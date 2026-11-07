"""Per-method loading and continuation for experiment_nn_lipschitz_constraint_new_v2.

The objective and optimizer are unchanged from the _new notebook: local manuscript
2026-10-22, Algorithm 1, equation (26), Section 5.1 / Appendix E.1; public reference
https://arxiv.org/abs/2602.08069v1 (equation 27, older settings).
See CORRECTIONS.md and ../THIRD_PARTY.txt for numerical notes and attribution.
"""
import copy
from datetime import datetime
import math
from pathlib import Path
import pickle
import platform
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from .GLAdSSNOptimizer_new import GlAdSSNOptimizer
from .utils_new import hessian_norm_penalty, make_param_vector_interface


class Net(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.fc1 = nn.Linear(2, 16, dtype=dtype)
        self.fc2 = nn.Linear(16, 16, dtype=dtype)
        self.fc3 = nn.Linear(16, 1, dtype=dtype)

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def read_run(path):
    # Only open trusted local experiment files: pickle can execute Python code.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


class RunManager:
    """Keep comparison methods separate; never overwrite an input result file.

    total_updates is a cumulative target, not an additional iteration count.
    Loading only copies stored tensors and histories; it never calls a trainer.
    Continuation restores Adam moments/scheduler or GLAd's cached Hessian,
    coefficient and reuse counter. Legacy exports need verified state recovery.
    """

    def __init__(self, base_file, output_root=None, checkpoint_every=100, print_every=100):
        self.base_file = Path(base_file).resolve()
        self.base = read_run(self.base_file)
        self.metadata = copy.deepcopy(self.base["metadata"])
        if self.metadata["objective"] != "finite_power_surrogate":
            raise ValueError("This notebook loads the finite-power NN experiment.")
        if self.metadata["architecture"] != [2, 16, 16, 1]:
            raise ValueError("Expected the original 2-16-16-1 network.")
        if self.metadata.get("device", "cpu") != "cpu":
            raise ValueError("This continuation helper uses the original CPU experiment.")
        if self.metadata.get("activation") != "tanh":
            raise ValueError("Expected the original tanh network.")
        self.dtype = getattr(torch, self.metadata["precision"].split(".")[-1])
        torch.set_default_dtype(self.dtype)
        torch.set_num_threads(self.metadata.get("threads", 1))
        self.X = self.base["training_inputs"].detach().cpu().clone()
        self.Y = self.base["training_targets"].detach().cpu().clone()
        self.initial_state = copy.deepcopy(self.base["initial_state"])
        # Clone row views once; avoid serializing their entire backing storage repeatedly.
        self.test_dataset = [(x.clone(), y.clone()) for x, y in self.base["test_dataset"]]
        self.records = {}
        self.output_root = Path(output_root or self.base_file.parent.parent / "new_v2")
        self.output_dir = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        if any(type(value) is not int or value < 0 for value in (checkpoint_every, print_every)):
            raise ValueError("Checkpoint and print intervals must be nonnegative integers.")

    def _compatible(self, data):
        for key in ("objective", "precision", "K", "normalization_eps", "penalty_coefficient",
                    "architecture", "activation", "adam_lr", "adam_scheduler", "grad_tol",
                    "lam_min", "max_backtracks", "solve_rtol"):
            if data["metadata"].get(key) != self.metadata.get(key):
                raise ValueError(f"Cannot mix runs with different {key} settings.")
        for key in ("training_inputs", "training_targets"):
            if not torch.equal(data[key], self.base[key]):
                raise ValueError(f"Cannot mix runs with different {key}.")
        if data["initial_state"].keys() != self.initial_state.keys() or any(
            not torch.equal(data["initial_state"][key], value)
            for key, value in self.initial_state.items()
        ):
            raise ValueError("Cannot mix runs with different starting weights.")

    def load(self, label, source=None):
        path = Path(source).resolve() if source is not None else self.base_file
        data = self.base if path == self.base_file else read_run(path)
        self._compatible(data)
        if label not in data["histories"]:
            raise KeyError(f"{label} is not in {path}")
        history = copy.deepcopy(data["histories"][label])
        info = copy.deepcopy(data["statuses"][label])
        n = len(history["loss"])
        if any(len(values) != n for values in history.values()) or info["accepted_steps"] != n - 1:
            raise ValueError(f"Inconsistent history lengths/counters for {label}.")
        self.records[label] = dict(
            history=history, info=info,
            weights=copy.deepcopy(data["final_model_states"][label]),
            checkpoint=copy.deepcopy(data.get("checkpoints", {}).get(label)),
            provenance=copy.deepcopy(data.get("method_provenance", {}).get(label, [])),
            source=str(path),
        )
        print(f"Loaded {label}: {n - 1:,} updates, gradient {history['grad_norm'][-1]:.3e}")
        return self.records[label]

    @staticmethod
    def period(label):
        if label == "Adam":
            return None
        if label == "GLAd-SSN":
            return 1
        if label.startswith("GLAd-SSN m="):
            period = int(label.split("=")[-1])
            if period > 0:
                return period
        raise ValueError(f"Unknown method {label!r}")

    def _runtime(self, label, weights=None):
        model = Net(dtype=self.dtype)
        model.load_state_dict(self.initial_state if weights is None else weights)
        model.train()
        if label == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.metadata["adam_lr"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, threshold=1e-5, min_lr=1e-6)
        else:
            optimizer = GlAdSSNOptimizer(
                model.parameters(), lambda0=1.0, m=self.metadata["max_backtracks"],
                number_of_steps_to_freeze=self.period(label), symmetrize=True,
                grad_tol=self.metadata["grad_tol"], lam_min=self.metadata["lam_min"],
                solve_rtol=self.metadata["solve_rtol"])
            scheduler = None
        pack, unpack, assign, _ = make_param_vector_interface(model)

        def loss_from_vec(vec):
            params = unpack(vec)
            penalty, _ = self._penalty(model, params)
            return F.mse_loss(functional_call(model, params, self.X), self.Y) + penalty

        return dict(model=model, optimizer=optimizer, scheduler=scheduler,
                    pack=pack, assign=assign, loss_from_vec=loss_from_vec)

    def _penalty(self, model, params):
        return hessian_norm_penalty(
            model, params, self.X, torch.ones_like(self.X),
            coeff=self.metadata["penalty_coefficient"], K=self.metadata["K"],
            normalization_eps=self.metadata["normalization_eps"])

    def measure(self, model):
        penalty, max_norm = self._penalty(model, dict(model.named_parameters()))
        loss = F.mse_loss(model(self.X), self.Y) + penalty
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        grad_norm = torch.cat([g.reshape(-1) for g in grads]).norm()
        return loss.item(), grad_norm.item(), max_norm.item()

    def _step(self, label, runtime):
        model, opt = runtime["model"], runtime["optimizer"]
        if label == "Adam":
            lr_used = opt.param_groups[0]["lr"]
            opt.zero_grad(set_to_none=True)
            penalty, _ = self._penalty(model, dict(model.named_parameters()))
            loss_before = F.mse_loss(model(self.X), self.Y) + penalty
            loss_before.backward()
            opt.step()
            runtime["scheduler"].step(loss_before.detach().item())
            return "accepted", lr_used
        opt.step(runtime["loss_from_vec"], runtime["pack"], runtime["assign"])
        state = opt.state
        return state["status"], dict(
            attempted_update=state["k"] if state["status"] == "accepted" else state["k"] + 1,
            status=state["status"], accepted_damping=state["accepted_lam"],
            next_coefficient=state["lambda_k"], trials=state["last_trials"],
            relative_linear_residual=state["last_linear_residual"],
            hessian_evaluations=state["hessian_evaluations"], linear_solves=state["linear_solves"])

    @staticmethod
    def _snapshot(runtime):
        return dict(optimizer=copy.deepcopy(runtime["optimizer"].state_dict()),
                    scheduler=copy.deepcopy(runtime["scheduler"].state_dict())
                    if runtime["scheduler"] is not None else None)

    @staticmethod
    def _restore(runtime, checkpoint):
        runtime["optimizer"].load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        if runtime["scheduler"] is not None:
            runtime["scheduler"].load_state_dict(copy.deepcopy(checkpoint["scheduler"]))

    def _recover(self, label, record):
        """Recover missing legacy state, refusing any nonidentical replay endpoint."""
        runtime = self._runtime(label, record["weights"])
        if record["checkpoint"] is not None:
            self._restore(runtime, record["checkpoint"])
            return runtime, "full checkpoint", 0.0
        count = record["info"]["accepted_steps"]
        period = self.period(label)
        started = perf_counter()
        if period is not None and count % period == 0:
            # At a refresh boundary the old Hessian will never be used again.
            info, state = record["info"], runtime["optimizer"].state
            state.update(k=count, freeze_counter=period if count else 0, H=None,
                         lambda_k=record["history"]["Lambda"][-1],
                         hessian_evaluations=info.get("hessian_evaluations", 0),
                         linear_solves=info.get("linear_solves", 0))
            kind = "legacy state at Hessian refresh boundary"
        else:
            print(f"Recovering missing {label} optimizer state: replaying {count:,} saved updates.")
            print("This runs only this method; the stored trajectory is kept unchanged.", flush=True)
            runtime = self._runtime(label)
            for i in range(count):
                status, diagnostic = self._step(label, runtime)
                if status != "accepted":
                    raise RuntimeError(f"State recovery failed at update {i + 1}: {status}")
                expected = (record["info"]["learning_rates"][i] if label == "Adam"
                            else record["history"]["Lambda"][i + 1])
                actual = diagnostic if label == "Adam" else diagnostic["next_coefficient"]
                if expected != actual:
                    raise RuntimeError(f"State recovery diverged at update {i + 1}; no continuation performed.")
                if (i + 1) % 1000 == 0:
                    print(f"  recovered {i + 1:,}/{count:,}", flush=True)
            if any(not torch.equal(value, record["weights"][key])
                   for key, value in runtime["model"].state_dict().items()):
                raise RuntimeError(
                    "Replay did not reproduce the saved weights exactly; no continuation performed. "
                    "Use the original CPU/PyTorch environment or action='run' for a separate fresh run.")
            kind = "legacy deterministic replay (bitwise endpoint verified)"
        # Also guard boundary recovery against inconsistent files/objective settings.
        measured = self.measure(runtime["model"])
        if any(not math.isclose(value, record["history"][key][-1], rel_tol=1e-10, abs_tol=1e-12)
               for key, value in zip(("loss", "grad_norm", "hessian_norm"), measured)):
            raise RuntimeError("Saved endpoint metrics do not match the current objective.")
        return runtime, kind, perf_counter() - started

    def choose(self, label, action="load", total_updates=None, source=None):
        """Execute a single cell's load/run/continue/skip choice."""
        self.period(label)  # Validate even when loading or skipping.
        if action == "skip":
            print(f"Skipped {label}; any already loaded result is kept.")
            return
        if action == "load":
            self.load(label, source)
            return
        if action not in {"run", "continue"}:
            raise ValueError("action must be 'load', 'run', 'continue', or 'skip'")
        if type(total_updates) is not int or total_updates < 0:
            raise ValueError("total_updates must be a nonnegative integer")
        if action == "continue":
            if source is not None or label not in self.records:
                self.load(label, source)
            record = self.records[label]
            if total_updates <= record["info"]["accepted_steps"]:
                print(f"{label} already has {record['info']['accepted_steps']:,} updates; nothing to run.")
                return
            runtime, recovery, recovery_seconds = self._recover(label, record)
        else:
            runtime = self._runtime(label)
            started = perf_counter()
            loss, grad, norm = self.measure(runtime["model"])
            history = dict(loss=[loss], time=[perf_counter() - started], grad_norm=[grad], hessian_norm=[norm])
            info = dict(status="initialized", accepted_steps=0,
                        normalization_eps=self.metadata["normalization_eps"],
                        penalty_coefficient=self.metadata["penalty_coefficient"],
                        power_iterations=self.metadata["K"], precision=str(self.dtype))
            if label == "Adam":
                info["learning_rates"] = []
            else:
                history["Lambda"] = [1.0]
                info.update(update_freq=self.period(label), step_diagnostics=[])
            record = dict(history=history, info=info, weights=None, checkpoint=None,
                          source=str(self.base_file), provenance=[])
            self.records[label] = record
            recovery, recovery_seconds = "fresh common initialization", 0.0
        record["provenance"].append(dict(
            action=action, start_updates=record["info"]["accepted_steps"], target_updates=total_updates,
            source=record["source"], recovery=recovery, recovery_seconds=recovery_seconds,
            python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform(),
            processor=platform.processor(), device="cpu", threads=torch.get_num_threads(),
            precision=str(self.dtype), timestamp=datetime.now().isoformat()))
        self._capture(label, runtime)
        self.save()  # Preserve recovered optimizer state before starting extra updates.
        self._train(label, runtime, total_updates)

    def _capture(self, label, runtime):
        record = self.records[label]
        record["weights"] = copy.deepcopy(runtime["model"].state_dict())
        record["checkpoint"] = self._snapshot(runtime)
        if label != "Adam":
            state = runtime["optimizer"].state
            record["info"].update(
                hessian_evaluations=state["hessian_evaluations"], linear_solves=state["linear_solves"],
                last_trials=state["last_trials"], last_linear_residual=state["last_linear_residual"],
                last_damping=state["accepted_lam"], next_coefficient=state["lambda_k"])

    def _train(self, label, runtime, total_updates):
        record = self.records[label]
        history, info = record["history"], record["info"]
        info["status"] = "running"
        try:
            while info["accepted_steps"] < total_updates:
                if history["grad_norm"][-1] <= self.metadata["grad_tol"]:
                    info["status"] = "converged"
                    break
                # Roll back an interrupted/invalid update to a fully logged iterate.
                before_weights = copy.deepcopy(runtime["model"].state_dict())
                before_state = self._snapshot(runtime)
                before_count = info["accepted_steps"]
                before_lengths = {key: len(values) for key, values in history.items()}
                diagnostics_key = "learning_rates" if label == "Adam" else "step_diagnostics"
                before_diagnostics = len(info[diagnostics_key])
                started = perf_counter()
                try:
                    status, diagnostic = self._step(label, runtime)
                    if status != "accepted":
                        info["status"] = status
                        info["step_diagnostics"].append(diagnostic)
                        history["time"][-1] += perf_counter() - started
                        break
                    loss, grad, norm = self.measure(runtime["model"])
                    if not all(math.isfinite(v) for v in (loss, grad, norm)):
                        raise FloatingPointError("Nonfinite endpoint; restored the previous accepted iterate.")
                    history["loss"].append(loss)
                    history["grad_norm"].append(grad)
                    history["hessian_norm"].append(norm)
                    history["time"].append(history["time"][-1] + perf_counter() - started)
                    info["accepted_steps"] += 1
                    info[diagnostics_key].append(diagnostic)
                    if label != "Adam":
                        history["Lambda"].append(runtime["optimizer"].state["lambda_k"])
                except BaseException:
                    runtime["model"].load_state_dict(before_weights)
                    self._restore(runtime, before_state)
                    info["accepted_steps"] = before_count
                    del info[diagnostics_key][before_diagnostics:]
                    for key, values in history.items():
                        del values[before_lengths[key]:]
                    raise
                n = info["accepted_steps"]
                if self.print_every and n % self.print_every == 0:
                    print(f"{label}: {n:,}/{total_updates:,}, loss={loss:.6g}, ||g||={grad:.3e}", flush=True)
                if self.checkpoint_every and n % self.checkpoint_every == 0:
                    self._capture(label, runtime)
                    self.save(quiet=True)
                if label != "Adam" and grad <= self.metadata["grad_tol"]:
                    info["status"] = "converged"
                    break
            else:
                info["status"] = "iterations_exceeded"
        except KeyboardInterrupt:
            info["status"] = "interrupted"
            print("Interrupted; saving the last fully logged iterate.")
            raise
        except Exception:
            info["status"] = "exception"
            raise
        finally:
            self._capture(label, runtime)
            self.save()

    def directory(self):
        if self.output_dir is None:
            self.output_dir = self.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.output_dir.mkdir(parents=True, exist_ok=False)
        return self.output_dir

    def save(self, quiet=False):
        if not self.records:
            raise ValueError("Load or run a method first.")
        metadata = copy.deepcopy(self.metadata)
        metadata.update(
            format_version=2, source_file=str(self.base_file),
            per_method_updates={label: r["info"]["accepted_steps"] for label, r in self.records.items()},
            timing_continuation="CPU perf_counter per update including diagnostics; legacy timings retained; "
                                "state recovery, checkpoint copies/I/O and printing excluded",
            adam_included="Adam" in self.records,
            lazy_periods=[self.period(label) for label in self.records if label != "Adam"])
        metadata["adam_epochs"] = self.records["Adam"]["info"]["accepted_steps"] if "Adam" in self.records else 0
        glad_counts = {r["info"]["accepted_steps"] for label, r in self.records.items() if label != "Adam"}
        # A mixed-budget comparison has no single GLAd iteration count.
        metadata["glad_epochs"] = next(iter(glad_counts)) if len(glad_counts) == 1 else None
        metadata["requested_glad_periods"] = metadata["lazy_periods"]
        targets = {}
        for label, record in self.records.items():
            history = record["history"]
            targets[label] = {}
            for target in self.metadata.get("gradient_targets", [1e-2, 1e-3, 1e-4]):
                hit = next((i for i, v in enumerate(history["grad_norm"]) if v <= target), None)
                targets[label][f"{target:.0e}"] = dict(
                    reached=hit is not None, updates=hit, seconds=history["time"][hit] if hit is not None else None)
        payload = dict(
            metadata=metadata, labels=list(self.records), initial_state=self.initial_state,
            training_inputs=self.X, training_targets=self.Y, test_dataset=self.test_dataset,
            histories={k: r["history"] for k, r in self.records.items()},
            statuses={k: r["info"] for k, r in self.records.items()},
            final_model_states={k: r["weights"] for k, r in self.records.items()},
            checkpoints={k: r["checkpoint"] for k, r in self.records.items() if r["checkpoint"] is not None},
            method_provenance={k: r["provenance"] for k, r in self.records.items()},
            method_sources={k: r["source"] for k, r in self.records.items()}, gradient_targets=targets)
        path = self.directory() / "nn_results_new_v2.pkl"
        temporary = path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        if not quiet:
            print("Saved:", path)
        return path
