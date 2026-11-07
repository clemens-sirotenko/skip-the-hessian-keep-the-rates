"""Focused split-run checks, with tiny data and budgets (not paper reproduction)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import copy
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from NN.nn_run_manager_new_v2 import Net, RunManager, read_run


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        torch.manual_seed(26)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "base.pkl"
        X = torch.randn(3, 2)
        data = dict(
            metadata=dict(objective="finite_power_surrogate", precision="torch.float64",
                          architecture=[2, 16, 16, 1], activation="tanh", K=1,
                          normalization_eps=1e-7, penalty_coefficient=1., adam_lr=1e-4,
                          adam_scheduler="original", grad_tol=0., lam_min=1e-9,
                          max_backtracks=25, solve_rtol=1e-10, threads=1),
            training_inputs=X, training_targets=torch.sin(X[:, :1]),
            initial_state=copy.deepcopy(Net().state_dict()),
            test_dataset=list(zip(X, torch.sin(X[:, :1]))))
        with self.base.open("wb") as stream:
            pickle.dump(data, stream)

    def tearDown(self):
        self.tmp.cleanup()

    def manager(self, base=None):
        return RunManager(base or self.base, self.root / "outputs", checkpoint_every=0, print_every=0)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def compare(self, left, right):
        self.assert_nested_equal(left["weights"], right["weights"])
        self.assert_nested_equal(left["checkpoint"], right["checkpoint"])
        for key in left["history"]:
            if key != "time":
                self.assertEqual(left["history"][key], right["history"][key])
        self.assertEqual(left["info"]["accepted_steps"], right["info"]["accepted_steps"])
        self.assertEqual(sorted(right["history"]["time"]), right["history"]["time"])

    def test_split_checkpoints_and_legacy_recovery(self):
        for label, split in [("Adam", 2), ("GLAd-SSN m=3", 2), ("GLAd-SSN m=3", 3)]:
            with self.subTest(label=label, split=split):
                full = self.manager()
                full.choose(label, "run", 5)
                partial = self.manager()
                partial.choose(label, "run", split)
                saved = partial.save()
                resumed = self.manager(saved)
                resumed.choose(label, "continue", 5)
                self.compare(full.records[label], resumed.records[label])
                # Remove state to exercise both replay and a refresh-boundary recovery.
                legacy = read_run(saved)
                del legacy["checkpoints"]
                legacy_file = self.root / "legacy.pkl"
                with legacy_file.open("wb") as stream:
                    pickle.dump(legacy, stream)
                recovered = self.manager(legacy_file)
                recovered.choose(label, "continue", 5)
                self.compare(full.records[label], recovered.records[label])
                # Cumulative target makes rerunning continue idempotent.
                recovered._step = lambda *args: self.fail("Already reached target")
                recovered.choose(label, "continue", 5)

    def test_load_never_trains_and_rejects_mixed_objective(self):
        original = self.manager()
        original.choose("Adam", "run", 1)
        saved = original.save()
        loaded = self.manager(saved)
        loaded._runtime = lambda *args: self.fail("Loading must not construct a trainer")
        loaded.choose("Adam", "load")
        self.compare(original.records["Adam"], loaded.records["Adam"])
        incompatible = read_run(saved)
        incompatible["metadata"]["normalization_eps"] = 0.1
        file = self.root / "incompatible.pkl"
        with file.open("wb") as stream:
            pickle.dump(incompatible, stream)
        with self.assertRaisesRegex(ValueError, "normalization_eps"):
            loaded.load("Adam", file)

    def test_interrupt_rolls_back_optimizer_and_weights(self):
        runner = self.manager()
        runner.choose("Adam", "run", 2)
        before = copy.deepcopy(runner.records["Adam"])
        step = runner._step

        def interrupted(*args):
            step(*args)
            raise KeyboardInterrupt

        runner._step = interrupted
        with self.assertRaises(KeyboardInterrupt):
            runner.choose("Adam", "continue", 5)
        self.compare(before, runner.records["Adam"])
        stored = self.manager(runner.save())
        stored.choose("Adam", "load")
        self.compare(before, stored.records["Adam"])

    def test_legacy_recovery_refuses_different_endpoint(self):
        runner = self.manager()
        runner.choose("Adam", "run", 2)
        legacy = read_run(runner.save())
        del legacy["checkpoints"]
        legacy["final_model_states"]["Adam"]["fc1.weight"][0, 0] += 1e-8
        path = self.root / "mismatched.pkl"
        with path.open("wb") as stream:
            pickle.dump(legacy, stream)
        recovered = self.manager(path)
        with self.assertRaisesRegex(RuntimeError, "saved weights exactly"):
            recovered.choose("Adam", "continue", 3)
        self.assertIsNone(recovered.output_dir)


if __name__ == "__main__":
    unittest.main()
