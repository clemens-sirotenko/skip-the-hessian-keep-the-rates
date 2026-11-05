"""Corrected GLAd-SSN; GLAdSSNOptimizer.py remains available for rollback.

Implements Algorithm 1 with p=1/2 and psi=0 in the local current manuscript.
See CORRECTIONS.md for numerical safeguards and the public paper reference;
../THIRD_PARTY.txt retains the framework and PyTorch credits.
"""
import math

import torch
# -------------------------
# Utilities
# -------------------------


def make_param_spec(model, trainable_only: bool = True):
    """
    Creates a consistent mapping between:
      - the model's (trainable) parameters
      - a single flattened parameter vector

    Returns:
      pack() -> (P,) tensor
      vec_to_param_dict(vec) -> {param_name: tensor_view}
      assign(vec) -> in-place copy vec -> model parameters
      total -> number of scalars in vec
    """
    names, shapes, numels, params = [], [], [], []

    for name, p in model.named_parameters():
        if (not trainable_only) or p.requires_grad:
            names.append(name)
            shapes.append(p.shape)
            numels.append(p.numel())
            params.append(p)

    total = sum(numels)

    def pack():
        # Current model params -> one flat vector (detached)
        return torch.cat([p.detach().reshape(-1) for p in params])

    def vec_to_param_list(vec):
        # Flat vector -> list of tensors (views into vec)
        out, off = [], 0
        for n, shape in zip(numels, shapes):
            out.append(vec[off : off + n].view(shape))
            off += n
        if off != vec.numel():
            raise ValueError(f"Vector has {vec.numel()} elements, expected {off}.")
        return out

    def vec_to_param_dict(vec):
        # Flat vector -> {name: tensor_view}
        return {k: v for k, v in zip(names, vec_to_param_list(vec))}

    @torch.no_grad()
    def assign(vec):
        # In-place update of model parameters from vec
        off = 0
        for p, n, shape in zip(params, numels, shapes):
            p.copy_(vec[off : off + n].view(shape))
            off += n
        if off != vec.numel():
            raise ValueError(f"Vector has {vec.numel()} elements, expected {off}.")

    return pack, vec_to_param_dict, assign, total


class GlAdSSNOptimizer(torch.optim.Optimizer):
    """Dense damped Newton steps with a Hessian reused at accepted iterates.

    m is the maximum number of backtracking increases (m+1 trials), NOT the
    paper's Hessian period. number_of_steps_to_freeze is that period.
    state['lambda_k'] stores the paper's coefficient Lambda_k, while
    state['accepted_lam'] records the actual damping lambda of the last step.
    alpha=.5 and beta=.25 give the paper's two acceptance inequalities.

    grad_tol=0 stops only at an exactly zero gradient. lam_min is a numerical
    damping floor; set it to zero for the unfloored paper formula. solve_rtol
    bounds ||(H+lambda*I)s-g|| / ||g||; None uses 100*machine epsilon.
    The direct/least-squares solves are numerical approximations, not exact
    subproblem solves. No positive-semidefinite projection is applied.
    """
    def __init__(
        self,
        params,
        alpha=0.5,
        beta=0.25,
        m=4,
        lambda0=1.0,
        number_of_steps_to_freeze=1,
        lam_min=1e-8,
        symmetrize=True,
        grad_tol=0.0,
        solve_rtol=None,
    ):
        if not math.isfinite(lambda0) or lambda0 <= 0:
            raise ValueError("lambda0 must be finite and positive")
        if not isinstance(m, int) or m < 0:
            raise ValueError("m must be a nonnegative integer")
        if not isinstance(number_of_steps_to_freeze, int) or number_of_steps_to_freeze < 1:
            raise ValueError("number_of_steps_to_freeze must be a positive integer")
        if not math.isfinite(lam_min) or lam_min < 0 or not math.isfinite(grad_tol) or grad_tol < 0:
            raise ValueError("lam_min and grad_tol must be finite and nonnegative")
        if solve_rtol is not None and (not math.isfinite(solve_rtol) or solve_rtol <= 0):
            raise ValueError("solve_rtol must be finite and positive")
        defaults = dict(
            alpha=alpha,
            beta=beta,
            m=m,
            number_of_steps_to_freeze=number_of_steps_to_freeze,
            lam_min=lam_min,
            symmetrize=symmetrize,
            grad_tol=grad_tol,
            solve_rtol=solve_rtol,
        )
        super().__init__(params, defaults)
        self.state["lambda_k"] = float(lambda0)
        self.state["k"] = 0
        self.state["freeze_counter"] = 0
        self.state["H"] = None
        self.state["status"] = "initialized"
        self.state["accepted_lam"] = None
        self.state["last_trials"] = 0
        self.state["hessian_evaluations"] = 0
        self.state["linear_solves"] = 0
        self.state["last_linear_residual"] = math.nan

    def step(self, loss_from_vec, pack, assign):
        """Return (f, g, next Lambda) at the returned/current parameter vector.

        loss_from_vec takes a flat (P,) vector and returns a scalar. pack()
        reads that vector; assign(vector) updates model parameters. Inspect
        state['status']: 'accepted', 'converged', or a failure reason. On
        failure, parameters and accepted-iteration counters do not advance.
        """
        group = self.param_groups[0]
        alpha = group["alpha"]
        beta = group["beta"]
        m = group["m"]
        number_of_steps_to_freeze = group["number_of_steps_to_freeze"]
        symmetrize = group["symmetrize"]
        self.state["last_trials"] = 0
        self.state["accepted_lam"] = None
        self.state["last_linear_residual"] = math.nan

        xk = pack().detach()

        # Evaluate f,g at current x
        x_var = xk.detach().requires_grad_(True)
        fk = loss_from_vec(x_var)
        gk = torch.autograd.grad(fk, x_var, create_graph=False)[0].detach()
        gk_norm = float(torch.linalg.vector_norm(gk))
        if not torch.isfinite(fk) or not torch.isfinite(gk).all() or not math.isfinite(gk_norm):
            self.state["status"] = "nonfinite_objective_or_gradient"
            return fk.detach(), gk, self.state["lambda_k"]
        if gk_norm <= group["grad_tol"]:
            self.state["status"] = "converged"
            return fk.detach(), gk, self.state["lambda_k"]

        # (Re)compute Hessian
        if (
            self.state["H"] is None
            or self.state["freeze_counter"] >= number_of_steps_to_freeze
        ):
            H = torch.autograd.functional.hessian(loss_from_vec, x_var).detach()
            if symmetrize:
                H = 0.5 * (H + H.T)
            self.state["H"] = H
            self.state["freeze_counter"] = 0
            self.state["hessian_evaluations"] += 1

        H = self.state["H"]
        if not torch.isfinite(H).all():
            self.state["status"] = "nonfinite_hessian"
            return fk.detach(), gk, self.state["lambda_k"]

        lambda_k = float(self.state["lambda_k"])
        n = xk.numel()
        Id = torch.eye(n, device=xk.device, dtype=xk.dtype)

        gradient_scale = math.sqrt(gk_norm)  # ||g||**p with p=1/2.
        # Raising the base coefficient once avoids repeated identical trials
        # when the damping floor is active. Otherwise this is exactly Lambda_k.
        trial_coefficient = max(lambda_k, group["lam_min"] / gradient_scale)
        solve_rtol = group["solve_rtol"]
        if solve_rtol is None:
            solve_rtol = 100 * torch.finfo(xk.dtype).eps
        self.state["status"] = "line_search_failed"

        for j in range(m + 1):
            if j:
                trial_coefficient *= 4.0
            lam = trial_coefficient * gradient_scale
            self.state["last_trials"] = j + 1
            if not math.isfinite(lam) or lam <= 0:
                self.state["status"] = "invalid_regularization"
                break
            A = H + lam * Id
            if not torch.isfinite(A).all():
                self.state["status"] = "invalid_regularization"
                break

            # Solve (H + lam Id) s = g
            try:
                self.state["linear_solves"] += 1
                s = torch.linalg.solve(A, gk)
            except torch.linalg.LinAlgError:
                try:
                    self.state["linear_solves"] += 1
                    s = torch.linalg.lstsq(A, gk.unsqueeze(1)).solution.squeeze(1)
                except torch.linalg.LinAlgError:
                    continue
            # Least squares need not solve the stationarity equation. Reject
            # inaccurate or nonfinite solutions and try stronger damping.
            relative_residual = float(torch.linalg.vector_norm(A @ s - gk)) / gk_norm
            self.state["last_linear_residual"] = relative_residual
            if not torch.isfinite(s).all() or not math.isfinite(relative_residual) or relative_residual > solve_rtol:
                continue

            x_cand = xk - s
            if not torch.isfinite(x_cand).all():
                continue
            if torch.equal(x_cand, xk):
                self.state["status"] = "numerical_stagnation"
                break

            # Candidate loss and gradient
            x_cand_var = x_cand.detach().requires_grad_(True)
            f_cand = loss_from_vec(x_cand_var)
            g_cand = torch.autograd.grad(f_cand, x_cand_var, create_graph=False)[
                0
            ].detach()
            if not torch.isfinite(f_cand) or not torch.isfinite(g_cand).all():
                continue

            step = xk - x_cand  # = s
            Fprime_term = g_cand.dot(step)
            g_cand_norm_squared = g_cand.dot(g_cand)

            # Acceptance criteria
            cond1 = Fprime_term >= (alpha / lam) * g_cand_norm_squared
            cond2 = (fk.detach() - f_cand.detach()) >= beta * lam * step.dot(step)

            if cond1 and cond2:
                assign(x_cand.detach())
                # Algorithm 1 updates Lambda, NOT the gradient-scaled lambda.
                self.state["lambda_k"] = trial_coefficient / 4.0
                self.state["accepted_lam"] = lam
                self.state["k"] += 1
                self.state["freeze_counter"] += 1
                self.state["status"] = "accepted"
                return f_cand.detach(), g_cand, self.state["lambda_k"]

        return fk.detach(), gk, self.state["lambda_k"]
