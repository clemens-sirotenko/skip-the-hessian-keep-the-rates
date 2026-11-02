"""GALAHAD GLRT adapter for the Euclidean cubic Newton subproblem.

Reference: Cartis, Gould and Toint (2011), Adaptive cubic regularisation
methods for unconstrained optimization. Part I, Section 6.2.
https://doi.org/10.1007/s10107-009-0286-5
https://ralna.github.io/galahad_docs/html/Python/glrt.html

This module calls the installed GALAHAD implementation; it does not implement
Lanczos or ARC. The outer acceptance/regularization rules live in methods.py.
"""
from threading import Lock

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import aslinearoperator


# The Python extension exposes one module-level native solver state. Serialize
# initialize -> reverse communication -> information -> terminate in a process.
_GLRT_LOCK = Lock()


def _load_glrt():
    try:
        from galahad import glrt
    except ImportError as error:
        raise ImportError(
            "GALAHAD GLRT is required for subproblem_solver='glrt'. Install "
            "the optimization package in this notebook's Python environment: "
            "%pip install -r MF/requirements-glrt.txt (from the repository root), "
            "or %pip install -r requirements-glrt.txt (from MF). "
            "The PyPI package is galahad-optrove, not galahad."
        ) from error
    return glrt


def _validate_controls(H, eps, max_iter):
    for name, value in (("H", H), ("eps", eps)):
        if (not np.isscalar(value) or np.iscomplexobj(value)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and strictly positive")
    if eps >= 1:
        raise ValueError("eps must be smaller than one")
    if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
        raise ValueError("max_iter must be a positive integer")


class GLRTCubicSubproblem:
    """Fixed g/A, with diagnostics accumulated across cubic backtracking trials.

    g is a real vector of shape (d,); A is a symmetric matrix, LinearOperator,
    or callable v -> A v. Each solve minimizes approximately

        g.T @ s + 0.5*s.T @ A @ s + H/3 * ||s||**3.

    H is the scalar cubic weight, not the Hessian. The outer method passes
    H=H_k/2. GLRT is called with power=3 and the identity norm/preconditioner.
    Each trial starts a fresh native solve; the same fixed Hessian callable
    is reused, but no Lanczos state is reused across trials or outer iterates.

    GLRT starts its Krylov space from g. A successful status and small model
    residual do not certify positive semidefiniteness outside that space.
    At g=0 this adapter returns zero without a negative-curvature search.
    Thus success denotes an inexact first-order model solve, not a guaranteed
    global minimizer. Use the separate dense reference for global validation.
    """

    def __init__(self, g, A, progress=None):
        if np.iscomplexobj(g):
            raise ValueError("g must be real")
        self.g = np.array(g, dtype=np.float64, copy=True)
        if (self.g.ndim != 1 or self.g.size == 0
                or not np.all(np.isfinite(self.g))):
            raise ValueError("g must be a finite nonempty vector")
        if callable(A):
            self.matvec = A
        else:
            if np.iscomplexobj(A):
                raise ValueError("A must be real")
            operator = aslinearoperator(A)
            if operator.shape != (self.g.size, self.g.size):
                raise ValueError("A must have shape (len(g), len(g))")
            self.matvec = operator.matvec
        self.progress = progress
        self.hvp_calls = self.iterations = 0
        self.last_iterations = 0
        self.native_status = None
        self.residual = self.relative_residual = np.nan
        self.inform = {}

    def _product(self, v):
        self.hvp_calls += 1
        value = np.asarray(self.matvec(v.copy()))
        if value.shape != self.g.shape or np.iscomplexobj(value):
            raise ValueError("A must return a real vector with the same shape as g")
        return np.asarray(value, dtype=np.float64)

    def solve(self, H, eps=1e-6, max_iter=400):
        """Return (step, model_change, status), preserving g and operator inputs.

        eps requests ||g + A s + H*||s||*s|| <= eps*||g||. The returned step
        is independently checked, allowing an additional float64 roundoff
        allowance 64*u*(||g|| + ||A s|| + H*||s||**2). fraction_opt=1 keeps
        the final computed Krylov solution in GLRT's reconstruction pass.

        max_iter limits GLRT iterations (itmax), not total HVPs. All Lanczos
        vectors are retained (extra_vectors=max_iter+1; O(d*max_iter) storage)
        to avoid a reconstruction-state issue in the 5.5.3 Python binding.
        Work counts every native HVP and one extra product for the final check.
        Iteration exhaustion/failure returns zero, NaN, and a failure status;
        it must not be interpreted as a usable converged step.
        """
        _validate_controls(H, eps, max_iter)
        glrt = _load_glrt()
        self.last_iterations = 0
        self.native_status = None
        self.residual = self.relative_residual = np.nan
        self.inform = {}
        def failure(status):
            if self.progress is not None:
                self.progress(f"GLRT failed: {status}; native status {self.native_status}, "
                              f"{self.last_iterations} iterations, relative model "
                              f"residual {self.relative_residual:.2e}")
            return np.zeros_like(self.g), np.nan, status

        g_norm = norm(self.g)
        if g_norm == 0:
            self.native_status = 0
            self.residual = self.relative_residual = 0.
            return np.zeros_like(self.g), 0., "success"

        with _GLRT_LOCK:
            options = glrt.initialize()
            try:
                # The 5.5.3 Python binding allocates a new x on every reverse
                # communication call, losing partial second-pass sums. Retain
                # all Lanczos vectors so reconstruction needs no such calls.
                # Include the starting vector: itmax slots alone are not enough
                # when the solve reaches exactly the iteration cap.
                # This uses a supported GLRT control, at O(d*max_iter) storage.
                options.update(unitm=True, print_level=0, itmax=int(max_iter),
                               extra_vectors=int(max_iter) + 1,
                               stopping_rule=3, stop_relative=float(eps),
                               stop_absolute=0., fraction_opt=1., rminvr_zero=0.,
                               f_0=0., impose_descent=False)
                glrt.load_options(options)
                r, v = self.g.copy(), np.zeros_like(self.g)
                status = 1
                while status > 0:
                    try:
                        status, step, r, v = glrt.solve_problem(
                            status, self.g.size, 3., float(H), r, v)
                    except Exception:
                        # Some GALAHAD bindings raise on a negative native exit.
                        # Only convert an actual native failure; propagate API
                        # misuse/programming errors with no native error code.
                        self.inform = glrt.information()
                        status = int(self.inform.get("status", 0))
                        if status >= 0:
                            raise
                        break
                    if status == 3:
                        value = self._product(v)
                        if not np.all(np.isfinite(value)):
                            # Preserve work already spent before this early
                            # reverse-communication exit, as on native exits.
                            self.inform = glrt.information()
                            self.native_status = int(self.inform.get("status", status))
                            self.last_iterations = int(self.inform.get("iter", 0))
                            self.iterations += self.last_iterations
                            return failure("nonfinite_hessian_product")
                        v[:] = value
                    elif status == 4:
                        r[:] = self.g
                    elif status == 2:
                        pass  # Identity inverse metric; normally not requested.
                    elif status > 0:
                        raise RuntimeError(f"Unexpected GLRT request {status}")
                self.inform = glrt.information()
                self.native_status = int(status if status < 0 else self.inform["status"])
                self.last_iterations = int(self.inform.get("iter", 0))
                self.iterations += self.last_iterations
                if self.native_status != 0:
                    if self.native_status == -18:
                        return failure("iterations_exceeded")
                    return failure(f"glrt_error_{self.native_status}")
                step = np.array(step, dtype=np.float64, copy=True)
            finally:
                glrt.terminate()

        if step.shape != self.g.shape or not np.all(np.isfinite(step)):
            return failure("nonfinite_subproblem")
        As = self._product(step)
        if not np.all(np.isfinite(As)):
            return failure("nonfinite_hessian_product")
        with np.errstate(over="ignore", invalid="ignore"):
            radius = np.float64(norm(step))
            shift = H * radius
            residual = self.g + As + shift * step
            self.residual = float(norm(residual, check_finite=False))
            self.relative_residual = self.residual / g_norm
            scale = g_norm + norm(As) + shift * radius
            allowance = 64 * np.finfo(float).eps * scale
            model = self.g @ step + .5 * step @ As + shift * radius**2 / 3
        if not np.all(np.isfinite([model, self.residual, scale])):
            return failure("numerical_error")
        if self.residual > eps * g_norm + allowance:
            # GLRT 5.5.3 may return native status 0 after reconstructing the
            # best available step at its iteration cap. The actual residual
            # remains authoritative; a budget exit is not convergence.
            if self.last_iterations >= max_iter:
                return failure("iterations_exceeded")
            return failure("residual_tolerance_not_met")
        if self.progress is not None:
            self.progress(f"GLRT: {self.last_iterations} iterations, "
                          f"relative model residual {self.relative_residual:.2e}, "
                          f"{self.hvp_calls} HVPs across this search")
        return step, float(model), "success"


def cubic_newton_step_glrt(g, H_mv, H, eps=1e-6, max_iter=400):
    """Solve a Euclidean cubic model with GALAHAD; see GLRTCubicSubproblem.

    g has shape (d,), H_mv applies a symmetric d-by-d Hessian, and H>0 is the
    coefficient in H/3*||s||**3. Return (step, model_change, status).
    Requires the optional galahad-optrove dependency only when called.
    """
    return GLRTCubicSubproblem(g, H_mv).solve(H, eps, max_iter)
