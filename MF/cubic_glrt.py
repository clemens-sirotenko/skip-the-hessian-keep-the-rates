"""Ask GALAHAD GLRT to approximately minimize a cubic Newton model.

For a step s, the predicted change in the objective is

    m(s) = g @ s + 0.5 * s @ A @ s + (H / 3) * ||s||**3.

Here g is the current gradient, A is the (generalized) Hessian, and H is
a positive scalar weight, NOT the Hessian. The cubic term penalizes large
steps. GLRT builds a smaller problem using products A @ v; we supply those
products without needing to form a dense Hessian.

Read solve() for the overall flow, then _run_glrt() for the library interface.
The outer method in methods.py decides whether to accept s and how to change H.
This is a supplementary MF baseline for Section 5.2 / Appendix E.2 of
https://arxiv.org/abs/2602.08069v1, not the linear subproblem in Algorithm 1.
See CUBIC_SUBPROBLEM.md for notebook and figure inputs.

Reference: Cartis, Gould and Toint (2011), Adaptive cubic regularisation
methods for unconstrained optimization. Part I, Section 6.2.
https://doi.org/10.1007/s10107-009-0286-5
https://ralna.github.io/galahad_docs/html/Python/glrt.html

The Lanczos algorithm itself lives in GALAHAD, not in this adapter.
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
            "GALAHAD GLRT is required for the cubic baseline. Install "
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
    """Keep g and A fixed while trying different cubic weights H.

    g is a real gradient of shape (d,). A is a symmetric (d, d) matrix,
    LinearOperator, or callable taking v of shape (d,) and returning A @ v.
    Symmetry is assumed, not checked. Optional progress(message) receives logs.

    Each solve starts GLRT from scratch. hvp_calls (Hessian-vector products)
    and iterations accumulate across calls; products include the final residual
    check. last_iterations, native_status, inform and the residuals describe
    only the most recent call.

    Success means a small model gradient, not a certified global minimum:
    GLRT's search space can miss negative curvature. At g=0 we return s=0,
    even if A is indefinite. Inputs are copied before giving them to GLRT.
    """

    def __init__(self, g, A, progress=None):
        if np.iscomplexobj(g):
            raise ValueError("g must be real")
        self.g = np.array(g, dtype=np.float64, copy=True)
        if (self.g.ndim != 1 or self.g.size == 0
                or not np.all(np.isfinite(self.g))):
            raise ValueError("g must be a finite nonempty vector")
        # Normalize all supported Hessian forms to one operation: matvec(v).
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

    def solve(self, H, eps=1e-6, max_iter=400):
        """Return (step, model_change, status), with step of shape (d,).

        H > 0 multiplies ||s||**3 / 3. The outer cubic method passes H_k/2
        because its model uses H_k*||s||**3/6; ARC passes sigma directly.
        eps in (0, 1) bounds the model-gradient norm relative to ||g||.
        max_iter caps GLRT iterations, not Hessian-vector products. Workspace
        uses O(d*max_iter) memory; see the binding workaround in _run_glrt().

        status is "success" or a failure reason. On failure return a zero
        step and NaN model change; the caller must check status before use.
        """
        # 1. Prepare this trial, keeping the cumulative work counters.
        _validate_controls(H, eps, max_iter)
        glrt = _load_glrt()
        self.last_iterations = 0
        self.native_status = None
        self.residual = self.relative_residual = np.nan
        self.inform = {}
        g_norm = norm(self.g)
        if g_norm == 0:
            self.native_status = 0
            self.residual = self.relative_residual = 0.
            return np.zeros_like(self.g), 0., "success"

        # 2. Let GLRT find a candidate using only Hessian-vector products.
        step, status = self._run_glrt(glrt, H, eps, max_iter)
        if status != "success":
            return self._failure(status)

        # 3. Check the returned step ourselves, even if GLRT reports success.
        if step.shape != self.g.shape or not np.all(np.isfinite(step)):
            return self._failure("nonfinite_subproblem")
        As = self._product(step)
        if not np.all(np.isfinite(As)):
            return self._failure("nonfinite_hessian_product")
        with np.errstate(over="ignore", invalid="ignore"):
            radius = np.float64(norm(step))
            shift = H * radius
            # grad m(s) = g + A@s + H*||s||*s should be close to zero.
            residual = self.g + As + shift * step
            self.residual = float(norm(residual, check_finite=False))
            self.relative_residual = self.residual / g_norm
            # Allow rounding error when the three gradient terms cancel.
            scale = g_norm + norm(As) + shift * radius
            allowance = 64 * np.finfo(float).eps * scale
            model = self.g @ step + .5 * step @ As + shift * radius**2 / 3
        if not np.all(np.isfinite([model, self.residual, scale])):
            return self._failure("numerical_error")
        if self.residual > eps * g_norm + allowance:
            # GLRT 5.5.3 can report success at its iteration cap even when
            # the reconstructed step misses the requested accuracy.
            if self.last_iterations >= max_iter:
                return self._failure("iterations_exceeded")
            return self._failure("residual_tolerance_not_met")
        if self.progress is not None:
            self.progress(f"GLRT: {self.last_iterations} iterations, "
                          f"relative model residual {self.relative_residual:.2e}, "
                          f"{self.hvp_calls} HVPs across this search")
        return step, float(model), "success"

    def _run_glrt(self, glrt, H, eps, max_iter):
        """Return (candidate, status), releasing native workspace on every exit.

        GLRT uses 'reverse communication': it pauses and returns a request
        code; we fill its work arrays and call it again with that same code.
        The loop below supplies operations, not the optimization algorithm.
        """
        with _GLRT_LOCK:
            options = glrt.initialize()
            try:
                options.update(
                    unitm=True,       # Euclidean norm: metric M is the identity.
                    print_level=0,
                    itmax=int(max_iter),
                    # Store all Lanczos vectors, including the starting vector.
                    # The 5.5.3 binding loses partial sums when reconstructing
                    # a step in a second pass; this avoids that second pass.
                    extra_vectors=int(max_iter) + 1,
                    stopping_rule=3,  # Target ||grad m(s)|| <= eps * ||g||.
                    stop_relative=float(eps),
                    stop_absolute=0.,
                    fraction_opt=1.,  # Do not settle for a fraction of reduction.
                    rminvr_zero=0.,   # Do not round small initial gradients to zero.
                    f_0=0.,          # We want the model change, so m(0) = 0.
                    impose_descent=False,  # No extra constraint g @ s < 0.
                )
                glrt.load_options(options)
                # r starts as g; v carries requests and our answers in place.
                r, v = self.g.copy(), np.zeros_like(self.g)
                status = 1  # Start a fresh solve (no reuse between trials).
                failure_reason = None
                while status > 0:
                    try:
                        # power=3 and weight=H give the term H*||s||**3/3.
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
                    if status == 3:    # Request: replace v with A @ v.
                        value = self._product(v)
                        if not np.all(np.isfinite(value)):
                            failure_reason = "nonfinite_hessian_product"
                            break
                        v[:] = value
                    elif status == 4:  # Request: restore the initial gradient.
                        r[:] = self.g
                    elif status == 2:  # Request: replace v with M^{-1} @ v.
                        pass          # M = I; normally skipped with unitm=True.
                    elif status > 0:
                        raise RuntimeError(f"Unexpected GLRT request {status}")

                # Status 0 means done; negative codes mean native failure.
                # Record spent work even when we stopped on a nonfinite A @ v.
                self.inform = glrt.information()
                self.native_status = int(status if status < 0
                                         else self.inform.get("status", status))
                self.last_iterations = int(self.inform.get("iter", 0))
                self.iterations += self.last_iterations
                if failure_reason is not None:
                    return None, failure_reason
                if self.native_status != 0:
                    if self.native_status == -18:
                        return None, "iterations_exceeded"
                    return None, f"glrt_error_{self.native_status}"
                return np.array(step, dtype=np.float64, copy=True), "success"
            finally:
                glrt.terminate()

    def _product(self, v):
        """Apply A, count the work, and protect GLRT's v from callback mutation."""
        self.hvp_calls += 1
        value = np.asarray(self.matvec(v.copy()))
        if value.shape != self.g.shape or np.iscomplexobj(value):
            raise ValueError("A must return a real vector with the same shape as g")
        return np.asarray(value, dtype=np.float64)

    def _failure(self, status):
        """Use the same failure result and log format for all numerical exits."""
        if self.progress is not None:
            self.progress(f"GLRT failed: {status}; native status {self.native_status}, "
                          f"{self.last_iterations} iterations, relative model "
                          f"residual {self.relative_residual:.2e}")
        return np.zeros_like(self.g), np.nan, status


def cubic_newton_step_glrt(g, H_mv, H, eps=1e-6, max_iter=400):
    """Solve a Euclidean cubic model with GALAHAD; see GLRTCubicSubproblem.

    g has shape (d,), H_mv applies a symmetric d-by-d Hessian, and H>0 is the
    coefficient in H/3*||s||**3. Return (step, model_change, status).
    Requires the optional galahad-optrove dependency only when called.
    """
    return GLRTCubicSubproblem(g, H_mv).solve(H, eps, max_iter)
