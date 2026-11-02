"""Dense reference solver for cubic regularization.

Model: g.T@s + .5*s.T@A@s + H/3*||s||_B**3. See CUBIC_SUBPROBLEM.md
and Cartis, Gould & Toint (2011), Theorem 3.1 and Sections 6.1 and 6.3:
https://doi.org/10.1007/s10107-009-0286-5 . This solves the subproblem;
the outer majorization rule lives in methods.py. Framework attribution
is retained in ../THIRD_PARTY.txt.
"""

import numpy as np
import scipy.linalg
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


def cubic_newton_step_nonconvex_new(g, H_mv, H, B=None, eps=1e-8,
                                   max_iter=200):
    """Dense global reference for g.T@s + .5*s.T@A@s + H/3*||s||_B**3.

    g has shape (n,); H_mv is a symmetric (n,n) matrix, LinearOperator,
    or callable v -> A@v. B is SPD (n,n), default identity. H>0; eps is
    the relative optimality/secular tolerance; max_iter bounds scalar steps.
    Returns (step, model_change, status), with zero/NaN on numerical failure.

    Materializes A and uses O(n**2) storage and O(n**3) work. The spectral
    solve checks stationarity and PSD curvature and handles the hard case.
    This is a floating-point solution to tolerance, not an exact certificate.
    No CG/MINRES solves or random starts. For the outer H_k/6 convention,
    pass H=H_k/2. See CGT (2011), Theorem 3.1, Sections 6.1 and 6.3, and
    CUBIC_SUBPROBLEM.md. The outer solver reuses preparation within a search.
    """
    _validate_cubic_nonconvex_controls(H, eps, max_iter)
    solve = _prepare_cubic_nonconvex(g, H_mv, B)
    return solve(H, eps=eps, max_iter=max_iter)


def _validate_cubic_nonconvex_controls(H, eps, max_iter):
    if not np.isscalar(H) or np.iscomplexobj(H) or not np.isreal(H) or not np.isfinite(H) or H <= 0:
        raise ValueError("H must be a finite positive scalar")
    if not np.isscalar(eps) or np.iscomplexobj(eps) or not np.isreal(eps) or not 0 < eps < 1:
        raise ValueError("eps must lie in (0, 1)")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")


def _prepare_cubic_nonconvex(g, H_mv, B=None, progress=None):
    """Prepare a fixed cubic model's quadratic part; return solve(H, eps, max_iter).

    Materialization, metric scaling, and eigendecomposition happen once here.
    The returned solver is valid only for these g, H_mv, and B; the outer loop
    creates a new one at every iterate. Only the cubic coefficient may change.
    Optional progress(message) reports the expensive preparation stages.
    """
    if np.iscomplexobj(g):
        raise ValueError("g must be real")
    g = np.array(g, dtype=np.float64, copy=True)
    if g.ndim != 1 or g.size == 0 or not np.all(np.isfinite(g)):
        raise ValueError("g must be a finite, nonempty vector")
    n = g.size
    roundoff = 32 * n * np.finfo(np.float64).eps

    def symmetric_matrix(value, name):
        if sparse.issparse(value):
            value = value.toarray()
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be a finite matrix of shape {(n, n)}")
        scale = np.max(np.abs(matrix))
        if np.max(np.abs(matrix - matrix.T)) > roundoff * scale:
            raise ValueError(f"{name} must be symmetric")
        return 0.5 * matrix + 0.5 * matrix.T

    if callable(H_mv) or isinstance(H_mv, LinearOperator):
        if progress is not None:
            from time import perf_counter
            progress(f"building dense Hessian ({n} Hessian-vector products)")
            last_report = perf_counter()
        matvec = H_mv if callable(H_mv) else H_mv.matvec
        A = np.empty((n, n))
        basis = np.zeros(n)
        for j in range(n):
            basis[j] = 1.0
            column = np.asarray(matvec(basis.copy()))
            if column.shape != (n,) or np.iscomplexobj(column):
                raise ValueError("H_mv must return a real vector of shape (n,)")
            A[:, j] = column
            basis[j] = 0.0
            if progress is not None and perf_counter() - last_report >= 5.0:
                progress(f"building dense Hessian: {j + 1}/{n} columns")
                last_report = perf_counter()
        A = symmetric_matrix(A, "H_mv")
    else:
        A = symmetric_matrix(H_mv, "H_mv")

    # y = L.T @ s turns ||s||_B into ||y||_2 when B = L @ L.T.
    L = None
    A_scaled, g_scaled = A, g
    if B is not None:
        B = symmetric_matrix(B, "B")
        try:
            L = scipy.linalg.cholesky(B, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError("B must be positive definite") from error
        left = scipy.linalg.solve_triangular(L, A, lower=True)
        A_scaled = scipy.linalg.solve_triangular(L, left.T, lower=True).T
        A_scaled = 0.5 * A_scaled + 0.5 * A_scaled.T
        g_scaled = scipy.linalg.solve_triangular(L, g, lower=True)

    def failure(status):
        return np.zeros(n), np.nan, status

    def failed_solve(H, eps=1e-8, max_iter=200):
        _validate_cubic_nonconvex_controls(H, eps, max_iter)
        return failure("numerical_error")

    if not (np.all(np.isfinite(A_scaled)) and np.all(np.isfinite(g_scaled))):
        return failed_solve
    if progress is not None:
        progress(f"eigendecomposition of {n} x {n} dense Hessian")
    try:
        eigenvalues, vectors = scipy.linalg.eigh(A_scaled)
    except np.linalg.LinAlgError:
        return failed_solve
    c = vectors.T @ g_scaled
    g_norm = scipy.linalg.norm(c)
    a_norm = np.max(np.abs(eigenvalues))
    lower = max(0.0, -eigenvalues[0])
    gaps = eigenvalues + lower

    def solve(H, eps=1e-8, max_iter=200):
        _validate_cubic_nonconvex_controls(H, eps, max_iter)
        H = float(H)

        def finish(z):
            y = vectors @ z
            step = y if L is None else scipy.linalg.solve_triangular(L.T, y, lower=False)
            # Unrepresentable model values are numerical failures, even when the
            # step itself is finite. NumPy scalars avoid Python overflow exceptions.
            with np.errstate(over="ignore", invalid="ignore"):
                radius = np.float64(scipy.linalg.norm(y, check_finite=False))
                shift = H * radius
                residual = scipy.linalg.norm(g_scaled + A_scaled @ y + shift * y,
                                             check_finite=False)
                residual_scale = g_norm + (a_norm + shift) * radius
                curvature_scale = max(a_norm, shift)
                linear = g @ step
                quadratic = 0.5 * step @ (A @ step)
                cubic = (shift / 3.0) * radius**2
                model = linear + quadratic + cubic
                model_scale = abs(linear) + abs(quadratic) + abs(cubic)
            if (not np.all(np.isfinite(step))
                    or not np.all(np.isfinite([shift, residual, residual_scale, model, model_scale]))
                    or residual > eps * residual_scale
                    or eigenvalues[0] + shift < -eps * curvature_scale
                    or model > eps * model_scale):
                return failure("numerical_error")
            return step, float(model), "success"

        if g_norm == 0 and lower == 0:
            return finish(np.zeros(n))

        # At the spectral boundary the inverse becomes a pseudoinverse. When g
        # is orthogonal to the nullspace, add the missing norm in that eigenspace.
        if lower > 0:
            null = gaps <= roundoff * a_norm
            null_gradient = scipy.linalg.norm(c[null])
            if null_gradient <= roundoff * g_norm:
                z = np.zeros(n)
                z[~null] = -c[~null] / gaps[~null]
                radius = scipy.linalg.norm(z)
                target = lower / H
                if np.isfinite(target) and radius <= target:
                    extra = target * np.sqrt(max(0.0, (1 - radius / target) * (1 + radius / target)))
                    if null_gradient > 0:
                        z[null] = -extra * (c[null] / null_gradient)
                    else:
                        z[np.flatnonzero(null)[0]] = extra
                    return finish(z)

        # Search in t = lambda - lower, so a near-hard-case root does not require
        # subtracting two nearly equal eigenvalue-sized numbers in denominators.
        def secular(t):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                z = -c / (gaps + t)
            radius = scipy.linalg.norm(z, check_finite=False)
            return z, radius, H * radius - (lower + t)

        # ||c/(d+lambda)|| <= ||c||/(d_min+lambda) provides an upper bound.
        # Rationalization avoids cancellation for positive d_min or tiny g.
        q = np.sqrt(H) * np.sqrt(g_norm)
        discriminant = np.hypot(eigenvalues[0], 2 * q)
        upper = q if eigenvalues[0] == 0 else q * (2 * q / (discriminant + abs(eigenvalues[0])))
        if not np.isfinite(upper) or upper <= 0:
            return failure("numerical_error")
        for _ in range(64):
            _, _, value = secular(upper)
            if value <= 0:
                break
            upper *= 2
            if not np.isfinite(upper):
                return failure("numerical_error")
        else:
            return failure("numerical_error")

        low, t = 0.0, upper
        for _ in range(max_iter):
            z, radius, value = secular(t)
            shift = lower + t
            if np.isfinite(radius) and abs(value) <= eps * max(shift, H * radius):
                return finish(z)
            if value > 0:
                low = t
            else:
                upper = t
            next_t = low + 0.5 * (upper - low)
            if np.isfinite(radius) and radius > 0:
                slope = -1 - H * radius * np.sum((z / radius)**2 / (gaps + t))
                newton_t = t - value / slope
                if np.isfinite(newton_t) and low < newton_t < upper:
                    next_t = newton_t
            if next_t == t or next_t <= 0:
                return failure("numerical_error")
            t = next_t
        return failure("iterations_exceeded")

    return solve
