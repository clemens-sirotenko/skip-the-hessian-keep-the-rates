import numpy as np



 


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of the function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

    def third_vec_vec(self, x, v):
        """
        Computes tensor-vector-vector product with the third derivative tensor
        D^3 f(x)[v, v].
        """
        raise NotImplementedError('Third derivative oracle is not implemented.')


class OracleCallsCounter(BaseSmoothOracle):
    """
    Wrapper to count oracle calls.
    """

    def __init__(self, oracle):
        self.oracle = oracle
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0
        self.hess_vec_calls = 0
        self.third_vec_vec_calls = 0
        self.penalty_calls = 0

    def func(self, x):
        self.func_calls += 1
        return self.oracle.func(x)

    def penalty(self, x):
        self.penalty_calls += 1
        return self.oracle.penalty(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.oracle.grad(x)

    def hess(self, x):
        self.hess_calls += 1
        return self.oracle.hess(x)
    
    def hess_gn(self, x):
        self.hess_calls += 1
        return self.oracle.hess_gn(x)
    
    def hess_vec(self, x, v):
        self.hess_vec_calls += 1
        return self.oracle.hess_vec(x, v)

    def third_vec_vec(self, x, v):
        self.third_vec_vec_calls += 1
        return self.oracle.third_vec_vec(x, v)
    

class PositiveMatrixFactorization:
    """Non-negative matrix factorization using the paper's revised equation (27).

    U has shape (m, r), V has shape (r, n), and M (the paper's Y) has
    shape (m, n). With reconstruction residual R = U V - M, minimize

        f(U,V) = 0.5 * ||R||_F^2
                 + 0.5*alpha*(||U||_F^2 + ||V||_F^2)
                 + 0.5*beta*(||(-U)_+||_F^2 + ||(-V)_+||_F^2).

    Here (-U)_+ = max(-U, 0) entrywise. Larger beta penalizes negative
    entries more strongly; beta is a direct weight, not an inverse weight.

    Migration from the previous code: alpha = old lam, beta = 2*old mu,
    and V = old V.T. Repack stored factor vectors after transposing V.
    The experiment settings alpha = 0.01, beta = 200 preserve the previous
    objective (lam = 0.01, mu = 100). The defaults likewise preserve the
    old default weights. This follows the revised formula supplied by the
    authors; older paper PDFs may still use alpha and 1/(2*beta) instead.

    z = [vec(U); vec(V)] is a vector of length (m + n)*r in row-major order.
    """

    def __init__(self, M, r, alpha=1e-4, beta=2.0):
        self.M = np.asarray(M, dtype=np.float64)
        self.m, self.n = self.M.shape
        self.r = int(r)
        self.alpha = float(alpha)  # Ridge term: alpha/2 times squared norms.
        self.beta = float(beta)    # Penalty term: beta/2 times squared negative parts.

    # ---- Convert between matrix factors and the solver's single vector ----
    def unpack(self, z):
        """Split a vector of length (m + n)*r into U (m, r) and V (r, n).

        Entries are stored row by row (NumPy's default C order): all of U,
        followed by all of V. This also splits a search direction into dU,
        dV, using exactly the same ordering as for the factor matrices.

        The returned arrays share memory with z for ordinary 1-D NumPy input:
        changing U or V in place also changes z. Copy them if needed.
        """
        m, n, r = self.m, self.n, self.r
        # The first m*r entries hold U; restore its m rows and r columns.
        U = z[:m*r].reshape(m, r)
        # The remaining r*n entries hold V; restore its r rows and n columns.
        V = z[m*r:].reshape(r, n)
        return U, V

    def pack(self, U, V):
        """Flatten U (m, r) and V (r, n) into one vector of length (m + n)*r.

        The solver works with a single vector rather than two matrices.
        Use the same layout for factor values, gradient blocks, and Hessian
        products. For matching shapes, unpack(pack(U, V)) recovers U and V.
        """
        # reshape(-1) lists entries row by row; -1 infers the vector length.
        # concatenate puts U first, then V, and allocates a new output array.
        return np.concatenate([U.reshape(-1), V.reshape(-1)], axis=0)

    # ---- objective ----
    def func(self, z):
        """Evaluate the revised equation (27), with R = U V - M."""
        U, V = self.unpack(z)
        R = U @ V - self.M  # (m, n)

        rec = 0.5 * np.sum(R * R)
        reg = 0.5 * self.alpha * (np.sum(U * U) + np.sum(V * V))
        pen = 0.5 * self.beta * (
            np.sum(np.maximum(-U, 0.0)**2) + np.sum(np.maximum(-V, 0.0)**2)
        )
        return rec + reg + pen

    def penalty(self, z):
        """Return beta/2 times the squared negative parts of U and V."""
        U, V = self.unpack(z)
        return 0.5 * self.beta * (
            np.sum(np.maximum(-U, 0.0)**2) + np.sum(np.maximum(-V, 0.0)**2)
        )

    # ---- gradient ----
    def grad(self, z):
        """Return the packed gradient; the data-fit blocks are R V^T and U^T R."""
        U, V = self.unpack(z)
        R = U @ V - self.M

        # Gradient of 0.5 * ||R||_F^2.
        gU = R @ V.T
        gV = U.T @ R

        # Gradient of alpha/2 times the squared factor norms.
        gU += self.alpha * U
        gV += self.alpha * V

        # d/du [beta/2 * max(-u, 0)^2] = beta * min(u, 0).
        gU += self.beta * np.minimum(U, 0.0)
        gV += self.beta * np.minimum(V, 0.0)

        return self.pack(gU, gV)

    # ---- Gauss-Newton approximation: matrix-free matvec ----
    def hess_gn(self, z):
        """Return a callable applying the Gauss-Newton approximation at z.

        For R = U V - M, its directional derivative is
            dR = dU V + U dV.
        The data-fit blocks are dR V^T and U^T dR. Gauss-Newton omits
        the residual terms R dV^T and dU^T R from the exact Hessian.
        Ridge and penalty curvature are included. For alpha, beta >= 0,
        the resulting operator is positive semidefinite.
        """
        U, V = self.unpack(z)
        # Freeze the evaluation point for reuse in lazy Hessian updates.
        U, V = U.copy(), V.copy()
        alpha, beta = self.alpha, self.beta

        # At zero entries choose zero generalized penalty curvature.
        maskU = (U < 0.0).astype(np.float64)
        maskV = (V < 0.0).astype(np.float64)

        def Hv(v):
            dU, dV = self.unpack(v)
            dR = dU @ V + U @ dV  # (m, n): directional change of R

            # Gauss-Newton contribution from 0.5 * ||R||_F^2.
            HvU = dR @ V.T
            HvV = U.T @ dR

            # Ridge curvature.
            HvU += alpha * dU
            HvV += alpha * dV

            # Nonnegativity penalty curvature.
            HvU += beta * (maskU * dU)
            HvV += beta * (maskV * dV)

            return self.pack(HvU, HvV)

        return Hv

    # ---- Exact Hessian / generalized Hessian selection ----
    def hess(self, z):
        """Return a callable applying the Hessian at z without forming it.

        R = U V - M and dR = dU V + U dV give the product rules
            d(R V^T) = dR V^T + R dV^T,
            d(U^T R) = dU^T R + U^T dR.
        These are the data-fit Hessian blocks. In expanded form their
        diagonal terms are dU (V V^T) and (U^T U) dV.
        Ridge and penalty curvature are added below. At zero factor entries,
        the penalty is not twice differentiable, but its gradient is
        semismooth. Choose zero from its generalized curvature [0, beta]
        for beta >= 0. Away from these kinks, this is the classical Hessian.
        """
        U, V = self.unpack(z)
        # Freeze the evaluation point for reuse in lazy Hessian updates.
        U, V = U.copy(), V.copy()
        R = U @ V - self.M  # (m, n): residual at the evaluation point
        alpha, beta = self.alpha, self.beta

        # At zero entries choose zero generalized penalty curvature.
        maskU = (U < 0.0).astype(np.float64)
        maskV = (V < 0.0).astype(np.float64)

        def Hv(v):
            dU, dV = self.unpack(v)
            dR = dU @ V + U @ dV  # (m, n): directional change of R

            # Product rule for the gradient blocks R @ V.T and U.T @ R.
            HvU = dR @ V.T + R @ dV.T
            HvV = dU.T @ R + U.T @ dR

            # Ridge curvature.
            HvU += alpha * dU
            HvV += alpha * dV

            # Nonnegativity penalty curvature.
            HvU += beta * (maskU * dU)
            HvV += beta * (maskV * dV)

            return self.pack(HvU, HvV)

        return Hv
