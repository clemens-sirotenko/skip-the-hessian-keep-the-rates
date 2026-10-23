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
    """
    Minimise over U in R^{m x r}, V in R^{n x r}:

        R = U V^T - M
        f(U,V) = 0.5 * ||R||_F^2
                 + 0.5*lam*(||U||_F^2 + ||V||_F^2)
                 + mu*(||(-U)_+||_F^2 + ||(-V)_+||_F^2)

    where (-U)_+ = max(-U,0) elementwise.

    Relation to the paper's equation (27):
        paper ridge coefficient alpha = lam / 2,
        paper penalty coefficient 1 / (2*beta) = mu.
    Thus, to use paper parameters alpha and beta, pass
        lam = 2*alpha, mu = 1 / (2*beta).
    The code stores V transposed relative to equation (27), so U @ V.T
    here corresponds to U @ V in that equation; M is the paper's Y.

    Appendix E.2 uses beta = 1/200, matching the notebooks' mu = 100.
    Its alpha = 0.01 would require lam = 0.02; the current notebooks use
    lam = 0.01 (effective alpha = 0.005). This discrepancy is unresolved.

    The variable z is the concatenation:
        z = [vec(U); vec(V)]  in R^{m*r + n*r}.
    """

    def __init__(self, M, r, lam=1e-4, mu=1.0):
        self.M = np.asarray(M, dtype=np.float64)
        self.m, self.n = self.M.shape
        self.r = int(r)
        self.lam = float(lam)  # Twice the paper's ridge coefficient alpha.
        self.mu = float(mu)    # Direct penalty multiplier: 1 / (2*beta).

    # ---- Convert between matrix factors and the solver's single vector ----
    def unpack(self, z):
        """Split a vector of length (m + n)*r into U (m, r) and V (n, r).

        Entries are stored row by row (NumPy's default C order): all of U,
        followed by all of V. This also splits a search direction into dU,
        dV, using exactly the same ordering as for the factor matrices.

        The returned arrays share memory with z for ordinary 1-D NumPy input:
        changing U or V in place also changes z. Copy them if needed.
        """
        m, n, r = self.m, self.n, self.r
        # The first m*r entries hold U; restore its m rows and r columns.
        U = z[:m*r].reshape(m, r)
        # The remaining n*r entries hold V; restore its n rows and r columns.
        V = z[m*r:].reshape(n, r)
        return U, V

    def pack(self, U, V):
        """Flatten U (m, r) and V (n, r) into one vector of length (m + n)*r.

        The solver works with a single vector rather than two matrices.
        Use the same layout for factor values, gradient blocks, and Hessian
        products. For matching shapes, unpack(pack(U, V)) recovers U and V.
        """
        # reshape(-1) lists entries row by row; -1 infers the vector length.
        # concatenate puts U first, then V, and allocates a new output array.
        return np.concatenate([U.reshape(-1), V.reshape(-1)], axis=0)

    # ---- objective ----
    def func(self, z):
        """Evaluate the objective with reconstruction residual R = U V^T - M."""
        U, V = self.unpack(z)
        R = U @ V.T - self.M  # (m,n)

        rec = 0.5 * np.sum(R * R)
        reg = 0.5 * self.lam * (np.sum(U * U) + np.sum(V * V))

        # hinge nonnegativity penalty: ||(-U)_+||^2 = sum(max(-U,0)^2)
        pen = self.mu * (np.sum(np.maximum(-U, 0.0)**2) + np.sum(np.maximum(-V, 0.0)**2))

        return rec + reg + pen

    def penalty(self, z):
        U, V = self.unpack(z)
        pen = self.mu * (np.sum(np.maximum(-U, 0.0)**2) + np.sum(np.maximum(-V, 0.0)**2))
        return pen
    
    # ---- gradient ----
    def grad(self, z):
        """Return the packed gradient; the data-fit blocks are R V and R^T U."""
        U, V = self.unpack(z)
        R = U @ V.T - self.M

        # grad of 0.5||R||^2
        gU = R @ V
        gV = R.T @ U

        # add grad of 0.5*lam(||U||^2  + ||V||^2)
        gU += self.lam * U
        gV += self.lam * V

        # add grad of mu * (||(-U)_+||^2 + ||(-V)_+||^2):
        # for u<0 : d/d u ( (-u)^2 ) = 2u    ; for u>=0 : 0
        gU += 2.0 * self.mu * (U * (U < 0.0))
        gV += 2.0 * self.mu * (V * (V < 0.0))

        return self.pack(gU, gV)

    # ---- Gauss-Newton approximation: matrix-free matvec ----
    def hess_gn(self, z):
        """Return a callable applying the Gauss-Newton approximation at z.

        For R = U V^T - M, its directional derivative is
            dR = dU V^T + U dV^T.
        The data-fit blocks are dR V and dR^T U. Gauss-Newton omits
        the residual terms R dV and R^T dU from the exact Hessian.
        Ridge and penalty curvature are included. For lam, mu >= 0,
        the resulting operator is positive semidefinite.
        """
        U, V = self.unpack(z)
        # Freeze the evaluation point for reuse in lazy Hessian updates.
        U, V = U.copy(), V.copy()
        lam, mu = self.lam, self.mu

        # At zero entries choose zero generalized penalty curvature.
        maskU = (U < 0.0).astype(np.float64)
        maskV = (V < 0.0).astype(np.float64)

        def Hv(v):
            dU, dV = self.unpack(v)
            dR = dU @ V.T + U @ dV.T  # (m, n): directional change of R

            # Gauss-Newton contribution from 0.5 * ||R||_F^2.
            HvU = dR @ V
            HvV = dR.T @ U

            # Ridge curvature.
            HvU += lam * dU
            HvV += lam * dV

            # Nonnegativity penalty curvature.
            HvU += 2.0 * mu * (maskU * dU)
            HvV += 2.0 * mu * (maskV * dV)

            return self.pack(HvU, HvV)

        return Hv

    # ---- Exact Hessian / generalized Hessian selection ----
    def hess(self, z):
        """Return a callable applying the Hessian at z without forming it.

        R = U V^T - M and dR = dU V^T + U dV^T give the product rules
            d(R V)   = dR V   + R dV,
            d(R^T U) = dR^T U + R^T dU.
        These are the data-fit Hessian blocks. Ridge and penalty curvature
        are added below. At zero factor entries, the penalty is not twice
        differentiable; choose zero from its generalized curvature [0, 2*mu]
        for mu >= 0. Away from these kinks, this is the classical Hessian.
        """
        U, V = self.unpack(z)
        # Freeze the evaluation point for reuse in lazy Hessian updates.
        U, V = U.copy(), V.copy()
        R = U @ V.T - self.M  # (m, n): residual at the evaluation point
        lam, mu = self.lam, self.mu

        # At zero entries choose zero generalized penalty curvature.
        maskU = (U < 0.0).astype(np.float64)
        maskV = (V < 0.0).astype(np.float64)

        def Hv(v):
            dU, dV = self.unpack(v)
            dR = dU @ V.T + U @ dV.T  # (m, n): directional change of R

            # Product rule for the gradient blocks R @ V and R.T @ U.
            HvU = dR @ V + R @ dV
            HvV = dR.T @ U + R.T @ dU

            # Ridge curvature.
            HvU += lam * dU
            HvV += lam * dV

            # Nonnegativity penalty curvature.
            HvU += 2.0 * mu * (maskU * dU)
            HvV += 2.0 * mu * (maskV * dV)

            return self.pack(HvU, HvV)

        return Hv
