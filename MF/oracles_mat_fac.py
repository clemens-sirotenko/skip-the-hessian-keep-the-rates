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

        f(U,V) = 0.5 * || U V^T - M ||_F^2
                 + 0.5*lam*(||U||_F^2 + ||V||_F^2)
                 + mu*(||(-U)_+||_F^2 + ||(-V)_+||_F^2)

    where (-U)_+ = max(-U,0) elementwise.

    The variable z is the concatenation:
        z = [vec(U); vec(V)]  in R^{m*r + n*r}.
    """

    def __init__(self, M, r, lam=1e-4, mu=1.0):
        self.M = np.asarray(M, dtype=np.float64)
        self.m, self.n = self.M.shape
        self.r = int(r)
        self.lam = float(lam)
        self.mu = float(mu)

    # ---- pack/unpack helpers ----
    def unpack(self, z):
        m, n, r = self.m, self.n, self.r
        U = z[:m*r].reshape(m, r)
        V = z[m*r:].reshape(n, r)
        return U, V

    def pack(self, U, V):
        return np.concatenate([U.reshape(-1), V.reshape(-1)], axis=0)

    # ---- objective ----
    def func(self, z):
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

    # ---- Gauss–Newton / generalized Hessian: matrix-free matvec ----
    def hess_gn(self, z):
        """
        Returns a function Hv(v) that computes (H_GN(z) v) WITHOUT forming H.

        H_GN is PSD-ish:
          - reconstruction term uses Gauss–Newton (J^T J) which is PSD
          - L2 adds lam*I
          - hinge penalty adds diagonal 2*mu on entries where U<0 or V<0
        """
        U, V = self.unpack(z)
        m, n, r = self.m, self.n, self.r
        lam, mu = self.lam, self.mu

        # Precompute Gram matrices for GN terms
        VtV = V.T @ V   # (r,r)
        UtU = U.T @ U   # (r,r)

        # Semismooth diagonal curvature masks for hinge penalty
        maskU = (U < 0.0).astype(np.float64)  # (m,r)
        maskV = (V < 0.0).astype(np.float64)  # (n,r)

        def Hv(v):
            dU, dV = self.unpack(v)

            A = dV.T @ V  # (r,r)
            B = dU.T @ U  # (r,r)

            HvU = dU @ VtV + U @ A
            HvV = dV @ UtU + V @ B


            # L2 curvature
            HvU += lam * dU
            HvV += lam * dV

            # hinge curvature: second derivative is 2*mu on negative entries, else 0
            HvU += (2.0 * mu) * (maskU * dU)
            HvV += (2.0 * mu) * (maskV * dV)

            return self.pack(HvU, HvV)

        return Hv
     
    def hess(self, z):
        """
        Returns Hv(v) computing the exact symmetric Hessian-vector product H*v.
        """
        U, V = self.unpack(z)
        m, n, r = self.m, self.n, self.r
        lam, mu = self.lam, self.mu

        # Precompute Gram matrices
        VtV = V.T @ V  # (r,r)
        UtU = U.T @ U  # (r,r)

        # Diagonal masks for hinge penalty
        maskU = (U < 0.0).astype(np.float64)
        maskV = (V < 0.0).astype(np.float64)

        def Hv(v):
            dU, dV = self.unpack(v)

            # symmetric exact Hessian-vector product
            # Step 1: GN-like term
            HvU = dU @ VtV + U @ (dV.T @ V)
            HvV = dV @ UtU + V @ (dU.T @ U)

            # Step 2: exact second-order coupling term
            # (dU V^T + U dV^T) @ V for U block
            # (dU V^T + U dV^T)^T @ U for V block
            HvU += (dU @ V.T + U @ dV.T) @ V
            HvV += (dU @ V.T + U @ dV.T).T @ U

            # Step 3: L2 regularization
            HvU += lam * dU
            HvV += lam * dV

            # Step 4: hinge penalty curvature
            HvU += 2.0 * mu * (maskU * dU)
            HvV += 2.0 * mu * (maskV * dV)

            return self.pack(HvU, HvV)

        return Hv
 

 