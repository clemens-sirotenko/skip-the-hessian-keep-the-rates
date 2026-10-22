

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
        self.func_at_hess_eval = []
        self.penalty_calls = 0

    def func(self, x):
        self.func_calls += 1
        return self.oracle.func(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.oracle.grad(x)

    def hess(self, x):
        self.hess_calls += 1
        self.func_at_hess_eval.append(self.oracle.func(x))
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
    
    def penalty(self, x):
        self.penalty_calls += 1
        return self.oracle.penalty(x)

