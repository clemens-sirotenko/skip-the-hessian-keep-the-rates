"""Call counters used by the MF solvers; framework attribution: ../THIRD_PARTY.txt."""


class OracleCallsCounter:
    """Count calls and retain objective samples at Hessian evaluations."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0
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

    def penalty(self, x):
        self.penalty_calls += 1
        return self.oracle.penalty(x)
