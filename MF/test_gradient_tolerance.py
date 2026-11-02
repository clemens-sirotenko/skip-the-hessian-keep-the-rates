"""Shared outer stopping behavior for methods used by the parallel MF notebook."""

import contextlib
import io
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import methods


class QuadraticOracle:
    def __init__(self):
        self.hessian_calls = 0

    def func(self, x):
        return 0.5 * x.dot(x)

    def grad(self, x):
        return x.copy()

    def hess(self, x):
        self.hessian_calls += 1
        return lambda v: v.copy()

    hess_gn = hess

    def penalty(self, x):
        return 0.0


class GradientToleranceTests(unittest.TestCase):
    cases = (
        (methods.gradient_method, "max_iter", {}),
        (methods.mf_leapssn, "n_iters", {"lss_method": "minres"}),
        (methods.mf_gladssn, "n_iters", {"lss_method": "minres", "p": 0.5}),
        (methods.mf_leapssn, "n_iters", {"lss_method": "minres", "use_gauss_newton": True}),
        (methods.mf_gladssn, "n_iters", {"lss_method": "minres", "use_gauss_newton": True}),
        (methods.cubic_newton_nonconvex_new, "n_iters", {"subproblem_solver": "dense"}),
    )

    def run_case(self, case, *, x=1.0, budget=30, oracle=None, **kwargs):
        solver, budget_name, defaults = case
        options = dict(defaults, **{budget_name: budget})
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return solver(oracle or QuadraticOracle(), np.array([x]), **options)

    def test_initial_point_at_tolerance_stops_before_hessian_work(self):
        for case in self.cases:
            for trace in (True, False):
                with self.subTest(solver=case[0].__name__, options=case[2], trace=trace):
                    oracle = QuadraticOracle()
                    x, status, history = self.run_case(
                        case, x=0.125, grad_tol=0.125, oracle=oracle, trace=trace)
                    assert_array_equal(x, [0.125])
                    self.assertEqual(status, "success, 0 iters")
                    self.assertEqual(oracle.hessian_calls, 0)
                    if trace:
                        self.assertEqual(history["grad_norm"], [0.125])
                    else:
                        self.assertIsNone(history)

    def test_tolerance_is_on_norm_not_squared_norm(self):
        for case in self.cases:
            with self.subTest(solver=case[0].__name__, options=case[2]):
                _, status, _ = self.run_case(case, x=0.2, budget=0, grad_tol=0.05)
                self.assertEqual(status, "iterations_exceeded")

    def test_stop_at_first_crossing_preserves_the_trajectory_prefix(self):
        for case in self.cases:
            with self.subTest(solver=case[0].__name__, options=case[2]):
                _, _, baseline = self.run_case(case, grad_tol=None)
                x, status, history = self.run_case(case, grad_tol=0.1)
                crossing = np.flatnonzero(np.asarray(baseline["grad_norm"]) <= 0.1)[0]
                self.assertEqual(status, f"success, {crossing} iters")
                self.assertEqual(len(history["grad_norm"]), crossing + 1)
                self.assertLessEqual(np.linalg.norm(x), 0.1)
                for key in ("func", "grad_norm", "penalty", "grad_calls"):
                    assert_array_equal(history[key], baseline[key][:crossing + 1])

    def test_success_takes_precedence_at_the_iteration_limit(self):
        for case in self.cases:
            with self.subTest(solver=case[0].__name__, options=case[2]):
                _, _, baseline = self.run_case(case, budget=1, grad_tol=None)
                tolerance = baseline["grad_norm"][-1]
                _, status, history = self.run_case(case, budget=1, grad_tol=tolerance)
                self.assertEqual(status, "success, 1 iters")
                self.assertEqual(len(history["grad_norm"]), 2)

    def test_gradient_method_none_retains_budget_only_stopping(self):
        _, status, history = self.run_case(self.cases[0], x=0.0, budget=3, grad_tol=None)
        self.assertEqual(status, "iterations_exceeded")
        self.assertEqual(len(history["grad_norm"]), 4)

    def test_gradient_method_uses_recorded_dual_norm_with_a_metric(self):
        x, status, history = self.run_case(
            self.cases[0], x=0.2, grad_tol=0.1, B=np.array([[4.0]]))
        assert_array_equal(x, [0.2])
        self.assertEqual(status, "success, 0 iters")
        self.assertEqual(history["grad_norm"], [0.1])


if __name__ == "__main__":
    unittest.main()
