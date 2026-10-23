"""Run from the repository root: python -B -m unittest discover -s MF -p "test_*.py"."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import methods
from utils import run_methods


class QuadraticOracle:
    def func(self, x):
        return 0.5 * x.dot(x)

    def grad(self, x):
        return x.copy()

    def hess(self, x):
        return lambda v: v.copy()

    def penalty(self, x):
        return 0.0


def exact_scalar_solve(operator, rhs, **kwargs):
    return rhs / operator.matvec(np.ones(1))[0], 0


class SearchFailureTests(unittest.TestCase):
    solvers = (methods.mf_leapssn, methods.mf_gladssn)

    def run_solver(self, solver, oracle=None, **kwargs):
        options = dict(n_iters=3, lss_method="minres")
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return solver(oracle or QuadraticOracle(), np.ones(1), **options)

    def test_rejected_finite_steps_never_update_the_iterate(self):
        for solver in self.solvers:
            with self.subTest(solver=solver.__name__):
                # A finite ascent step must not be used after exhausting retries.
                with patch.object(methods, "minres_compat", return_value=(np.ones(1), 0)):
                    x, status, history = self.run_solver(solver)
                assert_allclose(x, [1.0])
                self.assertEqual(status, "line_search_failed, 0 iters")
                self.assertEqual(history["func"], [0.5])

    def test_failed_solves_retry_a_changed_system_and_preserve_the_point(self):
        for solver in self.solvers:
            for failure in ("nan", "exception", "solver_status"):
                with self.subTest(solver=solver.__name__, failure=failure):
                    systems = []
                    def fail(operator, rhs, **kwargs):
                        systems.append(operator.matvec(np.ones(1))[0])
                        if failure == "exception":
                            raise ValueError("forced failure")
                        if failure == "solver_status":
                            return np.ones(1), -1
                        return np.full(1, np.nan), 0
                    with patch.object(methods, "minres_compat", side_effect=fail):
                        x, status, _ = self.run_solver(solver)
                    assert_allclose(x, [1.0])
                    self.assertTrue(status.startswith("line_search_failed"))
                    self.assertEqual(len(systems), 120)
                    self.assertTrue(np.all(np.diff(systems) > 0))

    def test_failed_search_returns_the_last_accepted_point(self):
        for solver in self.solvers:
            with self.subTest(solver=solver.__name__):
                calls = []
                def once(operator, rhs, **kwargs):
                    calls.append(1)
                    if len(calls) == 1:
                        return exact_scalar_solve(operator, rhs)
                    return np.ones(1), 0
                with patch.object(methods, "minres_compat", side_effect=once):
                    x, status, history = self.run_solver(solver)
                assert_allclose(x, [0.5])
                self.assertEqual(status, "line_search_failed, 1 iters")
                assert_allclose(history["func"], [0.5, 0.125])

    def test_nonfinite_trial_objective_or_gradient_is_rejected(self):
        for solver in self.solvers:
            for bad_value in ("func", "grad"):
                with self.subTest(solver=solver.__name__, bad_value=bad_value):
                    oracle = QuadraticOracle()
                    original = getattr(oracle, bad_value)
                    def bad(x):
                        if np.array_equal(x, np.ones(1)):
                            return original(x)
                        return np.inf if bad_value == "func" else np.full_like(x, np.nan)
                    setattr(oracle, bad_value, bad)
                    with patch.object(methods, "minres_compat", return_value=(np.array([-0.25]), 0)):
                        x, status, history = self.run_solver(solver, oracle)
                    assert_allclose(x, [1.0])
                    self.assertTrue(status.startswith("line_search_failed"))
                    self.assertEqual(history["func"], [0.5])

    def test_unrepresentable_step_stops_without_claiming_success(self):
        for solver in self.solvers:
            with self.subTest(solver=solver.__name__):
                with patch.object(methods, "minres_compat", return_value=(np.array([-1e-300]), 0)) as solve:
                    x, status, _ = self.run_solver(solver)
                assert_allclose(x, [1.0])
                self.assertEqual(status, "numerical_stagnation, 0 iters")
                self.assertEqual(solve.call_count, 1)

    def test_invalid_regularization_does_not_enter_the_linear_solver(self):
        for solver in self.solvers:
            with self.subTest(solver=solver.__name__):
                with patch.object(methods, "minres_compat") as solve:
                    x, status, _ = self.run_solver(solver, H_0=np.inf)
                assert_allclose(x, [1.0])
                self.assertTrue(status.startswith("invalid_regularization"))
                solve.assert_not_called()

    def test_valid_steps_still_decrease_the_quadratic_with_or_without_history(self):
        for solver in self.solvers:
            for trace in (True, False):
                with self.subTest(solver=solver.__name__, trace=trace):
                    with patch.object(methods, "minres_compat", side_effect=exact_scalar_solve):
                        x, status, history = self.run_solver(solver, trace=trace)
                    self.assertLess(np.linalg.norm(x), 0.5)
                    self.assertEqual(status, "iterations_exceeded")
                    if trace:
                        self.assertTrue(np.all(np.diff(history["func"]) < 0))
                    else:
                        self.assertIsNone(history)

    def test_experiment_runner_reports_failure_status(self):
        def failed():
            return np.ones(1), "line_search_failed, 0 iters", {"time": [0.0]}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_methods({"test": (failed, (), {})}, None, None)
        self.assertIn("Status: line_search_failed, 0 iters", output.getvalue())


if __name__ == "__main__":
    unittest.main()
