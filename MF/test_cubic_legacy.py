"""Regression checks for the legacy cubic API and its shared safe outer loop."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import methods
import methods_legacy as legacy
from cubic_subproblems import cubic_newton_step_nonconvex_new


class QuadraticOracle:
    def func(self, x):
        return 0.5 * x.dot(x)

    def grad(self, x):
        return x.copy()

    def hess(self, x):
        return lambda v: v.copy()

    def penalty(self, x):
        return 0.0


class LegacyCubicTests(unittest.TestCase):
    def test_legacy_step_handles_an_indefinite_hessian(self):
        step, model, status = legacy.cubic_newton_step(
            np.ones(1), lambda v: -2 * v, 1.0, eps=1e-11)
        self.assertEqual(status, "success")
        assert_allclose(step, [-1 - np.sqrt(2)], rtol=1e-10)
        self.assertLess(model, 0)

    def test_legacy_step_handles_the_zero_gradient_hard_case(self):
        step, model, status = legacy.cubic_newton_step(
            np.zeros(1), lambda v: -2 * v, 1.0)
        self.assertEqual(status, "success")
        assert_allclose(np.abs(step), [2.0])
        self.assertAlmostEqual(model, -4 / 3)

    def test_history_matches_the_accepted_point_and_keeps_trial_fields(self):
        x, status, history = legacy.cubic_newton(
            QuadraticOracle(), np.ones(1), n_iters=1, warnings=False)
        self.assertEqual(status, "iterations_exceeded")
        assert_allclose(x, [2 - np.sqrt(3)], rtol=1e-8)
        self.assertAlmostEqual(history["grad_norm"][-1], np.linalg.norm(x))
        self.assertAlmostEqual(history["func"][-1], 0.5 * x.dot(x))
        self.assertEqual(history["grad_calls"][-1], 2)
        assert_allclose(history["func_at_lssolve"], history["func"][1:])
        assert_allclose(history["grad_norm_at_lssolve"], history["grad_norm"][1:])

    def test_trace_false_and_scaled_norm(self):
        metric = np.array([[4.0]])
        expected, _, _ = cubic_newton_step_nonconvex_new(
            np.ones(1), np.eye(1), 0.5, B=metric)
        for trace in (True, False):
            with self.subTest(trace=trace):
                x, _, history = legacy.cubic_newton(
                    QuadraticOracle(), np.ones(1), n_iters=1, B=metric,
                    trace=trace, warnings=False)
                assert_allclose(x, 1 + expected)
                if not trace:
                    self.assertIsNone(history)

    def test_exhausted_search_never_applies_the_last_rejected_step(self):
        def solve(*args, **kwargs):
            return np.ones(1), -1.0, "success"

        with patch("methods._prepare_cubic_nonconvex", return_value=solve):
            x, status, history = legacy.cubic_newton(
                QuadraticOracle(), np.ones(1), n_iters=1, warnings=False)
        assert_allclose(x, [1.0])
        self.assertEqual(status, "line_search_failed, 0 iters")
        self.assertEqual(history["func"], [0.5])
        self.assertEqual(len(history["func_at_lssolve"]), 120)
        self.assertEqual(history["grad_calls"][-1], 121)

    def test_failed_subproblem_and_positive_model_are_not_accepted(self):
        for model, message in ((-1.0, "numerical_error"), (1.0, "success")):
            with self.subTest(model=model, message=message):
                with patch("methods._prepare_cubic_nonconvex", return_value=(
                    lambda *args, **kwargs: (np.array([-0.5]), model, message)
                )):
                    x, status, history = legacy.cubic_newton(
                        QuadraticOracle(), np.ones(1), n_iters=1, warnings=False)
                assert_allclose(x, [1.0])
                self.assertFalse(status.startswith("success"))
                self.assertEqual(history["func"], [0.5])

    def test_objective_stop_and_fixed_regularization_mode(self):
        x, status, history = legacy.cubic_newton(
            QuadraticOracle(), np.ones(1), f_star=0.5, warnings=False)
        self.assertEqual(status, "success, 0 iters")
        self.assertEqual(history["hess_calls"], [0])
        _, _, history = legacy.cubic_newton(
            QuadraticOracle(), np.ones(1), n_iters=2,
            adaptive_search=False, H_0=2.0, warnings=False)
        self.assertEqual(history["H_k"], [2.0, 2.0, 2.0])

    def test_small_nonzero_gradient_does_not_satisfy_zero_tolerance(self):
        _, status, history = methods.cubic_newton_nonconvex_new(
            QuadraticOracle(), np.array([1e-200]), n_iters=0, grad_tol=0.0)
        self.assertEqual(status, "iterations_exceeded")
        self.assertEqual(history["grad_norm"], [1e-200])

    def test_fixed_regularization_does_not_retry_an_unusable_subproblem(self):
        coefficients = []

        def solve(H, **kwargs):
            coefficients.append(H)
            return np.zeros(1), np.nan, "numerical_error"

        with patch("methods._prepare_cubic_nonconvex", return_value=solve):
            x, status, _ = legacy.cubic_newton(
                QuadraticOracle(), np.ones(1), n_iters=2, H_0=2.0,
                adaptive_search=False, warnings=False)
        assert_allclose(x, [1.0])
        self.assertEqual(coefficients, [1.0])
        self.assertEqual(status, "subproblem_numerical_error, 0 iters")


if __name__ == "__main__":
    unittest.main()
