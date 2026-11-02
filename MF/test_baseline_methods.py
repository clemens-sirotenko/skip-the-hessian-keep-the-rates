"""Regression checks for the non-cubic baseline solvers."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import methods
import methods_legacy as legacy
from oracles_mat_fac import PositiveMatrixFactorization


class QuadraticOracle:
    def __init__(self, sign=1.0):
        self.sign = sign
        self.gradient_calls = 0

    def func(self, x):
        return 0.5 * self.sign * x.dot(x)

    def grad(self, x):
        self.gradient_calls += 1
        return self.sign * x.copy()

    def hess(self, x):
        return self.sign * np.eye(x.size)

    def penalty(self, x):
        return 0.0


class BaselineMethodTests(unittest.TestCase):
    def quiet(self, solver, oracle=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return solver(oracle or QuadraticOracle(), np.ones(1), **kwargs)

    def test_vanilla_tolerance_checks_the_gradient(self):
        x, status, history = self.quiet(legacy.vanilla_newton, n_iters=3, grad_tol=1e-10)
        assert_array_equal(x, [0.0])
        self.assertEqual(status, "success, 1 iters")
        self.assertEqual(history["func"], [0.5, 0.0])

    def test_dense_newton_trace_false_matches_traced_run(self):
        for solver in (legacy.vanilla_newton, legacy.super_newton):
            with self.subTest(solver=solver.__name__):
                expected, expected_status, _ = self.quiet(solver, n_iters=2)
                actual, status, history = self.quiet(solver, n_iters=2, trace=False)
                assert_array_equal(actual, expected)
                self.assertEqual(status, expected_status)
                self.assertIsNone(history)

    def test_vanilla_failed_solve_preserves_the_point(self):
        with patch.object(methods.scipy.linalg, "solve", side_effect=np.linalg.LinAlgError):
            x, status, history = self.quiet(legacy.vanilla_newton, n_iters=2)
        assert_array_equal(x, [1.0])
        self.assertEqual(status, "linear_solve_failed, 0 iters")
        self.assertEqual(history["func"], [0.5])

    def test_super_newton_retries_indefinite_regularized_hessian(self):
        x, status, history = self.quiet(
            legacy.super_newton, QuadraticOracle(-1.0), n_iters=1, H_0=1.0)
        assert_allclose(x, [4.0 / 3.0])
        self.assertEqual(status, "iterations_exceeded")
        self.assertLess(history["func"][-1], history["func"][0])
        self.assertEqual(history["reg_parameter"], [4.0])

    def test_super_newton_zero_gradient_stops_before_regularization(self):
        x, status, history = legacy.super_newton(QuadraticOracle(), np.zeros(1))
        assert_array_equal(x, [0.0])
        self.assertEqual(status, "success, 0 iters")
        self.assertEqual(history["hess_calls"], [0])

    def test_super_newton_rejected_steps_do_not_update_iterate(self):
        oracle = QuadraticOracle()
        with patch.object(methods.scipy.linalg, "cho_solve", return_value=np.ones(1)):
            x, status, history = self.quiet(legacy.super_newton, oracle, n_iters=2)
        assert_array_equal(x, [1.0])
        self.assertEqual(status, "line_search_failed, 0 iters")
        self.assertEqual(history["func"], [0.5])
        self.assertEqual(history["grad_calls"][-1], oracle.gradient_calls)
        self.assertEqual(history["hess_calls"][-1], 1)

    def test_gradient_search_exhaustion_preserves_the_point(self):
        for solver in (methods.gradient_method, legacy.fast_gradient_method):
            for trace in (True, False):
                with self.subTest(solver=solver.__name__, trace=trace):
                    x, status, history = self.quiet(
                        solver, max_iter=1, L_0=1e-30, trace=trace)
                    assert_array_equal(x, [1.0])
                    self.assertEqual(status, "line_search_failed, 0 iters")
                    if trace:
                        self.assertEqual(history["func"], [0.5])
                        self.assertGreater(history["func_calls"][-1], 1)
                    else:
                        self.assertIsNone(history)

    def test_gradient_methods_reject_nonfinite_trial_gradient(self):
        class NonfiniteTrialGradient(QuadraticOracle):
            def grad(self, x):
                return x.copy() if np.array_equal(x, np.ones(1)) else np.full_like(x, np.nan)

        for solver in (methods.gradient_method, legacy.fast_gradient_method):
            with self.subTest(solver=solver.__name__):
                x, status, history = self.quiet(solver, NonfiniteTrialGradient(), max_iter=1)
                assert_array_equal(x, [1.0])
                self.assertEqual(status, "line_search_failed, 0 iters")
                self.assertEqual(history["func"], [0.5])

    def test_accelerated_history_uses_the_accepted_point_gradient(self):
        x, _, history = self.quiet(legacy.fast_gradient_method, max_iter=3, L_0=2.0)
        assert_allclose(history["grad_sqr_norm"], 2 * np.asarray(history["func"]))
        self.assertAlmostEqual(history["grad_sqr_norm"][-1], x.dot(x))

    def test_inverse_only_metric_matches_explicit_metric(self):
        metric = np.array([[3.0, 0.5], [0.5, 2.0]])
        inverse = np.linalg.inv(metric)
        for solver in (methods.gradient_method, legacy.fast_gradient_method):
            with self.subTest(solver=solver.__name__):
                x_0 = np.array([1.0, -2.0])
                expected, status, history = solver(
                    QuadraticOracle(), x_0, max_iter=5, B=metric, Binv=inverse)
                actual, actual_status, actual_history = solver(
                    QuadraticOracle(), x_0, max_iter=5, Binv=inverse)
                assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
                self.assertEqual(actual_status, status)
                for key in ("func", "grad_sqr_norm"):
                    assert_allclose(actual_history[key], history[key], rtol=1e-13, atol=1e-14)

    def test_alternating_failed_block_preserves_accepted_work_and_history(self):
        oracle = PositiveMatrixFactorization(np.ones((1, 1)), r=1, alpha=0.5, beta=1.0)
        x_0 = oracle.pack(np.ones((1, 1)), np.ones((1, 1)))
        for block, options in (("u", {"L_u0": 1e-40}), ("v", {"L_v0": 1e-40})):
            for trace in (True, False):
                with self.subTest(block=block, trace=trace):
                    x, status, history = methods.alternating_gradient_method(
                        oracle, x_0, max_iter=2, trace=trace, **options)
                    self.assertEqual(status, f"line_search_failed_{block}, 0 iters")
                    if block == "u":
                        assert_array_equal(x, x_0)
                    else:
                        self.assertLess(oracle.func(x), oracle.func(x_0))
                    if trace:
                        self.assertEqual(len(history["func"]), 1 if block == "u" else 2)
                        self.assertEqual(history["func"][-1], oracle.func(x))
                        self.assertAlmostEqual(history["grad_norm"][-1], np.linalg.norm(oracle.grad(x)))
                        self.assertTrue(all(len(v) == len(history["func"]) for v in history.values()))
                    else:
                        self.assertIsNone(history)

    def test_alternating_sweep_matches_oracle_gradients_at_each_block(self):
        oracle = PositiveMatrixFactorization(
            np.array([[1.0, -0.5], [0.3, 2.0]]), r=1, alpha=0.5, beta=3.0)
        x_0 = np.array([-0.4, 0.8, 0.6, -0.2])
        original = x_0.copy()
        expected = x_0.copy()
        expected[:2] -= oracle.grad(expected)[:2] / 16.0
        expected[2:] -= oracle.grad(expected)[2:] / 16.0

        x, status, history = methods.alternating_gradient_method(
            oracle, x_0, max_iter=1, L_u0=16.0, L_v0=16.0, line_search=False)
        assert_allclose(x, expected, rtol=1e-14, atol=1e-15)
        assert_array_equal(x_0, original)
        self.assertEqual(status, "iterations_exceeded")
        self.assertAlmostEqual(history["func"][-1], oracle.func(x))
        self.assertAlmostEqual(history["grad_norm"][-1], np.linalg.norm(oracle.grad(x)))

    def test_alternating_tolerance_uses_full_gradient_norm(self):
        oracle = PositiveMatrixFactorization(np.ones((1, 1)), r=1, alpha=0.5, beta=1.0)
        # Both block norms and the squared full norm are 0.5; the full norm is sqrt(0.5).
        for tolerance, expected in ((0.6, "iterations_exceeded"), (0.8, "success")):
            with self.subTest(tolerance=tolerance):
                x, status, _ = methods.alternating_gradient_method(
                    oracle, np.ones(2), max_iter=0, tol=tolerance)
                assert_array_equal(x, np.ones(2))
                self.assertEqual(status, expected)

    def test_alternating_converges_with_monotone_objective_and_trace_independence(self):
        oracle = PositiveMatrixFactorization(np.ones((1, 1)), r=1, alpha=0.5, beta=1.0)
        x, status, history = methods.alternating_gradient_method(
            oracle, np.ones(2), max_iter=100, tol=1e-8)
        untraced, untraced_status, no_history = methods.alternating_gradient_method(
            oracle, np.ones(2), max_iter=100, tol=1e-8, trace=False)
        self.assertEqual(status, "success")
        self.assertLessEqual(np.linalg.norm(oracle.grad(x)), 1e-8)
        assert_allclose(x, np.full(2, np.sqrt(0.5)), rtol=0, atol=1e-8)
        self.assertTrue(np.all(np.diff(history["func"]) <= 1e-14))
        assert_array_equal(untraced, x)
        self.assertEqual(untraced_status, status)
        self.assertIsNone(no_history)


if __name__ == "__main__":
    unittest.main()
