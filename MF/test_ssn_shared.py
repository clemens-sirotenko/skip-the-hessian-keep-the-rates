"""Regression checks for shared LeAP/GLAd steps and final work accounting."""

import contextlib
import io
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

import methods


class QuadraticOracle:
    def __init__(self, explicit=True):
        self.explicit = explicit
        self.grad_calls = self.hess_calls = self.work = 0

    def func(self, x):
        return 0.5 * x.dot(x)

    def grad(self, x):
        self.grad_calls += 1
        self.work += 1
        return x.copy()

    def hess(self, x):
        self.hess_calls += 1
        self.work += 1
        return np.eye(x.size) if self.explicit else lambda v: v.copy()

    def penalty(self, x):
        return 0.0


class SharedSSNTests(unittest.TestCase):
    def run_solver(self, solver, oracle=None, **options):
        settings = dict(n_iters=1, lss_method="minres")
        settings.update(options)
        with contextlib.redirect_stdout(io.StringIO()):
            return solver(oracle or QuadraticOracle(), np.ones(1), **settings)

    def test_only_glad_uses_the_explicit_hessian_breakdown_fallback(self):
        for solver in (methods.mf_leapssn, methods.mf_gladssn):
            with self.subTest(solver=solver.__name__):
                with patch.object(methods, "minres", side_effect=ZeroDivisionError) as solve:
                    x, status, _ = self.run_solver(solver)
                if solver is methods.mf_gladssn:
                    assert_array_equal(x, [0.5])
                    self.assertEqual(status, "iterations_exceeded")
                    self.assertEqual(solve.call_count, 1)
                else:
                    assert_array_equal(x, [1.0])
                    self.assertEqual(status, "line_search_failed, 0 iters")
                    self.assertEqual(solve.call_count, 120)

    def test_breakdown_does_not_materialize_a_callable_hessian(self):
        with patch.object(methods, "minres", side_effect=ZeroDivisionError) as solve:
            x, status, _ = self.run_solver(methods.mf_gladssn, QuadraticOracle(explicit=False))
        assert_array_equal(x, [1.0])
        self.assertEqual(status, "line_search_failed, 0 iters")
        self.assertEqual(solve.call_count, 120)

    def test_iteration_limited_minres_step_remains_eligible_but_cg_retries(self):
        for solver in (methods.mf_leapssn, methods.mf_gladssn):
            for inner in ("cg", "minres"):
                with self.subTest(solver=solver.__name__, inner=inner):
                    with patch.object(methods, inner, return_value=(np.array([-0.5]), 1)):
                        x, status, _ = self.run_solver(solver, lss_method=inner)
                    if inner == "minres":
                        assert_array_equal(x, [0.5])
                        self.assertEqual(status, "iterations_exceeded")
                    else:
                        assert_array_equal(x, [1.0])
                        self.assertEqual(status, "line_search_failed, 0 iters")

    def test_glad_zero_exponent_retains_its_own_regularization_update(self):
        for solver, options, expected_H in (
            (methods.mf_leapssn, {}, 0.5),
            (methods.mf_gladssn, {"p": 0.0}, 0.25),
        ):
            with self.subTest(solver=solver.__name__):
                with patch.object(methods, "minres", return_value=(np.array([-0.5]), 0)):
                    _, _, history = self.run_solver(solver, **options)
                self.assertEqual(history["H_k"][-1], expected_H)

    def test_failed_search_includes_work_without_changing_the_last_point(self):
        for solver in (methods.mf_leapssn, methods.mf_gladssn):
            with self.subTest(solver=solver.__name__):
                oracle = QuadraticOracle()

                class Clock:
                    @staticmethod
                    def now():
                        return datetime(2026, 1, 1) + timedelta(seconds=oracle.work)

                with patch.object(methods, "datetime", Clock):
                    with patch.object(methods, "minres", return_value=(np.ones(1), 0)):
                        x, status, history = self.run_solver(solver, oracle)
                assert_array_equal(x, [1.0])
                self.assertEqual(status, "line_search_failed, 0 iters")
                self.assertEqual(history["func"], [0.5])
                self.assertEqual(history["grad_norm"], [1.0])
                self.assertEqual(history["grad_calls"][-1], oracle.grad_calls)
                self.assertEqual(history["hess_calls"][-1], oracle.hess_calls)
                self.assertEqual(history["time"][-1], oracle.work)
                self.assertGreater(oracle.grad_calls, 1)


if __name__ == "__main__":
    unittest.main()
