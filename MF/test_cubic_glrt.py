"""GLRT integration checks; native tests require galahad-optrove."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import norm

from cubic_glrt import GLRTCubicSubproblem, cubic_newton_step_glrt
from methods import cubic_newton_nonconvex_new
from cubic_subproblems import cubic_newton_step_nonconvex_new
from oracles_mat_fac import PositiveMatrixFactorization


class ReplayGLRT:
    """Native-interface double that checks reverse-communication data flow."""

    def __init__(self, g, A, step, status=0, raises=False):
        self.g, self.A, self.step = g.copy(), A, step
        self.status, self.raises = status, raises
        self.closed = False
        self.calls = 0

    def initialize(self):
        return {}

    def load_options(self, options):
        self.options = options

    def solve_problem(self, status, n, power, weight, r, v):
        self.calls += 1
        assert power == 3.
        assert n == len(self.g)
        self.weight = weight
        if self.calls == 1:
            assert status == 1
            assert_allclose(r, self.g)
            r[:] = 123.  # Native workspace must not alias the caller's g.
            v[:] = np.arange(n) + 1.
            self.probe = v.copy()
            return 3, self.step, r, v
        if self.calls == 2:
            assert status == 3
            assert_allclose(v, self.A @ self.probe)
            return 4, self.step, r, v
        if self.calls == 3:
            assert status == 4
            assert_allclose(r, self.g)
            v[:] = self.probe
            return 2, self.step, r, v
        assert status == 2
        assert_allclose(v, self.probe)  # Identity inverse metric.
        if self.raises:
            raise RuntimeError("native failure")
        return self.status, self.step, r, v

    def information(self):
        return dict(status=self.status, iter=4, iter_pass2=1)

    def terminate(self):
        self.closed = True


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.A = np.diag([2., 3.])
        self.g = np.array([1., -2.])
        self.step, _, status = cubic_newton_step_nonconvex_new(
            self.g, self.A, .5, eps=1e-12)
        self.assertEqual(status, "success")

    def test_reverse_communication_inputs_residual_and_cost(self):
        native = ReplayGLRT(self.g, self.A, self.step)
        original = self.g.copy()
        problem = GLRTCubicSubproblem(self.g, self.A)
        with patch("cubic_glrt._load_glrt", return_value=native):
            step, model, status = problem.solve(.5, eps=1e-8, max_iter=40)
        self.assertEqual(status, "success")
        assert_allclose(self.g, original, atol=0, rtol=0)
        self.assertTrue(native.closed)
        self.assertEqual(native.weight, .5)
        self.assertEqual(native.options["itmax"], 40)
        self.assertEqual(native.options["extra_vectors"], 41)
        self.assertTrue(native.options["unitm"])
        self.assertEqual(native.options["fraction_opt"], 1.)
        self.assertEqual(problem.hvp_calls, 2)  # Requested product + final check.
        self.assertEqual(problem.iterations, 4)
        self.assertLess(problem.relative_residual, 1e-8)
        assert_allclose(model, self.g @ step + .5 * step @ self.A @ step
                        + .5 / 3 * norm(step)**3)

    def test_negative_status_and_raised_native_failure(self):
        for code, raises in [(-18, False), (-18, True), (-15, True)]:
            with self.subTest(code=code, raises=raises):
                native = ReplayGLRT(self.g, self.A, self.step, code, raises)
                problem = GLRTCubicSubproblem(self.g, self.A)
                with patch("cubic_glrt._load_glrt", return_value=native):
                    step, value, status = problem.solve(.5)
                self.assertEqual(status, "iterations_exceeded" if code == -18
                                 else "glrt_error_-15")
                assert_allclose(step, 0.)
                self.assertTrue(np.isnan(value))
                self.assertEqual(problem.native_status, code)
                self.assertTrue(native.closed)

    def test_unconverged_native_success_is_rejected(self):
        native = ReplayGLRT(self.g, self.A, np.zeros(2))
        with patch("cubic_glrt._load_glrt", return_value=native):
            _, value, status = cubic_newton_step_glrt(self.g, self.A, .5)
        self.assertEqual(status, "residual_tolerance_not_met")
        self.assertTrue(np.isnan(value))

    def test_operator_exception_and_nonfinite_product_release_native_state(self):
        def broken(v):
            raise ValueError("operator failure")

        for operator in (broken, lambda v: np.full_like(v, np.nan)):
            native = ReplayGLRT(self.g, self.A, self.step)
            with patch("cubic_glrt._load_glrt", return_value=native):
                if operator is broken:
                    with self.assertRaisesRegex(ValueError, "operator failure"):
                        cubic_newton_step_glrt(self.g, operator, .5)
                else:
                    self.assertEqual(cubic_newton_step_glrt(self.g, operator, .5)[2],
                                     "nonfinite_hessian_product")
            self.assertTrue(native.closed)

    def test_missing_dependency_has_install_instructions(self):
        with patch.dict("sys.modules", {"galahad": None}):
            with self.assertRaisesRegex(ImportError, "galahad-optrove"):
                cubic_newton_step_glrt(self.g, self.A, .5)

    def test_nonfinite_product_retains_native_work_before_termination(self):
        native = ReplayGLRT(self.g, self.A, self.step)
        problem = GLRTCubicSubproblem(self.g, lambda v: np.full_like(v, np.nan))
        with patch("cubic_glrt._load_glrt", return_value=native):
            _, value, status = problem.solve(.5)
        self.assertEqual(status, "nonfinite_hessian_product")
        self.assertTrue(np.isnan(value))
        self.assertEqual(problem.hvp_calls, 1)
        self.assertEqual(problem.last_iterations, 4)
        self.assertEqual(problem.iterations, 4)
        self.assertEqual(problem.native_status, native.status)
        self.assertTrue(native.closed)

    def test_invalid_controls_and_zero_gradient_policy(self):
        for kwargs in ({"H": 0}, {"H": np.inf}, {"H": 1, "eps": 1},
                       {"H": 1, "max_iter": 0}, {"H": 1, "max_iter": True}):
            with self.assertRaises(ValueError):
                cubic_newton_step_glrt(self.g, self.A, **kwargs)
        with patch("cubic_glrt._load_glrt"):
            step, value, status = cubic_newton_step_glrt(np.zeros(2), -self.A, 1.)
        assert_allclose(step, 0.)
        self.assertEqual((value, status), (0., "success"))

    def test_outer_failure_preserves_point_and_reports_work(self):
        class Quadratic:
            def func(_, x): return .5 * x @ self.A @ x
            def grad(_, x): return self.A @ x
            def hess(_, x): return lambda v: self.A @ v
            def penalty(_, x): return 0.

        x0 = np.ones(2)
        native = ReplayGLRT(self.A @ x0, self.A, self.step, -18)
        with patch("cubic_glrt._load_glrt", return_value=native):
            x, status, history = cubic_newton_nonconvex_new(
                Quadratic(), x0, subproblem_solver="glrt", n_iters=2, H_0=1.)
        assert_allclose(x, x0, atol=0, rtol=0)
        self.assertEqual(status, "subproblem_iterations_exceeded, 0 iters")
        self.assertEqual(history["subproblem_calls"], [1])
        self.assertEqual(history["glrt_status"], [-18])
        self.assertEqual(history["hess_vec_calls"], [1])
        self.assertEqual(native.weight, .5)  # Outer H_k / 6 -> GLRT sigma=H_k/2.


@unittest.skipUnless(importlib.util.find_spec("galahad"), "install galahad-optrove")
class NativeGLRTTests(unittest.TestCase):
    def test_solution_at_exact_iteration_cap_preserves_reconstruction(self):
        A = np.diag([2., 3.])
        g = np.array([1., -2.])
        problem = GLRTCubicSubproblem(g, A)
        step, _, status = problem.solve(.5, eps=1e-8, max_iter=2)
        self.assertEqual(status, "success")
        self.assertLess(norm(g + A @ step + .5 * norm(step) * step), 1e-8)
        self.assertEqual(problem.hvp_calls, 3)  # No reverse calls to reconstruct.

    def test_convex_and_generic_indefinite_models_against_dense(self):
        rng = np.random.default_rng(31)
        for spectrum in (np.arange(1., 13.), np.linspace(-5., 8., 12)):
            Q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
            A = Q @ np.diag(spectrum) @ Q.T
            g = rng.normal(size=12)
            for weight in (.7, 2.):
                problem = GLRTCubicSubproblem(g, lambda v: A @ v)
                step, model, status = problem.solve(weight, eps=1e-8, max_iter=100)
                _, expected, reference_status = cubic_newton_step_nonconvex_new(
                    g, A, weight, eps=1e-11)
                self.assertEqual((status, reference_status), ("success", "success"))
                assert_allclose(model, expected, rtol=2e-8, atol=1e-10)
                self.assertLessEqual(norm(g + A @ step + weight * norm(step) * step),
                                     1.1e-8 * norm(g))
                self.assertGreaterEqual(np.linalg.eigvalsh(A)[0] + weight * norm(step),
                                        -1e-8)

    def test_hidden_negative_curvature_is_not_a_global_certificate(self):
        A = np.diag([-10., 1.])
        g = np.array([0., 1.])
        step, model, status = cubic_newton_step_glrt(g, A, 1., eps=1e-8)
        _, global_model, _ = cubic_newton_step_nonconvex_new(g, A, 1.)
        self.assertEqual(status, "success")
        assert_allclose(g + A @ step + norm(step) * step, 0., atol=1e-8)
        self.assertLess(norm(step) - 10., 0.)
        self.assertLess(global_model, model - 1.)

    def test_native_iteration_limit_and_subsequent_fresh_solve(self):
        A = np.diag(np.linspace(-3., 100., 50))
        problem = GLRTCubicSubproblem(np.ones(50), A)
        _, _, status = problem.solve(1., eps=1e-10, max_iter=1)
        self.assertEqual(status, "iterations_exceeded")
        self.assertIn(problem.native_status, (0, -18))
        self.assertEqual(problem.solve(1., eps=1e-8, max_iter=200)[2], "success")

    def test_small_mf_outer_run_matches_dense_reference(self):
        for seed in (549, 498):
            rng = np.random.RandomState(seed)
            m, n, r = 6, 4, 2
            data = rng.rand(m, r) @ rng.rand(n, r).T + .5 * rng.randn(m, n)
            oracle = PositiveMatrixFactorization(data, r=r, alpha=10., beta=200.)
            initial = 1.5 * rng.randn((m + n) * r)
            x0 = oracle.pack(initial[:m*r].reshape(m, r), initial[m*r:].reshape(n, r).T)
            expected, ds, dh = cubic_newton_nonconvex_new(oracle, x0, n_iters=5, H_0=10.)
            x, status, h = cubic_newton_nonconvex_new(
                oracle, x0, n_iters=5, H_0=10., subproblem_solver="glrt", subproblem_tol=1e-8)
            self.assertEqual((status, ds), ("iterations_exceeded", "iterations_exceeded"))
            assert_allclose(h["func"], dh["func"], rtol=2e-7, atol=1e-8)
            assert_allclose(x, expected, rtol=1e-5, atol=2e-6)
            self.assertTrue(np.all(np.diff(h["func"]) <= 0.))
            self.assertTrue(all(s == 0 for s in h["glrt_status"][1:]))
            self.assertLess(max(h["subproblem_relative_residual"][1:]), 1.1e-8)


if __name__ == "__main__":
    unittest.main()
