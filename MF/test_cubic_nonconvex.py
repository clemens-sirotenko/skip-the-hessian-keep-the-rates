"""Global-optimality checks for the additional dense cubic subproblem solver."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator

from methods import cubic_newton_nonconvex_new
from cubic_subproblems import cubic_newton_step_nonconvex_new
from methods import _prepare_cubic_nonconvex
from oracles_mat_fac import PositiveMatrixFactorization


class CubicNonconvexTests(unittest.TestCase):
    def check_solution(self, g, A, H, B=None, operator=None):
        """Check the global characterization independently in original coordinates."""
        step, model, status = cubic_newton_step_nonconvex_new(
            g, A if operator is None else operator, H, B=B, eps=1e-11,
        )
        self.assertEqual(status, "success")
        metric = np.eye(len(g)) if B is None else B
        radius = np.sqrt(step @ metric @ step)
        shift = H * radius
        residual = g + A @ step + shift * (metric @ step)
        scale = linalg.norm(g) + (linalg.norm(A, 2) + shift * linalg.norm(metric, 2)) * linalg.norm(step)
        self.assertLessEqual(linalg.norm(residual), 5e-10 * scale)
        smallest = linalg.eigvalsh(A + shift * metric)[0]
        self.assertGreaterEqual(smallest, -5e-10 * (linalg.norm(A, 2) + shift * linalg.norm(metric, 2)))
        expected_model = g @ step + 0.5 * step @ A @ step + H * radius**3 / 3
        assert_allclose(model, expected_model, rtol=1e-10, atol=1e-14)
        self.assertLessEqual(model, 1e-12)
        return step, model

    def test_indefinite_scalar_selects_global_minimum(self):
        step, _ = self.check_solution(np.array([1.]), np.array([[-2.]]), 1.)
        assert_allclose(step, [-1 - np.sqrt(2)], rtol=1e-11)

    def test_positive_scalar_matches_analytic_solution(self):
        step, _ = self.check_solution(np.array([1.]), np.array([[2.]]), 1.)
        assert_allclose(step, [1 - np.sqrt(2)], rtol=1e-11)

    def test_zero_hessian_matches_analytic_solution(self):
        g = np.array([3., -4.])
        step, _ = self.check_solution(g, np.zeros((2, 2)), 2.)
        assert_allclose(step, -g / np.sqrt(2 * linalg.norm(g)), rtol=1e-11)

    def test_zero_gradient_escapes_negative_curvature(self):
        step, model = self.check_solution(np.zeros(2), np.diag([-2., 3.]), 1.)
        assert_allclose(np.abs(step), [2., 0.], atol=1e-12)
        self.assertAlmostEqual(model, -4 / 3)

    def test_zero_gradient_psd_including_singular_and_zero_matrices(self):
        for diagonal in ([1., 2.], [0., 2.], [0., 0.]):
            with self.subTest(diagonal=diagonal):
                step, model = self.check_solution(np.zeros(2), np.diag(diagonal), 1.)
                assert_allclose(step, 0.)
                self.assertEqual(model, 0.)

    def test_hard_case_adds_missing_eigenvector_component(self):
        step, _ = self.check_solution(np.array([0., 1.]), np.diag([-2., 1.]), 1.)
        self.assertAlmostEqual(linalg.norm(step), 2.)
        self.assertAlmostEqual(step[1], -1 / 3)
        self.assertAlmostEqual(abs(step[0]), np.sqrt(4 - 1 / 9))

    def test_orthogonal_gradient_can_still_require_an_interior_root(self):
        step, _ = self.check_solution(np.array([0., 10.]), np.diag([-2., 1.]), 1.)
        self.assertGreater(linalg.norm(step), 2.)
        self.assertAlmostEqual(step[0], 0.)

    def test_rotated_repeated_eigenvalues_and_near_hard_case(self):
        rng = np.random.default_rng(7)
        Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        A = Q @ np.diag([-2., -2., 1., 3.]) @ Q.T
        for small in (0., 1e-14, 1e-8):
            with self.subTest(small=small):
                g = Q @ np.array([small, 0., 1., .25])
                self.check_solution(g, A, 1.)

    def test_nonidentity_metric_matches_transformed_problem(self):
        L = np.array([[2., 0.], [.5, 1.]])
        B = L @ L.T
        D = np.diag([-2., 1.])
        g = L @ np.array([0., 1.])
        step, _ = self.check_solution(g, L @ D @ L.T, 1., B=B)
        transformed = L.T @ step
        assert_allclose(np.abs(transformed), [np.sqrt(4 - 1 / 9), 1 / 3], rtol=1e-10)

    def test_input_representations_and_no_mutation(self):
        A = np.array([[-2., .2], [.2, 1.]])
        g = np.array([1., -.3])
        original_A, original_g = A.copy(), g.copy()
        outputs = []
        for operator in (A, sparse.csr_matrix(A), lambda v: A @ v,
                         LinearOperator(A.shape, matvec=lambda v: A @ v)):
            outputs.append(self.check_solution(g, A, .7, operator=operator)[0])
        for step in outputs[1:]:
            assert_allclose(step, outputs[0], rtol=1e-12)
        assert_allclose(A, original_A, rtol=0, atol=0)
        assert_allclose(g, original_g, rtol=0, atol=0)

    def test_random_symmetric_problems_and_objective_scaling(self):
        rng = np.random.default_rng(23)
        for index in range(12):
            Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
            A = Q @ np.diag(rng.uniform(-3, 4, 5)) @ Q.T
            g = rng.normal(size=5)
            baseline, _ = self.check_solution(g, A, .7)
            for scale in (1e-10, 1e10):
                with self.subTest(index=index, scale=scale):
                    step, _ = self.check_solution(scale * g, scale * A, scale * .7)
                    assert_allclose(step, baseline, rtol=1e-8, atol=1e-10)

    def test_invalid_inputs_are_rejected(self):
        cases = [
            dict(H=0.), dict(H=-1.), dict(H=np.inf), dict(H=1.+0j),
            dict(g=np.array([np.nan, 1.])), dict(g=np.ones((2, 1))),
            dict(H_mv=np.ones((3, 3))), dict(H_mv=np.array([[1., 1.], [0., 1.]])),
            dict(H_mv=lambda v: np.ones(3)), dict(B=np.diag([1., 0.])),
            dict(B=np.diag([1., -1.])), dict(eps=0.), dict(eps=1e-8+0j), dict(max_iter=0),
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                options = dict(g=np.ones(2), H_mv=np.eye(2), H=1.)
                options.update(overrides)
                with self.assertRaises(ValueError):
                    cubic_newton_step_nonconvex_new(**options)

    def test_iteration_limit_is_not_success_or_zero_model_decrease(self):
        _, model, status = cubic_newton_step_nonconvex_new(
            np.ones(3), np.diag([-2., 1., 4.]), 1., eps=1e-14, max_iter=1,
        )
        self.assertEqual(status, "iterations_exceeded")
        self.assertTrue(np.isnan(model))

    def test_prepared_model_matches_fresh_solves_across_cubic_coefficients(self):
        A = np.diag([-2., 1.])
        for g in (np.array([0., 1.]), np.array([1e-12, 1.]), np.ones(2)):
            for B in (None, np.diag([2., .5])):
                solve = _prepare_cubic_nonconvex(g, A, B)
                # Cross between hard-case and interior solutions, then repeat.
                for H in (.1, 1., 10., .1):
                    with self.subTest(g=g, B=B, H=H):
                        step, model, status = solve(H, eps=1e-11)
                        expected_step, expected_model = self.check_solution(g, A, H, B=B)
                        self.assertEqual(status, "success")
                        assert_allclose(step, expected_step, rtol=0, atol=0)
                        self.assertEqual(model, expected_model)

    def test_unrepresentable_model_reports_numerical_failure(self):
        for g, A, H in (([1e100], [[-1e100]], 1e-100),
                        ([1e200], [[1.]], 1e-200)):
            with self.subTest(g=g):
                step, model, status = cubic_newton_step_nonconvex_new(g, A, H)
                self.assertEqual(status, "numerical_error")
                self.assertTrue(np.isnan(model))
                assert_allclose(step, 0.)

    def test_small_mf_steps_with_fresh_hessians_and_checked_acceptance(self):
        # This exercises the oracle-to-subproblem interface, not the old outer
        # cubic_newton implementation, which deliberately remains unchanged.
        rng = np.random.default_rng(549)
        m, n, rank = 6, 4, 2
        M = rng.random((m, rank)) @ rng.random((rank, n))
        oracle = PositiveMatrixFactorization(M, rank, alpha=.5, beta=200.)
        x = .5 * rng.normal(size=(m + n) * rank)
        coefficient = 1.
        initial_value = oracle.func(x)
        for _ in range(5):
            value, gradient, hess = oracle.func(x), oracle.grad(x), oracle.hess(x)
            accepted = False
            for _ in range(30):
                step, model, status = cubic_newton_step_nonconvex_new(
                    gradient, hess, coefficient, eps=1e-10,
                )
                self.assertEqual(status, "success")
                if oracle.func(x + step) <= value + model:
                    x = x + step
                    coefficient *= .5
                    accepted = True
                    break
                coefficient *= 2
            self.assertTrue(accepted)
            self.assertLessEqual(oracle.func(x), value)
        self.assertLess(oracle.func(x), initial_value)
        self.assertTrue(np.isfinite(linalg.norm(oracle.grad(x))))
        self.assertTrue(np.isfinite(oracle.penalty(x)))


class CubicOuterTests(unittest.TestCase):
    class Quadratic:
        def func(self, x):
            return .5 * x @ x

        def grad(self, x):
            return x.copy()

        def hess(self, x):
            return np.eye(x.size)

        def penalty(self, x):
            return 0.

    def test_subproblem_failure_retries_and_preserves_iterate(self):
        solve = Mock(return_value=(np.zeros(1), np.nan, "iterations_exceeded"))
        with patch("methods._prepare_cubic_nonconvex", return_value=solve):
            x, status, history = cubic_newton_nonconvex_new(
                self.Quadratic(), np.ones(1), n_iters=2, max_backtracks=2,
            )
        assert_allclose(x, [1.])
        self.assertEqual(status, "subproblem_iterations_exceeded, 0 iters")
        self.assertEqual(history["func"], [.5])
        self.assertEqual(history["subproblem_calls"][-1], 3)
        self.assertEqual([call.args[0] for call in solve.call_args_list], [.5, 1., 2.])

    def test_rejected_trial_is_never_applied(self):
        solve = Mock(return_value=(np.ones(1), -.1, "success"))
        with patch("methods._prepare_cubic_nonconvex", return_value=solve):
            x, status, history = cubic_newton_nonconvex_new(
                self.Quadratic(), np.ones(1), n_iters=2, max_backtracks=2,
            )
        assert_allclose(x, [1.])
        self.assertEqual(status, "line_search_failed, 0 iters")
        self.assertEqual(history["func"], [.5])

    def test_failure_after_acceptance_returns_consistent_final_history(self):
        calls = 0

        def one_good_preparation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _prepare_cubic_nonconvex(*args, **kwargs)
            return Mock(return_value=(np.zeros(1), np.nan, "numerical_error"))

        oracle = self.Quadratic()
        with patch("methods._prepare_cubic_nonconvex", side_effect=one_good_preparation):
            x, status, history = cubic_newton_nonconvex_new(
                oracle, np.ones(1), n_iters=3, max_backtracks=1,
            )
        self.assertEqual(status, "subproblem_numerical_error, 1 iters")
        self.assertEqual(len(history["func"]), 2)
        self.assertAlmostEqual(history["func"][-1], oracle.func(x))
        self.assertAlmostEqual(history["grad_norm"][-1], linalg.norm(oracle.grad(x)))
        self.assertEqual(history["subproblem_calls"][-1], 3)

    def test_trace_false_and_matching_gradient_history(self):
        oracle = self.Quadratic()
        x, status, history = cubic_newton_nonconvex_new(oracle, np.ones(2), n_iters=3)
        y, status_without_trace, no_history = cubic_newton_nonconvex_new(
            oracle, np.ones(2), n_iters=3, trace=False,
        )
        assert_allclose(x, y)
        self.assertEqual(status, status_without_trace)
        self.assertIsNone(no_history)
        self.assertTrue(np.all(np.diff(history["func"]) < 0))
        self.assertTrue(np.all(np.diff(history["time"]) >= 0))
        self.assertAlmostEqual(history["grad_norm"][-1], linalg.norm(oracle.grad(x)))

    def test_zero_gradient_with_negative_curvature_takes_a_step(self):
        class Quartic(self.Quadratic):
            def func(self, x):
                return -x[0]**2 + x[0]**4 / 4

            def grad(self, x):
                return -2 * x + x**3

            def hess(self, x):
                return np.diag(-2 + 3 * x**2)

        x, status, history = cubic_newton_nonconvex_new(
            Quartic(), np.zeros(1), n_iters=1, H_0=2.,
        )
        self.assertNotEqual(x[0], 0.)
        self.assertLess(history["func"][-1], 0.)
        self.assertEqual(status, "iterations_exceeded")

    def test_zero_gradient_with_psd_hessian_stops(self):
        x, status, _ = cubic_newton_nonconvex_new(self.Quadratic(), np.zeros(1))
        assert_allclose(x, 0.)
        self.assertEqual(status, "success, 0 iters")

    def test_unrepresentable_step_reports_stagnation(self):
        solve = Mock(return_value=(np.array([-1e-300]), -1e-300, "success"))
        with patch("methods._prepare_cubic_nonconvex", return_value=solve):
            x, status, _ = cubic_newton_nonconvex_new(self.Quadratic(), np.ones(1))
        self.assertEqual(status, "numerical_stagnation, 0 iters")
        assert_allclose(x, 1.)
        self.assertEqual(solve.call_count, 1)

    def test_reuse_preserves_trials_and_refreshes_at_each_new_point(self):
        class Quartic(self.Quadratic):
            def __init__(self):
                self.hessian_points = []
                self.products = 0

            def func(self, x):
                return -x @ x + .25 * np.sum(x**4)

            def grad(self, x):
                return -2 * x + x**3

            def hess(self, x):
                self.hessian_points.append(x.copy())
                diagonal = -2 + 3 * x**2

                def matvec(v):
                    self.products += 1
                    return diagonal * v

                return matvec

        initial = np.array([.1, .2])
        settings = dict(n_iters=3, H_0=.001)
        oracle = Quartic()
        with patch("methods.scipy.linalg.eigh", wraps=linalg.eigh) as eig:
            x, status, history = cubic_newton_nonconvex_new(oracle, initial, **settings)
        self.assertEqual(eig.call_count, 3)
        self.assertEqual(oracle.products, 3 * initial.size)
        self.assertEqual(len(oracle.hessian_points), 3)
        self.assertTrue(all(not np.array_equal(a, b) for a, b in
                            zip(oracle.hessian_points, oracle.hessian_points[1:])))
        self.assertGreater(history["subproblem_calls"][-1], 3)

        # Reference policy: rebuild the same model for every coefficient trial.
        def prepare_without_reuse(g, A, B=None):
            return lambda H, **kw: _prepare_cubic_nonconvex(g, A, B)(H, **kw)

        with patch("methods._prepare_cubic_nonconvex", side_effect=prepare_without_reuse):
            expected_x, expected_status, expected_history = cubic_newton_nonconvex_new(
                Quartic(), initial, **settings,
            )
        assert_allclose(x, expected_x, rtol=0, atol=0)
        self.assertEqual(status, expected_status)
        for key in history:
            if key != "time":
                assert_allclose(history[key], expected_history[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
