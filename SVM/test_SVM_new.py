"""Focused regression checks for the companion SVM files (no reference writes)."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

SVM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SVM_DIR))
import SVM_oracles as original
import SVM_oracles_new as corrected
from SVM_utils_new import _residual_series, plot_figures, plot_for_paper_2


def notebook_class(new=True):
    name = "experiment_SVM_new.ipynb" if new else "experiment_SVM.ipynb"
    nb = json.loads((SVM_DIR / name).read_text(encoding="utf-8"))
    namespace = {}
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            exec(source, namespace)
            if "class SupportVectorClassification" in source:
                return namespace["SupportVectorClassification"]
    raise AssertionError("Missing SVM class")


SVC = notebook_class()
OldSVC = notebook_class(False)


class Quadratic(corrected.OptimizationMethods):
    x = np.zeros((3, 2))

    def func(self, z):
        return 0.5 * z.dot(z)

    def grad(self, z):
        return z.copy()

    def grad_i(self, z, i):
        return z.copy()

    def hess(self, z):
        return np.eye(len(z))


class SVMRegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(43)
        self.x = np.c_[rng.normal(size=(120, 8)), np.ones(120)]
        self.y = rng.choice([-1, 1], 120)
        self.svc = SVC(self.x, self.y, 10.0)
        self.z = np.full(9, 0.5)

    def test_derivatives_and_sample_gradient_scaling(self):
        rng = np.random.default_rng(1)
        z, v = rng.normal(size=(2, 9))
        eps = 1e-6
        derivative = (self.svc.func(z + eps*v) - self.svc.func(z - eps*v)) / (2*eps)
        np.testing.assert_allclose(derivative, self.svc.grad(z).dot(v), rtol=1e-7)
        hv = (self.svc.grad(z + eps*v) - self.svc.grad(z - eps*v)) / (2*eps)
        np.testing.assert_allclose(hv, self.svc.hess(z) @ v, rtol=1e-7, atol=1e-6)
        average = np.mean([self.svc.grad_i(z, i) for i in range(120)], axis=0)
        np.testing.assert_allclose(average, self.svc.grad(z), rtol=1e-12, atol=1e-10)

    def test_valid_legacy_trajectories_are_preserved(self):
        old = OldSVC(self.x, self.y, 10.0)
        cases = [
            ("gradient_method", {"max_iter": 10}),
            ("cubic_newton", {"n_iters": 4}),
            ("leapssn", {"n_iters": 6, "alpha": .1, "beta": .1, "accept_tol": 1e-8}),
            ("gladssn", {"n_iters": 6, "alpha": .1, "beta": .1, "accept_tol": 1e-8,
                         "lazy_Hessians": True, "update_freq": 5}),
            ("svrg_bb", {"n_epochs": 10, "eta_0": 1e-5}),
        ]
        for name, kwargs in cases:
            with self.subTest(method=name), contextlib.redirect_stdout(io.StringIO()):
                np.random.seed(0)
                old_x, _, old_h = getattr(old, name)(self.z, **kwargs)
                np.random.seed(0)
                new_x, _, new_h = getattr(self.svc, name)(self.z, **kwargs)
                np.testing.assert_allclose(new_x, old_x, rtol=1e-12, atol=1e-11)
                np.testing.assert_allclose(new_h["func"], old_h["func"], rtol=1e-12, atol=1e-10)

    def test_rejected_svm_steps_do_not_change_iterate(self):
        svc = SVC(np.c_[np.zeros(101), np.ones(101)], np.r_[np.ones(100), -1], 1e4)
        z = np.array([0., 1.1])
        for name in ("gladssn", "leapssn"):
            with self.subTest(method=name):
                out, status, h = getattr(svc, name)(z, n_iters=1, adaptive_search_max_iter=1,
                                                   alpha=.1, beta=.1)
                np.testing.assert_array_equal(out, z)
                self.assertEqual(status, "adaptive_iterations_exceeded")
                self.assertEqual(len(h["func_at_acceptance"]), 0)
                self.assertEqual(h["matrix_inverses"][-1], 1)

    def test_gradient_and_cubic_failures_do_not_change_iterate(self):
        q, z = Quadratic(), np.ones(2)
        out, status, _ = q.gradient_method(z, max_iter=1, L_0=.1, line_search_max_iter=1)
        np.testing.assert_array_equal(out, z)
        self.assertEqual(status, "line_search_max_iter_reached")
        failed = (np.zeros(2), 0.0, "linalg_error")
        with patch.object(corrected.OptimizationMethods, "cubic_newton_step", return_value=failed):
            out, status, _ = q.cubic_newton(z, n_iters=1, adaptive_search=False)
            np.testing.assert_array_equal(out, z)
            self.assertEqual(status, "subproblem_linalg_error")
            out, status, _ = q.cubic_newton(z, n_iters=1)
            np.testing.assert_array_equal(out, z)
            self.assertEqual(status, "adaptive_iterations_exceeded")

    def test_cholesky_failure_retries_without_using_stale_step(self):
        factor = corrected.scipy.linalg.cho_factor
        for name in ("gladssn", "leapssn"):
            calls = []

            def fail_once(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise np.linalg.LinAlgError("synthetic factorization failure")
                return factor(*args, **kwargs)

            with self.subTest(method=name), patch.object(corrected.scipy.linalg, "cho_factor", fail_once):
                out, status, h = getattr(Quadratic(), name)(np.ones(2), n_iters=1,
                                                           adaptive_search_max_iter=3)
                self.assertEqual(status, "iterations_exceeded")
                self.assertLess(np.linalg.norm(out), np.sqrt(2))
                self.assertEqual(h["matrix_inverses"][-1], 1)

    def test_trace_false_matches_trace_true(self):
        for name in ("gradient_method", "cubic_newton", "leapssn", "gladssn", "svrg", "svrg_bb"):
            kwargs = ({"max_iter": 2} if name == "gradient_method" else
                      {"n_epochs": 2} if name.startswith("svrg") else {"n_iters": 2})
            with self.subTest(method=name):
                np.random.seed(0)
                expected, expected_status, _ = getattr(self.svc, name)(self.z, **kwargs)
                np.random.seed(0)
                actual, status, history = getattr(self.svc, name)(self.z, trace=False, **kwargs)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(status, expected_status)
                self.assertIsNone(history)

    def test_identity_metric_is_supported(self):
        for name in ("cubic_newton", "leapssn", "gladssn"):
            with self.subTest(method=name), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                expected, _, _ = getattr(self.svc, name)(self.z, n_iters=2)
                out, _, _ = getattr(self.svc, name)(self.z, n_iters=2, B=np.eye(9))
                np.testing.assert_allclose(out, expected, atol=1e-13)

    def test_stationary_initial_point_is_success(self):
        for name in ("gradient_method", "cubic_newton", "leapssn", "gladssn", "svrg", "svrg_bb"):
            with self.subTest(method=name), np.errstate(all="raise"):
                out, status, _ = getattr(Quadratic(), name)(np.zeros(2))
                self.assertTrue(status.startswith("success"), status)
                np.testing.assert_array_equal(out, np.zeros(2))
        step, value, status = corrected.OptimizationMethods.cubic_newton_step(np.zeros(2), np.eye(2), 1.)
        self.assertEqual(status, "success")
        self.assertEqual(value, 0.)

    def test_bb_zero_curvature_reports_failure_without_nan(self):
        class Linear(Quadratic):
            def func(self, z): return float(z.sum())
            def grad(self, z): return np.ones_like(z)
            def grad_i(self, z, i): return np.ones_like(z)
        with np.errstate(all="raise"):
            out, status, _ = Linear().svrg_bb(np.ones(2), n_epochs=3, eta_0=.1)
        self.assertEqual(status, "invalid_bb_curvature")
        self.assertTrue(np.isfinite(out).all())

    def test_hessian_and_sample_counters(self):
        _, _, h = self.svc.gladssn(self.z, n_iters=5, lazy_Hessians=True, update_freq=5)
        self.assertEqual(h["hess_calls"][-1], 1)
        self.assertEqual(len(h["func_at_hess_eval"]), 1)
        np.random.seed(0)
        _, _, h = self.svc.svrg_bb(self.z, n_epochs=2, eta_0=1e-5)
        self.assertEqual(h["grad_calls"][-1], 3)
        self.assertEqual(h["grad_i_calls"][-1], 480)

    def test_paper_acceptance_inequalities(self):
        out, _, h = self.svc.gladssn(self.z, n_iters=1)
        step = out - self.z
        g = self.svc.grad(out)
        lam = h["reg_parameter"][0]
        self.assertGreaterEqual(g.dot(-step), .5*g.dot(g)/lam)
        self.assertGreaterEqual(self.svc.func(self.z)-self.svc.func(out), .25*lam*step.dot(step))

    def test_wrappers_use_instance_dimension_and_return_status(self):
        small = SVC(self.x[:, [0, -1]], self.y, 10.)
        for name in ("solve_with_gradient_method", "solve_with_cubic_newton", "solve_with_leapssn",
                     "solve_with_gladssn", "solve_with_svrg_bb"):
            with self.subTest(method=name):
                out, status, history = getattr(small, name)()
                self.assertEqual(out.shape, (2,))
                self.assertIsInstance(status, str)
                self.assertTrue(history["func"])

    def test_plot_uses_outer_iterates_and_honors_filename(self):
        h = {"time": [0., 1., 2.], "func": [4., 2., 1.], "grad_norm": [3., 2., 0.],
             "func_at_lssolve": [100., 50., 20., 10., 5., 1.]}
        with tempfile.TemporaryDirectory() as directory, patch.object(plt, "show"):
            output = Path(directory) / "custom_new.pdf"
            fig, axes = plot_for_paper_2([h], ["GLAd-SSN"], n=1, m=None, mu_str="",
                                        colors=["black"], linewidths=[1], linestyles=["-"],
                                        filename=output)
            self.assertTrue(output.exists())
            self.assertEqual(len(axes[1].lines[0].get_xdata()), 3)
            self.assertEqual(axes[1].get_xlabel(), "Outer iteration")
            self.assertTrue(np.all(axes[1].lines[0].get_ydata() > 0))
            plt.close(fig)
            fig, axes = plot_figures([h], ["GLAd-SSN"], n=1, m=None, mu_str="",
                                    colors=["black"], linewidths=[1], linestyles=["-"],
                                    which_measure_2="func_at_lssolve")
            self.assertIn("Trial step", axes[1].get_xlabel())
            plt.close(fig)

    def test_nonmonotone_plot_threshold(self):
        values = _residual_series({"func": [10., .01, 20., .001]}, "func", 0., .1)
        np.testing.assert_array_equal(values, [10., .1])

    def test_extreme_finite_divergence_does_not_crash_plot(self):
        h = {"time": [0., 1., 2.], "func": [1., 1e260, 1e227]}
        with tempfile.TemporaryDirectory() as directory, patch.object(plt, "show"):
            output = Path(directory) / "divergence_new.pdf"
            fig, axes = plot_for_paper_2([h], ["SVRG-BB"], n=1, m=None, mu_str="",
                                        colors=["pink"], linewidths=[1], linestyles=["-"],
                                        f_star=0., filename=output)
            self.assertTrue(output.exists())
            self.assertEqual(max(axes[0].lines[0].get_ydata()), 1e260)
            plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main(verbosity=2)
