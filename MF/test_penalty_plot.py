"""Penalty plots must show zeros and later returns to a positive penalty."""

import contextlib
import io
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

from utils_new import plot_figures


class PenaltyPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def plot(self, values, measure="penalty"):
        history = {"func": np.ones(len(values)), "time": np.arange(len(values)),
                   measure: np.array(values)}
        with contextlib.redirect_stdout(io.StringIO()):
            figure = plot_figures(
                [history], ["method"], n=1, m=1, mu_str=None,
                colors=["blue"], linewidths=[2], linestyles=["-"],
                f_star=0, threshold=1e-10, max_iter=len(values)-1,
                which_measure_1=measure, which_measure_2=measure)
        return figure, history

    def test_positive_penalty_after_zero_is_not_truncated(self):
        values = [1.0, 0.0, 0.25, 0.0]
        figure, history = self.plot(values)
        for axis in figure.axes:
            assert_array_equal(axis.lines[0].get_xdata(), np.arange(4))
            assert_array_equal(axis.lines[0].get_ydata(), values)
            self.assertEqual(axis.get_yscale(), "symlog")
        assert_array_equal(history["penalty"], values)

    def test_all_zero_penalty_is_visible_without_log_warning(self):
        for length in (1, 4):
            with self.subTest(length=length), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                figure, _ = self.plot([0.0] * length)
                figure.canvas.draw()
            self.assertFalse(any("log" in str(w.message).lower() for w in caught))
            for axis in figure.axes:
                assert_array_equal(axis.lines[0].get_ydata(), np.zeros(length))
                self.assertLess(axis.get_ylim()[0], 0)
                self.assertGreater(axis.get_ylim()[1], 0)
                self.assertTrue(np.isfinite(axis.transData.transform((0, 0))).all())
                if length == 1:
                    self.assertEqual(axis.lines[0].get_marker(), "o")

    def test_gradient_plot_keeps_its_existing_threshold_rule(self):
        figure, _ = self.plot([1.0, 1e-12, 0.25], measure="grad_norm")
        for axis in figure.axes:
            self.assertEqual(axis.get_yscale(), "log")
            assert_array_equal(axis.lines[0].get_ydata(), [1.0, 1e-12])


if __name__ == "__main__":
    unittest.main()
