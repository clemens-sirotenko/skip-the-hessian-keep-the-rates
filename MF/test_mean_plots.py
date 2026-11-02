"""Check the arithmetic means and time alignment independently of rendering."""

import copy
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mean_plots import mean_gradient_histories, plot_mean_gradient_histories


def history(gradient, time):
    return dict(grad_norm=np.array(gradient, dtype=float), time=np.array(time, dtype=float))


class MeanPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_short_runs_and_unsuccessful_plateaus_keep_equal_weight(self):
        runs = {1: {"A": history([10, 0], [0, 1])},
                2: {"A": history([10, 4, 2, 1], [0, 2, 3, 4])},
                3: {"A": history([10, 6], [0, 1])}}
        result = mean_gradient_histories(runs, ["A"])
        # The zero and the unsuccessful 6 remain in every later mean.
        assert_array_equal(result["iteration"]["grid"], [0, 1, 2, 3])
        assert_allclose(result["iteration"]["means"]["A"], [10, 10 / 3, 8 / 3, 7 / 3])
        assert_allclose(result["runtime"]["means"]["A"], [10, 16 / 3, 10 / 3, 8 / 3, 7 / 3])

    def test_common_runtime_grid_keeps_steps_and_last_duplicate(self):
        runs = {1: {"A": history([8, 4, 2], [0, 1, 1])},
                2: {"A": history([8, 6, 0], [0.5, 2, 3])}}
        result = mean_gradient_histories(runs, ["A"])["runtime"]
        assert_array_equal(result["grid"], [0.5, 1, 2, 3])
        assert_array_equal(result["means"]["A"], [8, 5, 4, 1])

    def test_common_budget_across_methods_does_not_hide_rebounds(self):
        runs = {1: {"A": history([5, 0, 3], [0, 1, 2]),
                    "B": history([5, 4, 2, 1], [0, 3, 4, 5])}}
        before = copy.deepcopy(runs)
        result = mean_gradient_histories(runs, ["A", "B"])
        assert_array_equal(result["iteration"]["means"]["A"], [5, 0, 3, 3])
        assert_array_equal(result["runtime"]["grid"], [0, 1, 2, 3, 4, 5])
        assert_array_equal(result["runtime"]["means"]["A"], [5, 0, 3, 3, 3, 3])
        assert_array_equal(result["runtime"]["means"]["B"], [5, 5, 5, 4, 2, 1])
        for name in runs[1]:
            for key in runs[1][name]:
                assert_array_equal(runs[1][name][key], before[1][name][key])

    def test_single_initial_point_and_zero_are_visible(self):
        runs = {1: {"A": history([0], [0])}}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fig, result = plot_mean_gradient_histories(runs, {"A": "blue"})
            fig.canvas.draw()
        for ax in fig.axes:
            assert_array_equal(ax.lines[0].get_ydata(), [0])
            self.assertEqual(ax.lines[0].get_marker(), "o")
            self.assertLess(ax.get_ylim()[0], 0)
        self.assertEqual(len(result["iteration"]["grid"]), 1)

    def test_zoom_does_not_change_averages_or_stored_full_curves(self):
        runs = {1: {"A": history([10, 2], [0, 1])},
                2: {"A": history([10, 4, 1], [0, 2, 3])}}
        fig, result = plot_mean_gradient_histories(runs, {"A": "blue"}, max_iter=1, max_time=2)
        for ax, axis, limit in zip(fig.axes, ("runtime", "iteration"), (2, 1)):
            assert_array_equal(ax.lines[0].get_xdata(), result[axis]["grid"])
            assert_array_equal(ax.lines[0].get_ydata(), result[axis]["means"]["A"])
            self.assertEqual(ax.get_xlim()[1], limit)
        assert_array_equal(result["iteration"]["means"]["A"], [10, 3, 1.5])

    def test_cutoff_waits_for_every_seed_not_the_mean(self):
        runs = {1: {"A": history([1, 0, 0, 0], [0, 1, 3, 4]),
                    "B": history([2, 1], [0, 8])},
                2: {"A": history([1, 0.12, 0.1, 0.01], [0, 2, 4, 6]),
                    "B": history([2, 1], [0, 8])}}
        fig, result = plot_mean_gradient_histories(runs, {"A": "blue", "B": "red"}, target=0.1)
        for ax, axis, endpoint in zip(fig.axes, ("runtime", "iteration"), (4, 2)):
            line = next(line for line in ax.lines if line.get_label() == "A")
            self.assertEqual(line.get_xdata()[-1], endpoint)
            self.assertEqual(line.get_ydata()[-1], 0.05)
            self.assertTrue(result[axis]["endpoints"]["A"]["all_seeds_reached"])
        # Later data remain available, but the curve ends at the all-seed target.
        assert_allclose(result["iteration"]["means"]["A"], [1, 0.06, 0.05, 0.005])

    def test_missed_target_ends_at_own_last_record_without_dropping_seed(self):
        runs = {1: {"A": history([1, 0.01], [0, 1]),
                    "B": history([1, 0.5, 0.4, 0.3], [0, 4, 8, 10])},
                2: {"A": history([1, 0.3, 0.2], [0, 2, 3]),
                    "B": history([1, 0.5, 0.4, 0.3], [0, 4, 8, 10])}}
        fig, result = plot_mean_gradient_histories(runs, {"A": "blue", "B": "red"}, target=0.1)
        for ax, axis, endpoint in zip(fig.axes, ("runtime", "iteration"), (3, 2)):
            line = next(line for line in ax.lines if line.get_label() == "A")
            self.assertEqual(line.get_xdata()[-1], endpoint)
            self.assertAlmostEqual(line.get_ydata()[-1], 0.105)
            self.assertFalse(result[axis]["endpoints"]["A"]["all_seeds_reached"])

    def test_earlier_target_hit_that_rebounds_does_not_trigger_cutoff(self):
        runs = {1: {"A": history([1, 0.05, 0.2, 0.05], [0, 1, 2, 3])},
                2: {"A": history([1, 0.5, 0.05, 0.05], [0, 1, 2, 3])}}
        fig, result = plot_mean_gradient_histories(runs, {"A": "blue"}, target=0.1)
        for ax, axis in zip(fig.axes, ("runtime", "iteration")):
            self.assertEqual(ax.lines[0].get_xdata()[-1], 3)
            self.assertTrue(result[axis]["endpoints"]["A"]["all_seeds_reached"])


if __name__ == "__main__":
    unittest.main()
