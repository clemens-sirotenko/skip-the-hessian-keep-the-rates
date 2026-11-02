"""Representative plots select one shared seed and preserve its real traces."""

import copy
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

from representative_plots import (
    plot_representative_runs,
    plot_target_times,
    select_representative_seed,
    target_summary,
)


def history(gradient, *, times=None, objective=None, penalty=None):
    size = len(gradient)
    return {
        "grad_norm": np.asarray(gradient, dtype=float),
        "time": np.asarray(times if times is not None else np.arange(size), dtype=float),
        "func": np.asarray(objective if objective is not None else np.arange(size, 0, -1), dtype=float),
        "penalty": np.asarray(penalty if penalty is not None else np.zeros(size), dtype=float),
    }


def central_runs():
    fast = [10, 0.01]
    middle = [10, 2, 0.01]
    slow = [10, 2, 2, 0.01]
    return {
        30: {"A": history(slow), "B": history(fast)},
        10: {"A": history(fast), "B": history(slow)},
        20: {"A": history(middle), "B": history(middle)},
    }


class RepresentativeSelectionTests(unittest.TestCase):
    def test_central_seed_is_shared_across_methods(self):
        selection = select_representative_seed(central_runs(), ["A", "B"], [1, 0.1])
        self.assertEqual(selection["seed"], 20)
        self.assertEqual(selection["seeds"], [10, 20, 30])
        self.assertEqual(selection["methods"], ["A", "B"])
        self.assertEqual(selection["targets"], [1, 0.1])
        self.assertTrue(selection["informative"])
        self.assertLess(selection["scores"][20], selection["scores"][10])
        self.assertLess(selection["scores"][20], selection["scores"][30])

    def test_order_and_ties_have_a_deterministic_seed(self):
        runs = {8: {"A": history([10, 0.01])}, 3: {"A": history([10, 2, 0.01])}}
        for ordered in (runs, dict(reversed(list(runs.items())))):
            selection = select_representative_seed(ordered, ["A"], [0.1])
            self.assertEqual(selection["seed"], 3)
            self.assertAlmostEqual(selection["scores"][3], selection["scores"][8])

        tied = {8: {"A": history([10, 2])}, 3: {"A": history([10, 2, 1])}}
        selection = select_representative_seed(tied, ["A"], [0.1])
        self.assertEqual(selection["seed"], 3)
        self.assertFalse(selection["informative"])

    def test_missed_targets_are_retained_with_unequal_lengths(self):
        runs = {
            10: {"A": history([10, 0.01])},
            20: {"A": history([10, 2, 0.1])},
            30: {"A": history([10, 2, 2, 0.2, 0.11])},
        }
        selection = select_representative_seed(runs, ["A"], [1, 0.01])
        self.assertEqual(selection["seed"], 20)
        self.assertEqual(set(selection["scores"]), set(runs))
        self.assertTrue(all(np.isfinite(score) for score in selection["scores"].values()))

    def test_one_seed_has_zero_distance(self):
        selection = select_representative_seed({12: {"A": history([0])}}, ["A"], [1, 0.1])
        self.assertEqual(selection["seed"], 12)
        self.assertEqual(selection["scores"], {12: 0})
        self.assertFalse(selection["informative"])

    def test_runtime_objective_and_penalty_do_not_change_selection(self):
        runs = central_runs()
        altered = copy.deepcopy(runs)
        for seed, methods in altered.items():
            for values in methods.values():
                values["time"] *= seed * 17
                values["func"] -= seed * 100
                values["penalty"] += seed
        before = select_representative_seed(runs, ["A", "B"], [1, 0.1])
        after = select_representative_seed(altered, ["A", "B"], [1, 0.1])
        self.assertEqual(before["seed"], after["seed"])
        self.assertEqual(before["scores"], after["scores"])

    def test_summary_reports_first_hit_and_unreached_runs(self):
        runs = {
            10: {"A": history([10, 0.01, 1], times=[0, 0.5, 0.8], penalty=[1, 0, 0.2])},
            20: {"A": history([10, 2], times=[0, 0.7], penalty=[1, 0.3])},
        }
        statuses = {10: {"A": "iterations_exceeded"}, 20: {"A": "numerical_stagnation"}}
        rows = {row["seed"]: row for row in target_summary(runs, ["A"], 0.1, statuses)}
        reached, missed = rows[10], rows[20]
        self.assertTrue(reached["reached"])
        self.assertEqual(reached["method"], "A")
        self.assertEqual(reached["hit_iteration"], 1)
        self.assertEqual(reached["hit_time"], 0.5)
        self.assertEqual(reached["final_gradient"], 1)
        self.assertEqual(reached["final_penalty"], 0.2)
        self.assertEqual(reached["iterations"], 2)
        self.assertEqual(reached["runtime"], 0.8)
        self.assertEqual(reached["status"], statuses[10]["A"])
        self.assertFalse(missed["reached"])
        self.assertIsNone(missed["hit_iteration"])
        self.assertIsNone(missed["hit_time"])
        self.assertEqual(missed["status"], statuses[20]["A"])


class RepresentativePlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_full_raw_traces_use_one_seed_and_a_common_objective_reference(self):
        runs = {
            20: {
                "A": history([10, 0.001, 0.1], times=[0, 0.2, 0.4],
                             objective=[8, 5, 4], penalty=[1, 0, 0.5]),
                "B": history([10, 0], times=[0, 0.3], objective=[9, 3], penalty=[3, 0]),
            },
            10: {"A": history([10, 1], objective=[0, -100]),
                 "B": history([10, 1], objective=[0, -200])},
        }
        before = copy.deepcopy(runs)
        styles = {"A": {"color": "blue", "linestyle": "-"},
                  "B": {"color": "orange", "linestyle": "--"}}
        select_representative_seed(runs, list(styles), [1, 0.01])
        target_summary(runs, list(styles), 0.1)
        figures = plot_representative_runs(runs, 20, styles)
        self.assertEqual(set(figures), {"func", "grad_norm", "penalty"})
        for measure, figure in figures.items():
            self.assertEqual(len(figure.axes), 2)
            for axis_index, axis in enumerate(figure.axes):
                lines = {line.get_label(): line for line in axis.lines if line.get_label() in styles}
                self.assertEqual(set(lines), set(styles))
                for method, line in lines.items():
                    values = runs[20][method]
                    expected_y = values[measure] - (3 if measure == "func" else 0)
                    expected_x = values["time"] if axis_index == 0 else np.arange(len(expected_y))
                    assert_array_equal(line.get_ydata(), expected_y)
                    assert_array_equal(line.get_xdata(), expected_x)
        for seed, methods in before.items():
            for method, values in methods.items():
                for measure, expected in values.items():
                    assert_array_equal(runs[seed][method][measure], expected)

    def test_zero_gradient_and_penalty_remain_visible(self):
        styles = {"A": {"color": "blue", "linestyle": "-"}}
        for length in (1, 3):
            with self.subTest(length=length), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                runs = {5: {"A": history(np.zeros(length), objective=np.ones(length))}}
                figures = plot_representative_runs(runs, 5, styles)
                for figure in figures.values():
                    figure.canvas.draw()
                    for axis in figure.axes:
                        self.assertEqual(axis.get_yscale(), "symlog")
                        line = next(line for line in axis.lines if line.get_label() == "A")
                        assert_array_equal(line.get_ydata(), np.zeros(length))
                        self.assertTrue(np.isfinite(axis.transData.transform((0, 0))).all())
            self.assertFalse(any("log" in str(w.message).lower() for w in caught))

    def test_display_limits_do_not_crop_recorded_data(self):
        runs = {5: {"A": history([10, 1, 0.01], times=[0, 1, 3])}}
        styles = {"A": {"color": "blue", "linestyle": "-"}}
        figures = plot_representative_runs(runs, 5, styles, max_iter=1, max_time=0.5)
        for figure in figures.values():
            self.assertAlmostEqual(figure.axes[0].get_xlim()[1], 0.5)
            self.assertAlmostEqual(figure.axes[1].get_xlim()[1], 1)
            for axis in figure.axes:
                line = next(line for line in axis.lines if line.get_label() == "A")
                self.assertEqual(len(line.get_ydata()), 3)

    def test_target_times_keep_nonhits_and_mark_the_same_selected_seed(self):
        runs = {
            5: {"A": history([5, 0.1], times=[0, 0.5]),
                "B": history([5, 2, 1], times=[0, 0.3, 0.8])},
            9: {"A": history([5, 4], times=[0, 1.2]),
                "B": history([5, 0.01], times=[0, 1.4])},
        }
        styles = {"A": {"color": "blue", "linestyle": "-"},
                  "B": {"color": "orange", "linestyle": "--"}}
        rows = target_summary(runs, list(styles), 0.1)
        before = copy.deepcopy(rows)
        figure = plot_target_times(rows, styles, 0.1, selected_seed=5)
        axis = figure.axes[0]
        observations = [artist for artist in axis.collections if artist.get_facecolors().size]
        rings = [artist for artist in axis.collections if not artist.get_facecolors().size]
        observed_times = sorted(float(artist.get_offsets()[0, 0]) for artist in observations)
        ring_times = sorted(float(artist.get_offsets()[0, 0]) for artist in rings)
        assert_array_equal(observed_times, [0.5, 0.8, 1.2, 1.4])
        assert_array_equal(ring_times, [0.5, 0.8])
        self.assertTrue(all("1/2 reached" in label.get_text() for label in axis.get_yticklabels()))
        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
