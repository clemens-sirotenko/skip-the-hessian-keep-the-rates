"""Supplementary actual-run plots for the parallel MF experiment.

A medoid is an observed item minimizing its total dissimilarity to the others
(Kaufman & Rousseeuw, 1987). Here dissimilarity compares ranks of first-hit
iterations at fixed gradient targets, equally weighted across methods/targets.
This selection rule defines supplementary diagnostic figures.
All plots reuse recorded histories; selection does not use runtime or winners.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata
from utils_new import plot_figures


def _first_hit(history, target):
    gradient = np.asarray(history["grad_norm"])
    hits = np.flatnonzero(np.isfinite(gradient) & (gradient <= target))
    return int(hits[0]) if hits.size else None


def select_representative_seed(seed_histories, methods, targets):
    """Select one shared seed as the medoid of convergence-milestone ranks.

    Input: seed -> method -> history, with 1-D grad_norm arrays. For each
    method/target, rank first-hit iterations across seeds; unreached targets
    tie last. Average absolute rank differences over methods/targets, then
    average distances to the other seeds. Smallest seed breaks numerical ties.
    Runtime, objective, penalty and relative method rankings are not used.

    Returns the seed, scores and the complete selection record (JSON-ready).
    No common-length truncation, interpolation or extrapolation is used.
    A central milestone profile need not have a central penalty trajectory.
    """
    seeds, methods, targets = sorted(seed_histories), list(methods), list(targets)
    if not seeds or not methods or not targets:
        raise ValueError("Selection needs seeds, methods and gradient targets.")
    hits = np.full((len(seeds), len(methods), len(targets)), np.inf)
    for i, seed in enumerate(seeds):
        for j, method in enumerate(methods):
            for k, target in enumerate(targets):
                hit = _first_hit(seed_histories[seed][method], target)
                if hit is not None:
                    hits[i, j, k] = hit

    ranks = (rankdata(hits, axis=0, method="average") - 1) / max(1, len(seeds) - 1)
    distances = np.abs(ranks[:, None] - ranks[None, :]).mean(axis=(2, 3))
    scores = distances.sum(axis=1) / max(1, len(seeds) - 1)
    tied = np.flatnonzero(np.isclose(scores, scores.min(), rtol=1e-12, atol=1e-15))
    return {
        "seed": int(seeds[tied[0]]),
        "seeds": [int(seed) for seed in seeds],
        "methods": methods,
        "targets": [float(target) for target in targets],
        "scores": {int(seed): float(score) for seed, score in zip(seeds, scores)},
        "tied_seeds": [int(seeds[i]) for i in tied],
        "informative": bool(np.any(np.ptp(ranks, axis=0) > 0)),
        "first_hit_iterations": {
            int(seed): {method: [int(v) if np.isfinite(v) else None for v in hits[i, j]]
                        for j, method in enumerate(methods)}
            for i, seed in enumerate(seeds)
        },
        "normalized_ranks": ranks.tolist(),
        "distance": "mean absolute normalized first-hit rank difference",
        "tie_rule": "smallest seed (ties within floating-point roundoff)",
    }


def target_summary(seed_histories, methods, target, statuses=None):
    """One row per seed/method, retaining unreached targets and exit statuses.

    hit_time is the recorded time of the first gradient <= target, or None.
    Reaching a reporting target is distinct from the eventual solver exit.
    """
    rows = []
    for seed in sorted(seed_histories):
        for method in methods:
            h = seed_histories[seed][method]
            hit = _first_hit(h, target)
            rows.append(dict(
                seed=int(seed), method=method, reached=hit is not None,
                hit_iteration=hit, hit_time=float(h["time"][hit]) if hit is not None else None,
                final_gradient=float(h["grad_norm"][-1]), final_penalty=float(h["penalty"][-1]),
                iterations=len(h["grad_norm"]) - 1, runtime=float(h["time"][-1]),
                status=statuses[seed][method] if statuses is not None else "",
            ))
    return rows


def plot_representative_runs(seed_histories, seed, method_styles, *, zero_scale=1e-14,
                             penalty_zero_scale=None, max_iter=None, max_time=None,
                             selection_label="Representative"):
    """Use plot_figures for one seed's objective, gradient and penalty histories.

    method_styles maps names to color/linestyle dictionaries. Each figure has
    runtime and iteration panels, with no averaging or threshold truncation.
    The objective reference is the best observed value among these methods for
    this seed. A linear y-region below zero_scale preserves exact zeros; above
    it the scale is logarithmic. Optional x limits only change the view.
    penalty_zero_scale optionally sets a separate linear region for penalties.
    """
    histories = seed_histories[seed]
    reference = min(np.min(histories[name]["func"]) for name in method_styles)
    measures = [("func", "Objective - reference", reference),
                ("grad_norm", "Gradient norm", 0.0), ("penalty", "Weighted penalty", 0.0)]
    figures = {}
    for measure, ylabel, offset in measures:
        scale = penalty_zero_scale if measure == "penalty" and penalty_zero_scale is not None else zero_scale
        note = f"Circles: recorded endpoints. Linear below {scale:g}, log above."
        if max_iter is not None:
            note += f" Iteration view: 0-{max_iter:g}."
        if max_time is not None:
            note += f" Runtime view: 0-{max_time:g} s."
        if measure == "func":
            note += "\nReference: best objective observed across these methods for this seed."
        figures[measure] = plot_figures(
            [histories[name] for name in method_styles], list(method_styles),
            n=None, m=None, mu_str=None,
            colors=[style["color"] for style in method_styles.values()],
            linewidths=[2] * len(method_styles),
            linestyles=[style.get("linestyle", "-") for style in method_styles.values()],
            which_measure_1=measure, which_measure_2=measure, f_star=offset,
            threshold=None, max_time=max_time, max_iter=max_iter, zero_scale=scale,
            sharey=True, mark_endpoints=True, verbose=False, ylabel=ylabel,
            suptitle=f"Matrix Factorization: {selection_label.lower()} seed {seed}",
            note=note,
        )
    return figures


def plot_target_times(rows, method_styles, target, selected_seed):
    """Show every seed's first-hit time; mark non-hits at their recorded end.

    Crosses are not time-to-target estimates. Counts include every seed, and
    rings identify the seed used for the actual-run plots.
    """
    methods = list(method_styles)
    seeds = sorted({row["seed"] for row in rows})
    offsets = dict(zip(seeds, np.linspace(-0.18, 0.18, len(seeds)) if len(seeds) > 1 else [0]))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.46 * len(methods) + 1.7)))
    times = []
    for row in rows:
        x = row["hit_time"] if row["reached"] else row["runtime"]
        y = methods.index(row["method"]) + offsets[row["seed"]]
        color = method_styles[row["method"]]["color"]
        ax.scatter(x, y, color=color, marker="o" if row["reached"] else "x", s=34)
        if row["seed"] == selected_seed:
            ax.scatter(x, y, facecolors="none", edgecolors="black", s=100, linewidths=1)
        times.append(x)
    labels = []
    for method in methods:
        reached = sum(row["reached"] for row in rows if row["method"] == method)
        labels.append(f"{method}   ({reached}/{len(seeds)} reached)")
    ax.set_yticks(range(len(methods)), labels)
    ax.invert_yaxis()
    if all(t > 0 for t in times):
        ax.set_xscale("log")
    ax.set_xlabel("Recorded runtime (s)")
    ax.set_title(f"All {len(seeds)} seeds: first gradient norm <= {target:g}")
    ax.grid(True, axis="x", which="major", alpha=0.25)
    handles = [Line2D([], [], color="black", marker="o", linestyle="none", label="Target reached"),
               Line2D([], [], color="black", marker="x", linestyle="none", label="Not reached"),
               Line2D([], [], color="black", marker="o", markerfacecolor="none",
                      linestyle="none", label=f"Selected seed {selected_seed}")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
    fig.text(0.5, 0.015, "Crosses show last recorded runtime, not time to target. Parallel timings include shared CPU load.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig
