"""All-seed gradient means with terminal values retained after stopping."""

import numpy as np
import matplotlib.pyplot as plt


def mean_gradient_histories(seed_histories, methods):
    """Average gradient norms at common iteration and runtime budgets.

    Input: seed -> method -> history, with equally sized 1-D ``grad_norm``
    and cumulative ``time`` arrays. Times must be finite and nondecreasing.
    Every seed has equal weight, including unsuccessful runs. Each terminal
    norm is retained after its recorded end; earlier target hits do not stop
    the trace. No threshold replacement, interpolation or log averaging.

    Runtime uses the last observation at each timestamp (last duplicate wins).
    Its grid starts once every run has an observation and ends at the latest
    recorded time across methods. Iterations also extend to the longest run.
    Return {axis: {grid: 1-D array, means: {method: 1-D array}}}.
    """
    runs = {name: [histories[name] for histories in seed_histories.values()]
            for name in methods}
    all_runs = [h for histories in runs.values() for h in histories]
    if any(not np.all(np.isfinite(h["grad_norm"])) for h in all_runs):
        raise ValueError("Mean requires finite gradient histories; inspect failed runs.")
    iteration_grid = np.arange(max(len(h["grad_norm"]) for h in all_runs))
    first_time = max(h["time"][0] for h in all_runs)
    time_grid = np.unique(np.concatenate([h["time"] for h in all_runs]))
    time_grid = time_grid[time_grid >= first_time]

    summaries = {}
    for axis, grid in [("runtime", time_grid), ("iteration", iteration_grid)]:
        means = {}
        for name, histories in runs.items():
            values = []
            for h in histories:
                gradient = np.asarray(h["grad_norm"], dtype=float)
                index = (np.searchsorted(h["time"], grid, side="right") - 1
                         if axis == "runtime" else grid)
                values.append(gradient[np.minimum(index, len(gradient) - 1)])
            means[name] = np.mean(values, axis=0)
        summaries[axis] = dict(grid=grid, means=means)
    return summaries


def plot_mean_gradient_histories(seed_histories, method_colors, *,
                                 max_iter=None, max_time=None, zero_scale=1e-14,
                                 method_linestyles=None, target=None):
    """Plot all-seed arithmetic means, returning (figure, numeric summaries).

    Optional limits zoom the view without changing the computed means.
    The y axis is logarithmic above zero_scale and linear near zero so exact
    zeros remain visible. This is a mean of norms, not a norm of mean gradients.
    method_linestyles optionally maps method names to solid/dashed line styles.
    Each curve ends at the first shared-grid point where every seed's current
    norm is <= target, or at that method's last recording if the target is
    missed/None. The full means are retained in summaries; endpoints record
    the displayed cutoff and whether all seeds met the target there.
    """
    summaries = mean_gradient_histories(seed_histories, method_colors)
    method_linestyles = method_linestyles or {}
    legend_rows = (len(method_colors) + 3) // 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5 + 0.28 * (legend_rows - 1)), sharey=True)
    for ax, axis, limit, xlabel in zip(
        axes, ("runtime", "iteration"), (max_time, max_iter), ("Runtime (s)", "Iteration")
    ):
        summary = summaries[axis]
        grid = summary["grid"]
        summary["endpoints"] = {}
        for name, color in method_colors.items():
            histories = [methods[name] for methods in seed_histories.values()]
            last_record = max(h["time"][-1] if axis == "runtime" else len(h["grad_norm"]) - 1
                              for h in histories)
            end = max(0, int(np.searchsorted(grid, last_record, side="right")) - 1)
            reached = None
            if target is not None:
                below = []
                for h in histories:
                    index = (np.searchsorted(h["time"], grid[:end + 1], side="right") - 1
                             if axis == "runtime" else grid[:end + 1])
                    values = np.asarray(h["grad_norm"])[np.minimum(index, len(h["grad_norm"]) - 1)]
                    below.append(values <= target)
                hits = np.flatnonzero(np.all(below, axis=0))
                reached = bool(hits.size)
                if reached:
                    end = int(hits[0])
            x = grid[:end + 1].copy()
            # A run ending before other methods initialize still has a real endpoint.
            if last_record < x[0]:
                x[0] = last_record
            summary["endpoints"][name] = dict(index=end, x=float(x[-1]), all_seeds_reached=reached)
            ax.plot(x, summary["means"][name][:end + 1], color=color, label=name,
                    linestyle=method_linestyles.get(name, "-"),
                    linewidth=2, drawstyle="steps-post" if axis == "runtime" else "default",
                    marker="o" if len(x) == 1 else None)
        ax.set_yscale("symlog", linthresh=zero_scale)
        last_shown = max(endpoint["x"] for endpoint in summary["endpoints"].values())
        ax.set_xlim(0, limit if limit is not None else max(last_shown, 1e-12))
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(True, which="major", alpha=0.25)
    plotted = [values[:summary["endpoints"][name]["index"] + 1]
               for summary in summaries.values() for name, values in summary["means"].items()]
    if any(np.any(values == 0) for values in plotted):
        axes[0].set_ylim(bottom=-0.1 * zero_scale)
    if not any(np.any(values > 0) for values in plotted):
        axes[0].set_ylim(top=zero_scale)
    axes[0].set_ylabel("Mean gradient norm", fontsize=12)
    fig.suptitle(f"Matrix factorization: arithmetic mean over {len(seed_histories)} seeds", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=min(4, len(method_colors)), fontsize=9, frameon=False)
    note = "Final values retained after all exits, including failures; every seed remains in the mean."
    if target is not None:
        note += f"\nCurves end when all seeds <= {target:g}, otherwise at that method's last recording."
    else:
        note += "\nEach curve ends at that method's last recording."
    if any(style == "--" for style in method_linestyles.values()):
        note += "\nDashed: Gauss-Newton approximations."
    note += f"\nLinear below {zero_scale:g}, log above. Parallel timings include shared CPU load."
    if max_iter is not None:
        note += f" Iteration view: 0-{max_iter:g}."
    if max_time is not None:
        note += f" Runtime view: 0-{max_time:g} s."
    fig.text(0.5, 0.015, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.15, 1, 0.88 - 0.045 * (legend_rows - 1)))
    fig.subplots_adjust(left=0.085)
    return fig, summaries
