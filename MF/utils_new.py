"""Plots and all-seed arithmetic means for the MF notebooks.

Histories contain func, grad_norm, penalty and time. Plot thresholds affect
display only; penalty panels retain zeros and later rebounds.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import run_methods


def plot_figures(
    histories, labels, n, m, mu_str, colors, linewidths, linestyles,
    n_iters=None, threshold=1e-8, filename=None, f_star=None, suptitle=None,
    max_iter=1000, which_measure_1="func", which_measure_2="func_at_lssolve",
    ylabel="Func. residual", xlabel="Runtime (s)", *, max_time=None,
    zero_scale=None, sharey=False, runtime_steps=False, mark_endpoints=False,
    note=None, verbose=True, legend_layout="outside", marker_cycle=None, markevery=None,
):
    """Draw runtime/iteration panels for recorded or prepared histories.

    Each history supplies ``time`` and the two named measures; style lists
    follow ``labels``. The runtime measure aligns with time. The iteration
    measure may have a different length and uses indices starting at zero.
    This lets arithmetic means use their own runtime and iteration grids.

    Subtract f_star (default: best supplied objective) from both measures.
    threshold trims objective/gradient curves at their first hit; None disables
    trimming. n_iters optionally caps entries per method. Penalties keep zeros
    and later rebounds, using a linear region below threshold (fallback 1e-10).
    zero_scale explicitly sets that linear region for both panels, including
    gradient/objective plots. Above it the scale is logarithmic.

    max_time/max_iter only zoom. runtime_steps uses the latest-observation
    convention; mark_endpoints adds open circles. sharey links y limits.
    note adds a caption; verbose prints terminal plotted values. n, m, mu_str
    and filename are unused legacy arguments. Return a figure without saving it.
    legend_layout="paper" follows utils.plot_for_paper_2: a boxed, translucent
    three-column legend overlapping the panels, dashed grid and 10x4 figure.
    "outside" preserves the original legend between the panels. marker_cycle
    optionally supplies one marker per method; markevery controls marker spacing
    without thinning the plotted data (a float uses display-distance spacing).
    """
    if legend_layout not in ("outside", "paper"):
        raise ValueError("legend_layout must be 'outside' or 'paper'.")
    if f_star is None:
        f_star = min(np.min(h["func"]) for h in histories)
    paper_style = legend_layout == "paper"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4) if paper_style else (12, 4), sharey=sharey)
    measures = (which_measure_1, which_measure_2)
    penalty_scale = threshold if threshold is not None and threshold > 0 else 1e-10
    scales = [zero_scale if zero_scale is not None else
              penalty_scale if measure == "penalty" else None for measure in measures]
    for ax, scale in zip(axes, scales):
        if scale is None:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=scale)

    for i, history in enumerate(histories):
        values = [np.asarray(history[measure]) - f_star for measure in measures]
        if verbose:
            print(f"Last {ylabel} of {labels[i]}: {values[1][-1]}")
        for panel, (ax, measure, y) in enumerate(zip(axes, measures, values)):
            end = len(y) if n_iters is None else min(len(y), n_iters[i])
            if measure != "penalty" and threshold is not None:
                hits = np.flatnonzero(y <= threshold)
                if hits.size:
                    end = min(end, int(hits[0]) + 1)
            x = np.asarray(history["time"])[:end] if panel == 0 else np.arange(end)
            y = y[:end]
            marker = (marker_cycle[i] if marker_cycle is not None else
                      "o" if end == 1 and (measure == "penalty" or runtime_steps) else None)
            ax.plot(x, y, label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i],
                    drawstyle="steps-post" if panel == 0 and runtime_steps else "default",
                    marker=marker, markevery=markevery, markersize=5 if paper_style else 6)
            if mark_endpoints and end:
                ax.plot(x[-1], y[-1], "o", color=colors[i], markerfacecolor="white",
                        markersize=4, label="_nolegend_")

    for ax, measure, scale, limit in zip(axes, measures, scales, (max_time, max_iter)):
        if scale is not None:
            lines = [line for panel in (axes if sharey else [ax]) for line in panel.lines]
            if (measure == "penalty" and zero_scale is None
                    or any(np.any(line.get_ydata() == 0) for line in lines)):
                ax.set_ylim(bottom=-0.1 * scale)
            if not any(np.any(line.get_ydata() > 0) for line in lines):
                ax.set_ylim(top=scale)
        if limit is not None:
            ax.set_xlim(0, limit)
        if paper_style:
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            ax.grid(True)
    label_fontsize = 10 if paper_style else 14
    axes[0].set_xlabel(xlabel, fontsize=label_fontsize)
    axes[0].set_ylabel(ylabel, fontsize=label_fontsize)
    axes[1].set_xlabel("Iteration", fontsize=label_fontsize)
    if paper_style:
        axes[1].set_ylabel(ylabel, fontsize=label_fontsize)
    axes[1].tick_params(labelleft=True)
    if legend_layout == "outside":
        axes[0].legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    bottom = 0
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8)
        bottom = 0.025 + 0.045 * len(note.splitlines())
    if paper_style:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center",
            ncol=3, frameon=True, framealpha=0.8, fancybox=True,
            facecolor="white", edgecolor="0.8", fontsize=10,
        )
    fig.tight_layout(rect=(0, bottom, 1, 0.95 if paper_style else 1))
    return fig


def plot_penalty_iterations(
    histories, labels, colors, linewidths, linestyles, *, max_iter=None,
    ylim=None, zero_scale=1e-10, figsize=(6, 4), legend_loc="upper right",
    marker_cycle=None, markevery=None,
):
    """Plot recorded weighted penalties against update count on one panel.

    Each history supplies a 1-D ``penalty`` array including the initial point.
    Style lists follow labels. All recorded entries are retained, including
    zeros and later rebounds; no threshold cutoff or terminal extension is used.
    The axis is logarithmic above positive zero_scale and linear near zero.
    max_iter and ylim only zoom; None uses automatic limits. The boxed legend
    and grid follow the NMF Figure 3 / Figure 8 single-panel paper plots.
    Return the figure without saving or displaying it.
    """
    if not histories:
        raise ValueError("Select at least one penalty history.")
    if not np.isscalar(zero_scale) or not np.isfinite(zero_scale) or zero_scale <= 0:
        raise ValueError("zero_scale must be a positive finite number.")
    if max_iter is not None and (not np.isfinite(max_iter) or max_iter <= 0):
        raise ValueError("max_iter must be a positive number or None.")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_yscale("symlog", linthresh=zero_scale)
    has_zero, has_positive = False, False
    for i, history in enumerate(histories):
        penalty = np.asarray(history["penalty"])
        has_zero |= bool(np.any(penalty == 0))
        has_positive |= bool(np.any(penalty > 0))
        ax.plot(
            np.arange(len(penalty)), penalty, label=labels[i],
            color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i],
            marker=marker_cycle[i] if marker_cycle is not None else None,
            markevery=markevery, markersize=4,
        )
    ax.set_xlim(left=0, right=max_iter)
    if has_zero:
        ax.set_ylim(bottom=-0.1 * zero_scale)
    if not has_positive:
        ax.set_ylim(top=zero_scale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Penalty", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc=legend_loc, fontsize=8, frameon=True, framealpha=0.8,
              fancybox=True, facecolor="white", edgecolor="0.8")
    fig.tight_layout()
    return fig


def mean_histories(seed_histories, methods, *, measure="grad_norm"):
    """Average a recorded measure at common iteration and runtime budgets.

    Input: seed -> method -> history, with equally sized 1-D measure
    and cumulative ``time`` arrays. Times must be finite and nondecreasing.
    Every seed has equal weight, including unsuccessful runs. Each terminal
    value is retained after its recorded end; earlier target hits do not stop
    the trace. No threshold replacement, interpolation or log averaging.

    Runtime uses the last observation at each timestamp (last duplicate wins).
    Its grid starts once every run has an observation and ends at the latest
    recorded time across methods. Iterations also extend to the longest run.
    Return {axis: {grid: 1-D array, means: {method: 1-D array}}}.
    """
    runs = {name: [histories[name] for histories in seed_histories.values()]
            for name in methods}
    all_runs = [h for histories in runs.values() for h in histories]
    if any(not np.all(np.isfinite(h[measure])) for h in all_runs):
        raise ValueError(f"Mean requires finite {measure} histories; inspect failed runs.")
    iteration_grid = np.arange(max(len(h[measure]) for h in all_runs))
    first_time = max(h["time"][0] for h in all_runs)
    time_grid = np.unique(np.concatenate([h["time"] for h in all_runs]))
    time_grid = time_grid[time_grid >= first_time]

    summaries = {}
    for axis, grid in [("runtime", time_grid), ("iteration", iteration_grid)]:
        means = {}
        for name, histories in runs.items():
            values = []
            for h in histories:
                recorded = np.asarray(h[measure], dtype=float)
                index = (np.searchsorted(h["time"], grid, side="right") - 1
                         if axis == "runtime" else grid)
                values.append(recorded[np.minimum(index, len(recorded) - 1)])
            means[name] = np.mean(values, axis=0)
        summaries[axis] = dict(grid=grid, means=means)
    return summaries


def prepare_mean_histories(seed_histories, methods, *, measure="grad_norm",
                           target=None, flat_tail_rtol=None):
    """Return prepared plot histories and full means with display endpoints.

    Each method's plot history has time, runtime values and iteration values;
    the two value arrays may have different lengths. Full grids and means stay
    in summaries. Every seed has equal weight, with terminal values retained.

    End display at the first all-seed target hit, otherwise the last recording.
    If the target is missed/None, flat_tail_rtol optionally omits a terminal
    plateau once every seed stays within that relative tolerance of its final
    value (zero absolute tolerance). These cutoffs apply to the selected measure;
    leave both None to retain every recording, including zero penalty rebounds.
    This is not a solver convergence test.
    """
    summaries = mean_histories(seed_histories, methods, measure=measure)
    plot_histories = {name: {} for name in methods}
    for axis in ("runtime", "iteration"):
        summary = summaries[axis]
        grid = summary["grid"]
        summary["endpoints"] = {}
        for name in methods:
            histories = [seed_runs[name] for seed_runs in seed_histories.values()]
            last_record = max(h["time"][-1] if axis == "runtime" else len(h[measure]) - 1
                              for h in histories)
            end = max(0, int(np.searchsorted(grid, last_record, side="right")) - 1)
            reached = None
            if target is not None:
                below = []
                for h in histories:
                    index = (np.searchsorted(h["time"], grid[:end + 1], side="right") - 1
                             if axis == "runtime" else grid[:end + 1])
                    values = np.asarray(h[measure])[np.minimum(index, len(h[measure]) - 1)]
                    below.append(values <= target)
                hits = np.flatnonzero(np.all(below, axis=0))
                reached = bool(hits.size)
                if reached:
                    end = int(hits[0])
            flat_tail_trimmed = False
            cutoff = last_record
            if flat_tail_rtol is not None and not reached:
                plateau_starts = []
                for h in histories:
                    recorded = np.asarray(h[measure])
                    changed = np.flatnonzero(~np.isclose(
                        recorded, recorded[-1], rtol=flat_tail_rtol, atol=0))
                    start = int(changed[-1]) + 1 if changed.size else 0
                    plateau_starts.append(h["time"][start] if axis == "runtime" else start)
                cutoff = max(plateau_starts)
                plateau_end = max(0, int(np.searchsorted(grid, cutoff, side="right")) - 1)
                flat_tail_trimmed = cutoff < last_record
                end = min(end, plateau_end)
            x = grid[:end + 1].copy()
            # A run ending before other methods initialize still has a real endpoint.
            if cutoff < x[0]:
                x[0] = cutoff
            summary["endpoints"][name] = dict(index=end, x=float(x[-1]),
                                             all_seeds_reached=reached,
                                             flat_tail_trimmed=bool(flat_tail_trimmed))
            plot_histories[name][axis] = summary["means"][name][:end + 1]
            if axis == "runtime":
                plot_histories[name]["time"] = x
    return plot_histories, summaries


# Names used by the older gradient-only notebooks.
mean_gradient_histories = mean_histories
prepare_mean_gradient_histories = prepare_mean_histories


def plot_mean_gradient_histories(seed_histories, method_colors, *,
                                 max_iter=None, max_time=None, zero_scale=1e-14,
                                 method_linestyles=None, target=None, flat_tail_rtol=None):
    """Plot all-seed arithmetic means, returning (figure, numeric summaries).

    Prepare each panel's data and cutoffs, then draw both with plot_figures.
    Optional limits zoom the view without changing the computed means.
    The y axis is logarithmic above zero_scale and linear near zero so exact
    zeros remain visible. This is a mean of norms, not a norm of mean gradients.
    method_linestyles optionally maps method names to solid/dashed line styles.
    Each curve ends at the first shared-grid point where every seed's current
    norm is <= target, or at that method's last recording if the target is
    missed/None. The full means are retained in summaries; endpoints record
    the displayed cutoff and whether all seeds met the target there.
    If flat_tail_rtol is set and the target is missed/None, omit the terminal
    plateau once every seed stays within this relative tolerance of its own
    final norm. No absolute tolerance is used. This display cutoff does not
    imply convergence and never removes an earlier, temporary plateau.
    """
    plot_histories, summaries = prepare_mean_gradient_histories(
        seed_histories, method_colors, target=target, flat_tail_rtol=flat_tail_rtol)
    method_linestyles = method_linestyles or {}
    limits = {}
    for axis, limit in zip(("runtime", "iteration"), (max_time, max_iter)):
        last_shown = max(endpoint["x"] for endpoint in summaries[axis]["endpoints"].values())
        limits[axis] = limit if limit is not None else max(last_shown, 1e-12)
    note = f"All {len(seed_histories)} seeds retained after exits, including failures. "
    note += (f"Cutoff: all seeds <= {target:g}, otherwise last recording."
             if target is not None else "Cutoff: each method's last recording.")
    if flat_tail_rtol is not None:
        note += (f"\nTerminal plateaus omitted ({100 * flat_tail_rtol:g}% relative tolerance "
                 "per seed; display only).")
    if any(style == "--" for style in method_linestyles.values()):
        note += " Dashed: Gauss-Newton."
    note += f"\nLinear below {zero_scale:g}, log above. Parallel runtimes include shared CPU load."
    fig = plot_figures(
        list(plot_histories.values()), list(method_colors), n=None, m=None, mu_str=None,
        colors=list(method_colors.values()), linewidths=[2] * len(method_colors),
        linestyles=[method_linestyles.get(name, "-") for name in method_colors],
        which_measure_1="runtime", which_measure_2="iteration", f_star=0,
        threshold=None, max_time=limits["runtime"], max_iter=limits["iteration"],
        zero_scale=zero_scale, sharey=True, runtime_steps=True, verbose=False,
        ylabel="Mean gradient norm", suptitle="Matrix Factorization: arithmetic mean",
        note=note,
    )
    return fig, summaries


def plot_combined_gradient_histories(seed_histories, method_colors, *,
                                     axis="iteration", max_x=None, ylim=None):
    """Overlay per-seed norms, medians and quartiles for selected MF methods.

    seed_histories maps each seed to method histories. ``grad_norm`` and
    ``time`` are equal-length 1-D arrays, starting with the initial point;
    times are cumulative seconds, finite, nonnegative and nondecreasing.
    method_colors maps selected method names to Matplotlib colors.
    axis is 'iteration' or 'runtime'; max_x is an optional inclusive display
    bound in those units. All-seed summaries use common recorded support.
    Runtime uses the latest observation at each common timestamp, including
    the last observation at duplicate times, with no endpoint extrapolation.
    Return a new figure; leave all input histories and earlier plots intact.
    """
    if axis not in ("iteration", "runtime"):
        raise ValueError("axis must be 'iteration' or 'runtime'.")
    if not seed_histories or not method_colors:
        raise ValueError("Select at least one method and collect seed histories first.")
    if max_x is not None and (not np.isfinite(max_x) or max_x < 0):
        raise ValueError("max_x must be finite and nonnegative, or None.")

    runs_by_method = {}
    for method in method_colors:
        runs = []
        for seed, histories_for_seed in seed_histories.items():
            history = histories_for_seed[method]
            g = np.asarray(history["grad_norm"], dtype=float)
            t = np.asarray(history["time"], dtype=float)
            if (g.ndim != 1 or g.size == 0 or t.shape != g.shape
                    or not np.all(np.isfinite(g)) or np.any(g < 0)
                    or not np.all(np.isfinite(t)) or np.any(t < 0)
                    or np.any(np.diff(t) < 0)):
                raise ValueError(f"Invalid gradient/time history for {method}, seed {seed}.")
            runs.append((np.arange(g.size) if axis == "iteration" else t, g))
        runs_by_method[method] = runs

    def positive(values):
        return np.where(values > 0, values, np.nan)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    zero_present = False
    no_overlap = []
    for method, color in method_colors.items():
        runs = runs_by_method[method]
        for x, g in runs:
            ax.plot(x, positive(g), color=color, alpha=0.18, linewidth=0.8,
                    drawstyle="steps-post" if axis == "runtime" else "default",
                    marker="." if len(g) == 1 else None, label="_nolegend_")
            zero_present |= bool(np.any(g == 0))

        start = max(x[0] for x, _ in runs)
        end = min(x[-1] for x, _ in runs)
        if max_x is not None:
            end = min(end, max_x)
        if start > end:
            no_overlap.append(method)
            ax.plot([], [], color=color, linewidth=2.3, label=f"{method} (traces only)")
            continue

        if axis == "iteration":
            grid = np.arange(int(end) + 1)
            values = np.stack([g[:len(grid)] for _, g in runs])
        else:
            # Every change inside common support appears in the step summary.
            grid = np.unique(np.concatenate(
                [x[(x >= start) & (x <= end)] for x, _ in runs]
                + [np.array([start, end])]
            ))
            values = np.stack([
                g[np.searchsorted(x, grid, side="right") - 1] for x, g in runs
            ])
        lower, median, upper = np.percentile(values, [25, 50, 75], axis=0)
        ax.fill_between(grid, positive(lower), positive(upper), color=color,
                        alpha=0.16, step="post" if axis == "runtime" else None)
        ax.plot(grid, positive(median), color=color, linewidth=2.4, label=method,
                drawstyle="steps-post" if axis == "runtime" else "default")
        if median[-1] > 0:
            ax.plot(grid[-1], median[-1], marker="o", markersize=5,
                    markerfacecolor="white", markeredgecolor=color,
                    linestyle="none", label="_nolegend_")

    longest = max(x[-1] for runs in runs_by_method.values() for x, _ in runs)
    right = longest if max_x is None else max_x
    ax.set_xlim(0, right if right > 0 else 1)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("Iteration" if axis == "iteration" else "Runtime (s)")
    ax.set_ylabel("Gradient norm")
    title = f"Matrix factorization: {len(seed_histories)} seeds, gradient vs {axis}"
    runtime_zoomed = axis == "runtime" and max_x is not None and max_x < longest
    if runtime_zoomed:
        title += f" (first {max_x:g} s)"
    fig.suptitle(title)
    ax.set_title("Thick: median | Shaded: 25th-75th percentile | Faint: individual seeds",
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, which="major", alpha=0.2)
    note = "Open circles: end of each displayed all-seed summary."
    if runtime_zoomed:
        note += f"\nFull recorded runtime: {longest:,.1f} s; later observations are outside this view."
    if no_overlap:
        note += " No common recorded interval: " + ", ".join(no_overlap) + "."
    if zero_present:
        note += " Exact zeros omitted on the log axis."
    fig.supxlabel(note, fontsize=9)
    return fig
