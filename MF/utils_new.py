"""Experiment runners and plotting helpers for matrix factorization.

The multiple-seed notebook uses ``run_methods`` and ``plot_figures``. The other
plotting functions retain the original paper layouts and their legacy options.
Unless a function documents a label-to-history mapping, histories and styles
are parallel lists in method order. A history records objective ``func``, full
``grad_norm``, weighted nonnegativity ``penalty``, and elapsed ``time`` in seconds.
Plotting thresholds only affect displayed curves; they never stop optimization.
Penalty panels in ``plot_figures`` use a linear scale near zero and log above it.
"""

import numpy as np
import matplotlib.pyplot as plt

from oracles_mat_fac import *


def run_methods(methods_to_run, oracle, x_0):
    """Run each solver in the supplied method dictionary and collect histories.

    Each entry is ``label: (solver, args, kwargs)``. The solver must return
    ``(final_point, status, history)``, with a nonempty ``history["time"]`` in
    seconds. Methods run in dictionary order; their exit status and final
    elapsed time are printed. Only ``{label: history}`` is returned, so inspect
    the printed statuses before interpreting the curves as convergence.

    The oracle and starting point are passed through each entry's ``args``.
    The separate ``oracle`` and ``x_0`` parameters are unused compatibility
    arguments retained from the original notebook.
    """
    history_dict = {}

    for method_name, (func, args, kwargs) in methods_to_run.items():
        print(f"\nRunning {method_name}...")

        # Keep the recorded history; discard the solver's final factor vector.
        _, status, history = func(*args, **kwargs)
        print(f"Status: {status}")

        # Preserve the method label used by the notebook's plotting code.
        history_dict[method_name] = history

        print("Time: %.3f s" % history_dict[method_name]["time"][-1])

    return history_dict


def plot_figures(
    histories,
    labels,
    n,
    m,
    mu_str,
    colors,
    linewidths,
    linestyles,
    n_iters=None,
    threshold=1e-8,
    filename=None,
    f_star=None,
    suptitle=None,
    max_iter=1000,
    which_measure_1="func",
    which_measure_2="func_at_lssolve",
    ylabel="Func. residual",
    xlabel="Runtime (s)",
):
    """Plot two recorded measures against runtime and iteration; return the figure.

    ``histories`` is a list of history dictionaries. Each needs ``time`` and
    the keys named by ``which_measure_1`` (left panel) and ``which_measure_2``
    (right panel). Labels and style lists must follow the same method order.
    The left measure must align with the recorded times. Iteration indices
    start at zero, which represents the initial point in the MF histories.

    ``f_star`` is subtracted from both measures. If omitted, use the smallest
    recorded objective across the supplied histories, not a certified global
    optimum. Pass zero for gradient norms or penalties. ``n_iters`` optionally
    caps the number of recorded entries per method, including the initial one.
    ``threshold`` trims objective/gradient curves after their first threshold hit.
    Penalty curves retain every entry: zero penalty can be followed by positive
    values. Their y-axis is linear up to ``threshold`` and logarithmic above it,
    so exact zeros stay visible. A nonpositive/None threshold uses 1e-10 for this
    linear region. None disables trimming of other measures.

    ``max_iter`` limits only the iteration axis, so longer runtime curves
    remain visible. ``n``, ``m``, ``mu_str``, and ``filename`` are unused legacy
    arguments. Saving, showing, and closing the returned figure is up to the
    caller; this helper does not create a file.
    """
    if f_star is None:
        f_best = min(histories[0]["func"])
        for i in range(1, len(histories)):
            f_best = min(f_best, min(histories[i]["func"]))
    else:
        f_best = f_star

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4)
    )  # Leave space between the panels for the legend.
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    penalty_scale = threshold if threshold is not None and threshold > 0 else 1e-10
    for ax, measure in ((ax1, which_measure_1), (ax2, which_measure_2)):
        if measure == "penalty":
            ax.set_yscale("symlog", linthresh=penalty_scale)
        else:
            ax.set_yscale("log")

    def first_hit_index(resid, measure):
        if measure == "penalty" or threshold is None:
            return len(resid)
        hits = np.flatnonzero(resid <= threshold)
        return hits[0] + 1 if hits.size else len(resid)

    for i in range(len(histories)):
        resid = np.array(histories[i][which_measure_1]) - f_best
        resid_2 = np.array(histories[i][which_measure_2]) - f_best

        print(f"Last {ylabel} of {labels[i]}: {resid_2[-1]}")

        # Apply the optional entry cap and threshold separately to both panels.
        n_iter = n_iters[i] if n_iters is not None else resid.shape[0]
        n_iter = min(n_iter, first_hit_index(resid, which_measure_1))
        n_iter_2 = (
            n_iters[i] if n_iters is not None else resid_2.shape[0]
        )
        n_iter_2 = min(
            n_iter_2, first_hit_index(resid_2, which_measure_2)
        )

        ax1.plot(
            histories[i]["time"][0:n_iter],
            resid[0:n_iter],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
            marker="o" if which_measure_1 == "penalty" and n_iter == 1 else None,
        )
        ax2.plot(
            range(n_iter_2),
            resid_2[0:n_iter_2],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
            marker="o" if which_measure_2 == "penalty" and n_iter_2 == 1 else None,
        )

    for ax, measure in ((ax1, which_measure_1), (ax2, which_measure_2)):
        if measure == "penalty":
            # Leave room below zero so a zero curve does not hide behind the axis.
            ax.set_ylim(bottom=-0.1 * penalty_scale)
            if not any(np.any(line.get_ydata() > 0) for line in ax.lines):
                ax.set_ylim(top=penalty_scale)

    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax2.set_xlabel("Iteration", fontsize=14)
    if max_iter is not None:
        ax2.set_xlim(0, max_iter)
    ax1.legend(
        fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Place the legend outside the data axes.
    ax1.grid()
    ax2.grid()
    plt.tight_layout()  # Fit axis labels and the external legend.
    return fig


def plot_for_paper(
    histories,
    labels,
    n,
    m,
    mu_str,
    colors,
    linewidths,
    linestyles,
    n_iters=None,
    threshold=1e-8,
    filename=None,
    f_star=None,
    suptitle=None,
    max_iter=1000,
    pair_1=["time", "func"],
    pair_2=["None", "func"],
    pair_3=["time", "func"],
    pair_4=["None", "func"],
    ylabel="Func. residual",
):
    """Display the original four-panel objective/gradient plot and save a PDF.

    Histories must contain ``func``, ``grad_norm``, and ``time``. The top row
    shows objective minus the numeric reference supplied as ``f_star``; the
    bottom row shows the gradient norm. Columns show runtime and iteration.
    Labels and style lists must match the order of the history list.

    This legacy helper assumes non-increasing plotted values when locating
    the threshold with ``searchsorted``. Gradient norms need not satisfy this
    assumption; use ``plot_figures`` for first-crossing truncation of those
    curves. Neither plotting threshold changes a solver's stopping criterion.

    Layout and output are fixed: save ``plot_for_paper.pdf`` in the current
    directory, then show the figure. ``n``, ``m``, ``mu_str``, ``n_iters``,
    ``filename``, ``suptitle``, ``max_iter``, and ``pair_1`` through ``pair_4``
    are unused; ``ylabel`` is replaced by the panel-specific labels. The
    function returns None and leaves closing the figure to the caller.
    """
    all_pairs = [
        ["time", "func"],
        [None, "func"],
        ["time", "grad_norm"],
        [None, "grad_norm"],
    ]

    f_best_for_grad = 0
    f_best_for_func = f_star

    fig, axes = plt.subplots(2, 2, figsize=(10, 4))  # 2x2 grid
    axes = axes.flatten()  # Iterate over subplot axes in row-major order.

    for i in range(len(histories)):
        resid_all = []
        for which_measure in [
            all_pairs[0][1],
            all_pairs[1][1],
            all_pairs[2][1],
            all_pairs[3][1],
        ]:
            if (
                which_measure == "func_at_lssolve"
                or which_measure == "func_at_acceptance"
                or which_measure == "func"
            ):
                f_best = f_best_for_func
            else:
                f_best = f_best_for_grad
            resid_all.append(np.array(histories[i][which_measure]) - f_best)

        n_iter_all = []
        for q in range(4):
            n_iter_all.append(resid_all[q].shape[0])
        # searchsorted requires -residual to be sorted (residual non-increasing).
        for q in range(4):
            n_iter_all[q] = min(
                n_iter_all[q], np.searchsorted(-resid_all[q], -threshold) + 1
            )
        for j, ax in enumerate(axes):
            if all_pairs[j][0] is None:
                xquantity = range(n_iter_all[j])
            else:
                xquantity = histories[i][all_pairs[j][0]][0 : n_iter_all[j]]

            if True:
                ax.semilogy(
                    xquantity,
                    resid_all[j][0 : n_iter_all[j]],
                    label=labels[i],
                    linestyle=linestyles[i],
                    color=colors[i],
                    linewidth=linewidths[i],
                )
                ax.set_xlabel("Time" if all_pairs[j][0] is not None else "Iteration")
                if (
                    all_pairs[j][1] == "func"
                    or all_pairs[j][1] == "func_at_lssolve"
                    or all_pairs[j][1] == "func_at_acceptance"
                ):
                    ylabel = "Function residual"
                else:
                    ylabel = "Gradient norm"
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)


    # Shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plot_for_paper.pdf")  # Export as PDF in the current working directory.

    plt.show()


def plot_for_paper_2(
    histories,
    labels,
    n,
    m,
    mu_str,
    colors,
    linewidths,
    linestyles,
    n_iters=None,
    threshold=1e-8,
    filename=None,
    f_star=None,
    suptitle=None,
    max_iter=None,
    which_measure="func",
    marker_cycle=None,
    markevery_numbers=None,
):
    # Use the same measure in both panels; only the horizontal axis changes.
    """Display one measure against runtime and iteration, with method markers.

    Each history needs ``time`` and ``which_measure``. Objective keys (``func``,
    ``func_at_lssolve``, ``func_at_acceptance``) subtract the numeric ``f_star``;
    other measures are plotted relative to zero. Labels, styles, ``marker_cycle``,
    and ``markevery_numbers`` need one entry per history in matching order.
    Marker spacing is measured in recorded entries, not seconds.

    Threshold lookup uses ``searchsorted`` and assumes non-increasing values;
    it can trim nonmonotone gradient curves incorrectly. The runtime axis is
    fixed to 0--470 seconds. The iteration axis uses ``max_iter`` if supplied,
    otherwise it also spans 0--470.

    Save ``plot_MF_<which_measure>.pdf`` in the current directory and show the
    figure. ``n``, ``m``, ``mu_str``, ``n_iters``, ``filename``, and ``suptitle``
    are unused compatibility arguments. Return None; the caller closes the
    figure when it is no longer needed.
    """
    all_pairs = [["time", which_measure], [None, which_measure]]

    f_best_for_grad = 0
    f_best_for_func = f_star

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Runtime and iteration panels.
    axes = axes.flatten()  # Iterate over subplot axes in row-major order.

    for i in range(len(histories)):
        resid_all = []
        for which_measure in [all_pairs[0][1], all_pairs[1][1]]:
            if (
                which_measure == "func_at_lssolve"
                or which_measure == "func_at_acceptance"
                or which_measure == "func"
            ):
                f_best = f_best_for_func
            else:
                f_best = f_best_for_grad
            resid_all.append(np.array(histories[i][which_measure]) - f_best)

        n_iter_all = []
        for q in range(2):
            n_iter_all.append(resid_all[q].shape[0])
        # Legacy monotonicity assumption; gradient norms may violate it.
        for q in range(2):
            n_iter_all[q] = min(
                n_iter_all[q], np.searchsorted(-resid_all[q], -threshold) + 1
            )


        for j, ax in enumerate(axes):
            if all_pairs[j][0] is None:
                xquantity = range(n_iter_all[j])
            else:
                xquantity = histories[i][all_pairs[j][0]][0 : n_iter_all[j]]
            if all_pairs[j][0] == None and max_iter is not None:
                ax.set_xlim(0, max_iter)
            else:
                ax.set_xlim(0, 470)  # Fixed runtime range; also used if max_iter is None.
            if True:
                ax.semilogy(
                    xquantity,
                    resid_all[j][0 : n_iter_all[j]],
                    label=labels[i],
                    linestyle=linestyles[i],
                    color=colors[i],
                    linewidth=linewidths[i],
                    marker=marker_cycle[i],
                    markevery=markevery_numbers[i],
                    markersize=5
                )
                ax.set_xlabel("Runtime (s)" if all_pairs[j][0] is not None else "Iteration")
                if (
                    all_pairs[j][1] == "func"
                    or all_pairs[j][1] == "func_at_lssolve"
                    or all_pairs[j][1] == "func_at_acceptance"
                ):
                    ylabel = "Function residual"
                else:
                    ylabel = "Gradient norm"
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        "plot_MF_" + which_measure + ".pdf"
    )  # Export as PDF in the current working directory.
    plt.show()


def plot_for_paper_penalty(
    histories,
    labels,
    colors,
    linewidths,
    linestyles,
    x_pair=None,
    threshold=1e-8,
    filename="plot_MF_penalty.pdf",
    marker_cycle=None,
    max_time=None,
):
    # The oracle has already included beta/2 in this recorded penalty.
    """Display and save the weighted nonnegativity penalty for each method.

    ``history["penalty"]`` is beta/2 times the sum of squared negative factor
    entries. ``x_pair=None`` uses zero-based iteration indices; a history key
    such as ``"time"`` uses that recorded series instead. Despite its name,
    ``max_time`` limits whichever x-axis is selected, in that axis's units.

    ``threshold`` keeps the first entry at or below the threshold and trims
    the remainder; it does not stop optimization. Style lists, including
    ``marker_cycle``, must align with ``histories`` and ``labels``. The fixed
    marker-spacing list currently supports at most eight methods.

    Save to ``filename`` (by default ``plot_MF_penalty.pdf``), then show the
    figure. Return None and leave closing the figure to the caller.
    """
    y_pair = "penalty"
    # Sparse markers for the much longer gradient-descent history.
    markevery_numbers = [500000, 500, 500, 500, 500, 500, 500, 500]

    plt.figure(figsize=(6, 4))

    for i, hist in enumerate(histories):
        y_data = np.array(hist[y_pair])
        n_iter = len(y_data)

        # Keep the first threshold hit, even when the penalty is nonmonotone.
        below_threshold = np.where(y_data <= threshold)[0]
        if len(below_threshold) > 0:
            n_iter = min(n_iter, below_threshold[0] + 1)

        # Determine x-axis
        if x_pair is None:
            x_data = range(n_iter)
        else:
            x_data = hist[x_pair][:n_iter]

        # Plot
        plt.semilogy(
            x_data,
            y_data[:n_iter],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
            marker=marker_cycle[i],
            markevery=markevery_numbers[i],
            markersize=5
        )
    # max_time is an axis bound: iterations for x_pair=None, seconds for "time".
    if max_time is not None:
        plt.xlim(left=0, right=max_time)

    plt.xlabel("Iteration" if x_pair is None else x_pair)
    plt.ylabel("Penalty")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_particular_parameter(
    parameter_history,
    title=None,
    x_label="Iterations",
    y_label="Regularization parameter",
):
    """Plot a scalar parameter history against one-based entry indices.

    ``parameter_history`` contains one scalar per recorded entry, for example
    a regularization parameter. Axis labels and the title are configurable.
    The figure is left open for the notebook to display or save; this helper
    neither saves a file nor returns the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, len(parameter_history) + 1))

    ax.plot(x, parameter_history)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid()


def plot_graphs(
    histories,
    labels,
    n,
    m,
    mu_str,
    colors,
    linewidths,
    linestyles,
    n_iters=None,
    threshold=1e-8,
    filename=None,
    f_star=None,
    suptitle=None,
    max_iter=1000,
):
    """Plot objective residuals against iteration (left) and runtime (right).

    Each history needs matching ``func`` and ``time`` series. Labels and style
    lists must follow history order. ``f_star=None`` chooses the best observed
    objective across histories as a reference, not a known global optimum.

    ``n_iters`` optionally caps the number of entries per method. Threshold
    lookup assumes non-increasing residuals. ``max_iter`` additionally limits
    the number of entries drawn in the left panel, including the initial
    point; the runtime panel keeps the full threshold-trimmed series.

    ``n``, ``m``, and ``mu_str`` are unused legacy arguments. Save only when
    ``filename`` is supplied. Return None and leave the figure open for the
    caller to show or close.
    """
    if f_star is None:
        f_best = min(histories[0]["func"])
        for i in range(1, len(histories)):
            f_best = min(f_best, min(histories[i]["func"]))
    else:
        f_best = f_star

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    for i in range(len(histories)):
        resid = np.array(histories[i]["func"]) - f_best
        n_iter = n_iters[i] if n_iters is not None else resid.shape[0]
        # This legacy lookup assumes the objective residual is non-increasing.
        n_iter = min(n_iter, np.searchsorted(-resid, -threshold) + 1)

        ax1.semilogy(
            resid[0 : min(n_iter, max_iter)],
            label=labels[i],
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
        )
        ax2.semilogy(
            histories[i]["time"][0:n_iter],
            resid[0:n_iter],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
        )

    ax1.set_xlabel("Iterations", fontsize=14)
    ax1.set_ylabel("Func. residual", fontsize=14)
    ax2.set_xlabel("Time", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid()
    ax2.grid()
    if filename:
        print("output: %s" % filename)
        plt.savefig(filename)


def plot_mf_comparison(histories, styles, output_dir=None):
    """Show three comparison figures, optionally saving each as PNG and SVG.

    ``histories`` maps method labels to dictionaries containing equally long
    ``time``, ``func``, ``grad_norm``, and ``penalty`` series. ``styles`` maps
    the same labels to Matplotlib color/linestyle options. Each figure plots
    one measure against runtime and zero-based outer iteration. No objective
    reference is subtracted and no threshold truncation is applied.

    ``output_dir=None`` only displays plots. Otherwise create the directory
    if needed and save ``objective``, ``gradient_norm``, and ``penalty`` in
    both formats. If an existing image cannot be overwritten (for example,
    a Windows preview keeps it mapped), save a uniquely named copy and warn
    with its path; the old file remains at the original path. Other saving
    errors propagate. Every figure is closed, including after an error.
    Return None.
    """
    import errno
    import warnings
    from pathlib import Path
    from uuid import uuid4

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    measures = (
        ("func", "Objective F(U, V)", "objective"),
        ("grad_norm", "Full gradient norm", "gradient_norm"),
        ("penalty", "Weighted nonnegativity penalty", "penalty"),
    )
    for measure, ylabel, filename in measures:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        try:
            for label, history in histories.items():
                values = np.asarray(history[measure], dtype=float)
                for ax, x in zip(axes, (history["time"], np.arange(len(values)))):
                    ax.semilogy(x, values, label=label, linewidth=1.8, **styles[label])
            axes[0].set_xlabel("Runtime (s)")
            axes[1].set_xlabel("Outer iteration")
            axes[0].set_ylabel(ylabel)
            for ax in axes:
                ax.grid(True, which="both", alpha=0.25)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.85))
            if output_dir is not None:
                for extension in ("png", "svg"):
                    destination = output_dir / f"{filename}.{extension}"
                    try:
                        fig.savefig(destination, dpi=170, bbox_inches="tight")
                    except OSError as error:
                        # Windows can reject truncation of an image still held open.
                        if error.errno not in (errno.EINVAL, errno.EACCES) or not destination.is_file():
                            raise
                        alternative = destination.with_name(
                            f"{filename}_rerun_{uuid4().hex[:8]}.{extension}"
                        )
                        fig.savefig(alternative, dpi=170, bbox_inches="tight")
                        warnings.warn(
                            f"Could not overwrite {destination} ({error}). "
                            f"Saved the current plot to {alternative}. "
                            "Close any preview of the original image before retrying.",
                            RuntimeWarning, stacklevel=2,
                        )
            plt.show()
        finally:
            # Release the figure even if rendering or saving fails on a rerun.
            plt.close(fig)
