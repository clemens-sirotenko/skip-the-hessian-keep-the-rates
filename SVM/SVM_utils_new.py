import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np


def _best_observed(histories):
    """Finite objective reference from these runs; not a certified optimum."""
    values = np.concatenate([np.asarray(h["func"], dtype=float) for h in histories])
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite objective values to plot")
    return float(values.min())


def _residual_series(history, key, reference, threshold):
    """First threshold crossing, including its point; floor values for a log axis.

    Unlike searchsorted, this works for nonmonotone trial/gradient sequences.
    Objective subtraction has a scale-dependent floating-point floor. Terminal
    nonfinite entries are excluded rather than connected across a plotting gap.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive for a logarithmic plot")
    reference = reference if key.startswith("func") else 0.0
    values = np.asarray(history.get(key, []), dtype=float) - reference
    floor = max(threshold, 10 * np.finfo(float).eps * max(1.0, abs(reference)))
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        values = values[:bad[0]]
    reached = np.flatnonzero(values <= floor)
    if reached.size:
        values = values[:reached[0] + 1]
    return np.maximum(values, floor)


def _iteration_label(key, labels):
    if key.endswith("at_lssolve"):
        return "Trial step (including rejected trials)"
    return "Outer iteration / SVRG epoch" if "SVRG-BB" in labels else "Outer iteration"


def _safe_log_ticks(axes):
    """Keep extreme finite divergence visible without overflowing log tick powers."""
    for ax in axes:
        low, high = np.log10(ax.get_ylim())
        if high > 280 or high - low > 250:
            exponents = np.unique(np.linspace(max(-307, np.ceil(low)),
                                             min(308, np.floor(high)), 9).astype(int))
            ax.yaxis.set_major_locator(FixedLocator(10.0 ** exponents))
            ax.yaxis.set_minor_locator(NullLocator())


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
    which_measure_2="func",
    ylabel="Objective gap to best observed",
):
    if f_star is None:
        f_best = _best_observed(histories)
    else:
        f_best = f_star

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4)
    )  # Increased figure width for legend
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    for i in range(len(histories)):
        resid = _residual_series(histories[i], which_measure_1, f_best, threshold)
        resid_2 = _residual_series(histories[i], which_measure_2, f_best, threshold)

        # We always have n_iters=None
        n_iter = n_iters[i] if n_iters is not None else resid.shape[0]
        n_iter = min(n_iter, len(resid))
        n_iter_2 = n_iters[i] if n_iters is not None else resid_2.shape[0]
        n_iter_2 = min(n_iter_2, len(resid_2))

        ax1.semilogy(
            histories[i]["time"][0:n_iter],
            resid[0:n_iter],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
        )
        if n_iter_2:
            ax2.semilogy(
                range(n_iter_2),
                resid_2[0:n_iter_2],
                label=labels[i],
                linestyle=linestyles[i],
                color=colors[i],
                linewidth=linewidths[i],
            )

    ax1.set_xlabel("Time", fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    # ax2.set_xlabel('No. LS solves', fontsize=14)
    ax2.set_xlabel(_iteration_label(which_measure_2, labels), fontsize=14)
    ax1.legend(
        fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Moved legend outside
    ax1.grid()
    ax2.grid()
    _safe_log_ticks((ax1, ax2))
    plt.tight_layout()  # Added tight_layout after legend for better fit
    if filename is not None:
        fig.savefig(filename)
    return fig, (ax1, ax2)


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
    ylabel="Objective gap to best observed",
    clip_time=None,
):
    all_pairs = [
        ["time", "func"],
        [None, "func"],
        ["time", "grad_norm"],
        [None, "grad_norm"],
    ]

    f_best_for_grad = 0
    f_best_for_func = _best_observed(histories) if f_star is None else f_star

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
    axes = axes.flatten()  # turn into a simple list: axes[0], axes[1], axes[2], axes[3]

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
            resid_all.append(_residual_series(histories[i], which_measure, f_best, threshold))

        n_iter_all = []
        for q in range(4):
            n_iter_all.append(resid_all[q].shape[0])
            # TODO:

        for j, ax in enumerate(axes):
            if all_pairs[j][0] is None:
                xquantity = range(n_iter_all[j])
            else:
                xquantity = histories[i][all_pairs[j][0]][0 : n_iter_all[j]]
            if n_iter_all[j]:
                ax.semilogy(
                    xquantity,
                    resid_all[j][0 : n_iter_all[j]],  # j+1 if resid_all[0] is dummy
                    label=labels[i],
                    linestyle=linestyles[i],
                    color=colors[i],
                    linewidth=linewidths[i],
                )
                # clip time axis to 5 seconds
                if clip_time is not None and all_pairs[j][0] == "time":
                    ax.set_xlim(0, clip_time)
                ax.set_xlabel("Time" if all_pairs[j][0] is not None else _iteration_label(all_pairs[j][1], labels))
                if (
                    all_pairs[j][1] == "func"
                    or all_pairs[j][1] == "func_at_lssolve"
                    or all_pairs[j][1] == "func_at_acceptance"
                ):
                    ylabel = "Objective gap to best observed"
                else:
                    ylabel = "Gradient norm"
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)

    _safe_log_ticks(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if filename is not None:
        fig.savefig(filename)

    plt.show()
    return fig, axes

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
    max_iter=1000,
    pair_1=["time", "func"],
    pair_2=["None", "func"],
    pair_3=["time", "func"],
    pair_4=["None", "func"],
    ylabel="Objective gap to best observed",
    clip_time=None,
    m_cycle= [
    "None",   # None Gradient Method
    "P",   # plus-filled SVRG-BB
    "*",   # star Cubic Newton
    "s",   # square LeapSSN
    "^",   # triangle up GladSSN
    "D",   # diamond GladSSN second
    "x",   # x GladSSN third
    "v",   # triangle down GladSSN fourth
    "h",   # hexagon GladSSN fifth
    ],
    markevery_numbers = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
):
 
    all_pairs = [
        ["time", "func"],
        [None, "func"],
    ]
  
    f_best_for_grad = 0
    f_best_for_func = _best_observed(histories) if f_star is None else f_star

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 2x2 grid
    axes = axes.flatten()  # turn into a simple list: axes[0], axes[1], axes[2], axes[3]

    for i in range(len(histories)):
        resid_all = []
        for which_measure in [
            all_pairs[0][1],
            all_pairs[1][1]
        ]:
            if (
                which_measure == "func_at_lssolve"
                or which_measure == "func_at_acceptance"
                or which_measure == "func"
            ):
                f_best = f_best_for_func
            else:
                f_best = f_best_for_grad
            resid_all.append(_residual_series(histories[i], which_measure, f_best, threshold))

        n_iter_all = []
        for q in range(2):
            n_iter_all.append(resid_all[q].shape[0])
            # TODO:

        for j, ax in enumerate(axes):
            if all_pairs[j][0] is None:
                xquantity = range(n_iter_all[j])
            else:
                xquantity = histories[i][all_pairs[j][0]][0 : n_iter_all[j]]

            if all_pairs[j][0] == None:
                ax.set_xlim(0, max_iter)
            if n_iter_all[j]:
                ax.semilogy(
                    xquantity,
                    resid_all[j][0 : n_iter_all[j]],  # j+1 if resid_all[0] is dummy
                    label=labels[i],
                    linestyle=linestyles[i],
                    color=colors[i],
                    linewidth=linewidths[i],
                    # marker_size=marker_size,
                    markevery=markevery_numbers[i],
                    marker=m_cycle[i],
                )
                # clip time axis to 5 seconds
                if clip_time is not None and all_pairs[j][0] == "time":
                    ax.set_xlim(0, clip_time)
                ax.set_xlabel("Runtime (s)" if all_pairs[j][0] is not None else _iteration_label(all_pairs[j][1], labels))
                if (
                    all_pairs[j][1] == "func"
                    or all_pairs[j][1] == "func_at_lssolve"
                    or all_pairs[j][1] == "func_at_acceptance"
                ):
                    ylabel = "Objective gap to best observed"
                else:
                    ylabel = "Gradient norm"
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Original paper style, also used by MF/utils_new.py (legend_layout="paper").
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3,
        frameon=True, framealpha=0.8, fancybox=True,
        facecolor="white", edgecolor="0.8", fontsize=10,
    )

    _safe_log_ticks(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if filename is not None:
        fig.savefig(filename)

    plt.show()
    return fig, axes
