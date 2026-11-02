"""Runtime/iteration plots for the newer multiple-seed MF notebooks.

Histories contain func, grad_norm, penalty and time. Plot thresholds affect
display only; penalty panels retain zeros and later rebounds.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import run_methods


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
