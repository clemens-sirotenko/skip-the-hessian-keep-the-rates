import numpy as np
import matplotlib.pyplot as plt

from oracles_mat_fac import *


# from scipy.sparse import diags


def run_methods(methods_to_run, oracle, x_0):
    history_dict = {}

    for method_name, (func, args, kwargs) in methods_to_run.items():
        print(f"\nRunning {method_name}...")

        # Call the method
        _, status, history = func(*args, **kwargs)
        print(f"Status: {status}")

        # Store results
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
):
    if f_star is None:
        f_best = min(histories[0]["func"])
        for i in range(1, len(histories)):
            f_best = min(f_best, min(histories[i]["func"]))
    else:
        f_best = f_star

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4)
    )  # Increased figure width for legend
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    for i in range(len(histories)):
        resid = np.array(histories[i][which_measure_1]) - f_best
        resid_2 = np.array(histories[i][which_measure_2]) - f_best

        print(f"last func residual of {labels[i]}: {resid_2[-1]}")

        def first_hit_index(resid, threshold):
            ## just for nonmonotone residuals
            hits = np.flatnonzero(resid <= threshold)
            return (hits[0] + 1) if hits.size else len(resid)

        # We always have n_iters=None
        n_iter = n_iters[i] if n_iters is not None else resid.shape[0]
        n_iter = min(n_iter, first_hit_index(resid, threshold))
        # n_iter = min(n_iter, np.searchsorted(-resid, -threshold)+1) #for monotone residuals
        n_iter_2 = (
            n_iters[i] if n_iters is not None else resid_2.shape[0]
        )  # for nonmonotone residuals
        # n_iter_2 = min(n_iter_2, np.searchsorted(-resid_2, -threshold)+1) #for monotone residuals
        n_iter_2 = min(
            n_iter_2, first_hit_index(resid_2, threshold)
        )  # for nonmonotone residuals
        # print(n_iter,resid.shape[0])

        ax1.semilogy(
            histories[i]["time"][0:n_iter],
            resid[0:n_iter],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
        )
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
    ax2.set_xlabel("Iteration", fontsize=14)
    ax1.legend(
        fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Moved legend outside
    ax1.grid()
    ax2.grid()
    plt.tight_layout()  # Added tight_layout after legend for better fit


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
    all_pairs = [
        ["time", "func"],
        [None, "func"],
        ["time", "grad_norm"],
        [None, "grad_norm"],
    ]

    f_best_for_grad = 0
    f_best_for_func = f_star

    fig, axes = plt.subplots(2, 2, figsize=(10, 4))  # 2x2 grid
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
            resid_all.append(np.array(histories[i][which_measure]) - f_best)

        n_iter_all = []
        for q in range(4):
            n_iter_all.append(resid_all[q].shape[0])
            # TODO:
        for q in range(4):
            n_iter_all[q] = min(
                n_iter_all[q], np.searchsorted(-resid_all[q], -threshold) + 1
            )
        for j, ax in enumerate(axes):
            if all_pairs[j][0] is None:
                xquantity = range(n_iter_all[j])
            else:
                xquantity = histories[i][all_pairs[j][0]][0 : n_iter_all[j]]

            if True:  # all_pairs[j][0] != None or labels.index("Gradient Method") != i:
                ax.semilogy(
                    xquantity,
                    resid_all[j][0 : n_iter_all[j]],  # j+1 if resid_all[0] is dummy
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

                # Optional: title per subplot — change as needed
                # ax.set_title(f"Residual {j+1}")

    # Shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plot_for_paper.pdf")  # creates a PGF file (LaTeX/PGF code)

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
    # which_measure should be either "func" or "grad_norm"
    all_pairs = [["time", which_measure], [None, which_measure]]
    # markevery_numbers = [50000, 500, 500, 500, 500, 500, 500, 500]

    f_best_for_grad = 0
    f_best_for_func = f_star

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 2x2 grid
    axes = axes.flatten()  # turn into a simple list: axes[0], axes[1], axes[2], axes[3]

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
            # TODO:
        for q in range(2):
            n_iter_all[q] = min(
                n_iter_all[q], np.searchsorted(-resid_all[q], -threshold) + 1
            )

        # print("histories...", len(histories[i]["time"][0:n_iter_all[0]]))
        # print("residual 0...", len(resid_all[0][0:n_iter_all[0]]))

        for j, ax in enumerate(axes):
            if all_pairs[j][0] is None:
                xquantity = range(n_iter_all[j])
            else:
                xquantity = histories[i][all_pairs[j][0]][0 : n_iter_all[j]]
            if all_pairs[j][0] == None and max_iter is not None:
                ax.set_xlim(0, max_iter)
            else:
                ax.set_xlim(0, 470) # limit x-axis to 600 seconds for time plots
            # print(xquantity)
            # ax.semilogy(histories[i]["time"][0:n_iter_all[j]], resid_all[j][0:n_iter_all[j]],  # j+1 if resid_all[0] is dummy
            #             label=labels[i],
            #             linestyle=linestyles[i],
            #             color=colors[i],
            #             linewidth=linewidths[i])
            if True:  # all_pairs[j][0] != None or labels.index("Gradient Method") != i:
                ax.semilogy(
                    xquantity,
                    resid_all[j][0 : n_iter_all[j]],  # j+1 if resid_all[0] is dummy
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
    )  # creates a PGF file (LaTeX/PGF code)
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
    # Currently only plotting "penalty"
    # x_pair = "time"  # x-axis data is None => use iteration numbers
    y_pair = "penalty"
    markevery_numbers = [500000, 500, 500, 500, 500, 500, 500, 500]

    plt.figure(figsize=(6, 4))

    for i, hist in enumerate(histories):
        y_data = np.array(hist[y_pair])
        n_iter = len(y_data)

        # Find first index where residual <= threshold
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
    """
    Plots the regularization parameters over iterations.
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
    # plt.tight_layout()
    if filename:
        print("output: %s" % filename)
        plt.savefig(filename)


def plot_mf_comparison(histories, styles, output_dir=None):
    """Plot every recorded iterate against runtime and outer iteration.

    Both comparison notebooks use this helper. Styles map each method label
    to Matplotlib color/linestyle options. No objective reference is subtracted.
    If an existing image cannot be overwritten (for example, a Windows
    preview keeps it mapped), save a uniquely named copy and report its path.
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
            plt.close(fig)
