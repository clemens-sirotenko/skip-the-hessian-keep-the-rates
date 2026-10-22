import matplotlib.pyplot as plt
import numpy as np


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

        # We always have n_iters=None
        n_iter = n_iters[i] if n_iters is not None else resid.shape[0]
        n_iter = min(n_iter, np.searchsorted(-resid, -threshold) + 1)
        n_iter_2 = n_iters[i] if n_iters is not None else resid_2.shape[0]
        n_iter_2 = min(n_iter_2, np.searchsorted(-resid_2, -threshold) + 1)

        ax1.semilogy(
            histories[i]["time"][0:n_iter],
            resid[0:n_iter],
            label=labels[i],
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=linewidths[i],
        )
        if i != labels.index("Gradient Method"):
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
    clip_time=None,
):
    all_pairs = [
        ["time", "func"],
        [None, "func_at_lssolve"],
        ["time", "grad_norm"],
        [None, "grad_norm_at_lssolve"],
    ]

    f_best_for_grad = 0
    f_best_for_func = f_star

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
            if (
                all_pairs[j][0] != None or labels.index("Gradient Method") != i
            ):  # don't show gradient method here because it stretches the graph and we can't see detail
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
    plt.savefig("plot_SVM.pdf")  # creates a PGF file (LaTeX/PGF code)

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
    max_iter=1000,
    pair_1=["time", "func"],
    pair_2=["None", "func"],
    pair_3=["time", "func"],
    pair_4=["None", "func"],
    ylabel="Func. residual",
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
        [None, "func_at_lssolve"],
    ]
  
    f_best_for_grad = 0
    f_best_for_func = f_star

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
            resid_all.append(np.array(histories[i][which_measure]) - f_best)

        n_iter_all = []
        for q in range(2):
            n_iter_all.append(resid_all[q].shape[0])
            # TODO:
        for q in range(2):
            n_iter_all[q] = min(
                n_iter_all[q], np.searchsorted(-resid_all[q], -threshold) + 1
            )

        for j, ax in enumerate(axes):
            if all_pairs[j][0] is None:
                xquantity = range(n_iter_all[j])
            else:
                xquantity = histories[i][all_pairs[j][0]][0 : n_iter_all[j]]

            if all_pairs[j][0] == None:
                ax.set_xlim(0, max_iter)
            if (
                all_pairs[j][0] != None or labels.index("Gradient Method") != i
            ):  # don't show gradient method here because it stretches the graph and we can't see detail
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
    plt.savefig("plot_SVM_func.pdf")  # creates a PGF file (LaTeX/PGF code)

    plt.show()