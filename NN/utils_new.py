"""Corrected NN helpers; utils.py remains the original implementation.

Paper alignment and validation: CORRECTIONS.md. Third-party credits:
../THIRD_PARTY.txt. Parameter packing and legacy plotting helpers are unchanged.
"""
import math

import torch
from torch.func import functional_call


def make_param_vector_interface(model, trainable_only=True):
    names, shapes, numels, params = [], [], [], []
    for name, p in model.named_parameters():
        if (not trainable_only) or p.requires_grad:
            names.append(name)
            shapes.append(p.shape)
            numels.append(p.numel())
            params.append(p)
    total = sum(numels)

    def flatten_parameters():
        return torch.cat([p.detach().reshape(-1) for p in params])

    def vec_to_param_list(vec):
        out, off = [], 0
        for n, shape in zip(numels, shapes):
            out.append(vec[off : off + n].view(shape))
            off += n
        assert off == vec.numel() == total
        return out

    def vec_to_param_dict(vec):
        return {k: v for k, v in zip(names, vec_to_param_list(vec))}

    @torch.no_grad()
    def assign_parameters_from_vector(vec):
        off = 0
        for p, n, shape in zip(params, numels, shapes):
            p.copy_(vec[off : off + n].view(shape))
            off += n
        assert off == vec.numel() == total

    return flatten_parameters, vec_to_param_dict, assign_parameters_from_vector, total


def hessian_norm_penalty(
    model, params_dict, xb, v0, coeff=0.1, K=3, detach_power_iter=False,
    *, normalization_eps=1e-12,
):
    """Return (mean penalty, maximum sample norm estimate) for scalar outputs.

    xb and v0 have shape (N, d); model maps each input independently to a
    scalar. params_dict contains the differentiable network parameters.
    coeff is the paper's penalty lambda, so the penalty is
        coeff / (2*N) * sum_i relu(squared_norm_estimate_i - 1)**2.
    This implements local equation (26), Section 5.1 / Appendix E.1
    (public arXiv:2602.08069v1 equation (27)); see CORRECTIONS.md.

    K fixed power iterations approximate each input Hessian's spectral norm,
    i.e. its largest absolute eigenvalue. Normalize each sample separately
    using sqrt(||w_i||**2 + normalization_eps**2), as in local Remark E.1.
    The default normalization_eps=1e-12 reproduces the previous 1e-24 term.
    Increasing it changes the finite-power surrogate and can underestimate
    the norm more strongly; use the same value for every compared optimizer.
    Differentiate
    through the entire recursion by default. detach_power_iter=True is an
    explicit derivative approximation and is NOT used by the new notebook.

    The second result is a scalar maximum over the sampled inputs for plotting;
    it is an approximation, not a certified global Lipschitz bound.
    """
    if K < 1:
        raise ValueError("K must be at least one")
    if not math.isfinite(normalization_eps) or normalization_eps <= 0:
        raise ValueError("normalization_eps must be finite and positive")
    eps_squared = normalization_eps**2
    xb = xb.detach().requires_grad_(True)
    y = functional_call(model, params_dict, xb).sum()
    grad = torch.autograd.grad(y, xb, create_graph=True)[0]  # (N, d)

    v = v0.to(grad)
    v = v / torch.sqrt(v.square().sum(dim=1, keepdim=True) + eps_squared)

    Hv = None
    for _ in range(K):
        Hv = torch.autograd.grad(
            grad, xb, grad_outputs=v, retain_graph=True, create_graph=True
        )[0]
        v_next = Hv / torch.sqrt(Hv.square().sum(dim=1, keepdim=True) + eps_squared)
        v = v_next.detach() if detach_power_iter else v_next
    # Work with squared norms directly: differentiating sqrt at zero is
    # unnecessary for the penalty and can spoil its second derivatives.
    squared_norms = Hv.square().sum(dim=1)
    penalty = 0.5 * coeff * torch.relu(squared_norms - 1).square().mean()
    max_norm_estimate = squared_norms.max().sqrt()
    return penalty, max_norm_estimate


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
