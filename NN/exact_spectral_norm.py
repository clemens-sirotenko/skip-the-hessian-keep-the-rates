"""Direct sampled input-Hessian norms with optional Huber spectral smoothing.

This evaluates the spectral-norm penalty in local equation (26), Section 5.1
(arXiv:2602.08069v1 equation (27)), instead of Appendix E.1's finite-power
surrogate. Positive spectral_smoothing_eps enables a separate C1 Huber
surrogate with a semismooth gradient. See CORRECTIONS.md for scope and validation;
../THIRD_PARTY.txt retains PyTorch/framework attribution.
"""
import math

import torch
from torch.func import functional_call


class ActiveSpectralTieError(RuntimeError):
    """An active spectral maximum has no numerically resolved unique branch."""


# The exploratory notebook continues at ties. Set to "raise" to restore the
# original strict check, or pass tie_policy="raise" to a particular call.
DEFAULT_TIE_POLICY = "average"


def input_hessians_2d(model, params_dict, xb):
    """Return differentiable (N, 2, 2) input Hessians for scalar predictions.

    xb has shape (N, 2). Each prediction must depend only on its corresponding
    input row, as in the experiment's tanh MLP (no cross-sample operations).
    params_dict may contain views into a differentiable parameter vector.
    Both input differentiations retain their graphs so parameter gradients
    and parameter Hessians include the penalty's complete dependence.
    """
    if xb.ndim != 2 or xb.shape[1] != 2 or xb.shape[0] == 0:
        raise ValueError("Expected a nonempty input batch with shape (N, 2)")
    x = xb.detach().requires_grad_(True)
    pred = functional_call(model, params_dict, x)
    if pred.shape not in (x.shape[:1], (len(x), 1)):
        raise ValueError("The model must return one scalar per input row")
    gx = torch.autograd.grad(
        pred.sum(), x, create_graph=True, allow_unused=True
    )[0] if pred.requires_grad else None
    if gx is None:
        gx = x * 0.0 + pred.sum() * 0.0
    rows = []
    for j in range(2):
        component = gx[:, j].sum()
        row = torch.autograd.grad(
            component, x, create_graph=True, retain_graph=True, allow_unused=True
        )[0] if component.requires_grad else None
        if row is None:
            row = x * 0.0 + component * 0.0
        rows.append(row)
    hessians = torch.stack(rows, dim=1)
    return 0.5 * (hessians + hessians.transpose(-1, -2))


def smoothed_squared_spectral_norm_2d(hessians, epsilon):
    """Huber upper approximation to ||B||_2**2 for symmetric (..., 2, 2) B.

    Let A=B@B and r=sqrt((A00-A11)**2+4*A01**2), the squared-eigenvalue gap.
    Return (tr(A)+h_epsilon(r))/2, where h_epsilon(r)=r for r>=epsilon
    and r**2/(2*epsilon)+epsilon/2 for r<=epsilon. Thus this equals the
    exact squared spectral norm outside the tie band and exceeds it by at
    most epsilon/4 inside. This is the usual Huber absolute value plus a
    constant epsilon/2, chosen to agree with absolute value outside the band.

    The matrix-entry expression is C1 with a locally strongly semismooth
    gradient, generally not C2 at r=epsilon. It handles repeated and
    opposite-sign eigenvalue ties without eigenvector derivatives.
    Epsilon has squared-norm units and is distinct from power normalization
    epsilon. Larger values trade approximation accuracy for less sharp
    spectral transitions. Composition with the outer squared hinge retains
    a C1 objective with a locally semismooth gradient.
    """
    if hessians.shape[-2:] != (2, 2):
        raise ValueError("Expected symmetric matrices with shape (..., 2, 2)")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    squared = hessians @ hessians
    trace = squared[..., 0, 0] + squared[..., 1, 1]
    discriminant = (squared[..., 0, 0] - squared[..., 1, 1]).square() + 4*squared[..., 0, 1].square()
    inside = discriminant <= epsilon**2
    # torch.where evaluates both branches: avoid sqrt(0) even on the inactive
    # outer branch, otherwise its higher derivatives can contaminate the result.
    outer_radius = torch.sqrt(torch.where(inside, torch.full_like(discriminant, epsilon**2), discriminant))
    huber_radius = torch.where(inside, discriminant/(2*epsilon)+epsilon/2, outer_radius)
    return 0.5 * (trace + huber_radius)


def exact_hessian_norm_penalty(
    model, params_dict, xb, coeff=1.0, *, tie_policy=None, spectral_smoothing_eps=0.0,
):
    """Return (sampled penalty, detached EXACT maximum sampled spectral norm).

    For B_i = Hessian_x(model(x_i)), compute q_i = max_j eigenvalue_j(B_i)^2
    and coeff/(2*N) * sum_i relu(q_i - 1)^2. `Exact` means direct symmetric
    eigendecomposition up to floating-point error, not a global bound over x.
    There is no power iteration or random starting vector. With
    spectral_smoothing_eps=0 (default), the penalty uses the exact squared norm.
    A positive value uses smoothed_squared_spectral_norm_2d instead: this
    changes the objective to a Huber squared-norm surrogate followed by the
    same squared hinge, eliminating the need for a tie derivative selection.
    The returned plotting metric always uses the exact sampled norm, so the
    diagnostic is not inflated by the smoothing's known upper bias.

    In unsmoothed mode, eigenvalues used for the penalty remain differentiable. Inactive samples
    are excluded from that eigendecomposition: their penalty is locally zero,
    including at repeated eigenvalues inside the unit ball. This avoids NaNs
    in higher derivatives of otherwise inactive terms. At q_i=1 the zero
    second-order branch is selected, as for the squared-ReLU implementation.

    Active ties in absolute eigenvalue can make the objective non-C1. The
    default tie_policy="average" continues with averaged squared-eigenvalue
    derivatives. For a squared-eigenvalue gap <= 64*eps*max(1,q_i), retain the
    exact forward q_i but differentiate trace(B_i**2)/2 = ||B_i||_F**2/2.
    For symmetric 2-by-2 matrices this averages the two squared eigenvalues;
    at an exact tie it selects a valid subgradient of their maximum. It also
    avoids undefined eigenvector derivatives at repeated eigenvalues. The
    resulting curvature is an explicit approximation at flagged ties (and
    near ties), not a classical Hessian of a differentiable spectral maximum.
    Away from flagged ties, full exact-norm derivatives are unchanged. This
    policy does not guarantee continued convergence.

    tie_policy="raise" restores ActiveSpectralTieError. None uses the module's
    DEFAULT_TIE_POLICY. The maximum returned for plotting is detached and is
    not a differentiable loss output.
    """
    if not math.isfinite(coeff) or coeff < 0:
        raise ValueError("coeff must be finite and nonnegative")
    if not math.isfinite(spectral_smoothing_eps) or spectral_smoothing_eps < 0:
        raise ValueError("spectral_smoothing_eps must be finite and nonnegative")
    if tie_policy is None:
        tie_policy = DEFAULT_TIE_POLICY
    if tie_policy not in ("average", "raise"):
        raise ValueError("tie_policy must be 'average' or 'raise'")
    hessians = input_hessians_2d(model, params_dict, xb)
    # Branch selection is discrete; retained branches below still use the full
    # parameter graph. The metric includes all samples, including inactive ones.
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(hessians.detach())
        squared = eigenvalues.square()
        norm_squared = squared.amax(dim=-1)
        maximum = norm_squared.max().sqrt()
        active = norm_squared > 1.0
        gap = (squared[:, 0] - squared[:, 1]).abs()
        tolerance = 64 * torch.finfo(hessians.dtype).eps * norm_squared.clamp_min(1.0)
        tied = active & (gap <= tolerance)
    if spectral_smoothing_eps > 0:
        smoothed_squared = smoothed_squared_spectral_norm_2d(hessians, spectral_smoothing_eps)
        penalty = 0.5 * coeff * torch.relu(smoothed_squared - 1).square().mean()
        return penalty, maximum
    if tie_policy == "raise" and coeff and bool(tied.any()):
        indices = tied.nonzero(as_tuple=False).flatten().tolist()
        raise ActiveSpectralTieError(
            f"Active spectral-norm tie at sample indices {indices}; "
            "the exact objective has no numerically resolved smooth branch."
        )
    # Retain a zero derivative connection even if every penalty is inactive.
    penalty = hessians.square().sum() * 0.0
    if coeff and bool(active.any()):
        # Preserve the existing value and derivatives away from flagged ties.
        regular = active & ~tied
        if bool(regular.any()):
            values = torch.linalg.eigvalsh(hessians[regular])
            q = values.square().amax(dim=-1)
            penalty = penalty + (0.5 * coeff / len(xb)) * torch.relu(q - 1).square().sum()
        if bool(tied.any()):
            averaged_squared = 0.5 * hessians[tied].square().sum(dim=(-1, -2))
            # Exact forward values, with the explicitly documented derivative
            # selection. No sample or penalty contribution is discarded.
            q = norm_squared[tied] + (averaged_squared - averaged_squared.detach())
            penalty = penalty + (0.5 * coeff / len(xb)) * torch.relu(q - 1).square().sum()
    return penalty, maximum
