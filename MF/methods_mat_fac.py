"""Factor alignment and metrics used by the older MF notebooks.

U has shape (m,r), V has shape (r,n); oracle.pack/unpack define vector layout.
Solvers, including alternating descent, live in methods.py.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from oracles_mat_fac import PositiveMatrixFactorization


def _normalize_cols(A):
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def align_factors(U, V, U_true, V_true):
    """Align columns of U and matching rows of V, preserving U @ V."""
    r = U_true.shape[1]
    Un, Vn = _normalize_cols(U), _normalize_cols(V.T)
    Utn, Vtn = _normalize_cols(U_true), _normalize_cols(V_true.T)

    # Match each column of U together with the corresponding row of V.
    C = np.abs(Utn.T @ Un) + np.abs(Vtn.T @ Vn)  # (r,r)
    row, col = linear_sum_assignment(-C)
    perm = col

    U_p = U[:, perm]
    V_p = V[perm, :]

    # per-component scaling to best match U_true
    U_al = np.zeros_like(U_p)
    V_al = np.zeros_like(V_p)
    for k in range(r):
        u = U_p[:, k]
        v = V_p[k, :]
        ut = U_true[:, k]
        dk = (ut @ u) / ((u @ u) + 1e-12)
        if abs(dk) < 1e-12:
            dk = 1.0
        U_al[:, k] = dk * u
        V_al[k, :] = (1.0 / dk) * v

    return U_al, V_al


def mf_terms_and_metrics(
    oracle: PositiveMatrixFactorization, z, M_clean=None, U_true=None, V_true=None
):
    U, V = oracle.unpack(z)
    M_obs = oracle.M

    R = U @ V - M_obs
    rec = 0.5 * np.sum(R * R)
    pen = 0.5 * oracle.beta * (
        np.sum(np.maximum(-U, 0.0) ** 2) + np.sum(np.maximum(-V, 0.0) ** 2)
    )

    rel_obs = np.linalg.norm(U @ V - M_obs) / (np.linalg.norm(M_obs) + 1e-12)

    rel_clean = None
    if M_clean is not None:
        rel_clean = np.linalg.norm(U @ V - M_clean) / (
            np.linalg.norm(M_clean) + 1e-12
        )

    negU = np.maximum(-U, 0.0).mean()
    negV = np.maximum(-V, 0.0).mean()

    U_err = None
    V_err = None
    if U_true is not None and V_true is not None:
        U_al, V_al = align_factors(U, V, U_true, V_true)
        U_err = np.linalg.norm(U_al - U_true) / (np.linalg.norm(U_true) + 1e-12)
        V_err = np.linalg.norm(V_al - V_true) / (np.linalg.norm(V_true) + 1e-12)

    return dict(
        rec=rec,
        pen=pen,
        rel_obs=rel_obs,
        rel_clean=rel_clean if rel_clean is not None else np.nan,
        negU=negU,
        negV=negV,
        U_err=U_err if U_err is not None else np.nan,
        V_err=V_err if V_err is not None else np.nan,
    )
