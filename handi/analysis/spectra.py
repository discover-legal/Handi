"""Singular values, empirical spectral density, Marchenko-Pastur, power-law fits.

Conventions (Martin & Mahoney, "Implicit self-regularization", 2021):
  * W is N x M after orientation so that N = max(m, n) >= M = min(m, n).
  * Correlation matrix X = W^T W / N  (M x M), eigenvalues lambda_i = sigma_i^2 / N.
  * gamma = M / N in (0, 1]  (their Q = N / M = 1 / gamma).
  * MP bulk for iid entries of variance sigma2 (per element of W):
        lambda_± = sigma2 (1 ± sqrt(gamma))^2
        rho(x)   = sqrt((lambda_+ - x)(x - lambda_-)) / (2 pi sigma2 gamma x)  on [lambda_-, lambda_+]
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Singular values
# ----------------------------------------------------------------------------


def _device(use_gpu: bool):
    import torch

    return torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")


def singular_values(W: np.ndarray, use_gpu: bool = True, gram_ratio: float = 16.0) -> np.ndarray:
    """Descending singular values of W (float32 via torch, GPU if available).

    For very tall/wide matrices (max dim / min dim >= gram_ratio, e.g. the Qwen
    248k x 1024 embedding) the spectrum is taken from eigvalsh of the Gram matrix
    computed in float64 instead of a direct SVD. Returned as float64 numpy.
    """
    import torch

    W = np.asarray(W)
    m, n = W.shape
    dev = _device(use_gpu)
    if max(m, n) / min(m, n) >= gram_ratio:
        T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float64)).to(dev)
        G = T.T @ T if m >= n else T @ T.T
        ev = torch.linalg.eigvalsh(G).clamp_min(0.0)
        s = ev.sqrt().flip(0)
    else:
        T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).to(dev)
        s = torch.linalg.svdvals(T)
    return s.detach().cpu().numpy().astype(np.float64)


def aspect(W: np.ndarray) -> float:
    """gamma = min(m, n) / max(m, n)."""
    m, n = W.shape
    return min(m, n) / max(m, n)


def esd(W: np.ndarray, s: np.ndarray | None = None, use_gpu: bool = True) -> np.ndarray:
    """Eigenvalues of W^T W / N, N = max(m, n), ascending. Pass `s` to reuse singular values."""
    if s is None:
        s = singular_values(W, use_gpu=use_gpu)
    N = max(W.shape)
    return np.sort(np.asarray(s, dtype=np.float64) ** 2 / N)


# ----------------------------------------------------------------------------
# Marchenko-Pastur
# ----------------------------------------------------------------------------


def mp_edges(gamma: float, sigma2: float) -> tuple[float, float]:
    """(lambda_-, lambda_+) = sigma2 (1 -+ sqrt(gamma))^2."""
    r = np.sqrt(gamma)
    return float(sigma2 * (1 - r) ** 2), float(sigma2 * (1 + r) ** 2)


def mp_pdf(x, gamma: float, sigma2: float) -> np.ndarray:
    """MP density for the eigenvalues of W^T W / N (see module docstring). Zero outside the bulk."""
    x = np.asarray(x, dtype=np.float64)
    a, b = mp_edges(gamma, sigma2)
    out = np.zeros_like(x)
    inside = (x > a) & (x < b)
    xi = x[inside]
    out[inside] = np.sqrt((b - xi) * (xi - a)) / (2.0 * np.pi * sigma2 * gamma * xi)
    return out


def _mp_cdf_grid(gamma: float, sigma2: float, n_grid: int = 20001):
    """Numerical MP CDF on a grid over the bulk (trapezoid), normalized to 1."""
    a, b = mp_edges(gamma, sigma2)
    xs = np.linspace(a, b, n_grid)
    pdf = mp_pdf(xs, gamma, sigma2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs))])
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]  # absorbs the trapezoid error at the sqrt endpoints
    return xs, cdf


def mp_cdf(x, gamma: float, sigma2: float) -> np.ndarray:
    """MP CDF at x via numerical integration of mp_pdf (gamma <= 1: no atom at zero)."""
    xs, cdf = _mp_cdf_grid(gamma, sigma2)
    x = np.asarray(x, dtype=np.float64)
    return np.interp(x, xs, cdf, left=0.0, right=1.0)


def mp_median(gamma: float, sigma2: float = 1.0) -> float:
    xs, cdf = _mp_cdf_grid(gamma, sigma2)
    return float(np.interp(0.5, cdf, xs))


def ks_to_mp(eigs: np.ndarray, gamma: float, sigma2: float) -> float:
    """KS distance sup|F_emp - F_MP| over the sorted eigenvalues."""
    e = np.sort(np.asarray(eigs, dtype=np.float64))
    n = e.size
    F = mp_cdf(e, gamma, sigma2)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    return float(max(np.max(emp_hi - F), np.max(F - emp_lo)))


def estimate_sigma2(
    eigs: np.ndarray, gamma: float, sigma2_est: float | None = None, rounds: int = 5
) -> tuple[float, float]:
    """Bulk variance from the median: sigma2 = median(bulk) / median(MP, sigma2=1).

    Iterates: estimate, drop eigenvalues above the upper edge, re-estimate (<= `rounds`).
    Returns (sigma2, upper_edge).
    """
    e = np.sort(np.asarray(eigs, dtype=np.float64))
    m_unit = mp_median(gamma, 1.0)
    bulk = e
    sigma2 = sigma2_est if sigma2_est is not None else float(np.median(bulk) / m_unit)
    edge = mp_edges(gamma, sigma2)[1]
    for _ in range(rounds):
        new_bulk = e[e <= edge]
        if new_bulk.size == 0:
            break
        new_sigma2 = float(np.median(new_bulk) / m_unit)
        new_edge = mp_edges(gamma, new_sigma2)[1]
        converged = new_bulk.size == bulk.size and abs(new_sigma2 - sigma2) <= 1e-9 * max(sigma2, 1e-30)
        bulk, sigma2, edge = new_bulk, new_sigma2, new_edge
        if converged:
            break
    return sigma2, edge


def spike_count(eigs: np.ndarray, gamma: float, sigma2_est: float | None = None) -> int:
    """Number of eigenvalues above the MP upper edge, sigma2 estimated by `estimate_sigma2`."""
    e = np.asarray(eigs, dtype=np.float64)
    _, edge = estimate_sigma2(e, gamma, sigma2_est)
    return int(np.sum(e > edge))


# ----------------------------------------------------------------------------
# Power law (Clauset, Shalizi & Newman 2009, continuous case)
# ----------------------------------------------------------------------------


def _pl_alpha(tail: np.ndarray, xmin: float) -> float:
    return 1.0 + tail.size / np.sum(np.log(tail / xmin))


def _pl_ks(tail: np.ndarray, xmin: float, alpha: float) -> float:
    t = np.sort(tail)
    n = t.size
    F = 1.0 - (t / xmin) ** (1.0 - alpha)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    return float(max(np.max(emp_hi - F), np.max(F - emp_lo)))


def powerlaw_fit(
    eigs: np.ndarray,
    xmin_search: bool = True,
    xmin: float | None = None,
    min_tail: int = 10,
    max_candidates: int = 200,
) -> dict:
    """MLE alpha = 1 + n / sum ln(x/xmin); xmin chosen by minimum KS over the data values.

    Candidate xmin values are the unique data values that leave >= `min_tail` points
    in the tail, thinned to at most `max_candidates` evenly spaced in rank.
    Returns dict(alpha, xmin, ks, n_tail).
    """
    x = np.asarray(eigs, dtype=np.float64)
    x = np.sort(x[x > 0])
    if x.size < min_tail:
        return {"alpha": np.nan, "xmin": np.nan, "ks": np.nan, "n_tail": int(x.size)}
    if not xmin_search or xmin is not None:
        xm = float(xmin if xmin is not None else x[0])
        tail = x[x >= xm]
        if tail.size < 2:
            return {"alpha": np.nan, "xmin": xm, "ks": np.nan, "n_tail": int(tail.size)}
        alpha = _pl_alpha(tail, xm)
        return {"alpha": float(alpha), "xmin": xm, "ks": _pl_ks(tail, xm, alpha), "n_tail": int(tail.size)}
    cands = np.unique(x[: x.size - min_tail + 1])
    if cands.size > max_candidates:
        cands = cands[np.linspace(0, cands.size - 1, max_candidates).astype(int)]
    best = None
    for xm in cands:
        tail = x[x >= xm]
        if tail.size < min_tail:
            continue
        alpha = _pl_alpha(tail, xm)
        if not np.isfinite(alpha):
            continue
        ks = _pl_ks(tail, xm, alpha)
        if best is None or ks < best["ks"]:
            best = {"alpha": float(alpha), "xmin": float(xm), "ks": ks, "n_tail": int(tail.size)}
    return best


# ----------------------------------------------------------------------------
# Scalar summaries
# ----------------------------------------------------------------------------


def stable_rank(W: np.ndarray, s: np.ndarray | None = None, use_gpu: bool = True) -> float:
    """||W||_F^2 / sigma_max^2."""
    if s is None:
        s = singular_values(W, use_gpu=use_gpu)
    s = np.asarray(s, dtype=np.float64)
    return float(np.sum(s**2) / s[0] ** 2)


def participation_ratio(s: np.ndarray) -> float:
    """(sum s^2)^2 / sum s^4 (effective number of singular directions)."""
    p = np.asarray(s, dtype=np.float64) ** 2
    return float(p.sum() ** 2 / np.sum(p**2))


def sv_entropy(s: np.ndarray) -> float:
    """Shannon entropy (nats) of p_i = s_i^2 / sum s^2."""
    p = np.asarray(s, dtype=np.float64) ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
