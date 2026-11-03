"""Small statistics helpers shared by the hypothesis scripts."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as _sps


def null_summary(null_values) -> dict:
    v = np.asarray(null_values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {
        "mean": float(v.mean()) if v.size else float("nan"),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "n": int(v.size),
    }


def zscore(value: float, null_values) -> float:
    """(value - mean(null)) / sd(null, ddof=1); inf if sd == 0 and value != mean."""
    s = null_summary(null_values)
    if s["n"] < 2 or not np.isfinite(s["sd"]):
        return float("nan")
    if s["sd"] == 0:
        return 0.0 if value == s["mean"] else float(np.sign(value - s["mean"]) * np.inf)
    return float((value - s["mean"]) / s["sd"])


def ks_2samp(a, b) -> tuple[float, float]:
    """Two-sample KS: (statistic, p-value) via scipy."""
    r = _sps.ks_2samp(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return float(r.statistic), float(r.pvalue)


def bootstrap_ci(
    values,
    stat: Callable = np.mean,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap: (stat(values), lo, hi) at level 1 - alpha."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n, v.size))
    boots = np.array([stat(v[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(v)), float(lo), float(hi)
