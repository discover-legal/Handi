"""Shared argparse / bookkeeping for the hypothesis scripts (analysis/hNN_*.py).

Nothing scientific lives here: just model selection, tiers, source dtypes, the
null-draw count per tier and the pre-registered verdict thresholds (hypotheses.md section 0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from handi.analysis import registry

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"

# dtype of the weight matrices in the pinned checkpoints (docs/models/<model>.md)
SRC_DTYPE = {
    "gpt2": "F32",
    "pythia-70m": "F16",
    "pythia-160m": "F16",
    "qwen3.5-0.8b": "BF16",
    "tinystories-1m": "F32",
    "tinystories-3m": "F32",
    "tinystories-8m": "F32",
    "tinystories-33m": "F32",
}

Z_THRESH = 5.0        # |z| >= 5 for a scalar statistic
KS_FACTOR = 3.0       # KS(trained, null) > 3 x median KS(null, null)
TIER_NULLS = {"A": 20, "B": 20, "C": 100}


def parser(hyp: str, default_nulls: int | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{hyp} analysis")
    p.add_argument("--models", nargs="*", default=list(registry.MODELS), help="subset of registry.MODELS")
    p.add_argument("--nulls", type=int, default=default_nulls,
                   help="null draws per statistic (default: 20, 100 for TinyStories tier)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(TABLES / f"{hyp.lower()}.csv"))
    p.add_argument("--append", action="store_true", help="append to --out instead of truncating it")
    p.add_argument("--no-figures", action="store_true")
    return p


def n_nulls(args, model: str) -> int:
    if args.nulls is not None:
        return args.nulls
    return TIER_NULLS[registry.tier(model)]


def prepare_out(args) -> Path:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.append and out.exists():
        out.unlink()
    return out


def model_seed(seed: int, model: str) -> int:
    """Deterministic per-model seed from the run seed."""
    return int(np.random.SeedSequence([seed, sum(map(ord, model))]).generate_state(1)[0])


def finite_median(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")
