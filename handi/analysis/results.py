"""Long-format results CSV writer and raw array store."""
from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "results" / "raw"

COLUMNS = [
    "hypothesis", "model", "family", "layer", "component", "metric", "value",
    "null_kind", "null_n", "null_mean", "null_sd", "z", "ks", "ks_null_median",
    "n_rows", "n_cols", "src_dtype", "seed", "git_sha", "timestamp",
]


def git_sha(short: bool = False) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""
    return sha[:12] if short else sha


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResultsWriter:
    """Appends rows to a long-format CSV with exactly `COLUMNS`; writes the header once."""

    def __init__(self, path_csv: str | os.PathLike):
        self.path = Path(path_csv)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sha = git_sha()
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=COLUMNS, extrasaction="raise")
        if needs_header:
            self._w.writeheader()
            self._fh.flush()

    def row(self, **fields) -> dict:
        unknown = set(fields) - set(COLUMNS)
        if unknown:
            raise KeyError(f"unknown result columns {sorted(unknown)}")
        r = {c: "" for c in COLUMNS}
        r.update({k: ("" if v is None else v) for k, v in fields.items()})
        if r["git_sha"] == "":
            r["git_sha"] = self._sha
        if r["timestamp"] == "":
            r["timestamp"] = utc_now()
        self._w.writerow(r)
        self._fh.flush()
        return r

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_raw(hyp: str, model: str, component: str, root: str | os.PathLike | None = None, **arrays) -> Path:
    """np.savez_compressed to results/raw/<hyp>/<model>/<component>.npz (component '/' -> '__')."""
    base = Path(root) if root is not None else RAW_DIR
    out = base / hyp / model / (component.replace("/", "__") + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in arrays.items()})
    return out
