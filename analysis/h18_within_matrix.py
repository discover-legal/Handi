"""H18: within-matrix structure beyond the spectrum (hypotheses.md section 7).

Units: MLP neurons (mlp.up rows in, mlp.down columns out, mlp.up.b bias), attention head dims (attn.q/k/v rows,
attn.o columns).
(a) sparsity: mean excess kurtosis and mean Gini of |entries| per unit; null N5 (spectrum-matched).
(b) coupling: Spearman over units of (||up_i||, ||down_i||), (||up_i||, up_b_i), (||q_j||, ||k_j||); null N3 pairing
    permutation (200 draws).
Per layer rows plus layer=ALL aggregates (mean over layers; null = the same aggregate on each draw).

Run:  .venv/bin/python analysis/h18_within_matrix.py [--models ...] [--nulls N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from handi.analysis import results, stats  # noqa: E402
from handi.analysis.loader import load_canonical, n_layers  # noqa: E402
from handi.authoring.generators import spectrum_matched_gpu  # noqa: E402

HYP = "H18"
UNITS = {"mlp.up": ("row", "mlp.up.w"), "mlp.down": ("col", "mlp.down.w"), "attn.q": ("row", "attn.q.w"),
         "attn.k": ("row", "attn.k.w"), "attn.v": ("row", "attn.v.w"), "attn.o": ("col", "attn.o.w")}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unit_vectors(W: np.ndarray, how: str) -> torch.Tensor:
    T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).to(DEV)
    return T if how == "row" else T.T


def sparsity(Uv: torch.Tensor) -> tuple[float, float]:
    X = Uv - Uv.mean(1, keepdim=True)
    sd = X.pow(2).mean(1).sqrt().clamp_min(1e-12)
    kurt = (X / sd[:, None]).pow(4).mean(1) - 3.0
    A = Uv.abs().sort(dim=1).values
    n = A.shape[1]
    i = torch.arange(1, n + 1, device=A.device, dtype=A.dtype)
    gini = ((2 * i - n - 1)[None, :] * A).sum(1) / (n * A.sum(1).clamp_min(1e-12))
    return float(kurt.mean()), float(gini.mean())


def main() -> None:
    args = C.parser(HYP).parse_args()
    out = C.prepare_out(args)
    with results.ResultsWriter(out) as w:
        for name in args.models:
            t0 = time.time()
            canon, cfg, fam = load_canonical(name)
            L = n_layers(cfg, fam)
            n_null = C.n_nulls(args, name)
            gen = torch.Generator().manual_seed(args.seed)
            rng = np.random.default_rng(args.seed)
            base = dict(hypothesis=HYP, model=name, family=fam, src_dtype=C.SRC_DTYPE[name], seed=args.seed)
            # (a) sparsity per unit type
            for utype, (how, suffix) in UNITS.items():
                keys = [f"layers.{l}.{suffix}" for l in range(L) if f"layers.{l}.{suffix}" in canon]
                if not keys:
                    continue
                tr = {k: sparsity(unit_vectors(canon[k], how)) for k in keys}
                nulls = {k: [] for k in keys}
                for i in range(n_null):
                    for k in keys:
                        nulls[k].append(sparsity(unit_vectors(spectrum_matched_gpu(canon[k], gen), how)))
                for j, metric in enumerate(("kurtosis", "gini")):
                    for k in keys:
                        nv = [x[j] for x in nulls[k]]
                        ns = stats.null_summary(nv)
                        w.row(layer=int(k.split(".")[1]), component=utype, metric=metric, value=tr[k][j], null_kind="N5", null_n=ns["n"],
                              null_mean=ns["mean"], null_sd=ns["sd"], z=stats.zscore(tr[k][j], nv), n_rows=canon[k].shape[0],
                              n_cols=canon[k].shape[1], **base)
                    agg = float(np.mean([tr[k][j] for k in keys]))
                    agg_null = [float(np.mean([nulls[k][i][j] for k in keys])) for i in range(n_null)]
                    ns = stats.null_summary(agg_null)
                    w.row(layer="ALL", component=utype, metric=metric, value=agg, null_kind="N5", null_n=ns["n"], null_mean=ns["mean"],
                          null_sd=ns["sd"], z=stats.zscore(agg, agg_null), **base)
                    print(f"[{name}] {utype} {metric}: {agg:+.3f} vs N5 {ns['mean']:+.3f}+-{ns['sd']:.3f} z={stats.zscore(agg, agg_null):+.1f}", flush=True)
            # (b) coupling
            pairs = {"up_norm_vs_down_norm": ("mlp.up.w", "row", "mlp.down.w", "col"),
                     "up_norm_vs_up_bias": ("mlp.up.w", "row", "mlp.up.b", "vec"),
                     "q_norm_vs_k_norm": ("attn.q.w", "row", "attn.k.w", "row")}
            for metric, (ka, ha, kb, hb) in pairs.items():
                vals, nulls_l = [], []
                for l in range(L):
                    a_key, b_key = f"layers.{l}.{ka}", f"layers.{l}.{kb}"
                    if a_key not in canon or b_key not in canon:
                        continue
                    A = np.linalg.norm(canon[a_key], axis=1 if ha == "row" else 0)
                    B = canon[b_key] if hb == "vec" else np.linalg.norm(canon[b_key], axis=1 if hb == "row" else 0)
                    if len(A) != len(B):
                        continue      # GQA: k has fewer rows than q
                    rho = float(spearmanr(A, B).correlation)
                    nl = [float(spearmanr(A, B[rng.permutation(len(B))]).correlation) for _ in range(200)]
                    ns = stats.null_summary(nl)
                    w.row(layer=l, component="mlp" if "up" in metric else "attn", metric=metric, value=rho, null_kind="N3", null_n=200,
                          null_mean=ns["mean"], null_sd=ns["sd"], z=stats.zscore(rho, nl), n_rows=len(A), **base)
                    vals.append(rho); nulls_l.append(nl)
                if vals:
                    agg = float(np.mean(vals))
                    agg_null = list(np.mean(np.array(nulls_l), axis=0))
                    ns = stats.null_summary(agg_null)
                    w.row(layer="ALL", component="mlp" if "up" in metric else "attn", metric=metric, value=agg, null_kind="N3", null_n=200,
                          null_mean=ns["mean"], null_sd=ns["sd"], z=stats.zscore(agg, agg_null), **base)
                    print(f"[{name}] {metric}: rho={agg:+.3f} z={stats.zscore(agg, agg_null):+.1f}", flush=True)
            print(f"  {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
