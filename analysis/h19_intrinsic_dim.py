"""H19: intrinsic dimension of weight point clouds (hypotheses.md section 7).

Two-NN estimator (Facco et al. 2017), MLE form d = (N - 1) / sum_i log(r2_i / r1_i), on 20k sampled points of:
token embeddings and unembedding rows (null N6), MLP up rows and down columns per layer (null N5).
Linear effective rank exp(H(p)) of the centred cloud alongside.

Run:  .venv/bin/python analysis/h19_intrinsic_dim.py [--models ...] [--nulls N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from handi.analysis import results, stats  # noqa: E402
from handi.analysis.loader import load_canonical, n_layers  # noqa: E402
from handi.analysis.nulls import n6_cov_matched  # noqa: E402
from handi.authoring.generators import spectrum_matched_gpu  # noqa: E402

HYP = "H19"
N_POINTS = 20_000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def two_nn(X: np.ndarray, rng: np.random.Generator) -> float:
    if len(X) > N_POINTS:
        X = X[rng.choice(len(X), N_POINTS, replace=False)]
    T = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(DEV)
    sq = T.pow(2).sum(1)
    r1, r2 = [], []
    for s in range(0, len(T), 2048):
        D = sq[s:s + 2048, None] + sq[None, :] - 2 * T[s:s + 2048] @ T.T
        D[torch.arange(D.shape[0]), torch.arange(s, s + D.shape[0])] = float("inf")
        d2 = D.clamp_min(0).topk(2, dim=1, largest=False).values
        r1.append(d2[:, 0]); r2.append(d2[:, 1])
    r1, r2 = torch.cat(r1).sqrt(), torch.cat(r2).sqrt()
    ok = r1 > 0
    mu = (r2[ok] / r1[ok]).double()
    return float((ok.sum() - 1) / torch.log(mu).sum())


def effective_rank(X: np.ndarray) -> float:
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> None:
    args = C.parser(HYP).parse_args()
    out = C.prepare_out(args)
    with results.ResultsWriter(out) as w:
        for name in args.models:
            t0 = time.time()
            canon, cfg, fam = load_canonical(name)
            L = n_layers(cfg, fam)
            n_null = C.n_nulls(args, name)
            rng = np.random.default_rng(args.seed)
            gen = torch.Generator().manual_seed(args.seed)
            base = dict(hypothesis=HYP, model=name, family=fam, src_dtype=C.SRC_DTYPE[name], seed=args.seed)
            clouds = [("embed.tok", canon["embed.tok"], "N6")]
            if "unembed.w" in canon:
                clouds.append(("unembed.w", canon["unembed.w"], "N6"))
            for l in range(L):
                clouds.append((f"mlp.up:{l}", canon[f"layers.{l}.mlp.up.w"], "N5"))
                clouds.append((f"mlp.down:{l}", canon[f"layers.{l}.mlp.down.w"].T, "N5"))
            agg = {}
            for comp, X, nk in clouds:
                sub = X[rng.choice(len(X), N_POINTS, replace=False)] if len(X) > N_POINTS else X
                idim = two_nn(sub, rng)
                er = effective_rank(sub)
                nulls = []
                for _ in range(n_null):
                    if nk == "N6":
                        Xn = n6_cov_matched(sub, rng)
                    else:
                        Xn = spectrum_matched_gpu(X, gen)
                        Xn = Xn.T if comp.startswith("mlp.down") else Xn
                    nulls.append(two_nn(Xn, rng))
                ns = stats.null_summary(nulls)
                layer = comp.split(":")[1] if ":" in comp else "ALL"
                ctype = comp.split(":")[0]
                w.row(layer=layer, component=ctype, metric="intrinsic_dim", value=idim, null_kind=nk, null_n=ns["n"], null_mean=ns["mean"],
                      null_sd=ns["sd"], z=stats.zscore(idim, nulls), n_rows=X.shape[0], n_cols=X.shape[1], **base)
                w.row(layer=layer, component=ctype, metric="effective_rank", value=er, n_rows=X.shape[0], n_cols=X.shape[1], **base)
                agg.setdefault(ctype, []).append((idim, nulls, er))
                if layer == "ALL":
                    print(f"[{name}] {comp}: ID={idim:.1f} vs {nk} {ns['mean']:.1f}+-{ns['sd']:.2f} z={stats.zscore(idim, nulls):+.1f} | eff. rank {er:.0f}", flush=True)
            for ctype in ("mlp.up", "mlp.down"):
                vals = agg[ctype]
                m = float(np.mean([v[0] for v in vals]))
                mn = list(np.mean(np.array([v[1] for v in vals]), axis=0))
                ns = stats.null_summary(mn)
                w.row(layer="ALL", component=ctype, metric="intrinsic_dim", value=m, null_kind="N5", null_n=ns["n"], null_mean=ns["mean"],
                      null_sd=ns["sd"], z=stats.zscore(m, mn), **base)
                w.row(layer="ALL", component=ctype, metric="effective_rank", value=float(np.mean([v[2] for v in vals])), **base)
                print(f"[{name}] {ctype} (mean over layers): ID={m:.1f} vs N5 {ns['mean']:.1f} z={stats.zscore(m, mn):+.1f}", flush=True)
            print(f"  {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
