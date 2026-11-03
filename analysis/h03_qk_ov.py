"""H3: QK/OV circuits are low stable-rank with an identity-like component (hypotheses.md H3).

Per head of every attention layer: W_QK = W_Q W_K^T and W_OV = W_V W_O^T (d x d, rank <= hd);
stable rank, participation ratio, singular-value entropy and cos_F(W, I) = tr(W) / (||W||_F sqrt d).
Null: products of N1 Gaussian factors of the same shapes [d, hd], drawn independently for every
(draw, layer, head) slot (all metrics are scale-free, so the factor norms do not matter).
Rows per head (component "attn.h{h}.QK"), per layer ("attn.QK", mean over heads, null = per-draw
mean over heads) and per model (layer="ALL", mean over all heads, null = per-draw mean over all
heads). The ks columns hold KS(trained over-heads distribution, null draw) (median over draws)
and the median KS(null draw, null draw).

Run:  .venv/bin/python analysis/h03_qk_ov.py [--models ...] [--nulls N] [--seed S] [--out CSV]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from handi.analysis import plots, results, spectra, stats  # noqa: E402
from handi.analysis.loader import attn_heads, attn_layers, load_canonical  # noqa: E402

HYP = "H03"
CIRCUITS = ("QK", "OV")
METRICS = ("stable_rank", "participation_ratio", "sv_entropy", "cos_identity")
MAX_NULL_PAIRS = 200


def circuit_metrics(A: torch.Tensor, B: torch.Tensor) -> dict[str, np.ndarray]:
    """Metrics of W_b = A_b @ B_b^T for batched factors A, B [batch, d, hd]; each metric -> [batch]."""
    W = A @ B.transpose(1, 2)
    s = torch.linalg.svdvals(W).double().cpu().numpy()          # [batch, d]
    d = W.shape[1]
    p = s**2
    fro = np.sqrt(p.sum(axis=1))
    tr = torch.diagonal(W, dim1=1, dim2=2).sum(dim=1).double().cpu().numpy()
    pn = p / p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(pn > 0, pn * np.log(pn), 0.0), axis=1)
    return {
        "stable_rank": p.sum(axis=1) / p[:, 0],
        "participation_ratio": p.sum(axis=1) ** 2 / (p**2).sum(axis=1),
        "sv_entropy": ent,
        "cos_identity": np.where(fro > 0, tr / (fro * np.sqrt(d)), np.nan),
    }


def run_model(model: str, args, writer: results.ResultsWriter) -> None:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    layers = attn_layers(cfg, fam)
    n_null = C.n_nulls(args, model)
    seed = C.model_seed(args.seed, model)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev).manual_seed(seed)
    dtype = C.SRC_DTYPE[model]
    heads0 = attn_heads(weights, cfg, fam, layers[0])
    H = len(heads0)
    d, hd = heads0[0][0].shape
    L = len(layers)
    print(f"[{model}] {L} attention layers x {H} heads, d={d}, hd={hd}, {n_null} nulls", flush=True)

    # trained: [L, H] per metric per circuit
    tr = {c: {m: np.full((L, H), np.nan) for m in METRICS} for c in CIRCUITS}
    f = lambda arrs: torch.from_numpy(np.stack([np.ascontiguousarray(a) for a in arrs])).to(dev)
    for li, layer in enumerate(layers):
        heads = attn_heads(weights, cfg, fam, layer)
        for c, (ia, ib) in (("QK", (0, 1)), ("OV", (2, 3))):
            for m, v in circuit_metrics(f([h[ia] for h in heads]), f([h[ib] for h in heads])).items():
                tr[c][m][li] = v
    # null: [n_null, L, H]; products of iid Gaussians (circuit label irrelevant: same shapes)
    nl = {m: np.full((n_null, L, H), np.nan) for m in METRICS}
    for i in range(n_null):
        for li in range(L):
            A = torch.randn(H, d, hd, generator=gen, device=dev)
            B = torch.randn(H, d, hd, generator=gen, device=dev)
            for m, v in circuit_metrics(A, B).items():
                nl[m][i, li] = v
    print(f"  metrics done [{time.time() - t0:.0f}s]", flush=True)

    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n_null) for j in range(i + 1, n_null)]
    if len(pairs) > MAX_NULL_PAIRS:
        pairs = [pairs[k] for k in rng.choice(len(pairs), MAX_NULL_PAIRS, replace=False)]
    common = dict(hypothesis=HYP, model=model, family=fam, null_kind="N1", n_rows=d, n_cols=d,
                  src_dtype=dtype, seed=seed)
    for c in CIRCUITS:
        for m in METRICS:
            T = tr[c][m]
            N = nl[m]
            # over-heads distribution KS (all heads of all layers)
            ks_tn = C.finite_median([stats.ks_2samp(T.ravel(), N[i].ravel())[0] for i in range(n_null)])
            ks_nn = C.finite_median([stats.ks_2samp(N[i].ravel(), N[j].ravel())[0] for i, j in pairs])
            for li, layer in enumerate(layers):
                for h in range(H):
                    nv = N[:, li, h]
                    sm = stats.null_summary(nv)
                    writer.row(**common, layer=layer, component=f"attn.h{h}.{c}", metric=m, value=T[li, h],
                               null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"], z=stats.zscore(T[li, h], nv))
                nv = N[:, li, :].mean(axis=1)
                v = float(T[li].mean())
                sm = stats.null_summary(nv)
                writer.row(**common, layer=layer, component=f"attn.{c}", metric=m, value=v,
                           null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"], z=stats.zscore(v, nv))
            nv = N.reshape(n_null, -1).mean(axis=1)
            v = float(T.mean())
            sm = stats.null_summary(nv)
            writer.row(**common, layer="ALL", component=f"attn.{c}", metric=m, value=v,
                       null_n=sm["n"], null_mean=sm["mean"], null_sd=sm["sd"], z=stats.zscore(v, nv),
                       ks=ks_tn, ks_null_median=ks_nn)
            print(f"  {c} {m:20s} trained mean={v:.4f} null={sm['mean']:.4f}+-{sm['sd']:.4f} "
                  f"z={stats.zscore(v, nv):.1f} ks={ks_tn:.3f} ks_nn={ks_nn:.3f}", flush=True)
    results.save_raw(HYP.lower(), model, "head_metrics", layers=np.array(layers),
                     **{f"trained_{c}_{m}": tr[c][m] for c in CIRCUITS for m in METRICS},
                     **{f"null_{m}": nl[m] for m in METRICS})
    if not args.no_figures:
        figure(model, tr, nl, n_null)
    print(f"[{model}] done in {time.time() - t0:.0f}s", flush=True)


def figure(model: str, tr: dict, nl: dict, n_null: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    out_dir = C.FIGURES / "h03"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [("cos_identity", "QK", "cos_F(W_QK, I)"), ("cos_identity", "OV", "cos_F(W_OV, I)"),
              ("stable_rank", "QK", "stable rank of W_QK")]
    for ax, (m, c, label) in zip(axes, panels):
        t = tr[c][m].ravel()
        n = nl[m].ravel()
        lo, hi = np.nanmin(np.concatenate([t, n])), np.nanmax(np.concatenate([t, n]))
        bins = np.linspace(lo, hi, 40)
        ax.hist(n, bins=bins, density=True, color=plots.INK_SECONDARY, alpha=0.25, label=f"N1 product null ({n_null} draws)")
        ax.hist(t, bins=bins, density=True, color=pal[0], alpha=0.85, label="trained heads",
                rwidth=0.9)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"{label}: {t.size} heads")
        ax.legend()
    fig.suptitle(f"{model}: head-level QK/OV statistics vs random products", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / f"{model}_qk_ov.png")
    plt.close(fig)


def main() -> None:
    args = C.parser(HYP).parse_args()
    out = C.prepare_out(args)
    with results.ResultsWriter(out) as writer:
        for model in args.models:
            run_model(model, args, writer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
