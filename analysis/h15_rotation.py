"""H15: cross-layer rotation rule (hypotheses.md section 7, added 2026-11-04).

Per layer l, P_l = top-k right singular vectors of the layer's reader matrices stacked along their
output axis ([sum rows, d]); writers: top-k left singular vectors of the writers stacked along their
input axis ([d, sum cols]). Reader / writer types as in H7.

Geodesic on the Grassmannian from P_a to P_b (Edelman, Arias, Smith 1998):
    Q = P_b (P_a^T P_b)^{-1} - P_a = U tan(Theta) V^T,   Gamma(t) = P_a V cos(Theta t) + U sin(Theta t)
Gamma(1) spans P_b; Gamma(2) is "the same turn again". Overlap ov(A, B) = ||A^T B||_F^2 / k.

Statistics (per model, per side, per k in {8, 32} with k < d):
  s1_extrapolate   mean_l ov(Gamma_{l->l+1}(2), P_{l+2})          null: random turn (same Theta, Haar U in the
                                                                  complement of P_l, Haar V), 200 draws
  s1_stay          mean_l ov(P_{l+1}, P_{l+2})                     baseline, no rule (what H7 / A3 assume)
  s1_multi         mean_{t>=2} ov(Gamma_{0->1}(t), P_t)            null: random turn
  s2_turn_overlap  mean_l ov(U_l, U_{l+1})                         null: Haar k-subspaces (mean k/d), 200 draws
  lag_overlap      mean_l ov(P_l, P_{l+j}) per lag j (layer=j)     descriptive
  matrix_lag_overlap / matrix_allpair_overlap: the same per matrix (H7's items, lag-grouped), used by A6.

Run:  .venv/bin/python analysis/h15_rotation.py [--models ...] [--nulls 200] [--out results/tables/h15.csv]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from handi.analysis import registry, results, stats  # noqa: E402
from handi.analysis.loader import load_canonical, n_layers  # noqa: E402

HYP = "H15"
READER_TYPES = ["attn.q.w", "attn.k.w", "attn.v.w", "mlp.up.w", "mlp.gate.w", "linear.in_qkv.w", "linear.in_z.w"]
WRITER_TYPES = ["attn.o.w", "mlp.down.w", "linear.out.w"]
KS = (8, 32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def haar(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    Z = torch.randn(m, k, generator=gen).to(DEV)
    Q, R = torch.linalg.qr(Z)
    return Q * torch.where(torch.diagonal(R) >= 0, 1.0, -1.0)[None, :]


def haar_complement(P: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    Z = torch.randn(P.shape[0], k, generator=gen).to(DEV)
    Z = Z - P @ (P.T @ Z)
    Q, R = torch.linalg.qr(Z)
    return Q * torch.where(torch.diagonal(R) >= 0, 1.0, -1.0)[None, :]


def ov(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def top_basis(M: np.ndarray, kmax: int, side: str) -> torch.Tensor:
    T = torch.from_numpy(np.ascontiguousarray(M, dtype=np.float32)).to(DEV)
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    return (Vh[:kmax].T if side == "right" else U[:, :kmax]).contiguous()


def layer_stacks(w: dict, L: int):
    readers, writers = [], []
    for l in range(L):
        r = [w[f"layers.{l}.{t}"] for t in READER_TYPES if f"layers.{l}.{t}" in w]
        s = [w[f"layers.{l}.{t}"] for t in WRITER_TYPES if f"layers.{l}.{t}" in w]
        readers.append(np.concatenate(r, axis=0))
        writers.append(np.concatenate(s, axis=1))
    return readers, writers


def geodesic(Pa: torch.Tensor, Pb: torch.Tensor):
    M = Pa.T @ Pb
    Q = Pb @ torch.linalg.inv(M.double()).float() - Pa
    U, tan_t, Vh = torch.linalg.svd(Q, full_matrices=False)
    return U, torch.atan(tan_t), Vh.T


def gamma(Pa: torch.Tensor, U: torch.Tensor, theta: torch.Tensor, V: torch.Tensor, t: float) -> torch.Tensor:
    return (Pa @ V) * torch.cos(theta * t)[None, :] + U * torch.sin(theta * t)[None, :]


def run_side(P: list[torch.Tensor], k: int, n_null: int, gen: torch.Generator) -> dict:
    """P: per-layer [d, k] bases. Returns the statistics and their null draws."""
    L = len(P)
    d = P[0].shape[0]
    geo = [geodesic(P[l], P[l + 1]) for l in range(L - 1)]
    # S1: two-step extrapolation
    s1 = [ov(gamma(P[l], *geo[l], 2.0), P[l + 2]) for l in range(L - 2)]
    stay = [ov(P[l + 1], P[l + 2]) for l in range(L - 2)]
    s1_null = np.zeros(n_null)
    for i in range(n_null):
        vals = []
        for l in range(L - 2):
            U, theta, V = geo[l]
            Ur = haar_complement(P[l], k, gen)
            Vr = haar(k, k, gen)
            vals.append(ov(gamma(P[l], Ur, theta, Vr, 2.0), P[l + 2]))
        s1_null[i] = np.mean(vals)
    # S1 multi: one rule from layers 0, 1
    U0, th0, V0 = geo[0]
    multi = [ov(gamma(P[0], U0, th0, V0, float(t)), P[t]) for t in range(2, L)]
    multi_null = np.zeros(n_null)
    for i in range(n_null):
        Ur = haar_complement(P[0], k, gen)
        Vr = haar(k, k, gen)
        multi_null[i] = np.mean([ov(gamma(P[0], Ur, th0, Vr, float(t)), P[t]) for t in range(2, L)])
    # S2: turn-direction overlap
    s2 = [ov(geo[l][0], geo[l + 1][0]) for l in range(L - 2)]
    s2_null = np.array([np.mean([ov(haar(d, k, gen), haar(d, k, gen)) for _ in range(L - 2)]) for _ in range(n_null)])
    # lag overlaps
    lag = {j: float(np.mean([ov(P[l], P[l + j]) for l in range(L - j)])) for j in range(1, L)}
    angles = {l: geo[l][1].cpu().numpy() for l in range(L - 1)}
    return dict(s1=float(np.mean(s1)), stay=float(np.mean(stay)), s1_null=s1_null, multi=float(np.mean(multi)),
                multi_null=multi_null, s2=float(np.mean(s2)), s2_null=s2_null, lag=lag, angles=angles, d=d, L=L)


def matrix_lag_overlaps(w: dict, L: int, types: list[str], side: str, k: int) -> tuple[dict, float]:
    items = [(l, top_basis(w[f"layers.{l}.{t}"], k, side)) for l in range(L) for t in types if f"layers.{l}.{t}" in w]
    V = torch.stack([b for _, b in items])
    G = torch.einsum("adk,bdl->abkl", V, V).pow(2).sum((-1, -2)) / k
    lay = np.array([l for l, _ in items])
    lag_vals: dict[int, list] = {}
    allp = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            j = abs(int(lay[a] - lay[b]))
            if j == 0:
                continue
            lag_vals.setdefault(j, []).append(float(G[a, b]))
            allp.append(float(G[a, b]))
    return {j: float(np.mean(v)) for j, v in lag_vals.items()}, float(np.mean(allp))


def main() -> None:
    args = C.parser(HYP, default_nulls=200).parse_args()
    out = C.prepare_out(args)
    figdir = C.FIGURES / "h15"
    figdir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    with results.ResultsWriter(out) as w:
        for name in args.models:
            t0 = time.time()
            canon, cfg, fam = load_canonical(name)
            L = n_layers(cfg, fam)
            readers, writers = layer_stacks(canon, L)
            d = readers[0].shape[1]
            base = dict(hypothesis=HYP, model=name, family=fam, src_dtype=C.SRC_DTYPE[name], seed=args.seed)
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            for side, mats, sv_side, types in (("readers", readers, "right", READER_TYPES), ("writers", writers, "left", WRITER_TYPES)):
                for k in KS:
                    if k >= d:
                        continue
                    gen = torch.Generator().manual_seed(args.seed)
                    P = [top_basis(M, k, sv_side) for M in mats]
                    r = run_side(P, k, args.nulls, gen)
                    comp = f"{side}:k{k}"
                    common = dict(layer="ALL", component=comp, n_rows=L, n_cols=d, **base)
                    for metric, val, null, kind in (("s1_extrapolate", r["s1"], r["s1_null"], "random_turn"),
                                                    ("s1_multi", r["multi"], r["multi_null"], "random_turn"),
                                                    ("s2_turn_overlap", r["s2"], r["s2_null"], "Haar")):
                        ns = stats.null_summary(null)
                        w.row(metric=metric, value=val, null_kind=kind, null_n=ns["n"], null_mean=ns["mean"], null_sd=ns["sd"],
                              z=stats.zscore(val, null), **common)
                    w.row(metric="s1_stay", value=r["stay"], **common)
                    w.row(metric="s1_minus_stay", value=r["s1"] - r["stay"], **common)
                    w.row(metric="chance_overlap", value=k / d, **common)
                    for j, v in r["lag"].items():
                        w.row(metric="lag_overlap", value=v, layer=j, component=comp, n_rows=L, n_cols=d, **base)
                    for l, th in r["angles"].items():
                        w.row(metric="principal_angle_mean_deg", value=float(np.degrees(th).mean()), layer=l, component=comp,
                              n_rows=L, n_cols=d, **base)
                    mlag, mall = matrix_lag_overlaps(canon, L, types, sv_side, k)
                    for j, v in mlag.items():
                        w.row(metric="matrix_lag_overlap", value=v, layer=j, component=comp, n_rows=L, n_cols=d, **base)
                    w.row(metric="matrix_allpair_overlap", value=mall, **common)
                    z1 = stats.zscore(r["s1"], r["s1_null"])
                    print(f"[{name}] {comp}: S1={r['s1']:.4f} stay={r['stay']:.4f} null={r['s1_null'].mean():.4f} z={z1:.1f} | "
                          f"multi={r['multi']:.4f} (null {r['multi_null'].mean():.4f}) | S2={r['s2']:.4f} (chance {k / d:.4f}, "
                          f"z={stats.zscore(r['s2'], r['s2_null']):.1f}) | lag1={r['lag'][1]:.4f} matrix lag1={mlag.get(1, float('nan')):.4f} "
                          f"all={mall:.4f}", flush=True)
                    if k == 32 or (k == 8 and 32 >= d / 2):
                        ax = axes[0 if side == "readers" else 1]
                        js = sorted(r["lag"])
                        ax.plot(js, [r["lag"][j] for j in js], "o-", label=f"trained lag overlap (k={k})")
                        ax.axhline(k / d, ls=":", color="gray", label="chance")
                        ax.axhline(r["s1"], ls="--", color="C3", label=f"S1 extrapolate {r['s1']:.3f}")
                        ax.axhline(r["s1_null"].mean(), ls="--", color="C7", label=f"random-turn null {r['s1_null'].mean():.3f}")
                        ax.set_title(f"{name} {side}")
                        ax.set_xlabel("lag (layers)")
                        ax.set_ylabel("Grassmann overlap")
                        ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(figdir / f"{name}.png", dpi=120)
            plt.close(fig)
            print(f"  {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
