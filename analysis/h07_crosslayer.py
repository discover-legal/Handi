"""H7: cross-layer alignment of reader / writer subspaces (docs/hypotheses.md H7).

Readers (matrices [rows, d] that read the residual stream): attn.q.w, attn.k.w, attn.v.w, mlp.up.w,
plus mlp.gate.w and linear.in_qkv.w / linear.in_z.w for Qwen. Writers ([d, cols]): attn.o.w,
mlp.down.w, linear.out.w. Reader subspace = top-k right singular vectors (in R^d); writer subspace =
top-k left singular vectors. Grassmann overlap tr(V_a^T V_b V_b^T V_a) / k for every pair of items
within readers, within writers, and writer(l) vs reader(l+1).

Nulls: Haar random k-subspaces of R^d (closed-form mean k/d; sd from simulated draws of the whole
item set, 200 draws) and N5 spectrum-matched (10 draws, confirmatory: fewer than the 20-draw rule).
Re-run with the top-5 outlier dims projected out; direct cos_F(W_l, W_{l+1}) per matrix type vs N1.

Choices the pre-registration did not fix (listed in the report):
  * Outlier dims: column norms of every reader matrix, each matrix's norms divided by their own mean,
    averaged over all reader matrices; the top-5 dims of that profile are projected out.
  * After projection the subspaces live in a (d-5)-dimensional space, so the Haar null is simulated
    in d-5 dims (mean k/(d-5)).
  * Aggregate z: the aggregate (mean overlap over a set of pairs) is compared with the same aggregate
    computed on 200 Haar draws of the whole item set (so the null sd of the mean is simulated, not
    assumed sd/sqrt(n)).
  * k = 64 is skipped where k >= d (TinyStories-1M, d = 64: every 64-subspace is R^d).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: the Tk backend crashes at interpreter exit
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis import nulls, plots, registry  # noqa: E402
from handi.analysis.loader import load_canonical, n_layers  # noqa: E402
from handi.analysis.results import ResultsWriter, save_raw  # noqa: E402
from handi.analysis.stats import null_summary, zscore  # noqa: E402

HYP = "H07"
SRC_DTYPE = {"gpt2": "float32", "gpt_neox": "float16", "qwen3_5": "bfloat16", "gpt_neo": "float32"}
READER_TYPES = ["attn.q.w", "attn.k.w", "attn.v.w", "mlp.up.w", "mlp.gate.w", "linear.in_qkv.w", "linear.in_z.w"]
WRITER_TYPES = ["attn.o.w", "mlp.down.w", "linear.out.w"]
KS = (8, 32, 64)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect(w: dict, L: int):
    readers, writers = [], []   # (layer, type, matrix)
    for l in range(L):
        for t in READER_TYPES:
            if f"layers.{l}.{t}" in w:
                readers.append((l, t, w[f"layers.{l}.{t}"]))
        for t in WRITER_TYPES:
            if f"layers.{l}.{t}" in w:
                writers.append((l, t, w[f"layers.{l}.{t}"]))
    return readers, writers


def svd_bases(M: np.ndarray, kmax: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One SVD on the GPU: (U[:, :kmax], V[:, :kmax], all singular values), bases as [d, kmax]."""
    T = torch.from_numpy(np.ascontiguousarray(M, dtype=np.float32)).to(DEV)
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    return U[:, :kmax].contiguous(), Vh[:kmax].T.contiguous(), S


def haar_gpu(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Haar m x k with orthonormal columns, same construction as nulls.haar_stiefel (QR + sign fix), on the GPU."""
    Z = torch.randn(m, k, generator=gen).to(DEV)
    Q, R = torch.linalg.qr(Z)
    sgn = torch.where(torch.diagonal(R) >= 0, 1.0, -1.0)
    return Q * sgn[None, :]


def n5_gpu(shape: tuple[int, int], sigma: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """N5 spectrum-matched random U diag(sigma) V^T (nulls.n5_spectrum_matched re-done on the GPU for speed)."""
    m, n = shape
    k = min(m, n)
    U = haar_gpu(m, k, gen); V = haar_gpu(n, k, gen)
    return (U * sigma[:k][None, :]) @ V.T


def top_subspace_t(T: torch.Tensor, kmax: int, side: str) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    return (Vh[:kmax].T if side == "right" else U[:, :kmax]).contiguous()


def overlaps(Bs: list[torch.Tensor], k: int) -> torch.Tensor:
    """[n, n] Grassmann overlaps tr(V_a^T V_b V_b^T V_a) / k for a stack of [d, k] bases."""
    V = torch.stack(Bs)                         # [n, d, k]
    G = torch.einsum("adk,bdl->abkl", V, V)      # [n, n, k, k]
    return (G ** 2).sum((-1, -2)) / k


def haar_stack(n: int, d: int, k: int, gen: torch.Generator) -> list[torch.Tensor]:
    Z = torch.randn(n, d, k, generator=gen).to(DEV)
    Q, R = torch.linalg.qr(Z)
    return list(Q)


def pair_sets(readers, writers):
    """Index pairs for the three comparison groups (in the combined item list readers + writers)."""
    nr = len(readers)
    rr = [(a, b) for a in range(nr) for b in range(a + 1, nr)]
    ww = [(nr + a, nr + b) for a in range(len(writers)) for b in range(a + 1, len(writers))]
    wr = [(nr + a, b) for a, (la, _, _) in enumerate(writers) for b, (lb, _, _) in enumerate(readers) if lb == la + 1]
    return rr, ww, wr


def aggregate_keys(items, pairs):
    """Group pairs by (type_a, type_b) for cross-layer pairs, plus the whole group."""
    groups: dict[str, list] = {"all_crosslayer": []}
    for a, b in pairs:
        la, ta = items[a][0], items[a][1]
        lb, tb = items[b][0], items[b][1]
        if la == lb:
            groups.setdefault("same_layer", []).append((a, b))
            continue
        groups["all_crosslayer"].append((a, b))
        key = f"{ta}|{tb}" if ta <= tb else f"{tb}|{ta}"
        groups.setdefault(key, []).append((a, b))
    return groups


def run_model(name: str, args, W: ResultsWriter, figdir: Path):
    pal = plots.style()
    import matplotlib.pyplot as plt

    w, cfg, fam = load_canonical(name)
    L = n_layers(cfg, fam)
    d = w["embed.tok"].shape[1]
    src = SRC_DTYPE[fam]
    common = dict(hypothesis=HYP, model=name, family=fam, src_dtype=src, seed=args.seed)
    readers, writers = collect(w, L)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # outlier dims (top-5 by mean normalised reader column norm)
    prof = np.zeros(d)
    for _, _, M in readers:
        cn = np.linalg.norm(M, axis=0); prof += cn / cn.mean()
    prof /= len(readers)
    outl = np.argsort(-prof)[:5]
    print(f"  outlier dims (reader column norm): {outl.tolist()} (profile {np.round(prof[outl], 2).tolist()})")
    W.row(**common, layer=-1, component="readers", metric="outlier_dims", value=";".join(map(str, outl.tolist())), null_kind="")
    keep = np.ones(d, bool); keep[outl] = False

    for variant in ("full", "no_outlier"):
        if variant == "full":
            R_items = [(l, t, M) for l, t, M in readers]
            Wr_items = [(l, t, M) for l, t, M in writers]
            d_eff = d
        else:
            R_items = [(l, t, M[:, keep]) for l, t, M in readers]
            Wr_items = [(l, t, M[keep, :]) for l, t, M in writers]
            d_eff = d - 5
        items = R_items + Wr_items
        rr, ww, wr = pair_sets(R_items, Wr_items)
        n_items = len(items)
        kmax = max(k for k in KS if k < d_eff)
        t0 = time.time()
        bases, sigmas = [], []
        for _, _, M in R_items:
            U, V, S = svd_bases(M, kmax); bases.append(V); sigmas.append(S)
        for _, _, M in Wr_items:
            U, V, S = svd_bases(M, kmax); bases.append(U); sigmas.append(S)
        n5_bases = []     # [draw][item] -> [d, kmax]
        for _ in range(args.n5):
            draw = []
            for i, (_, _, M) in enumerate(R_items):
                draw.append(top_subspace_t(n5_gpu(M.shape, sigmas[i], gen), kmax, "right"))
            for j, (_, _, M) in enumerate(Wr_items):
                draw.append(top_subspace_t(n5_gpu(M.shape, sigmas[len(R_items) + j], gen), kmax, "left"))
            n5_bases.append(draw)
        print(f"  {variant}: SVDs for {n_items} items + {args.n5} N5 draws in {time.time() - t0:.0f}s")
        for k in KS:
            if k >= d_eff:
                print(f"  skip k={k} (>= d_eff={d_eff})")
                continue
            t0 = time.time()
            Bs = [B[:, :k] for B in bases]
            O = overlaps(Bs, k).cpu().numpy()
            # Haar null: 200 draws of the whole item set
            haar = np.stack([overlaps(haar_stack(n_items, d_eff, k, gen), k).cpu().numpy() for _ in range(args.haar)])
            n5 = [overlaps([B[:, :k] for B in draw], k).cpu().numpy() for draw in n5_bases]
            n5 = np.stack(n5)
            for gname, pairs in (("readers", rr), ("writers", ww), ("writer_to_next_reader", wr)):
                groups = aggregate_keys(items, pairs)
                for key, pp in groups.items():
                    if not pp:
                        continue
                    ia = np.array([p[0] for p in pp]); ib = np.array([p[1] for p in pp])
                    val = float(O[ia, ib].mean())
                    hv = haar[:, ia, ib].mean(1); n5v = n5[:, ia, ib].mean(1)
                    s = null_summary(hv); s5 = null_summary(n5v)
                    comp = f"{gname}:{key}:{variant}"
                    W.row(**common, layer=-1, component=comp, metric=f"grassmann_mean_k{k}", value=val, null_kind="Haar",
                          null_n=s["n"], null_mean=k / d_eff, null_sd=s["sd"], z=(val - k / d_eff) / s["sd"], n_rows=len(pp), n_cols=d_eff)
                    W.row(**common, layer=-1, component=comp, metric=f"grassmann_mean_k{k}", value=val, null_kind="N5",
                          null_n=s5["n"], null_mean=s5["mean"], null_sd=s5["sd"], z=zscore(val, n5v), n_rows=len(pp), n_cols=d_eff)
                # per-pair rows (cross-layer only, z against the single-pair Haar sd)
                pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])
                sd1 = float(haar[:, pa, pb].std())
                if not args.no_per_pair:
                    for a, b in pairs:
                        la, ta, _ = items[a]; lb, tb, _ = items[b]
                        W.row(**common, layer=la, component=f"{ta}@L{la}|{tb}@L{lb}:{variant}", metric=f"grassmann_k{k}",
                              value=float(O[a, b]), null_kind="Haar", null_n=args.haar, null_mean=k / d_eff, null_sd=sd1,
                              z=(O[a, b] - k / d_eff) / sd1, n_rows=1, n_cols=d_eff)
            save_raw(HYP.lower(), name, f"overlap_k{k}_{variant}", O=O, haar_mean=haar.mean(0), n5_mean=n5.mean(0),
                     layers=np.array([it[0] for it in items]), types=np.array([it[1] for it in items]))
            print(f"  {variant} k={k}: {n_items} items, {time.time() - t0:.0f}s; readers cross-layer mean "
                  f"{O[[p[0] for p in rr], [p[1] for p in rr]].mean():.4f} vs k/d {k / d_eff:.4f}; N5 {n5[:, [p[0] for p in rr], [p[1] for p in rr]].mean():.4f}")
            # figures: L x L heatmaps per type, k = 32, trained vs N5
            if k == 32 and variant == "full":
                types = sorted({t for _, t, _ in items}, key=lambda t: (READER_TYPES + WRITER_TYPES).index(t))
                fig, axes = plt.subplots(2, len(types), figsize=(2.6 * len(types) + 1, 5.4), squeeze=False)
                vmax = 0
                mats = {}
                for t in types:
                    idx = [i for i, it in enumerate(items) if it[1] == t]
                    mats[t] = (O[np.ix_(idx, idx)], n5.mean(0)[np.ix_(idx, idx)], [items[i][0] for i in idx])
                    off = ~np.eye(len(idx), dtype=bool)
                    vmax = max(vmax, mats[t][0][off].max() if off.any() else 0)
                for j, t in enumerate(types):
                    Ot, N5t, ls = mats[t]
                    for r, (M, lab) in enumerate(((Ot, "trained"), (N5t, "N5 mean"))):
                        ax = axes[r, j]
                        M = M.copy(); np.fill_diagonal(M, np.nan)
                        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
                        ax.grid(False)
                        ax.set_title(f"{t} ({lab})", fontsize=9)
                        ax.set_xticks(range(0, len(ls), max(1, len(ls) // 6))); ax.set_xticklabels([ls[i] for i in range(0, len(ls), max(1, len(ls) // 6))])
                        ax.set_yticks(range(0, len(ls), max(1, len(ls) // 6))); ax.set_yticklabels([ls[i] for i in range(0, len(ls), max(1, len(ls) // 6))])
                        if j == 0:
                            ax.set_ylabel("layer")
                fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=f"Grassmann overlap, k=32 (k/d = {k / d_eff:.3f})")
                fig.suptitle(f"{name}: layer x layer subspace overlap per matrix type, trained (top) vs N5 (bottom)", x=0.02, ha="left", fontsize=10)
                fig.savefig(figdir / f"h07_{name}_overlap_k32.png"); plt.close(fig)

    # direct cos_F between consecutive layers per matrix type, vs N1 (20 draws)
    for t in READER_TYPES + WRITER_TYPES:
        vals = []
        for l in range(L - 1):
            a, b = f"layers.{l}.{t}", f"layers.{l + 1}.{t}"
            if a in w and b in w and w[a].shape == w[b].shape:
                A, B = w[a], w[b]
                val = float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B)))
                n1 = []
                for _ in range(20):
                    X, Y = nulls.n1_gaussian(A, rng), nulls.n1_gaussian(B, rng)
                    n1.append(float(np.sum(X * Y) / (np.linalg.norm(X) * np.linalg.norm(Y))))
                s = null_summary(n1)
                W.row(**common, layer=l, component=f"{t}|L{l + 1}", metric="direct_cosF_consecutive", value=val, null_kind="N1",
                      null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(val, n1), n_rows=A.shape[0], n_cols=A.shape[1])
                vals.append(val)
        if vals:
            W.row(**common, layer=-1, component=t, metric="direct_cosF_consecutive_mean", value=float(np.mean(vals)), null_kind="N1",
                  null_mean=0.0, n_rows=len(vals))
            print(f"  direct cos_F {t}: mean {np.mean(vals):+.4f} max {np.max(np.abs(vals)):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(registry.MODELS))
    ap.add_argument("--nulls", type=int, default=10, help="N5 draws (spectrum-matched)")
    ap.add_argument("--haar", type=int, default=200, help="Haar draws for the null sd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/tables/h07.csv"))
    ap.add_argument("--no-per-pair", action="store_true", help="skip the one-row-per-pair output")
    args = ap.parse_args()
    args.n5 = args.nulls
    figdir = ROOT / "results/figures/h07"; figdir.mkdir(parents=True, exist_ok=True)
    with ResultsWriter(args.out) as W:
        for name in args.models:
            t0 = time.time(); print(f"== {name}")
            run_model(name, args, W, figdir)
            print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
