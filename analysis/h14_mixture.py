"""H14: spectral mixture decomposition (hypotheses.md section 7, added 2026-11-04).

Every 2-D weight matrix with min(n, m) >= 32 gives a profile
    p(t) = -log10(sigma(t n) / sigma_1),  t on a 128-point grid in (0, 1]   (non-negative, p(0) = 0)
read from the H1 raw store (results/raw/h01/<model>/<matrix>.npz: s_trained, shape).

Leave-one-model-out, for K in {1, 2, 3, 4, 6, 8}:
  S1  held-out explained variance: NMF X_train ~ W H (K components) on the other seven models' profiles;
      held-out weights by NNLS against H; EV = 1 - ||X_held - W_held H||^2 / ||X_held - mean||^2.
      Null: the same with H fitted on N1 Gaussian matrices of the training models' shapes (20 draws).
  S2  matrix-type identification: nearest-centroid classifier on the K normalized mixing weights,
      trained on the seven models, tested on the held-out model's matrices whose type exists in the
      training set. Null: 200 label permutations of the training set.
  S3  descriptive: mean purity (largest weight share) per model, component x type mean weights.
In-sample EV (fit and evaluate on the same model) is written as the ceiling.
Qwen's linear.* (DeltaNet) types are not in any other model: their held-out EV is written on its own
rows (component=linear_only) and they are excluded from S2.

Run:  .venv/bin/python analysis/h14_mixture.py [--models ...] [--nulls 20] [--out results/tables/h14.csv]
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from analysis.h01_spectra import split_name  # noqa: E402
from handi.analysis import plots, registry, results, stats  # noqa: E402

HYP = "H14"
RAW = C.REPO_ROOT / "results" / "raw" / "h01"
N_GRID = 128
GRID = np.linspace(0, 1, N_GRID + 1)[1:]          # t in (0, 1]
KS = (1, 2, 3, 4, 6, 8)
MIN_DIM = 32
N_PERM = 200
NMF_ITERS = 400


# ----------------------------------------------------------------------------- profiles
def profile(s: np.ndarray) -> np.ndarray:
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    s = np.maximum(s, s[0] * 1e-12)
    n = len(s)
    idx = np.minimum((GRID * n).astype(int), n - 1)
    return -np.log10(s[idx] / s[0])


def load_profiles(model: str) -> list[dict]:
    d = RAW / model
    if not d.is_dir():
        raise FileNotFoundError(f"{d}: run analysis/h01_spectra.py first")
    out = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        shape = tuple(int(v) for v in z["shape"])
        if min(shape) < MIN_DIM:
            continue
        layer, comp = split_name(f.stem)
        out.append({"model": model, "name": f.stem, "layer": layer, "type": comp, "shape": shape,
                    "p": profile(z["s_trained"])})
    return out


def n1_profile(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Profile of an N1 Gaussian matrix of this shape (profile is scale-free, so unit variance)."""
    n, m = shape
    big, small = max(n, m), min(n, m)
    W = rng.standard_normal((big, small), dtype=np.float32)
    G = W.T @ W
    ev = np.linalg.eigvalsh(G.astype(np.float64))
    s = np.sqrt(np.maximum(ev, 0.0))
    return profile(s)


# ----------------------------------------------------------------------------- NMF
def nmf(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = NMF_ITERS) -> tuple[np.ndarray, np.ndarray]:
    """Lee-Seung multiplicative updates for min ||X - W H||_F, W (n x k), H (k x g) >= 0."""
    n, g = X.shape
    scale = np.sqrt(X.mean() / k) + 1e-12
    W = rng.random((n, k)) * scale
    H = rng.random((k, g)) * scale
    eps = 1e-12
    for _ in range(iters):
        H *= (W.T @ X) / (W.T @ W @ H + eps)
        W *= (X @ H.T) / (W @ H @ H.T + eps)
    # normalize components to unit max so weights are comparable across K
    norm = H.max(axis=1, keepdims=True) + eps
    return W * norm.T, H / norm


def nnls_weights(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    return np.stack([nnls(H.T, x)[0] for x in X])


def explained_variance(X: np.ndarray, Xhat: np.ndarray) -> float:
    resid = float(((X - Xhat) ** 2).sum())
    total = float(((X - X.mean(axis=0)) ** 2).sum()) + 1e-12
    return 1.0 - resid / total


def nearest_centroid_acc(Wtr: np.ndarray, ytr: np.ndarray, Wte: np.ndarray, yte: np.ndarray) -> float:
    def norm(W):
        return W / (W.sum(axis=1, keepdims=True) + 1e-12)
    A, B = norm(Wtr), norm(Wte)
    labels = np.unique(ytr)
    cents = np.stack([A[ytr == c].mean(axis=0) for c in labels])
    pred = labels[np.argmin(((B[:, None, :] - cents[None, :, :]) ** 2).sum(-1), axis=1)]
    return float((pred == yte).mean())


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = C.parser(HYP, default_nulls=20).parse_args()
    out = C.prepare_out(args)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    data = {m: load_profiles(m) for m in args.models}
    for m, rows in data.items():
        print(f"[{m}] {len(rows)} matrices", flush=True)
    types_by_model = {m: {r["type"] for r in rows} for m, rows in data.items()}

    # N1 profiles per distinct shape, shared across draws (profile under N1 depends only on shape)
    shapes = sorted({r["shape"] for rows in data.values() for r in rows})
    n_null = args.nulls
    print(f"N1 library: {len(shapes)} shapes x {n_null} draws", flush=True)
    n1_prof = {}
    for sh in shapes:
        n1_prof[sh] = np.stack([n1_profile(sh, np.random.default_rng([args.seed, 1, *sh, i])) for i in range(n_null)])
    print(f"  N1 profiles done [{time.time() - t0:.0f}s]", flush=True)

    fig_rows = {}
    with results.ResultsWriter(out) as w:
        common = dict(hypothesis=HYP, seed=args.seed)
        for held in args.models:
            train_rows = [r for m, rows in data.items() if m != held for r in rows]
            test_rows = data[held]
            Xtr = np.stack([r["p"] for r in train_rows])
            shared_types = {t for m in args.models if m != held for t in types_by_model[m]}
            in_lib = np.array([r["type"] in shared_types for r in test_rows])
            Xte = np.stack([r["p"] for r in test_rows])
            ytr = np.array([r["type"] for r in train_rows])
            yte = np.array([r["type"] for r in test_rows])
            tier = registry.tier(held)
            fam = registry.family(held)
            nrows, ncols = Xte.shape
            base = dict(model=held, family=fam, src_dtype=C.SRC_DTYPE[held], n_rows=nrows, n_cols=ncols, **common)

            for k in KS:
                # trained library
                Wtr, H = nmf(Xtr, k, np.random.default_rng([args.seed, 2, k]))
                Wte = nnls_weights(Xte, H)
                ev_all = explained_variance(Xte, Wte @ H)
                ev_lib = explained_variance(Xte[in_lib], (Wte @ H)[in_lib]) if in_lib.any() else float("nan")
                ev_lin = explained_variance(Xte[~in_lib], (Wte @ H)[~in_lib]) if (~in_lib).sum() >= 2 else float("nan")
                # in-sample ceiling
                Wself, Hself = nmf(Xte, k, np.random.default_rng([args.seed, 3, k]))
                ev_self = explained_variance(Xte, Wself @ Hself)
                # N1 library null
                ev_null = []
                for i in range(n_null):
                    Xn = np.stack([n1_prof[r["shape"]][i] for r in train_rows])
                    _, Hn = nmf(Xn, k, np.random.default_rng([args.seed, 4, k, i]), iters=NMF_ITERS // 2)
                    Wn = nnls_weights(Xte, Hn)
                    ev_null.append(explained_variance(Xte, Wn @ Hn))
                ns = stats.null_summary(ev_null)
                z = stats.zscore(ev_all, ev_null)
                w.row(layer="ALL", component=f"K={k}", metric="s1_heldout_ev", value=ev_all, null_kind="N1_library",
                      null_n=ns["n"], null_mean=ns["mean"], null_sd=ns["sd"], z=z, **base)
                w.row(layer="ALL", component=f"K={k}", metric="s1_heldout_ev_shared_types", value=ev_lib, **base)
                if np.isfinite(ev_lin):
                    w.row(layer="ALL", component=f"K={k}|linear_only", metric="s1_heldout_ev", value=ev_lin, **base)
                w.row(layer="ALL", component=f"K={k}", metric="s1_insample_ev", value=ev_self, **base)

                # S2 type identification on shared types
                mask_tr = np.isin(ytr, np.array(sorted(shared_types & set(yte))))
                mask_te = in_lib
                if mask_te.sum() >= 2 and len(np.unique(yte[mask_te])) >= 2:
                    acc = nearest_centroid_acc(Wtr[mask_tr], ytr[mask_tr], Wte[mask_te], yte[mask_te])
                    prng = np.random.default_rng([args.seed, 5, k])
                    acc_null = [nearest_centroid_acc(Wtr[mask_tr], prng.permutation(ytr[mask_tr]), Wte[mask_te], yte[mask_te])
                                for _ in range(N_PERM)]
                    ns2 = stats.null_summary(acc_null)
                    counts = np.unique(yte[mask_te], return_counts=True)[1]
                    w.row(layer="ALL", component=f"K={k}", metric="s2_type_accuracy", value=acc, null_kind="label_perm",
                          null_n=ns2["n"], null_mean=ns2["mean"], null_sd=ns2["sd"], z=stats.zscore(acc, acc_null),
                          ks=float(counts.max() / counts.sum()), **base)   # ks column = majority-class rate

                # S3 purity and per-type weights
                Wn_ = Wte / (Wte.sum(axis=1, keepdims=True) + 1e-12)
                w.row(layer="ALL", component=f"K={k}", metric="s3_mean_purity", value=float(Wn_.max(axis=1).mean()), **base)
                for t in sorted(set(yte)):
                    sel = yte == t
                    for j in range(k):
                        w.row(layer="ALL", component=f"K={k}|{t}", metric=f"s3_weight_c{j}", value=float(Wn_[sel, j].mean()), **base)
                if k == 4:
                    fig_rows[held] = (H, Wn_, yte)
                print(f"  {held:16s} K={k} EV={ev_all:.3f} (shared {ev_lib:.3f}, self {ev_self:.3f}) "
                      f"N1lib={ns['mean']:.3f}+-{ns['sd']:.3f} z={z:.1f}  [{time.time() - t0:.0f}s]", flush=True)

            results.save_raw(HYP.lower(), held, "profiles", X=Xte, types=yte, grid=GRID)

    if not args.no_figures:
        figure(fig_rows, data)
    print(f"wrote {out}", flush=True)


def figure(fig_rows: dict, data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    fig_dir = C.FIGURES / HYP.lower()
    fig_dir.mkdir(parents=True, exist_ok=True)
    # components learned with each model held out (K=4) overlaid, plus type-weight bars for GPT-2 if present
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for i, (held, (H, _, _)) in enumerate(fig_rows.items()):
        for j in range(H.shape[0]):
            ax.plot(GRID, H[j], color=pal[j % len(pal)], alpha=0.5, lw=1)
    ax.set_xlabel("i / n"); ax.set_ylabel("-log10(sigma_i / sigma_1)")
    ax.set_title("K=4 component profiles (one set per held-out model)")
    ax = axes[1]
    key = "gpt2" if "gpt2" in fig_rows else next(iter(fig_rows))
    _, Wn_, yte = fig_rows[key]
    types = sorted(set(yte))
    bottoms = np.zeros(len(types))
    for j in range(Wn_.shape[1]):
        vals = np.array([Wn_[yte == t, j].mean() for t in types])
        ax.bar(range(len(types)), vals, bottom=bottoms, color=pal[j % len(pal)], label=f"c{j}")
        bottoms += vals
    ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("mean normalized weight"); ax.set_title(f"{key}: mixing weights by matrix type (held out)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "h14_components_and_weights.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
