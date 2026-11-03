"""H9: fractal / self-similar structure on the ordered axes, against N3 (docs/hypotheses.md H9).

(a) embed.pos (learned-pos models): top-10 PC series along position -> DFA-1 Hurst exponent,
    block-variance slope (iid: -1), 1-D box-counting dimension of the thresholded |PC| series, and
    2-D box-counting of the thresholded |W_pos| image, at quantile thresholds 0.5/0.7/0.8/0.9/0.95.
    Null N3 = row permutation of W_pos (100 draws).
(b) embed.tok ordered by token id (first 10,000 ids): same statistics; N3 row permutation
    (50 draws; Qwen 20).
(c) RoPE frequency axis: per head, QK energy per rotary pair ordered by frequency index; the series
    has 8 (Pythia) or 32 (Qwen) points, so DFA is meaningless and only the block-variance slope is
    reported, against N3 = permutation of the frequency order.
(d) hidden-dim control: box-counting of |W| for three representative matrices with rows and columns
    permuted as the null; the pre-registered expectation is null (this is a control).

Verdict (pre-registered): confirmed only if some ordered-axis statistic has |z| >= 5 AND reproduces
in >= 2 of the 3 tiers. A "fractal_verdict" aggregate row per model carries the max |z|.

Choices the pre-registration did not fix (listed in the report):
  * DFA-1 with 16 log-spaced box sizes from 4 to n/4; Hurst = slope of log F(s) vs log s.
  * Block-variance slope: block sizes powers of 2 from 1 to n/8; slope of log var(block means) vs log size.
  * 1-D box counting: mask = |x| > quantile_q(|x|); box sizes powers of 2 up to n/4; dimension =
    -slope of log N(s) vs log s. 2-D box counting: square boxes of size s on the [n, d] |W| mask,
    sizes powers of 2 up to min(n, d)/4.
  * PCs are taken by centred PCA of the rows (positions / token ids) in the trained matrix; under the
    row-permutation null the PCA is recomputed on the permuted matrix.
  * TinyStories W_pos has 2048 rows but the models were trained with a shorter context; the whole
    pre-registered matrix is used.
  * (d) uses layer L//2 attn.q.w, mlp.up.w, attn.o.w (Qwen: the first full-attention layer).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: the Tk backend crashes at interpreter exit
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis import plots, registry  # noqa: E402
from handi.analysis.loader import attn_heads, attn_layers, load_canonical, n_layers  # noqa: E402
from handi.analysis.results import ResultsWriter, save_raw  # noqa: E402
from handi.analysis.stats import null_summary, zscore  # noqa: E402

HYP = "H09"
SRC_DTYPE = {"gpt2": "float32", "gpt_neox": "float16", "qwen3_5": "bfloat16", "gpt_neo": "float32"}
TIER = {"gpt2": "A", "gpt_neox": "A", "qwen3_5": "B", "gpt_neo": "C"}
LEARNED_POS = {"gpt2", "gpt_neo"}
QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95)
N_PC = 10
N_TOK = 10_000


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------

def dfa_sizes(n: int, k: int = 16) -> np.ndarray:
    return np.unique(np.round(np.logspace(np.log10(4), np.log10(n // 4), k)).astype(int))


def dfa(x: np.ndarray, sizes: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """DFA-1: (Hurst exponent, sizes, F(s)). Profile = cumsum(x - mean); linear detrend per box."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if sizes is None:
        sizes = dfa_sizes(n)
    Y = np.cumsum(x - x.mean())
    F = np.empty(sizes.size)
    for i, s in enumerate(sizes):
        nb = n // s
        seg = Y[: nb * s].reshape(nb, s)
        t = np.arange(s, dtype=np.float64)
        t = t - t.mean()
        slope = seg @ t / (t @ t)
        resid = seg - seg.mean(1, keepdims=True) - slope[:, None] * t[None, :]
        F[i] = np.sqrt(np.mean(resid ** 2))
    H = float(np.polyfit(np.log(sizes), np.log(F), 1)[0])
    return H, sizes, F


def block_variance_slope(x: np.ndarray, max_frac: int = 8) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    sizes, v = [], []
    s = 1
    while s <= max(1, n // max_frac):
        nb = n // s
        bm = x[: nb * s].reshape(nb, s).mean(1)
        if nb >= 2:
            sizes.append(s); v.append(bm.var(ddof=1))
        s *= 2
    sizes, v = np.array(sizes, float), np.array(v)
    ok = v > 0
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(sizes[ok]), np.log(v[ok]), 1)[0])


def box_count_1d(mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n = mask.size
    sizes, counts = [], []
    s = 1
    while s <= n // 4:
        nb = n // s
        c = mask[: nb * s].reshape(nb, s).any(1).sum()
        sizes.append(s); counts.append(max(c, 1))
        s *= 2
    sizes, counts = np.array(sizes, float), np.array(counts, float)
    D = float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])
    return D, sizes, counts


def box_count_2d(mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n, m = mask.shape
    sizes, counts = [], []
    s = 1
    while s <= min(n, m) // 4:
        a, b = n // s, m // s
        c = mask[: a * s, : b * s].reshape(a, s, b, s).any(axis=(1, 3)).sum()
        sizes.append(s); counts.append(max(c, 1))
        s *= 2
    sizes, counts = np.array(sizes, float), np.array(counts, float)
    D = float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])
    return D, sizes, counts


def pc_series(M: np.ndarray, n_pc: int) -> np.ndarray:
    """[n_pc, n] scores of the top PCs of the rows of M (centred), ordered along the row axis."""
    X = M - M.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :n_pc] * S[:n_pc]).T


def ordered_stats(M: np.ndarray, n_pc: int, with_image: bool = True) -> dict[str, float]:
    """All ordered-axis statistics of a [n, d] matrix whose row order is the ordered axis."""
    out = {}
    pcs = pc_series(M, n_pc)
    for i, x in enumerate(pcs):
        out[f"pc{i + 1}.dfa_hurst"] = dfa(x)[0]
        out[f"pc{i + 1}.blockvar_slope"] = block_variance_slope(x)
        ax = np.abs(x)
        for q in QUANTILES:
            out[f"pc{i + 1}.boxdim1d_q{q}"] = box_count_1d(ax > np.quantile(ax, q))[0]
    if with_image:
        A = np.abs(M)
        for q in QUANTILES:
            out[f"image.boxdim2d_q{q}"] = box_count_2d(A > np.quantile(A, q))[0]
    return out


def write_block(W, common, layer, comp_prefix, M, n_pc, n_null, rng, axis_label, figdir, name, with_image=True):
    """Trained vs N3 (row permutation) for every ordered-axis statistic of M. Returns max |z| and its label."""
    t0 = time.time()
    tr = ordered_stats(M, n_pc, with_image)
    nulls_ = {k: [] for k in tr}
    for _ in range(n_null):
        st = ordered_stats(M[rng.permutation(M.shape[0])], n_pc, with_image)
        for k, v in st.items():
            nulls_[k].append(v)
    best = (0.0, None)
    for k, v in tr.items():
        s = null_summary(nulls_[k]); z = zscore(v, nulls_[k])
        W.row(**common, layer=layer, component=f"{comp_prefix}:{k.split('.')[0]}", metric=k.split(".")[1] + f"@{axis_label}",
              value=v, null_kind="N3", null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=z, n_rows=M.shape[0], n_cols=M.shape[1])
        if np.isfinite(z) and abs(z) > best[0]:
            best = (abs(z), f"{comp_prefix}:{k}")
    print(f"    {comp_prefix} [{axis_label}] {n_null} N3 draws, {time.time() - t0:.0f}s; max |z| = {best[0]:.1f} at {best[1]}; "
          f"pc1 hurst {tr['pc1.dfa_hurst']:.2f} (null {np.mean(nulls_['pc1.dfa_hurst']):.2f}), "
          f"pc1 blockvar {tr['pc1.blockvar_slope']:.2f} (null {np.mean(nulls_['pc1.blockvar_slope']):.2f})")
    # figures: DFA curve of PC1 trained vs N3 band; box-count log-log of the image at q=0.9
    pal = plots.style()
    import matplotlib.pyplot as plt

    x = pc_series(M, 1)[0]
    H, sizes, F = dfa(x)
    Fn = np.stack([dfa(pc_series(M[rng.permutation(M.shape[0])], 1)[0], sizes)[2] for _ in range(min(n_null, 30))])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.fill_between(sizes, Fn.min(0), Fn.max(0), color="#c9c8c4", alpha=0.6, lw=0, label="N3 range (row-permuted)")
    ax.plot(sizes, F, color=pal[0], marker="o", label=f"trained, H = {H:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("box size s"); ax.set_ylabel("DFA fluctuation F(s)")
    ax.set_title(f"{name}: DFA of PC1 along {axis_label}")
    ax.legend()
    ax = axes[1]
    if with_image:
        A = np.abs(M); D, bs, bc = box_count_2d(A > np.quantile(A, 0.9))
        Dn, _, bcn = box_count_2d(np.abs(M[rng.permutation(M.shape[0])]) > np.quantile(A, 0.9))
        ax.plot(bs, bcn, color="#9a9994", marker="s", label=f"N3 draw, D = {Dn:.2f}")
        ax.plot(bs, bc, color=pal[0], marker="o", label=f"trained, D = {D:.2f}")
        ax.set_title(f"{name}: box count of |W| > q0.9 image ({axis_label} x hidden)")
    else:
        ax_ = np.abs(x); D, bs, bc = box_count_1d(ax_ > np.quantile(ax_, 0.9))
        xp = np.abs(x[rng.permutation(x.size)]); Dn, _, bcn = box_count_1d(xp > np.quantile(xp, 0.9))
        ax.plot(bs, bcn, color="#9a9994", marker="s", label=f"N3 draw, D = {Dn:.2f}")
        ax.plot(bs, bc, color=pal[0], marker="o", label=f"trained, D = {D:.2f}")
        ax.set_title(f"{name}: box count of |PC1| > q0.9 along {axis_label}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("box size s"); ax.set_ylabel("boxes N(s)"); ax.legend()
    fig.savefig(figdir / f"h09_{name}_{comp_prefix.replace('.', '_')}.png"); plt.close(fig)
    save_raw(HYP.lower(), name, comp_prefix, **{k: v for k, v in tr.items()}, dfa_sizes=sizes, dfa_F=F, dfa_F_null=Fn)
    return best


def rope_pairs_energy(cfg, fam, w, layers):
    """Per head: energy per rotary pair ordered by frequency index, for W_Q and W_K."""
    if fam == "gpt_neox":
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        frac = (cfg.get("rope_parameters") or {}).get("partial_rotary_factor", cfg.get("rotary_pct", 0.25))
    else:
        hd = cfg["head_dim"]; frac = cfg["rope_parameters"].get("partial_rotary_factor", 1.0)
    rot = int(hd * frac); half = rot // 2
    out = []
    for l in layers:
        for h, (WQ, WK, WV, WO) in enumerate(attn_heads(w, cfg, fam, l)):
            for nm, M in (("q", WQ), ("k", WK)):
                e = (M[:, :half] ** 2).sum(0) + (M[:, half:rot] ** 2).sum(0)
                out.append((l, h, nm, e.astype(np.float64)))
    return out


def run_model(name, args, W, figdir):
    w, cfg, fam = load_canonical(name)
    src = SRC_DTYPE[fam]
    common = dict(hypothesis=HYP, model=name, family=fam, src_dtype=src, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    bests = []
    # (a) positional embeddings
    if fam in LEARNED_POS:
        P = w["embed.pos"].astype(np.float64)
        bests.append(write_block(W, common, -1, "embed.pos", P, N_PC, args.nulls_pos, rng, "position", figdir, name))
    # (b) token embeddings by id
    E = w["embed.tok"][:N_TOK].astype(np.float64)
    n_tok_null = args.nulls_tok_qwen if fam == "qwen3_5" else args.nulls_tok
    bests.append(write_block(W, common, -1, "embed.tok", E, N_PC, n_tok_null, rng, "token_id", figdir, name))
    # (c) RoPE frequency axis: block-variance slope only
    if fam not in LEARNED_POS:
        t0 = time.time()
        layers = attn_layers(cfg, fam)
        series = rope_pairs_energy(cfg, fam, w, layers)
        tr_vals, null_means = [], []
        per_head_z = []
        for l, h, nm, e in series:
            v = block_variance_slope(e, max_frac=2)
            nv = [block_variance_slope(e[rng.permutation(e.size)], max_frac=2) for _ in range(args.nulls_pos)]
            s = null_summary(nv); z = zscore(v, nv)
            W.row(**common, layer=l, component=f"attn.h{h}.{nm}", metric="blockvar_slope@rope_freq", value=v, null_kind="N3",
                  null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=z, n_rows=e.size)
            tr_vals.append(v); null_means.append(nv); per_head_z.append(z)
        # aggregate: mean slope over heads vs the mean over heads of each null draw
        agg = float(np.nanmean(tr_vals)); nm_arr = np.nanmean(np.array(null_means), axis=0)
        s = null_summary(nm_arr); z = zscore(agg, nm_arr)
        W.row(**common, layer=-1, component="attn.all_heads.qk", metric="blockvar_slope_mean@rope_freq", value=agg, null_kind="N3",
              null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=z, n_rows=len(series), n_cols=series[0][3].size)
        print(f"    rope_freq: {len(series)} series of {series[0][3].size} points, {time.time() - t0:.0f}s; mean block-variance slope "
              f"{agg:.2f} vs null {s['mean']:.2f} (z = {z:.1f}); DFA not reported (series too short)")
        bests.append((abs(z) if np.isfinite(z) else 0.0, "rope_freq:blockvar_slope_mean"))
    # (d) hidden-dim control
    t0 = time.time()
    L = n_layers(cfg, fam)
    lc = attn_layers(cfg, fam)[0] if fam == "qwen3_5" else L // 2
    ctrl_max = 0.0
    for t in ("attn.q.w", "mlp.up.w", "attn.o.w"):
        M = np.abs(w[f"layers.{lc}.{t}"].astype(np.float64))
        for q in QUANTILES:
            thr = np.quantile(M, q)
            v = box_count_2d(M > thr)[0]
            nv = [box_count_2d(M[rng.permutation(M.shape[0])][:, rng.permutation(M.shape[1])] > thr)[0] for _ in range(args.nulls_ctrl)]
            s = null_summary(nv); z = zscore(v, nv)
            ctrl_max = max(ctrl_max, abs(z) if np.isfinite(z) else 0.0)
            W.row(**common, layer=lc, component=f"{t}:control_unordered", metric=f"boxdim2d_q{q}@hidden_dims", value=v, null_kind="N3rc",
                  null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=z, n_rows=M.shape[0], n_cols=M.shape[1])
    W.row(**common, layer=lc, component="control_unordered", metric="max_abs_z@hidden_dims", value=ctrl_max, null_kind="N3rc", null_n=args.nulls_ctrl)
    print(f"    hidden-dim control (layer {lc}, rows+cols permuted null, {args.nulls_ctrl} draws): max |z| = {ctrl_max:.1f} "
          f"({'at null, as pre-registered' if ctrl_max < 5 else 'NOT at null'}) -- this is a control, not an ordered-axis test; {time.time() - t0:.0f}s")
    # verdict row
    best = max(bests, key=lambda b: b[0])
    W.row(**common, layer=-1, component="fractal_verdict", metric="max_abs_z_ordered_axes", value=best[0], null_kind="N3",
          n_rows=len(bests), n_cols=0)
    W.row(**common, layer=-1, component="fractal_verdict", metric="max_abs_z_statistic", value=best[1], null_kind="N3")
    print(f"  VERDICT {name} (tier {TIER[fam]}): max |z| over ordered-axis statistics = {best[0]:.1f} at {best[1]} -> "
          f"{'passes |z| >= 5 in this model' if best[0] >= 5 else 'below |z| = 5'}")
    return TIER[fam], best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(registry.MODELS))
    ap.add_argument("--nulls", type=int, default=None, help="override every null count")
    ap.add_argument("--nulls-pos", type=int, default=100)
    ap.add_argument("--nulls-tok", type=int, default=50)
    ap.add_argument("--nulls-tok-qwen", type=int, default=20)
    ap.add_argument("--nulls-ctrl", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/tables/h09.csv"))
    args = ap.parse_args()
    if args.nulls is not None:
        args.nulls_pos = args.nulls_tok = args.nulls_tok_qwen = args.nulls_ctrl = args.nulls
    figdir = ROOT / "results/figures/h09"; figdir.mkdir(parents=True, exist_ok=True)
    verdicts = {}
    with ResultsWriter(args.out) as W:
        for name in args.models:
            t0 = time.time(); print(f"== {name}")
            verdicts[name] = run_model(name, args, W, figdir)
            print(f"  done in {time.time() - t0:.0f}s")
        tiers_pass = {tier for tier, (z, _) in verdicts.values() if z >= 5}
        confirmed = len(tiers_pass) >= 2
        for name, (tier, (z, lab)) in verdicts.items():
            W.row(hypothesis=HYP, model=name, family=registry.family(name), src_dtype=SRC_DTYPE[registry.family(name)], seed=args.seed,
                  layer=-1, component="fractal_verdict", metric="confirmed_prereg_rule",
                  value=int(confirmed), null_kind="N3", n_rows=len(tiers_pass))
    print(f"\nTiers with any ordered-axis |z| >= 5: {sorted(tiers_pass) or 'none'} -> H9 {'CONFIRMED' if confirmed else 'NOT confirmed'} "
          f"under the pre-registered rule (|z| >= 5 in >= 2 tiers) over the models run.")


if __name__ == "__main__":
    main()
