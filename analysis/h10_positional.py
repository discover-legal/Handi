"""H10: positional structure.

Learned-position models (gpt2, tinystories-*), on embed.pos [n_ctx, d]:
  * PCA variance share of the top-10 PCs vs the MP edge of an n_ctx x d Gaussian with the same
    Frobenius norm (edge as a share = (1 + sqrt(gamma))^2 / d);
  * neighbour ratio: mean ||p_{i+1} - p_i|| / median ||p_i - p_j|| over 10^5 random pairs;
  * FFT along position of the top-5 PC time series: dominant period and spectral concentration
    (fraction of non-DC power in the top-3 frequency bins).
  Null N3 (row permutation, 100 draws). A row permutation leaves the spectrum (so the PCA share)
  unchanged, so the PCA share also gets an N1 row (same count of draws) for a non-degenerate z.
  The whole n_ctx range is the pre-registered unit; because TinyStories ships 2048 positions, a
  second component `embed.pos[:512]` repeats the test on the first 512 positions.

RoPE models (pythia-*, qwen3.5-0.8b), per attention head: rotary dims are the first
rot = head_dim * partial_rotary_factor dims of each head, paired rotate_half-style as
(j, j + rot/2) for j < rot/2, with inv_freq_j = theta^(-2j/rot) (transformers gpt_neox and
qwen3_5 modeling code). energy_j = ||W_Q[:, pair j]||_F^2 * ||W_K[:, pair j]||_F^2, normalized over
pairs; per-head entropy of the band distribution (nats; flat = ln(n_pairs)); dispersion of the
band profile across heads of a layer; fraction of Q / K energy in the non-rotary dims.
Null N1 (Gaussian W_Q, W_K, 20 draws). Qwen's per-head q_norm/k_norm gains are not applied
(raw W_Q, W_K as in the pre-registration).

Usage: .venv/bin/python analysis/h10_positional.py [--models ...] [--nulls N] [--seed S] [--out CSV]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis import nulls, plots, spectra  # noqa: E402
from handi.analysis.loader import attn_heads, attn_layers, load_canonical  # noqa: E402
from handi.analysis.results import ResultsWriter, save_raw  # noqa: E402
from handi.analysis.stats import ks_2samp, null_summary, zscore  # noqa: E402

HYP = "H10"
LEARNED = ["gpt2", "tinystories-1m", "tinystories-3m", "tinystories-8m", "tinystories-33m"]
ROPE = ["pythia-70m", "pythia-160m", "qwen3.5-0.8b"]
DEFAULT_MODELS = ["gpt2", "pythia-70m", "pythia-160m", "qwen3.5-0.8b"] + LEARNED[1:]
SRC_DTYPE = {"gpt2": "F32", "pythia-70m": "F16", "pythia-160m": "F16", "qwen3.5-0.8b": "BF16",
             "tinystories-1m": "F32", "tinystories-3m": "F32", "tinystories-8m": "F32", "tinystories-33m": "F32"}
N_PAIRS = 100_000
N_PCS = 5


# ----------------------------------------------------------------------------- learned positions
def pos_stats(P: np.ndarray, rng: np.random.Generator) -> dict:
    P = np.asarray(P, dtype=np.float64)
    n, d = P.shape
    Pc = P - P.mean(0)
    C = Pc.T @ Pc / (n - 1)
    evals, evecs = np.linalg.eigh(C)
    evals, evecs = evals[::-1].clip(0), evecs[:, ::-1]
    share = evals / evals.sum()
    sigma2 = float((P ** 2).sum()) / (n * d)
    gamma = min(n, d) / max(n, d)
    mp_share = spectra.mp_edges(gamma, sigma2)[1] / (d * sigma2)
    nb = np.linalg.norm(np.diff(P, axis=0), axis=1)
    i = rng.integers(0, n, N_PAIRS)
    j = rng.integers(0, n, N_PAIRS)
    m = i != j
    rp = np.linalg.norm(P[i[m]] - P[j[m]], axis=1)
    ts = Pc @ evecs[:, :N_PCS]                      # n x 5 PC time series
    F = np.abs(np.fft.rfft(ts, axis=0)) ** 2        # power, bins 0..n/2
    F[0] = 0.0
    freqs = np.fft.rfftfreq(n, d=1.0)
    dom = np.argmax(F, axis=0)
    dom_period = np.where(dom > 0, 1.0 / np.where(freqs[dom] > 0, freqs[dom], np.nan), np.nan)
    srt = np.sort(F, axis=0)[::-1]
    conc = srt[:3].sum(0) / F.sum(0).clip(1e-30)
    return {
        "pc_share": share[:10], "mp_edge_share": float(mp_share), "pc1_over_mp": float(share[0] / mp_share),
        "top10_share": float(share[:10].sum()),
        "neighbor_ratio": float(nb.mean() / np.median(rp)),
        "neighbor_ratio_median": float(np.median(nb) / np.median(rp)),
        "dominant_period": dom_period, "spectral_concentration": conc,
        "spectral_concentration_mean5": float(conc.mean()),
        "_ts": ts, "_power": F, "_nb": nb,
    }


def run_learned(model: str, n_null: int, seed: int, w: ResultsWriter, fig_dir: Path) -> dict:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    P_full = weights["embed.pos"]
    comps = [("embed.pos", P_full)]
    if P_full.shape[0] > 512:
        comps.append(("embed.pos[:512]", P_full[:512]))
    head = {}
    for comp, P in comps:
        n, d = P.shape
        rng = np.random.default_rng([seed, 10])
        tr = pos_stats(P, rng)
        n3 = [pos_stats(nulls.n3_permute(P, np.random.default_rng([seed, 3, k]), rows=True, cols=False), rng) for k in range(n_null)]
        n1 = [pos_stats(nulls.n1_gaussian(P, np.random.default_rng([seed, 1, k])), rng) for k in range(n_null)]
        base = dict(hypothesis=HYP, model=model, family=fam, layer="", component=comp, n_rows=n, n_cols=d,
                    src_dtype=SRC_DTYPE[model], seed=seed)

        def row(metric, value, null_kind, nv, **kw):
            s = null_summary(nv)
            w.row(**base, metric=metric, value=value, null_kind=null_kind, null_n=s["n"], null_mean=s["mean"],
                  null_sd=s["sd"], z=zscore(value, nv), **kw)

        for null_kind, nl in (("N3", n3), ("N1", n1)):
            for k in range(10):
                row(f"pc{k + 1}_share", float(tr["pc_share"][k]), null_kind, [x["pc_share"][k] for x in nl])
            row("top10_share", tr["top10_share"], null_kind, [x["top10_share"] for x in nl])
            row("pc1_over_mp", tr["pc1_over_mp"], null_kind, [x["pc1_over_mp"] for x in nl])
            row("neighbor_ratio", tr["neighbor_ratio"], null_kind, [x["neighbor_ratio"] for x in nl])
            row("neighbor_ratio_median", tr["neighbor_ratio_median"], null_kind, [x["neighbor_ratio_median"] for x in nl])
            row("spectral_concentration_mean5", tr["spectral_concentration_mean5"], null_kind, [x["spectral_concentration_mean5"] for x in nl])
            for k in range(N_PCS):
                row(f"spectral_concentration_pc{k + 1}", float(tr["spectral_concentration"][k]), null_kind,
                    [x["spectral_concentration"][k] for x in nl])
                row(f"dominant_period_pc{k + 1}", float(tr["dominant_period"][k]), null_kind, [x["dominant_period"][k] for x in nl])
            # distribution statistic: neighbour distances trained vs null; null-vs-null median
            ks_tr = [ks_2samp(tr["_nb"], x["_nb"])[0] for x in nl]
            ks_nn = [ks_2samp(nl[a]["_nb"], nl[b]["_nb"])[0] for a in range(min(len(nl), 15)) for b in range(a + 1, min(len(nl), 15))]
            w.row(**base, metric="ks_neighbor_dist", value=float(np.median(ks_tr)), null_kind=null_kind, null_n=len(nl),
                  ks=float(np.median(ks_tr)), ks_null_median=float(np.median(ks_nn)))
        w.row(**base, metric="mp_edge_share", value=tr["mp_edge_share"], null_kind="", null_n=0)
        save_raw(HYP, model, comp, pc_share=tr["pc_share"], pc_timeseries=tr["_ts"], power=tr["_power"],
                 dominant_period=tr["dominant_period"], spectral_concentration=tr["spectral_concentration"],
                 neighbor_dist=tr["_nb"], n3_neighbor_ratio=[x["neighbor_ratio"] for x in n3],
                 n3_conc=[x["spectral_concentration_mean5"] for x in n3])
        if comp == "embed.pos":
            figure_pos(model, tr, fig_dir)
        head[comp] = dict(pc1=float(tr["pc_share"][0]), mp=tr["mp_edge_share"], top10=tr["top10_share"],
                          nbr=tr["neighbor_ratio"], z_nbr_N3=zscore(tr["neighbor_ratio"], [x["neighbor_ratio"] for x in n3]),
                          conc=tr["spectral_concentration_mean5"],
                          z_conc_N3=zscore(tr["spectral_concentration_mean5"], [x["spectral_concentration_mean5"] for x in n3]),
                          periods=np.round(tr["dominant_period"], 1).tolist())
    print(f"{model} done in {time.time() - t0:.0f}s: {head}", flush=True)
    return head


def figure_pos(model: str, tr: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    ts = tr["_ts"]
    n = ts.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for k in range(N_PCS):
        ax.plot(np.arange(n), ts[:, k], color=pal[k], lw=1.4, label=f"PC{k + 1} ({100 * tr['pc_share'][k]:.1f}%)")
    ax.set_xlabel("position")
    ax.set_ylabel("projection")
    ax.set_title(f"{model} embed.pos: top-{N_PCS} PCs along position (neighbour ratio {tr['neighbor_ratio']:.3f})")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax = axes[1]
    F = tr["_power"]
    freqs = np.fft.rfftfreq(n)[1:]
    for k in range(N_PCS):
        ax.plot(freqs, F[1:, k] / F[1:, k].sum(), color=pal[k], lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (cycles / position)")
    ax.set_ylabel("normalized power")
    ax.set_title("FFT along position per PC")
    ax.grid(True, axis="both")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"h10_{model}_pos_pcs.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- RoPE
def rotary_layout(cfg: dict, fam: str, hd: int) -> tuple[int, np.ndarray]:
    if fam == "gpt_neox":
        pct = cfg.get("rotary_pct", 1.0)
        if "rope_parameters" in cfg:
            pct = cfg["rope_parameters"].get("partial_rotary_factor", pct)
        theta = cfg.get("rotary_emb_base", 10000)
    else:
        rp = cfg["rope_parameters"]
        pct = rp.get("partial_rotary_factor", 1.0)
        theta = rp["rope_theta"]
    rot = int(hd * pct)
    inv_freq = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    return rot, inv_freq


def band_energy(WQ: np.ndarray, WK: np.ndarray, rot: int) -> tuple[np.ndarray, float, float]:
    half = rot // 2
    eq = (WQ.astype(np.float64) ** 2).sum(0)         # per head-dim energy
    ek = (WK.astype(np.float64) ** 2).sum(0)
    e = (eq[:half] + eq[half:rot]) * (ek[:half] + ek[half:rot])
    return e / e.sum(), float(eq[rot:].sum() / eq.sum()), float(ek[rot:].sum() / ek.sum())


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_rope(model: str, n_null: int, seed: int, w: ResultsWriter, fig_dir: Path) -> dict:
    t0 = time.time()
    weights, cfg, fam = load_canonical(model)
    layers = attn_layers(cfg, fam)
    heads0 = attn_heads(weights, cfg, fam, layers[0])
    d, hd = heads0[0][0].shape
    rot, inv_freq = rotary_layout(cfg, fam, hd)
    nb = rot // 2
    flat_H = np.log(nb)
    base = dict(hypothesis=HYP, model=model, family=fam, src_dtype=SRC_DTYPE[model], seed=seed, n_rows=d, n_cols=hd)
    all_prof, all_ent, all_ent_null = [], [], []     # over (layer, head)
    disp_tr, disp_null = [], []
    nonrot_q, nonrot_k = [], []
    for li, layer in enumerate(layers):
        heads = attn_heads(weights, cfg, fam, layer)
        prof, ent, fq, fk = [], [], [], []
        null_prof = [[] for _ in range(n_null)]
        for h, (WQ, WK, _, _) in enumerate(heads):
            p, q_nr, k_nr = band_energy(WQ, WK, rot)
            prof.append(p); ent.append(entropy(p)); fq.append(q_nr); fk.append(k_nr)
            ne = []
            for k in range(n_null):
                r = np.random.default_rng([seed, 1, layer, h, k])
                pn, _, _ = band_energy(nulls.n1_gaussian(WQ, r), nulls.n1_gaussian(WK, r), rot)
                null_prof[k].append(pn); ne.append(entropy(pn))
            s = null_summary(ne)
            w.row(**base, layer=layer, component=f"head{h}", metric="band_entropy", value=ent[-1], null_kind="N1",
                  null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(ent[-1], ne))
            all_ent.append(ent[-1]); all_ent_null.append(ne)
        prof = np.stack(prof)
        all_prof.append(prof)
        disp = float(prof.std(0).mean())
        dn = [float(np.stack(np_).std(0).mean()) for np_ in null_prof]
        disp_tr.append(disp); disp_null.append(dn)
        s = null_summary(dn)
        w.row(**base, layer=layer, component="all_heads", metric="band_profile_dispersion", value=disp, null_kind="N1",
              null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(disp, dn))
        en = [float(np.mean([entropy(p) for p in np_])) for np_ in null_prof]
        s = null_summary(en)
        w.row(**base, layer=layer, component="all_heads", metric="band_entropy_mean", value=float(np.mean(ent)), null_kind="N1",
              null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(float(np.mean(ent)), en))
        for metric, vals in (("nonrotary_energy_frac_q", fq), ("nonrotary_energy_frac_k", fk)):
            # N1 reference is exact: (hd - rot) / hd
            w.row(**base, layer=layer, component="all_heads", metric=metric, value=float(np.mean(vals)), null_kind="N1",
                  null_n=0, null_mean=(hd - rot) / hd, null_sd=0.0)
        nonrot_q.append(float(np.mean(fq))); nonrot_k.append(float(np.mean(fk)))
        print(f"  {model} layer {layer} ({time.time() - t0:.0f}s)", flush=True)
    # model-level aggregates
    v = float(np.mean(all_ent))
    nv = [float(np.mean([e[k] for e in all_ent_null])) for k in range(n_null)]
    s = null_summary(nv)
    w.row(**base, layer="all", component="all_heads", metric="band_entropy_mean", value=v, null_kind="N1",
          null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(v, nv))
    w.row(**base, layer="all", component="all_heads", metric="band_entropy_flat", value=flat_H, null_kind="", null_n=0)
    vd = float(np.mean(disp_tr))
    nd = [float(np.mean([dl[k] for dl in disp_null])) for k in range(n_null)]
    s = null_summary(nd)
    w.row(**base, layer="all", component="all_heads", metric="band_profile_dispersion", value=vd, null_kind="N1",
          null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(vd, nd))
    ks_tr = [ks_2samp(all_ent, [e[k] for e in all_ent_null])[0] for k in range(n_null)]
    ks_nn = [ks_2samp([e[a] for e in all_ent_null], [e[b] for e in all_ent_null])[0] for a in range(n_null) for b in range(a + 1, n_null)]
    w.row(**base, layer="all", component="all_heads", metric="ks_band_entropy", value=float(np.median(ks_tr)), null_kind="N1",
          null_n=n_null, ks=float(np.median(ks_tr)), ks_null_median=float(np.median(ks_nn)))
    w.row(**base, layer="all", component="all_heads", metric="nonrotary_energy_frac_q", value=float(np.mean(nonrot_q)),
          null_kind="N1", null_n=0, null_mean=(hd - rot) / hd, null_sd=0.0)
    w.row(**base, layer="all", component="all_heads", metric="nonrotary_energy_frac_k", value=float(np.mean(nonrot_k)),
          null_kind="N1", null_n=0, null_mean=(hd - rot) / hd, null_sd=0.0)
    P = np.stack(all_prof)      # layers x heads x bands
    save_raw(HYP, model, "rope_bands", band_energy=P, inv_freq=inv_freq, layers=np.array(layers), band_entropy=np.array(all_ent).reshape(P.shape[:2]),
             null_entropy_mean=np.array(nv), rotary_dims=rot, head_dim=hd)
    figure_rope(model, P, inv_freq, layers, fig_dir)
    head = dict(entropy=v, flat=flat_H, z_entropy_N1=zscore(v, nv), dispersion=vd, z_disp_N1=zscore(vd, nd),
                nonrot_q=float(np.mean(nonrot_q)), nonrot_k=float(np.mean(nonrot_k)), nonrot_ref=(hd - rot) / hd)
    print(f"{model} done in {time.time() - t0:.0f}s: {head}", flush=True)
    return head


def figure_rope(model: str, P: np.ndarray, inv_freq: np.ndarray, layers: list[int], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots.style()
    Lh, H, nb = P.shape
    M = P.reshape(Lh * H, nb)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * nb + 3), max(4, 0.16 * Lh * H + 1.5)))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(float(M.max()), 2.0 / nb), interpolation="nearest")
    ax.set_yticks([i * H for i in range(Lh)])
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("layer (rows = heads)")
    step = max(1, nb // 16)
    ax.set_xticks(range(0, nb, step))
    ax.set_xticklabels([f"{2 * np.pi / inv_freq[j]:.3g}" for j in range(0, nb, step)], rotation=90, fontsize=7)
    ax.set_xlabel("rotary band (wavelength in positions, low index = high frequency)")
    ax.set_title(f"{model}: per-head QK energy share per rotary band (flat = {1 / nb:.3f})")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("energy share")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"h10_{model}_rope_bands.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--nulls", type=int, default=None, help="draws; default 100 (N3, learned pos) / 20 (N1, RoPE)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "h10.csv"))
    args = ap.parse_args()
    fig_dir = ROOT / "results" / "figures" / "h10"
    with ResultsWriter(args.out) as w:
        for m in args.models:
            if m in LEARNED:
                run_learned(m, args.nulls or 100, args.seed, w, fig_dir)
            elif m in ROPE:
                run_rope(m, args.nulls or 20, args.seed, w, fig_dir)
            else:
                raise SystemExit(f"{m}: H10 has no test for this model")


if __name__ == "__main__":
    main()
