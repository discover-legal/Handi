"""H6: previous-token and induction heads are identifiable from weights alone.

Weights-only (a) previous-token score, (b) K-composition, (c) copying score; (d) confirmation on
the real model (M2 copy accuracy, attention to the induction target). See docs/hypotheses.md H6.

Conventions (from handi.analysis.loader.attn_heads): W_Q, W_K, W_V, W_O are [d, hd] so that
W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O.T (both d x d).

Choices the pre-registration did not fix (listed in the report):
  * RoPE score: the pre-registered formula tr(W_Q^T R_delta W_K C) is read in the dimensionally
    consistent form tr(W_Q R_{-delta} W_K^T C) = E[x^T W_Q R_{-delta} W_K^T x] for a residual x with
    covariance C; R_{-delta} = R_m^T R_{m-delta} is the net rotation between a query at m and a key
    delta positions earlier (rotate_half layout, identity on the non-rotary dims).
  * Attention biases (GPT-2, Pythia) are ignored in all weight-only scores.
  * Qwen q_norm/k_norm: the (1 + w) gains enter as diagonal scalings of W_Q/W_K head columns (the RMS
    normalisation itself is nonlinear and is not modelled).
  * Copying score = cos_F(M, I) with M = W_E[:1000] W_OV W_U[:1000]^T (= tr(M) / (||M||_F sqrt(1000))).
  * Random tokens for (d) are uniform over the tokenizer vocabulary (Qwen: len(tokenizer) < config vocab).
  * M2 accuracy is scored on predictions made at positions 50..98 (targets 51..99), the positions
    where the repeat is determinable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: the Tk backend crashes at interpreter exit
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from handi.analysis import nulls, plots, registry  # noqa: E402
from handi.analysis.loader import attn_heads, attn_layers, load_canonical  # noqa: E402
from handi.analysis.results import ResultsWriter, save_raw  # noqa: E402
from handi.analysis.stats import null_summary, zscore  # noqa: E402

HYP = "H06"
SRC_DTYPE = {"gpt2": "float32", "gpt_neox": "float16", "qwen3_5": "bfloat16", "gpt_neo": "float32"}
LEARNED_POS = {"gpt2", "gpt_neo"}
N_POS = 256
DELTAS = np.arange(0, 65)
TOP_TOKENS = 1000


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def head_geometry(cfg: dict, fam: str) -> tuple[int, int]:
    """(n_heads, head_dim)."""
    if fam == "qwen3_5":
        return cfg["num_attention_heads"], cfg["head_dim"]
    if fam == "gpt_neo":
        return cfg["num_heads"], cfg["hidden_size"] // cfg["num_heads"]
    if fam == "gpt_neox":
        return cfg["num_attention_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]
    return cfg["n_head"], cfg["n_embd"] // cfg["n_head"]


def attn_scale(fam: str, hd: int) -> float:
    """GPT-Neo (TinyStories) has no 1/sqrt(hd); everyone else does."""
    return 1.0 if fam == "gpt_neo" else 1.0 / np.sqrt(hd)


def rotary_setup(cfg: dict, fam: str, hd: int) -> tuple[int, np.ndarray]:
    """(rotary_dim, inv_freq) exactly as transformers computes them for the family."""
    if fam == "gpt_neox":
        rp = cfg.get("rope_parameters") or {}
        base = rp.get("rope_theta", cfg.get("rotary_emb_base", 10000))
        frac = rp.get("partial_rotary_factor", cfg.get("rotary_pct", 0.25))
    else:  # qwen3_5
        rp = cfg["rope_parameters"]
        base, frac = rp["rope_theta"], rp.get("partial_rotary_factor", 1.0)
    dim = int(hd * frac)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    return dim, inv_freq


def rotation_matrices(hd: int, rot_dim: int, inv_freq: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """R[-delta] for each delta: [n_delta, hd, hd]; rotate_half layout (pair i <-> i + rot_dim/2).

    R(theta) on a pair (x1, x2): x1' = x1 cos - x2 sin, x2' = x2 cos + x1 sin (transformers' apply_rotary_pos_emb).
    The score between a query at m and a key at m - delta is q^T R(-delta * inv_freq) k.
    """
    half = rot_dim // 2
    out = np.zeros((deltas.size, hd, hd), dtype=np.float64)
    for n, delta in enumerate(deltas):
        R = np.eye(hd)
        th = -float(delta) * inv_freq
        c, s = np.cos(th), np.sin(th)
        for i in range(half):
            j = i + half
            R[i, i] = c[i]; R[i, j] = -s[i]
            R[j, i] = s[i]; R[j, j] = c[i]
        out[n] = R
    return out


def rope_curve(WQ: np.ndarray, WK: np.ndarray, C: np.ndarray, Rs: np.ndarray, scale: float) -> np.ndarray:
    """score(delta) = scale * tr(W_Q R_{-delta} W_K^T C) for every delta; W_Q, W_K [d, hd]."""
    M = WK.T @ C @ WQ                      # [hd, hd]; tr(R M) = sum_ij R_ij M_ji
    return scale * np.einsum("nij,ji->n", Rs, M)


def rope_stat(curve: np.ndarray) -> float:
    return float(curve[1] - curve[2:].mean())


def learned_pos_S(P: np.ndarray, WQ: np.ndarray, WK: np.ndarray, scale: float) -> np.ndarray:
    return scale * (P @ WQ) @ (P @ WK).T


_IDX = {}


def prev_token_stat(S: np.ndarray) -> float:
    """mean S[i, i-1] - mean S[i, j < i-1], over i >= 2."""
    n = S.shape[0]
    if n not in _IDX:
        i = np.arange(2, n)
        _IDX[n] = ((i, i - 1), np.tril_indices(n, k=-2))
    d1, il = _IDX[n]
    return float(S[d1].mean() - S[il].mean())


def kcomp_score(GK_m: np.ndarray, A: np.ndarray, GO_l: np.ndarray, nQK_m: float, nOV_l: float) -> float:
    """||W_QK^(m)T W_OV^(l)||_F / (||W_QK^(m)||_F ||W_OV^(l)||_F) with A = W_Q^(m)T W_V^(l) [hd, hd]."""
    num = np.sqrt(max(np.trace(GK_m @ A @ GO_l @ A.T), 0.0))
    return float(num / (nQK_m * nOV_l))


def frob_product_norm(GA: np.ndarray, GB: np.ndarray) -> float:
    """||A B^T||_F = sqrt(tr(GA GB)) with GA = A^T A, GB = B^T B."""
    return float(np.sqrt(max(np.trace(GA @ GB), 0.0)))


def copy_score(E: np.ndarray, U: np.ndarray, WV: np.ndarray, WO: np.ndarray) -> float:
    """cos_F(M, I), M = E W_V W_O^T U^T on the top token rows."""
    A = E @ WV            # [t, hd]
    B = U @ WO            # [t, hd]
    tr = float(np.sum(A * B))
    fro = np.sqrt(max(np.trace((A.T @ A) @ (B.T @ B)), 0.0))
    return tr / (fro * np.sqrt(E.shape[0]) + 1e-30)


# ----------------------------------------------------------------------------------------------
# weights-only analysis
# ----------------------------------------------------------------------------------------------

def run_weights(name: str, w, cfg, fam, args, W: ResultsWriter, figdir: Path):
    pal = plots.style()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    H, hd = head_geometry(cfg, fam)
    layers = attn_layers(cfg, fam)
    d = w["embed.tok"].shape[1]
    scale = attn_scale(fam, hd)
    src = SRC_DTYPE[fam]
    common = dict(hypothesis=HYP, model=name, family=fam, src_dtype=src, seed=args.seed)
    heads = {l: attn_heads(w, cfg, fam, l) for l in layers}
    if fam == "qwen3_5":   # q_norm / k_norm gains (1 + w) as diagonal scalings of the head dims
        for l in layers:
            gq = 1.0 + w[f"layers.{l}.attn.qnorm.w"]
            gk = 1.0 + w[f"layers.{l}.attn.knorm.w"]
            heads[l] = [(WQ * gq[None, :], WK * gk[None, :], WV, WO) for (WQ, WK, WV, WO) in heads[l]]
    nn = args.nulls
    prev_rows = []        # (layer, head, stat, z_primary, z_n1)
    curves = {}

    # ---------------- (a) previous-token score ----------------
    t0 = time.time()
    if fam in LEARNED_POS:
        P = w["embed.pos"][:N_POS].astype(np.float64)
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        # N3 row-permuted P: S(P[perm]) = S[perm][:, perm], so the null is evaluated by indexing S (identical result).
        perms = [rng.permutation(N_POS) for _ in range(nn)]
        best = None
        for l in layers:
            for h, (WQ, WK, WV, WO) in enumerate(heads[l]):
                WQ64, WK64 = WQ.astype(np.float64), WK.astype(np.float64)
                for variant, PP in (("raw", P), ("rownorm", Pn)):
                    S = learned_pos_S(PP, WQ64, WK64, scale)
                    val = prev_token_stat(S)
                    n3 = [prev_token_stat(S[pm][:, pm]) for pm in perms]
                    n1 = [prev_token_stat(learned_pos_S(PP, nulls.n1_gaussian(WQ, rng), nulls.n1_gaussian(WK, rng), scale)) for _ in range(nn)]
                    for nk, nv in (("N3", n3), ("N1", n1)):
                        s = null_summary(nv)
                        W.row(**common, layer=l, component=f"attn.h{h}",
                              metric=f"prev_token_score_{variant}", value=val, null_kind=nk,
                              null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=zscore(val, nv),
                              n_rows=N_POS, n_cols=d)
                    if variant == "raw":
                        z3, z1 = zscore(val, n3), zscore(val, n1)
                        prev_rows.append((l, h, val, z3, z1))
                        if best is None or min(z3, z1) > best[0]:
                            best = (min(z3, z1), l, h, S)
        # figure: S heatmap for the top learned-pos head
        z3, l, h, S = best
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        n_show = 48
        v = np.abs(S[:n_show, :n_show]).max()
        im = ax.imshow(np.tril(S[:n_show, :n_show]), cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.grid(False)
        ax.set_xlabel("key position j"); ax.set_ylabel("query position i")
        ax.set_title(f"{name}: S = P W_QK P^T, layer {l} head {h} (min z over N3, N1 = {z3:.1f})")
        fig.colorbar(im, ax=ax, shrink=0.8, label="score (causal part)")
        fig.savefig(figdir / f"h06_{name}_S_heatmap.png"); plt.close(fig)
    else:
        rot_dim, inv_freq = rotary_setup(cfg, fam, hd)
        Rs = rotation_matrices(hd, rot_dim, inv_freq, DELTAS)
        E = w["embed.tok"].astype(np.float64)
        C = np.cov(E, rowvar=False)
        for l in layers:
            for h, (WQ, WK, WV, WO) in enumerate(heads[l]):
                cur = rope_curve(WQ.astype(np.float64), WK.astype(np.float64), C, Rs, scale)
                val = rope_stat(cur)
                n1 = [rope_stat(rope_curve(nulls.n1_gaussian(WQ, rng).astype(np.float64),
                                           nulls.n1_gaussian(WK, rng).astype(np.float64), C, Rs, scale))
                      for _ in range(nn)]
                s = null_summary(n1)
                z1 = zscore(val, n1)
                W.row(**common, layer=l, component=f"attn.h{h}", metric="prev_token_score_rope", value=val,
                      null_kind="N1", null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=z1,
                      n_rows=d, n_cols=hd)
                prev_rows.append((l, h, val, z1, z1))
                curves[(l, h)] = (cur, s["sd"])
    # ranking: a head counts only if it beats BOTH pre-registered nulls (N3 permuted P, N1 random head),
    # so heads are ranked by min(z_N3, z_N1) (RoPE models: N1 only, both entries equal).
    prev_sorted = sorted(prev_rows, key=lambda r: -min(r[3], r[4]))
    print(f"  (a) previous-token: {time.time() - t0:.1f}s; top-3 by min(z_N3, z_N1):",
          [(l, h, round(z3, 1), round(z1, 1)) for l, h, _, z3, z1 in prev_sorted[:3]])
    for rank, (l, h, val, z3, z1) in enumerate(prev_sorted[:3]):
        W.row(**common, layer=l, component=f"attn.h{h}", metric=f"prev_token_top{rank + 1}_zmin", value=min(z3, z1),
              null_kind="N3&N1" if fam in LEARNED_POS else "N1", null_n=nn)
    if curves:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for i, (l, h, val, z, _) in enumerate(prev_sorted[:3]):
            cur, sd = curves[(l, h)]
            ax.fill_between(DELTAS, cur - 2 * sd, cur + 2 * sd, color=pal[i], alpha=0.12, lw=0)
            ax.plot(DELTAS, cur, color=pal[i], label=f"L{l} H{h} (z = {z:.1f})")
        ax.axvline(1, color=plots.INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xlabel("delta = query position - key position"); ax.set_ylabel("E[q^T R k] (scaled)")
        ax.set_title(f"{name}: RoPE previous-token curves, top-3 heads (band: 2 N1 null sd)")
        ax.legend()
        fig.savefig(figdir / f"h06_{name}_rope_curves.png"); plt.close(fig)

    # ---------------- (b) K-composition ----------------
    t0 = time.time()
    G = {}
    for l in layers:
        for h, (WQ, WK, WV, WO) in enumerate(heads[l]):
            GQ, GK, GV, GO = (M.T.astype(np.float64) @ M.astype(np.float64) for M in (WQ, WK, WV, WO))
            G[(l, h)] = dict(WQ=WQ, WK=WK, WV=WV, WO=WO, GK=GK, GO=GO, nQK=frob_product_norm(GQ, GK), nOV=frob_product_norm(GV, GO))
    n_kc = args.kc_nulls
    null_kc = []
    shape = (d, hd)
    for _ in range(n_kc):
        RQ, RK, RV, RO = (rng.standard_normal(shape) for _ in range(4))
        GKr, GOr = RK.T @ RK, RO.T @ RO
        null_kc.append(kcomp_score(GKr, RQ.T @ RV, GOr, frob_product_norm(RQ.T @ RQ, GKr), frob_product_norm(RV.T @ RV, GOr)))
    skc = null_summary(null_kc)
    pairs = []
    for i, l in enumerate(layers):
        for m in layers[i + 1:]:
            for hl in range(H):
                gl = G[(l, hl)]
                for hm in range(H):
                    gm = G[(m, hm)]
                    A = gm["WQ"].T.astype(np.float64) @ gl["WV"].astype(np.float64)
                    val = kcomp_score(gm["GK"], A, gl["GO"], gm["nQK"], gl["nOV"])
                    pairs.append((l, hl, m, hm, val, (val - skc["mean"]) / skc["sd"]))
    pairs.sort(key=lambda p: -p[5])
    print(f"  (b) K-composition: {len(pairs)} pairs, {time.time() - t0:.1f}s; top-5:",
          [(p[0], p[1], p[2], p[3], round(p[5], 1)) for p in pairs[:5]])
    for rank, (l, hl, m, hm, val, z) in enumerate(pairs[:5]):
        W.row(**common, layer=m, component=f"attn.h{hm}<-L{l}.h{hl}", metric=f"kcomp_top{rank + 1}", value=val,
              null_kind="N1", null_n=skc["n"], null_mean=skc["mean"], null_sd=skc["sd"], z=z, n_rows=d, n_cols=d)
    kc_vals = np.array([p[4] for p in pairs])
    W.row(**common, layer=-1, component="all_pairs", metric="kcomp_mean", value=float(kc_vals.mean()),
          null_kind="N1", null_n=skc["n"], null_mean=skc["mean"], null_sd=skc["sd"],
          z=(kc_vals.mean() - skc["mean"]) / skc["sd"], n_rows=len(pairs))
    W.row(**common, layer=-1, component="all_pairs", metric="kcomp_frac_z_ge_5", value=float(np.mean([p[5] >= 5 for p in pairs])),
          null_kind="N1", null_n=skc["n"], n_rows=len(pairs))
    save_raw(HYP.lower(), name, "kcomp", pairs=np.array([p[:5] for p in pairs]), null=np.array(null_kc))

    # ---------------- (c) copying score ----------------
    t0 = time.time()
    E = w["embed.tok"][:TOP_TOKENS].astype(np.float64)
    U = (w["unembed.w"] if "unembed.w" in w else w["embed.tok"])[:TOP_TOKENS].astype(np.float64)
    copy_rows = []
    for l in layers:
        for h, (WQ, WK, WV, WO) in enumerate(heads[l]):
            val = copy_score(E, U, WV.astype(np.float64), WO.astype(np.float64))
            n1 = [copy_score(E, U, nulls.n1_gaussian(WV, rng).astype(np.float64), nulls.n1_gaussian(WO, rng).astype(np.float64)) for _ in range(n_kc)]
            s = null_summary(n1)
            z = zscore(val, n1)
            W.row(**common, layer=l, component=f"attn.h{h}", metric="copy_score", value=val, null_kind="N1",
                  null_n=s["n"], null_mean=s["mean"], null_sd=s["sd"], z=z, n_rows=TOP_TOKENS, n_cols=TOP_TOKENS)
            copy_rows.append((l, h, val, z))
    copy_sorted = sorted(copy_rows, key=lambda r: -r[3])
    print(f"  (c) copying: {time.time() - t0:.1f}s; top-3:", [(l, h, round(float(v), 3), round(z, 1)) for l, h, v, z in copy_sorted[:3]])
    for rank, (l, h, val, z) in enumerate(copy_sorted[:3]):
        W.row(**common, layer=l, component=f"attn.h{h}", metric=f"copy_top{rank + 1}_z", value=z, null_kind="N1", null_n=n_kc)

    # aggregates (weights only)
    W.row(**common, layer=-1, component="model", metric="n_heads_prev_token_z_ge_5",
          value=int(sum(r[3] >= 5 for r in prev_rows)), null_kind="N3" if fam in LEARNED_POS else "N1", null_n=nn, n_rows=len(prev_rows))
    W.row(**common, layer=-1, component="model", metric="n_pairs_kcomp_z_ge_5", value=int(sum(p[5] >= 5 for p in pairs)),
          null_kind="N1", null_n=n_kc, n_rows=len(pairs))
    W.row(**common, layer=-1, component="model", metric="n_heads_copy_z_ge_5", value=int(sum(r[3] >= 5 for r in copy_rows)),
          null_kind="N1", null_n=n_kc, n_rows=len(copy_rows))
    return dict(prev=prev_sorted, kcomp=pairs, copy=copy_sorted, H=H, layers=layers, common=common)


# ----------------------------------------------------------------------------------------------
# (d) confirmation on the real model
# ----------------------------------------------------------------------------------------------

def run_confirm(name: str, fam: str, cfg: dict, res: dict, args, W: ResultsWriter, figdir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM
    import matplotlib.pyplot as plt

    pal = plots.style()
    snap = registry.snapshot_dir(name)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = Qwen3_5ForCausalLM if fam == "qwen3_5" else AutoModelForCausalLM
    model = cls.from_pretrained(snap, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(snap)
    V = min(len(tok), cfg["vocab_size"])
    g = torch.Generator().manual_seed(args.seed)
    half = args.period
    base = torch.randint(0, V, (args.n_seq, half), generator=g)
    ids = torch.cat([base, base], dim=1).to(dev)           # [n_seq, 2*half]
    layers, H = res["layers"], res["H"]
    ind_att = np.zeros((len(layers), H)); prev_att = np.zeros((len(layers), H))
    correct = 0; total = 0
    with torch.no_grad():
        for b in range(args.n_seq):
            out = model(ids[b:b + 1], output_attentions=True)
            logits = out.logits[0]
            pred = logits[half:2 * half - 1].argmax(-1)
            tgt = ids[b, half + 1:2 * half]
            correct += int((pred == tgt).sum()); total += int(tgt.numel())
            atts = [a for a in out.attentions if a is not None]
            assert len(atts) == len(layers), (len(atts), len(layers))
            qi = torch.arange(half, 2 * half, device=dev)
            for li, a in enumerate(atts):
                a = a[0]                                     # [H, T, T]
                ind_att[li] += a[:, qi, qi - half + 1].mean(-1).cpu().numpy()
                prev_att[li] += a[:, qi, qi - 1].mean(-1).cpu().numpy()
    ind_att /= args.n_seq; prev_att /= args.n_seq
    m2 = correct / total
    common = res["common"]
    W.row(**common, layer=-1, component="model", metric="m2_copy_accuracy", value=m2, null_kind="chance",
          null_mean=1.0 / V, n_rows=args.n_seq, n_cols=half)
    for li, l in enumerate(layers):
        for h in range(H):
            W.row(**common, layer=l, component=f"attn.h{h}", metric="attn_to_induction_target", value=float(ind_att[li, h]),
                  null_kind="uniform_causal", null_mean=float(np.mean(1.0 / (np.arange(half, 2 * half) + 1))), n_rows=args.n_seq)
            W.row(**common, layer=l, component=f"attn.h{h}", metric="attn_to_prev_token", value=float(prev_att[li, h]),
                  null_kind="uniform_causal", null_mean=float(np.mean(1.0 / (np.arange(half, 2 * half) + 1))), n_rows=args.n_seq)
    order = np.dstack(np.unravel_index(np.argsort(-ind_att.ravel()), ind_att.shape))[0]
    top_ind = [(layers[li], int(h), float(ind_att[li, h])) for li, h in order[:3]]
    order_p = np.dstack(np.unravel_index(np.argsort(-prev_att.ravel()), prev_att.shape))[0]
    top_prev = [(layers[li], int(h), float(prev_att[li, h])) for li, h in order_p[:3]]
    kc_later = {(p[2], p[3]) for p in res["kcomp"][:5]}
    kc_earlier = {(p[0], p[1]) for p in res["kcomp"][:5]}
    copy_top = {(r[0], r[1]) for r in res["copy"][:3]}
    prev_top = {(r[0], r[1]) for r in res["prev"][:3]}
    ind_set = {(l, h) for l, h, _ in top_ind}
    prev_set = {(l, h) for l, h, _ in top_prev}
    match_kc = len(ind_set & kc_later); match_copy = len(ind_set & copy_top)
    match_prev = len(prev_set & prev_top); match_prev_kc = len(prev_set & kc_earlier)
    for metric, val in (("match_induction_vs_kcomp_top5", match_kc), ("match_induction_vs_copy_top3", match_copy),
                        ("match_prevtoken_attn_vs_weights_top3", match_prev), ("match_prevtoken_attn_vs_kcomp_earlier", match_prev_kc)):
        W.row(**common, layer=-1, component="model", metric=metric, value=val, null_kind="", n_rows=3)
    for rank, (l, h, v) in enumerate(top_ind):
        W.row(**common, layer=l, component=f"attn.h{h}", metric=f"induction_attn_top{rank + 1}", value=float(v), null_kind="")
    print(f"  (d) M2 = {m2:.3f} (chance {1 / V:.2e}); top induction-attn heads {[(l, h, round(v, 3)) for l, h, v in top_ind]}; "
          f"top prev-attn heads {[(l, h, round(v, 3)) for l, h, v in top_prev]}; "
          f"matches: ind~kcomp {match_kc}/3, ind~copy {match_copy}/3, prev~weights {match_prev}/3, prev~kcomp-earlier {match_prev_kc}/3")
    # bar chart: attention to the induction target per head, candidates highlighted
    flat = [(layers[li], h, ind_att[li, h]) for li in range(len(layers)) for h in range(H)]
    fig, ax = plt.subplots(figsize=(max(6, 0.06 * len(flat) + 2), 3.4))
    x = np.arange(len(flat))
    colors = [pal[0] if (l, h) in kc_later or (l, h) in copy_top else "#c9c8c4" for l, h, _ in flat]
    ax.bar(x, [v for _, _, v in flat], color=colors, width=0.8)
    ax.set_ylabel("mean attention to i - 49"); ax.set_xlabel("head (layer-major order)")
    ax.set_title(f"{name}: attention to the induction target; blue = weight-only candidate (K-comp top-5 / copy top-3). M2 = {m2:.2f}")
    ticks = [i for i, (l, h, _) in enumerate(flat) if h == 0]
    ax.set_xticks(ticks); ax.set_xticklabels([f"L{flat[i][0]}" for i in ticks])
    fig.savefig(figdir / f"h06_{name}_induction_attn.png"); plt.close(fig)
    save_raw(HYP.lower(), name, "attn", induction=ind_att, prev=prev_att, layers=np.array(layers))
    del model; torch.cuda.empty_cache()
    return m2, top_ind


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(registry.MODELS))
    ap.add_argument("--nulls", type=int, default=50, help="N3 / N1 draws for the previous-token score")
    ap.add_argument("--kc-nulls", type=int, default=20, help="N1 draws for K-composition and copying")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/tables/h06.csv"))
    ap.add_argument("--n-seq", type=int, default=20)
    ap.add_argument("--period", type=int, default=50)
    ap.add_argument("--no-confirm", action="store_true")
    args = ap.parse_args()
    figdir = ROOT / "results/figures/h06"; figdir.mkdir(parents=True, exist_ok=True)
    with ResultsWriter(args.out) as W:
        for name in args.models:
            t0 = time.time()
            print(f"== {name}")
            w, cfg, fam = load_canonical(name)
            res = run_weights(name, w, cfg, fam, args, W, figdir)
            del w
            if not args.no_confirm:
                run_confirm(name, fam, cfg, res, args, W, figdir)
            print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
