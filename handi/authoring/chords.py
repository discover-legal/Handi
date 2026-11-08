"""Rung 4, "chords": conjunction neurons authored from corpus trigram statistics on top of the A8 direct path
(decided 2026-11-08). Pythia (GPT-NeoX: untied, RoPE, parallel residual) only. No gradient step.

Residual layout (d = B + K + K, K = head_dim):
    dims [0, B)          bigram subspace: rank-B factorisation of the smoothed log-bigram table (as A8, rank B)
    dims [B, B+K)        "current token" code slot: a fixed random +-1/sqrt(K) code per token, part of the embedding
    dims [B+K, B+2K)     "previous token" code slot: written by one previous-token head at layer 0 (RoPE recipe of
                         handi/authoring/circuits.py) that copies the current-code slot of position t-1
Every other attention head and every other write is zero. The unembedding reads the bigram dims only.

Neuron j in the MLP of layer l >= 1, for the trigram context (a, b) (b = current token, a = previous):
    pre-activation = code_b . x[cur slot] / m + code_a . x[prev slot] / m - 1.5      (x = the layer's LN2 output;
                     m = the measured magnitude of a true code in that slot, so a true pair scores ~2, a half match ~1)
    down column    = sigma_f * pinv(U_b) r_ab / gelu(0.5) * s
    r_ab[c]        = log((1 - l3 + l3 n(abc) / (n(ab) p2(c|b))) / (1 - l3)) for n(abc) > 0, else 0
                     (= log p3(c|a,b) - log p2(c|b) up to a constant that the softmax drops), p2 = the A8 bigram,
                     l3 = 0.5; sigma_f = the median final-norm scale of a token row; s = a global scale picked on
                     the calibration slice from {0.5, 1, 2}
Contexts: the (L-1) * d_mlp most frequent (a, b) with n(ab) >= 3. Builder defaults in docs/decisions.md (Rung 4).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from . import bigram
from .circuits import _rotary

RAW = Path(__file__).resolve().parents[2] / "results" / "raw" / "trigram"
L3 = 0.5
MIN_CTX = 3
DETECT_GAIN = 8.0                      # detector scale: true pair -> +4, half match -> -4, no match -> -12 (gelu ~ 0)
BIAS = -1.5 * DETECT_GAIN
GELU_HALF = (0.5 * DETECT_GAIN) * 0.5 * (1.0 + math.erf((0.5 * DETECT_GAIN) / math.sqrt(2.0)))   # gelu at a true pair
RIDGE = 1e-2                           # weighted least squares ridge, relative to the mean diagonal
SCALES = (0.05, 0.1, 0.25, 0.5, 1.0)
G1_GRID = (3.0, 10.0, 30.0, 100.0)
CALIB_TOKENS = 20_000


# ----------------------------------------------------------------------------- trigram counts and references
def trigram_counts(model: str, V: int) -> dict:
    """ctx_a, ctx_b, ctx_n: contexts (a, b) with counts; tri_ctx (index into contexts), tri_c, tri_n: triples."""
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{model}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["V"]) == V:
            return {k: z[k] for k in ("ctx_a", "ctx_b", "ctx_n", "tri_ctx", "tri_c", "tri_n")}
    train = bigram._tokens(model, "train", bigram.TRAIN_CHARS).astype(np.int64)
    key3 = (train[:-2] * V + train[1:-1]) * V + train[2:]
    u3, n3 = np.unique(key3, return_counts=True)
    ctx_key = u3 // V
    u2, inv2 = np.unique(ctx_key, return_inverse=True)
    n2 = np.bincount(inv2, weights=n3).astype(np.int64)
    out = dict(ctx_a=(u2 // V).astype(np.int64), ctx_b=(u2 % V).astype(np.int64), ctx_n=n2,
               tri_ctx=inv2.astype(np.int64), tri_c=(u3 % V).astype(np.int64), tri_n=n3.astype(np.int64))
    np.savez(f, V=V, **out)
    return out


def _p2_matrix(c: dict, V: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Interpolated bigram p2(c|b) as (sparse part csr [V, V], unigram p) so that p2 = sparse[b, c] + (1 - lam) p[c]."""
    uni = c["uni"].astype(np.float64) + bigram.ALPHA
    p = uni / uni.sum()
    ctx = np.maximum(c["uni"].astype(np.float64), 1.0)
    M = sp.csr_matrix((bigram.LAMBDA * c["vals"] / ctx[c["rows"]], (c["rows"], c["cols"])), shape=(V, V))
    return M, (1 - bigram.LAMBDA) * p


def trigram_reference(c: dict, t: dict, V: int, selected: np.ndarray | None = None) -> float:
    """Test cross-entropy of p3 = l3 n(abc)/n(ab) + (1 - l3) p2(c|b) on contexts with n(ab) >= MIN_CTX
    (restricted to `selected` context indices if given), p2 elsewhere."""
    M2, base = _p2_matrix(c, V)
    test = c["test"].astype(np.int64)
    a, b, cc = test[:-2], test[1:-1], test[2:]
    p2 = np.asarray(M2[b, cc]).ravel() + base[cc]
    ctx_lookup = sp.csr_matrix((np.arange(1, len(t["ctx_a"]) + 1), (t["ctx_a"], t["ctx_b"])), shape=(V, V))
    idx = np.asarray(ctx_lookup[a, b]).ravel() - 1                 # -1 where the context was never seen
    use = idx >= 0
    if selected is not None:
        sel = np.zeros(len(t["ctx_a"]), bool); sel[selected] = True
        use &= sel[np.maximum(idx, 0)]
    use &= t["ctx_n"][np.maximum(idx, 0)] >= MIN_CTX
    T3 = sp.csr_matrix((t["tri_n"].astype(np.float64), (t["tri_ctx"], t["tri_c"])), shape=(len(t["ctx_a"]), V))
    n3 = np.where(use, np.asarray(T3[np.maximum(idx, 0), cc]).ravel(), 0.0)
    nab = np.where(use, t["ctx_n"][np.maximum(idx, 0)], 1.0)
    p3 = np.where(use, L3 * n3 / nab + (1 - L3) * p2, p2)
    # first pair of the test slice is scored by the bigram only (no context); include it for comparability with M1
    first = -np.log(np.asarray(M2[test[0], test[1]]).ravel()[0] + base[test[1]])
    return float((first + (-np.log(p3)).sum()) / (len(test) - 1))


# ----------------------------------------------------------------------------- pieces
def _rope(x: np.ndarray, pos: np.ndarray, inv_freq: np.ndarray, rot: int) -> np.ndarray:
    """GPT-NeoX rotary on the first `rot` dims of x [hd] for every position in pos -> [len(pos), hd]."""
    out = np.repeat(x[None, :], len(pos), 0).astype(np.float64)
    half = rot // 2
    th = pos[:, None] * inv_freq[None, :]
    x1, x2 = x[:half], x[half:rot]
    out[:, :half] = x1 * np.cos(th) - x2 * np.sin(th)
    out[:, half:rot] = x2 * np.cos(th) + x1 * np.sin(th)
    return out


def prev_head_biases(cfg: dict, hd: int, n_pos: int = 512) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(bq, bk, g1, mean attention mass on offset -1) for the RoPE previous-token head (circuits.py recipe);
    g1 = the smallest grid value with mass >= 0.95, else the best."""
    rot, inv = _rotary(cfg, "gpt_neox", hd)
    half = rot // 2
    pos = np.arange(n_pos)
    best = None
    for g1 in G1_GRID:
        bq = np.zeros(hd); bk = np.zeros(hd)
        bq[:half], bq[half:rot] = g1 * np.cos(-inv), g1 * np.sin(-inv)
        bk[:half] = g1
        Q, K = _rope(bq, pos, inv, rot), _rope(bk, pos, inv, rot)
        S = Q @ K.T / math.sqrt(hd)
        S[np.triu_indices(n_pos, 1)] = -np.inf
        P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        mass = float(np.mean([P[i, i - 1] for i in range(1, n_pos)]))
        if best is None or mass > best[3]:
            best = (bq.astype(np.float32), bk.astype(np.float32), g1, mass)
        if mass >= 0.95:
            return bq.astype(np.float32), bk.astype(np.float32), g1, mass
    return best


# ----------------------------------------------------------------------------- the builder
def build_chords(canon: dict, cfg: dict, fam: str, seed: int, model: str, report: dict | None, neurons: bool = True):
    """A9 (neurons=True) / A9C control (neurons=False). Returns the canonical weight dict, or None if not GPT-NeoX."""
    if fam != "gpt_neox":
        return None
    from ..analysis.loader import n_layers
    V, d = canon["embed.tok"].shape
    L = n_layers(cfg, fam)
    H = cfg["num_attention_heads"]; hd = d // H; K = hd; B = d - 2 * K
    d_mlp = canon["layers.0.mlp.up.w"].shape[0]
    rng = np.random.default_rng([seed, 9])
    rep = {} if report is None else report

    # ---- direct path at rank B (A8 recipe, untied)
    c = bigram.counts(model, V)
    E_b, U_b, fit, _ = bigram.factorise(c, V, B, tied=False, seed=0)
    E_b = E_b - E_b.mean(axis=1, keepdims=True)
    r0 = float(np.median(np.linalg.norm(E_b, axis=1)))
    E_b = (E_b * (r0 / (np.linalg.norm(E_b, axis=1, keepdims=True) + 1e-9))).astype(np.float32)   # constant row norm:
    # the final LN divides each row by its own scale, so the direct path is unchanged, and every code slot now has
    # the same magnitude after every LN (the detector's operating point is token-independent)
    code = (rng.integers(0, 2, size=(V, K)) * 2 - 1).astype(np.float32) / math.sqrt(K)
    c_code = r0
    E = np.zeros((V, d), np.float32); E[:, :B] = E_b; E[:, B:B + K] = c_code * code
    w = {k: np.zeros_like(v) for k, v in canon.items()}
    w["embed.tok"] = E
    U = np.zeros((V, d), np.float32); U[:, :B] = U_b
    w["unembed.w"] = U
    w["ln_f.w"] = np.zeros(d, np.float32); w["ln_f.w"][:B] = 1.0
    uni = c["uni"].astype(np.float64) + bigram.ALPHA
    logp = np.log(uni / uni.sum())
    bvec, *_ = np.linalg.lstsq(U_b.astype(np.float64), logp - logp.mean(), rcond=None)
    w["ln_f.b"] = np.zeros(d, np.float32); w["ln_f.b"][:B] = bvec
    for l in range(L):
        w[f"layers.{l}.ln1.w"] = np.ones(d, np.float32); w[f"layers.{l}.ln2.w"] = np.ones(d, np.float32)
    T, ce_direct = bigram.direct_path_ce(w, fam, c["calib"], c["test"])
    w["unembed.w"] = (w["unembed.w"] / T).astype(np.float32)
    rep.update(a9_rank=B, a9_fit_ev=fit, a9_temperature=T, a9_direct_ce=ce_direct, a9_code_scale=c_code)

    # ---- previous-token head at layer 0, head 0: copies the current-code slot of t-1 into the previous-code slot
    sigma0 = math.sqrt((r0 ** 2 + c_code ** 2) / d)
    bq, bk, g1, mass = prev_head_biases(cfg, hd)
    w["layers.0.attn.q.b"][0:hd] = bq
    w["layers.0.attn.k.b"][0:hd] = bk
    v = np.zeros((hd, d), np.float32); v[np.arange(K), B + np.arange(K)] = 1.0          # read the current-code slot
    w["layers.0.attn.v.w"][0:hd] = v
    o = np.zeros((d, hd), np.float32); o[B + K + np.arange(K), np.arange(K)] = sigma0     # write it as a previous code
    w["layers.0.attn.o.w"][:, 0:hd] = o
    rep.update(a9_prev_gain=g1, a9_prev_mass=mass)
    if not neurons:
        rep.update(a9_neurons=0)
        return w

    # ---- contexts and neurons
    t = trigram_counts(model, V)
    n_neur = (L - 1) * d_mlp
    order = np.argsort(t["ctx_n"])[::-1]
    order = order[t["ctx_n"][order] >= MIN_CTX][:n_neur]
    n_neur = len(order)
    M2, base = _p2_matrix(c, V)
    T3 = sp.csr_matrix((t["tri_n"].astype(np.float64), (t["tri_ctx"], t["tri_c"])), shape=(len(t["ctx_a"]), V))
    a_sel, b_sel = t["ctx_a"][order], t["ctx_b"][order]
    nab = t["ctx_n"][order].astype(np.float64)
    # target: r_ab = log p3(. | a, b) - log q(. | b), q = what the model's own direct path says after token b
    # (so the neurons also repair the rank loss on the covered contexts); written through pinv(U_b)
    dev = bigram.DEVICE
    U_inst = torch.as_tensor(w["unembed.w"][:, :B], device=dev)
    Eb_t = torch.as_tensor(E_b, device=dev)
    gf = torch.as_tensor(w["ln_f.w"][:B], device=dev); bf = torch.as_tensor(w["ln_f.b"][:B], device=dev)
    base_t = torch.as_tensor(base, device=dev, dtype=torch.float32)
    Vcorr = np.zeros((n_neur, B), np.float32)
    for s0 in range(0, n_neur, 256):
        sl = slice(s0, min(s0 + 256, n_neur))
        bb = torch.as_tensor(b_sel[sl], device=dev)
        hrow = torch.cat([Eb_t[bb], torch.zeros(len(b_sel[sl]), d - B, device=dev)], 1)
        hn = bigram.final_norm(hrow, w, fam)[:, :B]
        logq = torch.log_softmax(hn @ U_inst.T, -1)                      # [n, V] the direct path after b
        blk = T3[order[sl]].toarray()
        p3 = torch.as_tensor(L3 * blk / nab[sl][:, None], device=dev, dtype=torch.float32) \
             + (1 - L3) * (torch.as_tensor(M2[b_sel[sl]].toarray(), device=dev, dtype=torch.float32) + base_t[None, :])
        r = torch.log(p3) - logq
        # weighted least squares per context: min_D sum_c p3(c) (u_c . D - r_c)^2 + ridge, so the B dims go where
        # the continuation mass is (the plain pseudo-inverse spends them on the tail)
        for i in range(r.shape[0]):
            wgt = p3[i] + 1e-6
            r_i = r[i] - (wgt * r[i]).sum() / wgt.sum()
            Uw = U_inst * wgt[:, None]
            A = U_inst.T @ Uw
            A = A + RIDGE * A.diagonal().mean() * torch.eye(B, device=dev)
            Vcorr[s0 + i] = torch.linalg.solve(A, Uw.T @ r_i).cpu().numpy()
        del logq, p3, r
    sigma_f = math.sqrt((r0 ** 2 + 2 * c_code ** 2) / d)
    m = c_code / sigma_f                                                  # magnitude of a true unit code after LN2
    up = np.zeros((n_neur, d), np.float32)
    up[:, B:B + K] = DETECT_GAIN * code[b_sel] / m
    up[:, B + K:B + 2 * K] = DETECT_GAIN * code[a_sel] / m
    down = (sigma_f * Vcorr / GELU_HALF).astype(np.float32)               # [n_neur, B]
    for l in range(1, L):
        s0, s1 = (l - 1) * d_mlp, min(l * d_mlp, n_neur)
        if s0 >= n_neur:
            break
        w[f"layers.{l}.mlp.up.w"][: s1 - s0] = up[s0:s1]
        w[f"layers.{l}.mlp.up.b"][: s1 - s0] = BIAS
        w[f"layers.{l}.mlp.down.w"][:B, : s1 - s0] = down[s0:s1].T
    # coverage and the achievable (capped) trigram line
    ctx_lookup = sp.csr_matrix((np.arange(1, len(t["ctx_a"]) + 1), (t["ctx_a"], t["ctx_b"])), shape=(V, V))
    cal = c["calib"].astype(np.int64)
    idx = np.asarray(ctx_lookup[cal[:-2], cal[1:-1]]).ravel() - 1
    sel = np.zeros(len(t["ctx_a"]), bool); sel[order] = True
    coverage = float(np.mean((idx >= 0) & sel[np.maximum(idx, 0)]))
    rep.update(a9_neurons=n_neur, a9_coverage_calib=coverage, a9_detector_m=m, a9_sigma_f=sigma_f,
               a9_ref_trigram=trigram_reference(c, t, V), a9_ref_trigram_capped=trigram_reference(c, t, V, order),
               a9_ref_bigram=bigram.references(c, V)["bigram"])
    w["_a9_scale_grid"] = np.array(SCALES, np.float32)                    # consumed by calibrate_scale
    return w


@torch.no_grad()
def calibrate(w: dict, cfg: dict, fam: str, model_name: str, report: dict | None) -> dict:
    """Empirical checks and the global down-scale on the calibration slice (closed-form grid). Also measures the
    neuron pre-activations: true-context mean / sd and the false-firing rate, and rescales the detector so a true
    pair scores 2.0 (the designed value) if the measured mean differs."""
    from .evaluate import load_authored
    from ..analysis.loader import n_layers
    grid = w.pop("_a9_scale_grid", None)
    rep = {} if report is None else report
    if grid is None:
        return w
    V, d = w["embed.tok"].shape
    L = n_layers(cfg, fam)
    c = bigram.counts(model_name, V)
    t = trigram_counts(model_name, V)
    # calibration on the WikiText validation split: the train slice used by A8 lies inside the counted text, and the
    # trigram counts would make an in-sample scale choice a memorisation choice
    cal = bigram._tokens(model_name, "validation", 400_000)[:CALIB_TOKENS + 1].astype(np.int64)
    H = cfg["num_attention_heads"]; hd = d // H; K = hd; B = d - 2 * K
    model, _ = load_authored(model_name, w)
    # ---- pre-activation check at layer 1 (the first neuron layer)
    captured = {}
    h = model.gpt_neox.layers[1].post_attention_layernorm.register_forward_hook(lambda m, i, o: captured.__setitem__("x", o))
    x = torch.as_tensor(cal[:1024][None, :], device=next(model.parameters()).device)
    model(x)
    h.remove()
    X = captured["x"][0].float().cpu().numpy()                               # [1024, d]
    up = w["layers.1.mlp.up.w"]; nb = int((np.abs(up).sum(1) > 0).sum())
    pre = X @ up[:nb].T                                                      # [1024, nb], before bias
    # which neuron (if any) is the true one at each position: context (prev, cur)
    ctx_lookup = sp.csr_matrix((np.arange(1, len(t["ctx_a"]) + 1), (t["ctx_a"], t["ctx_b"])), shape=(V, V))
    order = np.argsort(t["ctx_n"])[::-1]; order = order[t["ctx_n"][order] >= MIN_CTX]
    pos_of_ctx = {int(ci): j for j, ci in enumerate(order[:nb])}            # layer-1 neurons = the first d_mlp contexts
    idx = np.asarray(ctx_lookup[cal[:1023], cal[1:1024]]).ravel() - 1
    true_vals, false_max = [], []
    for p in range(1, 1024):
        j = pos_of_ctx.get(int(idx[p - 1]), None)
        row = pre[p].copy()
        if j is not None:
            true_vals.append(row[j]); row[j] = -np.inf
        false_max.append(row.max())
    true_mean = float(np.mean(true_vals)) if true_vals else float("nan")
    rep.update(a9_true_pre_mean=true_mean, a9_true_pre_sd=float(np.std(true_vals)) if true_vals else float("nan"),
               a9_false_pre_p999=float(np.quantile(false_max, 0.999)), a9_true_n=len(true_vals),
               a9_false_fire_frac=float(np.mean(np.array(false_max) + BIAS > 0)))
    del model; torch.cuda.empty_cache()
    # ---- global down scale by calibration cross-entropy
    from .evaluate import cross_entropy
    best = None
    base_down = {l: w[f"layers.{l}.mlp.down.w"].copy() for l in range(1, L)}
    for s in grid:
        for l in range(1, L):
            w[f"layers.{l}.mlp.down.w"] = (base_down[l] * s).astype(np.float32)
        model, _ = load_authored(model_name, w)
        ce = cross_entropy(model, cal, n_tokens=CALIB_TOKENS)[0]
        rep[f"a9_calib_ce_scale_{s:g}"] = ce
        if best is None or ce < best[1]:
            best = (s, ce)
        del model; torch.cuda.empty_cache()
    for l in range(1, L):
        w[f"layers.{l}.mlp.down.w"] = (base_down[l] * best[0]).astype(np.float32)
    rep.update(a9_scale=best[0], a9_calib_ce=best[1])
    return w


def a9_chords(canon, cfg, fam, seed, model=None, report=None):
    w = build_chords(canon, cfg, fam, seed, model, report, neurons=True)
    return None if w is None else calibrate(w, cfg, fam, model, report)


def a9c_control(canon, cfg, fam, seed, model=None, report=None):
    return build_chords(canon, cfg, fam, seed, model, report, neurons=False)
