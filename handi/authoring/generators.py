"""Rung 2 weight generators: each condition returns a full canonical weight dict for a model shape.

A0  fresh init by the family's init rule (control).
A1  spectrum-matched random: every 2-D matrix = U diag(sigma_trained) V^T with Haar U, V;
    token embeddings covariance-matched (N6) with random token assignment; 1-D vectors (norm gains,
    biases) keep their trained marginal under a random permutation (control; expected == A0).
A2  A1 + ordered-axis patterns (H10 smooth positions, H5 outlier dims, H4 frequency direction).
A3  A2 + cross-layer subspace alignment (H7) and QK/OV identity components (H3).
A4  A3 + explicit previous-token / induction circuits where H6 finds them.
A5  trained weights (upper bound); A5' / A5'' are produced by analysis/h02_lowrank_function.py.
Conditions A2-A4 take their parameters from the confirmed-pattern tables and are filled in only
after those hypotheses are evaluated (see docs/decisions.md for what was confirmed).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch

from ..analysis.nulls import n4_fresh_init, n5_spectrum_matched, n6_cov_matched

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _haar(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    g = torch.randn(m, k, generator=gen).to(DEVICE)
    q, r = torch.linalg.qr(g)
    return q * torch.sign(torch.diagonal(r))[None, :]


def spectrum_matched_gpu(W: np.ndarray, gen: torch.Generator) -> np.ndarray:
    """U diag(sigma_W) V^T with Haar U, V, computed on the GPU (Gram path for very tall matrices)."""
    T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).to(DEVICE)
    m, n = T.shape
    if max(m, n) / min(m, n) >= 16:
        G = (T.T @ T) if m >= n else (T @ T.T)
        s = torch.linalg.eigvalsh(G.double()).flip(0).clamp_min(0).sqrt().float()
    else:
        s = torch.linalg.svdvals(T)
    k = len(s)
    out = (_haar(m, k, gen) * s[None, :]) @ _haar(n, k, gen).T
    return out.cpu().numpy()


def a0_fresh(canon: dict[str, np.ndarray], cfg: dict, fam: str, seed: int, **_) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    fresh = n4_fresh_init(fam, cfg, rng)
    out = dict(canon)
    for k in canon:
        if k in fresh and fresh[k].shape == canon[k].shape:
            out[k] = fresh[k].astype(np.float32)
        elif canon[k].ndim == 1:
            # norm gains / biases the init table does not cover: family init value
            out[k] = _init_vector(k, canon[k], fam)
        else:
            out[k] = (rng.standard_normal(canon[k].shape) * 0.02).astype(np.float32)
    return out


def _init_vector(name: str, v: np.ndarray, fam: str) -> np.ndarray:
    if name.endswith(".b"):
        return np.zeros_like(v)
    if fam == "qwen3_5":
        return np.zeros_like(v)            # zero-centered RMSNorm: weight 0 means gain 1
    return np.ones_like(v)                 # LayerNorm gain 1


def a1_spectrum(canon: dict[str, np.ndarray], cfg: dict, fam: str, seed: int, **_) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    out = {}
    if couple_unembed and "unembed.w" in canon:
        # coupled N6: one Gaussian draw shared by embed and unembed, so u_t is a fixed linear map of e_t
        V = canon["embed.tok"].shape[0]
        Z = rng.standard_normal((V, d))
        out["embed.tok"] = _n6_from_z(canon["embed.tok"], Z)
        out["unembed.w"] = _n6_from_z(canon["unembed.w"], Z)
    for k, W in canon.items():
        if k in out:
            continue
        if k in ("embed.tok", "unembed.w"):
            out[k] = n6_cov_matched(W, rng).astype(np.float32)
        elif W.ndim == 2:
            out[k] = spectrum_matched_gpu(W, gen)
        elif W.ndim == 1:
            out[k] = rng.permutation(W).astype(np.float32)
        else:
            out[k] = rng.permutation(W.reshape(-1)).reshape(W.shape).astype(np.float32)
    if fam == "qwen3_5" and "unembed.w" not in canon:
        pass   # tied: the HF class ties lm_head to embed.tok after loading
    return out



# --------------------------------------------------------------------------------------------
# A2 / A2F / A3: confirmed-pattern conditions (docs/decisions.md "Rung 2 builder defaults")
# --------------------------------------------------------------------------------------------

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
K_ALIGN = 32          # subspace dimension at which H7 overlap was measured (grassmann_mean_k32)
READER_SUFFIX = (".attn.q.w", ".attn.k.w", ".attn.v.w", ".attn.gate.w", ".mlp.up.w", ".mlp.gate.w",
                 ".linear.in_qkv.w", ".linear.in_z.w", ".linear.in_b.w", ".linear.in_a.w")
WRITER_SUFFIX = (".attn.o.w", ".mlp.down.w", ".linear.out.w")
RESIDUAL_VECTORS = (".ln1.w", ".ln1.b", ".ln2.w", ".ln2.b", ".attn.o.b", ".mlp.down.b")


def _d_model(cfg: dict, fam: str) -> int:
    return cfg.get("n_embd") or cfg.get("hidden_size")


def _heads(cfg: dict, fam: str, d: int) -> tuple[int, int, int]:
    if fam == "qwen3_5":
        return cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    H = cfg.get("n_head") or cfg.get("num_heads") or cfg.get("num_attention_heads")
    return H, H, d // H


def _is_reader(k: str) -> bool:
    return k.endswith(READER_SUFFIX)


def _is_writer(k: str) -> bool:
    return k.endswith(WRITER_SUFFIX)


def _is_residual_vector(k: str) -> bool:
    return k.endswith(RESIDUAL_VECTORS) or k in ("ln_f.w", "ln_f.b")


def _svals(T: torch.Tensor) -> torch.Tensor:
    m, n = T.shape
    if max(m, n) / min(m, n) >= 16:
        G = (T.T @ T) if m >= n else (T @ T.T)
        return torch.linalg.eigvalsh(G.double()).flip(0).clamp_min(0).sqrt().float()
    return torch.linalg.svdvals(T)


def _orth_complement_haar(V0: torch.Tensor, r: int, gen: torch.Generator) -> torch.Tensor:
    """Haar r columns orthogonal to the columns of V0 [d, k0]."""
    d = V0.shape[0]
    if r == 0:
        return torch.zeros(d, 0, device=DEVICE)
    Z = torch.randn(d, r, generator=gen).to(DEVICE)
    Z = Z - V0 @ (V0.T @ Z)
    q, rr = torch.linalg.qr(Z)
    return q * torch.sign(torch.diagonal(rr))[None, :]


def _aligned_basis(d: int, k_full: int, shared: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """[d, k_full] orthonormal: first min(K_ALIGN, k_full) columns Haar inside span(shared), rest in the complement."""
    k0 = min(K_ALIGN, k_full)
    m = shared.shape[1]
    V0 = shared @ _haar(m, k0, gen)
    V1 = _orth_complement_haar(V0, k_full - k0, gen)
    return torch.cat([V0, V1], dim=1)


def _h07_overlap(model: str) -> float:
    import csv
    with open(TABLES / "h07.csv") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == model and r["metric"] == "grassmann_mean_k32" and r["component"] == "readers:all_crosslayer:full":
                return float(r["value"])
    raise KeyError(f"{model}: readers:all_crosslayer:full not in h07.csv")


def _h03_cos(model: str) -> dict[tuple[int, int, str], float]:
    """(layer, head, 'QK'|'OV') -> measured cos_F(W, I) from h03.csv."""
    import csv
    out = {}
    with open(TABLES / "h03.csv") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == model and r["metric"] == "cos_identity" and r["layer"] not in ("", "ALL"):
                m = re.fullmatch(r"attn\.h(\d+)\.(QK|OV)", r["component"])
                if m:
                    out[(int(r["layer"]), int(m.group(1)), m.group(2))] = float(r["value"])
    if not out:
        raise KeyError(f"{model}: no per-head cos_identity rows in h03.csv")
    return out


def _phase_randomized_positions(P: np.ndarray, gen: torch.Generator) -> np.ndarray:
    """W_pos with the trained PC spectrum and per-PC |FFT| profile along position, random phases,
    random residual directions (H10 pattern without trained directions)."""
    T = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float32)).to(DEVICE)
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)        # U [n, d], S [d]
    n = U.shape[0]
    F = torch.fft.rfft(U, dim=0)
    ph = torch.rand(F.shape, generator=gen).to(DEVICE) * 2 * math.pi
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    U2 = torch.fft.irfft(F.abs() * torch.exp(1j * ph), n=n, dim=0)
    # No re-orthonormalization: with ~d random-phase columns of length n the Gram matrix is far from the
    # identity, and any orthonormalization (QR or Loewdin) mixes high-frequency content into the
    # leading PCs and destroys the smoothness (GPT-2 neighbour ratio 0.04 -> 0.12..0.27). Keeping the
    # columns as they are preserves each PC's increment energy exactly; the singular values are then
    # only approximately the trained ones (reported by the smoke test).
    V2 = _haar(T.shape[1], T.shape[1], gen)
    return ((U2 * S[None, :]) @ V2.T).cpu().numpy()


def _rank_assign_pc1(E_new: np.ndarray, E_trained: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give token t the PC1 coordinate whose rank matches the smoothed trained PC1-vs-token-id curve (H4)."""
    V, d = E_new.shape
    mu = E_new.mean(axis=0)
    X = E_new - mu
    C = X.T @ X / (V - 1)
    w, U = np.linalg.eigh(C)
    u = U[:, -1]
    if u @ mu < 0:                                                # orient PC1 along the common (mean) direction
        u = -u
    a = X @ u
    mu_t = E_trained.mean(axis=0)
    Xt = E_trained - mu_t
    Ct = Xt.T @ Xt / (V - 1)
    wt, Ut = np.linalg.eigh(Ct)
    ut = Ut[:, -1]
    if ut @ mu_t < 0:
        ut = -ut
    c = Xt @ ut
    win = max(50, V // 200)
    c_s = np.convolve(c, np.ones(win) / win, mode="same")
    ids = np.arange(V)
    rho_t = _spearman(c, ids)                                     # measured correlation to reproduce
    a_sorted = np.sort(a)
    noise = rng.standard_normal(V) * c_s.std()
    lo, hi = 0.0, 50.0
    for _ in range(14):                                            # noise level so that spearman(a_new, id) ~ rho_t
        sig = 0.5 * (lo + hi)
        order = np.argsort(c_s + sig * noise, kind="stable")
        a_new = np.empty_like(a)
        a_new[order] = a_sorted
        rho = _spearman(a_new, ids)
        if abs(rho - rho_t) < 0.01:
            break
        if abs(rho) > abs(rho_t):
            lo = sig
        else:
            hi = sig
    return (E_new + np.outer(a_new - a, u)).astype(np.float32)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _identity_component_heads(out: dict, canon: dict, cfg: dict, fam: str, layer: int, cos: dict,
                              gen: torch.Generator) -> dict[str, float]:
    """Rebuild Q heads from K heads and O heads from V heads so that each head's W_QK and W_OV have the
    measured cos_F against the identity (H3). Singular values of Q and O come from the trained blocks (N5)."""
    d = _d_model(cfg, fam)
    H, Hkv, hd = _heads(cfg, fam, d)
    rep = H // Hkv
    pre = f"layers.{layer}.attn."
    q = torch.from_numpy(out[pre + "q.w"]).to(DEVICE).clone()
    k = torch.from_numpy(out[pre + "k.w"]).to(DEVICE)
    v = torch.from_numpy(out[pre + "v.w"]).to(DEVICE)
    o = torch.from_numpy(out[pre + "o.w"]).to(DEVICE).clone()
    qT = torch.from_numpy(canon[pre + "q.w"]).to(DEVICE)
    oT = torch.from_numpy(canon[pre + "o.w"]).to(DEVICE)
    err = []
    sqd = math.sqrt(d)

    def mix(base: torch.Tensor, s_new: torch.Tensor, s_base: torch.Tensor, target: float):
        """Return orthonormal [d, hd] columns aligned to `base` so that the product has cos_F ~ target."""
        prod = s_new * s_base
        beta0 = target * sqd * float(prod.norm()) / float(prod.sum())
        R = _haar(d, hd, gen)

        def build(beta):
            beta = max(-1.0, min(1.0, beta))
            M = beta * base + math.sqrt(1 - beta * beta) * R
            qq, rr = torch.linalg.qr(M)
            return qq * torch.sign(torch.diagonal(rr))[None, :]

        def achieved(Vn):
            # cos_F(V_n S_new S_base base^T, I) = sum_i prod_i (v_n,i . base_i) / (sqrt(d) ||prod||)
            return float((prod * (Vn * base).sum(0)).sum() / (sqd * prod.norm()))

        lo, hi = -1.0, 1.0
        beta = beta0
        Vn = build(beta)
        for _ in range(12):
            got = achieved(Vn)
            if abs(got - target) < 1e-3:
                break
            if got < target:
                lo = beta
            else:
                hi = beta
            beta = 0.5 * (lo + hi)
            Vn = build(beta)
        return Vn, achieved(Vn)

    for h in range(H):
        kv = h // rep
        Kb = k[kv * hd:(kv + 1) * hd]                       # [hd, d]
        Uk, Sk, Vkh = torch.linalg.svd(Kb, full_matrices=False)   # Uk [hd,hd], Vk = Vkh.T [d,hd]
        Vk = Vkh.T
        Sq = torch.linalg.svdvals(qT[h * hd:(h + 1) * hd])
        Vq, got = mix(Vk, Sq, Sk, cos.get((layer, h, "QK"), 0.0))
        q[h * hd:(h + 1) * hd] = (Uk * Sq[None, :]) @ Vq.T
        err.append(abs(got - cos.get((layer, h, "QK"), 0.0)))

        Vb = v[kv * hd:(kv + 1) * hd]                       # [hd, d]
        Uv, Sv, Vvh = torch.linalg.svd(Vb, full_matrices=False)   # W_V = Vv Sv Uv^T
        Vv = Vvh.T
        So = torch.linalg.svdvals(oT[:, h * hd:(h + 1) * hd])
        Uo, got = mix(Vv, So, Sv, cos.get((layer, h, "OV"), 0.0))
        o[:, h * hd:(h + 1) * hd] = (Uo * So[None, :]) @ Uv.T
        err.append(abs(got - cos.get((layer, h, "OV"), 0.0)))
    out[pre + "q.w"] = q.cpu().numpy()
    out[pre + "o.w"] = o.cpu().numpy()
    return {"mean_abs_cos_err": float(np.mean(err)), "max_abs_cos_err": float(np.max(err))}


def build(canon: dict[str, np.ndarray], cfg: dict, fam: str, seed: int, *, model: str | None = None,
          ordered: bool = False, token_rank: bool = False, align: bool = False, identity: bool = False,
          couple_unembed: bool = False, report: dict | None = None) -> dict[str, np.ndarray]:
    """A1 core with optional confirmed patterns: ordered (H10 + H5), token_rank (H4), align (H7),
    identity (H3). A2 = ordered; A2F = ordered + token_rank; A3 = ordered + align + identity."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    d = _d_model(cfg, fam)
    shared = None
    if align:
        if model is None:
            raise ValueError("align needs the model name to read h07.csv")
        ov = _h07_overlap(model)
        m = int(np.clip(round(K_ALIGN / ov), K_ALIGN, d))
        shared = _haar(d, m, gen)
        if report is not None:
            report["shared_dim"] = m
    out = {}
    if couple_unembed and "unembed.w" in canon:
        # coupled N6: one Gaussian draw shared by embed and unembed, so u_t is a fixed linear map of e_t
        V = canon["embed.tok"].shape[0]
        Z = rng.standard_normal((V, d))
        out["embed.tok"] = _n6_from_z(canon["embed.tok"], Z)
        out["unembed.w"] = _n6_from_z(canon["unembed.w"], Z)
    for k, W in canon.items():
        if k in out:
            continue
        if k in ("embed.tok", "unembed.w"):
            out[k] = n6_cov_matched(W, rng).astype(np.float32)
        elif k == "embed.pos" and ordered:
            out[k] = _phase_randomized_positions(W, gen)
        elif W.ndim == 2:
            if align and (_is_reader(k) or _is_writer(k)):
                T = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32)).to(DEVICE)
                mm, nn = T.shape
                s = _svals(T)
                kf = len(s)
                if _is_reader(k):
                    V = _aligned_basis(nn, kf, shared, gen)
                    U = _haar(mm, kf, gen)
                else:
                    U = _aligned_basis(mm, kf, shared, gen)
                    V = _haar(nn, kf, gen)
                out[k] = ((U * s[None, :]) @ V.T).cpu().numpy()
            else:
                out[k] = spectrum_matched_gpu(W, gen)
        elif W.ndim == 1:
            out[k] = rng.permutation(W).astype(np.float32)
        else:
            out[k] = rng.permutation(W.reshape(-1)).reshape(W.shape).astype(np.float32)

    if identity:
        if model is None:
            raise ValueError("identity needs the model name to read h03.csv")
        cos = _h03_cos(model)
        from ..analysis.loader import attn_layers
        errs = []
        for layer in attn_layers(cfg, fam):
            errs.append(_identity_component_heads(out, canon, cfg, fam, layer, cos, gen))
        if report is not None:
            report["cos_mean_abs_err"] = float(np.mean([e["mean_abs_cos_err"] for e in errs]))
            report["cos_max_abs_err"] = float(np.max([e["max_abs_cos_err"] for e in errs]))

    if ordered:
        # H5: one random placement of residual dims shared by every layer and matrix; magnitudes from the trained model
        perm = rng.permutation(d)
        inv = np.argsort(perm)
        for k, W in canon.items():
            if k in ("embed.tok", "unembed.w"):
                out[k] = np.ascontiguousarray(out[k][:, perm])
            elif W.ndim == 2 and _is_reader(k):
                target = np.linalg.norm(W, axis=0)[inv]
                cur = np.linalg.norm(out[k], axis=0) + 1e-12
                out[k] = (out[k] * (target / cur)[None, :]).astype(np.float32)
            elif W.ndim == 2 and _is_writer(k):
                target = np.linalg.norm(W, axis=1)[inv]
                cur = np.linalg.norm(out[k], axis=1) + 1e-12
                out[k] = (out[k] * (target / cur)[:, None]).astype(np.float32)
            elif W.ndim == 1 and _is_residual_vector(k) and W.shape[0] == d:
                out[k] = np.ascontiguousarray(W[inv]).astype(np.float32)
        if token_rank:
            for k in ("embed.tok", "unembed.w"):
                if k in canon:
                    out[k] = _rank_assign_pc1(out[k], canon[k][:, perm], rng)
    return out


def _n6_from_z(E: np.ndarray, Z: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    mu = E.mean(axis=0)
    C = np.cov(E, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    C_half = (evecs * np.sqrt(np.clip(evals, 0.0, None))[None, :]) @ evecs.T
    return (Z @ C_half + mu[None, :]).astype(np.float32)


def a4_circuit(canon, cfg, fam, seed, model=None, report=None):
    """A3 + explicit previous-token / induction circuit where H6 found one; None where it did not."""
    from .circuits import CIRCUITS, tune_and_install
    if model not in CIRCUITS:
        return None
    w = build(canon, cfg, fam, seed, model=model, ordered=True, align=True, identity=True, couple_unembed=True,
              report=report)
    return tune_and_install(w, canon, cfg, fam, model, seed, device="cpu", report=report)


def a2_ordered(canon, cfg, fam, seed, model=None, report=None):
    return build(canon, cfg, fam, seed, model=model, ordered=True, report=report)


def a2f_ordered_freq(canon, cfg, fam, seed, model=None, report=None):
    return build(canon, cfg, fam, seed, model=model, ordered=True, token_rank=True, report=report)


def a3_aligned(canon, cfg, fam, seed, model=None, report=None):
    return build(canon, cfg, fam, seed, model=model, ordered=True, align=True, identity=True, report=report)


CONDITIONS = {"A0": a0_fresh, "A1": a1_spectrum, "A2": a2_ordered, "A2F": a2f_ordered_freq, "A3": a3_aligned,
              "A4": a4_circuit}
