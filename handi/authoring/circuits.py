"""A4: explicit previous-token + induction circuit written into an A3 weight set (hypotheses.md section 5).

The two heads are placed where H6 found them (attention-confirmed heads; Qwen's previous-token head from
the weight statistic, since its attention test found none). Nothing in the circuit comes from trained
weights except the head locations and the A3 patterns it sits in; four scalar gains are tuned with the
model in the loop on 20 repeated-random-token sequences (a different seed from the M2 metric).

Previous-token head (layer lp, head hp)
  learned positions (GPT-2):  Q = g1 * (Vp A)^T * LN-compensation, K = Vp^T, where Vp = top-hd PCs of the
      authored W_pos (ranked by increment energy) and A is the least-squares shift c_i -> c_{i-1} in those
      coordinates; so q_i ~ c_{i-1} and the score peaks at j = i-1.
  RoPE with biases (Pythia):  q = b_q, k = b_k on the rotary pairs with b_q = g1 (cos(-w_j), sin(-w_j)),
      b_k = g1 (1, 0): score = g1^2/sqrt(hd) * sum_j cos(w_j (j - i + 1)), peaked at j = i-1.
  RoPE without biases (Qwen): the same two vectors are read from the embedding common direction mu
      (N6 keeps the trained mean), which every position carries.
  V = Pi1 (Haar hd x d, orthogonal to Vp for learned positions), O = beta * S1 (Haar d x hd):
      writes a hd-dim projection of the previous token's residual into a slot S1.
Induction head (layer li > lp, head hi)
  K = S1^T (reads the slot: ~ previous token of j), Q = g2 * Pi1 (current token of i); non-rotary dims
      only in RoPE models; score peaks where token(j-1) == token(i).
  V = Pi2 (Haar), O = alpha * S2 with S2 = Pi2^T (tied embeddings) or M^-T Pi2^T for untied embeddings,
      where the unembedding is the coupled N6 sample u_t = M (e_t - mu_e) + mu_u; the logit of token t
      then grows with (Pi2 e_t) . (Pi2 e_j), maximal at t = token(j).
Qwen specifics: the circuit heads' gate rows are zero (sigmoid 0.5); the layer's q/k RMSNorm gains are set
to 1 (zero-centered weight 0); the other Q/O heads of the GQA group that shares the overwritten K/V head
are zeroed. Everything else in the model is the A3 weight set.
"""
from __future__ import annotations

import itertools
import math
import re

import numpy as np
import torch

from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM

from ..analysis import registry
from ..canon import from_canonical

# (layer, head) of the previous-token head and of the induction head, from results/tables/h06.csv
# GPT-2 (4, 11), (5, 5) is excluded: no authored previous-token design for learned positions (decisions.md gap)
CIRCUITS = {
    "pythia-70m": ((2, 1), (3, 6)),
    "pythia-160m": ((3, 2), (4, 6)),
    "qwen3.5-0.8b": ((7, 0), (15, 7)),
}
ATTENUATION = 0.01   # decision of 2026-11-04: non-circuit residual writes in A4 are scaled by this factor
TUNE_SEED = 123
N_TUNE = 20
LENGTH = 50
GRID_GAIN = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
GRID_WRITE = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e4, 1e5)


def _dims(cfg: dict, fam: str):
    d = cfg.get("n_embd") or cfg.get("hidden_size")
    if fam == "qwen3_5":
        H, Hkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    else:
        H = cfg.get("n_head") or cfg.get("num_heads") or cfg.get("num_attention_heads")
        Hkv, hd = H, d // H
    return d, H, Hkv, hd


def _haar(m: int, k: int, rng: np.random.Generator, orth_to: np.ndarray | None = None) -> np.ndarray:
    Z = rng.standard_normal((m, k))
    if orth_to is not None:
        Z = Z - orth_to @ (orth_to.T @ Z)
    Q, R = np.linalg.qr(Z)
    return (Q * np.sign(np.diag(R))[None, :]).astype(np.float32)


def _rotary(cfg: dict, fam: str, hd: int) -> tuple[int, np.ndarray]:
    """(rotary_ndims, inv_freq[rot/2]) exactly as the HF class computes them."""
    if fam == "gpt_neox":
        rot = int(hd * cfg["rotary_pct"])
        base = cfg.get("rotary_emb_base", 10000)
    elif fam == "qwen3_5":
        rot = int(hd * cfg["rope_parameters"].get("partial_rotary_factor", 1.0))
        base = cfg["rope_parameters"]["rope_theta"]
    else:
        return 0, np.zeros(0)
    inv = 1.0 / (base ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    return rot, inv


class Circuit:
    def __init__(self, canon: dict, cfg: dict, fam: str, model: str, seed: int):
        self.cfg, self.fam, self.model = cfg, fam, model
        self.d, self.H, self.Hkv, self.hd = _dims(cfg, fam)
        self.rep = self.H // self.Hkv
        (self.lp, self.hp), (self.li, self.hi) = CIRCUITS[model]
        self.rng = np.random.default_rng([seed, 4])
        self.rot, self.inv_freq = _rotary(cfg, fam, self.hd)
        self.learned_pos = "embed.pos" in canon
        self.tied = "unembed.w" not in canon
        from ..analysis.loader import attn_layers, n_layers
        self.attn_layer_ids = attn_layers(cfg, fam)
        self.n_layers = n_layers(cfg, fam)

    # ------------------------------------------------------------------ pieces that do not depend on gains
    def prepare(self, w: dict[str, np.ndarray]) -> None:
        d, hd = self.d, self.hd
        Vp = None
        if self.learned_pos:
            P = w["embed.pos"].astype(np.float64)
            U, S, Vt = np.linalg.svd(P, full_matrices=False)
            inc = (np.diff(U, axis=0) ** 2).sum(0) * S ** 2            # increment energy per PC
            pick = np.argsort(inc)[::-1][:hd]
            Vp = Vt[pick].T                                            # [d, hd]
            n = min(len(P), 1024)
            C = P[:n] @ Vp                                             # [n, hd] coordinates
            # bilinear form on the curve fitted to a lag-1 target: C A C^T ~ T, T[i, i-1] = 1
            T = np.zeros((n, n)); T[np.arange(1, n), np.arange(0, n - 1)] = 1.0
            Cp = np.linalg.pinv(C)
            A = Cp @ T @ Cp.T
            self.Wq_prev = (Vp @ A).T.astype(np.float32)               # q_i = A^T c_i
            self.Wk_prev = Vp.T.astype(np.float32)                     # k_j = c_j
        self.Vp = Vp
        # token-reading projections and the slot are mutually orthogonal (and orthogonal to the position
        # subspace), so the slot written by the previous-token head cannot leak into the token readers
        self.Pi1 = _haar(d, hd, self.rng, orth_to=Vp).T               # [hd, d]
        used = self.Pi1.T if Vp is None else np.concatenate([Vp, self.Pi1.T], axis=1)
        self.S1 = _haar(d, hd, self.rng, orth_to=used)                 # [d, hd]
        used = np.concatenate([used, self.S1], axis=1)
        self.Pi2 = _haar(d, hd, self.rng, orth_to=used).T
        # final-norm gain compensation: logits = U (g_f * n + b_f), so write through 1/g_f
        gf = w["ln_f.w"].astype(np.float64)
        if self.fam == "qwen3_5":
            gf = 1.0 + gf
        inv_gf = (1.0 / np.where(np.abs(gf) < 1e-3, 1e-3, gf)).astype(np.float32)
        if self.tied:
            self.S2 = (inv_gf[:, None] * self.Pi2.T).astype(np.float32)
        else:
            E, Uw = w["embed.tok"].astype(np.float64), w["unembed.w"].astype(np.float64)
            Ce, Cu = np.cov(E, rowvar=False), np.cov(Uw, rowvar=False)
            def sqrt_sym(C, inv=False):
                ev, Q = np.linalg.eigh(C); ev = np.clip(ev, 1e-10, None)
                return (Q * (ev ** (-0.5 if inv else 0.5))[None, :]) @ Q.T
            M = sqrt_sym(Cu) @ sqrt_sym(Ce, inv=True)                  # u = M (e - mu_e) + mu_u (coupled N6)
            self.S2 = (inv_gf[:, None] * (np.linalg.inv(M).T @ self.Pi2.T)).astype(np.float32)
        if self.fam == "qwen3_5":
            mu = w["embed.tok"].mean(axis=0).astype(np.float64)
            self.mu_hat = (mu / np.linalg.norm(mu)).astype(np.float32)

    def _ln(self, w: dict, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """(1/gain, bias) of the attention-input norm, to read the residual in its own units."""
        g = w[f"layers.{layer}.ln1.w"].astype(np.float64)
        if self.fam == "qwen3_5":
            g = 1.0 + g
        inv = 1.0 / np.where(np.abs(g) < 1e-3, 1e-3, g)
        b = w.get(f"layers.{layer}.ln1.b", np.zeros(self.d))
        return inv.astype(np.float32), b.astype(np.float32)

    # ------------------------------------------------------------------ write the circuit for given gains
    def install(self, w: dict[str, np.ndarray], g1: float, beta: float, g2: float, alpha: float) -> dict:
        w = dict(w)
        d, hd, rot = self.d, self.hd, self.rot
        for (layer, head) in ((self.lp, self.hp), (self.li, self.hi)):
            kv = head // self.rep
            pre = f"layers.{layer}.attn."
            for n in ("q", "k", "v", "o"):
                w[pre + n + ".w"] = w[pre + n + ".w"].copy()
                if pre + n + ".b" in w:
                    w[pre + n + ".b"] = w[pre + n + ".b"].copy()
            if self.fam == "qwen3_5":
                w[pre + "gate.w"] = w[pre + "gate.w"].copy()
                w[pre + "qnorm.w"] = np.zeros_like(w[pre + "qnorm.w"])
                w[pre + "knorm.w"] = np.zeros_like(w[pre + "knorm.w"])
                for sib in range(kv * self.rep, (kv + 1) * self.rep):
                    if sib != head:
                        w[pre + "q.w"][sib * hd:(sib + 1) * hd] = 0
                        w[pre + "o.w"][:, sib * hd:(sib + 1) * hd] = 0
                        w[pre + "gate.w"][sib * hd:(sib + 1) * hd] = 0
                w[pre + "gate.w"][head * hd:(head + 1) * hd] = 0
        inv_p, b_p = self._ln(w, self.lp)
        inv_i, b_i = self._ln(w, self.li)

        def put(layer, head, q, k, v, o):
            kv = head // self.rep
            pre = f"layers.{layer}.attn."
            w[pre + "q.w"][head * hd:(head + 1) * hd] = q
            w[pre + "k.w"][kv * hd:(kv + 1) * hd] = k
            w[pre + "v.w"][kv * hd:(kv + 1) * hd] = v
            w[pre + "o.w"][:, head * hd:(head + 1) * hd] = o

        def bias(layer, head, name, vec):
            key = f"layers.{layer}.attn.{name}.b"
            if key in w:
                kvh = head // self.rep if name in ("k", "v") else head
                w[key][kvh * hd:(kvh + 1) * hd] = vec

        # ---- previous-token head
        q = np.zeros((hd, d), np.float32); k = np.zeros((hd, d), np.float32)
        bq = np.zeros(hd, np.float32); bk = np.zeros(hd, np.float32)
        if self.learned_pos:
            q = g1 * self.Wq_prev * inv_p[None, :]
            k = self.Wk_prev * inv_p[None, :]
            bq, bk = -(q @ b_p), -(k @ b_p)
        else:
            half = rot // 2
            phi = -self.inv_freq
            if self.fam == "gpt_neox":
                bq[:half], bq[half:rot] = g1 * np.cos(phi), g1 * np.sin(phi)
                bk[0:half] = g1
            else:                                                       # Qwen: constant from the common direction
                q[:half] = np.cos(phi)[:, None] * self.mu_hat[None, :]
                q[half:rot] = np.sin(phi)[:, None] * self.mu_hat[None, :]
                k[0:half] = self.mu_hat[None, :]
        v = self.Pi1 * inv_p[None, :]
        o = beta * self.S1
        put(self.lp, self.hp, q, k, v, o)
        bias(self.lp, self.hp, "q", bq); bias(self.lp, self.hp, "k", bk); bias(self.lp, self.hp, "v", -(v @ b_p))

        # ---- induction head
        k2 = self.S1.T * inv_i[None, :]
        q2 = g2 * self.Pi1 * inv_i[None, :]
        if rot:
            q2[:rot] = 0; k2[:rot] = 0
        v2 = self.Pi2 * inv_i[None, :]
        o2 = alpha * self.S2
        put(self.li, self.hi, q2, k2, v2, o2)
        bias(self.li, self.hi, "q", -(q2 @ b_i)); bias(self.li, self.hi, "k", -(k2 @ b_i)); bias(self.li, self.hi, "v", -(v2 @ b_i))
        return w


# ---------------------------------------------------------------------------------------------- tuning
def _load_once(name: str, device: str):
    snap = str(registry.snapshot_dir(name))
    fam = registry.family(name)
    cls = Qwen3_5ForCausalLM if fam == "qwen3_5" else AutoModelForCausalLM
    model = cls.from_pretrained(snap, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    return model, fam


def _set_weights(model, name: str, fam: str, cfg: dict, w: dict) -> None:
    ct = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in w.items()}
    kw = {}
    if fam == "gpt2" and not any(k.startswith("transformer.") for k in model.state_dict()):
        kw["prefix"] = ""
    sd = from_canonical(fam, ct, cfg, **kw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not re.search(r"\.(attn|attention)\.bias$|masked_bias|inv_freq|lm_head\.weight$", m)]
    if missing or unexpected:
        raise RuntimeError(f"authored load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()


@torch.no_grad()
def _score(model, x: torch.Tensor, circ: Circuit) -> tuple[float, float, float]:
    out = model(x[:, :-1], output_attentions=True)
    logits = out.logits[:, LENGTH - 1:]
    pred = logits.argmax(-1)
    tgt = x[:, LENGTH:]
    acc = float((pred == tgt).float().mean())
    correct = logits.gather(-1, tgt[..., None])[..., 0]
    other = logits.scatter(-1, tgt[..., None], float("-inf")).max(-1).values
    _score.last_margin = float((correct - other).mean())
    T = x.shape[1] - 1
    att = out.attentions
    if len(att) != circ.n_layers:                                     # Qwen returns only the full-attention layers
        att = {l: a for l, a in zip(circ.attn_layer_ids, att)}
    att_p = att[circ.lp][:, circ.hp]                                 # [B, T, T]
    idx = torch.arange(1, T)
    prev = float(att_p[:, idx, idx - 1].mean())
    att_i = att[circ.li][:, circ.hi]
    idx2 = torch.arange(LENGTH, T)
    ind = float(att_i[:, idx2, idx2 - LENGTH + 1].mean())
    return acc, prev, ind


def attenuate(w: dict[str, np.ndarray], cfg: dict, fam: str, model_name: str, s: float = ATTENUATION) -> dict:
    """Scale every residual write that is not one of the two circuit heads by s (A4 definition)."""
    w = dict(w)
    (lp, hp), (li, hi) = CIRCUITS[model_name]
    _, _, _, hd = _dims(cfg, fam)
    for k in list(w):
        if k.endswith((".mlp.down.w", ".mlp.down.b", ".linear.out.w", ".attn.o.b")):
            w[k] = (w[k] * s).astype(np.float32)
        elif k.endswith(".attn.o.w"):
            layer = int(k.split(".")[1])
            keep = [hp] if layer == lp else ([hi] if layer == li else [])
            m = np.full_like(w[k], s)
            for h in keep:
                m[:, h * hd:(h + 1) * hd] = 1.0
            w[k] = (w[k] * m).astype(np.float32)
    return w


def tune_and_install(w: dict[str, np.ndarray], canon: dict, cfg: dict, fam: str, model_name: str, seed: int,
                     device: str = "cpu", report: dict | None = None, attenuation: float | None = ATTENUATION) -> dict[str, np.ndarray]:
    if attenuation is not None:
        w = attenuate(w, cfg, fam, model_name, attenuation)
    circ = Circuit(canon, cfg, fam, model_name, seed)
    circ.prepare(w)
    model, _ = _load_once(model_name, device)
    V = cfg["vocab_size"]
    rng = np.random.default_rng(TUNE_SEED)
    x = rng.integers(0, V, size=(N_TUNE, LENGTH))
    x = torch.tensor(np.concatenate([x, x], 1), device=device)
    tune_g1 = fam != "qwen3_5"
    tune_g2 = fam != "qwen3_5"
    params = {"g1": 1.0 if tune_g1 else 1.0, "beta": 10.0, "g2": 1.0 if tune_g2 else 1.0, "alpha": 10.0}
    grids = {"g1": GRID_GAIN if tune_g1 else (1.0,), "beta": GRID_WRITE, "g2": GRID_GAIN if tune_g2 else (1.0,), "alpha": GRID_WRITE}
    # staged objectives: g1 -> previous-token attention; beta, g2 -> induction attention;
    # alpha -> copy accuracy then logit margin. Two passes.
    objective = {"g1": lambda r: r[1], "beta": lambda r: r[2], "g2": lambda r: r[2], "alpha": lambda r: (r[0], r[3])}
    for _pass in range(2):
        for key in ("g1", "beta", "g2", "alpha"):
            if len(grids[key]) == 1:
                continue
            results = []
            for val in grids[key]:
                trial = dict(params); trial[key] = val
                _set_weights(model, model_name, fam, cfg, circ.install(w, **trial))
                acc, prev, ind = _score(model, x, circ)
                results.append(((acc, prev, ind, _score.last_margin), val))
            _, val = max(results, key=lambda r: objective[key](r[0]))
            params[key] = val
    final = circ.install(w, **params)
    _set_weights(model, model_name, fam, cfg, final)
    acc, prev, ind = _score(model, x, circ)
    if report is not None:
        report.update({"a4_params": params, "a4_tune_copy_acc": acc, "a4_prev_attn": prev, "a4_induction_attn": ind,
                       "a4_logit_margin": _score.last_margin, "a4_heads": CIRCUITS[model_name], "a4_attenuation": attenuation})
    del model
    return final
