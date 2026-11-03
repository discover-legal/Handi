"""Seeded null generators (plan Part B: N1..N7) and canonical fresh-init (N4).

Every generator takes a numpy array and a `np.random.Generator`; nothing here
touches a model class. Canonical state-dict names:

    embed.tok [V,d]            embed.pos [n_ctx,d] (learned-pos families only)
    layers.{i}.ln1.w/.b        layers.{i}.ln2.w/.b   (RMSNorm families: .w only)
    layers.{i}.attn.q.w [H*hd,d]  .k.w [Hkv*hd,d]  .v.w [Hkv*hd,d]  .o.w [d,H*hd]
    layers.{i}.mlp.up.w [I,d]  layers.{i}.mlp.gate.w [I,d] (SwiGLU only)
    layers.{i}.mlp.down.w [d,I]
    layers.{i}.linear.{in_qkv,in_z,in_b,in_a,out}.w   (Qwen3.5 linear_attention layers)
    ln_f.w/.b                  unembed.w [V,d]
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# N1..N3, N5..N7: matrix-level nulls
# ----------------------------------------------------------------------------


def n1_gaussian(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """iid Gaussian, same shape and Frobenius norm as W."""
    W = np.asarray(W, dtype=np.float32)
    Z = rng.standard_normal(W.shape, dtype=np.float32)
    nz = np.linalg.norm(Z)
    if nz == 0:
        return Z
    return Z * (np.float32(np.linalg.norm(W)) / nz)


def n2_shuffle(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute all elements of W (destroys row/col structure, keeps the value histogram)."""
    W = np.asarray(W)
    flat = W.reshape(-1)
    return flat[rng.permutation(flat.size)].reshape(W.shape)


def n3_permute(
    W: np.ndarray, rng: np.random.Generator, rows: bool = True, cols: bool = True
) -> np.ndarray:
    """Independent row and/or column permutation (same singular values, destroyed order)."""
    W = np.asarray(W)
    out = W
    if rows:
        out = out[rng.permutation(W.shape[0]), :]
    if cols:
        out = out[:, rng.permutation(W.shape[1])]
    return np.ascontiguousarray(out)


def haar_stiefel(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed m x k matrix with orthonormal columns (QR with sign correction)."""
    Z = rng.standard_normal((m, k))
    Q, R = np.linalg.qr(Z, mode="reduced")
    d = np.diag(R)
    s = np.where(d >= 0, 1.0, -1.0)
    return Q * s[None, :]


def n5_spectrum_matched(
    W: np.ndarray, rng: np.random.Generator, sigma: np.ndarray | None = None
) -> np.ndarray:
    """U diag(sigma_W) V^T with Haar U [m,k], V [n,k]; k = min(m, n).

    `sigma` may be passed to skip recomputing the SVD of W.
    """
    W = np.asarray(W, dtype=np.float32)
    m, n = W.shape
    k = min(m, n)
    if sigma is None:
        sigma = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    sigma = np.asarray(sigma, dtype=np.float64)[:k]
    U = haar_stiefel(m, k, rng)
    V = haar_stiefel(n, k, rng)
    return ((U * sigma[None, :]) @ V.T).astype(np.float32)


def n6_cov_matched(E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian rows with the same mean and d x d covariance as the rows of E [V, d]."""
    E = np.asarray(E, dtype=np.float64)
    V, d = E.shape
    mu = E.mean(axis=0)
    C = np.cov(E, rowvar=False)  # d x d, unbiased
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 0.0, None)
    C_half = (evecs * np.sqrt(evals)[None, :]) @ evecs.T  # symmetric square root
    Z = rng.standard_normal((V, d))
    return (Z @ C_half + mu[None, :]).astype(np.float32)


def n7_assignment_shuffle(E: np.ndarray, rng: np.random.Generator, idx) -> np.ndarray:
    """Shuffle the rows of E among the index list `idx` (token-assignment null)."""
    E = np.asarray(E)
    idx = np.asarray(idx)
    out = E.copy()
    out[idx] = E[idx[rng.permutation(idx.size)]]
    return out


# ----------------------------------------------------------------------------
# Canonical shapes and N4 fresh init
# ----------------------------------------------------------------------------

FAMILIES = ("gpt2", "gpt_neo", "gpt_neox", "qwen3_5")


def canonical_shapes(family: str, config: dict) -> dict[str, tuple[int, ...]]:
    """{canonical name: shape} for a family from its HF config dict (Qwen: text_config)."""
    shapes: dict[str, tuple[int, ...]] = {}
    if family == "gpt2":
        d = config["n_embd"]
        L = config["n_layer"]
        H = config["n_head"]
        V = config["vocab_size"]
        n_ctx = config["n_positions"]
        I = config.get("n_inner") or 4 * d
        hd = d // H
        shapes["embed.tok"] = (V, d)
        shapes["embed.pos"] = (n_ctx, d)
        for i in range(L):
            p = f"layers.{i}."
            shapes[p + "ln1.w"] = (d,)
            shapes[p + "ln1.b"] = (d,)
            shapes[p + "attn.q.w"] = (H * hd, d)
            shapes[p + "attn.k.w"] = (H * hd, d)
            shapes[p + "attn.v.w"] = (H * hd, d)
            shapes[p + "attn.o.w"] = (d, H * hd)
            shapes[p + "ln2.w"] = (d,)
            shapes[p + "ln2.b"] = (d,)
            shapes[p + "mlp.up.w"] = (I, d)
            shapes[p + "mlp.down.w"] = (d, I)
        shapes["ln_f.w"] = (d,)
        shapes["ln_f.b"] = (d,)
        shapes["unembed.w"] = (V, d)
    elif family == "gpt_neo":
        d = config["hidden_size"]
        L = config["num_layers"]
        H = config["num_heads"]
        V = config["vocab_size"]
        n_ctx = config["max_position_embeddings"]
        I = config.get("intermediate_size") or 4 * d
        hd = d // H
        shapes["embed.tok"] = (V, d)
        shapes["embed.pos"] = (n_ctx, d)
        for i in range(L):
            p = f"layers.{i}."
            shapes[p + "ln1.w"] = (d,)
            shapes[p + "ln1.b"] = (d,)
            shapes[p + "attn.q.w"] = (H * hd, d)
            shapes[p + "attn.k.w"] = (H * hd, d)
            shapes[p + "attn.v.w"] = (H * hd, d)
            shapes[p + "attn.o.w"] = (d, H * hd)
            shapes[p + "ln2.w"] = (d,)
            shapes[p + "ln2.b"] = (d,)
            shapes[p + "mlp.up.w"] = (I, d)
            shapes[p + "mlp.down.w"] = (d, I)
        shapes["ln_f.w"] = (d,)
        shapes["ln_f.b"] = (d,)
        shapes["unembed.w"] = (V, d)
    elif family == "gpt_neox":
        d = config["hidden_size"]
        L = config["num_hidden_layers"]
        H = config["num_attention_heads"]
        V = config["vocab_size"]
        I = config["intermediate_size"]
        hd = d // H
        shapes["embed.tok"] = (V, d)  # rotary: no embed.pos
        for i in range(L):
            p = f"layers.{i}."
            shapes[p + "ln1.w"] = (d,)
            shapes[p + "ln1.b"] = (d,)
            shapes[p + "attn.q.w"] = (H * hd, d)
            shapes[p + "attn.k.w"] = (H * hd, d)
            shapes[p + "attn.v.w"] = (H * hd, d)
            shapes[p + "attn.o.w"] = (d, H * hd)
            shapes[p + "ln2.w"] = (d,)
            shapes[p + "ln2.b"] = (d,)
            shapes[p + "mlp.up.w"] = (I, d)
            shapes[p + "mlp.down.w"] = (d, I)
        shapes["ln_f.w"] = (d,)
        shapes["ln_f.b"] = (d,)
        shapes["unembed.w"] = (V, d)
    elif family == "qwen3_5":
        d = config["hidden_size"]
        L = config["num_hidden_layers"]
        H = config["num_attention_heads"]
        Hkv = config["num_key_value_heads"]
        hd = config["head_dim"]
        I = config["intermediate_size"]
        V = config["vocab_size"]
        layer_types = config.get("layer_types")
        if layer_types is None:
            k = config.get("full_attention_interval", 4)
            layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(L)
            ]
        Hk = config["linear_num_key_heads"]
        dk = config["linear_key_head_dim"]
        Hv = config["linear_num_value_heads"]
        dv = config["linear_value_head_dim"]
        shapes["embed.tok"] = (V, d)  # RoPE: no embed.pos
        for i in range(L):
            p = f"layers.{i}."
            shapes[p + "ln1.w"] = (d,)
            if layer_types[i] == "full_attention":
                shapes[p + "attn.q.w"] = (H * hd, d)
                shapes[p + "attn.k.w"] = (Hkv * hd, d)
                shapes[p + "attn.v.w"] = (Hkv * hd, d)
                shapes[p + "attn.o.w"] = (d, H * hd)
            else:
                shapes[p + "linear.in_qkv.w"] = (2 * Hk * dk + Hv * dv, d)
                shapes[p + "linear.in_z.w"] = (Hv * dv, d)
                shapes[p + "linear.in_b.w"] = (Hv, d)
                shapes[p + "linear.in_a.w"] = (Hv, d)
                shapes[p + "linear.out.w"] = (d, Hv * dv)
            shapes[p + "ln2.w"] = (d,)
            shapes[p + "mlp.up.w"] = (I, d)
            shapes[p + "mlp.gate.w"] = (I, d)
            shapes[p + "mlp.down.w"] = (d, I)
        shapes["ln_f.w"] = (d,)
        shapes["unembed.w"] = (V, d)
    else:
        raise ValueError(f"unknown family {family}; known: {FAMILIES}")
    return shapes


_N_LAYERS_KEY = {"gpt2": "n_layer", "gpt_neo": "num_layers", "gpt_neox": "num_hidden_layers", "qwen3_5": "num_hidden_layers"}


def _n_layers(family: str, config: dict) -> int:
    return config[_N_LAYERS_KEY[family]]


def _d_model(family: str, config: dict) -> int:
    return config["n_embd"] if family == "gpt2" else config["hidden_size"]


def _is_output_proj(name: str) -> bool:
    return name.endswith("attn.o.w") or name.endswith("mlp.down.w") or name.endswith("linear.out.w")


def _is_norm(name: str) -> bool:
    return ".ln1." in name or ".ln2." in name or name.startswith("ln_f.")


def n4_fresh_init(family: str, config: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Fresh-init state dict (canonical names) following the family's init rule.

    gpt2:     N(0, 0.02) everywhere, attn.o / mlp.down scaled by 1/sqrt(2*n_layers),
              LN gain 1 bias 0, pos emb N(0, 0.02).
    gpt_neo:  N(0, 0.02) everywhere, LN 1/0.
    gpt_neox: small_init N(0, sqrt(2/(5d))) for input-side matrices (embed, q/k/v,
              mlp.up, unembed); wang_init N(0, 2/(n_layers*sqrt(d))) for output
              projections (attn.o, mlp.down); LN 1/0.
    qwen3_5:  N(0, 0.02) everywhere, RMSNorm weight 0 (zero-centered).

    Where the HF config ties input and output embeddings (gpt2, gpt_neo default,
    qwen3_5 tie_word_embeddings=True) `unembed.w` is a copy of `embed.tok`.
    """
    shapes = canonical_shapes(family, config)
    L = _n_layers(family, config)
    d = _d_model(family, config)
    tied = {
        "gpt2": config.get("tie_word_embeddings", True),
        "gpt_neo": config.get("tie_word_embeddings", True),
        "gpt_neox": config.get("tie_word_embeddings", False),
        "qwen3_5": config.get("tie_word_embeddings", False),
    }[family]

    def normal(shape, std):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    out: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        if _is_norm(name):
            if name.endswith(".b"):
                out[name] = np.zeros(shape, dtype=np.float32)
            elif family == "qwen3_5":
                out[name] = np.zeros(shape, dtype=np.float32)
            else:
                out[name] = np.ones(shape, dtype=np.float32)
            continue
        if name == "unembed.w" and tied:
            continue
        if family == "gpt2":
            std = 0.02
            if _is_output_proj(name):
                std = 0.02 / np.sqrt(2.0 * L)
        elif family == "gpt_neo":
            std = 0.02
        elif family == "gpt_neox":
            if _is_output_proj(name):
                std = 2.0 / (L * np.sqrt(d))
            else:
                std = np.sqrt(2.0 / (5.0 * d))
        elif family == "qwen3_5":
            std = 0.02
        else:  # pragma: no cover
            raise ValueError(family)
        out[name] = normal(shape, std)
    if tied:
        out["unembed.w"] = out["embed.tok"].copy()
    return out
