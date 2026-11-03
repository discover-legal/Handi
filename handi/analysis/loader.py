"""Load any registered model's weights in the canonical schema as float32 numpy arrays, raw
(through the safetensors header reader, never through a model class)."""
from __future__ import annotations

import numpy as np
import torch

from ..canon import family_of, to_canonical, is_parameter_buffer
from . import registry
from .safetensors_np import load_all, open_checkpoint


def load_canonical(name: str) -> tuple[dict[str, np.ndarray], dict, str]:
    """Returns (canonical weights as float32 numpy, HF config dict (text_config for Qwen), family)."""
    cfg = registry.load_config(name)
    snap = registry.snapshot_dir(name)
    registry.weights_path(name)   # converts TinyStories .bin -> safetensors on first use
    raw: dict[str, np.ndarray] = {}
    for shard in open_checkpoint(snap):
        raw.update(load_all(shard))
    raw = {k: v for k, v in raw.items() if not is_parameter_buffer(k) and not k.startswith(("model.visual.", "mtp."))}
    fam = registry.family(name)
    hf = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in raw.items()}
    canon = to_canonical(fam, hf, cfg)
    return {k: v.numpy() for k, v in canon.items()}, cfg, fam


def attn_heads(canon: dict[str, np.ndarray], cfg: dict, fam: str, layer: int):
    """Per-head (W_Q, W_K, W_V, W_O) as lists of [d, hd] / [hd, d] arrays, expanding GQA kv heads."""
    q, k, v, o = (canon[f"layers.{layer}.attn.{n}.w"] for n in "qkvo")
    d = q.shape[1]
    if fam == "qwen3_5":
        H, Hkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    elif fam == "gpt_neo":
        H = cfg["num_heads"]; hd = d // H; Hkv = H
    elif fam == "gpt_neox":
        H = cfg["num_attention_heads"]; hd = d // H; Hkv = H
    else:
        H = cfg["n_head"]; hd = d // H; Hkv = H
    rep = H // Hkv
    heads = []
    for h in range(H):
        kv = h // rep
        heads.append((q[h * hd:(h + 1) * hd].T, k[kv * hd:(kv + 1) * hd].T, v[kv * hd:(kv + 1) * hd].T, o[:, h * hd:(h + 1) * hd]))
    return heads   # each: W_Q [d,hd], W_K [d,hd], W_V [d,hd], W_O [d,hd] so W_QK = W_Q @ W_K.T, W_OV = W_V @ W_O.T


def attn_layers(cfg: dict, fam: str) -> list[int]:
    if fam == "qwen3_5":
        return [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    n = cfg.get("n_layer") or cfg.get("num_layers") or cfg.get("num_hidden_layers")
    return list(range(n))


def n_layers(cfg: dict, fam: str) -> int:
    return cfg.get("n_layer") or cfg.get("num_layers") or cfg.get("num_hidden_layers")
