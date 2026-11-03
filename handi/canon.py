"""Canonical tensor schema shared by authoring and analysis, with lossless converters to and from
the Hugging Face state dicts of the four families (gpt2, gpt_neo, gpt_neox, qwen3_5).

Canonical names (all Linear weights are [out, in]):
    embed.tok [V, d]            embed.pos [n_ctx, d] (learned-position models only)
    layers.{i}.ln1.w / .b       layers.{i}.ln2.w / .b
    layers.{i}.attn.q.w [H*hd, d]  .k.w [Hkv*hd, d]  .v.w [Hkv*hd, d]  .o.w [d, H*hd]   (+ .b biases)
    layers.{i}.attn.gate.w [H*hd, d]       (qwen3_5 attention output gate, split out of q_proj)
    layers.{i}.attn.qnorm.w / .knorm.w     (qwen3_5 per-head q/k RMSNorm)
    layers.{i}.mlp.up.w [I, d]  .gate.w [I, d] (SwiGLU only)  .down.w [d, I]   (+ .b)
    layers.{i}.linear.{in_qkv,in_z,in_b,in_a,out}.w, .conv.w, .a_log, .dt_bias, .norm.w   (qwen3_5 Gated DeltaNet)
    ln_f.w / .b                 unembed.w [V, d] (absent when tied)
Non-parameter buffers (causal masks, rotary inv_freq) are dropped; `from_canonical` recreates the
family's parameter names only, which is what `load_state_dict(strict=False)` needs.
"""
from __future__ import annotations

import re

import torch

FAMILIES = ("gpt2", "gpt_neo", "gpt_neox", "qwen3_5")


def family_of(config: dict) -> str:
    mt = config.get("model_type")
    if mt in ("qwen3_5", "qwen3_5_text"):
        return "qwen3_5"
    if mt in FAMILIES:
        return mt
    raise ValueError(f"unknown model_type {mt!r}")


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


# --------------------------------------------------------------------------------------------
# GPT-2 (Conv1D: weights stored [in, out])
# --------------------------------------------------------------------------------------------

def _gpt2_to_canon(sd: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    p = "transformer." if any(k.startswith("transformer.") for k in sd) else ""
    d = cfg["n_embd"]
    out = {"embed.tok": sd[p + "wte.weight"], "embed.pos": sd[p + "wpe.weight"],
           "ln_f.w": sd[p + "ln_f.weight"], "ln_f.b": sd[p + "ln_f.bias"]}
    for i in range(cfg["n_layer"]):
        h = f"{p}h.{i}."
        c = f"layers.{i}."
        out[c + "ln1.w"], out[c + "ln1.b"] = sd[h + "ln_1.weight"], sd[h + "ln_1.bias"]
        out[c + "ln2.w"], out[c + "ln2.b"] = sd[h + "ln_2.weight"], sd[h + "ln_2.bias"]
        w = sd[h + "attn.c_attn.weight"].T          # [3d, d]
        b = sd[h + "attn.c_attn.bias"]
        for j, n in enumerate("qkv"):
            out[c + f"attn.{n}.w"] = w[j * d:(j + 1) * d]
            out[c + f"attn.{n}.b"] = b[j * d:(j + 1) * d]
        out[c + "attn.o.w"] = sd[h + "attn.c_proj.weight"].T
        out[c + "attn.o.b"] = sd[h + "attn.c_proj.bias"]
        out[c + "mlp.up.w"] = sd[h + "mlp.c_fc.weight"].T
        out[c + "mlp.up.b"] = sd[h + "mlp.c_fc.bias"]
        out[c + "mlp.down.w"] = sd[h + "mlp.c_proj.weight"].T
        out[c + "mlp.down.b"] = sd[h + "mlp.c_proj.bias"]
    return out


def _gpt2_from_canon(cn: dict[str, torch.Tensor], cfg: dict, prefix: str = "transformer.") -> dict[str, torch.Tensor]:
    p = prefix
    sd = {p + "wte.weight": cn["embed.tok"], p + "wpe.weight": cn["embed.pos"],
          p + "ln_f.weight": cn["ln_f.w"], p + "ln_f.bias": cn["ln_f.b"]}
    for i in range(cfg["n_layer"]):
        h, c = f"{p}h.{i}.", f"layers.{i}."
        sd[h + "ln_1.weight"], sd[h + "ln_1.bias"] = cn[c + "ln1.w"], cn[c + "ln1.b"]
        sd[h + "ln_2.weight"], sd[h + "ln_2.bias"] = cn[c + "ln2.w"], cn[c + "ln2.b"]
        sd[h + "attn.c_attn.weight"] = torch.cat([cn[c + f"attn.{n}.w"] for n in "qkv"], 0).T.contiguous()
        sd[h + "attn.c_attn.bias"] = torch.cat([cn[c + f"attn.{n}.b"] for n in "qkv"], 0)
        sd[h + "attn.c_proj.weight"] = cn[c + "attn.o.w"].T.contiguous()
        sd[h + "attn.c_proj.bias"] = cn[c + "attn.o.b"]
        sd[h + "mlp.c_fc.weight"] = cn[c + "mlp.up.w"].T.contiguous()
        sd[h + "mlp.c_fc.bias"] = cn[c + "mlp.up.b"]
        sd[h + "mlp.c_proj.weight"] = cn[c + "mlp.down.w"].T.contiguous()
        sd[h + "mlp.c_proj.bias"] = cn[c + "mlp.down.b"]
    if "unembed.w" in cn:
        sd["lm_head.weight"] = cn["unembed.w"]
    return sd


# --------------------------------------------------------------------------------------------
# GPT-Neo (TinyStories): Linear layout, no qkv bias, out_proj bias, local/global attention
# --------------------------------------------------------------------------------------------

def _gpt_neo_to_canon(sd, cfg):
    p = "transformer."
    out = {"embed.tok": sd[p + "wte.weight"], "embed.pos": sd[p + "wpe.weight"],
           "ln_f.w": sd[p + "ln_f.weight"], "ln_f.b": sd[p + "ln_f.bias"]}
    for i in range(cfg["num_layers"]):
        h, c = f"{p}h.{i}.", f"layers.{i}."
        out[c + "ln1.w"], out[c + "ln1.b"] = sd[h + "ln_1.weight"], sd[h + "ln_1.bias"]
        out[c + "ln2.w"], out[c + "ln2.b"] = sd[h + "ln_2.weight"], sd[h + "ln_2.bias"]
        for n in "qkv":
            out[c + f"attn.{n}.w"] = sd[h + f"attn.attention.{n}_proj.weight"]
        out[c + "attn.o.w"] = sd[h + "attn.attention.out_proj.weight"]
        out[c + "attn.o.b"] = sd[h + "attn.attention.out_proj.bias"]
        out[c + "mlp.up.w"], out[c + "mlp.up.b"] = sd[h + "mlp.c_fc.weight"], sd[h + "mlp.c_fc.bias"]
        out[c + "mlp.down.w"], out[c + "mlp.down.b"] = sd[h + "mlp.c_proj.weight"], sd[h + "mlp.c_proj.bias"]
    return out


def _gpt_neo_from_canon(cn, cfg):
    p = "transformer."
    sd = {p + "wte.weight": cn["embed.tok"], p + "wpe.weight": cn["embed.pos"],
          p + "ln_f.weight": cn["ln_f.w"], p + "ln_f.bias": cn["ln_f.b"]}
    for i in range(cfg["num_layers"]):
        h, c = f"{p}h.{i}.", f"layers.{i}."
        sd[h + "ln_1.weight"], sd[h + "ln_1.bias"] = cn[c + "ln1.w"], cn[c + "ln1.b"]
        sd[h + "ln_2.weight"], sd[h + "ln_2.bias"] = cn[c + "ln2.w"], cn[c + "ln2.b"]
        for n in "qkv":
            sd[h + f"attn.attention.{n}_proj.weight"] = cn[c + f"attn.{n}.w"]
        sd[h + "attn.attention.out_proj.weight"] = cn[c + "attn.o.w"]
        sd[h + "attn.attention.out_proj.bias"] = cn[c + "attn.o.b"]
        sd[h + "mlp.c_fc.weight"], sd[h + "mlp.c_fc.bias"] = cn[c + "mlp.up.w"], cn[c + "mlp.up.b"]
        sd[h + "mlp.c_proj.weight"], sd[h + "mlp.c_proj.bias"] = cn[c + "mlp.down.w"], cn[c + "mlp.down.b"]
    if "unembed.w" in cn:
        sd["lm_head.weight"] = cn["unembed.w"]
    return sd


# --------------------------------------------------------------------------------------------
# GPT-NeoX (Pythia): fused qkv interleaved per head, parallel residual, untied unembed
# --------------------------------------------------------------------------------------------

def _neox_to_canon(sd, cfg):
    p = "gpt_neox."
    d, H = cfg["hidden_size"], cfg["num_attention_heads"]
    hd = d // H
    unembed = sd["lm_head.weight"] if "lm_head.weight" in sd else sd["embed_out.weight"]   # model vs checkpoint name
    out = {"embed.tok": sd[p + "embed_in.weight"], "unembed.w": unembed,
           "ln_f.w": sd[p + "final_layer_norm.weight"], "ln_f.b": sd[p + "final_layer_norm.bias"]}
    for i in range(cfg["num_hidden_layers"]):
        h, c = f"{p}layers.{i}.", f"layers.{i}."
        out[c + "ln1.w"], out[c + "ln1.b"] = sd[h + "input_layernorm.weight"], sd[h + "input_layernorm.bias"]
        out[c + "ln2.w"], out[c + "ln2.b"] = sd[h + "post_attention_layernorm.weight"], sd[h + "post_attention_layernorm.bias"]
        w = sd[h + "attention.query_key_value.weight"].view(H, 3, hd, d)
        b = sd[h + "attention.query_key_value.bias"].view(H, 3, hd)
        for j, n in enumerate("qkv"):
            out[c + f"attn.{n}.w"] = w[:, j].reshape(H * hd, d)
            out[c + f"attn.{n}.b"] = b[:, j].reshape(H * hd)
        out[c + "attn.o.w"], out[c + "attn.o.b"] = sd[h + "attention.dense.weight"], sd[h + "attention.dense.bias"]
        out[c + "mlp.up.w"], out[c + "mlp.up.b"] = sd[h + "mlp.dense_h_to_4h.weight"], sd[h + "mlp.dense_h_to_4h.bias"]
        out[c + "mlp.down.w"], out[c + "mlp.down.b"] = sd[h + "mlp.dense_4h_to_h.weight"], sd[h + "mlp.dense_4h_to_h.bias"]
    return out


def _neox_from_canon(cn, cfg, unembed_name: str = "lm_head.weight"):
    """`unembed_name` is 'lm_head.weight' for a loaded model, 'embed_out.weight' for the raw checkpoint."""
    p = "gpt_neox."
    d, H = cfg["hidden_size"], cfg["num_attention_heads"]
    hd = d // H
    sd = {p + "embed_in.weight": cn["embed.tok"], unembed_name: cn["unembed.w"],
          p + "final_layer_norm.weight": cn["ln_f.w"], p + "final_layer_norm.bias": cn["ln_f.b"]}
    for i in range(cfg["num_hidden_layers"]):
        h, c = f"{p}layers.{i}.", f"layers.{i}."
        sd[h + "input_layernorm.weight"], sd[h + "input_layernorm.bias"] = cn[c + "ln1.w"], cn[c + "ln1.b"]
        sd[h + "post_attention_layernorm.weight"], sd[h + "post_attention_layernorm.bias"] = cn[c + "ln2.w"], cn[c + "ln2.b"]
        w = torch.stack([cn[c + f"attn.{n}.w"].view(H, hd, d) for n in "qkv"], 1)   # [H, 3, hd, d]
        b = torch.stack([cn[c + f"attn.{n}.b"].view(H, hd) for n in "qkv"], 1)
        sd[h + "attention.query_key_value.weight"] = w.reshape(3 * d, d).contiguous()
        sd[h + "attention.query_key_value.bias"] = b.reshape(3 * d).contiguous()
        sd[h + "attention.dense.weight"], sd[h + "attention.dense.bias"] = cn[c + "attn.o.w"], cn[c + "attn.o.b"]
        sd[h + "mlp.dense_h_to_4h.weight"], sd[h + "mlp.dense_h_to_4h.bias"] = cn[c + "mlp.up.w"], cn[c + "mlp.up.b"]
        sd[h + "mlp.dense_4h_to_h.weight"], sd[h + "mlp.dense_4h_to_h.bias"] = cn[c + "mlp.down.w"], cn[c + "mlp.down.b"]
    return sd


# --------------------------------------------------------------------------------------------
# Qwen3.5 text model: hybrid Gated DeltaNet / gated attention, RMSNorm, SwiGLU
# --------------------------------------------------------------------------------------------

_QWEN_LINEAR = {"in_proj_qkv.weight": "in_qkv.w", "in_proj_z.weight": "in_z.w", "in_proj_b.weight": "in_b.w",
                "in_proj_a.weight": "in_a.w", "out_proj.weight": "out.w", "conv1d.weight": "conv.w",
                "A_log": "a_log", "dt_bias": "dt_bias", "norm.weight": "norm.w"}


def _qwen_prefix(sd) -> str:
    for p in ("model.language_model.", "model."):
        if any(k.startswith(p + "layers.") for k in sd):
            return p
    raise KeyError("no qwen3_5 language-model tensors found")


def _qwen_to_canon(sd, cfg):
    cfg = text_config(cfg)
    p = _qwen_prefix(sd)
    H, Hkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    d = cfg["hidden_size"]
    out = {"embed.tok": sd[p + "embed_tokens.weight"], "ln_f.w": sd[p + "norm.weight"]}
    if "lm_head.weight" in sd and not cfg.get("tie_word_embeddings", False):
        out["unembed.w"] = sd["lm_head.weight"]
    for i, kind in enumerate(cfg["layer_types"]):
        h, c = f"{p}layers.{i}.", f"layers.{i}."
        out[c + "ln1.w"] = sd[h + "input_layernorm.weight"]
        out[c + "ln2.w"] = sd[h + "post_attention_layernorm.weight"]
        out[c + "mlp.gate.w"] = sd[h + "mlp.gate_proj.weight"]
        out[c + "mlp.up.w"] = sd[h + "mlp.up_proj.weight"]
        out[c + "mlp.down.w"] = sd[h + "mlp.down_proj.weight"]
        if kind == "full_attention":
            qg = sd[h + "self_attn.q_proj.weight"].view(H, 2, hd, d)
            out[c + "attn.q.w"] = qg[:, 0].reshape(H * hd, d)
            out[c + "attn.gate.w"] = qg[:, 1].reshape(H * hd, d)
            out[c + "attn.k.w"] = sd[h + "self_attn.k_proj.weight"]
            out[c + "attn.v.w"] = sd[h + "self_attn.v_proj.weight"]
            out[c + "attn.o.w"] = sd[h + "self_attn.o_proj.weight"]
            out[c + "attn.qnorm.w"] = sd[h + "self_attn.q_norm.weight"]
            out[c + "attn.knorm.w"] = sd[h + "self_attn.k_norm.weight"]
        else:
            for hf, cn in _QWEN_LINEAR.items():
                out[c + "linear." + cn] = sd[h + "linear_attn." + hf]
    return out


def _qwen_from_canon(cn, cfg, prefix: str = "model."):
    cfg = text_config(cfg)
    p = prefix
    H, hd, d = cfg["num_attention_heads"], cfg["head_dim"], cfg["hidden_size"]
    sd = {p + "embed_tokens.weight": cn["embed.tok"], p + "norm.weight": cn["ln_f.w"]}
    if "unembed.w" in cn:
        sd["lm_head.weight"] = cn["unembed.w"]
    for i, kind in enumerate(cfg["layer_types"]):
        h, c = f"{p}layers.{i}.", f"layers.{i}."
        sd[h + "input_layernorm.weight"] = cn[c + "ln1.w"]
        sd[h + "post_attention_layernorm.weight"] = cn[c + "ln2.w"]
        sd[h + "mlp.gate_proj.weight"] = cn[c + "mlp.gate.w"]
        sd[h + "mlp.up_proj.weight"] = cn[c + "mlp.up.w"]
        sd[h + "mlp.down_proj.weight"] = cn[c + "mlp.down.w"]
        if kind == "full_attention":
            qg = torch.stack([cn[c + "attn.q.w"].view(H, hd, d), cn[c + "attn.gate.w"].view(H, hd, d)], 1)
            sd[h + "self_attn.q_proj.weight"] = qg.reshape(H * hd * 2, d).contiguous()
            sd[h + "self_attn.k_proj.weight"] = cn[c + "attn.k.w"]
            sd[h + "self_attn.v_proj.weight"] = cn[c + "attn.v.w"]
            sd[h + "self_attn.o_proj.weight"] = cn[c + "attn.o.w"]
            sd[h + "self_attn.q_norm.weight"] = cn[c + "attn.qnorm.w"]
            sd[h + "self_attn.k_norm.weight"] = cn[c + "attn.knorm.w"]
        else:
            for hf, cname in _QWEN_LINEAR.items():
                sd[h + "linear_attn." + hf] = cn[c + "linear." + cname]
    return sd


# --------------------------------------------------------------------------------------------

_TO = {"gpt2": _gpt2_to_canon, "gpt_neo": _gpt_neo_to_canon, "gpt_neox": _neox_to_canon, "qwen3_5": _qwen_to_canon}
_FROM = {"gpt2": _gpt2_from_canon, "gpt_neo": _gpt_neo_from_canon, "gpt_neox": _neox_from_canon, "qwen3_5": _qwen_from_canon}


def to_canonical(family: str, hf_sd: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    return _TO[family](hf_sd, config)


def from_canonical(family: str, canon: dict[str, torch.Tensor], config: dict, **kw) -> dict[str, torch.Tensor]:
    return _FROM[family](canon, config, **kw)


_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def component_of(name: str) -> tuple[int | None, str]:
    """('layers.3.attn.q.w') -> (3, 'attn.q.w'); ('embed.tok') -> (None, 'embed.tok')."""
    m = _LAYER_RE.match(name)
    return (int(m.group(1)), m.group(2)) if m else (None, name)


def is_parameter_buffer(hf_name: str) -> bool:
    """HF state-dict entries that are not parameters (masks, rotary tables) and are dropped."""
    return re.search(r"(^|\.)(attn|attention)\.bias$|masked_bias$|inv_freq$", hf_name) is not None
