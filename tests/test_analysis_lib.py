"""Unit tests for handi.analysis (library only; no hypothesis scripts)."""
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from handi.analysis import nulls, registry, results, safetensors_np, spectra


def test_bf16_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    x32 = rng.standard_normal((37, 53)).astype(np.float32) * 3.0
    t_bf16 = torch.from_numpy(x32).to(torch.bfloat16)
    p = tmp_path / "w.safetensors"
    save_file({"w": t_bf16, "h": torch.from_numpy(x32).half(), "f": torch.from_numpy(x32)}, str(p))
    got = safetensors_np.load_tensor(p, "w")
    ref = t_bf16.to(torch.float32).numpy()
    assert got.dtype == np.float32 and got.shape == x32.shape
    assert np.array_equal(got, ref)
    assert np.array_equal(safetensors_np.load_tensor(p, "h"), torch.from_numpy(x32).half().float().numpy())
    assert np.array_equal(safetensors_np.load_tensor(p, "f"), x32)
    assert safetensors_np.tensor_names(p) == ["f", "h", "w"]
    assert set(safetensors_np.load_all(p, filter=lambda n: n != "f")) == {"h", "w"}
    assert safetensors_np.open_checkpoint(p) == [p]


def test_n1_preserves_frobenius_norm():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((40, 70)).astype(np.float32) * 0.3
    Z = nulls.n1_gaussian(W, rng)
    assert Z.shape == W.shape
    assert np.isclose(np.linalg.norm(Z), np.linalg.norm(W), rtol=1e-5)


def test_n5_spectrum_matched_and_haar():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((60, 90)).astype(np.float32)
    W[:5] *= 10  # some spikes
    s_w = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    Z = nulls.n5_spectrum_matched(W, rng)
    s_z = np.linalg.svd(Z.astype(np.float64), compute_uv=False)
    assert Z.shape == W.shape
    assert np.allclose(s_z, s_w, rtol=1e-4)
    U = nulls.haar_stiefel(200, 50, rng)
    assert np.allclose(U.T @ U, np.eye(50), atol=1e-10)
    # Haar invariance sanity: column entries have mean ~0 and variance ~1/m
    assert abs(U.mean()) < 0.01
    assert np.isclose(U.var() * 200, 1.0, rtol=0.05)


def test_mp_fit_on_gaussian():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((2000, 1000)) * 0.5
    eigs = spectra.esd(W, use_gpu=False)
    gamma = spectra.aspect(W)
    assert gamma == 0.5
    ks = spectra.ks_to_mp(eigs, gamma, 0.25)
    assert ks < 0.02
    assert spectra.spike_count(eigs, gamma) <= 1
    sigma2, _ = spectra.estimate_sigma2(eigs, gamma)
    assert np.isclose(sigma2, 0.25, rtol=0.05)
    # closed form: pdf integrates to 1
    a, b = spectra.mp_edges(gamma, 0.25)
    xs = np.linspace(a, b, 200001)
    assert np.isclose(np.trapezoid(spectra.mp_pdf(xs, gamma, 0.25), xs), 1.0, atol=2e-3)


def test_powerlaw_fit_recovers_alpha():
    rng = np.random.default_rng(4)
    alpha = 3.0
    xmin = 2.0
    x = xmin * (1.0 + rng.pareto(alpha - 1.0, size=5000))  # pdf ~ x^-alpha for x >= xmin
    fit = spectra.powerlaw_fit(x)
    assert abs(fit["alpha"] - alpha) < 0.3
    assert fit["n_tail"] >= 10
    fit_fixed = spectra.powerlaw_fit(x, xmin=xmin)
    assert abs(fit_fixed["alpha"] - alpha) < 0.1


def test_spectral_summaries():
    s = np.array([3.0, 1.0, 1.0])
    assert np.isclose(spectra.participation_ratio(s), 121.0 / 83.0)
    assert np.isclose(spectra.sv_entropy(np.array([1.0, 1.0])), np.log(2))
    W = np.diag([2.0, 1.0, 1.0])
    assert np.isclose(spectra.stable_rank(W, use_gpu=False), 6.0 / 4.0)


def test_canonical_shapes_gpt2_real_config():
    cfg = registry.load_config("gpt2")
    shapes = nulls.canonical_shapes("gpt2", cfg)
    assert shapes["embed.tok"] == (50257, 768)
    assert shapes["embed.pos"] == (1024, 768)
    layers = {int(k.split(".")[1]) for k in shapes if k.startswith("layers.")}
    assert layers == set(range(12))
    assert shapes["layers.0.attn.o.w"] == (768, 768)
    assert shapes["layers.0.mlp.up.w"] == (3072, 768)
    assert shapes["layers.0.mlp.down.w"] == (768, 3072)
    assert "layers.0.mlp.gate.w" not in shapes


def test_canonical_shapes_qwen_real_config():
    cfg = registry.load_config("qwen3.5-0.8b")
    shapes = nulls.canonical_shapes("qwen3_5", cfg)
    assert shapes["embed.tok"] == (248320, 1024)
    assert "embed.pos" not in shapes
    assert shapes["layers.3.attn.q.w"] == (8 * 256, 1024)
    assert shapes["layers.3.attn.k.w"] == (2 * 256, 1024)
    assert "layers.0.attn.q.w" not in shapes
    assert shapes["layers.0.linear.in_qkv.w"] == (2 * 16 * 128 + 16 * 128, 1024)
    assert shapes["layers.0.linear.out.w"] == (1024, 16 * 128)
    assert shapes["layers.0.mlp.gate.w"] == (3584, 1024)
    assert "ln_f.b" not in shapes


def test_n4_fresh_init_small():
    cfg = {"n_embd": 16, "n_layer": 2, "n_head": 2, "n_positions": 8, "vocab_size": 10}
    sd = nulls.n4_fresh_init("gpt2", cfg, np.random.default_rng(0))
    assert set(sd) == set(nulls.canonical_shapes("gpt2", cfg))
    assert np.all(sd["layers.0.ln1.w"] == 1) and np.all(sd["layers.0.ln1.b"] == 0)
    assert np.array_equal(sd["unembed.w"], sd["embed.tok"])
    neox = {"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "max_position_embeddings": 8, "vocab_size": 10, "intermediate_size": 64}
    sd2 = nulls.n4_fresh_init("gpt_neox", neox, np.random.default_rng(0))
    assert "embed.pos" not in sd2
    assert not np.array_equal(sd2["unembed.w"], sd2["embed.tok"])


def test_results_writer(tmp_path: Path):
    p = tmp_path / "h01.csv"
    with results.ResultsWriter(p) as w:
        w.row(hypothesis="H1", model="gpt2", family="gpt2", layer=0, component="attn.q",
              metric="alpha", value=2.5, null_kind="N1", null_n=20, null_mean=4.0, null_sd=0.1, z=-15.0,
              n_rows=768, n_cols=768, src_dtype="F32", seed=0)
    lines = p.read_text().splitlines()
    assert lines[0] == ",".join(results.COLUMNS)
    assert len(lines) == 2
    row = dict(zip(results.COLUMNS, lines[1].split(",")))
    assert row["metric"] == "alpha" and row["value"] == "2.5" and row["timestamp"]
    # append keeps a single header
    with results.ResultsWriter(p) as w:
        w.row(hypothesis="H1", model="gpt2", metric="x", value=1)
    assert p.read_text().count("hypothesis,model") == 1
    out = results.save_raw("h01", "gpt2", "layers.0/attn.q", root=tmp_path, s=np.arange(3))
    assert out.exists() and np.load(out)["s"].tolist() == [0, 1, 2]
