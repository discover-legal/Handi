# Handi: hand-authored LLM weights, zero gradient descent

## Context

This project tests whether a frontier model (Claude) can hand-author the weights of a
transformer language model: no training, every weight emitted by code from an explicit design.
Prompted by Takehi, Teoh, Zhang & Khattab (MIT CSAIL, Nov 2026, "Sparks of Weight-Space
Programming"), where an LLM wrote a 1.48M-weight Go policy as a 336-line program and beat a
PPO-trained net. Nobody has written a language model this way; the paper's Gemma caveat is that
weight edits "could not add an ability the model lacked". This is a science experiment, not a
product. The ultimate target is **law tasks (LegalBench)**, not math. Repo is empty.

## Decisions (3 Nov 2026)

- **Python only, GPU.** No Rust. PyTorch CUDA wheel for the RTX 5060 (Blackwell, cu128).
- **Runtime = Hugging Face transformers model classes.** Authored state dicts load straight into
  the real architecture classes, so KV cache, GPU, generation, and exact fidelity to real LLMs
  come for free. Our code is only weight authoring and analysis.
- **Rung 1 adder runs on the Qwen3.5 architecture**, not GPT-2.
- **Ladder:** Rung 1 tiny Qwen-architecture adder (pipeline proof). Rung 2 weights for real
  LM shapes authored from patterns found in real weights. No grammar-chatbot rung.
- **Analysis targets, all three tiers, as real science with hypotheses and nulls:**
  GPT-2 small + Pythia-70M/160M; Qwen3.5-0.8B; TinyStories 1M/3M/8M/33M.
- **Corpora:** WikiText-103 test + TinyStories validation for cross-entropy; **LegalBench** as
  the law benchmark.
- **Token-id rank admissible** as a frequency proxy in authored embeddings (condition A2),
  reported on its own line.
- Honest framing carried in the plan: under the null, pattern-authored weights do no better than
  spectrum-matched random matrices. A positive result must beat that control.

## Machine

24 cores, 62 GB RAM, RTX 5060 8 GB, 485 GB free. System Python 3.12 has no numpy;
`uv` at ~/.local/bin/uv.

## Confirmed architecture facts

**Qwen3.5-0.8B** (from its HF config.json and model card): `model_type qwen3_5_text` inside a
multimodal wrapper with a `text_config`; hidden 1024; 24 layers = 6 blocks of
[3x Gated DeltaNet + FFN, 1x gated full attention + FFN] (`full_attention_interval 4`,
`layer_types` list); full attention: 8 query heads, 2 KV heads (GQA), head_dim 256, q/k RMSNorm,
RoPE; linear attention: 16 heads, key/value head dim 128; FFN SwiGLU intermediate 3584; RMSNorm;
vocab 248320, tied embeddings; bf16 on disk. The multimodal wrapper and vision tower are ignored:
load the text model only. Gated DeltaNet in transformers has a pure-torch fallback; the
`flash-linear-attention` / `causal-conv1d` kernels are optional speedups.

Other families (GPT-2 Conv1D layout, Pythia fused interleaved QKV + 25% rotary + parallel
residual, GPT-Neo no attention scaling + alternating local window 256, TinyStories likely
`.bin`-only) are handled by transformers itself; the analysis layer maps each family's state
dict to one canonical schema.

## Part A: authoring library + Rung 1 adder (Python)

### Layout

```
Handi/
  pyproject.toml                 uv project; deps: torch (cu128), transformers, safetensors,
                                 huggingface_hub, numpy, scipy, matplotlib, datasets, tokenizers
  handi/
    author/
      mat.py                     design primitives on torch tensors: zeros, identity, one_hot_row,
                                 set_block, block_diag, hconcat/vconcat, scaled, add_outer, permutation
      space.py                   Space: named slots in the residual stream (alloc, embed, basis)
      head.py                    HeadProgram: qk.match / qk.bias_from_key / qk.table,
                                 ov.copy / ov.copy_affine; packs into q/k/v/o (+ GQA head mapping)
      mlp.py                     Lookup (one hidden neuron per truth-table entry) and Piecewise
                                 steps, emitted for SwiGLU (gate/up/down) and plain GELU/ReLU MLPs
      norms.py                   RMSNorm / LayerNorm absorption: given a design where every
                                 position carries a known slot-norm, compute the gain vector that
                                 makes the norm a fixed affine map; q/k-norm gains for Qwen attention
      export.py                  canonical schema -> HF state dict per family (qwen3_5, gpt2,
                                 gpt_neox, gpt_neo); load_into(model_class, config, weights)
    canon.py                     canonical tensor schema shared by author/ and analysis/ ;
                                 HF state dict <-> canonical for all four families
    adder/
      design.py                  the Rung 1 adder, built from author primitives, emits a tiny
                                 Qwen3.5 text config + state dict
      run.py                     add(a, b): format prompt, generate 11 tokens with KV cache, parse
      tokenizer.py               13-symbol digit tokenizer (0-9, +, =, BOS) as a HF PreTrainedTokenizer
  tests/
    test_author_primitives.py
    test_adder.py                10 AdderBoard edge cases + 10k pairs from random.Random(2025), 100%
    test_canon_roundtrip.py      HF -> canonical -> HF equals original for each family
    fixtures/adder_pairs.json
  tools/download_models.py       pinned snapshot_download of all 7 checkpoints
  models/                        gitignored, HF_HOME
```

### Rung 1 adder on Qwen3.5 architecture

Tiny `Qwen3_5TextConfig`: `layer_types` all `full_attention` (no DeltaNet layers; authoring a
delta-rule recurrence is a separate, harder problem and is listed as a gap below), hidden 64,
2 layers, 2 query heads / 2 KV heads (GQA trivial), head_dim 16, intermediate 200, RMSNorm,
SwiGLU, RoPE, q/k RMSNorm, vocab 13, untied unembed. Everything the real Qwen attention does
must be absorbed by the design:

- **RMSNorm before attention/MLP:** every position's residual is built to carry the same number
  of unit-magnitude active slots, so RMSNorm is a constant scale; the gain vector undoes it.
  Verified by a test asserting the post-norm vector equals the designed one.
- **q/k RMSNorm per head:** normalizes q and k, so attention logits are bounded by the q/k norm
  gains; the margin constant S lives in those gains instead of in W_Q/W_K magnitudes.
- **RoPE:** rotates q/k by position. The design uses position-dependent matching, so either
  (a) place matching slots in the highest-frequency rotary dims and absorb rotation, or
  (b) use `rope_theta` huge so rotation over 34 positions is negligible for the matching dims.
  Default (b), with the exact angle budget asserted in a test; (a) is the fallback.
- **SwiGLU:** Lookup neurons use gate = up = the matching pattern so `silu(g) * u` acts as a
  sharp AND; `down` writes the output vector.
- **Attention output gate (Qwen3.5 gated attention):** gate projection set so the gate is open
  (constant) at every position.

Format and circuit (unchanged): `BOS a0..a9 + b0..b9 = s0..s10`, MSB-first, 34 positions,
vocab 13. Slots: const.one, tok.digit[10], tok.kind[3], pos.col[11], pos.seg[3], work.a[10],
work.b[10], flags G/P[2], carry[2], out.digit[10] (62 of 64 dims). Layer 1 heads A/B fetch the
a- and b-digit of the same column; MLP1 flags Generate (a+b>=10) / Propagate (a+b==9).
Layer 2 head D is the carry-lookahead: attends from the prediction position to the nearest lower
column that generates a carry and is not blocked by a propagate chain, BOS as the no-carry sink;
MLP2 computes (a+b+carry) mod 10. Unembed reads out.digit, non-digit rows biased down.
Since Qwen has no learned positional embedding, column/segment slots are written by a
position-indexed table only during design validation; in the real model they must come from
RoPE-based matching, which is why option (a) above is the honest fallback if (b) fails. If
neither yields the column slots, that is a gap to report, not a substitute to invent.

AdderBoard verification (fetched from its README/verify.py): pairs uniform in
[0, 9_999_999_999], >= 99% on 10 fixed edge cases + 10,000 pairs from `random.Random(2025)`;
forward() must be tensor-in logits-out with no task-specific control flow; same inference code
must work with other weights. Using the stock HF class satisfies the last two by construction.

### Verification (Part A)

- `tests/test_author_primitives.py`: shapes; a Lookup over two 3-way one-hots reproduces its
  table under the real SwiGLU MLP module; norm absorption test; RoPE angle-budget test.
- `tests/test_canon_roundtrip.py`: for each family, HF state dict -> canonical -> HF is exact
  and the model's logits are unchanged.
- `tests/test_adder.py`: build, save as safetensors, load into `Qwen3_5ForCausalLM` (text),
  generate with `use_cache=True`, 100% on edge cases + 10k pairs; attention-weight assertions
  (> 0.99 at the intended key via `output_attentions`).
- CLI `python -m handi.adder.run 123 456`.

### Part A steps

1. `uv init`, deps, confirm torch sees the GPU, `transformers` version supports `qwen3_5`.
2. `tools/download_models.py`; dump every config + tensor-name list to `docs/models/`.
3. `handi/canon.py` + roundtrip tests on GPT-2 and TinyStories-1M (fast), then Pythia, Qwen.
4. `handi/author/{mat,space,head,mlp,norms,export}.py` + primitive tests.
5. `handi/adder/{tokenizer,design,run}.py` + adder test. **Rung 1 done.**

## Part B: weight analysis + Rung 2 protocol

### Governing rule

Every measurement is computed identically on trained weights and on a family of null weight
sets. A pattern is "confirmed" only if the trained value is outside the null by a pre-registered
margin (|z| >= 5 over >= 20 null draws, or KS > 3x the null-vs-null KS) and reproduces in >= 2 of
the 3 model families. Hidden-dim axes are permutation-symmetric, so any "picture" of a weight
matrix has no meaningful x/y order; ordered axes are position (learned pos), RoPE frequency index,
and token id. The exotic hypotheses get their fair test on those axes.

### Hypotheses (frozen in `docs/hypotheses.md` before any trained-model result is produced)

| # | Hypothesis | Test | Null | Prior |
|---|---|---|---|---|
| H1 | Spectra are heavy-tailed with spikes above the Marchenko-Pastur edge (Martin & Mahoney) | SVD, power-law fit, spike count, KS to MP | N1, N2, N4 | ~95% |
| H2 | Function lives in the top-k singular directions: randomize bulk vs randomize spikes, compare held-out loss | loss under both ablations | N4 | ~75% MLP/OV |
| H3 | QK/OV circuits are low stable-rank with identity-like component (Trockman & Kolter mimetic init) | stable rank, cos_F to I | N1 products | ~85% / ~70% |
| H4 | Token embeddings anisotropic, PC1 correlates with token frequency | mean cosine, PCA vs MP, Spearman vs token id | N1, N6 | ~95% / ~80% |
| H5 | Outlier residual dims consistent across layers (massive activations) | norm kurtosis per dim, cross-layer overlap | N1, N4 | ~90% |
| H6 | Previous-token + induction heads identifiable from weights alone (Olsson et al.) | positional score matrix offset -1, K-composition, copy trace | N1, N3 | ~95% |
| H7 | Cross-layer alignment of reader/writer subspaces beyond k/d | Grassmann overlap, also with outlier dims removed | Haar, N5 | ~80% / ~50% |
| H8 | Spectral shape universal across model size | normalized ESD KS across sizes | N1 per size | ~55% |
| H9 | **Fractal / self-similar structure** (project hypothesis): Hurst/DFA, block-variance slope, box-counting on ordered axes vs row/col-permuted self | DFA, box-counting | N3 | ~10%, ~0% on hidden axes |
| H10 | Learned positions smooth and low-dim; RoPE heads frequency-specialized | PCA, neighbor ratio, FFT, per-band QK energy | N3, N1 | ~90% |
| H11 | **Number/modular/elliptic-curve structure** (project hypothesis): helix fit with periods 2/5/10/100 (Kantamneni & Tegmark), DFT peaks (Nanda), implicit cubic fit vs equal-param helix | R^2, DFT, cubic residual | N7, N1 | helix ~75%; cubic <2% |
| H12 | MLP neuron geometry shows superposition signature; in/out cosine biased | pairwise cosine tails, per-neuron cos | N1, N5 | ~80% |
| H13 | **Capstone (Rung 2):** weights authored from confirmed second-order patterns beat shape-matched random init on held-out loss | A-ladder below | A0/A1 | ~10% |

Qwen-specific additions: Gated DeltaNet layers' projection matrices get H1/H3/H5/H7/H12; the
DeltaNet state-decay and gate parameters get their own descriptive table (no hypothesis yet,
reported as observations).

Nulls: N1 Gaussian iid same shape/norm; N2 element-shuffle; N3 row/col permutation (same
spectrum, destroyed order); N4 fresh init per family's init rule; N5 spectrum-matched random
`U diag(sigma) V^T` with Haar U,V; N6 covariance-matched embeddings; N7 token-assignment
shuffle. All seeded, >= 20 draws (100 for TinyStories).

### Analysis layout

```
docs/hypotheses.md                 frozen pre-registration
docs/results-schema.md, docs/decisions.md, docs/models/<model>.md (config + tensor list)
handi/analysis/{registry,safetensors_np,nulls,spectra,subspaces,ordered,numbers,stats,results,plots}.py
analysis/h01_spectra.py ... h12_mlp_geometry.py, run_all.py
handi/authoring/{generators,circuits,evaluate}.py     Rung 2
results/{manifest.json, tables/hNN.csv, tables/rung2.csv, raw/, figures/, summary.md}
```

Own safetensors header reader (bf16 via uint16<<16) so raw tensors never pass through a model
class during analysis. Results are long-format CSV with null mean/sd/n/z per row;
`summary.md` is generated, never hand-edited. Plots follow the dataviz skill. SVDs on GPU via
torch where it helps (Qwen 248k x 1024 embedding via the Gram matrix).

Cost: downloads < 4 GB; whole hypothesis suite a few GPU-hours; Rung 2 evaluation a few hours.

### Rung 2 protocol (H13)

Conditions, each adds one confirmed pattern; every set is a full state dict loaded into the
real HF class and run with KV cache:
- **A0** fresh init (control). **A1** spectrum-matched random everywhere + covariance-matched
  embeddings (control; pre-registered expectation: equals A0).
- **A2** A1 + ordered-axis patterns: smooth W_pos with matched PC spectrum (H10), outlier dims
  placed consistently (H5), embedding common direction scaled by token-frequency rank (H4,
  admitted post-freeze, reported separately).
- **A3** A2 + cross-layer subspace alignment fitted to measured overlaps (H7) + identity
  components in QK/OV at measured cos_F (H3).
- **A4** A3 + explicit previous-token and induction heads where the analysis finds them (H6),
  built with `handi/author`. Validates the circuit-reading; its success does not count toward
  H13 and is reported separately as "function from an explicitly authored circuit".
- **A5** trained (upper bound), **A5'** bulk randomized, **A5''** spikes randomized (H2).

Metrics: **M1** held-out cross-entropy on WikiText-103 test and TinyStories validation with
bootstrap CI, reference lines uniform ln V, unigram, bigram, trained; **M2** copy accuracy on
repeated random-token sequences; **M3** sampled-text bigram KL vs corpus and dictionary-word
fraction; **M4 LegalBench** accuracy on a fixed subset of binary-label tasks (chosen at step 1
of Part B from the `nguha/legalbench` task list, favoring short-prompt classification tasks),
with the trained Qwen3.5-0.8B score as the upper bound and majority-class as the floor.
5 seeds per condition.

Pre-registered outcomes: null = A1-A3 within 0.1 nat of A0 and chance on M2/M4.
Positive-weak = some A_k (k<=3) beats A1 by >= 0.2 nat with non-overlapping CIs in two families.
Positive-strong = some A_k (k<=3) reaches the unigram line without token-identity input (~5%).
Law-positive = any authored A_k (k<=4) beats majority-class on any LegalBench task with CI.

### Part B steps

1. Registry, pinned downloads, manifest, `docs/models/*.md`; pick the LegalBench task subset
   and record it in `docs/decisions.md`.
2. `safetensors_np.py` + canonical loading; sanity check against transformers logits per family.
3. `nulls.py`, `spectra.py`, `stats.py`, `results.py`, `plots.py`; unit-check MP closed form.
4. Freeze and commit `docs/hypotheses.md`.
5. Run h01, h03, h04, h05, h12. `summary.md` v1.
6. Run h06, h07, h10 (feed Rung 2 directly).
7. Run h09, h11, h08.
8. Run h02 on GPT-2, Pythia-160M, TinyStories-33M, Qwen3.5-0.8B.
9. Rung 2: generators A0-A4 for all four families, load into HF classes, evaluate M1-M4,
   `rung2.csv`.
10. Final `summary.md` and H13 decision against the pre-registered outcomes.

## Routine defaults (may be overridden)

- AdderBoard 10k pairs dumped once from `random.Random(2025)` into `tests/fixtures/`.
- Adder uses full-attention-only Qwen3.5 layers; DeltaNet-layer authoring is a listed gap.
- Thresholds as stated. TinyStories `.bin` -> safetensors conversion allowed, one time.
- Qwen H11 run as the digit-level adaptation (Qwen tokenizes digits singly).
- torch cu128 wheel; fall back to CPU automatically if the GPU is unsupported by the wheel.

## Gaps (not solvable with what was decided)

- Hand-authoring a Gated DeltaNet layer (delta-rule recurrence) has no design yet. Rung 1 avoids
  it; Rung 2 conditions for Qwen fill DeltaNet layers with spectrum-matched random only.
- If RoPE-based column matching (options a/b above) cannot produce clean column slots at d=64,
  the adder needs a different positional scheme; I will report, not substitute.

## Sequencing

1. Part A steps 1-5 (Rung 1 adder, 100% on AdderBoard set). Commit.
2. Part B steps 1-4. Commit.
3. Part B steps 5-8 (the science). Commit per hypothesis table.
4. Part B steps 9-10 (Rung 2). Commit.

## Verification (end to end)

- `uv run pytest` green; `python -m handi.adder.run` prints 100% on edge cases + 10k pairs, run
  through the stock Qwen3.5 class with KV cache on the GPU.
- Canonical roundtrip tests pass on all four families; trained models generate normally after
  a roundtrip, proving the loader is faithful.
- `uv run analysis/run_all.py` regenerates every table and `summary.md` from a seed.
- Rung 2 table shows A0..A5 on M1-M4 with CIs; H13 verdict follows the pre-registered rule.
