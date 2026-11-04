# Decisions log

Decisions were taken by the project owner; defaults the owner did not set were taken by the builder (Claude Fable 5.1) and are marked as such. Dates are when the decision was made.

| date | decision |
|---|---|
| 2026-11-03 | Project: hand-author LLM weights with zero gradient descent, as a science experiment; ultimate target law tasks (LegalBench), not math. |
| 2026-11-03 | Python only, GPU (RTX 5060, torch cu128). Rust rejected. |
| 2026-11-03 | Runtime = stock Hugging Face transformers model classes; authored state dicts load into them (KV cache, fidelity to real LLMs). |
| 2026-11-03 | Rung 1 adder runs on the Qwen3.5 architecture, MSB-first with carry-lookahead. |
| 2026-11-03 | Ladder: Rung 1 adder, then Rung 2 weights authored from patterns found in real weights. No grammar-chatbot rung. |
| 2026-11-03 | Analysis models, all three tiers: GPT-2 small, Pythia-70M/160M; Qwen3.5-0.8B; TinyStories 1M/3M/8M/33M. Designed as real science with hypotheses and nulls. |
| 2026-11-03 | Qwen3.5-0.8B is a full engine target (its architecture runs via the stock class), not analysis-only. |
| 2026-11-03 | Corpora: WikiText-103 test + TinyStories validation, every model scored on both. LegalBench as the law benchmark (metric M4). |
| 2026-11-03 | Token-id rank is admissible as a frequency proxy in authored embeddings (condition A2), reported separately. |

## Routine defaults taken by the builder (may be overridden)

| date | default |
|---|---|
| 2026-11-03 | AdderBoard test pairs dumped once from `random.Random(2025)` into `tests/fixtures/adder_pairs.json`. |
| 2026-11-03 | Adder uses full-attention-only Qwen3.5 layers (d=128, 16 heads x 32, partial RoPE 0.25, rope_theta 10, SwiGLU 200). Gated DeltaNet authoring is a listed gap. |
| 2026-11-03 | Null thresholds: 20 draws (100 for TinyStories), abs(z) >= 5, KS > 3x null median. |
| 2026-11-03 | TinyStories `pytorch_model.bin` converted once to safetensors with torch. |
| 2026-11-03 | Qwen H11 run as the digit-level adaptation. |

## Gaps (cannot be built from what is decided)

| date | gap |
|---|---|
| 2026-11-03 | No design exists for hand-authoring a Gated DeltaNet (delta-rule recurrence) layer. Rung 1 avoids it; Rung 2 fills Qwen DeltaNet layers with spectrum-matched random only. |

## LegalBench subset for metric M4 (builder default, chosen 2026-11-03 before any Rung 2 run)

Ten binary tasks with short inputs (average under 500 characters) and at least ~90 test items,
evaluated zero-shot with each task's LegalBench base prompt, scored by comparing the log-probability
of the two label strings; at most 500 items per task (fixed seed subsample where larger):
`hearsay`, `overruling`, `nys_judicial_ethics`, `definition_classification`,
`citation_prediction_classification`, `proa`, `diversity_1`, `diversity_6`, `cuad_governing_law`,
`contract_nli_limited_use`.

## Amendments and validity notes (after the freeze; dated)

| date | note |
|---|---|
| 2026-11-03 | H9 box-counting statistic is INVALID as implemented: the pre-registered hidden-dim control (must be null) gives z of 24 to 71 in tiers A/B because the N3 null sd is degenerate (~1e-4). Box-counting rows in h09.csv are discounted; the H9 verdict rests on DFA/Hurst and block-variance only, which show smoothness (H > 1, slopes -0.4) on W_pos, token-id and RoPE-frequency axes, i.e. long-range correlation that the pre-registration assigns to H10, not self-similarity. Reported as: no evidence of fractal geometry. |
| 2026-11-03 | H6 K-composition N1 null is not discriminative (57% of GPT-2 pairs pass 5 sigma). The H6 verdict rests on the previous-token statistic (RoPE form picks the true heads in both Pythias) and on the attention/M2 confirmation. Weight-only induction candidates coincide with attention-confirmed heads only in Pythia. |
| 2026-11-03 | H6 RoPE statistic implemented as tr(W_Q R_{-delta} W_K^T C); the pre-registered expression was dimensionally inconsistent. |
| 2026-11-04 | Owner added H14 (spectral mixture decomposition) as a post-freeze hypothesis after seeing H1-H11 results; text in hypotheses.md section 7; measured by analysis/h14_mixture.py. Not pre-registered: reported as exploratory with its own nulls. |

## Rung 2 builder defaults for A2 / A2F / A3 (chosen 2026-11-04 before any A2/A3 run; may be overridden)

- **A2 = A1 + H10 + H5.** W_pos: uncentered SVD of the trained W_pos; every position-series PC keeps its |FFT|
  with random phases (DC and Nyquist kept real), no re-orthonormalization (it destroys the smoothness),
  residual directions Haar. Outlier dims: one random permutation of residual dims per seed shared by every
  layer; every reader matrix's column norms and writer matrix's row norms, and every residual-indexed vector
  (LN gains/biases, o/down biases), are set to the trained profile at the permuted dims; the N6 embedding's
  columns are permuted the same way so the covariance's large dims coincide.
- **A2F = A2 + H4.** PC1 of the N6 embeddings (oriented along the mean) is re-assigned across tokens so that
  its Spearman correlation with token id equals the trained value (rank assignment by the smoothed trained
  PC1-vs-id curve plus noise, noise level bisected to the measured correlation). Reported on its own line, as
  pre-registered; A3 builds on A2, not A2F.
- **A3 = A2 + H7 + H3.** Shared residual subspace: one Haar basis of dimension m = round(32 / overlap) with
  overlap = h07 `readers:all_crosslayer:full` at k=32 (GPT-2 m=267, Pythia 384, Qwen 464, TinyStories-1M 62);
  the top-32 right singular vectors of every reader and left singular vectors of every writer are Haar inside
  that subspace, the rest Haar in its complement; singular values from N5. Identity components: for each head,
  Q is rebuilt from K (same head-dim basis, right subspace = beta*V_K + sqrt(1-beta^2)*Haar, beta bisected so
  that cos_F(W_QK, I) equals the head's h03 value) and O from V likewise for W_OV; Q and O singular values
  from the trained blocks. Achieved cos_F within 0.005 of the measured value on average after the H5 rescale.
- Known deviation: the authored GPT-2 position curve has neighbour ratio 0.10 against 0.04 trained (random
  phases lose the trained phase coherence across PCs); TinyStories within 0.06.

