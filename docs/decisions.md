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

## Gaps found while building A4 (2026-11-04; undecided)

| date | gap |
|---|---|
| 2026-11-04 | **A4 cannot be hosted in A3 as defined.** The explicit previous-token + induction circuit (handi/authoring/circuits.py, placed at the H6 heads) is verified in isolation: on Pythia-70M with every other layer's residual write silenced it copies 92% of a repeated random sequence (previous-token attention 1.00, induction attention 0.99). Inside the real A3 weight set it copies 0%: the spectrum-matched random layers write junk into the residual at 20-100x the token-embedding norm, so at the circuit's layers only ~3% of what a head can read is token identity (trained models have the same residual norms but keep token identity in dedicated subspaces; random layers pollute every subspace equally). Attenuating all non-circuit output writes (attn.o, mlp.down) by a factor s gives copy accuracy s=0.1: 1%, s=0.03: 59%, s=0.01: 89%. Options: (a) define A4 as A3 with non-circuit writes attenuated by a stated s, reported as such; (b) report A4 as not buildable on A3 and the circuit reading as validated only in isolation. Not built until decided. |
| 2026-11-04 | **No authored previous-token head for learned-position models (GPT-2, TinyStories).** A rank-hd bilinear form fitted on the authored position curve to a lag-1 target reaches 85% lag-1 argmax on pure positions (the same on the trained W_pos), but 5% inside the model: token embeddings project onto the position subspace at the random rate (trained GPT-2 measured 0.083 = 64/768, no orthogonality to exploit), and the lag-1 score gap is of the order of the score scale. RoPE models (Pythia, Qwen) have a clean design (constant q/k on the rotary pairs with a one-position phase offset) and are unaffected. Without an owner-approved design A4 is RoPE-only. |

## A4 definition (decided 2026-11-04)

- A4 = A3 + the explicit previous-token/induction circuit at the H6 heads, with every residual write that is
  not one of the two circuit heads (attention output of other heads, MLP down, DeltaNet out, their biases)
  scaled by 0.01. Reported as its own line with the factor stated; does not count toward H13 (pre-registered).
- A4 is built for the RoPE models only (Pythia-70M, Pythia-160M, Qwen3.5-0.8B). GPT-2 and TinyStories get no
  A4 row; the learned-position previous-token design stays an open gap.


## H15 / H16 and conditions A6 / A7 (decided 2026-11-04)

| date | decision |
|---|---|
| 2026-11-04 | Owner added H15 (cross-layer rotation rule) and H16 (golden-ratio / Fibonacci structured fill) after seeing H1-H14 and Rung 2 A0-A3. Text in hypotheses.md section 7. Measured by analysis/h15_rotation.py (H15) and Rung 2 conditions A6 (H15 applied) and A7 (H16), handi/authoring/generators.py. Post-freeze: reported as exploratory with their own nulls, not counted in the H13 verdict. |

Builder defaults for A6 / A7 (chosen before any run; may be overridden):
- **A6 chain.** One rotation R shared by readers and writers: R = exp(phi * A) with A the skew-symmetric generator of d/2 Haar-chosen 2-planes (so R rotates every plane by phi). Layer l's subspace S_l = R^l S_0 (readers of layer l read inside S_l, writers of layer l write inside S_l). (m, phi) by nested bisection on the closed-form expectation ov_j = (k / m^2) ||S_0^T R^j S_0||_F^2 (Haar k=32 bases inside S_l and S_{l+j}) so that ov_1 and the all-pair mean match the trained reader values from h15.csv; phi is clipped to [0, pi/2] and m to [32, d], and the achieved values are written to the build report. phi = 0 reduces A6 to A3.
- **A7 sequence.** u_n = frac((n + 2^24 * seed) * phi) consumed in row-major order across all draws in the order the builder makes them (deterministic), clipped to [1e-10, 1 - 1e-10], z = ndtri(u). Permutations of 1-D vectors = argsort of the next n values. Products of the sequence with the covariance square root and the QR sign fix are exactly as in A1.
- Both run on all eight models with 5 seeds, after A4.

## H16 / A7 validity note (builder diagnostic, 2026-11-04, why the golden-ratio fill scores better)

Pythia-70M seed 0, 20k WikiText tokens. A7's lower cross-entropy comes from the embeddings, not the weight
matrices: golden embeddings with Gaussian matrices score 27.7 nats (A7 27.4); golden matrices with Gaussian
embeddings 41.8 (A1 44.8). The golden-sequence N6 embedding has effective rank 22 of 512 (trained 440, Gaussian
N6 437): consecutive rows of frac(n phi) consumed row-major are shifts of one another, so token vectors collapse
into a ~22-dimensional set and the output distribution flattens (entropy 6.3 nats vs 0.9 for A1). At the best
output temperature every authored fill scores 10.85 nats = the uniform line (A0 10.83): none knows anything
about the next word; A7 is merely less confidently wrong. H16 passes its mechanical rule but the mechanism is
rank collapse of the fill, not structure that helps. Reported as such in results/summary.md.
