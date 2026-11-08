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

## H17 / H18 / H19 and condition A8 (decided 2026-11-05)

| date | decision |
|---|---|
| 2026-11-05 | Decided: the three untested structures (weights against data; nonlinear within-matrix structure; hidden / intrinsic dimension) to be built. Added as H17, H18, H19 in hypotheses.md section 7, measured by analysis/h17_direct_path.py, h18_within_matrix.py, h19_intrinsic_dim.py; H17 applied as Rung 2 condition A8 (handi/authoring/generators.py, handi/authoring/bigram.py). A8 is data-derived (corpus bigram counts), so it leaves Rung 2's weights-only premise; reported on its own line. |

Builder defaults for A8 (chosen before any run; may be overridden): lambda = 0.7 and add-0.5 unigram as in the M1 bigram reference; rank = d; randomized range finder with 10 power iterations and 16 oversampling columns; temperature grid 0.05..500 (33 log-spaced values) fitted on train tokens 100k-200k; final-norm gain clamped to |g| >= 0.05 before division; write attenuation 0.01 as A4 (owner's factor), applied to attn.o, mlp.down, linear.out and their biases.

### A8 run note (2026-11-05, after the eight-model run)

- Untied models (Pythia-70M, Pythia-160M): the rank-d factorisation fits the smoothed log-bigram target
  (block explained variance 0.999); direct-path CE 7.17–7.47 on WikiText, below the unigram line (7.50)
  and far below A0 (11.0); still above the bigram line (6.10). Full-model CE equals the direct-path CE:
  the attenuated layers above add nothing. Copy accuracy 0; LegalBench at majority class.
- Tied models (GPT-2, TinyStories 1M–33M): the tied branch fails. E E^T is positive semidefinite, but the
  symmetrised log-bigram target is indefinite (its dominant eigenvalue is negative, from the rank-1 log p
  term); keeping only the positive eigenpairs gives block explained variance −55 and direct-path CE 10.25,
  i.e. the uniform line. The tied A8 numbers (9.9–10.6 nats) therefore show no bigram knowledge; they pass
  the positive-weak test against A3 and A0 only because the spectrum-matched fills are far worse than fresh
  and the tied A8 embedding is near-uniform. Gap: a tied factorisation that can carry a sign (for example
  via a signed final-norm gain) was not designed; it was not decided.
- Qwen3.5-0.8B: the A8 build ran out of GPU memory in the fit-quality diagnostic block of
  bigram.factorise (a dense V x 2048 one-hot matrix at V = 151936), before any cell was scored. Qwen is
  tied, so the same gap applies. No Qwen A8 rows.

| date | decision |
|---|---|
| 2026-11-05 | Decision: the signed final-norm gain for the tied A8 branch: (T+T^T)/2 ~ E diag(sign) E^T from the d eigenpairs of largest absolute value; sign installed in ln_f.w (Qwen: 1 + w). GPT-2 seed 0: fit 0.999, direct-path CE 7.61, 381 of 768 gains negative. |
| 2026-11-05 | Decision: A8 zeroes the residual writes and the position embeddings. A8 now zeroes every residual write (attn.o, mlp.down, linear.out, with biases) and embed.pos instead of scaling by the A4 factor 0.01. Reason: on GPT-2 the x0.01 writes cost 0.6 nat and the A3 position fill 0.3 nat over the direct path (8.44 vs 7.61); with both zero the full model equals its direct path (7.60). All A8 rows re-run under this definition. |

### A8 rerun note (2026-11-06, signed tied gain, writes and position embeddings zeroed)

All eight models, 5 seeds. Factorisation fit 0.999 everywhere (tied models: 381 of 768 gains negative on
GPT-2 and TinyStories-33M, 510 of 1024 on Qwen). Full-model WikiText CE now equals the direct-path CE in
every model. Means over seeds, WikiText: Pythia-160M 7.27, Pythia-70M 7.39 (below the unigram line 7.50);
GPT-2 7.61, TinyStories-33M 7.60, 8M 7.66, 3M 7.76, 1M 7.92 (unigram 7.50); Qwen 7.66 (unigram 7.30).
Bigram line 5.9–6.2 in all. Copy accuracy 0; LegalBench at majority class. The tied gap to the unigram
line is the loss of the symmetrisation (T + T^T)/2: a tied model's logits must be symmetric in (a, b).

## Rung 3: closed-form edits to the trained Qwen3.5-0.8B for LegalBench (decided 2026-11-07)

| date | decision |
|---|---|
| 2026-11-07 | Owner asked whether the trained Qwen3.5-0.8B can be hand-tweaked, no training, to do well on LegalBench, and whether the ability is still present. Rung 3 = closed-form weight edits of the unembedding rows of the two label tokens, fit on data, no gradient step. Script analysis/run_rung3.py, rows in results/tables/rung3.csv (hypothesis R3). Data-derived like A8: reported on its own line. |

Builder defaults (may be overridden): the edit is applied to an untied copy of lm_head (same class, tie flag
off) so the input embeddings of the label words in the few-shot prompt are untouched. Conditions E0 (as is),
E1 (contextual calibration: equal mean label log-prob on the fit fold, written as row_t += delta_t mu_hat with
mu_hat the unit mean final-norm state), E2 (E1 + ridge readout from the final-norm state at the answer position
to the label, written into the label rows, constant term through mu_hat). Two folds by item parity on the M4
test sample; every item is scored by the edit fit on the other fold. Ridge lambda = n_fit on standardised
features; alpha = 2 x sd(label-logit gap) / sd(readout) on the fit fold. Diagnostic: per-layer ridge probe on
the residual stream at the answer position (not an edit). First-label-token scoring (argmax over the two label
logits equals the M4 argmax for single-token labels; multi-token labels flagged in the log). The fold-0 E2 edit
is also applied to the real lm_head and rescored with the M4 scorer on one task; the two differ only through the
constant term, carried by mu_hat whose projection varies slightly per item.

### Rung 3 run note (2026-11-07)

All ten tasks, two-fold. Mean accuracy over tasks: majority 0.546; E0 (as is) 0.568; E1 (calibration) 0.673;
E2 (calibration + readout written into the label rows) 0.775; best per-layer probe 0.842. E2 is at or above
0.83 on overruling (0.94), definition_classification (0.97), proa (0.85), diversity_1 (0.91), cuad_governing_law
(1.00), contract_nli_limited_use (0.83); 0.74 on diversity_6; 0.63 on hearsay; at or below majority on
nys_judicial_ethics (0.56) and citation_prediction_classification (0.32, n = 108, label not decodable at any
layer). Where the label is linearly decodable, it is decodable from the middle layers (9 to 19) at least as
well as from the final state. The fold-0 E2 edit applied to the real lm_head and rescored by the M4 scorer on
hearsay gives 0.638 vs 0.617 offline (constant term through mu_hat). Fix during the run: logits computed at the
last position only (full-sequence logits ran out of GPU memory on contract_nli_limited_use).

## Rung 4, "chords": trigram conjunction neurons on the A8 direct path (decided 2026-11-08)

| date | decision |
|---|---|
| 2026-11-08 | Decided: the next chord beyond a bare note: conjunction neurons ("after these two tokens, this third") authored from corpus trigram statistics into the MLP layers, with the previous-token circuit. Module handi/authoring/chords.py; conditions A9 (neurons) and A9C (same layout, no neurons); rows in results/tables/rung4.csv. Data-derived, reported on its own line. Pythia only (untied, RoPE, parallel residual); GPT-2 would need the position embedding redesign A8 zeroed. |

Builder defaults: residual layout B + K + K (K = head_dim = 64): bigram subspace at rank B = d - 2K, a fixed
random +-1/sqrt(K) code per token in a "current" slot, a "previous" slot written by one RoPE previous-token head
at layer 0 (the circuits.py recipe; gain from the grid {3, 10, 30, 100}, smallest with >= 0.95 mass on offset -1).
Embedding rows scaled to a common norm (the final LN divides by it, so the direct path is unchanged, and every code
slot has the same magnitude after every LN). Neurons in layers 1..L-1, one per context (a, b), the (L-1) x d_mlp
most frequent with n(ab) >= 3: detector = code_b on the current slot + code_a on the previous slot, gain 8, bias
-12 (true pair +4, half match -4, no match -12: exact GELU at -4 is 1e-4, at -12 zero; the first build with a true
pair at +0.5 and idle neurons at -1.5 leaked -0.1 per idle neuron and summed to garbage). Down column: weighted
least squares on the bigram subspace of r_ab = log p3(.|a,b) - log q(.|b), q = the model's own direct path after
b, weights p3 (so the B dims go where the mass is), ridge 1e-2, times the LN scale, divided by gelu(+4), times a
global scale chosen on the first 20k tokens of the WikiText VALIDATION split from {0.05, 0.1, 0.25, 0.5, 1}
(the train calibration slice lies inside the counted text; with trigram counts an in-sample choice is a
memorisation choice). p3 = 0.5 n(abc)/n(ab) + 0.5 p2(c|b). Reference lines reported: interpolated trigram on
the test slice, and the same capped to the selected contexts (what the neurons could reach at best).

Closed-form findings on the way (Pythia-70M, docs for the record): the detector is exact once rows have a common
norm (true pair 15.97 +- 0.03 at gain 8, false-firing positions 0.1%). The correction is limited by LayerNorm
geometry: the final LN fixes the output norm, so a correction vector of norm v shrinks the bigram signal by
1/sqrt(1 + (v/r0)^2); with the plain pseudo-inverse v was 3x the row norm and the neurons made things worse at
every scale. Replacing the token row by a context-fitted row (weighted LS, rescaled to the row norm) was also
worse than the base row. Fitting a row to a softmax target properly is a convex problem with no closed form
(iterative Newton / logistic regression), which the project's premise excludes; listed as a gap.

## Hearsay direction (decided 2026-11-08)

| date | decision |
|---|---|
| 2026-11-08 | Script analysis/hearsay_direction.py, rows results/tables/hearsay_direction.csv (hypothesis R3D). Direction = unit mean difference of the residual at the answer position between Yes and No hearsay items, per layer, two-fold; tests: readout accuracy, cosine and cross-readout against the nine other tasks' directions and against U[Yes]-U[No], steering at the best layer (last position only, gamma x sd in {-4..4}) on hearsay and on the other tasks, the converse, and a logit-lens / top-item reading. Direction saved to results/raw/hearsay_direction_L11.npy. |

Result. The direction exists and is task-specific: it reads hearsay at 0.73 (best layer 11; above chance from
layer 1), its cosine with the output Yes-No direction is -0.02 and with the other tasks' directions at most
0.21, and no other task's direction reads hearsay above 0.62. Two caveats. (1) It is not causal where it was
tested: adding up to +-4 sd of it at the last position of layer 11 leaves the answers unchanged (fraction Yes
0.245 -> 0.255 at +2, +4; 0.234 at -2) and changes 0-4% of answers on the other tasks; the other tasks'
directions are equally inert on hearsay. The answer is not computed from this direction at that place; steering
at every position, or at other layers, was not run (one design, not decided). (2) It tracks form as much as
law: the items projecting highest are long litigation narratives with a quoted statement or testimony
("Corporation A sues B... introduces a statement by B's CEO"), the lowest are the short template items ("On the
issue of whether X, the fact that Y"), which are mostly No. The logit lens of the direction is uninterpretable
(mid-layer). So: a direction that separates the hearsay items, carried along by the model, not shown to be the
direction the model answers with.

### Rung 4 run note (2026-11-08)

Three seeds, scale chosen on the validation split. WikiText CE: Pythia-70M A9C 7.274 -> A9 7.235 (10,240 neurons,
coverage 40% of positions, scale 0.1); Pythia-160M A9C 7.255 -> A9 7.258 (33,792 neurons, coverage 54%, scale
0.05, the smallest in the grid: every larger scale is worse on validation). TinyStories CE and copy accuracy
unchanged. Reference lines: bigram 6.10, trigram 6.02, trigram capped to the selected contexts 6.03 (70M) / 6.02
(160M). Verdict: the chord circuit is exact (detection) but the knowledge it can write through a LayerNormed
rank-B bottleneck in closed form is 0.04 nat on the small model and nothing on the larger one, where more
neurons add more residual noise than correction. Writing a sharpened distribution into a fixed-norm residual
needs a per-context fit to a softmax target, which has no closed form; gap, open whether iterative
convex solves are admissible. With this corpus (1.8M train tokens) the trigram line itself is only 0.08 nat
below the bigram line, so even a perfect chord layer would gain little here; a larger corpus would raise the
ceiling for both the bigram direct path and the chords.
