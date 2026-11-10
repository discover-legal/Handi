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

## Post-freeze amendment: iterative optimisation admitted for the chord output (decided 2026-11-08)

| date | decision |
|---|---|
| 2026-11-08 | Decision: iterative optimisation is admitted for the chord output, on condition of a proper train/validation/test split against overfitting Condition A10 (analysis/run_rung4_fit.py): the A9 build unchanged and frozen (direct path, codes, previous-token head, detectors, closed-form down columns as the starting point); only the chord neurons' down-projection columns (bigram-subspace rows) are free; Adam on the full model's next-token CE over the WikiText TRAIN split (the counted text); early stopping on the VALIDATION split (tokens 20k..70k, disjoint from the 20k slice that chose the A9 scale), patience 6 evaluations of 100 steps, at most 4000 steps; reported once on the standard M1 TEST slice. Rows in results/tables/rung4.csv, component A10. This is the project's first gradient step; it is confined to the chord outputs and reported on its own line, never merged with the closed-form results. |

## Hearsay steering sweep (decided 2026-11-08)

| date | decision |
|---|---|
| 2026-11-08 | Script analysis/hearsay_steer.py, rows results/tables/hearsay_steer.csv (hypothesis R3S). Directions: H = hearsay mean difference, two-fold (an item is steered by the direction fit on the other fold, scaled by that fold's projection sd); O = the overruling direction (reads hearsay at 0.62 in the first run), fit on all overruling items; HO = the two added. Sweep: all 24 layers x {last position, every position} x gamma in {-4, -2, +2, +4} on the 94 hearsay items; side effects on the nine other tasks at the best H and best HO setting. |

### Hearsay steering sweep result (2026-11-08)

Unsteered: fraction Yes 0.245, accuracy 0.468; majority (all No) 0.564. Over all 24 layers x {last, all positions}
x gamma in {-4,-2,+2,+4} x {H, O, HO} (576 settings) the best hearsay accuracy of any setting is 0.564 (H alone
0.553, O alone 0.564, HO 0.564), i.e. nothing beats answering No to everything. What does move: the fraction of
Yes answers. H is inert in layers 0-3, nudges the fraction from layer 4 on, and in layers 14-23 moves it between
0.11 and 0.57, with the sign REVERSED relative to its readout (gamma -4, the "No" side, raises Yes). O behaves as a
plain answer knob in the late layers: at layer 23, gamma -4 gives 0.00 Yes and +4 gives 0.98 Yes, at layer 20 0.00 /
0.90, with accuracy 0.44-0.56 throughout; the overruling direction is the output Yes/No axis of the trained model,
not legal content. HO is the vector sum and behaves like a weaker O (the two directions have cosine -0.19 at the
late layers, so they partly cancel). Steering every position instead of only the last changes little (same bests
within one item). Side effects of the best H setting (layer 4, all positions, +4): 2-30% of answers change on the
other tasks (diversity_6 0.30, citation 0.15); of the best HO setting (layer 14, last, -4): 0-35% (diversity_1
0.35, nys_judicial_ethics 0.20). Reading: adding a constant vector can only move the model's decision threshold,
so the best it can reach is the accuracy of the model's own ranking of the items at the best threshold, which is
not above majority here (E1 calibration in Rung 3: 0.49). The item-specific information the probe reads at 0.73-0.79
is in the residual, but it is read out by a weight (E2 0.63, probe 0.79), not by a bias. So "the 0.62 and the 0.73
steered together" make no difference to accuracy, and O alone makes a very large difference to the answer with no
difference to its correctness.

### A10 run note (2026-11-09)

WikiText test CE, three seeds: Pythia-70M A9C 7.274 -> A9 7.235 -> A10 6.717 +- 0.002 (all three seeds still
improving at the 4,000-step cap; it was not raised: the last thousand steps gained 0.02 nat and the gap
to the bigram line, 6.10, is 0.6); Pythia-160M A9C 7.255 -> A9 7.258 -> A10 6.663 +- 0.006 (stopped by the
validation rule at steps 3,300 to 3,600; validation 6.60). Train loss 5.5 to 5.9 with validation still falling, so
no memorisation of the batch. TinyStories CE improves too (70M 8.28 -> 8.17, 160M 8.17 -> 8.16) with no TinyStories
text in the fit. Copy accuracy stays 0. Reading: the chord circuit's detection was never the limit; its closed-form
output was. With the output fitted under a train/validation/test split, the chords recover half of the rank loss
of the direct path, and the larger model's extra neurons, which hurt in closed form, help. The first 160M run
hung after a laptop sleep (process alive, no output for 26 h, GPU idle); it was killed and the 160M seeds rerun
from scratch; the 70M rows were already on disk and are untouched. Report page: docs/report.html.

### Rung 3 transfer test (2026-11-10)

Script analysis/run_rung3_transfer.py, rows results/tables/rung3_transfer.csv (R3T), exact offline computation.
(1) Fresh items, edit fitted on the whole 500-item sample and scored on items never seen: overruling 0.674 ->
0.940 (n = 500), definition_classification 0.590 -> 0.980 (500), cuad_governing_law 0.617 -> 1.000 (376). The
other seven tasks have no unseen items (the sample is the whole test set), so their two-fold numbers stand.
(2) Cross-task: the edit rewrites the shared " Yes" / " No" rows, so one task's edit changes 2% to 97% of
another task's answers, with that task's accuracy moving anywhere from -0.45 to +0.31 (mean +0.02): a scramble, not a consistent harm or help. (3) All ten edits summed into one output table:
overruling 0.82, definition 0.93, cuad 0.97 survive; the other seven fall back to the majority rate or below
(hearsay 0.56, proa 0.51, nys 0.48). Reading: each edit is a task-specific classifier written into two rows
that every task shares; it generalises to new items of its own task and does not coexist with the others. A
model edited for several tasks would need one row pair per task (distinct label tokens) or an edit deeper than
the output table.

### Rung 3 transfer, remaining items (2026-11-10)

definition_classification: the 337 test items not yet scored (1,337 total = 500 sampled + 500 fresh + 337):
majority 0.540, as shipped 0.561, edited 0.979. cuad_governing_law has no remaining items (876 = 500 + 376). The
LegalBench train splits (8 and 6 items) score 1.00 as shipped and edited. Every item of both tasks has now been
scored by an edit that did not see it. Both tasks are lexically marked (defining sentences say "means" / "is
defined as"; governing-law clauses say "governed by the laws of"); the finding is that the model as shipped
could not express a distinction it computes. Cookbook for the edit: docs/cookbook.md.

### Rung 3 cue test (2026-11-10)

Script analysis/run_rung3_cue.py, rows results/tables/rung3_cue.csv (R3C). Edit fitted on the 500-item sample,
scored on the never-seen items with the cue phrases masked to "[...]" or paraphrased into plain words.
cuad_governing_law (376 items, 99% of Yes items carry a cue): original 1.000, masked 0.997, paraphrased 0.997
(as shipped 0.62 -> 0.48 / 0.49). definition_classification (500 items, 60% of Yes items carry a listed cue):
original 0.980, masked 0.972, paraphrased 0.972 (as shipped 0.59 -> 0.55 / 0.58). The edit does not depend on
the stock wording; it reads the clause's content (a jurisdiction attached to the agreement; a term given a
meaning). The cue list for definitions is incomplete (many definitional sentences quote a dictionary without
"means"), which bounds what the masking can show for that task.

### Rung 3 control tests (2026-11-09)

Script analysis/run_rung3_controls.py, rows results/tables/rung3_controls.csv (R3X). Five controls on the three
tasks with never-seen items, same fit sample, same never-seen items and same formula as the transfer test; every
number is accuracy on the never-seen items (overruling 500, definition_classification 500, cuad_governing_law
376). Builder defaults: 20 label permutations; 20 random subsets per fit size; word counts over words present in
at least two fit items, same standardised ridge with lambda = n (a lambda grid of n/100 to 100n picked on the
never-seen items was no better except 0.997 -> 1.000 on governing law); near-copy = word-set Jaccard to any fit
item; untrained = the A0 block (fresh init by the family's rule), two seeds.

| | overruling | definition | governing law |
|---|---|---|---|
| majority | 0.508 | 0.524 | 0.521 |
| trained, as shipped | 0.674 | 0.590 | 0.617 |
| trained, calibrated (uses no labels) | 0.838 | 0.888 | 0.912 |
| trained, edited | 0.940 | 0.980 | 1.000 |
| 1. shuffled labels: read-out alone (mean of 20) | 0.479 | 0.493 | 0.476 |
| 1. shuffled labels: full edit (mean of 20) | 0.565 | 0.578 | 0.576 |
| 2. word counts, same formula, same 500 labels | 0.910 | 0.862 | 0.997 |
| 3. edit fitted on 10 items (mean, min of 20) | 0.862, 0.638 | 0.933, 0.894 | 0.997, 0.992 |
| 3. edit fitted on 50 items (mean, min of 20) | 0.904, 0.890 | 0.949, 0.928 | 0.999, 0.992 |
| 4. no fit item at Jaccard >= 0.5 (items kept) | 0.939 (488) | 0.980 (498) | 1.000 (212) |
| 4. clauses from contracts absent from the fit sample | | | 1.000 (144) |
| 5. untrained model, edited (seed 0, seed 1) | 0.832, 0.840 | 0.856, 0.896 | 0.960, 0.955 |

Reading. (1) The read-out cannot manufacture a score from 500 labels: with shuffled labels it is at chance; the
full edit keeps 0.57 because calibration uses no labels. (4) The score does not come from near-copies or shared
contracts. (3) Ten labelled items already give 0.86 / 0.93 / 1.00. (2) and (5) bound what the trained model
contributes: the same ridge on word counts, and the same recipe on a network with random weights, reach most of
the score on these three tasks. Governing law is solved by word counts (0.997), so its 1.000 is not evidence of
anything the trained model knows. The trained state is worth +0.03 on overruling and +0.12 on definitions over
word counts, and +0.10 / +0.10 / +0.04 over the untrained network. This corrects the earlier notes: "the model
as shipped could not express a distinction it computes" and "it reads the clause's content" are supported for
definition_classification, weakly for overruling, and not shown for cuad_governing_law, where the words alone
suffice. Not run: the word-count baseline and the untrained model at small fit sizes (10 to 100 labels), which
is where the trained state would be expected to matter most.

### Rung 3 on all two-label LegalBench tasks, 15 labelled examples each (2026-11-10)

Decided by the owner: run all of LegalBench; use few examples for each, 15. Script analysis/run_rung3_all.py,
rows results/tables/rung3_all.csv (R3A), trained Qwen3.5-0.8B. 132 of the 162 tasks have two labels and were
run; none failed. The other 30 have more than two labels, which the two-row edit does not cover: a gap, not
worked around. Prompts are each task's LegalBench base_prompt.txt (the ten already in tests/fixtures are
byte-identical to that source; the other 122 were copied from the LegalBench repository and are not committed,
because the datasets carry differing licences). Builder defaults: per task, 15 items drawn at random (not
balanced by label) from the collected items (the 500-item sample plus up to 500 never-seen items), 20 draws,
seed 300; the edit and the word-count ridge are fitted on the same 15 and scored on every other collected item;
lambda = 15. The same pass also records the full-data scores (two-fold on the sample; fit on 500 and scored on
never-seen items for the 30 tasks with more than 500 items). The row-edit form of the formula (fit_edit) is used
throughout, so two-fold numbers for the original ten tasks differ slightly from rung3.csv, whose offline gap
added alpha x read-out where the row edit adds 2 alpha x read-out (proa 0.853 there, 0.874 here).

Mean over the 132 tasks, 15 labelled examples:

| | majority | as shipped | calibrated | edited | word counts |
|---|---|---|---|---|---|
| accuracy | 0.595 | 0.590 | 0.657 | 0.788 | 0.711 |
| balanced accuracy | 0.500 | 0.562 | 0.659 | 0.733 | 0.651 |

Balanced accuracy, 15 examples: edited above as shipped on 124 tasks, above word counts on 105 (mean +0.082).
At or above 0.90 on 22 tasks (word counts 1, as shipped 0); at or above 0.80 on 53 (13, 1). By group (edited /
word counts): CUAD clauses, 38 tasks, 0.878 / 0.745; contract NLI, 14, 0.845 / 0.738; OPP-115, 9, 0.777 / 0.647;
Learned Hands, 16, 0.696 / 0.590; diversity, 6, 0.691 / 0.503; MAUD, 19, 0.594 / 0.626; supply chain, 10,
0.541 / 0.546. In plain accuracy the edit is below the majority rate on 33 tasks, nearly all lopsided ones
(MAUD, supply chain), where 15 random labels cannot beat always answering the common label.

With the full data the word-count baseline closes most of the gap: two-fold mean accuracy over 132 tasks,
edited 0.860 against word counts 0.819; never-seen items of the 30 large tasks, fitted on 500, edited 0.896
against word counts 0.861 (as shipped 0.550). Reading: the trained model's state is worth most when labels are
few (+0.08 balanced accuracy at 15 labels, +0.04 at 500), and on clause-type tasks. It does not help where the
answer needs a long document (MAUD, supply chain) or recall of law with nothing to read
(international_citizenship_questions: 0.55 at 15 labels, 0.66 at 500, word counts 0.59 / 0.69).

Limits of this run. Prompts longer than 1,024 tokens keep only their last 1,024, as in every earlier run: 75
tasks have some truncated prompts, and in 34 (all Learned Hands, most MAUD) more than half are, so there the
model reads the end of the item without the instruction or the worked examples. Not run: the untrained model at
15 labels; label-balanced draws of the 15.

### Rung 3 formula extended to more than two labels; edited model against the raw model on 153 tasks (2026-11-10)

Decided by the owner: extend the formula, try all of LegalBench, benchmark the edited model against the raw
model. The K-label form (analysis/run_rung3_all.py, fit_edit_k) is the two-label formula written per label:
calibration unchanged (it was already per label); one standardised ridge per label, from h to +1 for that
label's items and -1 for the others, lambda = n; U[t_k] += alpha w_k + (alpha b_k / p_mean) mu_hat; alpha =
2 x spread of the calibrated gap / spread of the read-out, both as the root mean square over label pairs of
sd(score_k - score_j), the read-out's halved. The answer is the label with the highest first-token score. For
two labels it is the same edit: row differences against fit_edit at 3e-16 relative, 0 of 18,000 predictions
differ on synthetic states, and proa, ucc_v_common_law and cuad_price_restrictions rerun through the K-label
code reproduce all 60 recorded rows. The 132 two-label results were therefore kept, not rerun.

21 further tasks run (3 to 8 labels): abercrombie, canada_tax_court_outcomes, function_of_decision_section,
insurance_policy_interpretation, oral_argument_question_purpose, scalr, 15 MAUD multiple-choice tasks. Builder
default for scalr: its labels are the digits 0 to 4, which this tokenizer writes as a space token then the
digit, so the prompt ends with that space and the digit rows are the label rows. Gap, not worked around, 9
tasks: citation_prediction_open, definition_extraction, rule_qa, sara_numeric, ssla_company_defendants,
ssla_individual_defendants, ssla_plaintiff (free-text answers, no fixed label set); successor_liability
(answers are sets of labels); unfair_tos (Unilateral change and Unilateral termination share their first
token, and the edit acts on the first token).

Balanced accuracy, mean over tasks (per-task table: docs/legalbench_benchmark.md; rows R3A in rung3_all.csv):

| tasks | n | chance | raw | edited, 15 labels | word counts, 15 labels | edited, two-fold | word counts, two-fold |
|---|---|---|---|---|---|---|---|
| all run | 153 | 0.463 | 0.524 | 0.683 | 0.607 | 0.774 | 0.726 |
| two labels | 132 | 0.500 | 0.562 | 0.733 | 0.651 | 0.816 | 0.765 |
| 3 to 8 labels | 21 | 0.228 | 0.281 | 0.365 | 0.329 | 0.507 | 0.478 |
| 3 to 8 labels, not MAUD | 6 | 0.225 | 0.277 | 0.389 | 0.268 | 0.503 | 0.313 |
| 3 to 8 labels, MAUD | 15 | 0.229 | 0.282 | 0.356 | 0.353 | 0.508 | 0.544 |

Edited above raw on 141 of 153 tasks with 15 labels (148 two-fold); above word counts on 116. Below raw with
15 labels on 12: citation_prediction_classification, insurance_policy_interpretation, three MAUD tasks,
nys_judicial_ethics, personal_jurisdiction, sara_entailment, scalr, telemarketing_sales_rule,
textualism_tool_dictionaries, maud_intervening_event. Reading: the extension works where the two-label edit
works (abercrombie 0.23 to 0.47 with 15 labels and 0.59 two-fold, word counts 0.18 and 0.14; canada tax court
0.34 to 0.68 and 0.90) and not where it does not (MAUD: no better than word counts; scalr and
insurance_policy_interpretation, which need reasoning over the choices: at or below raw). With 15 labels spread
over up to 8 labels many labels have one fit item or none, so the 15-label scores of the many-label tasks are
low. Raw = the shipped model with its answer restricted to the label tokens, as in every earlier run; one edit
per task. Same limits as the entry above (prompts cut to their last 1,024 tokens; random, not label-balanced,
draws of the 15). The 21 new base_prompt.txt copies are uncommitted with the other 122.

### Rung 3 on the many-label tasks with more labelled examples (2026-11-10)

Decided by the owner: run the many-label tasks with more labelled examples. Run: the 21 tasks with 3 to 8
labels, the same formula, fitted on 15, 30, 60 and 120 labelled examples (rows R3A in
results/tables/rung3_all_sizes.csv, written by analysis/run_rung3_all.py --sizes 15 30 60 120). Builder
defaults, flagged: the sizes 30, 60 and 120; each size is 20 random draws, not label-balanced, scored on every
other collected item; a size is skipped when fewer than 30 items would be left to score, so 60 runs on 19 tasks
and 120 on 11. Prompts cut to their last 1,024 tokens as before, so only the number of examples changes; the
15-example and two-fold rows reproduce rung3_all.csv exactly. The file also holds one two-label MAUD task that
the task pattern picked up; it is left out of the numbers below.

Balanced accuracy, mean over tasks (chance 0.228 over the 21):

| labelled examples | tasks | raw | edited | word counts | edited above raw | edited above word counts |
|---|---|---|---|---|---|---|
| 15 | 21 | 0.281 | 0.365 | 0.329 | 17 | 11 |
| 30 | 21 | 0.281 | 0.420 | 0.386 | 18 | 10 |
| 60 | 19 | 0.286 | 0.470 | 0.443 | 18 | 11 |
| 120 | 11 | 0.309 | 0.617 | 0.584 | 11 | 8 |
| half the sample (two-fold) | 21 | 0.281 | 0.507 | 0.478 | 20 | 11 |

The same 11 tasks at every size: edited 0.414, 0.472, 0.526, 0.617 at 15, 30, 60, 120 (word counts 0.345,
0.397, 0.469, 0.584). Plain accuracy over the 21: raw 0.394; edited 0.551, 0.592, 0.626 (19 tasks), 0.718 (11
tasks); word counts 0.528, 0.577, 0.610, 0.697.

Reading: more examples help steadily, 5 to 9 points of balanced accuracy per doubling, and the many-label
scores at 15 examples were low because 15 examples over up to 8 labels leave some labels with one example or
none. The split of the earlier entry holds at every size. The six tasks that are not MAUD: edited 0.389, 0.424,
0.436, 0.498 (4 tasks) against word counts 0.268, 0.279, 0.287, 0.368; the edited model is above word counts on
all six. The 15 MAUD tasks: edited 0.356, 0.419, 0.486, 0.685 (7 tasks) against word counts 0.353, 0.429, 0.514,
0.707; no better than word counts at any size. insurance_policy_interpretation gets worse with more examples
(0.334, 0.318, 0.296; raw 0.360) and scalr stays at raw (0.351 to 0.400; raw 0.379): what the model's state
does not hold, more examples do not add. Implementation change, no change of result: fit_edit_k now solves each
ridge in the n x n form when there are fewer examples than state dimensions (the same solution; row edits agree
to 3e-14 relative and 0 of 6,885 predictions differ on saved states).

### Free-text tasks: per-word read-out for the extraction tasks (2026-11-10)

Decided by the owner: build the extraction read-out, after the whole-prompt Qwen run and the Phi-4-mini run.
Scope: definition_extraction and the three ssla tasks, whose answers are words copied from the input. The same
closed-form fit scores each input word "part of the answer" or not from the model's state at that word; the
words that score high are copied out. This is a read-out beside the model, not an edit of what the model writes:
the model's own reply is unchanged. Word counts are the baseline. rule_qa, citation_prediction_open and
sara_numeric need recall or arithmetic and are not in scope. Also decided by the owner: runs use fewer resources
and may take longer (lowest CPU and disk priority, 2 CPU threads).

### Rung 3 with whole prompts on Qwen (2026-11-10)

Decided by the owner: do not cut long prompts. Run: the 90 tasks whose prompts were cut to the last 1,024 tokens,
read whole (--max-len 0), sizes 15, 30, 60, 120. Rows: results/tables/rung3_all_uncut.csv. 89 of 90 ran.
learned_hands_housing ran out of GPU memory twice (longest prompt 6,593 tokens on the 8 GB card): a gap.

Balanced accuracy, mean over the 89 tasks, 15 labelled examples (word counts read the whole text in both runs):

| | raw | edited | word counts |
|---|---|---|---|
| cut prompts (rung3_all.csv) | 0.483 | 0.636 | 0.589 |
| whole prompts | 0.503 | 0.670 | 0.589 |

Edited better with whole prompts on 76 tasks, worse on 12; edited above word counts on 69 of 89 (59 cut).
Largest gains: learned_hands (15 tasks) 0.690 to 0.793, textualism (2) 0.557 to 0.706; MAUD, CUAD and
supply_chain change by 0 to 3 points. Whole prompts at 15, 30, 60, 120 examples: edited 0.670, 0.703, 0.744
(80 tasks), 0.790 (55 tasks); word counts 0.589, 0.637, 0.687, 0.733.

### Rung 3 on a second model: Phi-4-mini-instruct (2026-11-10)

Decided by the owner: try another model, a Nemotron or a Phi. Phi-4-mini-instruct (3.8B, MIT licence) is my
choice; Nemotron 3 Nano 4B needs its own model code and was not tried (a gap). Same prompts, same formula, 15
labelled examples, prompts kept to their last 4,096 tokens for GPU memory (17 tasks cut). Rows:
results/tables/rung3_all_phi.csv, all 153 label tasks ran. Against Qwen of rung3_all.csv (prompts cut to 1,024),
balanced accuracy, mean over tasks:

| tasks | Qwen raw | Qwen edited | Phi raw | Phi edited | word counts |
|---|---|---|---|---|---|
| 132 two-label, 15 examples | 0.562 | 0.733 | 0.679 | 0.763 | 0.651 |
| 132 two-label, half the sample | 0.562 | 0.816 | 0.678 | 0.842 | 0.765 |
| 21 many-label, 15 examples | 0.281 | 0.365 | 0.399 | 0.431 | 0.329 |
| 21 many-label, half the sample | 0.281 | 0.507 | 0.399 | 0.555 | 0.478 |

Reading: Phi is better than Qwen before the edit (raw higher on 108 of 132 two-label tasks) and the edit still
adds to it; edited Phi is above word counts on 120 of 132 two-label tasks at 15 examples (Qwen 105). The prompt
lengths differ between the two runs (4,096 against 1,024 tokens), so part of the gap may be prompt length.
Both runs at low priority (owner: use fewer resources, at most about 30% of the CPU): 7 of 24 cores, nice 19.

### Extraction read-out: results (2026-11-10)

Run as decided above (analysis/run_rung3_extract.py, rows results/tables/rung3_extract.csv, hypothesis R3X):
Qwen3.5-0.8B, the four tasks, 500 items each, prompts whole (longest 3,318 tokens), 15, 30, 60, 120 labelled
examples and half the sample. Scored with LegalBench's own scorers. My choices, stated when built: a word is
picked when its read-out score is above 0; both a weighted fit (answer words and other words count equally)
and an unweighted one are reported, because answer words are rare and neither suits every task; the word
baseline uses the word, its neighbours and a capital-letter flag. Ceiling = the best any per-word pick can do
(the labels themselves turned into an answer).

| task | raw model | read-out 15 / 30 / 60 / 120 / half | word baseline 120 | ceiling |
|---|---|---|---|---|
| definition_extraction (accuracy) | 0.670 | 0.657 / 0.694 / 0.719 / 0.734 / 0.768 | 0.406 | 0.886 |
| ssla_company_defendants (F1) | 0.339 | 0.051 / 0.111 / 0.118 / 0.118 / 0.110 | 0.038 | 0.109 |
| ssla_individual_defendants (F1) | 0.233 | 0.221 / 0.221 / 0.218 / 0.214 / 0.214 | 0.111 | 0.309 |
| ssla_plaintiff (F1) | 0.103 | 0.745 unweighted at every size | 0.745 | 0.908 |

(weighted read-out shown except for ssla_plaintiff.)

Reading: definition_extraction works. From 30 labelled examples the read-out beats the model's own answer, up to
0.768 against 0.670, and is far above the word baseline. ssla_company_defendants cannot work this way: the
answer is the full legal name ("Akorn, Inc.") and the text mostly says "Akorn", so only 15% of answers are in
the text and the ceiling (0.109) is below the model's own answer (0.339), which writes the full name.
ssla_individual_defendants: the read-out is level with the model's own answer, below the ceiling; the ceiling
is low because LegalBench's ssla scorer adds false positives once per answer (as written in its code), which
punishes items with many names. ssla_plaintiff: 378 of 500 items are "Not named". Unweighted, the read-out
picks no word on any item (checked: 0 of 250), so its 0.745 is "Not named" every time (0.746 on all 500);
weighted, it picks too many words and gets worse with more examples (0.624 down to 0.115). The model's own
answer scores 0.103. No version finds the named plaintiffs.

### LegalBench prompts out of the repository again (2026-11-10)

Decided by the owner: take the prompt files out. Earlier today 147 base_prompt.txt copies (143 label tasks, the
four extraction tasks) were committed and pushed at the owner's request, which reversed the earlier choice of
keeping them local because the LegalBench datasets carry differing licences. They are now untracked again (kept
on disk, ignored by .gitignore); the original ten stay tracked. They remain in the git history of commits
9fd59f3 and a02c44b; removing them from history needs a rewrite of the public history and a force push, which
was not done.

### LegalBench: our scores beside the paper's published scores (2026-11-10)

Decided by the owner: compare with published scores. Source: the LegalBench paper (arXiv 2308.11462 v1),
appendix Tables 60 to 79, per-task scores of GPT-4, GPT-3.5, Claude-1 and the 13B models, read from the PDF text
(results/tables/legalbench_paper_scores.csv, percent as printed; rows whose names the PDF wraps are missing).
Script analysis/compare_legalbench_paper.py, rows results/tables/legalbench_paper_compare.csv. 147 of our 153
tasks are in both, 143 with every column. Balanced accuracy, mean over the 143:

| | two labels (123) | 3 to 8 labels (20) | all (143) |
|---|---|---|---|
| GPT-4 (paper) | 0.849 | 0.510 | 0.802 |
| GPT-3.5 (paper) | 0.780 | 0.455 | 0.735 |
| Claude-1 (paper) | 0.760 | 0.337 | 0.701 |
| Flan-T5-XXL, 11B (paper) | 0.738 | 0.257 | 0.671 |
| LLaMA-2-13B (paper) | 0.564 | 0.296 | 0.527 |
| Qwen as shipped | 0.564 | 0.284 | 0.525 |
| Qwen edited, 15 labels | 0.745 | 0.370 | 0.692 |
| Qwen edited, half the sample | 0.829 | 0.513 | 0.785 |
| Phi-4-mini as shipped | 0.690 | 0.407 | 0.650 |
| Phi-4-mini edited, 15 labels | 0.776 | 0.438 | 0.729 |
| Phi-4-mini edited, half the sample | 0.855 | 0.563 | 0.814 |
| word counts, 15 labels | 0.654 | 0.329 | 0.609 |

Reading: with 15 labels, edited Qwen (0.8B) is level with Flan-T5-XXL and between Claude-1 and LLaMA-2-13B;
edited Phi-4-mini is between GPT-3.5 and Claude-1. With half the sample (up to 250 labels), edited Qwen is near
GPT-4 (0.785 against 0.802; at or above GPT-4 on 63 of 143 tasks) and edited Phi is above it (0.814; 69 of 143).
Not like for like: the paper's models answer by few-shot generation from 0 to 8 worked examples, with no
labelled fit; ours are fitted on 15 or up to 250 labelled items of the task; item sets differ. Extraction, paper
against our read-out (best size): definition_extraction GPT-4 0.818, ours 0.768; ssla_plaintiff GPT-4 0.920,
ours 0.746; ssla_company_defendants GPT-4 0.650, ours 0.118; ssla_individual_defendants GPT-4 0.296, ours 0.221.

### LexGLUE: CaseHOLD run, SCOTUS skipped, LEDGAR read beside the model (2026-11-10)

Decided by the owner: try LexGLUE (coastalcph/lex_glue, CC BY 4.0 per its dataset card). Correction recorded: I
first said its single-answer tasks needed nothing new; LexGLUE has no prompts, LEDGAR has 100 types and SCOTUS
opinions are long, so each needed a decision. Owner: CaseHOLD with the proposed prompt (citing passage, the five
holdings numbered 0 to 4, "Answer:", digits as the answer rows as for scalr, no worked examples;
tests/fixtures/lexglue_prompts/case_hold.txt); SCOTUS skipped (opinions 12,500 tokens typical, up to 91,600, do
not fit the 8 GB card whole); LEDGAR read beside the model (below).

CaseHOLD (analysis/run_rung3_lexglue.py, rows results/tables/rung3_lexglue.csv), Qwen, test split, everything
else as run_rung3_all.py. Balanced accuracy (chance 0.20, classes balanced):

| | 15 | 30 | 60 | 120 | half the sample | fit on 500, never-seen 500 |
|---|---|---|---|---|---|---|
| as shipped | 0.280 | 0.280 | 0.280 | 0.280 | 0.270 | 0.284 |
| calibrated (uses no labels) | 0.458 | 0.464 | 0.465 | 0.466 | 0.464 | 0.466 |
| edited | 0.349 | 0.388 | 0.396 | 0.418 | 0.424 | 0.436 |
| word counts | 0.200 | 0.196 | 0.200 | 0.200 | 0.216 | 0.208 |

(calibrated row in accuracy; the classes are balanced, so accuracy and balanced accuracy agree to 0.01.)
Reading: removing the model's lean towards some digits takes it from 0.28 to 0.46 with no labels at all; the
read-out on top makes it worse at every size (0.35 to 0.44). The right digit depends on which of five holdings
matches the passage, and its position is random, so a read-out per digit from the final state has nothing
stable to learn. Word counts are at chance for the same reason. On this task the calibration alone is the
useful part.
