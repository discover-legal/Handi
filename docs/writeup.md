# Handi: hand-authored LLM weights, zero gradient descent — write-up

Date: 2026-11-06. Project owner: hordruma. Builder: Claude Fable 5.1.
Everything cited here is in this repository: hypotheses in `docs/hypotheses.md`, decisions and validity
notes in `docs/decisions.md`, per-row results in `results/tables/*.csv`, the machine-generated verdicts in
`results/summary.md`, figures in `results/figures/`.

## 1. The question

Do the weights of trained transformer language models contain regular mathematical structure strong
enough that weights *authored* from that structure, with no training at all, produce a network that does
anything language-like? The ultimate target was law tasks (LegalBench).

The project was prompted by "Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2 Nov 2026,
https://rikiyat.github.io/blog/weight-space-programming/), in which GPT-6 Astra wrote a 1.48M-weight Go network as a
program and edited the weights of Qwen3.5-0.8B without gradients. Hand-built and compiled transformers exist for
algorithmic tasks (RASP, Weiss et al. 2021; Tracr, Lindner et al. 2023; the AdderBoard entries), and initialisation
from the statistics of trained weights has been studied (Trockman & Kolter 2023). As far as we know, nobody had
tried to write the weights of a full-size language model from statistics measured in trained ones, with no
training; that is what Part B and Rung 2 do.

## 2. What was built

**Rung 1 (pipeline proof).** A 10-digit adder in the stock Qwen3.5 architecture, every one of its 4,119
nonzero weights emitted from an explicit design (residual-stream slots, RoPE relative-offset heads,
SwiGLU truth tables, RMSNorm gain absorption). It loads into the real Hugging Face class, decodes with the
KV cache, and scores 10,010 of 10,010 on the AdderBoard edge cases and random pairs. So the pipeline
"design -> state dict -> real architecture -> correct behaviour" works when the function is known.

**Part B (the science).** Nineteen hypotheses about the structure of trained weights, each measured
identically on eight trained models (GPT-2 small, Pythia-70M and 160M, Qwen3.5-0.8B, TinyStories 1M, 3M,
8M, 33M) and on families of null weight sets of the same shape. A pattern counts as confirmed only if the
trained value lies five or more standard deviations outside the null in at least two of the three model
tiers. Thirteen hypotheses were frozen before any measurement; six more (H14 to H19) were added by the
owner after seeing results and are reported separately as amendments.

**Rung 2 (the capstone, H13).** Full-size weights authored from the confirmed patterns, loaded into the
real model classes, and scored on held-out cross-entropy (WikiText-103, TinyStories), a copy task, and ten
LegalBench classification tasks. Conditions form a ladder: A0 fresh random init; A1 spectrum-matched random
(every matrix has the trained singular values on random directions); A2 adds outlier dimensions and
embedding anisotropy; A3 adds the shared cross-layer subspace and identity-like attention circuits; A4
adds an explicitly authored previous-token and induction circuit; A5 is the trained model. Post-freeze
additions: A6 (rotation chain, H15), A7 (golden-ratio fill, H16), A8 (direct path from corpus bigrams,
H17). Five seeds per cell, 95% confidence intervals, on all eight models.

## 3. What the trained weights contain

Fourteen of the eighteen structural hypotheses (H13 is the authoring capstone) pass the pre-registered rule (one of them, H9, only mechanically: on
inspection it shows no fractal structure, so thirteen are confirmed in substance); most with
z-scores in the hundreds or thousands. The trained weights are nothing like random matrices.

| hypothesis | verdict | what it says |
|---|---|---|
| H1 heavy-tailed spectra | confirmed | singular-value spectra have power-law tails and spikes above the random edge |
| H2 spikes carry function | supported, mixed | for QK circuits, keeping the spikes loses less than keeping the bulk; for OV it is mixed; embeddings cannot be reduced either way |
| H3 low-rank, identity-like circuits | confirmed | QK and OV circuits have far lower stable rank than random and a positive projection on the identity |
| H4 embedding anisotropy | confirmed | mean pairwise cosine far above zero; a dominant common direction |
| H5 outlier residual dimensions | confirmed | the same few residual dimensions are anomalous in every layer |
| H6 heads identifiable from weights | confirmed | previous-token heads are found from weights alone; induction heads via the attention check; the weight-only K-composition statistic was not discriminative |
| H7 cross-layer subspace alignment | confirmed | readers across layers share a subspace far more than chance |
| H8 spectral-shape universality | inconclusive | profiles collapse across GPT-2 and Pythia but not against Qwen |
| H9 fractal structure | no evidence | the surviving signal on ordered axes is long-range smoothness (H10), not self-similarity; the box-counting statistic failed its own control and was discounted |
| H10 positional structure | confirmed | learned positions are smooth, low-dimensional, with sinusoidal leading components; RoPE heads specialise by frequency band |
| H11 number structure in embeddings | confirmed in part | a period-100 helix in digit embeddings on Qwen and TinyStories; no cubic or elliptic-curve structure anywhere |
| H12 MLP neuron geometry | confirmed | heavy tails of near-parallel neurons and biased in/out alignment |
| H14 spectral mixture | confirmed (amendment) | a few component profiles explain held-out spectra; they do not identify matrix type |
| H15 rotation rule | refuted (amendment) | the turn direction between layers beats a random turn, but repeating it never beats staying where you are; consecutive turns are anti-aligned |
| H17 direct path is a bigram model | confirmed (amendment) | the trained embedding-to-unembedding path beats a token-shuffle null in 7 of 8 models, but sits within 0.1 nat of uniform: it knows almost nothing on its own |
| H18 within-matrix structure | confirmed (amendment) | every unit type is heavier-tailed than spectrum-matched random, strongest in output projections; a neuron's input norm is tied to its bias |
| H19 intrinsic dimension | confirmed (amendment) | weight clouds lie on low-dimensional curved sheets: token embeddings have intrinsic dimension 14 to 42 against covariance-matched nulls of 125 to 185, MLP down-projections about a third of their null |

H16 (golden-ratio fill) passed its mechanical rule and is discussed under Rung 2 because the mechanism
turned out to be an artefact.

## 4. What authoring from those patterns produces (Rung 2)

Held-out cross-entropy on WikiText-103 in nats, mean over five seeds. Lower is better. The uniform line
is log(vocabulary), about 10.8 for the 50k vocabularies and 12.4 for Qwen. The unigram line is what a
model knowing only word frequencies scores; the bigram line, one knowing only word pairs.

| model | A0 fresh | A1 spectrum | A3 patterns | A4 circuit | A6 rotation | A7 golden | A8 bigram path | unigram | bigram | A5 trained |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2 | 10.97 | 32.2 | 33.9 | — | 32.4 | 21.7 | 7.61 | 7.50 | 6.15 | 3.58 |
| pythia-70m | 11.04 | 43.4 | 43.0 | 35.4 | 44.0 | 26.7 | 7.39 | 7.50 | 6.10 | 4.16 |
| pythia-160m | 11.03 | 31.7 | 32.5 | 27.4 | 31.7 | 19.0 | 7.27 | 7.50 | 6.10 | 3.61 |
| qwen3.5-0.8b | 12.62 | 15.5 | 15.5 | 19.1 | 15.5 | 14.2 | 7.66 | 7.30 | 5.86 | 3.13 |
| tinystories-1m | 10.84 | 14.5 | 14.6 | — | 14.3 | 12.2 | 7.92 | 7.50 | 6.15 | 9.91 |
| tinystories-3m | 10.84 | 13.5 | 13.7 | — | 13.8 | 13.4 | 7.76 | 7.50 | 6.15 | 9.31 |
| tinystories-8m | 10.88 | 13.0 | 12.9 | — | 13.0 | 11.4 | 7.66 | 7.50 | 6.15 | 9.83 |
| tinystories-33m | 10.97 | 13.0 | 12.9 | — | 12.9 | 11.3 | 7.60 | 7.50 | 6.15 | 10.64 |

(A2 and A2F are within noise of A1 and A3 and are in `results/summary.md`. A4 exists only where H6 found
a circuit. The trained TinyStories models score poorly on WikiText because they were trained on children's
stories; on TinyStories they score 1.2 to 2.1.)

**H13 verdict: NULL.** The pre-registered test asked whether any pattern-authored condition (A1 to A3)
beats fresh random init or the spectrum-matched control. None does. The failure is in the wrong direction
and is large: every spectrum-matched condition is 2 to 32 nats *worse* than fresh random init. The reason
is simple once seen. Trained spectra have very large singular values; placed on random directions they
produce confident wrong logits. A fresh init is near-uniform and therefore cannot be very wrong. Adding the
outlier dimensions, the embedding anisotropy, the shared subspace, the identity-like circuits (A3) or the
rotation chain (A6) changes nothing measurable: all of these are statements about *which directions*
matter, and none says *which token* each direction serves.

**A4, the explicit circuit.** The authored previous-token and induction circuit works as a circuit: copy
accuracy 0.84 to 0.96 on Pythia-70M, Pythia-160M and Qwen. It does not help cross-entropy, because
copying earlier tokens is a small part of predicting text, and it had to be hosted with every other
residual write attenuated by 0.01, so the rest of the network contributes nothing.

**A7, the golden-ratio fill (H16).** Replacing Gaussian draws with a golden-ratio Kronecker sequence
scores 1 to 17 nats better than the Gaussian fill in seven of eight models, which passes the mechanical
rule. The diagnostic shows why: the structured sequence makes the embedding matrix collapse to an
effective rank of about 22 of 512, which flattens the output distribution. At the best output temperature
every fill, A7 included, scores the uniform line. A7 is less confidently wrong, not better informed.

**A8, the direct path from corpus bigrams (H17 applied).** This condition steps outside the weights-only
premise: it takes the smoothed corpus bigram table, factorises it in closed form (randomised SVD, or a
signed symmetric eigendecomposition for models with tied embeddings, the sign carried by the final-norm
gain) and writes the factors into the embedding and unembedding, with the layers above zeroed. No gradient
step. The result is the first authored model in the project that knows anything: 7.3 to 7.9 nats, 2.9
to 5.0 nats below fresh init, with the two untied Pythia models below the unigram line. The six
tied-embedding models sit 0.1 to 0.4 nat above it, and the reason is exact: a tied model's direct-path
logit for "b after a" must equal that for "a after b", so only the symmetric half of the bigram table can
be carried. All eight remain well above the bigram line, because a rank-d factorisation keeps only part of
the table and the layers above contribute nothing. Copy accuracy is zero and LegalBench is at the majority
class in every A8 cell, as expected of a model that is a bigram table.

**LegalBench.** Across 490 authored-condition comparisons, six cells beat the majority class at a 95%
confidence interval, against about 24 expected by chance; none reproduces in a second tier. No authored
condition does anything on law tasks.

## 5. What this means

1. **The structure is real and large.** Trained weights have heavy-tailed spectra, low-rank identity-like
   circuits, outlier dimensions, shared subspaces, smooth positions, heavy-tailed units, and point clouds
   on low-dimensional curved sheets. Every one of these is many standard deviations from random.

2. **Second-order structure does not carry function.** Everything measured by H1 to H7, H10, H12, H14 and
   H15 is invariant to which token each direction serves. Those statistics describe the *container*, not
   the *content*. Authoring the container alone produces a model that is worse than random, because the
   container is shaped to hold large, confident signals and the signals are absent.

3. **Content comes from data.** The one condition that works is the one that brings in the corpus (A8),
   and it works exactly as far as the data it brings: a bigram model, and a half-bigram model where the
   architecture ties its embeddings. This is the paper's Gemma caveat ("Editing weights could remove an obstacle that hid an ability, but it could
   not add an ability the model lacked") in a stronger form: weight edits
   cannot add an ability the weights do not encode, and the encoding is the token-to-direction pairing,
   which the second-order statistics do not see.

4. **The curved sheets (H19) are the most interesting lead left.** An intrinsic dimension of 14 to 42 for
   token embeddings in a 512- to 1024-dimensional space is not a statement about the spectrum; it is a
   statement about a nonlinear manifold that the tokens sit on. What parametrises it, and whether a tied
   factorisation that respects it would close the gap to the bigram line, were not tested.

5. **What would be needed to go further.** Not more weight statistics. Either a way to write the
   token-to-direction pairing from data at ranks above the embedding width (that is, authoring the MLP
   layers from n-gram or feature tables rather than leaving them random), or an explicit algorithmic
   design as in Rung 1, scaled to language. Both are outside this project's decided scope and are listed
   as gaps, not started.

## 6. Scope limits and honest caveats

- Eight small models (1M to 0.8B parameters). The structural findings reproduce across three
  architectures and three tiers, but nothing here is measured on a frontier-scale model.
- Rung 2 scoring used 100k held-out tokens per corpus and five seeds per cell; the confidence intervals in
  `results/summary.md` are per-cell bootstraps, and the cross-condition tests are the pre-registered ones.
- Six hypotheses (H14 to H19) and three conditions (A6 to A8) were added after results were seen. They are
  reported outside the H13 verdict and marked as amendments in every table.
- Three measurement instruments failed their own controls and were discounted before any verdict: the H9
  box-counting statistic, the H6 weight-only K-composition statistic, and the first tied-embedding branch
  of A8 (positive-only factorisation). Each is recorded with its date in `docs/decisions.md`.
- Gated DeltaNet layers in Qwen3.5 were filled with spectrum-matched random in every condition; no
  authoring design for them exists.
