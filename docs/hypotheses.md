# Pre-registration: structure in trained transformer weights, and whether it can be authored

Frozen 3 November 2026, before any measurement on trained weights. Changes after this commit are
amendments and must be listed in `docs/decisions.md` with a date and reason.

## 0. Question and governing rule

**Question.** Do the weights of trained transformer language models contain regular mathematical
structure strong enough that weights *authored* from that structure, with zero training, produce a
network that does anything language-like?

**Rule.** Every measurement is computed identically on the trained weights and on a family of null
weight sets of the same shape. A pattern counts as **confirmed** only if the trained value lies
outside the null distribution by the pre-registered margin:

- scalar statistic: |z| >= 5 against >= 20 null draws (100 for TinyStories), AND the effect
  reproduces in at least 2 of the 3 model tiers;
- distribution statistic: KS(trained, null) > 3 x median KS(null, null).

Multiple comparisons: ~13 hypotheses x 8 models x ~100 matrices means per-matrix hits at z = 3 are
expected by chance. Only layer-aggregated or tier-reproduced effects are reported as confirmed.
Every results row carries the null mean, sd, n and z so the thresholds can be re-applied.

**Permutation symmetry.** Hidden-dimension axes (residual dims, neurons, head dims) are
permutation-symmetric: any consistent permutation yields an identical function. A "picture" of a
weight matrix therefore has no meaningful x/y order, and any geometric claim on those axes is
refuted trivially by row/column permutation. The only ordered axes are: position (learned
positional embeddings), RoPE frequency index, and token id (BPE ids are roughly merge order, a
frequency proxy). The exotic hypotheses (H9, H11) get their test on those axes.

## 1. Models

| tier | model | family | params | notes |
|---|---|---|---|---|
| A | GPT-2 small | gpt2 | 124M | learned pos, LN, GELU, tied |
| A | Pythia-70M, Pythia-160M | gpt_neox | 70M / 160M | RoPE 25%, parallel residual, untied |
| B | Qwen3.5-0.8B | qwen3_5 | 0.8B | 18 Gated DeltaNet + 6 gated-attention layers, GQA 8/2, head_dim 256, RMSNorm, SwiGLU, RoPE 25%, tied |
| C | TinyStories-1M/3M/8M/33M | gpt_neo | 1M-33M | learned pos, LN, no attention scaling, local window 256 on alternate layers |

Revisions are pinned in `results/manifest.json`; configs and tensor lists in `docs/models/`.

## 2. Notation

d = hidden size, h = head dim, V = vocab, L = layers. Per head: W_QK = W_Q W_K^T (d x d, rank <= h),
W_OV = W_V W_O (d x d, rank <= h), per Elhage et al. 2021. ESD = empirical spectral density of
W^T W. MP = Marchenko-Pastur with gamma = min(n,m)/max(n,m), sigma^2 = ||W||_F^2 / (n m).

## 3. Null models

- **N1** Gaussian iid, same shape, same Frobenius norm (ESD -> MP exactly).
- **N2** element shuffle of the trained matrix (same marginal, iid): separates heavy-tailed entries from correlated structure.
- **N3** row and/or column permutation of the trained matrix (identical spectrum, destroyed axis order): the null for every ordered-axis claim.
- **N4** fresh initialization of the full architecture by the family's init rule (GPT-2: N(0, 0.02), residual projections scaled 1/sqrt(2L); GPT-Neo: N(0, 0.02); Pythia: small_init / wang_init; Qwen: N(0, 0.02), RMSNorm weight 0). Answers "training or architecture?".
- **N5** spectrum-matched random: U diag(sigma_trained) V^T with Haar U, V. Keeps H1 exactly, kills everything about singular vectors. Also Rung 2 control A1.
- **N6** covariance-matched Gaussian embeddings (keeps anisotropy, random token assignment).
- **N7** token-assignment shuffle among a set of tokens (number-token tests).

All nulls seeded; seeds recorded per row.

## 4. Hypotheses

### H1. Heavy-tailed spectra with spikes above the MP edge
- **Statement.** The ESD of each trained weight matrix is not MP: a power-law tail with exponent alpha in roughly [2, 6] and a countable number of spike eigenvalues above lambda+ estimated from the bulk.
- **Measurement.** SVD per matrix; Clauset-Shalizi-Newman power-law MLE with xmin search; spike count with sigma^2 from the bulk median (iterated); KS distance to MP.
- **Refuted if** KS to MP for trained matrices is within the null-vs-null range and spike count is within what N1 gives (0-1).
- **Literature.** Martin & Mahoney 2021 (JMLR); Martin, Peng, Mahoney 2021; Thamm, Staats, Rosenow 2022; Staats, Thamm, Rosenow 2023.
- **Nulls.** N1, N2, N4. **Prior.** ~95% (tiers A, B); ~70% TinyStories-1M (tiny matrices; spike count is the robust statistic there).

### H2. Function is concentrated in the low-rank part (the spikes carry the model)
- **Statement.** Keeping the top-k singular triplets (k = spike count, or 20% of directions) and replacing the rest with a bulk-matched Gaussian changes held-out loss far less than randomizing the top-k and keeping the bulk.
- **Measurement.** Held-out loss (M1 corpora) for {trained; top-k kept + bulk randomized; top-k randomized + bulk kept}, per matrix type.
- **Refuted if** both ablations are equally destructive (loss ~ ln V).
- **Literature.** Thamm et al. 2022; Sharma, Ash, Misra 2023 (LASER).
- **Nulls.** N4 (both ablations do nothing on a fresh-init model). **Prior.** ~75% for MLP and OV, ~50% for QK and embeddings. This is the key input to Rung 2: if function lives in the singular *vectors*, matching spectra cannot carry function.

### H3. QK/OV circuits are low stable-rank with an identity-like component
- **Statement.** (a) stable rank ||W||_F^2 / ||W||_2^2 of W_QK and W_OV is well below that of a random product of d x h Gaussians; (b) cos_F(W_QK, I) > 0 and cos_F(W_OV, I) != 0 (often negative).
- **Measurement.** Stable rank, participation ratio, singular-value entropy; Frobenius cosine against I; distributions over heads.
- **Refuted if** distributions over heads are KS-indistinguishable from products of random d x h matrices.
- **Literature.** Elhage et al. 2021; Trockman & Kolter 2023 (mimetic initialization): closest prior work to this project's Rung 2 idea; authored W_QK ~ aI + noise, W_OV ~ -bI speed training but are not tested at zero training.
- **Nulls.** N1 products. **Prior.** (a) ~85%; (b) ~70% tier A, ~50% tier C.

### H4. Token-embedding anisotropy with a dominant common direction tied to frequency
- **Statement.** Mean pairwise cosine >> 0; PC1 variance share far above the MP allowance; projection on PC1 correlates with token id (frequency proxy) and with corpus unigram counts.
- **Measurement.** Mean cosine over 10^6 random pairs; PCA variance ratios vs MP; Spearman vs token id and vs unigram counts from the M1 corpora.
- **Refuted if** mean cosine within 3 sigma of 0 (sigma ~ 1/sqrt(d)), PC1 within the MP edge, correlation ~ 0.
- **Literature.** Ethayarajh 2019; Mu & Viswanath 2018; Gao et al. 2019; Timkey & van Schijndel 2021.
- **Nulls.** N1, N6. **Prior.** ~95% anisotropy; ~80% frequency correlation in tier A.

### H5. Outlier residual dimensions, consistent across layers
- **Statement.** A handful of residual dims (GPT-2 small candidates 138, 378, 447) have anomalous LN gain/bias and anomalous column norms in reader matrices and row norms in writer matrices, and the same dims are anomalous across most layers.
- **Measurement.** Per-dim kurtosis of reader column norms and writer row norms and of LN parameters; top-5 dims per layer; count shared across >= L/2 layers vs hypergeometric expectation.
- **Refuted if** kurtosis within the null range and cross-layer overlap at chance.
- **Literature.** Kovaleva et al. 2021; Dettmers et al. 2022; Sun et al. 2024 (massive activations); Timkey & van Schijndel 2021.
- **Nulls.** N1, N4. **Prior.** ~90% tier A, ~80% Qwen, ~40% TinyStories-1M/3M.

### H6. Previous-token and induction heads are identifiable from weights alone
- **Statement.** Some head at layer l prefers offset -1 through its QK circuit acting on positional information; some head at layer m > l has K-composition with that head's OV far above random and an OV circuit that copies (positive projection of W_E W_OV W_U onto the identity over tokens).
- **Measurement (weights only).** Learned-pos models: S = P W_QK P^T, mean of the first lower off-diagonal minus the mean of other causal entries, in null sd units. RoPE models: tr(W_Q^T R_delta W_K C) for delta = 0..64 with C the embedding covariance; a previous-token head peaks at delta = 1. K-composition: ||W_QK^(m)T W_OV^(l)||_F / (||W_QK^(m)||_F ||W_OV^(l)||_F) vs random. Copying: trace of W_E W_OV W_U on the top-1000 tokens vs null.
- **Measurement (confirmation).** Attention to position i - period + 1 on a repeated random-token sequence; copy accuracy (M2).
- **Refuted if** no head exceeds 5 sigma on offset preference, no pair exceeds 5 sigma on K-composition, and M2 is at chance.
- **Literature.** Olsson et al. 2022; Elhage et al. 2021.
- **Nulls.** N1 head-shaped matrices; N3 row-permuted P. **Prior.** ~95% tiers A, B, TinyStories-8M/33M; ~60% TinyStories-1M.

### H7. Cross-layer alignment of reader/writer subspaces
- **Statement.** Top-k right singular subspaces of all reader matrices (W_Q, W_K, W_V, W_in across layers) overlap far more than random k-subspaces of R^d (expected k/d); writers' left subspaces align with the next layer's readers. Direct cos_F between consecutive layers' matrices is also tested and expected small.
- **Measurement.** Grassmann overlap tr(V_a^T V_b V_b^T V_a)/k for k in {8, 32, h}, as L x L matrices per matrix type; re-run with the top-5 outlier dims (H5) projected out.
- **Refuted if** overlaps within the random-subspace null.
- **Literature.** Elhage et al. 2021; Sun et al. 2024 (layers as painters); Men et al. 2024 (ShortGPT).
- **Nulls.** Haar subspaces; N5. **Prior.** ~80% above null; ~50% survives outlier removal; ~20% direct cosine (true weight sharing).

### H8. Spectral-shape universality across model size
- **Statement.** Normalized profiles sigma_i/sigma_1 vs i/n and fitted alpha per matrix type collapse across TinyStories 1M -> 33M, Pythia 70M -> 160M, GPT-2 and Qwen to within trained-vs-trained KS, much smaller than trained-vs-random.
- **Measurement.** Pairwise KS between normalized ESDs; alpha vs log(params) per type.
- **Refuted if** cross-size KS comparable to trained-vs-null KS.
- **Literature.** Martin & Mahoney (alpha tracks training quality: predicts non-universality).
- **Nulls.** N1 at each size, in normalized coordinates. **Prior.** ~55%; partial confirmation expected.

### H9. Fractal / self-similar structure (project hypothesis)
- **Statement (testable form).** On the ordered axes (position in W_pos; RoPE frequency index inside W_Q/W_K; token id in W_E) the signal is long-range correlated with a non-trivial scaling exponent (Hurst H != 0.5 or block-variance exponent != the iid value), and the thresholded |W| > t pattern has a box-counting dimension that differs from the row/column-permuted matrix. On unordered axes the pre-registered expectation is exactly null.
- **Measurement.** Detrended fluctuation analysis along the ordered axis for the top-10 PC time series; block-averaging variance vs block size (log-log slope); box-counting dimension of thresholded masks at 5 thresholds; all against N3.
- **Refuted if** every statistic within the N3 range on every axis. A power-law ESD (H1) is scale-free in the spectral sense and is **not** counted as fractal geometry.
- **Literature.** None specific to transformer weights.
- **Nulls.** N3. **Prior.** ~10% for anything fractal-like, mostly "positional embeddings are smooth along position", which is long-range correlation (H10), not self-similarity. ~0% on hidden-dim axes. A clean pre-registered negative is a result.

### H10. Positional structure: smooth low-dimensional learned positions; frequency specialization of RoPE heads
- **Statement.** Learned W_pos (GPT-2, TinyStories) is low-dimensional, smooth along position (neighbor distance << random-pair distance), leading PCs approximately sinusoidal. RoPE models: heads differ systematically in which frequency bands carry QK energy.
- **Measurement.** PCA share vs MP; ratio ||p_{i+1} - p_i|| / ||p_i - p_j||; FFT along position per PC; per-head per-band QK energy vs uniform.
- **Refuted if** W_pos indistinguishable from its row-permuted self; per-band energy flat within null.
- **Literature.** Wang & Chen 2020; Barbero et al. 2024 (round and round we go).
- **Nulls.** N3 for W_pos; N1 for W_Q/W_K. **Prior.** ~90%. Directly reusable for authoring.

### H11. Number / modular / elliptic-curve structure in embeddings (project hypothesis)
- **A (helix).** Embeddings of integer tokens n (GPT-2, Pythia: most of 0..999 are single tokens; Qwen: single digits only, tested on 0..9 and on two-token pairs) are well fit by the basis [n, cos(2 pi n/T), sin(2 pi n/T)] for T in {2, 5, 10, 100}, with R^2 far above random periods and above a shuffled n-to-embedding assignment.
- **B (modular Fourier).** The DFT along n of each embedding PC has power concentrated at periods 10, 100, 2, 5 relative to a shuffled assignment.
- **C (elliptic curve, stated plainly).** A finite elliptic curve's group is abelian, and abelian structure on tokens is testable only as additivity (the linear/helix part of A) or cyclic Fourier structure (B). The only distinct testable claim is geometric: number embeddings projected to their top 2-3 PCs lie near an algebraic cubic y^2 = x^3 + a x + b. Test: least-squares implicit cubic fit; residual vs shuffled assignment and vs a helix with the same parameter count.
- **Refuted if** A/B: R^2 and DFT peaks within the shuffle null; C: cubic residual not better than an equal-parameter helix.
- **Literature.** Kantamneni & Tegmark 2025 (helix with these periods in GPT-J, Pythia-6.9B, Llama-3.1-8B); Nanda et al. 2023 (Fourier features in grokked modular arithmetic; transfers only as B); Zhou et al. 2024.
- **Nulls.** N7, N1. **Prior.** A/B ~75% GPT-2 and Pythia-160M, ~60% Pythia-70M, ~70% Qwen (digit-level), ~15% TinyStories. C < 2%: no mechanism; expected to lose to a helix at equal parameter count. Run it; report it.

### H12. MLP neuron geometry: superposition signature and in/out alignment
- **Statement.** Rows of W_in have a pairwise-cosine distribution with excess kurtosis and a heavy tail of near-parallel/antipodal pairs; cos(W_in[i], W_out[i]) per neuron is biased rather than centered at 0; the fraction with |cos| > 3/sqrt(d) far exceeds the Gaussian 0.3%.
- **Measurement.** Pairwise-cosine histogram (10^6 pairs), kurtosis, tail fractions; per-neuron in/out cosine distribution.
- **Refuted if** all within N1 range.
- **Literature.** Elhage et al. 2022 (toy models of superposition); Geva et al. 2021/2022; Gurnee et al. 2023.
- **Nulls.** N1, N5. **Prior.** ~80% tail; ~70% in/out bias.

### H13 (capstone, Rung 2). Second-order statistics suffice for function
- **Statement.** Weights sampled to match the confirmed patterns (spectra, embedding covariance, cross-layer alignment, positional smoothness, outlier profile) give held-out loss materially better than shape-matched random initialization.
- **Expected outcome under the null.** No better than spectrum-matched random: A1 ~ A0 ~ ln V (~10.8 nats for a 50k vocab).
- **Prior.** ~10%.

Qwen-specific: Gated DeltaNet projection matrices get H1/H3/H5/H7/H12; the DeltaNet state-decay (A_log, dt_bias) and conv parameters get a descriptive table with no hypothesis.

## 5. Rung 2 protocol

### Conditions (each adds one confirmed pattern; every set is a full state dict loaded into the stock HF class and run with the KV cache)
- **A0** fresh init (N4). Control.
- **A1** spectrum-matched random (N5) for every matrix; embeddings N6 with random token assignment; norm gains matched marginally. Control; pre-registered expectation: equals A0.
- **A2** A1 + ordered-axis patterns: W_pos as a smooth curve with the trained PC spectrum and FFT profile (H10); outlier dims placed consistently at measured magnitudes (H5); embedding common direction scaled by token-id rank (H4). Token-id rank is admitted by the owner as a frequency proxy and reported on its own line.
- **A3** A2 + cross-layer alignment (H7) via a shared random residual basis with mixing fitted to the measured overlap matrices; QK/OV identity components at measured cos_F (H3). Singular values from N5.
- **A4** A3 + explicit previous-token and induction circuits at the layers where H6 finds them, built with `handi/author`. Validates the circuit-reading and the engine; **its success does not count toward H13** and is reported separately as "function from an explicitly authored circuit".
- **A5** trained (upper bound); **A5'** trained with bulk randomized; **A5''** trained with top-k randomized (H2).

### Metrics
- **M1** mean next-token cross-entropy (nats) on WikiText-103 test and TinyStories validation, >= 100k tokens each, bootstrap CI over 1k-token chunks; reference lines: uniform ln V, unigram entropy, bigram entropy, A5.
- **M2** copy accuracy on 200 sequences of 50 random tokens repeated twice (second half); chance 1/V.
- **M3** sampled-text statistics: 50k tokens at temperature 1; KL of sample bigrams to corpus bigrams vs unigram sampling; fraction of whole dictionary words.
- **M4** LegalBench accuracy on a fixed subset of binary-label tasks (chosen at Part B step 1 and recorded in `docs/decisions.md` before any Rung 2 run), trained Qwen3.5-0.8B as upper bound, majority class as floor.
- 5 seeds per condition.

### What counts as what
- **Null outcome:** A1, A2 (without token-id input), A3 all within 0.1 nats of A0 on M1, at chance on M2 and M4, M3 no better than unigram sampling.
- **Positive-weak:** some A_k (k <= 3) beats A1 by >= 0.2 nats on M1 with non-overlapping 95% CIs across 5 seeds in at least two tiers.
- **Positive-strong:** some A_k (k <= 3) reaches or beats the unigram line without token-identity input (~5%).
- **A2 with token-id rank** reaching the unigram line is a legitimate, modest positive ("anisotropy encodes frequency"), never conflated with syntax.
- **Law-positive:** any authored A_k (k <= 4) beats majority class on any LegalBench task with CI.
- A4 succeeding on M2 is expected (known construction) and validates the pipeline only.

## 6. Order of work
1. Registry, pinned downloads, manifest, `docs/models/*.md`, LegalBench subset chosen.
2. Raw loader and canonical schema, checked against transformers logits per family.
3. Nulls, spectra, stats, results writer; MP closed form unit-checked on Gaussians.
4. **Freeze this document.**
5. H1, H3, H4, H5, H12. 6. H6, H7, H10. 7. H9, H11, H8. 8. H2. 9. Rung 2. 10. Summary and H13 verdict.
