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
