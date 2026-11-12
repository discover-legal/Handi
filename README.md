# Handi: hand-authored language-model weights

An experiment in writing the weights of a transformer language model without training it, and in what
that shows about where a trained model keeps its knowledge, including its knowledge of law.

Repository: https://github.com/discover-legal/Handi

**Start with the report:** [`docs/report.html`](docs/report.html) (open it in a browser). It is written for
readers who are not machine-learning specialists and defines every term it uses.

## What is here

| Path | What it is |
|---|---|
| `docs/report.html` | The report: question-led, with charts drawn from the result tables |
| `docs/writeup.md` | Technical write-up of the weight-structure work and Rungs 1 and 2 (the law work is in the report, the cookbook and `docs/decisions.md`) |
| `docs/hypotheses.md` | The nineteen hypotheses (thirteen pre-registered, six added after results) and their pass rules |
| `docs/decisions.md` | Every decision, dated, including those taken after results were seen; run notes |
| `docs/cookbook.md` | Recipe and code for writing a task read-out into a model's output rows (the Rung 3 edit) |
| `docs/plan.md` | The original plan |
| `results/tables/*.csv` | Every measured number, one row each, with the git commit and timestamp of its run |
| `results/summary.md` | Machine-generated verdicts per hypothesis |
| `analysis/` | One script per experiment (`h01_*.py` to `h19_*.py`, `run_rung2.py`, `run_rung3.py` and `run_rung3_*.py` for the law work (`run_rung3_all.py` runs all of LegalBench), `hearsay_direction.py`, `hearsay_steer.py`, `run_rung4_fit.py`) |
| `handi/authoring/` | The weight-authoring code: `generators.py` (conditions A0 to A9), `bigram.py`, `circuits.py`, `chords.py` |
| `handi/adder/` | Rung 1: a hand-written 10-digit adder in the Qwen3.5 architecture |
| `tests/` | Tests and the LegalBench prompt templates |

## Models and data

Weight structure and hand-written models: GPT-2 small, Pythia-70M, Pythia-160M, Qwen3.5-0.8B, TinyStories
1M/3M/8M/33M (downloaded from the Hugging Face Hub into `models/`, which is not committed). Law work: Qwen3.5-0.8B,
Phi-4-mini-instruct, and Qwen3.5-2B, 4B and 9B (4-bit). Corpora: WikiText-103, TinyStories, LegalBench (Guha et al.
2023, https://arxiv.org/abs/2308.11462; 157 tasks run), LexGLUE (CaseHOLD, LEDGAR).

## Running

```
uv sync
.venv/bin/python analysis/run_rung2.py --models pythia-70m --conditions A0 A1 A8 --seeds 0
.venv/bin/python analysis/run_rung3.py
```

Every script writes rows to `results/tables/` through a single writer that stamps the commit and time. An 8 GB
GPU is enough for the 0.8B models and Phi-4-mini. It is not enough for everything: the Qwen3.5-4B ran out of
memory on 36 long tasks (left out), and the 9B runs only in 4-bit, which needs `bitsandbytes` (not in the lockfile;
install it separately).

## Credits

Project owner: hordruma. Builder: Claude (Fable 5.1, later Opus 5.5), which wrote the code, ran the experiments and
drafted the documents under the owner's decisions. Prompted by Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight-Space Programming: LLMs can natively read & write neural network weights" (MIT CSAIL, 2 Nov 2026,
https://rikiyat.github.io/blog/weight-space-programming/). Prior work the project builds on is listed at the end of
the report. The read-out edit is packaged as a tool: https://github.com/discover-legal/lmreadout.
