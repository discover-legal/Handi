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
| `docs/writeup.md` | The full technical write-up |
| `docs/hypotheses.md` | The nineteen pre-registered hypotheses and their pass rules |
| `docs/decisions.md` | Every decision, dated, including those taken after results were seen; run notes |
| `docs/cookbook.md` | Recipe and code for writing a task read-out into a model's output rows (the Rung 3 edit) |
| `docs/plan.md` | The original plan |
| `results/tables/*.csv` | Every measured number, one row each, with the git commit and timestamp of its run |
| `results/summary.md` | Machine-generated verdicts per hypothesis |
| `analysis/` | One script per experiment (`h01_*.py` to `h19_*.py`, `run_rung2.py`, `run_rung3.py`, `hearsay_direction.py`, `hearsay_steer.py`, `run_rung4_fit.py`) |
| `handi/authoring/` | The weight-authoring code: `generators.py` (conditions A0 to A9), `bigram.py`, `circuits.py`, `chords.py` |
| `handi/adder/` | Rung 1: a hand-written 10-digit adder in the Qwen3.5 architecture |
| `tests/` | Tests and the LegalBench prompt templates |

## Models and data

GPT-2 small, Pythia-70M, Pythia-160M, Qwen3.5-0.8B, TinyStories 1M/3M/8M/33M (downloaded from the Hugging Face
Hub into `models/`, which is not committed). Corpora: WikiText-103, TinyStories, LegalBench (ten classification
tasks).

## Running

```
uv sync
.venv/bin/python analysis/run_rung2.py --models pythia-70m --conditions A0 A1 A8 --seeds 0
.venv/bin/python analysis/run_rung3.py
```

Every script writes rows to `results/tables/` through a single writer that stamps the commit and time. An 8 GB
GPU is enough for everything here.

## Credits

Project owner: hordruma. Builder: Claude (Fable 5.1), which wrote the code, ran the experiments and drafted the
documents under the owner's decisions. Prompted by Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space
Programming" (MIT CSAIL, 2026).
