"""Our LegalBench scores beside the scores the LegalBench paper reports (owner, 2026-11-10: compare with
published scores).

Paper: Guha et al., "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language
Models", arXiv 2308.11462 v1, appendix Tables 60 to 79 (per-task scores of the commercial and 13B models; the
rule-application Table 59, scored correctness / analysis, is left out). results/tables/legalbench_paper_scores.csv
holds those rows as printed (percent), read from the PDF's text with pdftotext -layout and parsed by the table
headers; rows whose task name the PDF wraps over two lines are missing.
The paper's models answer by few-shot generation (0 to 8 worked examples in the prompt, temperature 0), scored by
balanced accuracy for classification, accuracy for definition_extraction, F1 for the ssla tasks. Ours: balanced
accuracy, the model as shipped restricted to the label tokens, and the edit fitted on 15 labelled items (mean
of 20 draws) or on half the sample (two-fold). The item sets also differ (ours: at most 1,000 items per task).
So this is a rough comparison.
Writes results/tables/legalbench_paper_compare.csv: one row per task found in both.
"""
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results/tables"
MODELS = ["GPT-4", "GPT-3.5", "Claude-1", "Flan-T5-XXL", "LLaMA-2-13B"]


def load(p):
    return {(r["component"], r["metric"]): float(r["value"]) for r in csv.DictReader(open(p)) if r["value"] != ""}


def main():
    P = {r["task"]: {k: float(v) / 100 for k, v in r.items() if k != "task" and v} for r in csv.DictReader(open(T / "legalbench_paper_scores.csv"))}
    Q, F, U = load(T / "rung3_all.csv"), load(T / "rung3_all_phi.csv"), load(T / "rung3_all_uncut.csv")
    ours = sorted({m[5:] for (_, m) in Q if m.startswith("done_")})
    rows = []
    for t in ours:
        if t not in P:
            continue
        g = lambda D, c, p: D.get((c, f"{p}_bal_{t}"))
        r = {"task": t, "labels": int(Q.get(("n_labels", f"info_{t}"), 2)), **{m: P[t].get(m) for m in MODELS},
             "qwen_raw": g(Q, "shipped", "few15"), "qwen_edit15": g(Q, "edited", "few15"),
             "qwen_edit15_whole": g(U, "edited", "few15") or g(Q, "edited", "few15"), "qwen_edit_half": g(Q, "edited", "twofold"),
             "phi_raw": g(F, "shipped", "few15"), "phi_edit15": g(F, "edited", "few15"), "phi_edit_half": g(F, "edited", "twofold"),
             "wordcount15": g(Q, "wordcount", "few15")}
        rows.append(r)
    cols = list(rows[0])
    with open(T / "legalbench_paper_compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    full = [r for r in rows if all(r[c] is not None for c in cols)]
    print(f"{len(rows)} tasks in both, {len(full)} with every column")
    for name, sel in [("all", lambda r: True), ("two labels", lambda r: r["labels"] == 2), ("3 to 8 labels", lambda r: r["labels"] > 2)]:
        s = [r for r in full if sel(r)]
        print(name, len(s), " ".join(f"{c}={st.mean(r[c] for r in s):.3f}" for c in cols[2:]))
    for c in ["qwen_edit15", "qwen_edit15_whole", "qwen_edit_half", "phi_edit15", "phi_edit_half"]:
        print(c, "at or above:", ", ".join(f"{m} {sum(r[c] >= r[m] for r in full)}" for m in MODELS), f"of {len(full)}")


if __name__ == "__main__":
    main()
