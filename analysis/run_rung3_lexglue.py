"""Rung 3 on LexGLUE CaseHOLD (owner, 2026-11-10: try LexGLUE; CaseHOLD with the proposed prompt; SCOTUS skipped).

CaseHOLD (coastalcph/lex_glue, config case_hold, test split, 3,600 items, CC BY 4.0): a passage citing a case,
with <HOLDING> where the cited holding goes, and five candidate holdings; the answer is the number of the right
one. Prompt (tests/fixtures/lexglue_prompts/case_hold.txt, owner-approved proposal): the citing passage, the five
holdings numbered 0 to 4, ending in "Answer:"; no worked examples. The digits are the answer rows, as for scalr
(the tokenizer writes " 3" as a space token then the digit, so the prompt ends with the space).
Everything else is run_rung3_all.py unchanged: the same K-label formula, word-count baseline on the item's own
fields, protocols few{N} (sizes given on the command line), twofold on the 500-item sample, fresh on up to 500
other items. This script only swaps in the CaseHOLD items and the prompt folder, then calls its main().
Rows: results/tables/rung3_lexglue.csv (hypothesis R3A).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis.run_rung3_all as R  # noqa: E402

_cache = {}


def lexglue_items(task, max_items=500, seed=0):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    if task != "case_hold":
        raise ValueError(task)
    if "rows" not in _cache:
        df = pd.read_parquet(hf_hub_download("coastalcph/lex_glue", "case_hold/test-00000-of-00001.parquet", repo_type="dataset"))
        _cache["rows"] = [{"context": r.context, **{f"choice_{i}": str(e) for i, e in enumerate(r.endings)}, "answer": str(r.label)}
                          for r in df.itertuples()]
    rows = _cache["rows"]
    if len(rows) > max_items:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), max_items, replace=False))]
    return rows


R.PROMPTS = ROOT / "tests/fixtures/lexglue_prompts"
R.EV.legalbench_items = lexglue_items

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(ROOT / "results/tables/rung3_lexglue.csv")]
    if "--tasks" not in sys.argv:
        sys.argv += ["--tasks", "case_hold"]
    R.main()
