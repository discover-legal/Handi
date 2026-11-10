"""Does the Rung 3 edit read the cue phrase or the clause's function? For the two near-perfect tasks, the edit is
fitted on the 500-item sample as before and scored on the 500 never-seen items three ways: the original text, the
text with the cue phrases masked out ("[...]"), and the text with the cue phrases paraphrased into plain words that
keep the meaning but drop the stock wording. Rows: results/tables/rung3_cue.csv (R3C)."""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from handi.analysis.results import ResultsWriter
from handi.authoring import evaluate as EV
from analysis.run_rung3 import collect
from analysis.run_rung3_transfer import fit_edit, apply, acc_of

CUES = {
  "cuad_governing_law": [
    (r"governed by and construed in accordance with the laws? of", "subject to the rules of"),
    (r"governed by and construed (?:under|in accordance with)", "subject to"),
    (r"construed (?:and enforced )?in accordance with the laws? of", "read under the rules of"),
    (r"governed by the laws? of", "subject to the rules of"),
    (r"governed (?:by|under)", "subject to"),
    (r"in accordance with the laws? of", "under the rules of"),
    (r"the laws? of", "the rules of"),
    (r"\bconstrued\b", "read"),
    (r"\bgoverning law\b", "applicable rules"),
    (r"\blaws?\b", "rules"),
  ],
  "definition_classification": [
    (r"\bis defined (?:as|to mean|to include)\b", "is"),
    (r"\bshall mean\b", "shall be"),
    (r"\bmeans\b", "is"),
    (r"\bmeaning\b", "sense"),
    (r"\bdefinitions?\b", "description"),
    (r"\bdefines?\b", "describes"),
    (r"\bdefined\b", "described"),
    (r"\brefers? to\b", "is"),
    (r"\bthe term\b", "the word"),
  ],
}

def variants(text, cues):
    masked, para = text, text; hit = False
    for pat, rep in cues:
        if re.search(pat, masked, flags=re.I):
            hit = True
        masked = re.sub(pat, "[...]", masked, flags=re.I)
        para = re.sub(pat, rep, para, flags=re.I)
    return masked, para, hit

def main():
    model, tok = EV.load_trained("qwen3.5-0.8b")
    with ResultsWriter(str(ROOT / "results/tables/rung3_cue.csv")) as w:
        base = dict(hypothesis="R3C", model="qwen3.5-0.8b", family="qwen3_5", src_dtype="BF16", seed=0, layer="ALL")
        for task, cues in CUES.items():
            template = (ROOT / "tests/fixtures/legalbench_prompts" / f"{task}.txt").read_text()
            all_rows = EV.legalbench_items(task, 10 ** 9); rows = EV.legalbench_items(task, 500)
            labels = sorted({r["answer"] for r in all_rows}); ft = [tok(" " + l, add_special_tokens=False)["input_ids"][0] for l in labels]
            seen = {tuple(sorted(r.items())) for r in rows}; fresh = [r for r in all_rows if tuple(sorted(r.items())) not in seen]
            rng = np.random.default_rng(1)
            if len(fresh) > 500: fresh = [fresh[i] for i in sorted(rng.choice(len(fresh), 500, replace=False))]
            H, LOG, _ = collect(model, tok, [EV.fill_prompt(template, r) for r in rows]); gold = np.array([labels.index(r["answer"]) for r in rows])
            e = fit_edit(H, LOG, gold, ft)
            gf = np.array([labels.index(r["answer"]) for r in fresh])
            hits = np.array([variants(r["text"], cues)[2] for r in fresh])
            print(f"\n[{task}] fresh n={len(gf)}; items containing a cue phrase: Yes {hits[gf == 1].mean():.2f}, No {hits[gf == 0].mean():.2f}", flush=True)
            ex = next(r for r in fresh if variants(r["text"], cues)[2] and r["answer"] == "Yes")
            m, p, _ = variants(ex["text"], cues)
            print(f"  example original:    {ex['text'][:150]}\n  example masked:      {m[:150]}\n  example paraphrased: {p[:150]}", flush=True)
            for name, idx in [("original", 0), ("masked", 1), ("paraphrased", 2)]:
                texts = [r["text"] if idx == 0 else variants(r["text"], cues)[idx - 1] for r in fresh]
                prompts = [EV.fill_prompt(template, {**r, "text": t}) for r, t in zip(fresh, texts)]
                Hf, LOGf, _ = collect(model, tok, prompts)
                p0 = (LOGf[:, ft[1]] > LOGf[:, ft[0]]).astype(int); L1 = apply(LOGf, Hf, e); p1 = (L1[:, ft[1]] > L1[:, ft[0]]).astype(int)
                a0, a1 = float((p0 == gf).mean()), float((p1 == gf).mean())
                rec1 = float((p1[gf == 1] == 1).mean()); spec1 = float((p1[gf == 0] == 0).mean())
                rec_hit = float((p1[(gf == 1) & hits] == 1).mean()) if ((gf == 1) & hits).any() else float("nan")
                for k, v in [("acc_shipped", a0), ("acc_edited", a1), ("edited_recall_yes", rec1), ("edited_spec_no", spec1), ("edited_recall_yes_cue_items", rec_hit)]:
                    w.row(metric=f"{k}_{task}", component=name, value=v, null_n=len(gf), **base)
                print(f"  {name:12s} as shipped={a0:.3f} edited={a1:.3f} | edited: Yes items correctly Yes={rec1:.3f} (cue-bearing Yes items {rec_hit:.3f}), No items correctly No={spec1:.3f}", flush=True)

if __name__ == "__main__":
    main()
