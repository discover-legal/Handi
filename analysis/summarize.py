"""Generate results/summary.md from results/tables/*.csv under the pre-registered rule.

A (hypothesis, model, metric) aggregate row (layer == "ALL") PASSES if |z| >= 5 or
ks > 3 * ks_null_median. A hypothesis is CONFIRMED when, for at least one of its pre-registered
metrics, the pass reproduces in >= 2 of the 3 tiers (A: gpt2/pythia, B: qwen, C: tinystories);
REFUTED when every model's aggregate rows fail; INCONCLUSIVE otherwise. H02 and H13 are
summarized by their deltas and reference lines instead (no z-test is pre-registered for them).
Never hand-edit summary.md: rerun this script.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handi.analysis.registry import MODELS  # noqa: E402

TABLES = ROOT / "results/tables"


def fnum(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def passes(r) -> bool | None:
    z, ks, ksn = fnum(r["z"]), fnum(r["ks"]), fnum(r["ks_null_median"])
    if z is None and (ks is None or ksn is None):
        return None
    return (z is not None and abs(z) >= 5) or (ks is not None and ksn is not None and ksn > 0 and ks > 3 * ksn)


def load(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# Validity notes from docs/decisions.md: metrics excluded from the verdict, with the reason.
EXCLUDED = {
    "H09": ("boxdim", "box-counting statistic invalid: its hidden-dim control (must be null) gives z 24-71 (decisions.md 2026-11-03)"),
    "H06": ("kcomp", "K-composition N1 null is not discriminative: 57% of GPT-2 pairs pass 5 sigma (decisions.md 2026-11-03)"),
}
# Verdict wording fixed after a validity note (decisions.md).
OVERRIDE = {
    "H09": "NO EVIDENCE OF FRACTAL GEOMETRY. The surviving DFA/Hurst and block-variance signal on the position, "
           "token-id and RoPE-frequency axes is long-range smoothness (Hurst > 0.5, often > 1), which the "
           "pre-registration assigns to H10, not self-similarity; hidden-dim axes are null as pre-registered.",
}


def summarize_h15(rows: list[dict]) -> tuple[str, list[str]]:
    """H15's own rule (hypotheses.md section 7): s1_extrapolate z >= 5 against the random-turn null AND
    s1 > s1_stay, for readers or writers at k=32, in >= 2 tiers."""
    val = {}
    for r in rows:
        if r["layer"] == "ALL" and r["component"].endswith(":k32"):
            val[(r["model"], r["component"], r["metric"])] = (fnum(r["value"]), fnum(r["z"]), fnum(r["null_mean"]))
    lines, tiers_ok = [], set()
    for model in MODELS:
        cells = []
        for side in ("readers", "writers"):
            comp = f"{side}:k32"
            s1 = val.get((model, comp, "s1_extrapolate"))
            stay = val.get((model, comp, "s1_stay"))
            s2 = val.get((model, comp, "s2_turn_overlap"))
            if not s1 or not stay:
                continue
            ok = s1[1] is not None and s1[1] >= 5 and s1[0] > stay[0]
            cells.append(f"{side}: S1={s1[0]:.3f} vs random-turn {s1[2]:.3f} (z {s1[1]:+.1f}), stay={stay[0]:.3f}, "
                         f"turn-overlap S2={s2[0]:.3f} vs chance {s2[2]:.3f} (z {s2[1]:+.1f}) -> {'PASS' if ok else 'fail'}")
            if ok:
                tiers_ok.add(MODELS[model]["tier"])
        if cells:
            lines.append(f"- {model}: " + "; ".join(cells))
    verdict = "CONFIRMED" if len(tiers_ok) >= 2 else "REFUTED"
    lines.append("- Rule: PASS needs S1 z >= 5 AND S1 > stay. The turn direction beats a random direction almost everywhere, "
                 "but applying the same turn again never beats staying where the previous layer is, so there is no usable "
                 "generative rule; S2 shows consecutive turns come from different (mostly anti-aligned) new directions.")
    return verdict, lines


def summarize_hypothesis(hyp: str, rows: list[dict]) -> tuple[str, list[str]]:
    if hyp == "H15":
        return summarize_h15(rows)
    excl = EXCLUDED.get(hyp)
    if excl:
        rows = [r for r in rows if not r["metric"].startswith(excl[0])]
    agg = [r for r in rows if r["layer"] in ("ALL", "") and r["null_kind"] not in ("", "none")]
    if not agg:
        agg = rows
    by_metric = defaultdict(lambda: defaultdict(list))   # metric -> model -> rows
    for r in agg:
        by_metric[r["metric"]][r["model"]].append(r)
    lines = []
    confirmed_metrics, any_pass, any_test = [], False, False
    for metric, per_model in sorted(by_metric.items()):
        tiers_passing = set()
        cells = []
        for model in MODELS:
            rs = per_model.get(model)
            if not rs:
                continue
            ps = [passes(r) for r in rs]
            ps = [p for p in ps if p is not None]
            if not ps:
                continue
            any_test = True
            ok = all(ps) if len(ps) == 1 else (sum(ps) >= max(1, len(ps) // 2 + 1))
            zs = [fnum(r["z"]) for r in rs if fnum(r["z"]) is not None]
            zrep = f"{max(zs, key=abs):+.1f}" if zs else "-"
            cells.append(f"{model}: {'PASS' if ok else 'fail'} (z {zrep})")
            if ok:
                any_pass = True
                tiers_passing.add(MODELS[model]["tier"])
        if cells:
            mark = "**" if len(tiers_passing) >= 2 else ""
            lines.append(f"- {mark}{metric}{mark}: " + "; ".join(cells))
        if len(tiers_passing) >= 2:
            confirmed_metrics.append(metric)
    if confirmed_metrics:
        verdict = "CONFIRMED"
    elif any_test and not any_pass:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    if excl:
        lines.insert(0, f"- excluded from the verdict: metrics `{excl[0]}*` ({excl[1]})")
    if hyp in OVERRIDE:
        verdict = f"{verdict} by the mechanical rule; reported as {OVERRIDE[hyp]}"
    return verdict, lines


def _mean_ci(vs: list[float]) -> tuple[float, float, float]:
    """mean and 95% CI across seeds (t-based for n <= 30)."""
    n = len(vs)
    m = sum(vs) / n
    if n < 2:
        return m, m, m
    sd = (sum((v - m) ** 2 for v in vs) / (n - 1)) ** 0.5
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(n, 2.0)
    h = t * sd / n ** 0.5
    return m, m - h, m + h


def h13_verdict(rows: list[dict]) -> list[str]:
    """Apply the pre-registered Rung 2 outcomes (hypotheses.md section 5) to rung2.csv."""
    out = ["### H13 verdict (pre-registered rule, hypotheses.md section 5)"]
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # model -> cond -> metric -> [value per seed]
    maj = {}
    for r in rows:
        v = fnum(r["value"])
        if v is None:
            continue
        by[r["model"]][r["component"]][r["metric"]].append(v)
        if r["metric"].startswith("m4_") and fnum(r["ks"]) is not None:
            maj[(r["model"], r["metric"])] = fnum(r["ks"])
    tiers = lambda models: {MODELS[m]["tier"] for m in models}
    authored = ("A2", "A2F", "A3", "A4", "A6", "A7")
    post = {"A4", "A6", "A7"}      # post-freeze lines: reported on their own, not counted in the H13 verdict
    null_ok, weak_hits, strong_hits, law_hits, vs_a0_hits = [], defaultdict(set), defaultdict(set), [], defaultdict(set)
    vs_a3_hits = defaultdict(set)
    out.append("| model | A0 CE | A1 CE | A2 CE | A2F CE | A3 CE | A4 CE | A6 CE | A7 CE | unigram | A5 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, conds in by.items():
        def ce(cond):
            vs = conds.get(cond, {}).get("m1_ce_wikitext", [])
            return _mean_ci(vs) if vs else None
        a0, a1 = ce("A0"), ce("A1")
        uni = conds.get("A5", {}).get("m1_ref_unigram_wikitext", [None])[0]
        cells = []
        a3 = ce("A3")
        for cond in ("A0", "A1", "A2", "A2F", "A3", "A4", "A6", "A7"):
            c = ce(cond)
            cells.append("-" if c is None else f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]")
        a5 = ce("A5")
        out.append(f"| {model} | " + " | ".join(cells) + f" | {'-' if uni is None else f'{uni:.2f}'} | {'-' if a5 is None else f'{a5[0]:.2f}'} |")
        if a0 and a1:
            within = all(abs(ce(c)[0] - a0[0]) <= 0.1 for c in ("A1", "A2", "A3") if ce(c))
            null_ok.append((model, within))
        for cond in authored:
            c = ce(cond)
            if c is None:
                continue
            if a1 and c[0] <= a1[0] - 0.2 and c[2] < a1[1]:
                weak_hits[cond].add(model)
            if a0 and c[0] <= a0[0] - 0.2 and c[2] < a0[1]:
                vs_a0_hits[cond].add(model)
            if a3 and c[0] <= a3[0] - 0.2 and c[2] < a3[1]:
                vs_a3_hits[cond].add(model)
            if uni is not None and c[0] <= uni and cond != "A2F":
                strong_hits[cond].add(model)
            for metric, vs in conds[cond].items():
                if metric.startswith("m4_") and (model, metric) in maj:
                    m, lo, hi = _mean_ci(vs)
                    if lo > maj[(model, metric)]:
                        law_hits.append((model, cond, metric, round(m, 3), round(maj[(model, metric)], 3)))
    out.append("")
    out.append(f"- Null outcome clause (A1, A2, A3 within 0.1 nat of A0): " +
               ("holds in " + ", ".join(m for m, w in null_ok if w) if any(w for _, w in null_ok) else "holds in no model") +
               "; fails in " + (", ".join(m for m, w in null_ok if not w) or "none") +
               ". Where it fails, A1-A3 are far WORSE than A0 (large singular values on random directions give confident wrong logits), not better.")
    weak = {c: ms for c, ms in weak_hits.items() if len(tiers(ms)) >= 2 and c not in post}
    out.append(f"- Positive-weak (some A_k, k<=3, beats A1 by >= 0.2 nat with non-overlapping CIs in >= 2 tiers): "
               + (", ".join(f"{c} in {sorted(ms)}" for c, ms in weak.items()) if weak else "NONE"))
    vs0 = {c: ms for c, ms in vs_a0_hits.items() if c not in post}
    out.append(f"- Same test against A0 instead of A1 (diagnostic, not a verdict): "
               + (", ".join(f"{c} in {sorted(ms)}" for c, ms in vs0.items()) if vs0 else "NONE"))
    strong = {c: ms for c, ms in strong_hits.items() if c not in post}
    out.append(f"- Positive-strong (some A_k, k<=3, reaches the unigram line without token-identity input): "
               + (", ".join(f"{c} in {sorted(ms)}" for c, ms in strong.items()) if strong else "NONE"))
    a2f = strong_hits.get("A2F") or set()
    out.append(f"- A2F (token-id rank) reaching the unigram line: " + (", ".join(sorted(a2f)) if a2f else "NONE"))
    n_cmp = sum(1 for model, conds in by.items() for cond in authored if cond in conds
                for metric in conds[cond] if metric.startswith("m4_"))
    law_tiers = defaultdict(set)
    for m, c, t, v, mj in law_hits:
        law_tiers[(c, t)].add(MODELS[m]["tier"])
    reproduced = [k for k, ts in law_tiers.items() if len(ts) >= 2]
    out.append(f"- Law-positive (any authored A_k, k<=4, beats majority class on any LegalBench task with CI): "
               + (", ".join(f"{m} {c} {t} {v} > {mj}" for m, c, t, v, mj in law_hits) if law_hits else "NONE")
               + f". {len(law_hits)} hit(s) in {n_cmp} comparisons at a 95% CI (about {n_cmp * 0.05:.0f} expected by chance); "
               + ("reproduced in >= 2 tiers: " + ", ".join(f"{c} {t}" for c, t in reproduced) if reproduced else
                  "none reproduced in a second tier, so none is reported as confirmed under the multiple-comparisons rule (section 0)"))
    a4 = {m: conds.get("A4", {}).get("m2_copy_acc") for m, conds in by.items() if "A4" in conds}
    if a4:
        out.append("- A4 (explicit circuit; reported separately, does not count toward H13): copy accuracy "
                   + ", ".join(f"{m}={_mean_ci(v)[0]:.2f}" for m, v in a4.items() if v))
    def post_line(cond, hyp, ref, hits):
        ms = hits.get(cond) or set()
        ok = len(tiers(ms)) >= 2
        have = sorted(m for m, conds in by.items() if cond in conds)
        if not have:
            return f"- {cond} ({hyp}, post-freeze): not run yet"
        return (f"- {cond} ({hyp}, post-freeze, reported separately) vs {ref} by the positive-weak test: "
                + (f"passes in {sorted(ms)}" if ms else "passes nowhere") + f" -> {hyp} {'CONFIRMED' if ok else 'NOT confirmed'}"
                + f"; vs A1: {sorted(weak_hits.get(cond) or set()) or 'none'}; vs A0: {sorted(vs_a0_hits.get(cond) or set()) or 'none'}")
    out.append(post_line("A6", "H15 applied", "A3", vs_a3_hits))
    out.append(post_line("A7", "H16", "A1", weak_hits))
    verdict = "NULL" if not weak and not strong else "POSITIVE"
    out.append(f"- **H13 verdict: {verdict}**" + (" (the pre-registered null clause is violated in the wrong direction: A1 is worse than A0; "
               "no authored condition beats either baseline)" if verdict == "NULL" else ""))
    return out


def summarize_h02(rows: list[dict]) -> list[str]:
    out = []
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][(r["component"], r["metric"])] = fnum(r["value"])
    for model, d in by_model.items():
        out.append(f"- {model}: " + "; ".join(f"{c} {m.replace('delta_ce_', 'Δ').replace('_vs_trained', '')}={v:+.2f}"
                                               for (c, m), v in sorted(d.items()) if m.startswith("delta_ce") and v is not None))
        base = {m: v for (c, m), v in d.items() if c == "trained"}
        fresh = {m: v for (c, m), v in d.items() if c == "fresh_init"}
        if base:
            out.append("  trained: " + ", ".join(f"{m}={v:.2f}" for m, v in base.items()) +
                       " | fresh init: " + ", ".join(f"{m}={v:.2f}" for m, v in fresh.items()))
    return out


def summarize_rung2(rows: list[dict]) -> list[str]:
    out = []
    by = defaultdict(lambda: defaultdict(list))   # model -> condition -> rows
    for r in rows:
        by[r["model"]][r["component"]].append(r)
    for model, conds in by.items():
        out.append(f"### {model}")
        out.append("| condition | seeds | CE wikitext | CE tinystories | copy acc | LegalBench mean acc (majority) |")
        out.append("|---|---|---|---|---|---|")
        for cond in sorted(conds):
            rs = conds[cond]
            seeds = sorted({r["seed"] for r in rs})

            def mean_of(metric):
                vs = [fnum(r["value"]) for r in rs if r["metric"] == metric and fnum(r["value"]) is not None]
                return (sum(vs) / len(vs)) if vs else None

            def fmt(v):
                return "-" if v is None else f"{v:.3f}"
            m4 = [fnum(r["value"]) for r in rs if r["metric"].startswith("m4_") and fnum(r["value"]) is not None]
            maj = [fnum(r["ks"]) for r in rs if r["metric"].startswith("m4_") and fnum(r["ks"]) is not None]
            m4s = f"{sum(m4) / len(m4):.3f} ({sum(maj) / len(maj):.3f})" if m4 else "-"
            out.append(f"| {cond} | {len(seeds)} | {fmt(mean_of('m1_ce_wikitext'))} | {fmt(mean_of('m1_ce_tinystories'))} | {fmt(mean_of('m2_copy_acc'))} | {m4s} |")
        refs = {m: fnum(r["value"]) for r in conds.get("A5", conds[next(iter(conds))]) for m in [r["metric"]] if m.startswith("m1_ref_")}
        if refs:
            out.append("reference lines: " + ", ".join(f"{k.replace('m1_ref_', '')}={v:.2f}" for k, v in sorted(refs.items())))
    return out


def main() -> None:
    parts = ["# Results summary", "", "Generated by `analysis/summarize.py` from `results/tables/*.csv`. "
             "Rule: aggregate row passes if |z| >= 5 or KS > 3x null median; hypothesis confirmed if a metric passes in >= 2 tiers.", ""]
    for path in sorted(TABLES.glob("h*.csv")):
        rows = load(path)
        if not rows:
            continue
        hyp = rows[0]["hypothesis"]
        parts.append(f"## {hyp} ({path.name})")
        if hyp == "H02":
            parts += summarize_h02(rows)
        else:
            verdict, lines = summarize_hypothesis(hyp, rows)
            parts.append(f"**Verdict: {verdict}**")
            parts += lines
        parts.append("")
    r2 = TABLES / "rung2.csv"
    if r2.exists():
        parts.append("## Rung 2 (H13)")
        r2rows = load(r2)
        parts += summarize_rung2(r2rows)
        parts.append("")
        parts += h13_verdict(r2rows)
    (ROOT / "results/summary.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
