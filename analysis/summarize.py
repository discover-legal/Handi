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


def summarize_hypothesis(hyp: str, rows: list[dict]) -> tuple[str, list[str]]:
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
    return verdict, lines


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
        parts += summarize_rung2(load(r2))
    (ROOT / "results/summary.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
