"""H8: spectral-shape universality across model size (hypotheses.md H8).

Reads the raw singular values saved by analysis/h01_spectra.py (results/raw/h01/<model>/<matrix>.npz:
s_trained, s_n1 [<=5 draws]) and the per-matrix alpha from results/tables/h01.csv.
Per component type (embed.tok, attn.q.w, ..., mlp.up.w, ...):
  * normalized profile sigma_i/sigma_1 vs i/n, pooled over layers of a model;
  * ks_profile(A,B) = two-sample KS between the pooled normalized singular values of models A and B;
    null = KS(trained A, N1 draw of B's matrices) and KS(N1 draw of A, trained B) (trained-vs-random
    in normalized coordinates), so z says how far cross-model KS sits from trained-vs-random KS and
    ks_null_median is the median trained-vs-random KS;
  * alpha_slope = OLS slope of the per-model median alpha vs log10(parameter count), with the
    N1 per-model median alpha regressed the same way as the null (from h01.csv, null_mean of N1 rows)
    and alpha_r the Pearson r.
Figure: profiles overlaid across models per component type.

Run:  .venv/bin/python analysis/h08_universality.py [--models ...] [--h01 results/tables/h01.csv] [--out CSV]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from analysis.h01_spectra import split_name  # noqa: E402
from handi.analysis import plots, registry, results, stats  # noqa: E402

HYP = "H08"
RAW = C.REPO_ROOT / "results" / "raw" / "h01"
N_GRID = 200
PROFILE_TYPES = ("embed.tok", "attn.q.w", "attn.k.w", "attn.v.w", "attn.o.w", "mlp.up.w", "mlp.down.w")


def load_model(model: str) -> dict[str, list[dict]]:
    """component type -> list of {s, s_n1, shape} over its matrices."""
    out = defaultdict(list)
    d = RAW / model
    if not d.is_dir():
        raise FileNotFoundError(f"{d}: run analysis/h01_spectra.py first")
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        layer, comp = split_name(f.stem)
        out[comp].append({"s": z["s_trained"], "s_n1": z["s_n1"], "shape": tuple(z["shape"]), "layer": layer})
    return out


def normalized(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return s / s[0]


def pooled(entries: list[dict], key: str = "s", draw: int | None = None) -> np.ndarray:
    if draw is None:
        return np.concatenate([normalized(e[key]) for e in entries])
    return np.concatenate([normalized(e[key][draw % e[key].shape[0]]) for e in entries])


def mean_profile(entries: list[dict]) -> np.ndarray:
    grid = np.linspace(0, 1, N_GRID)
    prof = []
    for e in entries:
        s = normalized(e["s"])
        x = np.arange(s.size) / max(s.size - 1, 1)
        prof.append(np.interp(grid, x, s))
    return np.mean(prof, axis=0)


def read_h01(path: Path) -> dict:
    """(model, component) -> {'alpha': median trained alpha over layers, 'alpha_n1': median N1 null mean,
    'params': total elements of that component's matrices}; plus model -> total 2-D params."""
    per = defaultdict(lambda: {"alpha": [], "alpha_n1": [], "params": 0})
    total = defaultdict(int)
    seen = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["null_kind"] != "N1" or r["layer"] == "ALL":
                continue
            key = (r["model"], r["layer"], r["component"])
            if key not in seen:
                seen.add(key)
                n = int(r["n_rows"]) * int(r["n_cols"])
                per[(r["model"], r["component"])]["params"] += n
                total[r["model"]] += n
            if r["metric"] == "alpha":
                per[(r["model"], r["component"])]["alpha"].append(float(r["value"]))
                per[(r["model"], r["component"])]["alpha_n1"].append(float(r["null_mean"]))
    return per, total


def slope(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    b = np.polyfit(x[ok], y[ok], 1)[0]
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return float(b), float(r)


def main() -> None:
    p = C.parser(HYP)
    p.add_argument("--h01", default=str(C.TABLES / "h01.csv"))
    args = p.parse_args()
    out = C.prepare_out(args)
    models = [m for m in args.models if (RAW / m).is_dir()]
    data = {m: load_model(m) for m in models}
    per, total = read_h01(Path(args.h01))
    types = sorted({c for m in models for c in data[m]})
    rng = np.random.default_rng(args.seed)
    with results.ResultsWriter(out) as w:
        for comp in types:
            have = [m for m in models if comp in data[m]]
            # pairwise profile KS
            for a, b in combinations(have, 2):
                ta, tb = pooled(data[a][comp]), pooled(data[b][comp])
                v = stats.ks_2samp(ta, tb)[0]
                nd = min(min(e["s_n1"].shape[0] for e in data[a][comp]), min(e["s_n1"].shape[0] for e in data[b][comp]))
                null = [stats.ks_2samp(ta, pooled(data[b][comp], "s_n1", k))[0] for k in range(nd)]
                null += [stats.ks_2samp(pooled(data[a][comp], "s_n1", k), tb)[0] for k in range(nd)]
                sm = stats.null_summary(null)
                w.row(hypothesis=HYP, model=a, family=registry.family(a), layer="ALL", component=comp,
                      metric=f"ks_profile({a},{b})", value=v, null_kind="N1", null_n=sm["n"], null_mean=sm["mean"],
                      null_sd=sm["sd"], z=stats.zscore(v, null), ks=v, ks_null_median=C.finite_median(null),
                      src_dtype=C.SRC_DTYPE[a], seed=args.seed)
            # alpha vs log10(params)
            xs = [np.log10(total[m]) for m in have if (m, comp) in per]
            ys = [C.finite_median(per[(m, comp)]["alpha"]) for m in have if (m, comp) in per]
            yn = [C.finite_median(per[(m, comp)]["alpha_n1"]) for m in have if (m, comp) in per]
            b, r = slope(xs, ys)
            bn, rn = slope(xs, yn)
            w.row(hypothesis=HYP, model="ALL", family="", layer="ALL", component=comp, metric="alpha_slope",
                  value=b, null_kind="N1", null_n=1, null_mean=bn, null_sd=float("nan"), z=float("nan"),
                  seed=args.seed)
            w.row(hypothesis=HYP, model="ALL", family="", layer="ALL", component=comp, metric="alpha_r",
                  value=r, null_kind="N1", null_n=1, null_mean=rn, null_sd=float("nan"), z=float("nan"),
                  seed=args.seed)
            print(f"{comp:18s} models={len(have)} alpha_slope={b:+.3f} (r={r:+.2f}; N1 slope {bn:+.3f})")
    if not args.no_figures:
        figure(models, data)
    print(f"wrote {out}")


def figure(models, data) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = plots.style()
    out_dir = C.FIGURES / "h08"
    out_dir.mkdir(parents=True, exist_ok=True)
    order = [m for m in registry.MODELS if m in models]     # fixed hue per model, registry order
    color = {m: pal[i] for i, m in enumerate(order)}
    types = [t for t in PROFILE_TYPES if any(t in data[m] for m in order)]
    ncol = 4
    nrow = int(np.ceil(len(types) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    grid = np.linspace(0, 1, N_GRID)
    for ax, comp in zip(axes.ravel(), types):
        for m in order:
            if comp not in data[m]:
                continue
            ax.plot(grid, mean_profile(data[m][comp]), color=color[m], lw=2, label=m)
        n1 = [{"s": e["s_n1"][0]} for m in order if comp in data[m] for e in data[m][comp]]
        ax.plot(grid, mean_profile(n1), color=plots.INK_SECONDARY, lw=1.2, label="N1 (all models)")
        ax.set_yscale("log")
        ax.set_title(comp)
        ax.set_xlabel("i / n")
        ax.set_ylabel("sigma_i / sigma_1")
    for ax in axes.ravel()[len(types):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=3)
    fig.suptitle("normalized singular-value profiles across model size (mean over layers)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_dir / "profiles.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
