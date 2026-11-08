"""Steering sweep for the hearsay direction (decided 2026-11-08).

Directions (Qwen3.5-0.8B, residual at the answer position):
  H  hearsay mean-difference direction, two-fold (an item is steered with the direction fit on the other fold)
  O  overruling mean-difference direction (fit on all overruling items; it reads hearsay at 0.62 in hearsay_direction.py)
  HO the two added, each scaled by the sd of its projection on hearsay items
Sweep: every layer x position mode {last, all} x gamma in {-4, -2, +2, +4} (x sd), on the 94 hearsay items.
Side effects: at the best HO and the best H setting (by hearsay accuracy), the fraction of changed answers on the
nine other tasks. Rows: results/tables/hearsay_steer.csv (hypothesis R3S).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import evaluate as EV  # noqa: E402
from analysis.run_rung3 import collect  # noqa: E402
from analysis.hearsay_direction import task_data, mean_diff  # noqa: E402

DEVICE = EV.DEVICE
GAMMAS = (-4.0, -2.0, 2.0, 4.0)


class Steer:
    """Adds self.vec to the residual at layer `layer`: last position only, or every position."""

    def __init__(self, model, layer: int, all_pos: bool):
        self.vec, self.all_pos = None, all_pos
        self.h = model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.vec is None:
            return out
        hs = (out[0] if isinstance(out, tuple) else out).clone()
        if self.all_pos:
            hs += self.vec
        else:
            hs[:, -1] += self.vec
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self.h.remove()


@torch.no_grad()
def answers(model, tok, prompts, first_tok, steer=None, vecs=None, max_len=1024):
    """vecs: per-item steering vector [n, d] (numpy) or None."""
    out = []
    for i, p in enumerate(prompts):
        if steer is not None:
            steer.vec = None if vecs is None else torch.as_tensor(vecs[i], dtype=torch.float32, device=DEVICE)
        ids = tok(p, add_special_tokens=False)["input_ids"][-max_len:]
        x = torch.tensor([ids], device=DEVICE)
        hs = model.model(x).last_hidden_state[:, -1]
        lg = model.lm_head(hs)[0]
        out.append(int(lg[first_tok[1]] > lg[first_tok[0]]))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--task", default="hearsay")
    ap.add_argument("--second", default="overruling")
    ap.add_argument("--others", nargs="*", default=[t for t in EV.LEGALBENCH_TASKS if t != "hearsay"])
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results/tables/hearsay_steer.csv"))
    a = ap.parse_args()
    model, tok = EV.load_trained(a.model)
    n_layers = len(model.model.layers)
    layers = a.layers or list(range(n_layers))

    t0 = time.time()
    prompts, gold, labels, ft = task_data(tok, a.task)
    _, _, RES = collect(model, tok, prompts)                       # [n, L, d]
    n = len(gold)
    p2, g2, _, _ = task_data(tok, a.second)
    _, _, RES2 = collect(model, tok, p2)
    print(f"collected {a.task} n={n}, {a.second} n={len(g2)} ({time.time() - t0:.0f}s)", flush=True)
    fold = np.arange(n) % 2

    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3S", model=a.model, family="qwen3_5", src_dtype="BF16", seed=0, n_rows=n)
        base_ans = answers(model, tok, prompts, ft)
        base_acc = float((base_ans == gold).mean())
        w.row(metric="unsteered_frac_yes", layer="ALL", component=a.task, value=float(base_ans.mean()), **base)
        w.row(metric="unsteered_acc", layer="ALL", component=a.task, value=base_acc, **base)
        print(f"unsteered: frac Yes={base_ans.mean():.3f} acc={base_acc:.3f}", flush=True)

        best = {}                                                  # dirset -> (acc, layer, mode, gamma, frac_yes)
        for L in layers:
            X = RES[:, L].astype(np.float64)
            # H: two-fold direction and scale (fit on the other fold); O: all overruling items
            wh = {f: mean_diff(X[fold != f], gold[fold != f]) for f in (0, 1)}
            sh = {f: float((X[fold != f] @ wh[f]).std()) for f in (0, 1)}
            wo = mean_diff(RES2[:, L].astype(np.float64), g2)
            so = float((X @ wo).std())
            dirs = {
                "H": np.stack([sh[fold[i]] * wh[fold[i]] for i in range(n)]),
                "O": np.tile(so * wo, (n, 1)),
            }
            dirs["HO"] = dirs["H"] + dirs["O"]
            cos = float(wh[0] @ wo)
            w.row(metric="cos_H_O", layer=L, component="H|O", value=cos, **base)
            for mode in ("last", "all"):
                steer = Steer(model, L, all_pos=(mode == "all"))
                line = []
                for ds, D in dirs.items():
                    for g in GAMMAS:
                        ans = answers(model, tok, prompts, ft, steer, g * D)
                        fy, acc = float(ans.mean()), float((ans == gold).mean())
                        w.row(metric="steer_frac_yes", layer=L, component=f"{ds}|{mode}|gamma={g:+.0f}", value=fy, ks=g, **base)
                        w.row(metric="steer_acc", layer=L, component=f"{ds}|{mode}|gamma={g:+.0f}", value=acc, ks=g, **base)
                        line.append(f"{ds}{g:+.0f}:{fy:.2f}/{acc:.2f}")
                        key = (ds, mode)
                        if key not in best or acc > best[key][0]:
                            best[key] = (acc, L, mode, g, fy)
                steer.vec = None; steer.remove()
                print(f"L={L:2d} {mode:4s} cos(H,O)={cos:+.2f} | " + " ".join(line) + f" ({time.time() - t0:.0f}s)", flush=True)

        print("best (frac Yes / acc) per direction set and position mode:", flush=True)
        for key, (acc, L, mode, g, fy) in sorted(best.items()):
            w.row(metric="best_steer_acc", layer=L, component=f"{key[0]}|{mode}|gamma={g:+.0f}", value=acc, ks=g, **base)
            print(f"  {key[0]:2s} {mode:4s}: layer {L} gamma {g:+.0f}: frac Yes={fy:.3f} acc={acc:.3f} (unsteered {base_acc:.3f})", flush=True)

        # side effects on the other tasks at the best H and best HO setting (over both position modes)
        for ds in ("H", "HO"):
            acc, L, mode, g, fy = max((best[(ds, m)] for m in ("last", "all")), key=lambda r: r[0])
            X = RES[:, L].astype(np.float64)
            wh_all = mean_diff(X, gold); sh_all = float((X @ wh_all).std())
            vec = g * sh_all * wh_all
            if ds == "HO":
                wo = mean_diff(RES2[:, L].astype(np.float64), g2); vec = vec + g * float((X @ wo).std()) * wo
            steer = Steer(model, L, all_pos=(mode == "all"))
            changed = {}
            for task in a.others:
                po, go, lo, fo = task_data(tok, task)
                b0 = answers(model, tok, po, fo, steer, None)
                b1 = answers(model, tok, po, fo, steer, np.tile(vec, (len(po), 1)))
                changed[task] = float((b0 != b1).mean())
                acc0, acc1 = float((b0 == go).mean()), float((b1 == go).mean())
                w.row(metric="side_effect_frac_changed", layer=L, component=f"{ds}|{mode}|gamma={g:+.0f}|{task}", value=changed[task], ks=acc1 - acc0, **base)
            steer.vec = None; steer.remove()
            print(f"side effects of best {ds} (layer {L}, {mode}, gamma {g:+.0f}): " + " ".join(f"{t}={v:.2f}" for t, v in changed.items()), flush=True)


if __name__ == "__main__":
    main()
