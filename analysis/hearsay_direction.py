"""Isolate the "hearsay" direction found by the Rung 3 probe in the trained Qwen3.5-0.8B (decided 2026-11-08).

Direction: at each layer, the unit mean-difference of the residual stream at the answer position between
hearsay items answered Yes and No (fit on one fold, tested on the other, both ways). Then, at the best layer L*:
  1. readout accuracy of the direction alone (two-fold), vs the per-layer ridge probe of Rung 3;
  2. specificity: cosine of the hearsay direction with the same-layer directions of the nine other tasks and
     with the unembedding difference U[Yes] - U[No]; readout accuracy of each other task's direction ON hearsay;
  3. causality: add gamma * s * w_hat to the residual at layer L* (last position only), gamma in a grid, s = the
     sd of the projection on hearsay items; record the fraction answered Yes and the accuracy on hearsay, and the
     fraction of answers that change on every other task (the same steer, same gamma); and the converse: steer
     hearsay with the other tasks' directions;
  4. reading: top / bottom tokens of the direction through the unembedding (logit lens), and the hearsay items
     with the largest and smallest projection.
Rows: results/tables/hearsay_direction.csv (hypothesis R3D). Log: results/tables/hearsay_direction.log.
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
from analysis.run_rung3 import collect, ridge  # noqa: E402

DEVICE = EV.DEVICE
GAMMAS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)


def task_data(tok, task):
    template = (ROOT / "tests/fixtures/legalbench_prompts" / f"{task}.txt").read_text()
    rows = EV.legalbench_items(task, 500)
    labels = sorted({r["answer"] for r in rows})
    prompts = [EV.fill_prompt(template, r) for r in rows]
    gold = np.array([labels.index(r["answer"]) for r in rows])
    first_tok = [tok(" " + lab, add_special_tokens=False)["input_ids"][0] for lab in labels]
    return prompts, gold, labels, first_tok


def mean_diff(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-9)


def twofold_dir_acc(X, y):
    n = len(y); pred = np.zeros(n, dtype=int)
    for fold in (0, 1):
        fit, ev = np.arange(n) % 2 != fold, np.arange(n) % 2 == fold
        w = mean_diff(X[fit], y[fit])
        p = X @ w
        thr = 0.5 * (p[fit][y[fit] == 1].mean() + p[fit][y[fit] == 0].mean())
        pred[ev] = (p[ev] > thr).astype(int)
    return float((pred == y).mean())


class Steer:
    def __init__(self, model, layer: int):
        self.vec = None
        self.h = model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.vec is None:
            return out
        if isinstance(out, tuple):
            hs = out[0].clone(); hs[:, -1] += self.vec; return (hs,) + tuple(out[1:])
        hs = out.clone(); hs[:, -1] += self.vec; return hs

    def remove(self):
        self.h.remove()


@torch.no_grad()
def answers(model, tok, prompts, first_tok, max_len=1024):
    out = []
    for p in prompts:
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
    ap.add_argument("--others", nargs="*", default=[t for t in EV.LEGALBENCH_TASKS if t != "hearsay"])
    ap.add_argument("--out", default=str(ROOT / "results/tables/hearsay_direction.csv"))
    a = ap.parse_args()
    model, tok = EV.load_trained(a.model)
    n_layers = len(model.model.layers)

    t0 = time.time()
    data = {}
    for task in [a.task] + a.others:
        prompts, gold, labels, first_tok = task_data(tok, task)
        H, LOG, RES = collect(model, tok, prompts)
        data[task] = dict(prompts=prompts, gold=gold, labels=labels, first_tok=first_tok, RES=RES, LOG=LOG)
        print(f"collected {task} n={len(gold)} ({time.time() - t0:.0f}s)", flush=True)

    T = data[a.task]
    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3D", model=a.model, family="qwen3_5", src_dtype="BF16", seed=0, n_rows=len(T["gold"]))
        # 1. per-layer accuracy of the mean-difference direction and of the ridge probe
        accs = []
        for l in range(RES_L := T["RES"].shape[1]):
            X = T["RES"][:, l].astype(np.float64)
            acc_dir = twofold_dir_acc(X, T["gold"])
            n = len(T["gold"]); pred = np.zeros(n, dtype=int)
            for fold in (0, 1):
                fit, ev = np.arange(n) % 2 != fold, np.arange(n) % 2 == fold
                pw, pb = ridge(X[fit], 2.0 * T["gold"][fit] - 1.0, lam=float(fit.sum()))
                pred[ev] = ((X[ev] @ pw + pb) > 0).astype(int)
            acc_ridge = float((pred == T["gold"]).mean())
            accs.append(acc_dir)
            w.row(metric="meandiff_acc", layer=l, component=a.task, value=acc_dir, **base)
            w.row(metric="ridge_acc", layer=l, component=a.task, value=acc_ridge, **base)
        Lstar = int(np.argmax(accs))
        print(f"[{a.task}] mean-difference direction accuracy by layer: " + " ".join(f"{l}:{v:.2f}" for l, v in enumerate(accs)), flush=True)
        print(f"[{a.task}] best layer L*={Lstar} acc={accs[Lstar]:.3f}", flush=True)
        w.row(metric="best_layer", layer=Lstar, component=a.task, value=accs[Lstar], **base)

        X = T["RES"][:, Lstar].astype(np.float64)
        w_hear = mean_diff(X, T["gold"])                         # full-sample direction for the causal / reading tests
        proj = X @ w_hear
        s = float(proj.std())

        # 2. specificity: cosines and cross-task readout
        U = model.lm_head.weight.detach().float().cpu().numpy()
        yes_no = U[T["first_tok"][1]] - U[T["first_tok"][0]]
        cos_yn = float(w_hear @ yes_no / np.linalg.norm(yes_no))
        w.row(metric="cos_with_unembed_yes_minus_no", layer=Lstar, component=a.task, value=cos_yn, **base)
        print(f"[{a.task}] cos(direction, U[Yes]-U[No]) = {cos_yn:+.3f}", flush=True)
        other_dirs = {}
        for task in a.others:
            Xo = data[task]["RES"][:, Lstar].astype(np.float64)
            wo = mean_diff(Xo, data[task]["gold"])
            other_dirs[task] = wo
            cos = float(w_hear @ wo)
            # other task's direction read on hearsay (threshold fit on hearsay fold, two-fold)
            n = len(T["gold"]); pred = np.zeros(n, dtype=int)
            p = X @ wo
            for fold in (0, 1):
                fit, ev = np.arange(n) % 2 != fold, np.arange(n) % 2 == fold
                thr = 0.5 * (p[fit][T["gold"][fit] == 1].mean() + p[fit][T["gold"][fit] == 0].mean())
                pred[ev] = (p[ev] > thr).astype(int)
            acc_cross = float((pred == T["gold"]).mean())
            w.row(metric="cos_with_other_task_direction", layer=Lstar, component=task, value=cos, **base)
            w.row(metric="other_direction_acc_on_hearsay", layer=Lstar, component=task, value=acc_cross, **base)
            print(f"[{a.task}] vs {task}: cos={cos:+.3f}; {task}'s direction reads hearsay at acc={acc_cross:.3f}", flush=True)

        # 3. causality: steer at L*, last position
        steer = Steer(model, Lstar)
        ft = T["first_tok"]
        base_ans = answers(model, tok, T["prompts"], ft)
        print(f"[{a.task}] unsteered: frac Yes={base_ans.mean():.3f} acc={(base_ans == T['gold']).mean():.3f}", flush=True)
        for g in GAMMAS:
            steer.vec = torch.as_tensor(g * s * w_hear, dtype=torch.float32, device=DEVICE)
            ans = answers(model, tok, T["prompts"], ft)
            frac_yes, acc = float(ans.mean()), float((ans == T["gold"]).mean())
            w.row(metric="steer_frac_yes", layer=Lstar, component=f"gamma={g:+.0f}", value=frac_yes, ks=g, **base)
            w.row(metric="steer_acc", layer=Lstar, component=f"gamma={g:+.0f}", value=acc, ks=g, **base)
            line = f"[{a.task}] steer gamma={g:+.0f}: frac Yes={frac_yes:.3f} acc={acc:.3f}"
            if g in (-2.0, 2.0):
                changed = {}
                for task in a.others:
                    D = data[task]
                    steer.vec = None
                    b0 = answers(model, tok, D["prompts"], D["first_tok"])
                    steer.vec = torch.as_tensor(g * s * w_hear, dtype=torch.float32, device=DEVICE)
                    b1 = answers(model, tok, D["prompts"], D["first_tok"])
                    changed[task] = float((b0 != b1).mean())
                    w.row(metric="steer_frac_changed_other_task", layer=Lstar, component=f"{task}|gamma={g:+.0f}", value=changed[task], ks=g, **base)
                line += " | other tasks changed: " + " ".join(f"{t}={v:.2f}" for t, v in changed.items())
            print(line, flush=True)
        # converse: other tasks' directions on hearsay at gamma = +-2
        for task in a.others:
            wo = other_dirs[task]; so = float((X @ wo).std())
            res = []
            for g in (-2.0, 2.0):
                steer.vec = torch.as_tensor(g * so * wo, dtype=torch.float32, device=DEVICE)
                ans = answers(model, tok, T["prompts"], ft)
                res.append(float(ans.mean()))
                w.row(metric="other_direction_steer_frac_yes_on_hearsay", layer=Lstar, component=f"{task}|gamma={g:+.0f}", value=float(ans.mean()), ks=g, **base)
            print(f"[{a.task}] steered by {task}'s direction: frac Yes at gamma -2 / +2 = {res[0]:.3f} / {res[1]:.3f}", flush=True)
        steer.vec = None; steer.remove()

        # 4. reading the direction
        gf = 1.0 + model.model.norm.weight.detach().float().cpu().numpy()
        lens = U @ (gf * w_hear)
        top, bot = np.argsort(lens)[::-1][:25], np.argsort(lens)[:25]
        print(f"[{a.task}] logit lens, tokens pushed UP by the direction: {[tok.decode([int(i)]) for i in top]}", flush=True)
        print(f"[{a.task}] logit lens, tokens pushed DOWN: {[tok.decode([int(i)]) for i in bot]}", flush=True)
        order = np.argsort(proj)
        rows = EV.legalbench_items(a.task, 500)
        print(f"[{a.task}] items with the LARGEST projection (direction says Yes):", flush=True)
        for i in order[::-1][:5]:
            print(f"   proj={proj[i]:+.2f} gold={T['labels'][T['gold'][i]]}: {rows[i]['text'][:160]}", flush=True)
        print(f"[{a.task}] items with the SMALLEST projection (direction says No):", flush=True)
        for i in order[:5]:
            print(f"   proj={proj[i]:+.2f} gold={T['labels'][T['gold'][i]]}: {rows[i]['text'][:160]}", flush=True)
        np.save(ROOT / "results/raw" / f"hearsay_direction_L{Lstar}.npy", w_hear.astype(np.float32))


if __name__ == "__main__":
    main()
