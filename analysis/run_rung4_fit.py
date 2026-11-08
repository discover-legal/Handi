"""Rung 4, condition A10: the A9 chord layer with its OUTPUT columns fitted by iterative optimisation (decided 2026-11-08).

Everything else is the A9 closed-form build (detectors, previous-token head, direct path, codes), frozen. Free
parameters: the down-projection columns of the chord neurons (layers 1..L-1, rows of the bigram subspace only).
Objective: next-token cross-entropy of the full model on the WikiText TRAIN split (the text the counts came from).
Model selection: validation split only (early stopping on validation CE, patience). Reporting: the standard M1
test slice (first 100k test tokens), M2 copy, written to results/tables/rung4.csv as component A10, alongside A9/A9C.

Run: .venv/bin/python analysis/run_rung4_fit.py --models pythia-70m pythia-160m --seeds 0 1 2
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

from handi.analysis.loader import load_canonical, n_layers  # noqa: E402
from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import bigram, chords  # noqa: E402
from handi.authoring.evaluate import cross_entropy, evaluate_all, load_authored  # noqa: E402
from analysis.run_rung2 import rows_from  # noqa: E402

DEVICE = bigram.DEVICE
CTX, BATCH = 512, 8
VAL_TOKENS = 50_000          # validation window: tokens 20k..70k of the validation split (20k = the A9 scale slice)
EVAL_EVERY, PATIENCE, MAX_STEPS = 100, 6, 4000


def fit(model, w: dict, cfg: dict, fam: str, name: str, lr: float, log) -> dict:
    V, d = w["embed.tok"].shape
    L = n_layers(cfg, fam)
    H = cfg["num_attention_heads"]; K = d // H; B = d - 2 * K
    train = bigram._tokens(name, "train", bigram.TRAIN_CHARS).astype(np.int64)
    val = bigram._tokens(name, "validation", 400_000)[chords.CALIB_TOKENS: chords.CALIB_TOKENS + VAL_TOKENS + 1].astype(np.int64)
    for p in model.parameters():
        p.requires_grad_(False)
    params, masks = [], []
    for l in range(1, L):
        nb = int((np.abs(w[f"layers.{l}.mlp.up.w"]).sum(1) > 0).sum())
        if nb == 0:
            continue
        p = model.gpt_neox.layers[l].mlp.dense_4h_to_h.weight            # [d, d_mlp]
        p.requires_grad_(True)
        m = torch.zeros_like(p); m[:B, :nb] = 1.0
        params.append(p); masks.append(m)
    n_free = int(sum(m.sum().item() for m in masks))
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(0)

    def val_ce():
        model.eval()
        with torch.no_grad():
            return cross_entropy(model, val, n_tokens=VAL_TOKENS)[0]

    best_ce, best_step, best_state, bad = val_ce(), 0, [p.detach().clone() for p in params], 0
    log(f"  start: val CE {best_ce:.4f}, {n_free} free weights in {len(params)} layers")
    curve = [(0, best_ce)]
    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        model.train()
        starts = rng.integers(0, len(train) - CTX - 1, size=BATCH)
        x = torch.as_tensor(np.stack([train[s:s + CTX + 1] for s in starts]), device=DEVICE)
        logits = model(x[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p, m in zip(params, masks):
            p.grad.mul_(m)
        opt.step()
        if step % EVAL_EVERY == 0:
            ce = val_ce()
            curve.append((step, ce))
            log(f"  step {step:5d} train {loss.item():.4f} val {ce:.4f} best {best_ce:.4f}@{best_step} ({time.time() - t0:.0f}s)")
            if ce < best_ce - 1e-4:
                best_ce, best_step, bad = ce, step, 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    with torch.no_grad():
        for p, s in zip(params, best_state):
            p.copy_(s)
        for p in params:
            p.requires_grad_(False)
    model.eval()
    return dict(a10_free_weights=n_free, a10_best_step=best_step, a10_val_ce_start=curve[0][1], a10_val_ce_best=best_ce,
                a10_steps_run=curve[-1][0], a10_lr=lr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["pythia-70m"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--which", nargs="*", default=["m1", "m2"])
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung4.csv"))
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    with ResultsWriter(a.out) as wr:
        for name in a.models:
            canon, cfg, fam = load_canonical(name)
            for seed in a.seeds:
                t0 = time.time()
                report = {}
                w = chords.a9_chords(canon, cfg, fam, seed, model=name, report=report)      # closed-form A9 = start point
                log(f"{name} A10 seed={seed}: A9 build {{scale={report.get('a9_scale')}, neurons={report.get('a9_neurons')}}}")
                model, tok = load_authored(name, w)
                report.update(fit(model, w, cfg, fam, name, a.lr, log))
                res = evaluate_all(model, tok, a.which, 100_000)
                rows_from(res, wr, hypothesis="H13", model=name, family=fam, layer="ALL", component="A10", seed=seed, src_dtype="F16")
                for k, v in report.items():
                    if k.startswith("a10_") or k in ("a9_scale", "a9_neurons", "a9_ref_trigram", "a9_ref_trigram_capped", "a9_ref_bigram"):
                        wr.row(hypothesis="H13", model=name, family=fam, layer="ALL", component="A10", metric=k, value=float(v), seed=seed, src_dtype="F16")
                m1 = {r["corpus"]: round(r["nats"], 3) for r in res.get("m1", [])}
                log(f"{name:16s} A10 seed={seed} m1={m1} m2={res.get('m2')} val {report['a10_val_ce_start']:.3f}->{report['a10_val_ce_best']:.3f} "
                    f"@step {report['a10_best_step']} ({time.time() - t0:.0f}s)")
                del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
