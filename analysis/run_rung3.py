"""Rung 3: closed-form weight edits to the trained Qwen3.5-0.8B for LegalBench (decided 2026-11-07). No gradient step anywhere.

Conditions, each a weight edit of the unembedding rows of the label tokens (on an untied copy of lm_head):
  E0  trained model as is (first-label-token scoring, same argmax as the M4 scorer for single-token labels)
  E1  contextual calibration: row_t += delta_t * mu_hat, so every label's mean log-prob on the fit half is equal
  E2  E1 + linear readout: ridge regression from the final-norm hidden state at the answer position to the label
      (+1 / -1), fit on the fit half, written into the label rows (row_A += alpha w, row_B -= alpha w, constant via mu_hat)
Diagnostic (not a weight edit): per-layer ridge probe accuracy on the residual stream at the answer position,
to locate where (if anywhere) the label is linearly decodable.

Protocol: the M4 test sample of each task (<= 500 items, seed 0) is split in two folds by item parity; edits are
fit on one fold and scored on the other, both ways; the reported accuracy is over all items, each scored by the
edit fit without it. Builder defaults (docs/decisions.md, Rung 3): ridge lambda = n_fit on standardised features;
alpha such that sd(alpha * readout) on the fit fold = 2 x sd(original label-logit gap) on the fit fold.

Run:  .venv/bin/python analysis/run_rung3.py [--model qwen3.5-0.8b] [--tasks ...] [--out results/tables/rung3.csv]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handi.analysis.results import ResultsWriter  # noqa: E402
from handi.authoring import evaluate as EV  # noqa: E402

DEVICE = EV.DEVICE
SRC = {"qwen3_5": "BF16", "gpt_neox": "F16", "gpt2": "F32", "gpt_neo": "F32"}


@torch.no_grad()
def collect(model, tok, prompts: list[str], max_len: int = 1024):
    """Per prompt: final-norm hidden state at the last position [d], full last-position logits [V] (CPU),
    and the residual stream at the last position for every layer [L+1, d]."""
    H, LOG, RES = [], [], []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"][-max_len:]
        x = torch.tensor([ids], device=DEVICE)
        out = model.model(x, output_hidden_states=True)       # logits only at the last position (long prompts)
        hs = out.hidden_states
        H.append(hs[-1][0, -1].float().cpu())
        LOG.append(model.lm_head(hs[-1][:, -1])[0].float().cpu())
        RES.append(torch.stack([h[0, -1].float().cpu() for h in hs[:-1]]))
    return torch.stack(H).numpy().astype(np.float64), torch.stack(LOG).numpy().astype(np.float64), torch.stack(RES).numpy()


def ridge(X: np.ndarray, y: np.ndarray, lam: float):
    """Standardised ridge with intercept; returns (w in original feature units, b)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w_z = np.linalg.solve(A, Z.T @ (y - y.mean()))
    w = w_z / sd
    b = y.mean() - mu @ w
    return w, b


def wilson_half(acc: float, n: int) -> float:
    return 1.96 * math.sqrt(max(acc * (1 - acc), 1e-12) / n) if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--tasks", nargs="*", default=EV.LEGALBENCH_TASKS)
    ap.add_argument("--out", default=str(ROOT / "results/tables/rung3.csv"))
    ap.add_argument("--verify-task", default="hearsay", help="apply the E2 edit to the real lm_head and rescore with the M4 scorer")
    a = ap.parse_args()

    model, tok = EV.load_trained(a.model)
    fam = "qwen3_5" if "qwen" in a.model else model.config.model_type
    # untie: the edits touch the output rows only (decisions.md, Rung 3)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    W0 = model.lm_head.weight.detach().clone()
    d = W0.shape[1]

    with ResultsWriter(a.out) as w:
        base = dict(hypothesis="R3", model=a.model, family=fam, src_dtype=SRC.get(fam, "F32"), seed=0)
        for task in a.tasks:
            t0 = time.time()
            template = (ROOT / "tests/fixtures/legalbench_prompts" / f"{task}.txt").read_text()
            rows = EV.legalbench_items(task, 500)
            labels = sorted({r["answer"] for r in rows})
            if len(labels) != 2:
                print(f"[{task}] {len(labels)} labels; Rung 3 edits are two-label only, skipped", flush=True)
                continue
            first_tok = [tok(" " + lab, add_special_tokens=False)["input_ids"][0] for lab in labels]
            multi = any(len(tok(" " + lab, add_special_tokens=False)["input_ids"]) > 1 for lab in labels)
            prompts = [EV.fill_prompt(template, r) for r in rows]
            gold = np.array([labels.index(r["answer"]) for r in rows])
            n = len(rows)
            majority = max(Counter(gold.tolist()).values()) / n
            H, LOG, RES = collect(model, tok, prompts)
            lab_logits = LOG[:, first_tok]                       # [n, 2]
            gap0 = lab_logits[:, 1] - lab_logits[:, 0]           # > 0 predicts label 1
            mu_hat = H.mean(0); mu_hat /= np.linalg.norm(mu_hat)
            proj = H @ mu_hat                                    # ~ constant across items (post-norm)

            pred = {k: np.zeros(n, dtype=int) for k in ("E0", "E1", "E2")}
            probe_pred = np.zeros((RES.shape[1], n), dtype=int)
            edits = {}
            for fold in (0, 1):
                fit, ev = np.arange(n) % 2 != fold, np.arange(n) % 2 == fold
                # E0
                pred["E0"][ev] = (gap0[ev] > 0).astype(int)
                # E1 calibration: equalise mean log-prob of the two labels on the fit fold (shared normaliser: use logits)
                lp = LOG[fit] - np.logaddexp.reduce(LOG[fit], axis=1, keepdims=True)
                mean_lp = lp[:, first_tok].mean(0)
                delta = -(mean_lp - mean_lp.mean())                 # per label logit shift
                shift = delta / proj.mean()                          # row_t += shift_t * mu_hat  ->  logit_t += shift_t * proj
                gap1 = gap0 + (shift[1] - shift[0]) * proj
                pred["E1"][ev] = (gap1[ev] > 0).astype(int)
                # E2 readout
                y = 2.0 * gold[fit] - 1.0
                wv, b = ridge(H[fit], y, lam=float(fit.sum()))
                yhat_fit = H[fit] @ wv + b
                alpha = 2.0 * gap1[fit].std() / (yhat_fit.std() + 1e-9)
                gap2 = gap1 + alpha * (H @ wv + b)
                pred["E2"][ev] = (gap2[ev] > 0).astype(int)
                edits[fold] = dict(shift=shift, wv=wv, b=b, alpha=alpha)
                # per-layer probe (diagnostic)
                for l in range(RES.shape[1]):
                    pw, pb = ridge(RES[fit, l].astype(np.float64), y, lam=float(fit.sum()))
                    probe_pred[l, ev] = ((RES[ev, l] @ pw + pb) > 0).astype(int)

            common = dict(layer="ALL", n_rows=n, n_cols=d, **base)
            line = []
            for k in ("E0", "E1", "E2"):
                acc = float((pred[k] == gold).mean())
                w.row(metric=f"m4_{task}", component=k, value=acc, null_kind="normal95", null_n=n,
                      null_mean=acc - wilson_half(acc, n), null_sd=acc + wilson_half(acc, n), ks=majority, **common)
                line.append(f"{k}={acc:.3f}")
            best_l, best_acc = -1, -1.0
            for l in range(RES.shape[1]):
                acc = float((probe_pred[l] == gold).mean())
                w.row(metric=f"probe_{task}", component=f"probe:{l}", layer=l, value=acc, null_kind="normal95", null_n=n,
                      null_mean=acc - wilson_half(acc, n), null_sd=acc + wilson_half(acc, n), ks=majority, n_rows=n, n_cols=d, **base)
                if acc > best_acc:
                    best_l, best_acc = l, acc
            w.row(metric=f"probe_best_{task}", component=f"probe:{best_l}", layer=best_l, value=best_acc, ks=majority,
                  n_rows=n, n_cols=d, **base)
            print(f"[{task}] n={n} majority={majority:.3f} multi_token_labels={multi} | {' '.join(line)} | "
                  f"best probe layer {best_l} acc={best_acc:.3f} | final-state probe acc={float((probe_pred[-1] == gold).mean()):.3f} | {time.time() - t0:.0f}s", flush=True)

            # verification: apply the fold-0 E2 edit to the real lm_head and rescore the fold-0 eval half with the M4 scorer
            if task == a.verify_task:
                e = edits[0]
                Wn = W0.clone()
                for j, t in enumerate(first_tok):
                    Wn[t] += torch.as_tensor(e["shift"][j] * mu_hat, dtype=Wn.dtype, device=Wn.device)
                sgn = [-1.0, 1.0]
                for j, t in enumerate(first_tok):
                    Wn[t] += torch.as_tensor(sgn[j] * e["alpha"] * e["wv"] + sgn[j] * e["alpha"] * e["b"] / proj.mean() * mu_hat,
                                             dtype=Wn.dtype, device=Wn.device)
                model.lm_head.weight.data.copy_(Wn)
                ev = np.arange(n) % 2 == 0
                sc = np.stack([EV._label_logprob(model, tok, [p for p, m in zip(prompts, ev) if m], lab) for lab in labels], 1)
                acc_real = float((sc.argmax(1) == gold[ev]).mean())
                acc_off = float((pred["E2"][ev] == gold[ev]).mean())
                model.lm_head.weight.data.copy_(W0)
                w.row(metric=f"verify_{task}_E2_real_model", component="E2", value=acc_real, ks=acc_off, **common)
                print(f"  verify [{task}] fold-0 eval half: E2 via real edited lm_head + M4 scorer = {acc_real:.3f}; offline = {acc_off:.3f}", flush=True)


if __name__ == "__main__":
    main()
