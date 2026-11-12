# Cookbook: pulling a classifier out of a model's weights

How to take a trained causal language model that answers a Yes/No classification task badly, find out
whether the answer is linearly readable from its internal state, and if it is, write a read-out into its own output
table so that the plain model answers well. No fine-tuning or gradient descent: a calibration and a ridge regression,
fitted in closed form on labelled examples (both known tools; see the report's "Prior work and credit"). This is the recipe behind the Rung 3 result (Qwen3.5-0.8B
on LegalBench: 0.57 to 0.78 mean over ten tasks; 0.94, 0.98 and 1.00 on never-seen items of overruling,
definition classification and governing law).

## What you need

- A causal LM you can load with Hugging Face `transformers` and whose weights you can edit (any size; the
  recipe was run on 0.8B on an 8 GB GPU).
- A few hundred labelled examples of one task, with two labels that are each one token after a space
  (" Yes" / " No"). Multi-token labels need the first-token variant noted at the end.
- A prompt template that ends exactly where the answer goes, e.g. `...\nLabel:`.

## The idea in one paragraph

A transformer turns the prompt into an internal state at the last position: a vector `h` of `d` numbers (1,024
for Qwen3.5-0.8B) after the final normalisation. The output table `U` (`lm_head.weight`, one row per token)
scores every token as `U[token] . h`; the answer is the higher of `U[Yes] . h` and `U[No] . h`. If a linear
read-out of `h` predicts the label well on held-out examples, the answer is linearly readable there (which is not
the same as the model understanding the task: compare with word counts, check 5 below). Writing that read-out
into the two rows, `U[Yes] += a w` and `U[No] -= a w`, makes the plain model use it. A calibration term removes
the model's built-in lean towards one label.

## Step 0: look before you edit

Run a linear probe first. If the probe on held-out examples is at the majority rate, the model does not know
the task in a linearly readable form at this position, and no output-row edit can help (citation prediction and
NY judicial ethics in LegalBench were like this). The probe below doubles as the read-out, so this costs
nothing extra.

## The recipe

```python
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "path/or/hub-id"                      # the model
LABELS = ["No", "Yes"]                       # label 1 is the "positive" side of the read-out
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).cuda().eval()
tok = AutoTokenizer.from_pretrained(PATH)
lab = [tok(" " + l, add_special_tokens=False)["input_ids"][0] for l in LABELS]   # one token each
assert all(len(tok(" " + l, add_special_tokens=False)["input_ids"]) == 1 for l in LABELS)

@torch.no_grad()
def collect(prompts, max_len=1024):
    """Final-norm state h [n, d] and full logits [n, V] at the last position of each prompt."""
    H, LOG = [], []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"][-max_len:]
        hs = model.model(torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states[-1][:, -1]
        H.append(hs[0].float().cpu()); LOG.append(model.lm_head(hs)[0].float().cpu())
    return torch.stack(H).double().numpy(), torch.stack(LOG).double().numpy()

def ridge(X, y, lam):
    """Standardised ridge with intercept; returns w (in raw feature units) and b."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w_z = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    w = w_z / sd
    return w, y.mean() - mu @ w

def fit_edit(H, LOG, gold):
    """gold in {0, 1}. Returns {token_id: delta_row} for the two label rows."""
    n = len(gold)
    # 1. calibration: equalise the labels' mean log-probability by shifting each row along the mean state
    mu_hat = H.mean(0); mu_hat /= np.linalg.norm(mu_hat)
    proj = H @ mu_hat                                     # nearly constant across examples (post-norm)
    lp = LOG - np.logaddexp.reduce(LOG, axis=1, keepdims=True)
    mean_lp = lp[:, lab].mean(0)
    shift = -(mean_lp - mean_lp.mean()) / proj.mean()     # logit_t += shift_t * proj
    gap1 = (LOG[:, lab[1]] - LOG[:, lab[0]]) + (shift[1] - shift[0]) * proj
    # 2. read-out: ridge from h to +-1, scaled so its spread is twice the calibrated logit gap's spread
    w, b = ridge(H, 2.0 * gold - 1.0, lam=float(n))
    alpha = 2.0 * gap1.std() / ((H @ w + b).std() + 1e-9)
    rows = {}
    for j, t in enumerate(lab):
        sgn = (-1.0, 1.0)[j]
        rows[t] = shift[j] * mu_hat + sgn * alpha * w + sgn * alpha * b / proj.mean() * mu_hat
    return rows

def predict(LOG, H, rows=None):
    L = LOG.copy()
    if rows:
        for t, dr in rows.items():
            L[:, t] += H @ dr
    return (L[:, lab[1]] > L[:, lab[0]]).astype(int)

# ---- data: prompts that end where the answer goes, gold in {0, 1} (index into LABELS)
prompts, gold = ..., np.array([...])
H, LOG = collect(prompts)
n = len(gold); fold = np.arange(n) % 2

# ---- step 0 + honest test: fit on one half, score the other, both ways
pred0, pred2 = np.zeros(n, int), np.zeros(n, int)
for f in (0, 1):
    fit, ev = fold != f, fold == f
    rows = fit_edit(H[fit], LOG[fit], gold[fit])
    pred0[ev] = predict(LOG[ev], H[ev])
    pred2[ev] = predict(LOG[ev], H[ev], rows)
print("majority", max(gold.mean(), 1 - gold.mean()), "as shipped", (pred0 == gold).mean(), "edited", (pred2 == gold).mean())
# if "edited" is at majority, stop: the answer is not linearly readable here.

# ---- write the edit into the model (fit on everything now that it is validated)
rows = fit_edit(H, LOG, gold)
if getattr(model.config, "tie_word_embeddings", False):           # untie, or the edit also changes the input embedding
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
with torch.no_grad():
    for t, dr in rows.items():
        model.lm_head.weight[t] += torch.as_tensor(dr, dtype=model.lm_head.weight.dtype, device=model.lm_head.weight.device)
model.save_pretrained("edited-model"); tok.save_pretrained("edited-model")
```

## Checks that matter

1. **Held-out first.** Never report the in-sample number. In-sample, every task scored 0.89 to 1.00; held-out
   is the real figure.
2. **Verify on the live model.** After writing the rows, re-run your ordinary scorer on the held-out half. In our
   check on hearsay the live model (bfloat16) scored 0.638 and the offline arithmetic (float64) 0.617, one
   example of 47 apart; if they differ by more than a close call or two, the hidden state
   you collected is not the one `lm_head` sees (check that `hidden_states[-1]` is post-norm for your
   architecture).
3. **One task per label pair.** The edit lives in the " Yes" and " No" rows, which every Yes/No prompt reads.
   It will change the model's Yes/No behaviour on other tasks (2% to 97% of answers in our test), and two
   tasks' edits summed into the same rows mostly cancel. For several tasks, give each its own label tokens
   (e.g. " A"/" B" vs " C"/" D") or keep separate edited copies.
4. **It adds no knowledge.** If the probe cannot read the answer, nothing here will create it.
5. **Check what the read-out is reading.** A classifier can score well by spotting a stock phrase. Test it:
   delete or paraphrase the phrases you suspect in the held-out items and rescore. On governing-law clauses
   the edited Qwen kept 0.997 after "governed by the laws of" was deleted from every clause (the model as
   shipped fell to a coin flip), and 0.97 on definitions after "means" / "is defined as" went. That rules out
   the one stock phrase, not everything else in the words: the same formula on plain word counts scores 0.997 on
   governing law, so its score says nothing about what the model knows. Run the word-count control too (and,
   if you can, the same recipe on a random-weight copy of the model: it reached 0.83 to 0.96 on our three tasks
   with a few hundred labels). This check is cheap and belongs in the write-up.
   Script: `analysis/run_rung3_cue.py`.

## Variants

- **Multi-token labels** (e.g. " Relevant" vs " Irrelevant" if they tokenise into several pieces): use the
  first token of each label for the rows and score by first token; check the first tokens differ.
- **More than two labels:** fit one ridge per label (one-vs-rest), add `alpha_k w_k` to row k, and calibrate
  all rows to equal mean log-probability.
- **Deeper edits:** the probe is usually better a few layers before the end (0.84 mean vs 0.78 for the
  output edit in our runs). Reading from an earlier layer needs a path from that layer to the output, which
  this recipe does not build.
- **Choosing the strength `alpha`:** `2 x` the calibrated gap's spread worked for every task we tried; it is
  a default, not a tuned value. A grid on the fit half is a reasonable refinement.
