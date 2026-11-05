"""H17: the direct path (embedding -> final norm -> unembedding) against corpus bigrams (hypotheses.md section 7).

S1  direct-path cross-entropy on WikiText-103 test bigram pairs at the temperature fitted on a disjoint train
    slice; references uniform / add-0.5 unigram / interpolated bigram; null N7 = token-assignment shuffle of
    the unembedding rows (embedding rows where tied).
S2  Spearman(log unigram count, ||e_t||) and (log count, ||u_t||); null N7.

Run:  .venv/bin/python analysis/h17_direct_path.py [--models ...] [--nulls N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import _common as C  # noqa: E402
from handi.analysis import registry, results, stats  # noqa: E402
from handi.analysis.loader import load_canonical  # noqa: E402
from handi.authoring import bigram  # noqa: E402

HYP = "H17"


def main() -> None:
    args = C.parser(HYP).parse_args()
    out = C.prepare_out(args)
    with results.ResultsWriter(out) as w:
        for name in args.models:
            t0 = time.time()
            canon, cfg, fam = load_canonical(name)
            V = canon["embed.tok"].shape[0]
            tied = "unembed.w" not in canon
            c = bigram.counts(name, V)
            refs = bigram.references(c, V)
            n_null = C.n_nulls(args, name)
            rng = np.random.default_rng(args.seed)
            base = dict(hypothesis=HYP, model=name, family=fam, src_dtype=C.SRC_DTYPE[name], seed=args.seed, layer="ALL",
                        n_rows=V, n_cols=canon["embed.tok"].shape[1])
            T, ce = bigram.direct_path_ce(canon, fam, c["calib"], c["test"])
            null = []
            for i in range(n_null):
                perm = rng.permutation(V)
                if tied:
                    null.append(bigram.direct_path_ce(canon, fam, c["calib"], c["test"], E=canon["embed.tok"][perm], U=canon["embed.tok"][perm])[1])
                else:
                    null.append(bigram.direct_path_ce(canon, fam, c["calib"], c["test"], U=canon["unembed.w"][perm])[1])
            ns = stats.null_summary(null)
            comp = "direct_path:tied" if tied else "direct_path"
            w.row(component=comp, metric="s1_ce", value=ce, null_kind="N7", null_n=ns["n"], null_mean=ns["mean"], null_sd=ns["sd"],
                  z=stats.zscore(ce, null), ks=T, **base)      # ks column = fitted temperature
            for k, v in refs.items():
                w.row(component=comp, metric=f"ref_{k}", value=v, **base)
            w.row(component=comp, metric="s1_ce_minus_unigram", value=ce - refs["unigram"], **base)
            # S2 frequency vs row norms
            logc = np.log(c["uni"] + 0.5)
            for mat, key in (("embed", "embed.tok"), ("unembed", "unembed.w")):
                if key not in canon:
                    continue
                norms = np.linalg.norm(canon[key], axis=1)
                rho = float(spearmanr(logc, norms).correlation)
                nulls2 = [float(spearmanr(logc, norms[rng.permutation(V)]).correlation) for _ in range(200)]
                ns2 = stats.null_summary(nulls2)
                w.row(component=mat, metric="s2_spearman_logcount_norm", value=rho, null_kind="N7", null_n=ns2["n"], null_mean=ns2["mean"],
                      null_sd=ns2["sd"], z=stats.zscore(rho, nulls2), **base)
            print(f"[{name}] direct path CE={ce:.3f} (T={T:.2f}) | N7 {ns['mean']:.3f}+-{ns['sd']:.3f} z={stats.zscore(ce, null):+.1f} | "
                  f"uniform {refs['uniform']:.2f} unigram {refs['unigram']:.2f} bigram {refs['bigram']:.2f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
