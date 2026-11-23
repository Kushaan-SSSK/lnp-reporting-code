"""Sensitivity checks requested in co-author review.

1. Encoding: one-hot encoding of lipid identity (fitted within each
   training fold; unseen lipids -> all zeros) in place of frequency encoding.
2. Ranges: target values written as ranges ("50-142") take the midpoint
   instead of the leading number.
3. Folds: how many publications and formulations each study-grouped fold
   holds out.

Each check reruns the five model families under study-grouped CV and
leave-one-study-out, and reports the best R2 next to the main analysis.

Output: results/extensions/sensitivity_results.json
Run:    python analysis/extensions/sensitivity.py
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import OUT, R, load, targets
from features import LIPID_COLS


class FoldSafeOneHot:
    """One-hot lipid identity, categories taken from the training fold only."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.cats_ = {c: sorted(X[c].astype(str).unique()) for c in self.columns}
        return self

    def transform(self, X):
        parts = []
        for c in self.columns:
            v = X[c].astype(str)
            parts.append(pd.DataFrame({f"{c}=={k}": (v == k).astype(int)
                                       for k in self.cats_[c]}, index=X.index))
        return pd.concat(parts, axis=1)


def best(raw, static, y, g):
    grp = {m: R.grouped_cv(raw, static, y, g, f)["r2"] for m, f in R.MODELS.items()}
    lo = {m: R.loso(raw, static, y, g, f)["r2"] for m, f in R.MODELS.items()}
    return {"grouped": grp, "loso": lo,
            "best_grouped": max(grp.values()), "best_loso": max(lo.values())}


def midpoint(v):
    """Midpoint for values written as ranges; otherwise the pipeline's parser."""
    from features import to_numeric_value
    if isinstance(v, str):
        m = re.match(r"\s*(\d+\.?\d*)\s*(?:-|–|to)\s*(\d+\.?\d*)", v.replace(",", ""))
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
    return to_numeric_value(v)


def main():
    df = load()
    out = {}
    original_encoder = R.FoldSafeFrequencyEncoder
    for target, label, raw, static, y, g, _ in targets(df):
        e = {"label": label, "n": int(len(y)), "n_publications": int(len(np.unique(g)))}
        # 3. fold composition (same splitter as the main analysis)
        folds = []
        for tr, te in GroupKFold(n_splits=10).split(static, y, g):
            folds.append({"test_publications": int(len(np.unique(g[te]))),
                          "test_formulations": int(len(te))})
        e["folds"] = folds
        # 1. one-hot encoding
        R.FoldSafeFrequencyEncoder = lambda cols: FoldSafeOneHot(cols)
        try:
            e["onehot"] = best(raw, static, y, g)
        finally:
            R.FoldSafeFrequencyEncoder = original_encoder
        e["frequency"] = best(raw, static, y, g)
        out[target] = e
        print(label, "one-hot", e["onehot"]["best_grouped"], e["onehot"]["best_loso"],
              "| frequency", e["frequency"]["best_grouped"], e["frequency"]["best_loso"],
              "| folds", [(f["test_publications"], f["test_formulations"]) for f in folds],
              flush=True)

    # 2. midpoint of ranges: rebuild targets with the alternative parser
    import features
    original = features.to_numeric_value
    import common
    n_ranges = {}
    for t in R.TARGETS:
        s = df[t].dropna().astype(str)
        n_ranges[t] = int(s.str.match(r"\s*\d+\.?\d*\s*(?:-|–|to)\s*\d").sum())
    common.to_numeric_value = midpoint
    try:
        for target, label, raw, static, y, g, _ in common.targets(df):
            out[target]["n_range_values"] = n_ranges[target]
            out[target]["midpoint"] = best(raw, static, y, g)
            out[target]["midpoint"]["icc1"] = R.icc1(y, g).get("icc1")
            out[target]["midpoint"]["n"] = int(len(y))
            print(label, "midpoint", out[target]["midpoint"]["best_grouped"],
                  out[target]["midpoint"]["best_loso"], "icc", out[target]["midpoint"]["icc1"],
                  "ranges", n_ranges[target], flush=True)
    finally:
        common.to_numeric_value = original
    (OUT / "sensitivity_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
