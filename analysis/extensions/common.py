"""Shared set-up for the extension analyses.

Every extension imports the main pipeline (run_real_analysis.py, one folder
up) and rebuilds each target exactly as it does, so preprocessing, folds and
exclusions are identical. Only the question being asked changes.

Works in both layouts:
  repository:      <root>/analysis/extensions/  with  <root>/data, <root>/results
  Zenodo deposit:  <root>/code/extensions/      with  <root>/data, <root>/results
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))  # analysis/ or code/

import run_real_analysis as R  # noqa: E402
from features import build_static_features, to_numeric_value  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "extensions"
OUT.mkdir(parents=True, exist_ok=True)


def first_existing(*paths: Path) -> Path:
    """Return the first path that exists, or raise naming all candidates."""
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("none of: " + ", ".join(str(p) for p in paths))


# The main pipeline writes results/real_results.json; the Zenodo deposit
# also ships a copy at its top level.
REAL_RESULTS = first_existing(ROOT / "results" / "real_results.json",
                              ROOT / "real_results.json")


def load():
    """The parsed Atlas, with the pipeline's grouping column.

    The pipeline groups on paper_doi cast to string, so the one publication
    without a DOI forms its own group ('nan'). That gives 63 groups.
    """
    df = pd.read_csv(DATA / "atlas_parsed.csv", low_memory=False)
    df["group"] = df["paper_doi"].astype(str)
    return df


def targets(df, extra=None):
    """Yield (target, label, raw, static, y, groups) exactly as the pipeline builds them.

    extra: optional Series aligned to df, carried through the same filtering
    and returned as an extra element when given.
    """
    static_all = build_static_features(df)
    groups_all = df["group"].values
    for target, label in R.TARGETS.items():
        mask = df[target].notna().values
        y = df.loc[mask, target].apply(to_numeric_value).values.astype(float)
        ok = np.isfinite(y)
        raw = df.loc[mask].iloc[ok].reset_index(drop=True)
        static = static_all.loc[mask].iloc[ok].reset_index(drop=True)
        y, g = y[ok], groups_all[mask][ok]
        ex = None if extra is None else extra[mask][ok].reset_index(drop=True)
        lo, hi = np.percentile(y, [0.5, 99.5])
        keep = (y >= lo) & (y <= hi)
        raw = raw[keep].reset_index(drop=True)
        static = static[keep].reset_index(drop=True)
        y, g = y[keep], g[keep]
        if ex is not None:
            ex = ex[keep].reset_index(drop=True)
        yield target, label, raw, static, y, g, ex
