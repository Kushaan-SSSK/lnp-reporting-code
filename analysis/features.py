"""
Feature construction for the LNP Atlas analysis.

Kept separate from the modelling code because two of these decisions are
forced by defects in the source data and must be auditable:

  1. The molar-ratio column does not use a consistent component order, so
     positional parsing assigns mole percentages to the wrong lipids. We use
     an order-invariant (sorted) representation instead.
  2. Frequency encoding of lipid identity must be fitted INSIDE each
     cross-validation fold. Computing it on the full dataset leaks
     information across the split and inflates apparent performance.

Both are documented in the manuscript rather than silently handled.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def sorted_molar_ratio(s) -> list[float]:
    """Order-invariant summary of the four component mole percentages.

    The same formulation appears in the source as both '50:10:38.5:1.5' and
    '50:1.5:38.5:10'. Across the dataset the smallest value sits at index 1 in
    58.7% of records and at index 3 in 33.7%, so component order cannot be
    inferred from position. Sorting is invariant to that ambiguity and still
    distinguishes compositions: 225 raw strings collapse to 190 distinct
    sorted vectors, which is the number of genuinely different compositions.

    The cost is that the model cannot know which lipid a given mole percentage
    belongs to. That is a limitation of the source column, not a modelling
    choice, and it is reported as such.
    """
    if not isinstance(s, str):
        return [np.nan] * 4
    head = s.split("(")[0]  # drop parenthetical commentary
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", head)][:4]
    if len(nums) != 4:
        return [np.nan] * 4
    return sorted(nums)


def first_number(s) -> float:
    if not isinstance(s, str):
        return np.nan
    m = re.search(r"(\d+\.?\d*)", s.replace(",", ""))
    return float(m.group(1)) if m else np.nan


def parse_frr(s) -> float:
    """'3:1 (aqueous:ethanol)' -> 3.0"""
    if not isinstance(s, str):
        return np.nan
    m = re.search(r"(\d+\.?\d*)\s*:\s*(\d+\.?\d*)", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else np.nan
    return first_number(s)


def to_numeric_value(v) -> float:
    """Central value of a reported measurement.

    Values appear as plain numbers, as 'mean +/- SD' and as ranges. The
    leading number is the point estimate; dispersion is discarded because a
    formulation-level model cannot use it.
    """
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if not isinstance(v, str):
        return np.nan
    m = re.search(r"-?\d+\.?\d*", v.replace(",", ""))
    return float(m.group(0)) if m else np.nan


# Categorical columns that get frequency encoding, fitted within-fold.
LIPID_COLS = ["ionizable_lipid", "peg_lipid", "sterol_lipid", "helper_lipid"]


class FoldSafeFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoding fitted on training rows only.

    Encoding a category by how often it occurs is standard for
    high-cardinality identifiers, but the frequencies must come from the
    training partition. Fitting on the whole dataset lets test-fold
    composition influence the training representation, which inflates
    apparent performance. Unseen categories map to zero.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.maps_ = {
            c: X[c].astype(str).value_counts(normalize=True).to_dict()
            for c in self.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.columns:
            out[f"{c}_freq"] = X[c].astype(str).map(self.maps_[c]).fillna(0.0)
        return out


def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that involve no fitting and so carry no leakage risk."""
    X = pd.DataFrame(index=df.index)

    ratios = df["lipid_molar_ratio"].apply(sorted_molar_ratio).apply(pd.Series)
    ratios.columns = ["molar_r1", "molar_r2", "molar_r3", "molar_r4"]
    X = pd.concat([X, ratios], axis=1)

    X["frr"] = df["proc_flow_rate_ratio"].apply(parse_frr)
    # The column is nominally mL/min, but only 2 of its 343 values state a
    # unit, and both are mL/h (17000 mL/h), so recorded values span 0.06 to
    # 17000. A log transform keeps the column usable as recorded.
    tfr = df["proc_total_flow_rate_ml_min"].apply(first_number)
    X["log_total_flow_rate"] = np.log10(tfr.where(tfr > 0))

    method = df["proc_synthesis_method"].astype(str)
    X["is_microfluidic"] = method.str.contains("microfluid", case=False,
                                               na=False).astype(int)
    X["is_ethanol_injection"] = method.str.contains(
        "ethanol inject|pipette|hand", case=False, na=False).astype(int)

    tt = df["target_type"].astype(str).str.lower()
    X["cargo_mrna"] = tt.str.contains("mrna", na=False).astype(int)
    X["cargo_sirna"] = tt.str.contains("sirna", na=False).astype(int)

    # Mixing time (91.5% missing) and preparation temperature (88.4% missing)
    # are deliberately excluded. An imputed column that is nine parts
    # imputation carries the imputation, not the measurement.
    return X
