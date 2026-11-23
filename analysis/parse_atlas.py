"""
Parse the LNP Atlas dataset into a structured, analysable table.

Source: Song S, Baek J, Seo S. "A Comprehensive Dataset of Lipid Nanoparticle
Compositions and Properties for Nucleic Acid Delivery." Scientific Data (2025).
DOI 10.1038/s41597-025-06456-w. Data: Zenodo DOI 10.5281/zenodo.17243733,
CC BY 4.0. Used here with attribution.

The central measurement of this study: the Atlas records all eleven
manufacturing parameters inside one free-text `synthesis_info` column. Every
row carries all eleven LABELS, so a naive presence check returns 100%. What
matters is whether a VALUE follows the label. This script separates the two.

Outputs (to data/):
  atlas_parsed.csv      one row per formulation, process fields split out
  atlas_coverage.csv    per-field reporting rates with Wilson intervals
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SRC = DATA / "LNP_Atlas_DB_202509_v1.csv"

# The eleven process keys the Atlas compresses into synthesis_info.
PROCESS_KEYS = [
    "synthesis_method", "device_used", "total_flow_rate_ml_min",
    "flow_rate_ratio", "aqueous_phase_composition",
    "organic_phase_composition", "mixing_time_sec", "temperature_condition",
    "post_processing", "sterilization", "storage_buffer",
]

# Strings that are syntactically present but carry no information.
NULL_TOKENS = {
    "", "na", "n/a", "not reported", "not specified", "none", "not stated",
    "unknown", "not available", "not applicable", "nr", "-", "--", "not given",
    "not described", "not mentioned", "unspecified", "not provided",
}


def split_synthesis(text: str) -> dict[str, str]:
    """Split the free-text column on its known keys.

    Splitting on the known key set rather than on every 'word:' pattern avoids
    being fooled by colons inside values, e.g. a flow rate ratio of '3:1'.
    """
    if not isinstance(text, str):
        return {k: "" for k in PROCESS_KEYS}
    pattern = "|".join(re.escape(k) for k in PROCESS_KEYS)
    parts = list(re.finditer(rf"({pattern})\s*:", text))
    out = {k: "" for k in PROCESS_KEYS}
    for i, m in enumerate(parts):
        end = parts[i + 1].start() if i + 1 < len(parts) else len(text)
        value = text[m.end():end].strip().strip(";").strip()
        out[m.group(1)] = value
    return out


def is_populated(value: str) -> bool:
    """True only if a real value follows the label."""
    if not isinstance(value, str):
        return False
    v = value.strip().strip(";").strip().lower()
    return v not in NULL_TOKENS and len(v) > 0


def is_quantitative(value: str) -> bool:
    """True if the value contains a number, i.e. it is usable as a covariate.

    'total_flow_rate_ml_min: 12' is modellable. 'mixing_time_sec: overnight'
    is populated but not quantitative, and cannot enter a numeric model.
    """
    return is_populated(value) and bool(re.search(r"\d", value))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        return 1

    # The released Atlas file is CP949 (Korean), not UTF-8: its plus-minus
    # signs are the byte pair A1 BE, which only that family decodes. Latin-1
    # was used here before, and because Latin-1 maps every byte it never
    # raises -- it just turned each plus-minus into "\xa1\xbe" and carried the
    # mojibake into the parsed dataset. Decoding as CP949 recovers all 613 of
    # them. This changes no reported number, because every value is read by
    # taking the leading number and the mangling sat only in the separator,
    # but the published Supporting Information should be legible.
    df = pd.read_csv(SRC, encoding="cp949")
    print(f"Loaded {len(df)} formulations from {df.paper_doi.nunique()} "
          f"publications ({int(df.paper_year.min())}-{int(df.paper_year.max())}).")

    parsed = df.synthesis_info.apply(split_synthesis).apply(pd.Series)
    parsed.columns = [f"proc_{c}" for c in parsed.columns]
    out = pd.concat([df, parsed], axis=1)
    out.to_csv(DATA / "atlas_parsed.csv", index=False, encoding="utf-8")

    rows = []
    n = len(df)

    # Composition and characterization fields: a value is present or it is not.
    direct = {
        "ionizable_lipid": "Composition",
        "peg_lipid": "Composition",
        "sterol_lipid": "Composition",
        "helper_lipid": "Composition",
        "lipid_molar_ratio": "Composition",
        "ionizable_lipid_smiles": "Composition",
        "particle_size_nm_std": "Physicochemical",
        "pdi_std": "Physicochemical",
        "zeta_potential_mv_std": "Physicochemical",
        "encapsulation_efficiency_percent_std": "Physicochemical",
        "loading_capacity_std": "Physicochemical",
        "target_type": "Cargo",
        "nucleic_acid_sequence": "Cargo",
        "bioactivity_profile": "Biological",
    }
    for col, domain in direct.items():
        if col not in df.columns:
            continue
        k = int(df[col].apply(lambda v: is_populated(str(v)) if pd.notna(v)
                              else False).sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"field": col, "domain": domain, "n": n, "n_populated": k,
                     "pct_populated": round(100 * k / n, 1),
                     "pct_quantitative": np.nan,
                     "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1)})

    # Process fields: label presence and value presence are different things.
    for key in PROCESS_KEYS:
        col = f"proc_{key}"
        k = int(out[col].apply(is_populated).sum())
        q = int(out[col].apply(is_quantitative).sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"field": key, "domain": "Process", "n": n,
                     "n_populated": k, "pct_populated": round(100 * k / n, 1),
                     "pct_quantitative": round(100 * q / n, 1),
                     "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1)})

    cov = pd.DataFrame(rows).sort_values(["domain", "pct_populated"],
                                         ascending=[True, False])
    cov.to_csv(DATA / "atlas_coverage.csv", index=False, encoding="utf-8")

    print("\nReporting completeness by domain (mean % of formulations):")
    for dom, g in cov.groupby("domain"):
        print(f"  {dom:18s} {g.pct_populated.mean():5.1f}%  ({len(g)} fields)")

    print("\nProcess fields: label always present, value often absent")
    pr = cov[cov.domain == "Process"]
    for _, r in pr.iterrows():
        quant = ("-" if pd.isna(r.pct_quantitative)
                 else f"{r.pct_quantitative:5.1f}%")
        print(f"  {r.field:30s} value {r.pct_populated:5.1f}%   numeric {quant}")

    print(f"\nWrote {DATA/'atlas_parsed.csv'} and {DATA/'atlas_coverage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
