"""
Modelling on the real LNP Atlas data, mirroring the PLGA companion paper.

Design matched to Sharma et al., Pharmaceutics 2026, 18(7), 805:
  - formulation-grouped cross-validation (no within-study leakage)
  - DOI-level leave-one-study-out (LOSO) for cross-laboratory transport
  - a synthetic positive control through the identical machinery
  - ICC(1) variance decomposition by study identity
  - leverage-based applicability domain

All categorical encoding, imputation and scaling are fitted INSIDE each
training fold. Frequency encoding in particular must not see the test fold.

Targets are the physicochemical attributes reported in comparable units.
Transfection efficiency is not modelled; its units are not comparable across
studies, which is reported as a finding instead.

Writes every number to results/real_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    LIPID_COLS, FoldSafeFrequencyEncoder, build_static_features,
    to_numeric_value,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 20260904

TARGETS = {
    "particle_size_nm_std": "Particle size (nm)",
    "pdi_std": "Polydispersity index",
    "encapsulation_efficiency_percent_std": "Encapsulation efficiency (%)",
}

MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "linear": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, random_state=SEED, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=SEED),
    "svr_rbf": lambda: SVR(kernel="rbf"),
}


def metrics(y, p) -> dict:
    r = y - p
    ss_res = float(np.sum(r**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else float("nan"),
        "mae": round(float(np.mean(np.abs(r))), 4),
        "rmse": round(float(np.sqrt(np.mean(r**2))), 4),
        "n": int(len(y)),
    }


def fit_predict_fold(raw_tr, raw_te, static_tr, static_te, y_tr, model):
    """One fold, with every transform fitted on the training rows only."""
    enc = FoldSafeFrequencyEncoder(LIPID_COLS).fit(raw_tr)
    Xtr = pd.concat([static_tr.reset_index(drop=True),
                     enc.transform(raw_tr).reset_index(drop=True)], axis=1)
    Xte = pd.concat([static_te.reset_index(drop=True),
                     enc.transform(raw_te).reset_index(drop=True)], axis=1)

    imp = SimpleImputer(strategy="mean").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    m = model()
    m.fit(sc.transform(imp.transform(Xtr)), y_tr)
    return m.predict(sc.transform(imp.transform(Xte)))


def grouped_cv(raw, static, y, groups, model, n_splits=10) -> dict:
    k = min(n_splits, len(np.unique(groups)))
    if k < 2:
        return {"note": "too few groups"}
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(static, y, groups):
        oof[te] = fit_predict_fold(raw.iloc[tr], raw.iloc[te],
                                   static.iloc[tr], static.iloc[te],
                                   y[tr], model)
    m = ~np.isnan(oof)
    out = metrics(y[m], oof[m])
    out["n_folds"] = k
    return out


def loso(raw, static, y, groups, model) -> dict:
    if len(np.unique(groups)) < 3:
        return {"note": "too few studies"}
    P, T = [], []
    for tr, te in LeaveOneGroupOut().split(static, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        P.append(fit_predict_fold(raw.iloc[tr], raw.iloc[te],
                                  static.iloc[tr], static.iloc[te],
                                  y[tr], model))
        T.append(y[te])
    if not P:
        return {"note": "no valid folds"}
    out = metrics(np.concatenate(T), np.concatenate(P))
    out["n_studies_held_out"] = len(P)
    return out


def random_split_cv(raw, static, y, model, n_splits=10) -> dict:
    """Ordinary random k-fold, IGNORING which study each formulation came from.

    This is how most published nanoparticle models are validated. It is
    reported here not as a result but as a comparator: formulations from one
    laboratory land in both training and test folds, so the model can
    memorise a study-specific offset and score well without having learned
    anything transferable. The gap between this and the grouped estimate is
    the size of that illusion.
    """
    oof = np.full(len(y), np.nan)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in cv.split(static, y):
        oof[te] = fit_predict_fold(raw.iloc[tr], raw.iloc[te],
                                   static.iloc[tr], static.iloc[te],
                                   y[tr], model)
    m = ~np.isnan(oof)
    out = metrics(y[m], oof[m])
    out["n_folds"] = n_splits
    return out


def study_mean_baseline(y, groups) -> dict:
    """How much variance does 'which laboratory made it' explain, in sample?

    Not a predictive model, since a new study has no mean. It bounds the
    variance that study identity alone accounts for.
    """
    d = pd.DataFrame({"y": y, "g": groups})
    pred = d.groupby("g").y.transform("mean")
    return metrics(d.y.values, pred.values)


def icc1(values, groups) -> dict:
    d = pd.DataFrame({"v": values, "g": groups})
    sizes = d.groupby("g").size()
    d = d[d.g.isin(sizes[sizes >= 2].index)]
    k = d.g.nunique()
    if k < 2:
        return {"note": "insufficient studies"}
    grand = d.v.mean()
    n_i = d.groupby("g").size().values
    means = d.groupby("g").v.mean().values
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(sum(((g.v - g.v.mean()) ** 2).sum() for _, g in d.groupby("g")))
    dfb, dfw = k - 1, len(d) - k
    if dfw <= 0:
        return {"note": "insufficient replication"}
    msb, msw = ssb / dfb, ssw / dfw
    n0 = (len(d) - np.sum(n_i**2) / len(d)) / dfb
    vb = max(0.0, (msb - msw) / n0)
    icc = vb / (vb + msw) if (vb + msw) > 0 else float("nan")
    f = msb / msw if msw > 0 else float("nan")
    return {
        "icc1": round(float(icc), 4), "n_studies": int(k),
        "n_obs": int(len(d)), "F": round(float(f), 3),
        "p_value": (float(1 - stats.f.cdf(f, dfb, dfw))
                    if np.isfinite(f) else None),
    }


def applicability_domain(X: pd.DataFrame) -> dict:
    Xf = StandardScaler().fit_transform(
        SimpleImputer(strategy="mean").fit_transform(X))
    n, p = Xf.shape
    lev = np.diag(Xf @ np.linalg.pinv(Xf.T @ Xf) @ Xf.T)
    h = 3 * p / n
    return {"h_star": round(float(h), 4), "n": int(n), "n_features": int(p),
            "pct_in_domain": round(100 * float(np.mean(lev <= h)), 1),
            "n_out_of_domain": int(np.sum(lev > h))}


def positive_control(raw, static, groups) -> dict:
    """Known linear signal through the identical fold machinery."""
    rng = np.random.default_rng(SEED)
    base = StandardScaler().fit_transform(
        SimpleImputer(strategy="mean").fit_transform(static))
    y = base @ rng.normal(size=base.shape[1]) + rng.normal(
        scale=0.01, size=base.shape[0])
    return {
        "grouped_cv": grouped_cv(raw, static, y, groups, MODELS["ridge"]),
        "loso": loso(raw, static, y, groups, MODELS["ridge"]),
    }


def endpoint_heterogeneity(df: pd.DataFrame) -> dict:
    s = df["bioactivity_profile"].dropna().astype(str)
    units = {
        "relative light units": r"\bRLU\b",
        "percent expression or positive cells":
            r"%\s*(?:GFP|positive|expression|transfect)",
        "fold change": r"fold[- ]change|fold increase",
        "IC50 or EC50": r"\b(?:IC|EC)50\b",
        "percent knockdown or silencing": r"knockdown|silencing",
        "mean fluorescence intensity": r"\bMFI\b",
        "photons or radiance": r"photon|radiance|p/s",
        # Both micro characters are spelled as escapes rather than literals.
        # The Atlas uses U+03BC (Greek small mu, 310 occurrences in this
        # field), not U+00B5 (micro sign); a literal here is also silently
        # corrupted when the file is read under a non-UTF-8 default encoding,
        # which undercounted this convention as 61 instead of 73.
        "protein mass per volume":
            "\\b(?:ng|pg|ug|[\u00b5\u03bc]g)/m[lL]\\b",
    }
    counts = {k: int(s.str.contains(v, case=False, regex=True).sum())
              for k, v in units.items()}
    return {"n_with_bioactivity": int(len(s)),
            "distinct_unit_conventions_detected":
                sum(1 for v in counts.values() if v),
            "counts": counts}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = pd.read_csv(DATA / "atlas_parsed.csv", low_memory=False)
    print(f"Dataset: {len(df)} formulations, {df.paper_doi.nunique()} publications\n")

    static_all = build_static_features(df)
    groups_all = df["paper_doi"].astype(str).values

    out: dict = {
        "dataset": {
            "source": "LNP Atlas (Song, Baek & Seo, Scientific Data 2025)",
            "data_doi": "10.5281/zenodo.17243733",
            "paper_doi": "10.1038/s41597-025-06456-w",
            "licence": "CC BY 4.0",
            "n_formulations": int(len(df)),
            "n_publications": int(df.paper_doi.nunique()),
            "year_min": int(df.paper_year.min()),
            "year_max": int(df.paper_year.max()),
            "n_static_features": int(static_all.shape[1]),
            "n_encoded_features": len(LIPID_COLS),
            "static_feature_names": list(static_all.columns),
            "note": ("Frequency encodings for the four lipid identity columns "
                     "are fitted within each training fold, so the total "
                     "feature count is static + encoded."),
        },
        "seed": SEED,
        "targets": {},
    }

    for target, label in TARGETS.items():
        mask = df[target].notna().values
        y = df.loc[mask, target].apply(to_numeric_value).values.astype(float)
        ok = np.isfinite(y)
        raw = df.loc[mask].iloc[ok].reset_index(drop=True)
        static = static_all.loc[mask].iloc[ok].reset_index(drop=True)
        y, g = y[ok], groups_all[mask][ok]

        lo, hi = np.percentile(y, [0.5, 99.5])
        keep = (y >= lo) & (y <= hi)
        raw = raw[keep].reset_index(drop=True)
        static = static[keep].reset_index(drop=True)
        y, g = y[keep], g[keep]

        print(f"=== {label} ===")
        print(f"  n = {len(y)} formulations from {len(np.unique(g))} studies")

        entry = {
            "label": label, "n": int(len(y)),
            "n_studies": int(len(np.unique(g))),
            "mean": round(float(np.mean(y)), 3),
            "sd": round(float(np.std(y)), 3),
            "grouped_cv": {}, "loso": {},
        }
        for name, factory in MODELS.items():
            entry["grouped_cv"][name] = grouped_cv(raw, static, y, g, factory)
            entry["loso"][name] = loso(raw, static, y, g, factory)
            print(f"    {name:18s} CV R2={entry['grouped_cv'][name].get('r2'):>9}"
                  f"   LOSO R2={entry['loso'][name].get('r2'):>9}")

        # The comparator that makes the null interpretable: same data, same
        # model, only the split changes.
        entry["random_split_cv"] = {
            name: random_split_cv(raw, static, y, factory)
            for name, factory in MODELS.items()
        }
        best_rand = max(entry["random_split_cv"].values(),
                        key=lambda v: v.get("r2", -9))["r2"]
        best_grp = max(entry["grouped_cv"].values(),
                       key=lambda v: v.get("r2", -9))["r2"]
        entry["optimism_gap"] = round(best_rand - best_grp, 4)
        print(f"    random-split best R2 = {best_rand}   "
              f"grouped best R2 = {best_grp}   "
              f"optimism gap = {entry['optimism_gap']}")

        entry["study_mean_baseline"] = study_mean_baseline(y, g)
        entry["icc1"] = icc1(y, g)
        entry["applicability_domain"] = applicability_domain(static)
        entry["positive_control"] = positive_control(raw, static, g)

        print(f"    study-mean baseline R2 (in-sample) = "
              f"{entry['study_mean_baseline']['r2']}")
        print(f"    ICC(1) = {entry['icc1'].get('icc1')}")
        print(f"    positive control: CV R2="
              f"{entry['positive_control']['grouped_cv'].get('r2')}, "
              f"LOSO R2={entry['positive_control']['loso'].get('r2')}\n")
        out["targets"][target] = entry

    out["endpoint_heterogeneity"] = endpoint_heterogeneity(df)
    print("Endpoint heterogeneity:",
          out["endpoint_heterogeneity"]["distinct_unit_conventions_detected"],
          "distinct unit conventions")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "real_results.json").write_text(json.dumps(out, indent=2),
                                               encoding="utf-8")
    print(f"\nWrote {RESULTS/'real_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
