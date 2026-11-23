"""
Analysis pipeline for the LNP reporting-gap study.

Mirrors the validation design of the companion PLGA paper
(Pharmaceutics 2026, 18(7), 805) so results are directly comparable:

  1. Reporting-completeness statistics from the extraction sheet.
  2. Formulation-grouped k-fold cross-validation (no within-formulation leakage).
  3. DOI-level leave-one-study-out (LOSO) validation for cross-laboratory transport.
  4. A SYNTHETIC POSITIVE CONTROL. This is non-negotiable: without it, a negative
     LOSO R-squared cannot be distinguished from a pipeline bug.
  5. ICC(1) variance decomposition by study identity.
  6. Applicability-domain (leverage / Williams plot) analysis.

Every numeric result is written to results/results.json with a stable key so
the manuscript never hard-codes a number that the code did not produce.

Run:  python lnp_analysis.py --extraction ../data/extraction.csv
      python lnp_analysis.py --selftest        (validates the pipeline)
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_SEED = 20260904
N_SPLITS = 10

# Scoring scheme for the reporting audit.
SCORE_MAP = {"1": 1.0, "0.5": 0.5, "0": 0.0}
# Distinct missingness codes. Never conflate these.
MISSING_CODES = {"NR", "NA", "IND"}


# --------------------------------------------------------------------------
# Reporting completeness
# --------------------------------------------------------------------------
def completeness_by_field(scores: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """Percent of studies reporting each field.

    'NA' (not applicable) is excluded from the denominator, because a study
    with no in vivo work cannot fail to report an in vivo field. 'NR' and
    'IND' remain in the denominator. Conflating these three is the single
    most criticised flaw in literature-derived datasets.
    """
    rows = []
    for field in codebook["field_id"]:
        if field not in scores.columns:
            continue
        col = scores[field].astype(str)
        applicable = col[col != "NA"]
        n_applicable = len(applicable)
        if n_applicable == 0:
            continue
        numeric = applicable.map(SCORE_MAP)
        n_full = int((numeric == 1.0).sum())
        n_partial = int((numeric == 0.5).sum())
        n_absent = n_applicable - n_full - n_partial
        meta = codebook.loc[codebook["field_id"] == field].iloc[0]
        # Wilson interval: correct for proportions near 0 or 1, where the
        # normal approximation fails. Several fields will sit near both bounds.
        lo, hi = wilson_ci(n_full, n_applicable)
        rows.append(
            {
                "field_id": field,
                "domain": meta["domain"],
                "field_name": meta["field_name"],
                "priority": meta["priority"],
                "n_applicable": n_applicable,
                "n_fully_reported": n_full,
                "n_partially_reported": n_partial,
                "n_not_reported": n_absent,
                "pct_fully_reported": round(100 * n_full / n_applicable, 1),
                "pct_any_reporting": round(100 * (n_full + n_partial) / n_applicable, 1),
                "ci95_low": round(100 * lo, 1),
                "ci95_high": round(100 * hi, 1),
                "mean_score": round(float(numeric.mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_fully_reported", ascending=False)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def domain_summary(per_field: pd.DataFrame) -> pd.DataFrame:
    """Mean completeness per domain. The headline comparison of the paper."""
    agg = (
        per_field.groupby("domain")
        .agg(
            n_fields=("field_id", "count"),
            mean_pct_fully_reported=("pct_fully_reported", "mean"),
            median_pct_fully_reported=("pct_fully_reported", "median"),
            min_pct=("pct_fully_reported", "min"),
            max_pct=("pct_fully_reported", "max"),
        )
        .round(1)
        .reset_index()
    )
    return agg.sort_values("mean_pct_fully_reported", ascending=False)


def compare_domains(per_field: pd.DataFrame, a: str, b: str) -> dict:
    """Mann-Whitney U test between two domains' completeness distributions.

    Non-parametric because per-field percentages are bounded and not normal.
    """
    xa = per_field.loc[per_field["domain"] == a, "pct_fully_reported"].values
    xb = per_field.loc[per_field["domain"] == b, "pct_fully_reported"].values
    if len(xa) < 2 or len(xb) < 2:
        return {"comparison": f"{a} vs {b}", "note": "insufficient fields"}
    u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    return {
        "comparison": f"{a} vs {b}",
        "n_fields_a": len(xa),
        "n_fields_b": len(xb),
        "median_a": round(float(np.median(xa)), 1),
        "median_b": round(float(np.median(xb)), 1),
        "U": float(u),
        "p_value": float(p),
    }


def temporal_trend(scores: pd.DataFrame, codebook: pd.DataFrame) -> dict:
    """Has reporting completeness changed over time?

    Spearman correlation between publication year and each study's mean score.
    Either direction is publishable; nobody has asked the question.
    """
    if "year" not in scores.columns:
        return {"note": "no year column"}
    field_ids = [f for f in codebook["field_id"] if f in scores.columns]
    per_study = []
    for _, row in scores.iterrows():
        vals = [SCORE_MAP[str(row[f])] for f in field_ids
                if str(row[f]) in SCORE_MAP]
        if vals:
            per_study.append({"year": row["year"], "mean_score": float(np.mean(vals))})
    df = pd.DataFrame(per_study)
    if len(df) < 3:
        return {"note": "insufficient studies"}
    rho, p = stats.spearmanr(df["year"], df["mean_score"])
    return {
        "n_studies": len(df),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "spearman_rho": round(float(rho), 3),
        "p_value": float(p),
        "mean_score_by_year": {
            int(y): round(float(g["mean_score"].mean()), 3)
            for y, g in df.groupby("year")
        },
    }


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
def build_stacked_ensemble(seed: int = RANDOM_SEED):
    """Base learners spanning tree, kernel and linear hypothesis classes.

    Hyperparameters are FIXED, not tuned. The models are diagnostic probes for
    information content, not optimisation targets. Consistent performance
    across structurally different families is evidence about the data, not
    about any one algorithm.
    """
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        "svr_rbf": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "linear": LinearRegression(),
    }


def make_pipeline(model, seed: int = RANDOM_SEED) -> Pipeline:
    """Imputation and scaling are fit INSIDE each fold, never on the full data."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "r2": round(r2, 4),
        "mae": round(float(np.mean(np.abs(resid))), 4),
        "rmse": round(float(np.sqrt(np.mean(resid**2))), 4),
        "n": int(len(y_true)),
    }


def grouped_cv(X, y, groups, model, n_splits=N_SPLITS, seed=RANDOM_SEED) -> dict:
    """Formulation-grouped k-fold. Each formulation sits in exactly one fold,
    so the model must generalise to unseen formulations rather than
    interpolating extra rows of a curve it has already seen."""
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        return {"note": "too few groups"}
    cv = GroupKFold(n_splits=k)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups):
        pipe = make_pipeline(model, seed)
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
    mask = ~np.isnan(oof)
    out = _metrics(y[mask], oof[mask])
    out["n_folds"] = k
    return out


def loso(X, y, groups, model, seed=RANDOM_SEED) -> dict:
    """Leave-one-study-out at DOI level: the cross-laboratory transport test.

    This is the central experiment. In the PLGA companion paper this went
    negative for all three targets while grouped CV was merely weak, which is
    the signature of study-level effects the covariates cannot capture.
    """
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        return {"note": "too few studies"}
    logo = LeaveOneGroupOut()
    preds, trues = [], []
    per_study = []
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = make_pipeline(model, seed)
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        preds.append(p)
        trues.append(y[te])
        per_study.append(
            {"study": str(groups[te][0]), "n": int(len(te)),
             "mae": round(float(np.mean(np.abs(y[te] - p))), 4)}
        )
    if not preds:
        return {"note": "no valid folds"}
    yh = np.concatenate(preds)
    yt = np.concatenate(trues)
    out = _metrics(yt, yh)          # pooled across all held-out studies
    out["n_studies"] = len(per_study)
    out["per_study"] = per_study
    return out


def synthetic_positive_control(X, groups, seed=RANDOM_SEED) -> dict:
    """Inject a target that IS a known linear function of the features, then
    run it through the identical grouped-CV and LOSO machinery.

    A correctly implemented pipeline recovers this almost perfectly
    (R-squared near 1.00). If it does not, the folds, scaling or grouping are
    broken and no null result from this pipeline may be interpreted.
    """
    rng = np.random.default_rng(seed)
    Xf = SimpleImputer(strategy="mean").fit_transform(X)
    Xf = StandardScaler().fit_transform(Xf)
    coefs = rng.normal(size=Xf.shape[1])
    y_syn = Xf @ coefs + rng.normal(scale=0.01, size=Xf.shape[0])
    # The injected signal is linear, so the probe must be able to represent a
    # linear function. A random forest cannot extrapolate a high-dimensional
    # linear surface from a few hundred points, so using one here would
    # understate the machinery's ability to recover a signal that is present
    # and would defeat the purpose of the control. A tree-based control is
    # reported alongside it for completeness.
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    return {
        "grouped_cv": grouped_cv(X, y_syn, groups, ridge, seed=seed),
        "loso": loso(X, y_syn, groups, ridge, seed=seed),
        "grouped_cv_tree_probe": grouped_cv(X, y_syn, groups, rf, seed=seed),
        "interpretation": (
            "Grouped-CV and LOSO R-squared near 1.00 for the linear probe "
            "confirm the validation machinery, including within-fold scaling "
            "and imputation and the grouping variable, recovers a strong "
            "signal when one is present. A null result on the real targets "
            "therefore reflects the data, not the pipeline."
        ),
    }


def icc1(values: np.ndarray, groups: np.ndarray) -> dict:
    """ICC(1): proportion of variance attributable to study identity.

    A high ICC means laboratories differ systematically in ways the recorded
    covariates do not explain, which is precisely what breaks transport.
    """
    df = pd.DataFrame({"v": values, "g": groups})
    counts = df.groupby("g").size()
    keep = counts[counts >= 2].index
    df = df[df["g"].isin(keep)]
    k_groups = df["g"].nunique()
    if k_groups < 2:
        return {"note": "fewer than two studies with >=2 formulations"}
    grand = df["v"].mean()
    n_i = df.groupby("g").size().values
    means = df.groupby("g")["v"].mean().values
    ss_between = float(np.sum(n_i * (means - grand) ** 2))
    ss_within = float(
        sum(((g["v"] - g["v"].mean()) ** 2).sum() for _, g in df.groupby("g"))
    )
    df_b, df_w = k_groups - 1, len(df) - k_groups
    if df_w <= 0:
        return {"note": "insufficient within-study replication"}
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    n0 = (df.shape[0] - np.sum(n_i**2) / df.shape[0]) / df_b
    var_b = max(0.0, (ms_b - ms_w) / n0)
    icc = var_b / (var_b + ms_w) if (var_b + ms_w) > 0 else float("nan")
    f = ms_b / ms_w if ms_w > 0 else float("nan")
    p = 1 - stats.f.cdf(f, df_b, df_w) if np.isfinite(f) else float("nan")
    return {
        "icc1": round(float(icc), 4),
        "n_studies": int(k_groups),
        "n_observations": int(len(df)),
        "F": round(float(f), 3),
        "p_value": float(p),
    }


def applicability_domain(X: np.ndarray) -> dict:
    """Leverage-based applicability domain (Williams plot).

    Warning leverage h* = 3p/N. Points above it are extrapolations. If
    filtering to in-domain points does not rescue performance, the limit is
    missing variables rather than covariate extrapolation alone.
    """
    Xf = SimpleImputer(strategy="mean").fit_transform(X)
    Xf = StandardScaler().fit_transform(Xf)
    n, p = Xf.shape
    try:
        hat = Xf @ np.linalg.pinv(Xf.T @ Xf) @ Xf.T
        lev = np.diag(hat)
    except np.linalg.LinAlgError:
        return {"note": "singular matrix"}
    h_star = 3 * p / n
    return {
        "n": int(n),
        "n_features": int(p),
        "h_star": round(float(h_star), 4),
        "n_in_domain": int(np.sum(lev <= h_star)),
        "n_out_of_domain": int(np.sum(lev > h_star)),
        "pct_in_domain": round(100 * float(np.mean(lev <= h_star)), 1),
    }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def selftest() -> dict:
    """Validate the pipeline on data with KNOWN structure.

    Two regimes are checked:
      A. Learnable  - target is a clean function of the features.
      B. Study-confounded - a large per-study offset the features cannot see,
         reproducing the information-limited regime the PLGA paper reported.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_studies, per_study, n_feat = 20, 8, 10
    groups = np.repeat(np.arange(n_studies), per_study)
    X = rng.normal(size=(n_studies * per_study, n_feat))
    coefs = rng.normal(size=n_feat)

    y_learn = X @ coefs + rng.normal(scale=0.1, size=len(groups))
    study_offset = rng.normal(scale=8.0, size=n_studies)[groups]
    y_conf = X @ coefs + study_offset + rng.normal(scale=0.1, size=len(groups))

    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    out = {
        "regime_A_learnable": {
            "grouped_cv": grouped_cv(X, y_learn, groups, LinearRegression()),
            "loso": loso(X, y_learn, groups, LinearRegression()),
        },
        "regime_B_study_confounded": {
            "grouped_cv": grouped_cv(X, y_conf, groups, rf),
            "loso": loso(X, y_conf, groups, rf),
            "icc1": icc1(y_conf, groups),
        },
        "positive_control": synthetic_positive_control(X, groups),
        "applicability_domain": applicability_domain(X),
    }

    checks = {
        "A_grouped_cv_recovers_signal":
            out["regime_A_learnable"]["grouped_cv"]["r2"] > 0.90,
        "A_loso_recovers_signal":
            out["regime_A_learnable"]["loso"]["r2"] > 0.90,
        "B_loso_degrades_vs_cv":
            out["regime_B_study_confounded"]["loso"]["r2"]
            < out["regime_B_study_confounded"]["grouped_cv"]["r2"],
        "B_icc_detects_study_effect":
            out["regime_B_study_confounded"]["icc1"]["icc1"] > 0.5,
        "positive_control_near_perfect":
            out["positive_control"]["grouped_cv"]["r2"] > 0.90,
    }
    out["checks"] = checks
    out["all_passed"] = bool(all(checks.values()))
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", type=Path, default=None)
    ap.add_argument("--codebook", type=Path,
                    default=Path(__file__).parent.parent / "protocol" / "extraction_codebook.csv")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent.parent / "results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "provenance": {
            "seed": RANDOM_SEED,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
    }

    if args.selftest:
        results["selftest"] = selftest()
        status = "PASSED" if results["selftest"]["all_passed"] else "FAILED"
        print(f"Pipeline self-test: {status}")
        for name, ok in results["selftest"]["checks"].items():
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        a = results["selftest"]["regime_A_learnable"]
        b = results["selftest"]["regime_B_study_confounded"]
        print(f"\n  Regime A (learnable):        grouped CV R2={a['grouped_cv']['r2']}, "
              f"LOSO R2={a['loso']['r2']}")
        print(f"  Regime B (study-confounded): grouped CV R2={b['grouped_cv']['r2']}, "
              f"LOSO R2={b['loso']['r2']}, ICC(1)={b['icc1']['icc1']}")

    if args.extraction and args.extraction.exists():
        codebook = pd.read_csv(args.codebook)
        scores = pd.read_csv(args.extraction)
        per_field = completeness_by_field(scores, codebook)
        per_field.to_csv(args.outdir / "completeness_by_field.csv", index=False)
        results["n_studies"] = int(len(scores))
        results["completeness_by_domain"] = domain_summary(per_field).to_dict("records")
        results["composition_vs_process"] = compare_domains(
            per_field, "Composition", "Process"
        )
        results["temporal_trend"] = temporal_trend(scores, codebook)
        print(f"\nExtraction analysed: {len(scores)} studies, "
              f"{len(per_field)} fields scored.")
    elif not args.selftest:
        print("No extraction file supplied. Run with --selftest to validate the "
              "pipeline, or pass --extraction once extraction has begun.")

    out_path = args.outdir / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
