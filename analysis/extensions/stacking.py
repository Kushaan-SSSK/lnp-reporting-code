"""Stacked-ensemble sensitivity analysis.

Combines ridge, random forest, gradient boosting and RBF SVR through a
cross-validated ridge meta-learner (ordinary least squares is omitted as it
duplicates ridge). Runs the same three partitioning schemes as the main
analysis, and first reruns gradient boosting under grouped CV as a check
that the pipeline reproduces real_results.json.

Output: results/extensions/stacking_results.json
Run:    python analysis/extensions/stacking.py
"""
import json

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from common import OUT, REAL_RESULTS, R, load, targets


def stack():
    return StackingRegressor(
        estimators=[(k, f()) for k, f in R.MODELS.items() if k != "linear"],
        final_estimator=RidgeCV(),
        cv=KFold(5, shuffle=True, random_state=R.SEED), n_jobs=-1)


def main():
    ref = json.loads(REAL_RESULTS.read_text())
    out = {}
    for target, label, raw, static, y, g, _ in targets(load()):
        check = R.grouped_cv(raw, static, y, g, R.MODELS["gradient_boosting"])["r2"]
        expected = ref["targets"][target]["grouped_cv"]["gradient_boosting"]["r2"]
        assert check == expected, f"pipeline drift: {check} != {expected}"
        res = {"label": label, "n": int(len(y)),
               "random": R.random_split_cv(raw, static, y, stack),
               "grouped": R.grouped_cv(raw, static, y, g, stack),
               "loso": R.loso(raw, static, y, g, stack)}
        res["optimism_gap"] = round(res["random"]["r2"] - res["grouped"]["r2"], 4)
        out[target] = res
        print(label, {k: res[k]["r2"] for k in ("random", "grouped", "loso")}, flush=True)
    (OUT / "stacking_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
