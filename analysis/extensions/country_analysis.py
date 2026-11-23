"""Does the laboratory's country explain the between-study differences?

For each target:
  1. variance explained by country vs by study (ICC(1), and in-sample R2
     of a predictor that returns the group mean);
  2. study-grouped CV with and without country as a feature
     (United States / Canada / other);
  3. transfer: random forest trained on US laboratories and tested on all
     others, and the reverse.

Input:  data/paper_countries.csv (from countries.py), or the identical
        S7_lab_country.csv shipped with the supplementary material
Output: results/extensions/country_results.json
Run:    python analysis/extensions/country_analysis.py
"""
import json

import pandas as pd

from common import DATA, OUT, ROOT, R, first_existing, load, targets


def main():
    df = load()
    pc = pd.read_csv(first_existing(DATA / "paper_countries.csv",
                                    ROOT / "S7_lab_country.csv"))
    df["country"] = df.group.map(dict(zip(pc.paper_doi.astype(str), pc.lab_country)))
    assert df.country.notna().all(), "publication without a country"
    country3 = df.country.where(df.country.isin(["US", "CA"]), "other")

    out = {"publications_by_country": pc.lab_country.value_counts().to_dict(),
           "formulations_by_country": df.country.value_counts().to_dict(),
           "targets": {}}
    for target, label, raw, static, y, g, c3 in targets(df, extra=country3):
        cty = raw.country.values
        e = {"label": label, "n": int(len(y)),
             "n_by_country": c3.value_counts().to_dict(),
             "icc1_study": R.icc1(y, g).get("icc1"),
             "icc1_country": R.icc1(y, cty).get("icc1"),
             "r2_study_mean": R.study_mean_baseline(y, g)["r2"],
             "r2_country_mean": R.study_mean_baseline(y, cty)["r2"],
             "grouped_r2": {}, "grouped_r2_with_country": {}, "transfer_r2": {}}
        with_country = static.copy()
        with_country["country_US"] = (c3 == "US").astype(int).values
        with_country["country_CA"] = (c3 == "CA").astype(int).values
        for m in ("ridge", "random_forest", "gradient_boosting"):
            e["grouped_r2"][m] = R.grouped_cv(raw, static, y, g, R.MODELS[m])["r2"]
            e["grouped_r2_with_country"][m] = R.grouped_cv(
                raw, with_country, y, g, R.MODELS[m])["r2"]
        us = (c3 == "US").values
        for name, train in (("train_US_test_other", us), ("train_other_test_US", ~us)):
            pred = R.fit_predict_fold(raw[train], raw[~train], static[train],
                                      static[~train], y[train],
                                      R.MODELS["random_forest"])
            e["transfer_r2"][name] = R.metrics(y[~train], pred)["r2"]
        out["targets"][target] = e
        print(label, json.dumps(e), flush=True)
    (OUT / "country_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
