"""Write the extension results into the supplementary tables.

  S2_model_results.csv      + stacked ensemble rows (3 targets x 3 splits)
  S7_lab_country.csv        one row per publication: lab country and source
  S8_country_analysis.csv   one row per target: country vs study variance,
                            grouped R2 with/without country, transfer R2
  S9_literature_coverage.csv  PubMed counts and database overlap (Fig. 1b)
  S10_sensitivity.csv       one-hot encoding and range-midpoint checks
  S11_representativeness.csv  how far the 63 publications resemble the field

In the repository: writes to submission/supplementary/. zenodo_deposit/ is a
frozen, self-contained copy with its own S6_code/ and is not touched.

Run after stacking.py, countries.py, country_analysis.py, coverage.py,
sensitivity.py and representativeness.py:
    python analysis/extensions/make_supplementary.py
"""
import json
import shutil

import pandas as pd

from common import DATA, OUT, ROOT, first_existing

REPO_SUPP = ROOT / "supplementary"
IN_REPO = REPO_SUPP.exists()
SUPP = REPO_SUPP if IN_REPO else ROOT
SPLIT = {"random": "random", "grouped": "grouped", "loso": "leave_one_study_out"}
CODE = ["common.py", "stacking.py", "countries.py", "country_analysis.py",
        "coverage.py", "sensitivity.py", "representativeness.py",
        "make_supplementary.py"]


def main():
    # S2: append stacked ensemble
    s2 = pd.read_csv(SUPP / "S2_model_results.csv")
    s2 = s2[s2.model != "stacked_ensemble"]
    stack = json.loads((OUT / "stacking_results.json").read_text())
    rows = []
    for split in ("random", "grouped", "loso"):
        for t, r in stack.items():
            base = s2[s2.target == r["label"]].iloc[0]
            m = r[split]
            rows.append(dict(target=r["label"], n=int(base.n), n_studies=int(base.n_studies),
                             model="stacked_ensemble", split=SPLIT[split],
                             r2=m["r2"], mae=m["mae"], rmse=m["rmse"]))
    s2 = pd.concat([s2, pd.DataFrame(rows)], ignore_index=True)
    s2.to_csv(SUPP / "S2_model_results.csv", index=False)

    # S7: lab country per publication
    pc = pd.read_csv(first_existing(DATA / "paper_countries.csv",
                                    SUPP / "S7_lab_country.csv"))
    pc.to_csv(SUPP / "S7_lab_country.csv", index=False)

    # S8: country analysis per target
    cr = json.loads((OUT / "country_results.json").read_text())
    s8 = []
    for t, e in cr["targets"].items():
        s8.append(dict(
            target=e["label"], n=e["n"],
            n_US=e["n_by_country"].get("US", 0), n_CA=e["n_by_country"].get("CA", 0),
            n_other=e["n_by_country"].get("other", 0),
            icc1_study=e["icc1_study"], icc1_country=e["icc1_country"],
            r2_study_mean_in_sample=e["r2_study_mean"],
            r2_country_mean_in_sample=e["r2_country_mean"],
            best_grouped_r2=max(e["grouped_r2"].values()),
            best_grouped_r2_with_country=max(e["grouped_r2_with_country"].values()),
            transfer_r2_train_US_test_other=e["transfer_r2"]["train_US_test_other"],
            transfer_r2_train_other_test_US=e["transfer_r2"]["train_other_test_US"]))
    pd.DataFrame(s8).to_csv(SUPP / "S8_country_analysis.csv", index=False)

    # S9: literature coverage
    cov = json.loads((OUT / "coverage.json").read_text())
    s9 = pd.DataFrame([
        (f"PubMed: lipid nanoparticle(s) in title/abstract, 2005-2025, "
         f"excluding reviews (retrieved {cov['retrieved']})", cov["pubmed"]["all"]),
        ("PubMed: of which published 2021-2025", cov["pubmed"]["all_2021_2025"]),
        ("PubMed: and mention mRNA, siRNA or nucleic acid", cov["pubmed"]["nucleic_acid"]),
        ("PubMed: and mention a particle characterization term", cov["pubmed"]["nucleic_acid_characterised"]),
        ("PubMed: same base query including review article types",
         cov["pubmed"]["all_including_reviews"]),
        ("PubMed: of which published 2021-2025, including reviews",
         cov["pubmed"]["all_2021_2025_including_reviews"]),
        ("LNP Atlas publications", cov["atlas_publications"]),
        ("LNPDB publications", cov["lnpdb_publications"]),
        ("Publications in both databases (matched by PMID)", cov["overlap"]),
        ("Publications in either database", cov["union"]),
    ], columns=["quantity", "count"])
    s9.to_csv(SUPP / "S9_literature_coverage.csv", index=False)

    # S10: encoding and range sensitivity (from sensitivity.py)
    sens = json.loads((OUT / "sensitivity_results.json").read_text())
    s10 = []
    for t, e in sens.items():
        for variant, key in (("frequency encoding, leading value (main analysis)", "frequency"),
                             ("one-hot encoding, leading value", "onehot"),
                             ("frequency encoding, midpoint of ranges", "midpoint")):
            for split, col in (("grouped", "grouped"), ("leave_one_study_out", "loso")):
                for model, r2 in e[key][col].items():
                    s10.append(dict(target=e["label"], variant=variant, split=split,
                                    model=model, r2=r2))
    pd.DataFrame(s10).to_csv(SUPP / "S10_sensitivity.csv", index=False)

    # S11: how far the 63 publications resemble the field
    rep = json.loads((OUT / "representativeness.json").read_text())
    y, j, r = rep["years"], rep["journals"], rep["reach"]
    s11 = pd.DataFrame([
        ("Publications in the LNP Atlas", rep["publications_total"]),
        ("Publications carrying a publication year", y["n_publications"]),
        ("Publications without a year, excluded from the year comparison",
         rep["publications_without_year"]),
        ("Published 2021 onward, observed", y["observed_2021_onward"]),
        ("Published 2021 onward, expected from the field's year mix",
         y["expected_2021_onward"]),
        ("Atlas share published 2021 onward", y["atlas_share_2021_onward"]),
        ("Field share published 2021 onward", y["field_share_2021_onward"]),
        (f"Chi-square goodness of fit ({y['bins']})", y["chi2"]),
        ("Chi-square p value", y["p_value"]),
        ("Distinct journals", j["n_distinct_journals"]),
        ("Most common journal", j["most_common_journal"]),
        ("Share from the most common journal", j["most_common_journal_share"]),
        ("Share from the five most common journals", j["top5_share"]),
        ("Herfindahl index of journal concentration", j["herfindahl_index"]),
        ("Primary research reports in the field, 2005-2025",
         r["field_primary_reports_2005_2025"]),
        ("Atlas publications as a share of the field", r["atlas_share_of_field"]),
        ("Publications reached by either database", r["either_database_publications"]),
        ("Either database as a share of the field", r["either_database_share_of_field"]),
        ("Primary research reports per curated publication",
         r["reports_per_curated_publication"]),
    ], columns=["quantity", "value"])
    s11.to_csv(SUPP / "S11_representativeness.csv", index=False)

    if IN_REPO:
        ext = SUPP / "S6_code" / "extensions"
        ext.mkdir(parents=True, exist_ok=True)
        for f in CODE:
            shutil.copy2(ROOT / "analysis" / "extensions" / f, ext / f)
    print("S2 rows:", len(s2))
    print(pd.DataFrame(s8).to_string())
    print(s9.to_string())
    print(s11.to_string())


if __name__ == "__main__":
    main()
