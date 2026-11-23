"""Is the LNP Atlas a representative sample of the field?

The reporting-completeness figures are measured on 63 publications drawn from a
literature of 7,730 primary research reports. That supports a claim about what
is available to anyone reusing the literature through a curated resource, but
not a claim about the literature as a whole unless the sample resembles it.

Three comparisons, all against counts already retrieved by coverage.py:

  year distribution   Atlas publication years vs the PubMed year histogram,
                      compared with a chi-square goodness-of-fit test on the
                      binned counts. Expected counts below five make the
                      per-year test unreliable, so years are pooled into the
                      same 2005-2020 / 2021-2025 split the manuscript uses.
  journal spread      how concentrated the 63 are: distinct journals, the
                      share from the most common one, and a Herfindahl index.
  database reach      the 63, and the 95 reached by either database, as a
                      share of the field.

Writes results/extensions/representativeness.json.
"""
import json
from collections import Counter

import pandas as pd
from scipy import stats

from common import DATA, OUT

COVERAGE = OUT / "coverage.json"
RECENT_FROM = 2021


def year_comparison(atlas_years, field_by_year):
    """Chi-square goodness of fit, Atlas years against the field's year mix.

    Pooled into two bins because the per-year expected counts for a sample of
    63 fall below five in most years, which the test does not tolerate.
    """
    field = {int(y): n for y, n in field_by_year.items()}
    field_total = sum(field.values())
    field_recent = sum(n for y, n in field.items() if y >= RECENT_FROM)

    observed = [
        sum(1 for y in atlas_years if y < RECENT_FROM),
        sum(1 for y in atlas_years if y >= RECENT_FROM),
    ]
    p_recent = field_recent / field_total
    expected = [len(atlas_years) * (1 - p_recent), len(atlas_years) * p_recent]

    chi2, p = stats.chisquare(observed, expected)
    return {
        "n_publications": len(atlas_years),
        "observed_pre_2021": observed[0],
        "observed_2021_onward": observed[1],
        "expected_pre_2021": round(expected[0], 1),
        "expected_2021_onward": round(expected[1], 1),
        "atlas_share_2021_onward": round(observed[1] / len(atlas_years), 3),
        "field_share_2021_onward": round(p_recent, 3),
        "chi2": round(float(chi2), 3),
        "p_value": float(f"{p:.3g}"),
        "bins": f"pre-{RECENT_FROM} vs {RECENT_FROM} onward",
    }


def journal_concentration(journals):
    counts = Counter(j for j in journals if isinstance(j, str) and j.strip())
    total = sum(counts.values())
    top_journal, top_n = counts.most_common(1)[0]
    herfindahl = sum((n / total) ** 2 for n in counts.values())
    return {
        "n_publications_with_journal": total,
        "n_distinct_journals": len(counts),
        "most_common_journal": top_journal,
        "most_common_journal_n": top_n,
        "most_common_journal_share": round(top_n / total, 3),
        "top5_share": round(sum(n for _, n in counts.most_common(5)) / total, 3),
        "herfindahl_index": round(herfindahl, 4),
        "counts": dict(counts.most_common()),
    }


def main():
    cov = json.loads(COVERAGE.read_text())
    df = pd.read_csv(DATA / "atlas_parsed.csv", low_memory=False)

    # One row per publication, not per formulation.
    pubs = df.drop_duplicates(subset="paper_doi")[
        ["paper_doi", "paper_year", "paper_journal"]
    ]
    years = [int(y) for y in pubs["paper_year"].dropna()]

    field_total = cov["pubmed"]["all"]
    atlas_n = cov["atlas_publications"]
    union_n = cov["union"]

    out = {
        "years": year_comparison(years, cov["pubmed"]["by_year"]),
        "publications_total": len(pubs),
        "publications_without_year": int(len(pubs) - len(years)),
        "journals": journal_concentration(pubs["paper_journal"]),
        "reach": {
            "field_primary_reports_2005_2025": field_total,
            "atlas_publications": atlas_n,
            "atlas_share_of_field": round(atlas_n / field_total, 5),
            "either_database_publications": union_n,
            "either_database_share_of_field": round(union_n / field_total, 5),
            "reports_per_curated_publication": round(field_total / union_n, 1),
        },
        "source": {
            "coverage_retrieved": cov["retrieved"],
            "atlas_publications_with_year": len(years),
        },
    }

    (OUT / "representativeness.json").write_text(json.dumps(out, indent=2))

    y = out["years"]
    print(f"years: Atlas {y['atlas_share_2021_onward']:.1%} from {RECENT_FROM} onward "
          f"vs field {y['field_share_2021_onward']:.1%}, "
          f"chi2={y['chi2']}, p={y['p_value']}")
    j = out["journals"]
    print(f"journals: {j['n_distinct_journals']} distinct, "
          f"top journal {j['most_common_journal']} at {j['most_common_journal_share']:.1%}, "
          f"top five {j['top5_share']:.1%}")
    r = out["reach"]
    print(f"reach: {r['atlas_publications']}/{r['field_primary_reports_2005_2025']} "
          f"= {r['atlas_share_of_field']:.2%}; either database "
          f"{r['either_database_share_of_field']:.2%} "
          f"(one in {r['reports_per_curated_publication']:.0f})")


if __name__ == "__main__":
    main()
