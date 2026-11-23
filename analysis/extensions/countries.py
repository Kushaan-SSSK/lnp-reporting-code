"""Look up the laboratory country of each source publication in OpenAlex.

Lab country = country of the corresponding author's institution. Where
OpenAlex flags no corresponding author, the last author's institution is
used. Three publications have no affiliation data in OpenAlex; they are
assigned from the affiliations printed in the articles (MANUAL below).

Output: data/paper_countries.csv (one row per publication)
Run:    python analysis/extensions/countries.py   (needs internet access)
"""
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import DATA, load

MANUAL = {
    "10.1016/j.bbamem.2005.02.001": "CA",   # Protiva Biotherapeutics, Burnaby
    "10.1016/j.jconrel.2005.06.014": "CA",  # Protiva Biotherapeutics, Burnaby
    "10.1073/pnas.1209367109": "US",        # Novartis Vaccines, Cambridge MA
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lnp-audit"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def countries(authorship):
    found = {i.get("country_code") for i in authorship.get("institutions", [])}
    found |= set(authorship.get("countries", []) or [])
    return sorted(c for c in found if c)


def main():
    df = load()
    papers = df.groupby("group").agg(title=("paper_title", "first"),
                                     n_formulations=("lnp_id", "size")).reset_index()
    rows = []
    for _, p in papers.iterrows():
        doi = p.group
        if doi != "nan":
            d = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            d = d.replace("https://dx.doi.org/", "")
            w = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(d))
        else:
            w = get("https://api.openalex.org/works?per-page=1&search="
                    + urllib.parse.quote(p.title))["results"][0]
        au = w.get("authorships", [])
        corr = [a for a in au if a.get("is_corresponding")]
        lead = corr[0] if corr else (au[-1] if au else {})
        lab = ";".join(countries(lead)).split(";")[0]
        source = "openalex_corresponding" if corr else "openalex_last_author"
        if doi in MANUAL:
            lab, source = MANUAL[doi], "manual_from_article"
        rows.append(dict(paper_doi=doi, paper_title=p.title,
                         n_formulations=int(p.n_formulations), lab_country=lab,
                         country_source=source,
                         all_author_countries=";".join(sorted(
                             {c for a in au for c in countries(a)}))))
        time.sleep(0.15)
    pd.DataFrame(rows).to_csv(DATA / "paper_countries.csv", index=False)


if __name__ == "__main__":
    main()
