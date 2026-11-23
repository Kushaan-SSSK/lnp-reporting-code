"""How much of the LNP literature do the curated databases cover?

PubMed counts for 2005-2025 (publication date) for three nested queries,
and the overlap between LNP Atlas and LNPDB source publications by PMID.

Review and systematic review publication types are excluded from every
count, so the denominators describe primary research reports rather than
all indexed records. The unfiltered totals are emitted alongside, under
the `*_including_reviews` keys, so both framings are available.

Note: summing per-year PubMed counts overcounts, because a record whose
electronic and print dates fall in different years is returned for both
years. Totals here come from a single range query.

PubMed is a live index: counts shift as records are added and as
publication types are reclassified, so this file records the retrieval
date and a rerun will not reproduce it exactly.

LNPDB is read from the LNPDB GitHub repository (MIT licence):
https://github.com/evancollins1/LNPDB, data/LNPDB_for_LiON/LNPDB.csv
One of its 43 Publication_link values is a vendor product page, not a
publication, so it has 42 publications, all with PMIDs.

Output: results/extensions/coverage.json
Run:    python analysis/extensions/coverage.py   (needs internet access)
"""
import datetime as dt
import io
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import OUT, load

E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
LNPDB_URL = ("https://raw.githubusercontent.com/evancollins1/LNPDB/main/"
             "data/LNPDB_for_LiON/LNPDB.csv")
# Review and systematic review publication types are excluded so that every
# denominator counts primary research reports, which are the records that may
# contain a formulation to reuse. The unfiltered total is reported alongside.
NOREV = ' NOT review[pt] NOT "systematic review"[pt]'
BASE_ALL = '("lipid nanoparticle"[Title/Abstract] OR "lipid nanoparticles"[Title/Abstract])'
BASE = BASE_ALL + NOREV
NA = BASE + ' AND (mRNA OR siRNA OR "nucleic acid")'
CHAR = NA + (' AND ("encapsulation efficiency" OR PDI OR polydispersity'
             ' OR "zeta potential" OR "particle size")')
YEARS = range(2005, 2026)


def esearch(**params):
    time.sleep(0.4)
    q = dict(db="pubmed", retmode="json", **params)
    return json.load(urllib.request.urlopen(E + urllib.parse.urlencode(q), timeout=60))["esearchresult"]


def count(term, lo=2005, hi=2025):
    return int(esearch(term=term, rettype="count", datetype="pdat",
                       mindate=lo, maxdate=hi)["count"])


def main():
    res = {"pubmed": {"all": count(BASE), "nucleic_acid": count(NA),
                      "nucleic_acid_characterised": count(CHAR),
                      "all_2021_2025": count(BASE, 2021, 2025),
                      "all_including_reviews": count(BASE_ALL),
                      "all_2021_2025_including_reviews": count(BASE_ALL, 2021, 2025)},
           "retrieved": dt.date.today().isoformat(),
           "query": {"base": BASE_ALL, "review_filter": NOREV.strip()}}
    # Per-year series for Figure 1a. Previously absent, which is why the figure
    # could not be regenerated from the deposited data.
    res["pubmed"]["by_year"] = {str(y): count(BASE, y, y) for y in YEARS}
    res["pubmed"]["by_year_including_reviews"] = {
        str(y): count(BASE_ALL, y, y) for y in YEARS}
    atlas = load()
    dois = sorted({re.sub(r"^https?://(dx\.)?doi\.org/", "", d).strip()
                   for d in atlas.paper_doi.dropna().astype(str)})
    pmids = {}
    for d in dois:
        ids = esearch(term=d + "[doi]")["idlist"]
        pmids[d] = ids[0] if len(ids) == 1 else None
    atlas_pmids = {v for v in pmids.values() if v}
    lnpdb = pd.read_csv(io.BytesIO(urllib.request.urlopen(LNPDB_URL, timeout=120).read()),
                        low_memory=False)
    lnpdb_pmids = {str(int(x)) for x in lnpdb.Publication_PMID.dropna()}
    overlap = atlas_pmids & lnpdb_pmids
    res.update({
        "atlas_publications": int(atlas.group.nunique()),
        "atlas_with_pmid": len(atlas_pmids),
        "atlas_dois_without_pmid": [d for d, v in pmids.items() if not v],
        "lnpdb_publications": len(lnpdb_pmids),
        "overlap": len(overlap),
        "union": int(atlas.group.nunique()) + len(lnpdb_pmids) - len(overlap),
    })
    print(json.dumps(res, indent=1))
    (OUT / "coverage.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
