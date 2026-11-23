# Incomplete manufacturing parameter reporting in lipid nanoparticle literature

Analysis code and derived data for a secondary analysis of the LNP Atlas, a
curated database of 1,092 lipid nanoparticle formulations drawn from 63
publications.

The question: does the published record contain enough information to predict
particle attributes across laboratories? The answer it arrives at is no, and
the reason is what the literature omits rather than what the models can do.

## What it finds

Composition is reported well and manufacturing is not. Lipid identities and
molar ratios are present for 95.5% of formulations; mixing time carries a value
for 13.3%, sterilization for 27.0% and total flow rate for 31.7%. The mixing
device is named for 72.6% of formulations but quantified for 5.5%.

That asymmetry sets a ceiling on prediction. Splitting the data at random gives
coefficients of determination up to 0.585. Splitting so that no publication
appears in both training and test data drops the same models on the same data to
0.097 or below, across five model families and three targets. A synthetic
positive control passes through the identical pipeline at 0.9999 or above, and
95 to 98% of formulations sit inside the applicability domain, so the limit is
neither the pipeline nor extrapolation. Source publication alone accounts for 46
to 58% of variance.

## Layout

```
analysis/
  parse_atlas.py          Atlas CSV -> data/atlas_parsed.csv, atlas_coverage.csv
  features.py             feature construction; isolates two source-data defects
  lnp_analysis.py         pipeline, with --selftest positive and negative controls
  run_real_analysis.py    grouped CV, leave-one-study-out, ICC(1), leverage
  extensions/
    common.py             shared setup; reuses the main pipeline's folds exactly
    countries.py          lab country per publication, from OpenAlex
    country_analysis.py   country versus study identity as variance sources
    coverage.py           PubMed counts and Atlas/LNPDB overlap (needs network)
    representativeness.py whether the 63 publications resemble the field
    sensitivity.py        one-hot encoding and range-midpoint checks
    stacking.py           stacked ensemble across the three splits
    make_fig1.py          Figure 1
    make_fig2.py          Figure 2
    make_supplementary.py Supplementary Tables S2, S7-S11
protocol/                 extraction codebook for a future de novo study
data/                     parsed and derived data (see Data below)
results/                  every number the analysis produces, as JSON
supplementary/            Supplementary Tables S1-S11
```

## Reproducing

Needs Python with numpy, pandas, scipy, scikit-learn and matplotlib.

```
python analysis/lnp_analysis.py --selftest      # controls, no source data needed
python analysis/parse_atlas.py                  # needs the Atlas CSV, see Data
python analysis/run_real_analysis.py
python analysis/extensions/representativeness.py
python analysis/extensions/make_supplementary.py
```

`coverage.py` and `countries.py` query PubMed and OpenAlex and so depend on live
services; `results/extensions/coverage.json` holds the retrieved counts and the
retrieval date, and everything downstream reads that file rather than the network.

All partitioning uses the fixed seed 20260904, and model hyperparameters are
fixed rather than tuned, so results are reproducible run to run. Note that the
counts in `coverage.json` are not: PubMed is a live index.

## Data

`data/atlas_parsed.csv` is the Atlas file plus eleven `proc_*` columns parsed
out of its free-text `synthesis_info` field. No original values were changed.

The source database is not redistributed here. Get it from Zenodo:

- LNP Atlas, https://doi.org/10.5281/zenodo.17243733, CC BY 4.0
  (Song, S., Baek, J. & Seo, S.)
- place `LNP_Atlas_DB_202509_v1.csv` in `data/`

The released file this analysis used has SHA-256 `624ba842...bbe83cb0` and was
downloaded on 4 September 2026. It is read as CP949, which is the encoding it
ships in.

## Two things worth knowing about the source data

Both are handled in `features.py` rather than silently worked around, because
both constrain what any model built on this data can learn.

The molar-ratio column does not use a consistent component order: the smallest
value sits at index 1 in 58.7% of records and index 3 in 33.7%, so position does
not identify the lipid. The four mole percentages therefore enter the model
sorted by magnitude, which is invariant to the ambiguity but means a model
cannot know which lipid a given percentage belongs to.

Total flow rate is nominally in mL/min, but only 2 of its 343 values state a
unit and both state mL/h, so recorded values span 0.06 to 17,000. The column is
modelled on a logarithmic scale as recorded rather than converted on a guess.

## License

CC BY 4.0. See LICENSE.

The parsed dataset is redistributed under CC BY 4.0, the same license as the
source database, with attribution to the Atlas authors.
