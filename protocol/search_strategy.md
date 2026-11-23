# Search Strategy (PRISMA-S)

Record the date each source was last searched, and paste each strategy **exactly as run**. Do not paraphrase a search string after the fact.

## Scoping counts already obtained

Run against the NCBI E-utilities API on **4 September 2026**. These are title/abstract-restricted and are a floor, not a ceiling.

| Query | Count |
|---|---|
| `("lipid nanoparticle"[Tiab] OR "lipid nanoparticles"[Tiab])` | 11,793 |
| above `AND (mRNA OR siRNA OR "nucleic acid")` | 4,527 |
| above `AND (encapsulation efficiency OR polydispersity OR zeta potential OR particle size)` | 384 |
| `"lipid nanoparticle*"[Tiab] AND 2019[dp]` | 454 |
| `"lipid nanoparticle*"[Tiab] AND 2021[dp]` | 842 |
| `"lipid nanoparticle*"[Tiab] AND 2024[dp]` | 1,570 |
| `"lipid nanoparticle*"[Tiab] AND 2025[dp]` | 2,213 |

Growth 2019 to 2025 is 4.9x. Roughly 60% of the LNP literature has appeared since 2021. This motivates the temporal analysis and dates the "standards should be set now" argument.

## 1. PubMed / MEDLINE (platform: PubMed)

Date last searched: ________

```
("lipid nanoparticle"[Tiab] OR "lipid nanoparticles"[Tiab] OR "LNP"[Tiab] OR "LNPs"[Tiab]
 OR "ionizable lipid*"[Tiab] OR "cationic lipid*"[Tiab] OR "lipoplex*"[Tiab]
 OR "SNALP"[Tiab] OR "stable nucleic acid lipid particle*"[Tiab])
AND ("mRNA"[Tiab] OR "siRNA"[Tiab] OR "saRNA"[Tiab] OR "self-amplifying RNA"[Tiab]
 OR "plasmid DNA"[Tiab] OR "pDNA"[Tiab] OR "nucleic acid*"[Tiab] OR "oligonucleotide*"[Tiab])
AND ("encapsulation efficiency"[Tiab] OR "polydispersity"[Tiab] OR "zeta potential"[Tiab]
 OR "particle size"[Tiab] OR "hydrodynamic diameter"[Tiab])
AND 2018:2025[dp]
NOT (review[pt] OR editorial[pt] OR comment[pt] OR letter[pt])
```

## 2. Scopus (platform: Elsevier)

Date last searched: ________

```
TITLE-ABS-KEY(("lipid nanoparticle*" OR "LNP" OR "ionizable lipid*" OR "SNALP")
AND ("mRNA" OR "siRNA" OR "nucleic acid" OR "plasmid DNA"))
AND PUBYEAR > 2017 AND PUBYEAR < 2026
AND DOCTYPE(ar) AND LANGUAGE(english)
```

## 3. Web of Science Core Collection (platform: Clarivate)

Date last searched: ________

```
TS=(("lipid nanoparticle*" OR "LNP*" OR "ionizable lipid*")
AND ("mRNA" OR "siRNA" OR "nucleic acid*"))
AND DT=(Article) AND PY=(2018-2025)
```

## 4. Embase (platform: Elsevier)

Date last searched: ________

```
('lipid nanoparticle'/exp OR 'lipid nanoparticle*':ti,ab OR 'ionizable lipid*':ti,ab)
AND ('mrna':ti,ab OR 'sirna':ti,ab OR 'nucleic acid':ti,ab)
AND [article]/lim AND [english]/lim AND [2018-2025]/py
```

## Supplementary sources

- **Backward citation chasing** on all included reviews and on MIRIBEL.
- **LNP Atlas** (Zenodo 10.5281/zenodo.17243732, CC BY 4.0) — reuse with attribution; record which rows were reused rather than independently extracted.
- **LNPDB** (lnpdb.molcube.com, REST API).
- **Patents** — reported separately with their own flow diagram. CPC: `(A61K9/127* OR C12N15/88) AND (A61K47/* OR A61K31/7105)`. Deduplicate to simple families via DOCDB/INPADOC.

## Screening

Two stages: title/abstract, then full text. Dual independent screening on a random 20% subset; report **Cohen's kappa and raw percent agreement together** (kappa paradoxes are guaranteed for items reported by ~98% or ~3% of papers). Calibrate on 50 records first and iterate until kappa >= 0.80. Resolve disagreements by discussion, with a third reviewer for deadlocks.

Check every included record against the **Retraction Watch Database**; record the date and re-run before final acceptance.

## Eligibility

**Include:** primary research articles, 2018-2025, English, indexed in Web of Science Core Collection and/or Scopus, reporting de novo preparation of nucleic-acid-carrying lipid nanoparticles with at least one quantitative physicochemical measurement.

**Exclude:** reviews, editorials, comments, letters; preprints; conference abstracts and theses (length limits make reporting completeness non-comparable — state this justification explicitly); liposomes without an ionizable or cationic lipid; small-molecule-only lipid carriers; purely computational papers with no new formulation.

Define the corpus by **indexing** rather than by predatory-journal judgement. It is objective, reproducible, and avoids a contested exercise.
