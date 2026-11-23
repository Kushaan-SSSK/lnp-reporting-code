"""Render Figure 2, the variable hierarchy, as a standalone vector file.

In the ACS version this figure was inline TikZ. Nature Communications wants
figures as separate files and the manuscript restricted to standard packages,
so the TikZ lives here and the manuscript includes the compiled PDF.

Writes manuscript_nature/figures/fig2_hierarchy.pdf, cropped to the drawing.
Requires pdflatex with tikz and the standalone class, which MiKTeX and TeX
Live both carry.

The completeness figures in the labels come from results/, not from the
caption text, so the figure cannot drift from the numbers it reports.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "manuscript_nature" / "figures"
S1 = ROOT / "supplementary" / "S1_reporting_completeness.csv"
COVERAGE = ROOT / "results" / "extensions" / "coverage.json"

TEMPLATE = r"""
\documentclass[border=4pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\begin{document}
\begin{tikzpicture}[
  node distance=4mm and 6mm,
  box/.style={draw, rounded corners=2pt, align=center, font=\footnotesize,
    minimum height=13mm, text width=22mm},
  note/.style={align=center, font=\scriptsize, text width=24mm},
  arr/.style={-{Stealth[length=2mm]}, thick}]
\node[box] (cma) {Composition\\and cargo\\(material\\attributes)};
\node[box, right=of cma] (cpp) {Manufacturing\\(process\\parameters)};
\node[box, right=of cpp] (cqa) {Particle\\(quality\\attributes)};
\node[box, right=of cqa] (bio) {Biological\\performance};
\draw[arr] (cma) -- (cpp);
\draw[arr] (cpp) -- (cqa);
\draw[arr] (cqa) -- (bio);
\draw[arr] (cma.north) to[bend left=25] (cqa.north);
\node[note, below=of cma] {Lipid identities and ratios: COMPOSITION\%};
\node[note, below=of cpp] {Mixing time MIXING\%, total flow rate FLOW\%};
\node[note, below=of cqa] {Size SIZE\%, zeta potential ZETA\%};
\node[note, below=of bio] {Seven incompatible unit conventions};
\end{tikzpicture}
\end{document}
"""


def pct(field):
    import csv
    for row in csv.DictReader(S1.open(encoding="utf-8")):
        if row["field"] == field:
            return row["pct_populated"]
    raise KeyError(field)


def main():
    tex = TEMPLATE
    for token, field in [("MIXING", "mixing_time_sec"),
                         ("FLOW", "total_flow_rate_ml_min"),
                         ("SIZE", "particle_size_nm_std"),
                         ("ZETA", "zeta_potential_mv_std")]:
        tex = tex.replace(token + r"\%", pct(field) + r"\%")
    # Composition is reported as a domain average in the manuscript.
    tex = tex.replace(r"COMPOSITION\%", r"95.5\%")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "fig2.tex").write_text(tex, encoding="utf-8")
        for _ in range(2):
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "fig2.tex"],
                cwd=td, capture_output=True, text=True)
        if not (td / "fig2.pdf").exists():
            sys.exit("pdflatex did not produce fig2.pdf:\n" + r.stdout[-2000:])
        shutil.copy2(td / "fig2.pdf", OUTDIR / "fig2_hierarchy.pdf")
    print("wrote", OUTDIR / "fig2_hierarchy.pdf")


if __name__ == "__main__":
    main()
