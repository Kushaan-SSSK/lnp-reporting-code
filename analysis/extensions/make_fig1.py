"""Figure 1: growth of the LNP primary research literature, and the share of it
reached by curated databases.

Reads results/extensions/coverage.json, which coverage.py writes, so the figure
and the numbers quoted in the text come from one source. Nothing here is
hardcoded except the panel labels.

Outputs, into the manuscript figures directory:
  fig1_pubmed_trend.pdf    vector, for LaTeX
  fig1_pubmed_trend.tiff    1200 dpi, for ACS production
  fig1_pubmed_trend.eps    vector, alternate production format

ACS Molecular Pharmaceutics figure rules observed here:
  - double-column width 300-504 pt; this figure is 360 pt
  - lettering no smaller than 4.5 pt at final size; smallest here is 6 pt
  - line art at 1200 dpi for the raster export
  - no meaning carried by colour alone: the 2021-onward split is labelled, and
    panel (b) is a single-hue sequence read by position and direct label
  - greyscale-safe: the two bar tones differ in lightness, not only in hue

Run: python analysis/extensions/make_fig1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
COVERAGE = ROOT / "results" / "extensions" / "coverage.json"

# Destinations: every manuscript source tree that exists, so neither format
# can silently keep a stale figure.
DESTS = [ROOT / "manuscript" / "figures",
         ROOT / "manuscript_springer" / "figures",
         ROOT / "manuscript_nature" / "figures"]

# Standalone code release: no manuscript tree, so write beside the results.
if not any(d.parent.exists() for d in DESTS):
    DESTS = [ROOT / "figures"]

# Figure width in inches per target. ACS takes a 300-504 pt double column
# (360 pt = 5 in); Nature Communications specifies a 180 mm double column,
# which is 7.09 in, so the Nature copy is drawn at its own size rather than
# scaled in LaTeX.
WIDTH_IN = {"manuscript": 5.0, "manuscript_springer": 5.0,
            "manuscript_nature": 180 / 25.4}

# An ordinal pair: one hue, monotone light-to-dark, so the pre-2021 and
# 2021-onward bars are told apart by lightness and survive greyscale printing.
# Validated with the data-viz validator in --ordinal mode (monotone L, step gap
# >= 0.06, light end >= 2:1 on the surface, hue spread 5 degrees): all pass.
# The categorical check is the wrong one here and fails a two-step ramp by design.
BLUE_LIGHT = "#8bb8e8"
BLUE_DARK = "#1b4f8f"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel_a(ax, by_year: dict[str, int], share: float) -> None:
    years = sorted(int(y) for y in by_year)
    counts = [by_year[str(y)] for y in years]
    colors = [BLUE_DARK if y >= 2021 else BLUE_LIGHT for y in years]

    # 2px-equivalent gap between adjacent bars at this scale.
    ax.bar(years, counts, color=colors, width=0.78, linewidth=0)

    ax.set_xlabel("Publication year")
    ax.set_ylabel("PubMed records")
    ax.set_xticks([2005, 2009, 2013, 2017, 2021, 2025])
    ax.set_xlim(2003.8, 2026.2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)

    # Direct label instead of a legend: two tones, one of which is the point.
    peak = max(counts)
    ax.annotate(f"2021 onward:\n{share:.0f}% of records",
                xy=(2020.6, peak * 0.99), fontsize=7.5, color=INK,
                ha="right", va="top", linespacing=1.3)
    # Panel label outside the axes, so it cannot collide with the marks.
    ax.text(-0.20, 1.06, "(a)", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")


def panel_b(ax, rows: list[tuple[str, int]]) -> None:
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    y = list(range(len(rows)))[::-1]

    # Funnel: PubMed subsets in the sequential hue, database counts lighter, so
    # the two kinds of quantity are distinguishable in print and in greyscale.
    colors = [BLUE_DARK, BLUE_DARK, BLUE_DARK, BLUE_LIGHT, BLUE_LIGHT, BLUE_LIGHT]
    ax.barh(y, values, color=colors[:len(rows)], height=0.62, linewidth=0)

    for yi, v in zip(y, values):
        ax.text(v * 1.18, yi, f"{v:,}", va="center", ha="left",
                fontsize=7.5, color=INK)

    ax.set_xscale("log")
    ax.set_xlim(1, 40000)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Records or publications, 2005 to 2025 (log scale)")
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(axis="y", length=0)
    ax.text(-0.20, 1.06, "(b)", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")


def main() -> None:
    cov = json.loads(COVERAGE.read_text(encoding="utf-8"))
    pm = cov["pubmed"]
    share = pm["all_2021_2025"] / pm["all"] * 100

    rows = [
        ("All primary research\nreports", pm["all"]),
        ("Mention nucleic acid cargo", pm["nucleic_acid"]),
        ("Also mention particle\ncharacterization", pm["nucleic_acid_characterised"]),
        ("In LNP Atlas or LNPDB", cov["union"]),
        ("LNP Atlas", cov["atlas_publications"]),
        ("LNPDB", cov["lnpdb_publications"]),
    ]

    wrote = []
    for d in DESTS:
        if not d.parent.exists():
            continue
        width = WIDTH_IN.get(d.parent.name, 5.0)
        # Height tracks width so the aspect ratio, and therefore the effective
        # font size at final width, is unchanged between targets.
        fig, (ax_a, ax_b) = plt.subplots(
            2, 1, figsize=(width, 4.8 * width / 5.0),
            gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.46})
        panel_a(ax_a, pm["by_year"], share)
        panel_b(ax_b, rows)
        # Left margin carries the longest category label in panel (b); the
        # panel labels sit in that margin too.
        fig.subplots_adjust(left=0.345, right=0.955, top=0.94, bottom=0.095)

        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig1_pubmed_trend.pdf")
        fig.savefig(d / "fig1_pubmed_trend.eps")
        fig.savefig(d / "fig1_pubmed_trend.tiff", dpi=1200,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
        wrote.append(d)

    print(f"retrieved {cov['retrieved']}; 2021-onward share {share:.1f}%")
    for d in wrote:
        print(f"  wrote {d / 'fig1_pubmed_trend.pdf'} (+ .eps, .tiff)")
    if not wrote:
        raise SystemExit("no manuscript figures directory found")


if __name__ == "__main__":
    main()
