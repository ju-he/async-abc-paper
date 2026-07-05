"""Single source of truth for paper-figure styling (review II.7).

Every paper-facing figure script (``experiments/scripts/make_*_fig.py``)
imports this module instead of rolling its own ``rcParams``/color constants.
The rules it encodes:

* **Figures are drawn at final print width.** sn-jnl's ``\\textwidth`` is
  372 pt = 5.147 in; a figure included at ``0.6\\linewidth`` is therefore
  drawn 3.09 in wide via :func:`fig_size` so that an 8 pt label prints at
  8 pt. Never draw 10 in wide and let LaTeX downscale — that is how the
  previous figure set ended up with 4–6 pt effective fonts.
* **TrueType fonts** (``pdf.fonttype = 42``): Springer production rejects
  the Type 3 fonts matplotlib embeds by default.
* **One palette, one label set.** Okabe–Ito colors (colorblind- and
  grayscale-safe) with fixed method→color/label/marker assignments, so the
  asynchronous method is the same blue in every figure.
* **Serif/STIX math** to match the sn-mathphys text font, so θ₁ in a figure
  looks like θ₁ in its caption.
* **No in-figure titles** — journals want captions only. Use
  :func:`panel_tag` for "(a)"/"(b)" corner tags.
* **Vendored figure data**: :func:`save_paper_figure` writes the PDF into
  the LaTeX figures directory and the exact plotted data as CSVs under
  ``experiments/data/paper_figures/<name>/`` so every figure regenerates
  from the repository alone (Declarations promise), independent of
  purge-prone cluster scratch.
"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib

# sn-jnl \textwidth = 372 pt at 72.27 pt/in.
TEXTWIDTH_IN = 372.0 / 72.27

# Okabe–Ito. Async and sync are additionally distinguished by marker and
# linestyle everywhere (grayscale print, CVD redundancy).
COLORS = {
    "async": "#0072B2",  # blue
    "sync": "#D55E00",  # vermillion
    "rejection": "#009E73",  # bluish green
    "reference": "#000000",  # truth/reference lines
    "neutral": "#999999",  # grid variants, secondary annotations
}

LABELS = {
    "async": "Asynchronous (ours)",
    "sync": "Synchronous baseline",
    "rejection": "Rejection ABC",
}

MARKERS = {"async": "o", "sync": "s", "rejection": "^"}
LINESTYLES = {"async": "-", "sync": "--", "rejection": ":"}

RC = {
    # TrueType instead of Type 3 — required by Springer production.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Match the sn-mathphys serif text and math fonts.
    "font.family": "serif",
    "mathtext.fontset": "stix",
    # Sizes are FINAL print sizes; figures are drawn at print width.
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Repository-anchored output locations.
_REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = _REPO_ROOT / "latex" / "sn-article-template" / "figures"
DATA_DIR = _REPO_ROOT / "experiments" / "data" / "paper_figures"


def apply() -> None:
    """Install the paper rcParams. Call once, before creating any figure."""
    matplotlib.rcParams.update(RC)


def fig_size(width_frac: float, aspect: float = 0.68) -> tuple:
    """Figure size (inches) for a ``width_frac``·``\\linewidth`` figure.

    ``aspect`` is height/width; 0.68 suits single-panel line plots. Draw at
    this size and include with the same fraction in LaTeX so fonts print at
    their nominal point size.
    """
    width = TEXTWIDTH_IN * float(width_frac)
    return (width, width * float(aspect))


def panel_tag(ax, text: str) -> None:
    """Put an "(a)"-style tag in the axes corner (instead of a title)."""
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def save_paper_figure(
    fig,
    name: str,
    data: Optional[Dict[str, "object"]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
    """Save a paper figure PDF plus its vendored input data.

    Parameters
    ----------
    fig:
        The matplotlib figure.
    name:
        Figure basename without extension, e.g. ``"fig_straggler_throughput"``.
    data:
        Optional mapping ``csv_name -> pandas.DataFrame`` with the exact data
        plotted; each frame is committed to
        ``experiments/data/paper_figures/<name>/<csv_name>.csv``.
    metadata:
        Extra provenance merged into the ``_meta.json`` written next to the
        vendored data (git hash and timestamp are always included).

    Returns
    -------
    dict
        ``{"pdf": Path, ...}`` — the PDF lands in the LaTeX figures dir.
    """
    from .export import save_figure

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    data_dir = DATA_DIR / name
    if data:
        data_dir.mkdir(parents=True, exist_ok=True)
        for csv_name, frame in data.items():
            csv_path = data_dir / f"{csv_name}.csv"
            frame.to_csv(csv_path, index=False)
            out[f"csv:{csv_name}"] = csv_path

    # save_figure writes <stem>.pdf/.png/_meta.json; keep the PDF in the
    # LaTeX tree and the provenance next to the vendored data.
    saved = save_figure(
        fig,
        FIGURES_DIR / name,
        metadata={
            "figure": name,
            "vendored_data_dir": str(data_dir) if data else None,
            **(metadata or {}),
        },
    )
    out.update(saved)
    return out
