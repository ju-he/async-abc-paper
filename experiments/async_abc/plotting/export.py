"""Figure export utilities.

Every figure is saved as `.pdf` + `.png` plus optional `_data.csv` and
always a `_meta.json` with provenance fields (git hash, timestamp).
"""
import csv
import json
import shutil
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.figure


def get_git_hash() -> str:
    """Return the short HEAD git hash, or ``'unknown'`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _save_png_via_pdftoppm(pdf_path: Path, png_path: Path, dpi: int = 150) -> bool:
    """Rasterize *pdf_path* into *png_path* using pdftoppm when available."""
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return False

    try:
        subprocess.run(
            [
                pdftoppm,
                "-singlefile",
                "-png",
                "-r",
                str(int(dpi)),
                str(pdf_path),
                str(png_path.with_suffix("")),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return png_path.exists()


def _save_png_via_convert(pdf_path: Path, png_path: Path, dpi: int = 150) -> bool:
    """Rasterize *pdf_path* into *png_path* using ImageMagick convert when available."""
    convert = shutil.which("convert")
    if not convert:
        return False

    try:
        subprocess.run(
            [
                convert,
                "-density",
                str(int(dpi)),
                str(pdf_path),
                str(png_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return png_path.exists()


def _save_png_from_pdf(pdf_path: Path, png_path: Path, dpi: int = 150) -> bool:
    """Best-effort PDF-to-PNG rasterization that avoids Matplotlib's Agg path."""
    return _save_png_via_pdftoppm(pdf_path, png_path, dpi=dpi) or _save_png_via_convert(
        pdf_path, png_path, dpi=dpi
    )


def save_figure(
    fig: matplotlib.figure.Figure,
    path_stem: Union[str, Path],
    data: Optional[Dict[str, List]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
    """Save *fig* to ``<path_stem>.pdf``, ``<path_stem>.png``, and metadata.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    path_stem:
        Path without extension.  Parent directory must already exist.
    data:
        Optional dict mapping column names to lists of values, written as
        ``<path_stem>_data.csv``.  Pass ``None`` to skip.
    metadata:
        Optional extra metadata to merge into ``<path_stem>_meta.json``.

    Returns
    -------
    dict
        Mapping containing ``'pdf'`` and ``'meta'`` and, when PNG
        rasterization succeeds, ``'png'``. Includes ``'csv'`` when *data*
        is provided.
    """
    stem = Path(path_stem)
    out: Dict[str, Path] = {}

    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    png_written = _save_png_from_pdf(pdf_path, png_path, dpi=150)
    if not png_written:
        try:
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            png_written = png_path.exists()
        except Exception:
            pass
    if not png_written:
        warnings.warn(
            f"Could not rasterize {pdf_path.name} to PNG; keeping PDF only.",
            stacklevel=2,
        )
    out["pdf"] = pdf_path
    if png_written:
        out["png"] = png_path

    if data is not None:
        csv_path = Path(str(stem) + "_data.csv")
        columns = list(data.keys())
        rows = list(zip(*[data[c] for c in columns]))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        out["csv"] = csv_path

    meta_path = Path(str(stem) + "_meta.json")
    meta = {
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
        "pdf": str(pdf_path),
        "png": str(png_path) if png_written else None,
    }
    if metadata:
        meta.update(metadata)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    out["meta"] = meta_path

    import matplotlib.pyplot as plt
    plt.close(fig)

    return out
