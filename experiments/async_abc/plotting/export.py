"""Figure export utilities.

Every figure is saved as `.pdf` + `.png` plus optional `_data.csv` and
always a `_meta.json` with provenance fields (git hash, timestamp).
"""
import csv
import json
import subprocess
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


def save_figure(
    fig: matplotlib.figure.Figure,
    path_stem: Union[str, Path],
    data: Optional[Dict[str, List]] = None,
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

    Returns
    -------
    dict
        Mapping of ``{'pdf': Path, 'png': Path, 'meta': Path}`` (plus
        ``'csv': Path`` when *data* is provided).
    """
    stem = Path(path_stem)
    out: Dict[str, Path] = {}

    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    out["pdf"] = pdf_path
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
        "png": str(png_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    out["meta"] = meta_path

    import matplotlib.pyplot as plt
    plt.close(fig)

    return out
