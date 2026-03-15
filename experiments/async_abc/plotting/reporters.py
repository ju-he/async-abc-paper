"""High-level plot reporters called by runner scripts.

Each function reads already-collected records (or CSVs) and produces the
standard set of figures for a given experiment type.
"""
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from .common import (
    archive_evolution_plot,
    compute_wasserstein,
    posterior_plot,
    scaling_plot,
    sensitivity_heatmap,
)
from .export import save_figure


def _param_names(records: List[ParticleRecord]) -> List[str]:
    for r in records:
        if r.params:
            return list(r.params.keys())
    return []


def _final_population(records: List[ParticleRecord]) -> List[ParticleRecord]:
    """Return particles from the lowest recorded tolerance (final population)."""
    accepted = [r for r in records if r.tolerance is not None]
    if not accepted:
        return records  # fall back to all records
    min_tol = min(r.tolerance for r in accepted)
    return [r for r in accepted if r.tolerance == min_tol]


def plot_posterior(records: List[ParticleRecord], output_dir: OutputDir) -> None:
    """Posterior histogram for each parameter from the final population."""
    final = _final_population(records)
    if not final:
        return
    for param in _param_names(final):
        samples = np.array([r.params[param] for r in final], dtype=float)
        stem = output_dir.plots / f"posterior_{param}"
        posterior_plot(samples, param_name=param, path_stem=stem)


def plot_archive_evolution(records: List[ParticleRecord], output_dir: OutputDir) -> None:
    """Mean tolerance vs. simulation step, averaged across replicates."""
    # Collect (step, tolerance) for accepted particles only
    by_step: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        if r.tolerance is not None:
            by_step[r.step].append(r.tolerance)

    if not by_step:
        return

    steps = sorted(by_step)
    mean_tols = [float(np.mean(by_step[s])) for s in steps]

    stem = output_dir.plots / "archive_evolution"
    archive_evolution_plot(
        sim_counts=np.array(steps, dtype=float),
        tolerances=np.array(mean_tols, dtype=float),
        path_stem=stem,
    )


def plot_scaling_summary(throughput_rows: List[Dict[str, Any]], output_dir: OutputDir) -> None:
    """Scaling + efficiency plot from throughput rows."""
    if not throughput_rows:
        return

    # Average throughput per n_workers
    by_n: Dict[int, List[float]] = defaultdict(list)
    for row in throughput_rows:
        n = int(row["n_workers"])
        t = float(row["throughput_sims_per_s"])
        by_n[n].append(t)

    if 1 not in by_n:
        return  # need baseline

    throughput_by_workers = {n: float(np.mean(ts)) for n, ts in sorted(by_n.items())}
    stem = output_dir.plots / "scaling"
    scaling_plot(throughput_by_workers, path_stem=stem)


def plot_sensitivity_summary(
    data_dir: Path,
    grid: Dict[str, List],
    output_dir: OutputDir,
) -> None:
    """Heatmap of mean final tolerance across a 2-parameter sensitivity grid.

    Uses the first two grid axes as rows/columns; additional axes are ignored.
    """
    axes = list(grid.keys())
    if len(axes) < 2:
        return

    row_key, col_key = axes[0], axes[1]
    row_vals = grid[row_key]
    col_vals = grid[col_key]

    # Build lookup: (row_val, col_val) → mean final tolerance
    tol_map: Dict = {}
    for csv_path in sorted(data_dir.glob("sensitivity_*.csv")):
        rows = _read_csv(csv_path)
        if not rows:
            continue
        tolerances = [
            float(r["tolerance"]) for r in rows
            if r.get("tolerance") not in ("", None)
        ]
        if not tolerances:
            continue
        mean_tol = float(np.mean(tolerances[-max(1, len(tolerances) // 10):]))

        # Parse variant name from filename stem
        stem_str = csv_path.stem[len("sensitivity_"):]  # strip prefix
        # stem_str looks like "k10_perturbation_scale0.4_scheduler_typeacceptance_rate"
        rv = _parse_variant_value(stem_str, row_key)
        cv = _parse_variant_value(stem_str, col_key)
        if rv is not None and cv is not None:
            tol_map[(rv, cv)] = mean_tol

    if not tol_map:
        return

    matrix = np.full((len(row_vals), len(col_vals)), np.nan)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            key = (rv, cv)
            if key in tol_map:
                matrix[i, j] = tol_map[key]

    row_labels = [str(v) for v in row_vals]
    col_labels = [str(v) for v in col_vals]
    stem = output_dir.plots / "sensitivity_heatmap"
    sensitivity_heatmap(matrix, row_labels=row_labels, col_labels=col_labels, path_stem=stem)


def plot_ablation_summary(
    data_dir: Path,
    variants: List[Dict[str, Any]],
    output_dir: OutputDir,
) -> None:
    """Bar chart comparing mean final tolerance across ablation variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_names = [v.get("name", f"v{i}") for i, v in enumerate(variants)]
    mean_tols = []

    for name in variant_names:
        csv_path = data_dir / f"ablation_{name}.csv"
        if not csv_path.exists():
            mean_tols.append(float("nan"))
            continue
        rows = _read_csv(csv_path)
        tolerances = [
            float(r["tolerance"]) for r in rows
            if r.get("tolerance") not in ("", None)
        ]
        if tolerances:
            mean_tols.append(float(np.mean(tolerances[-max(1, len(tolerances) // 10):])))
        else:
            mean_tols.append(float("nan"))

    fig, ax = plt.subplots(figsize=(max(5, len(variant_names) * 1.2), 4))
    x = np.arange(len(variant_names))
    ax.bar(x, mean_tols, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=30, ha="right")
    ax.set_ylabel("mean final tolerance ε")
    ax.set_title("Ablation comparison")
    fig.tight_layout()

    data = {"variant": variant_names, "mean_final_tolerance": mean_tols}
    stem = output_dir.plots / "ablation_comparison"
    save_figure(fig, stem, data=data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_variant_value(stem_str: str, key: str):
    """Extract the value for *key* from a variant name string like 'k10_...'."""
    # The variant stem encodes keys without underscores between key and value,
    # e.g. "k10", "perturbation_scale0.4", "scheduler_typeacceptance_rate"
    import re
    pattern = re.escape(key) + r"([^_]+(?:_[^0-9][^_]*)*)"
    m = re.search(pattern, stem_str)
    if not m:
        return None
    raw = m.group(1)
    # Try numeric first
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
