"""Standard figure types for the async-ABC paper experiments.

Each function creates a matplotlib figure, optionally annotates it, and
delegates persistence to :func:`~async_abc.plotting.export.save_figure`.
All functions return the dict produced by ``save_figure``.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .export import save_figure


# ---------------------------------------------------------------------------
# Posterior distribution plot
# ---------------------------------------------------------------------------

def posterior_plot(
    samples: np.ndarray,
    param_name: str,
    path_stem: Union[str, Path],
    true_value: Optional[float] = None,
    bins: int = 30,
) -> Dict[str, Path]:
    """KDE / histogram of posterior samples for a single parameter.

    Parameters
    ----------
    samples:
        1-D array of posterior samples.
    param_name:
        Name used for the x-axis label and figure title.
    path_stem:
        Destination path without extension.
    true_value:
        If provided, draws a vertical line at the true parameter value.
    bins:
        Number of histogram bins.

    Returns
    -------
    dict
        Paths produced by :func:`save_figure`.
    """
    samples = np.asarray(samples, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(samples, bins=bins, density=True, alpha=0.6, color="steelblue", label="posterior")
    if true_value is not None:
        ax.axvline(true_value, color="crimson", linewidth=1.5, linestyle="--", label="true")
        ax.legend(frameon=False)
    ax.set_xlabel(param_name)
    ax.set_ylabel("density")
    ax.set_title(f"Posterior: {param_name}")
    fig.tight_layout()

    data = {param_name: samples.tolist()}
    return save_figure(fig, path_stem, data=data)


# ---------------------------------------------------------------------------
# Scaling / efficiency plot
# ---------------------------------------------------------------------------

def scaling_plot(
    throughput_by_workers: Dict[int, float],
    path_stem: Union[str, Path],
) -> Dict[str, Path]:
    """Strong-scaling efficiency plot.

    Efficiency is defined as::

        efficiency(n) = throughput(n) / (n * throughput(1))

    Parameters
    ----------
    throughput_by_workers:
        Mapping ``{n_workers: simulations_per_second}``.
    path_stem:
        Destination path without extension.

    Returns
    -------
    dict
        Paths produced by :func:`save_figure`.
    """
    ns = sorted(throughput_by_workers.keys())
    ts = [throughput_by_workers[n] for n in ns]
    t1 = throughput_by_workers[1]
    efficiencies = [t / (n * t1) for n, t in zip(ns, ts)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax_t = axes[0]
    ax_t.plot(ns, ts, "o-", color="steelblue")
    ax_t.plot(ns, [t1 * n for n in ns], "--", color="grey", label="ideal")
    ax_t.set_xlabel("workers")
    ax_t.set_ylabel("throughput (sim/s)")
    ax_t.set_title("Throughput")
    ax_t.legend(frameon=False)

    ax_e = axes[1]
    ax_e.plot(ns, efficiencies, "s-", color="darkorange")
    ax_e.axhline(1.0, color="grey", linestyle="--")
    ax_e.set_ylim(0, 1.1)
    ax_e.set_xlabel("workers")
    ax_e.set_ylabel("efficiency")
    ax_e.set_title("Parallel efficiency")

    fig.tight_layout()

    data = {
        "n_workers": ns,
        "throughput": ts,
        "efficiency": efficiencies,
    }
    return save_figure(fig, path_stem, data=data)


# ---------------------------------------------------------------------------
# Archive / tolerance evolution plot
# ---------------------------------------------------------------------------

def archive_evolution_plot(
    sim_counts: np.ndarray,
    tolerances: np.ndarray,
    path_stem: Union[str, Path],
) -> Dict[str, Path]:
    """Plot tolerance schedule over simulation count.

    Parameters
    ----------
    sim_counts:
        1-D array of cumulative simulation counts (x-axis).
    tolerances:
        Corresponding tolerance values (y-axis).
    path_stem:
        Destination path without extension.

    Returns
    -------
    dict
        Paths produced by :func:`save_figure`.
    """
    sim_counts = np.asarray(sim_counts, dtype=float)
    tolerances = np.asarray(tolerances, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(sim_counts, tolerances, "o-", color="steelblue")
    ax.set_xlabel("simulations")
    ax.set_ylabel("tolerance ε")
    ax.set_title("Tolerance schedule")
    fig.tight_layout()

    data = {"sim_count": sim_counts.tolist(), "tolerance": tolerances.tolist()}
    return save_figure(fig, path_stem, data=data)


# ---------------------------------------------------------------------------
# Sensitivity heatmap
# ---------------------------------------------------------------------------

def sensitivity_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    path_stem: Union[str, Path],
) -> Dict[str, Path]:
    """Heatmap for sensitivity / ablation results.

    Parameters
    ----------
    data:
        2-D array of shape ``(len(row_labels), len(col_labels))``.
    row_labels, col_labels:
        Axis tick labels.
    path_stem:
        Destination path without extension.

    Returns
    -------
    dict
        Paths produced by :func:`save_figure`.
    """
    data = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels)), max(3, len(row_labels))))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.tight_layout()

    csv_data = {"row": row_labels}
    for j, col in enumerate(col_labels):
        csv_data[col] = data[:, j].tolist()
    return save_figure(fig, path_stem, data=csv_data)


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------

def compute_wasserstein(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """1-D Wasserstein (Earth Mover's) distance between two sample sets.

    Parameters
    ----------
    samples_a, samples_b:
        1-D arrays of samples from two distributions.

    Returns
    -------
    float
        Wasserstein-1 distance.
    """
    from scipy.stats import wasserstein_distance
    a = np.asarray(samples_a, dtype=float).ravel()
    b = np.asarray(samples_b, dtype=float).ravel()
    return float(wasserstein_distance(a, b))
