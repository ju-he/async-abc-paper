"""Standard figure types for the async-ABC paper experiments.

Each function creates a matplotlib figure, optionally annotates it, and
delegates persistence to :func:`~async_abc.plotting.export.save_figure`.
All functions return the dict produced by ``save_figure``.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def posterior_comparison_plot(
    method_samples: Dict[str, np.ndarray],
    param_name: str,
    path_stem: Union[str, Path],
    true_value: Optional[float] = None,
    bins: int = 30,
) -> Dict[str, Path]:
    """Overlaid posterior histograms comparing multiple methods."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    cmap = plt.get_cmap("tab10")
    for idx, (method, samples) in enumerate(sorted(method_samples.items())):
        samples = np.asarray(samples, dtype=float)
        if samples.size == 0:
            continue
        ax.hist(samples, bins=bins, density=True, alpha=0.45, color=cmap(idx % 10), label=method)
    if true_value is not None:
        ax.axvline(true_value, color="crimson", linewidth=1.5, linestyle="--", label="true")
    ax.legend(frameon=False, fontsize="small")
    ax.set_xlabel(param_name)
    ax.set_ylabel("density")
    ax.set_title(f"Posterior: {param_name}")
    fig.tight_layout()

    data: Dict[str, list] = {}
    for method, samples in sorted(method_samples.items()):
        data[method] = np.asarray(samples, dtype=float).tolist()
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

    _int_fmt = matplotlib.ticker.FuncFormatter(lambda x, _: str(int(x)) if x == int(x) else str(x))

    ax_t = axes[0]
    ax_t.plot(ns, ts, "o-", color="steelblue")
    ax_t.plot(ns, [t1 * n for n in ns], "--", color="grey", label="ideal")
    ax_t.set_xscale("log", base=2)
    ax_t.xaxis.set_major_formatter(_int_fmt)
    ax_t.set_xlabel("workers")
    ax_t.set_ylabel("throughput (sim/s)")
    ax_t.set_title("Throughput")
    ax_t.legend(frameon=False)

    ax_e = axes[1]
    ax_e.plot(ns, efficiencies, "s-", color="darkorange")
    ax_e.axhline(1.0, color="grey", linestyle="--")
    ax_e.set_ylim(0, 1.1)
    ax_e.set_xscale("log", base=2)
    ax_e.xaxis.set_major_formatter(_int_fmt)
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
    facet_labels: Optional[List[str]] = None,
    facet_row_labels: Optional[List[str]] = None,
    facet_col_labels: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Heatmap for sensitivity / ablation results.

    Parameters
    ----------
    data:
        2-D array of shape ``(len(row_labels), len(col_labels))`` or
        3-D array of shape ``(n_facets, len(row_labels), len(col_labels))``.
    row_labels, col_labels:
        Axis tick labels.
    path_stem:
        Destination path without extension.
    facet_labels:
        Optional labels for the first axis of 3-D data.

    Returns
    -------
    dict
        Paths produced by :func:`save_figure`.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(col_labels)), max(3.5, 0.9 * len(row_labels))))
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("mean final tolerance", rotation=270, labelpad=14)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        fig.tight_layout(pad=1.0)

        csv_data = {"row": row_labels}
        for j, col in enumerate(col_labels):
            csv_data[col] = data[:, j].tolist()
        return save_figure(fig, path_stem, data=csv_data)

    if data.ndim == 4:
        row_facets, col_facets = data.shape[:2]
        facet_row_labels = facet_row_labels or [f"row_facet_{i}" for i in range(row_facets)]
        facet_col_labels = facet_col_labels or [f"col_facet_{j}" for j in range(col_facets)]
        fig, axes = plt.subplots(
            row_facets,
            col_facets,
            figsize=(max(4.0, 1.1 * len(col_labels)) * col_facets + 0.8, max(3.2, 0.9 * len(row_labels)) * row_facets + 0.4),
            squeeze=False,
        )
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        for row_idx in range(row_facets):
            for col_idx in range(col_facets):
                ax = axes[row_idx, col_idx]
                im = ax.imshow(data[row_idx, col_idx], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(
                    f"tol x {facet_row_labels[row_idx]} | scheduler={facet_col_labels[col_idx]}",
                    fontsize=10,
                )
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, rotation=45, ha="right")
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels)
                if row_idx == row_facets - 1:
                    ax.set_xlabel("perturbation_scale")
                if col_idx == 0:
                    ax.set_ylabel("k")
        fig.subplots_adjust(wspace=0.30, hspace=0.42, bottom=0.16, right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel("mean final tolerance", rotation=270, labelpad=14)

        csv_data = {
            "tol_init_multiplier": [],
            "scheduler_type": [],
            "k": [],
            "perturbation_scale": [],
            "value": [],
        }
        for facet_row_idx, facet_row_label in enumerate(facet_row_labels):
            for facet_col_idx, facet_col_label in enumerate(facet_col_labels):
                for row_idx, row_label in enumerate(row_labels):
                    for col_idx, col_label in enumerate(col_labels):
                        csv_data["tol_init_multiplier"].append(str(facet_row_label))
                        csv_data["scheduler_type"].append(str(facet_col_label))
                        csv_data["k"].append(str(row_label))
                        csv_data["perturbation_scale"].append(str(col_label))
                        csv_data["value"].append(float(data[facet_row_idx, facet_col_idx, row_idx, col_idx]))
        return save_figure(fig, path_stem, data=csv_data)

    if data.ndim != 3:
        raise ValueError("sensitivity_heatmap expects 2-D, 3-D, or 4-D data.")

    n_facets = data.shape[0]
    facet_labels = facet_labels or [f"facet_{i}" for i in range(n_facets)]
    fig, axes = plt.subplots(
        1,
        n_facets,
        figsize=(max(4.0, 1.1 * len(col_labels)) * n_facets + 0.7, max(3.4, 0.9 * len(row_labels))),
        squeeze=False,
    )
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    for idx in range(n_facets):
        ax = axes[0, idx]
        im = ax.imshow(data[idx], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(str(facet_labels[idx]), fontsize=10)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
    fig.subplots_adjust(wspace=0.30, bottom=0.20, right=0.88)
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("mean final tolerance", rotation=270, labelpad=14)

    csv_data = {"facet": [], "row": [], "col": [], "value": []}
    for facet_idx, facet_label in enumerate(facet_labels):
        for row_idx, row_label in enumerate(row_labels):
            for col_idx, col_label in enumerate(col_labels):
                csv_data["facet"].append(str(facet_label))
                csv_data["row"].append(str(row_label))
                csv_data["col"].append(str(col_label))
                csv_data["value"].append(float(data[facet_idx, row_idx, col_idx]))
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


# ---------------------------------------------------------------------------
# Phase 3 figures
# ---------------------------------------------------------------------------

def gantt_plot(records, ax=None):
    """Horizontal worker timeline using sim_start_time/sim_end_time metadata."""
    timed = [
        record for record in records
        if record.worker_id is not None
        and record.sim_start_time is not None
        and record.sim_end_time is not None
    ]
    created_fig = ax is None
    if ax is None:
        lane_count = len({(int(record.replicate), str(record.worker_id)) for record in timed})
        fig_height = min(10.0, max(3.5, 0.6 * max(1, lane_count)))
        fig, ax = plt.subplots(figsize=(7, fig_height))
    else:
        fig = ax.figure

    if not timed:
        ax.set_title("Worker timeline")
        ax.set_xlabel("wall-clock time")
        ax.set_ylabel("worker")
        return fig

    def _worker_sort_key(value):
        try:
            return (0, int(str(value)))
        except (TypeError, ValueError):
            return (1, str(value))

    replicates = sorted({int(record.replicate) for record in timed})
    show_replicate = len(replicates) > 1
    lanes = sorted(
        {(int(record.replicate), str(record.worker_id)) for record in timed},
        key=lambda item: (item[0], _worker_sort_key(item[1])),
    )
    worker_to_y = {lane: idx for idx, lane in enumerate(lanes)}
    methods = sorted({record.method for record in timed})
    cmap = plt.get_cmap("tab10")
    method_colors = {method: cmap(i % 10) for i, method in enumerate(methods)}

    for record in timed:
        lane = (int(record.replicate), str(record.worker_id))
        ax.barh(
            worker_to_y[lane],
            float(record.sim_end_time) - float(record.sim_start_time),
            left=float(record.sim_start_time),
            height=0.7,
            color=method_colors[record.method],
            alpha=0.85,
        )

    worker_ticks = list(worker_to_y.values())
    worker_labels = [
        f"rep {replicate} | worker {worker_id}" if show_replicate else f"worker {worker_id}"
        for replicate, worker_id in worker_to_y.keys()
    ]
    if len(worker_labels) > 20:
        step = max(1, int(np.ceil(len(worker_labels) / 20.0)))
        worker_labels = [label if idx % step == 0 else "" for idx, label in enumerate(worker_labels)]
    ax.set_yticks(worker_ticks)
    ax.set_yticklabels(worker_labels)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("replicate | worker" if show_replicate else "worker")
    ax.set_title("Worker timeline")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=method_colors[method], alpha=0.85)
        for method in methods
    ]
    ax.legend(handles, methods, frameon=False, loc="best")
    if created_fig:
        fig.tight_layout()
    return fig


def idle_fraction_plot(
    sigma_levels: List[float],
    idle_fractions: List[float],
    path_stem: Union[str, Path],
) -> Dict[str, Path]:
    """Bar chart of worker idle fraction per sigma level."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = [f"σ={s}" for s in sigma_levels]
    ax.bar(labels, idle_fractions, color="steelblue", alpha=0.75)
    ax.set_xlabel("heterogeneity (σ)")
    ax.set_ylabel("idle fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Worker idle fraction")
    fig.tight_layout()
    data = {"sigma": [float(s) for s in sigma_levels], "idle_fraction": idle_fractions}
    return save_figure(fig, path_stem, data=data)


def throughput_over_time_plot(
    time_bins: Dict[str, np.ndarray],
    throughput_bins: Dict[str, np.ndarray],
    path_stem: Union[str, Path],
) -> Dict[str, Path]:
    """Line plot of simulation throughput over time, one line per condition."""
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(sorted(time_bins)):
        ax.plot(time_bins[label], throughput_bins[label], "o-",
                color=cmap(idx % 10), label=label, markersize=3)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("completions / bin")
    ax.set_title("Throughput over time")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    data: Dict[str, list] = {}
    for label in sorted(time_bins):
        data[f"{label}_time"] = time_bins[label].tolist()
        data[f"{label}_throughput"] = throughput_bins[label].tolist()
    return save_figure(fig, path_stem, data=data)


def idle_fraction_comparison_plot(
    sigma_levels: List[float],
    idle_by_sigma_replicate: Dict[float, List[float]],
    path_stem: Union[str, Path],
) -> Dict[str, Path]:
    """Idle fraction vs. sigma with per-replicate points and mean line."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sigmas = sorted(sigma_levels)
    means = []
    for s in sigmas:
        vals = idle_by_sigma_replicate[s]
        ax.scatter([s] * len(vals), vals, s=25, alpha=0.5, color="steelblue")
        means.append(float(np.mean(vals)) if vals else float("nan"))
    ax.plot(sigmas, means, "o-", color="darkorange", label="mean", linewidth=2)
    ax.set_xlabel("heterogeneity (σ)")
    ax.set_ylabel("idle fraction")
    ax.set_ylim(0, max(1, max(means) * 1.1) if means else 1)
    ax.set_title("Idle fraction vs. heterogeneity")
    ax.legend(frameon=False)
    fig.tight_layout()
    data = {"sigma": sigmas, "mean_idle_fraction": means}
    return save_figure(fig, path_stem, data=data)


def quality_vs_time_plot(quality_df: pd.DataFrame, ax=None):
    """Deprecated wrapper for the wall-time posterior quality diagnostic."""
    return posterior_quality_plot(quality_df, axis_kind="wall_time", ax=ax)


def _state_kind_short(state_kind: str) -> str:
    """Compact label for ``state_kind`` used in plot legends."""
    _MAP = {
        "archive_reconstruction": "archive",
        "generation_population": "generation",
        "accepted_prefix": "prefix",
    }
    return _MAP.get(state_kind, state_kind)


def _build_state_kind_map(quality_df) -> dict:
    """Return ``{method: short_state_kind}`` from a quality DataFrame."""
    if "state_kind" not in quality_df.columns:
        return {}
    return (
        quality_df.groupby("method")["state_kind"]
        .first()
        .map(_state_kind_short)
        .to_dict()
    )


def posterior_quality_plot(quality_df: pd.DataFrame, axis_kind: str, ax=None):
    """Posterior quality curve for a configurable x-axis."""
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    title = {
        "wall_time": "Posterior quality vs. wall-clock time",
        "posterior_samples": "Posterior quality vs. posterior samples",
        "attempt_budget": "Posterior quality vs. simulation attempts",
    }[axis_kind]
    xlabel = {
        "wall_time": "wall-clock time",
        "posterior_samples": "posterior samples",
        "attempt_budget": "simulation attempts",
    }[axis_kind]

    if quality_df.empty:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("wasserstein")
        return fig

    sk_map = _build_state_kind_map(quality_df)

    for (method, replicate), group in quality_df.groupby(["method", "replicate"], sort=True):
        group = group.sort_values("axis_value")
        sk_suffix = f" [{sk_map[method]}]" if method in sk_map else ""
        label = (
            f"{method}{sk_suffix}"
            if quality_df["replicate"].nunique() == 1
            else f"{method}{sk_suffix} (rep {replicate})"
        )
        x = group["axis_value"].to_numpy(dtype=float)
        y = group["wasserstein"].to_numpy(dtype=float)
        semantics = group["time_semantics"].dropna().unique().tolist()
        is_generation_snapshot = semantics and all(value == "generation_end" for value in semantics)
        if not is_generation_snapshot and len(group) > 1:
            ax.step(x, y, where="post", linewidth=1.2, alpha=0.7, label=label)
            ax.scatter(x, y, s=18, marker="o")
        else:
            ax.scatter(x, y, s=24, marker="s", label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("wasserstein")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    if created_fig:
        fig.tight_layout()
    return fig


def threshold_summary_plot(
    summary_df: pd.DataFrame,
    axis_kind: str,
    ax=None,
    *,
    include_replicates: bool = False,
):
    """Point summary of time/budget required to reach a target posterior quality."""
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    xlabel = {
        "wall_time": "wall-clock time to target",
        "posterior_samples": "posterior samples to target",
        "attempt_budget": "simulation attempts to target",
    }[axis_kind]
    title = {
        "wall_time": "Wall-clock time to target posterior quality",
        "posterior_samples": "Posterior samples to target quality",
        "attempt_budget": "Simulation attempts to target posterior quality",
    }[axis_kind]

    if summary_df.empty:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("method")
        return fig

    methods = list(dict.fromkeys(summary_df["method"].tolist()))
    y_positions = {method: idx for idx, method in enumerate(methods)}
    for method, group in summary_df.groupby("method", sort=False):
        valid = group.loc[group["axis_value_to_threshold"].notna()].copy()
        if valid.empty:
            continue
        x = valid["axis_value_to_threshold"].to_numpy(dtype=float)
        y = np.full(len(valid), y_positions[method], dtype=float)
        if include_replicates:
            ax.scatter(x, y, s=24, alpha=0.35, label=method)
        mean_x = float(np.mean(x))
        if len(x) >= 2:
            from scipy.stats import t

            sem = float(np.std(x, ddof=1) / np.sqrt(len(x)))
            half_width = float(t.ppf(0.975, len(x) - 1) * sem)
        else:
            half_width = float("nan")
        xerr = None if not np.isfinite(half_width) else np.array([[half_width], [half_width]])
        ax.errorbar(
            [mean_x],
            [y_positions[method]],
            xerr=xerr,
            fmt="D",
            markersize=7,
            capsize=4 if xerr is not None else 0,
            markeredgecolor="black",
            linewidth=1.2,
            label=None if include_replicates else method,
        )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("method")
    ax.set_title(title)
    if len(methods) <= 8:
        ax.legend(frameon=False, fontsize=8)
    if created_fig:
        fig.tight_layout()
    return fig


def corner_plot(records, param_names: List[str], true_params: Optional[Dict] = None,
                method_labels: Optional[List[str]] = None, ax=None):
    """Pairwise posterior scatter/hist grid, optionally color-coded by method."""
    if not param_names:
        fig, axis = plt.subplots(figsize=(4, 3))
        axis.set_title("Posterior corner")
        return fig

    if ax is None:
        fig, axes = plt.subplots(len(param_names), len(param_names), figsize=(3 * len(param_names), 3 * len(param_names)))
    else:
        axes = ax
        fig = axes.figure if hasattr(axes, "figure") else axes[0, 0].figure

    if len(param_names) == 1:
        axes = np.array([[axes]])
    elif not isinstance(axes, np.ndarray):
        axes = np.asarray(axes)

    if method_labels:
        cmap = plt.get_cmap("tab10")
        method_colors = {m: cmap(i % 10) for i, m in enumerate(method_labels)}
        groups = {m: [r for r in records if r.method == m] for m in method_labels}
    else:
        groups = {"_all": records}
        method_colors = {"_all": "steelblue"}

    for row_idx, y_name in enumerate(param_names):
        for col_idx, x_name in enumerate(param_names):
            axis = axes[row_idx, col_idx]
            for method, group_records in groups.items():
                color = method_colors[method]
                x = np.asarray([r.params[x_name] for r in group_records if x_name in r.params], dtype=float)
                if row_idx == col_idx:
                    axis.hist(x, bins=20, density=True, color=color, alpha=0.45,
                              label=method if method != "_all" else None)
                else:
                    y = np.asarray([r.params[y_name] for r in group_records if y_name in r.params], dtype=float)
                    axis.scatter(x, y, s=12, alpha=0.5, color=color,
                                 label=method if method != "_all" else None)
            if true_params:
                if x_name in true_params:
                    axis.axvline(float(true_params[x_name]), color="crimson", linestyle="--", linewidth=0.9)
                if row_idx != col_idx and y_name in true_params:
                    axis.axhline(float(true_params[y_name]), color="crimson", linestyle="--", linewidth=0.9)

            if row_idx == len(param_names) - 1:
                axis.set_xlabel(x_name)
            if col_idx == 0:
                axis.set_ylabel(y_name)

    if method_labels:
        handles = [plt.Rectangle((0, 0), 1, 1, color=method_colors[m], alpha=0.45) for m in method_labels]
        fig.legend(handles, method_labels, loc="upper right", frameon=False, fontsize="small")
    fig.suptitle("Posterior corner", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def tolerance_trajectory_plot(trajectory_df: pd.DataFrame, ax=None):
    """Line plot of tolerance versus wall-clock time."""
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if trajectory_df.empty:
        ax.set_title("Tolerance trajectory")
        ax.set_xlabel("wall-clock time")
        ax.set_ylabel("tolerance")
        return fig

    for (method, replicate), group in trajectory_df.groupby(["method", "replicate"], sort=True):
        group = group.sort_values("wall_time")
        label = method if trajectory_df["replicate"].nunique() == 1 else f"{method} (rep {replicate})"
        ax.plot(group["wall_time"], group["tolerance"], marker="o", label=label)

    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("tolerance")
    ax.set_yscale("log")
    ax.set_title("Tolerance trajectory")
    ax.legend(frameon=False, fontsize=8)
    if created_fig:
        fig.tight_layout()
    return fig
