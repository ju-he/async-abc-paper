"""Posterior convergence summaries.

This module computes posterior quality curves and time-to-threshold metrics
for comparing ABC inference methods.  The central metric is the 1-Wasserstein
distance between the current posterior approximation and the true parameter
vector (treated as a point mass).  In the 1-D case this reduces to the mean
absolute deviation from the truth; in higher dimensions a sliced-Wasserstein
approximation is used when the POT library is available, falling back to a
coordinate-wise average otherwise.

Each inference method produces "observable states" at different granularities:

- **async_propulate_abc** (``state_kind='archive_reconstruction'``):
  After each simulation attempt the current archive is reconstructed from
  all accepted particles below the running tolerance.  This gives per-event
  resolution.

- **pyabc_smc / abc_smc_baseline** (``state_kind='generation_population'``):
  A snapshot of the population is taken at the end of each SMC generation.
  Resolution equals the number of generations (typically 5-1000).

- **rejection_abc** (``state_kind='accepted_prefix'``):
  The running prefix of accepted particles grows by one per acceptance.

To compare methods on a common time axis, use ``checkpoint_strategy='time_uniform'``
which resamples all methods onto an evenly-spaced wall-time grid using
last-observation-carried-forward (LOCF).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ._helpers import records_to_frame
from .final_state import base_method_name

QUALITY_CURVE_COLUMNS = [
    "method",
    "replicate",
    "axis_kind",
    "axis_value",
    "checkpoint_id",
    "n_particles_used",
    "attempt_count",
    "time_semantics",
    "record_kind",
    "state_kind",
    "wall_time",
    "posterior_samples",
    "wasserstein",
    "posterior_mean_l2",
]

THRESHOLD_COLUMNS = [
    "method",
    "replicate",
    "axis_kind",
    "axis_value_to_threshold",
    "checkpoint_id",
    "n_particles_used",
    "attempt_count",
    "time_semantics",
    "record_kind",
    "state_kind",
    "wall_time",
    "posterior_samples",
    "wasserstein",
    "wall_time_to_threshold",
    "attempts_to_threshold",
    "posterior_samples_to_threshold",
]

_SYNC_METHODS = {"pyabc_smc", "abc_smc_baseline"}


def _posterior_mean_l2(frame: pd.DataFrame, true_params: dict[str, float]) -> float:
    """Normalized L2 from posterior mean to true params (in [0,1] space)."""
    param_cols = [c for c in true_params if c in frame.columns]
    if not param_cols:
        return float("nan")
    means = frame[param_cols].mean()
    diffs = np.array([means[p] - true_params[p] for p in param_cols])
    return float(np.linalg.norm(diffs) / np.sqrt(len(param_cols)))


def _wasserstein_to_true_params(
    frame: pd.DataFrame,
    true_params: dict[str, float],
    n_projections: int,
) -> float:
    """Compute the 1-Wasserstein distance between posterior samples and the truth.

    The true parameter vector is treated as a point mass (Dirac delta), so:

    - **1-D case**: W1 = mean |sample_i - true_value|, computed via
      ``scipy.stats.wasserstein_distance`` (exact).
    - **Multi-D case**: sliced Wasserstein distance via ``ot.sliced_wasserstein_distance``
      with *n_projections* random directions.  Falls back to coordinate-wise
      average W1 when the POT library is not installed.

    Parameters
    ----------
    frame : pd.DataFrame
        Posterior samples with columns matching the keys of *true_params*.
    true_params : dict
        Ground-truth parameter values.
    n_projections : int
        Number of random projections for the sliced approximation (multi-D only).

    Returns
    -------
    float
        Non-negative distance, or ``nan`` if *frame* is empty.
    """
    if len(frame) == 0:
        return float("nan")
    param_names = list(true_params.keys())
    samples = frame[param_names].to_numpy(dtype=float)
    target = np.tile(
        np.asarray([true_params[name] for name in param_names], dtype=float),
        (len(frame), 1),
    )
    if len(param_names) == 1:
        return float(wasserstein_distance(samples[:, 0], target[:, 0]))
    try:
        import ot
    except ImportError:
        # Fallback keeps multi-parameter diagnostics available in lightweight
        # test environments without POT.
        return float(
            np.mean(
                [
                    wasserstein_distance(samples[:, idx], target[:, idx])
                    for idx in range(samples.shape[1])
                ]
            )
        )
    return float(
        ot.sliced_wasserstein_distance(
            samples,
            target,
            n_projections=n_projections,
        )
    )


def posterior_quality_curve(
    records,
    true_params: dict[str, float],
    axis_kind: str,
    checkpoint_strategy: str = "all",
    checkpoint_count: int | None = None,
    *,
    archive_size: int | None = None,
    n_projections: int = 50,
    max_eval_points: int | None = 500,
) -> pd.DataFrame:
    """Compute posterior quality over observable states for a chosen axis.

    Parameters
    ----------
    max_eval_points:
        Cap the number of evaluation indices in ``_async_archive_rows`` to
        avoid O(n^2) blow-up when the record count is large (e.g. 48 k records
        from a 48-worker async run).  Evaluation indices are spread uniformly
        across the record range so the quality curve retains its shape.  Set to
        ``None`` to evaluate at every record (original behaviour).
    """
    if not true_params:
        return pd.DataFrame(columns=QUALITY_CURVE_COLUMNS)
    frame = _prepare_quality_frame(records, true_params)
    if frame.empty:
        return pd.DataFrame(columns=QUALITY_CURVE_COLUMNS)
    if axis_kind not in {"wall_time", "posterior_samples", "attempt_budget"}:
        raise ValueError(f"Unsupported axis_kind: {axis_kind}")

    rows: list[dict[str, object]] = []
    for (method, replicate), group in frame.groupby(["method", "replicate"], sort=True):
        rows.extend(
            _observable_quality_rows(
                group,
                true_params=true_params,
                axis_kind=axis_kind,
                archive_size=archive_size,
                n_projections=n_projections,
                max_eval_points=max_eval_points,
            )
        )

    quality_df = pd.DataFrame(rows, columns=QUALITY_CURVE_COLUMNS)
    if quality_df.empty:
        return pd.DataFrame(columns=QUALITY_CURVE_COLUMNS)
    return _apply_checkpoint_strategy(quality_df, checkpoint_strategy, checkpoint_count)


def wasserstein_at_checkpoints(
    records,
    true_params: dict[str, float],
    checkpoint_steps: list[int],
    n_projections: int = 50,
    *,
    archive_size: int | None = None,
) -> pd.DataFrame:
    """Deprecated wrapper for the legacy checkpoint API."""
    warnings.warn(
        "wasserstein_at_checkpoints() is deprecated; use posterior_quality_curve().",
        DeprecationWarning,
        stacklevel=2,
    )
    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="attempt_budget",
        checkpoint_strategy="all",
        archive_size=archive_size,
        n_projections=n_projections,
    )
    if quality_df.empty:
        return pd.DataFrame(columns=["method", "replicate", "step", "wall_time", "wasserstein"])

    rows: list[dict[str, object]] = []
    for (method, replicate), group in quality_df.groupby(["method", "replicate"], sort=True):
        ordered = group.sort_values("attempt_count")
        for checkpoint in sorted(set(int(step) for step in checkpoint_steps)):
            subset = ordered.loc[ordered["attempt_count"] <= checkpoint]
            if subset.empty:
                continue
            last = subset.iloc[-1]
            rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "step": checkpoint,
                    "wall_time": float(last["wall_time"]),
                    "wasserstein": float(last["wasserstein"]),
                }
            )
    return pd.DataFrame(rows)


def time_to_threshold(
    records,
    true_params: dict[str, float],
    target_wasserstein: float,
    *,
    axis_kind: str = "wall_time",
    archive_size: int | None = None,
    min_particles: int = 1,
    n_projections: int = 50,
) -> pd.DataFrame:
    """Return the earliest observable state that reaches a target distance."""
    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind=axis_kind,
        checkpoint_strategy="all",
        archive_size=archive_size,
        n_projections=n_projections,
    )
    if quality_df.empty:
        return pd.DataFrame(columns=THRESHOLD_COLUMNS)

    rows: list[dict[str, object]] = []
    for (method, replicate), group in quality_df.groupby(["method", "replicate"], sort=True):
        ordered = group.sort_values("axis_value")
        reached = ordered.loc[
            (ordered["wasserstein"] <= target_wasserstein)
            & (ordered["posterior_samples"] >= int(max(1, min_particles)))
        ]
        if reached.empty:
            rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "axis_kind": axis_kind,
                    "axis_value_to_threshold": np.nan,
                    "checkpoint_id": np.nan,
                    "n_particles_used": np.nan,
                    "attempt_count": np.nan,
                    "time_semantics": ordered["time_semantics"].iloc[-1],
                    "record_kind": ordered["record_kind"].iloc[-1],
                    "state_kind": ordered["state_kind"].iloc[-1],
                    "wall_time": np.nan,
                    "posterior_samples": np.nan,
                    "wasserstein": np.nan,
                    "wall_time_to_threshold": np.nan,
                    "attempts_to_threshold": np.nan,
                    "posterior_samples_to_threshold": np.nan,
                }
            )
            continue

        first = reached.iloc[0]
        rows.append(
            {
                "method": method,
                "replicate": replicate,
                "axis_kind": axis_kind,
                "axis_value_to_threshold": float(first["axis_value"]),
                "checkpoint_id": int(first["checkpoint_id"]),
                "n_particles_used": int(first["n_particles_used"]),
                "attempt_count": int(first["attempt_count"]),
                "time_semantics": first["time_semantics"],
                "record_kind": first["record_kind"],
                "state_kind": first["state_kind"],
                "wall_time": float(first["wall_time"]),
                "posterior_samples": int(first["posterior_samples"]),
                "wasserstein": float(first["wasserstein"]),
                "wall_time_to_threshold": float(first["wall_time"]),
                "attempts_to_threshold": int(first["attempt_count"]),
                "posterior_samples_to_threshold": int(first["posterior_samples"]),
            }
        )
    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)


def _prepare_quality_frame(records, true_params: dict[str, float]) -> pd.DataFrame:
    frame = records_to_frame(records)
    if frame.empty:
        return frame

    frame = frame.copy()
    for name in true_params:
        prefixed = f"param_{name}"
        if name not in frame.columns and prefixed in frame.columns:
            frame[name] = frame[prefixed]

    required = [name for name in true_params if name not in frame.columns]
    if required:
        return pd.DataFrame(columns=frame.columns)

    for column in ("record_kind", "time_semantics", "attempt_count", "generation"):
        if column not in frame.columns:
            frame[column] = np.nan
    if "sim_end_time" not in frame.columns:
        frame["sim_end_time"] = np.nan
    if "sim_start_time" not in frame.columns:
        frame["sim_start_time"] = np.nan

    frame["wall_time"] = pd.to_numeric(frame["wall_time"], errors="coerce").fillna(0.0)
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce").fillna(0).astype(int)
    frame["attempt_count"] = pd.to_numeric(frame["attempt_count"], errors="coerce")
    frame["generation"] = pd.to_numeric(frame["generation"], errors="coerce")
    frame["loss"] = pd.to_numeric(frame["loss"], errors="coerce")
    if "tolerance" in frame.columns:
        frame["tolerance"] = pd.to_numeric(frame["tolerance"], errors="coerce")
    for name in true_params:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def _observable_quality_rows(
    group: pd.DataFrame,
    *,
    true_params: dict[str, float],
    axis_kind: str,
    archive_size: int | None,
    n_projections: int,
    max_eval_points: int | None = 500,
) -> list[dict[str, object]]:
    method = base_method_name(str(group["method"].iloc[0]))
    if method == "async_propulate_abc":
        return _async_archive_rows(
            group,
            true_params=true_params,
            axis_kind=axis_kind,
            archive_size=archive_size,
            n_projections=n_projections,
            max_eval_points=max_eval_points,
        )
    if method in _SYNC_METHODS:
        return _sync_generation_rows(
            group,
            true_params=true_params,
            axis_kind=axis_kind,
            n_projections=n_projections,
        )
    if method == "rejection_abc":
        return _accepted_prefix_rows(
            group,
            true_params=true_params,
            axis_kind=axis_kind,
            n_projections=n_projections,
        )
    return _generic_prefix_rows(
        group,
        true_params=true_params,
        axis_kind=axis_kind,
        n_projections=n_projections,
    )


def _async_archive_rows(
    group: pd.DataFrame,
    *,
    true_params: dict[str, float],
    axis_kind: str,
    archive_size: int | None,
    n_projections: int,
    max_eval_points: int | None = 500,
) -> list[dict[str, object]]:
    """Reconstruct the archive at each simulation event for an async method.

    After each event the archive is built from all accepted particles whose
    loss is below the running (minimum) tolerance at that point.  The archive
    is optionally truncated to *archive_size* lowest-loss particles.

    Each checkpoint row has ``state_kind='archive_reconstruction'``.

    When *max_eval_points* is set and the record count exceeds it, evaluation
    indices are spread uniformly across the full range (always including the
    last record) so the quality curve retains its shape while keeping runtime
    linear in *max_eval_points* instead of quadratic in the record count.
    """
    ordered = group.sort_values(["wall_time", "sim_end_time", "step"]).reset_index(drop=True).copy()
    if ordered["attempt_count"].isna().all():
        ordered["attempt_count"] = np.arange(1, len(ordered) + 1, dtype=int)
    else:
        ordered["attempt_count"] = ordered["attempt_count"].ffill()
        missing = ordered["attempt_count"].isna()
        ordered.loc[missing, "attempt_count"] = np.arange(1, int(missing.sum()) + 1, dtype=int)
        ordered["attempt_count"] = ordered["attempt_count"].astype(int)
    ordered["record_kind"] = ordered["record_kind"].fillna("simulation_attempt")
    ordered["time_semantics"] = ordered["time_semantics"].fillna("event_end")

    rows: list[dict[str, object]] = []
    checkpoint_id = 0
    if ordered["tolerance"].notna().sum() == 0:
        return rows

    # --- Pre-compute arrays for O(1) per-checkpoint lookups ---
    tol_notna = ordered["tolerance"].notna().to_numpy()
    cum_tol_min = ordered["tolerance"].expanding().min().to_numpy()
    loss_values = ordered["loss"].to_numpy(dtype=float)

    # Pre-compute global sort rank by (loss, wall_time, step) so we can
    # avoid re-sorting the archive at every checkpoint.
    sort_cols = ordered[["loss", "wall_time", "step"]]
    global_sort_order = np.lexsort(
        (sort_cols["step"].to_numpy(), sort_cols["wall_time"].to_numpy(), sort_cols["loss"].to_numpy())
    )
    global_sort_rank = np.empty(len(ordered), dtype=int)
    global_sort_rank[global_sort_order] = np.arange(len(ordered))

    param_names = list(true_params.keys())
    param_values = ordered[param_names].to_numpy(dtype=float)

    n_records = len(ordered)
    if max_eval_points is not None and n_records > max_eval_points > 0:
        eval_indices = sorted(
            {int(round(x)) for x in np.linspace(0, n_records - 1, num=max_eval_points)}
        )
    else:
        eval_indices = list(range(n_records))

    for idx in eval_indices:
        if not tol_notna[idx]:
            continue
        if not tol_notna[: idx + 1].any():
            continue
        epsilon = cum_tol_min[idx]
        if np.isnan(epsilon):
            continue
        archive_mask = tol_notna[: idx + 1] & (loss_values[: idx + 1] < epsilon)
        if not archive_mask.any():
            continue
        archive_idx = np.flatnonzero(archive_mask)
        archive_idx = archive_idx[np.argsort(global_sort_rank[archive_idx])]
        if archive_size is not None and int(archive_size) > 0:
            archive_idx = archive_idx[: int(archive_size)]
        checkpoint_id += 1
        archive_df = pd.DataFrame(param_values[archive_idx], columns=param_names)
        current = ordered.iloc[idx]
        rows.append(
            _quality_row(
                current=current,
                state=archive_df,
                true_params=true_params,
                axis_kind=axis_kind,
                checkpoint_id=checkpoint_id,
                n_projections=n_projections,
                state_kind="archive_reconstruction",
            )
        )
    return rows


def _sync_generation_rows(
    group: pd.DataFrame,
    *,
    true_params: dict[str, float],
    axis_kind: str,
    n_projections: int,
) -> list[dict[str, object]]:
    """Emit one checkpoint per SMC generation for a synchronous method.

    Each generation's population is used as-is (no archive reconstruction).
    Wall-time and attempt_count are taken from the last particle in each
    generation group.

    Each checkpoint row has ``state_kind='generation_population'``.
    """
    ordered = group.loc[
        group["record_kind"].isna()
        | (group["record_kind"] == "population_particle")
    ].sort_values(["generation", "wall_time", "step"]).reset_index(drop=True).copy()
    if ordered.empty:
        return []
    ordered["record_kind"] = ordered["record_kind"].fillna("population_particle")
    ordered["time_semantics"] = ordered["time_semantics"].fillna("generation_end")
    if ordered["generation"].isna().all():
        ordered["_generation_id"] = ordered.groupby(["wall_time", "tolerance"], dropna=False).ngroup()
    else:
        ordered["_generation_id"] = ordered["generation"].ffill().fillna(0).astype(int)

    rows: list[dict[str, object]] = []
    cumulative_attempts = 0
    checkpoint_id = 0
    for _, generation_group in ordered.groupby("_generation_id", sort=True):
        generation_group = generation_group.sort_values("step").reset_index(drop=True)
        current = generation_group.iloc[-1]
        attempt_series = pd.to_numeric(generation_group["attempt_count"], errors="coerce")
        if attempt_series.notna().any() and float(attempt_series.max()) > 0.0:
            cumulative_attempts = int(attempt_series.max())
        else:
            cumulative_attempts += len(generation_group)
            current = current.copy()
            current["attempt_count"] = cumulative_attempts
        checkpoint_id += 1
        rows.append(
            _quality_row(
                current=current,
                state=generation_group,
                true_params=true_params,
                axis_kind=axis_kind,
                checkpoint_id=checkpoint_id,
                n_projections=n_projections,
                state_kind="generation_population",
            )
        )
    return rows


def _accepted_prefix_rows(
    group: pd.DataFrame,
    *,
    true_params: dict[str, float],
    axis_kind: str,
    n_projections: int,
) -> list[dict[str, object]]:
    ordered = group.sort_values(["wall_time", "step"]).reset_index(drop=True).copy()
    if ordered["attempt_count"].isna().all():
        ordered["attempt_count"] = ordered["step"].astype(int)
    else:
        ordered["attempt_count"] = ordered["attempt_count"].ffill().fillna(ordered["step"]).astype(int)
    ordered["record_kind"] = ordered["record_kind"].fillna("accepted_particle")
    ordered["time_semantics"] = ordered["time_semantics"].fillna("event_end")
    return _prefix_rows(
        ordered,
        true_params=true_params,
        axis_kind=axis_kind,
        n_projections=n_projections,
        state_kind="accepted_prefix",
    )


def _generic_prefix_rows(
    group: pd.DataFrame,
    *,
    true_params: dict[str, float],
    axis_kind: str,
    n_projections: int,
) -> list[dict[str, object]]:
    ordered = group.sort_values(["wall_time", "step"]).reset_index(drop=True).copy()
    if ordered["attempt_count"].isna().all():
        ordered["attempt_count"] = np.arange(1, len(ordered) + 1, dtype=int)
    else:
        ordered["attempt_count"] = ordered["attempt_count"].ffill().fillna(ordered["step"]).astype(int)
    ordered["record_kind"] = ordered["record_kind"].fillna("evaluation")
    ordered["time_semantics"] = ordered["time_semantics"].fillna("event_end")
    return _prefix_rows(
        ordered,
        true_params=true_params,
        axis_kind=axis_kind,
        n_projections=n_projections,
        state_kind="prefix",
    )


def _prefix_rows(
    ordered: pd.DataFrame,
    *,
    true_params: dict[str, float],
    axis_kind: str,
    n_projections: int,
    state_kind: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for checkpoint_id in range(1, len(ordered) + 1):
        state = ordered.iloc[:checkpoint_id].copy()
        current = state.iloc[-1]
        rows.append(
            _quality_row(
                current=current,
                state=state,
                true_params=true_params,
                axis_kind=axis_kind,
                checkpoint_id=checkpoint_id,
                n_projections=n_projections,
                state_kind=state_kind,
            )
        )
    return rows


def _quality_row(
    *,
    current: pd.Series,
    state: pd.DataFrame,
    true_params: dict[str, float],
    axis_kind: str,
    checkpoint_id: int,
    n_projections: int,
    state_kind: str,
) -> dict[str, object]:
    wall_time = float(pd.to_numeric(pd.Series([current["wall_time"]]), errors="coerce").fillna(0.0).iloc[0])
    posterior_samples = int(len(state))
    attempt_count = int(pd.to_numeric(pd.Series([current["attempt_count"]]), errors="coerce").fillna(posterior_samples).iloc[0])
    axis_value = _axis_value_for(
        axis_kind,
        wall_time=wall_time,
        posterior_samples=posterior_samples,
        attempt_count=attempt_count,
    )
    return {
        "method": current["method"],
        "replicate": int(current["replicate"]),
        "axis_kind": axis_kind,
        "axis_value": float(axis_value),
        "checkpoint_id": int(checkpoint_id),
        "n_particles_used": posterior_samples,
        "attempt_count": attempt_count,
        "time_semantics": current["time_semantics"],
        "record_kind": current["record_kind"],
        "state_kind": state_kind,
        "wall_time": wall_time,
        "posterior_samples": posterior_samples,
        "wasserstein": _wasserstein_to_true_params(state, true_params, n_projections=n_projections),
        "posterior_mean_l2": _posterior_mean_l2(state, true_params),
    }


def _axis_value_for(
    axis_kind: str,
    *,
    wall_time: float,
    posterior_samples: int,
    attempt_count: int,
) -> float:
    if axis_kind == "wall_time":
        return float(wall_time)
    if axis_kind == "posterior_samples":
        return float(posterior_samples)
    if axis_kind == "attempt_budget":
        return float(attempt_count)
    raise ValueError(f"Unsupported axis_kind: {axis_kind}")


def _apply_checkpoint_strategy(
    quality_df: pd.DataFrame,
    checkpoint_strategy: str,
    checkpoint_count: int | None,
) -> pd.DataFrame:
    if checkpoint_strategy == "all" or quality_df.empty:
        return quality_df.reset_index(drop=True)

    if checkpoint_strategy == "time_uniform":
        return _apply_time_uniform_strategy(quality_df, checkpoint_count)

    if checkpoint_strategy != "quantile":
        raise ValueError(f"Unsupported checkpoint_strategy: {checkpoint_strategy}")

    selected = []
    for (_, _), group in quality_df.groupby(["method", "replicate"], sort=True):
        ordered = group.sort_values("axis_value").reset_index(drop=True)
        if checkpoint_count is None or checkpoint_count <= 0 or checkpoint_count >= len(ordered):
            selected.append(ordered)
            continue
        indices = sorted({int(round(x)) for x in np.linspace(0, len(ordered) - 1, num=checkpoint_count)})
        selected.append(ordered.iloc[indices].copy())
    if not selected:
        return pd.DataFrame(columns=QUALITY_CURVE_COLUMNS)
    return pd.concat(selected, ignore_index=True)


def _apply_time_uniform_strategy(
    quality_df: pd.DataFrame,
    checkpoint_count: int | None,
) -> pd.DataFrame:
    """Resample all methods onto a shared, evenly-spaced wall-time grid.

    For each (method, replicate) group the function picks the row whose
    ``wall_time`` is closest-but-not-exceeding each grid point (LOCF /
    last-observation-carried-forward).  This gives every method the same
    number of evaluation checkpoints regardless of native granularity.
    """
    if checkpoint_count is None or checkpoint_count <= 0:
        return quality_df.reset_index(drop=True)

    # Build a shared time grid spanning [0, max_wall_time] across ALL methods.
    t_max = float(quality_df["wall_time"].max())
    t_min = float(quality_df["wall_time"].min())
    if t_max <= t_min:
        return quality_df.reset_index(drop=True)
    grid = np.linspace(t_min, t_max, num=checkpoint_count)

    selected: list[pd.DataFrame] = []
    for (method, replicate), group in quality_df.groupby(
        ["method", "replicate"], sort=True
    ):
        ordered = group.sort_values("wall_time").reset_index(drop=True)
        wall_times = ordered["wall_time"].to_numpy(dtype=float)

        # For each grid point find the latest checkpoint at or before that time.
        indices = np.searchsorted(wall_times, grid, side="right") - 1
        # Keep only valid indices (>= 0) and deduplicate.
        valid_mask = indices >= 0
        unique_indices = sorted(set(int(i) for i in indices[valid_mask]))

        if not unique_indices:
            continue

        rows = ordered.iloc[unique_indices].copy()
        # Re-number checkpoint_id for the resampled output.
        rows = rows.reset_index(drop=True)
        rows["checkpoint_id"] = np.arange(1, len(rows) + 1)

        # If we got fewer checkpoints than requested because multiple grid
        # points mapped to the same row, replicate the last observation for
        # each remaining grid point (LOCF).
        if len(rows) < checkpoint_count:
            # Map each grid point to its LOCF row
            locf_rows = []
            for g_idx, (g_time, raw_idx) in enumerate(zip(grid, indices)):
                if raw_idx < 0:
                    continue
                row = ordered.iloc[int(raw_idx)].copy()
                row["checkpoint_id"] = g_idx + 1
                row["wall_time"] = float(g_time)
                row["axis_value"] = float(g_time)
                locf_rows.append(row)
            if locf_rows:
                rows = pd.DataFrame(locf_rows).reset_index(drop=True)

        selected.append(rows)

    if not selected:
        return pd.DataFrame(columns=QUALITY_CURVE_COLUMNS)
    return pd.concat(selected, ignore_index=True)
