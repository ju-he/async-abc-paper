"""Posterior convergence summaries."""

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


def _wasserstein_to_true_params(
    frame: pd.DataFrame,
    true_params: dict[str, float],
    n_projections: int,
) -> float:
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
) -> pd.DataFrame:
    """Compute posterior quality over observable states for a chosen axis."""
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
) -> list[dict[str, object]]:
    method = base_method_name(str(group["method"].iloc[0]))
    if method == "async_propulate_abc":
        return _async_archive_rows(
            group,
            true_params=true_params,
            axis_kind=axis_kind,
            archive_size=archive_size,
            n_projections=n_projections,
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
) -> list[dict[str, object]]:
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

    for idx in range(len(ordered)):
        prefix = ordered.iloc[: idx + 1].copy()
        observed = prefix.loc[prefix["tolerance"].notna()].copy()
        if observed.empty:
            continue
        epsilon = float(observed["tolerance"].min())
        archive = observed.loc[observed["loss"] < epsilon].copy()
        if archive.empty:
            continue
        archive = archive.sort_values(["loss", "wall_time", "step"]).reset_index(drop=True)
        if archive_size is not None and int(archive_size) > 0:
            archive = archive.iloc[: int(archive_size)].copy()
        current = prefix.iloc[-1]
        if pd.isna(current["tolerance"]):
            continue
        checkpoint_id += 1
        rows.append(
            _quality_row(
                current=current,
                state=archive,
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
