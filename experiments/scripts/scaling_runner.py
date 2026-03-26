#!/usr/bin/env python3
"""Scaling experiment over workers and population size with budgeted summaries.

This runner executes a grid over ``(n_workers, k, method, replicate)`` and
persists three sharded artifacts:

* ``raw_results_w<N>_k<K>.csv``: tagged ParticleRecord rows
* ``throughput_summary_w<N>_k<K>.csv``: one final-state summary row per run
* ``budget_summary_w<N>_k<K>.csv``: one fixed-budget summary row per run/budget

Aggregate CSVs and plots are rebuilt from shards so cluster jobs can safely
fill the grid in parallel without clobbering shared outputs.
"""

from __future__ import annotations

import csv
import logging
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.analysis import final_state_results, posterior_quality_curve
from async_abc.analysis.final_state import base_method_name
from async_abc.benchmarks import make_benchmark
from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord, load_records, write_records
from async_abc.plotting.reporters import plot_scaling_grid
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.runner import (
    compute_scaling_factor,
    format_duration,
    make_arg_parser,
    run_method_distributed,
    write_timing_comparison_csv,
    write_timing_csv,
)
from async_abc.utils.seeding import make_seeds

logger = logging.getLogger(__name__)

_THROUGHPUT_FIELDNAMES = [
    "base_method",
    "method_variant",
    "k",
    "n_workers",
    "replicate",
    "seed",
    "requested_max_simulations",
    "max_wall_time_s",
    "elapsed_wall_time_s",
    "n_simulations",
    "throughput_sims_per_s",
    "final_quality_wasserstein",
    "final_n_particles",
    "final_tolerance",
    "state_kind",
    "test_mode",
]

_BUDGET_FIELDNAMES = [
    "base_method",
    "method_variant",
    "k",
    "n_workers",
    "replicate",
    "seed",
    "budget_s",
    "requested_max_simulations",
    "max_wall_time_s",
    "elapsed_wall_time_s",
    "attempts_by_budget",
    "posterior_samples_by_budget",
    "quality_wasserstein_by_budget",
    "best_tolerance_by_budget",
    "test_mode",
]

_THROUGHPUT_SHARD_RE = re.compile(r"throughput_summary_w(?P<n_workers>\d+)_k(?P<k>\d+)\.csv$")
_BUDGET_SHARD_RE = re.compile(r"budget_summary_w(?P<n_workers>\d+)_k(?P<k>\d+)\.csv$")
_RAW_SHARD_RE = re.compile(r"raw_results_w(?P<n_workers>\d+)_k(?P<k>\d+)\.csv$")


def _interp_time(w: int, measured: dict[int, float]) -> float:
    """Log-linearly interpolate/extrapolate wall time for worker count *w*."""
    keys = sorted(measured)
    if w <= keys[0]:
        return measured[keys[0]] * keys[0] / w
    if w >= keys[-1]:
        return measured[keys[-1]]
    lo = max(k for k in keys if k <= w)
    hi = min(k for k in keys if k >= w)
    if lo == hi:
        return measured[lo]
    t_lo, t_hi = measured[lo], measured[hi]
    frac = (math.log(w) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(t_lo) + frac * (math.log(t_hi) - math.log(t_lo)))


def _simulation_count(records: Iterable[ParticleRecord]) -> int:
    """Return a method-agnostic simulation-attempt count."""
    records = list(records)
    attempt_records = [record for record in records if record.record_kind == "simulation_attempt"]
    if attempt_records:
        return len(attempt_records)

    attempt_counts = [
        int(record.attempt_count)
        for record in records
        if record.attempt_count is not None
    ]
    if attempt_counts:
        return max(attempt_counts)
    return len(records)


def _attempt_count_upto(records: Iterable[ParticleRecord], budget_s: float) -> int:
    subset = [record for record in records if float(record.wall_time) <= float(budget_s)]
    if not subset:
        return 0
    return _simulation_count(subset)


def _elapsed_wall_time(records: Iterable[ParticleRecord]) -> float:
    return max((float(record.wall_time) for record in records), default=0.0)


def _best_tolerance_upto(records: Iterable[ParticleRecord], budget_s: float) -> float:
    values = [
        float(record.tolerance)
        for record in records
        if record.tolerance is not None and float(record.wall_time) <= float(budget_s)
    ]
    return min(values) if values else float("nan")


def _final_tolerance(result_records: Iterable[ParticleRecord]) -> float:
    values = [float(record.tolerance) for record in result_records if record.tolerance is not None]
    return min(values) if values else float("nan")


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    try:
        path.chmod(0o644)
    except OSError:
        pass


def _shard_path(data_dir: Path, stem: str, n_workers: int, k: int) -> Path:
    return data_dir / f"{stem}_w{int(n_workers)}_k{int(k)}.csv"


def _throughput_shard_path(data_dir: Path, n_workers: int, k: int) -> Path:
    return _shard_path(data_dir, "throughput_summary", n_workers, k)


def _budget_shard_path(data_dir: Path, n_workers: int, k: int) -> Path:
    return _shard_path(data_dir, "budget_summary", n_workers, k)


def _raw_shard_path(data_dir: Path, n_workers: int, k: int) -> Path:
    return _shard_path(data_dir, "raw_results", n_workers, k)


def _sort_throughput_rows(rows: Iterable[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("base_method", "")),
            int(row["k"]),
            int(row["n_workers"]),
            int(row["replicate"]),
            int(row["seed"]),
        ),
    )


def _sort_budget_rows(rows: Iterable[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["budget_s"]),
            str(row.get("base_method", "")),
            int(row["k"]),
            int(row["n_workers"]),
            int(row["replicate"]),
            int(row["seed"]),
        ),
    )


def _sort_records(records: Iterable[ParticleRecord]) -> list[ParticleRecord]:
    return sorted(
        records,
        key=lambda record: (
            str(record.method),
            int(record.replicate),
            int(record.seed),
            float(record.wall_time),
            int(record.step),
            str(record.record_kind or ""),
        ),
    )


def _row_key(row: dict, cols: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(col, "")) for col in cols)


def _record_key(record: ParticleRecord) -> tuple[Any, ...]:
    return (
        str(record.method),
        int(record.replicate),
        int(record.seed),
        int(record.step),
        tuple(sorted((str(key), float(value)) for key, value in record.params.items())),
        float(record.loss),
        "" if record.weight is None else float(record.weight),
        "" if record.tolerance is None else float(record.tolerance),
        float(record.wall_time),
        "" if record.worker_id is None else str(record.worker_id),
        "" if record.sim_start_time is None else float(record.sim_start_time),
        "" if record.sim_end_time is None else float(record.sim_end_time),
        "" if record.generation is None else int(record.generation),
        "" if record.record_kind is None else str(record.record_kind),
        "" if record.time_semantics is None else str(record.time_semantics),
        "" if record.attempt_count is None else int(record.attempt_count),
    )


def _dedupe_rows(rows: Iterable[dict], key_cols: list[str]) -> list[dict]:
    merged: dict[tuple[str, ...], dict] = {}
    for row in rows:
        merged[_row_key(row, key_cols)] = row
    return list(merged.values())


def _dedupe_records(records: Iterable[ParticleRecord]) -> list[ParticleRecord]:
    merged: dict[tuple[Any, ...], ParticleRecord] = {}
    for record in records:
        merged[_record_key(record)] = record
    return list(merged.values())


def _load_sharded_rows(output_dir: OutputDir, stem: str, pattern: re.Pattern[str]) -> list[dict]:
    shard_paths = [path for path in sorted(output_dir.data.glob(f"{stem}_w*_k*.csv")) if pattern.fullmatch(path.name)]
    if shard_paths:
        rows: list[dict] = []
        for path in shard_paths:
            rows.extend(_read_rows(path))
        return rows
    return _read_rows(output_dir.data / f"{stem}.csv")


def _load_scaling_rows(output_dir: OutputDir) -> list[dict]:
    return _load_sharded_rows(output_dir, "throughput_summary", _THROUGHPUT_SHARD_RE)


def _load_budget_rows(output_dir: OutputDir) -> list[dict]:
    return _load_sharded_rows(output_dir, "budget_summary", _BUDGET_SHARD_RE)


def _load_scaling_records(output_dir: OutputDir) -> list[ParticleRecord]:
    shard_paths = [path for path in sorted(output_dir.data.glob("raw_results_w*_k*.csv")) if _RAW_SHARD_RE.fullmatch(path.name)]
    if shard_paths:
        records: list[ParticleRecord] = []
        for path in shard_paths:
            records.extend(load_records(path))
        return records
    return load_records(output_dir.data / "raw_results.csv")


def _find_completed_scaling(output_dir: OutputDir, key_cols: list[str]) -> set[tuple[str, ...]]:
    rows = _load_scaling_rows(output_dir)
    return {_row_key(row, key_cols) for row in rows}


def _tagged_method_name(base_method: str, k: int, n_workers: int) -> str:
    return f"{base_method}__k{int(k)}__w{int(n_workers)}"


def _tag_records(records: Iterable[ParticleRecord], method_variant: str) -> list[ParticleRecord]:
    tagged: list[ParticleRecord] = []
    for record in records:
        tagged.append(
            ParticleRecord(
                method=method_variant,
                replicate=int(record.replicate),
                seed=int(record.seed),
                step=int(record.step),
                params=dict(record.params),
                loss=float(record.loss),
                weight=record.weight,
                tolerance=record.tolerance,
                wall_time=float(record.wall_time),
                worker_id=record.worker_id,
                sim_start_time=record.sim_start_time,
                sim_end_time=record.sim_end_time,
                generation=record.generation,
                record_kind=record.record_kind,
                time_semantics=record.time_semantics,
                attempt_count=record.attempt_count,
            )
        )
    return tagged


def _true_params_from_cfg(example_records: Iterable[ParticleRecord], benchmark_cfg: Dict[str, Any]) -> Dict[str, float]:
    param_names: list[str] = []
    for record in example_records:
        if record.params:
            param_names = list(record.params.keys())
            break
    true_params: Dict[str, float] = {}
    for name in param_names:
        key = f"true_{name}"
        if key in benchmark_cfg:
            true_params[name] = float(benchmark_cfg[key])
    return true_params


def _quality_curve_by_wall_time(
    records: list[ParticleRecord],
    *,
    true_params: Dict[str, float],
    archive_size: int | None,
):
    if not true_params:
        return None
    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="wall_time",
        checkpoint_strategy="all",
        archive_size=archive_size,
    )
    if quality_df.empty:
        return None
    return quality_df.sort_values("axis_value")


def _final_summary_row(
    records: list[ParticleRecord],
    *,
    base_method: str,
    method_variant: str,
    k: int,
    n_workers: int,
    replicate: int,
    seed: int,
    requested_max_simulations: int,
    max_wall_time_s: float | None,
    test_mode: bool,
    true_params: Dict[str, float],
) -> dict:
    elapsed = _elapsed_wall_time(records)
    n_sims = _simulation_count(records)
    throughput = n_sims / elapsed if elapsed > 0 else float("inf")

    final_results = final_state_results(records, archive_size=int(k))
    final_state = final_results[0] if final_results else None
    final_n_particles = int(final_state.n_particles_used) if final_state is not None else 0
    final_tolerance = _final_tolerance(final_state.records if final_state is not None else [])
    state_kind = final_state.state_kind if final_state is not None else ""

    quality_df = _quality_curve_by_wall_time(
        records,
        true_params=true_params,
        archive_size=int(k),
    )
    final_quality = float("nan")
    if quality_df is not None and not quality_df.empty:
        final_quality = float(quality_df.iloc[-1]["wasserstein"])
        if not state_kind:
            state_kind = str(quality_df.iloc[-1]["state_kind"])
        if final_n_particles == 0:
            final_n_particles = int(quality_df.iloc[-1]["posterior_samples"])

    return {
        "base_method": base_method,
        "method_variant": method_variant,
        "k": int(k),
        "n_workers": int(n_workers),
        "replicate": int(replicate),
        "seed": int(seed),
        "requested_max_simulations": int(requested_max_simulations),
        "max_wall_time_s": "" if max_wall_time_s is None else float(max_wall_time_s),
        "elapsed_wall_time_s": float(elapsed),
        "n_simulations": int(n_sims),
        "throughput_sims_per_s": float(throughput),
        "final_quality_wasserstein": float(final_quality),
        "final_n_particles": int(final_n_particles),
        "final_tolerance": float(final_tolerance) if math.isfinite(float(final_tolerance)) else "",
        "state_kind": str(state_kind),
        "test_mode": bool(test_mode),
    }


def _budget_summary_rows(
    records: list[ParticleRecord],
    *,
    base_method: str,
    method_variant: str,
    k: int,
    n_workers: int,
    replicate: int,
    seed: int,
    requested_max_simulations: int,
    max_wall_time_s: float | None,
    wall_time_budgets_s: list[float],
    test_mode: bool,
    true_params: Dict[str, float],
) -> list[dict]:
    elapsed = _elapsed_wall_time(records)
    quality_df = _quality_curve_by_wall_time(
        records,
        true_params=true_params,
        archive_size=int(k),
    )
    rows: list[dict] = []
    for budget_s in sorted({float(value) for value in wall_time_budgets_s if float(value) > 0}):
        attempts = _attempt_count_upto(records, budget_s)
        best_tolerance = _best_tolerance_upto(records, budget_s)
        posterior_samples = 0
        quality = float("nan")
        if quality_df is not None and not quality_df.empty:
            eligible = quality_df.loc[quality_df["axis_value"] <= budget_s]
            if not eligible.empty:
                last = eligible.iloc[-1]
                posterior_samples = int(last["posterior_samples"])
                quality = float(last["wasserstein"])
        rows.append(
            {
                "base_method": base_method,
                "method_variant": method_variant,
                "k": int(k),
                "n_workers": int(n_workers),
                "replicate": int(replicate),
                "seed": int(seed),
                "budget_s": float(budget_s),
                "requested_max_simulations": int(requested_max_simulations),
                "max_wall_time_s": "" if max_wall_time_s is None else float(max_wall_time_s),
                "elapsed_wall_time_s": float(min(elapsed, budget_s)),
                "attempts_by_budget": int(attempts),
                "posterior_samples_by_budget": int(posterior_samples),
                "quality_wasserstein_by_budget": float(quality),
                "best_tolerance_by_budget": float(best_tolerance) if math.isfinite(float(best_tolerance)) else "",
                "test_mode": bool(test_mode),
            }
        )
    return rows


def _requested_max_simulations(inference_cfg: Dict[str, Any], *, n_workers: int, k: int, policy: Dict[str, Any]) -> int:
    candidates = [int(inference_cfg.get("max_simulations", 0) or 0)]
    if policy:
        candidates.append(int(policy.get("min_total", 0) or 0))
        candidates.append(int(policy.get("per_worker", 0) or 0) * int(n_workers))
        candidates.append(int(policy.get("k_factor", 0) or 0) * int(k))
    return max(1, max(candidates))


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _cleanup_combo_artifacts(
    output_dir: OutputDir,
    *,
    methods: list[str],
    n_workers: int,
    k: int,
    seeds: list[int],
) -> None:
    _remove_path(_throughput_shard_path(output_dir.data, n_workers, k))
    _remove_path(_budget_shard_path(output_dir.data, n_workers, k))
    _remove_path(_raw_shard_path(output_dir.data, n_workers, k))

    tag = f"w{int(n_workers)}_k{int(k)}"
    for replicate, seed in enumerate(seeds):
        _remove_path(output_dir.logs / f"propulate_rep{replicate}_seed{seed}__{tag}")
        for method in methods:
            _remove_path(output_dir.logs / f"{method}_rep{replicate}_seed{seed}__{tag}_attempts")


def _merge_and_write_combo_records(output_dir: OutputDir, *, n_workers: int, k: int, records: list[ParticleRecord]) -> None:
    path = _raw_shard_path(output_dir.data, n_workers, k)
    existing = load_records(path)
    merged = _sort_records(_dedupe_records([*existing, *records]))
    if merged:
        write_records(path, merged)


def _merge_and_write_combo_rows(
    *,
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
    key_cols: list[str],
    sort_fn,
) -> None:
    existing = _read_rows(path)
    merged = sort_fn(_dedupe_rows([*existing, *rows], key_cols))
    if merged:
        _write_rows_atomic(path, merged, fieldnames)


def rebuild_scaling_outputs(
    output_dir: OutputDir,
    cfg: dict,
    *,
    fallback_rows: list[dict] | None = None,
    fallback_budget_rows: list[dict] | None = None,
    fallback_records: list[ParticleRecord] | None = None,
) -> list[dict]:
    """Rebuild aggregate scaling CSVs/plots/metadata from per-combination shards."""
    aggregate_rows = _load_scaling_rows(output_dir)
    budget_rows = _load_budget_rows(output_dir)
    aggregate_records = _load_scaling_records(output_dir)

    if not aggregate_rows and fallback_rows:
        aggregate_rows = list(fallback_rows)
    if not budget_rows and fallback_budget_rows:
        budget_rows = list(fallback_budget_rows)
    if not aggregate_records and fallback_records:
        aggregate_records = list(fallback_records)

    aggregate_rows = _sort_throughput_rows(aggregate_rows)
    budget_rows = _sort_budget_rows(budget_rows)
    aggregate_records = _sort_records(aggregate_records)

    if aggregate_rows:
        _write_rows_atomic(
            output_dir.data / "throughput_summary.csv",
            aggregate_rows,
            _THROUGHPUT_FIELDNAMES,
        )
    if budget_rows:
        _write_rows_atomic(
            output_dir.data / "budget_summary.csv",
            budget_rows,
            _BUDGET_FIELDNAMES,
        )
    if aggregate_records:
        write_records(output_dir.data / "raw_results.csv", aggregate_records)

    plots_cfg = cfg.get("plots", {})
    should_plot = bool(plots_cfg.get("scaling_curve") or plots_cfg.get("efficiency"))
    if should_plot and aggregate_rows and budget_rows:
        plot_scaling_grid(
            throughput_rows=aggregate_rows,
            budget_rows=budget_rows,
            output_dir=output_dir,
        )

    worker_counts = sorted({int(row["n_workers"]) for row in aggregate_rows}) if aggregate_rows else []
    k_values = sorted({int(row["k"]) for row in aggregate_rows}) if aggregate_rows else []
    scaling_cfg = cfg.get("scaling", {})
    wall_time_budgets_s = [float(value) for value in scaling_cfg.get("wall_time_budgets_s", [])]
    wall_time_limit_s = scaling_cfg.get("wall_time_limit_s")
    if wall_time_limit_s in (None, "") and wall_time_budgets_s:
        wall_time_limit_s = max(wall_time_budgets_s)
    write_metadata(
        output_dir,
        cfg,
        extra={
            "worker_counts": worker_counts,
            "k_values": k_values,
            "wall_time_budgets_s": wall_time_budgets_s,
            "wall_time_limit_s": wall_time_limit_s,
            "max_simulations_policy": scaling_cfg.get("max_simulations_policy", {}),
        },
    )
    return aggregate_rows


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Scaling experiment.")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        dest="n_workers",
        help="Run only this specific worker count (for HPC: match --ntasks=N).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    if args.finalize_only:
        if is_root_rank():
            rebuild_scaling_outputs(output_dir, cfg)
        return

    benchmark = make_benchmark(cfg["benchmark"])
    scaling_cfg = cfg.get("scaling", {})
    worker_counts = list(scaling_cfg.get("worker_counts", [cfg["inference"].get("n_workers", 1)]))
    if test_mode:
        worker_counts = list(scaling_cfg.get("test_worker_counts", worker_counts))
    elif args.n_workers is not None:
        worker_counts = [int(args.n_workers)]

    k_values = list(scaling_cfg.get("k_values", [cfg["inference"].get("k", 100)]))
    if test_mode:
        k_values = list(scaling_cfg.get("test_k_values", k_values))

    wall_time_budgets_s = [
        float(value) for value in scaling_cfg.get("wall_time_budgets_s", [])
        if float(value) > 0
    ]
    wall_time_limit_s = scaling_cfg.get("wall_time_limit_s")
    if wall_time_limit_s in (None, "") and wall_time_budgets_s:
        wall_time_limit_s = max(wall_time_budgets_s)
    wall_time_limit_s = None if wall_time_limit_s in (None, "") else float(wall_time_limit_s)
    if not wall_time_budgets_s and wall_time_limit_s is not None:
        wall_time_budgets_s = [float(wall_time_limit_s)]
    max_simulations_policy = dict(scaling_cfg.get("max_simulations_policy", {}))

    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    if not args.extend and is_root_rank():
        for n_workers in worker_counts:
            for k in k_values:
                _cleanup_combo_artifacts(
                    output_dir,
                    methods=list(cfg["methods"]),
                    n_workers=int(n_workers),
                    k=int(k),
                    seeds=seeds,
                )

    done = (
        _find_completed_scaling(output_dir, ["n_workers", "k", "base_method", "replicate"])
        if args.extend
        else set()
    )

    aggregate_throughput_rows: list[dict] = []
    aggregate_budget_rows: list[dict] = []
    aggregate_records: list[ParticleRecord] = []
    experiment_start = time.time()

    try:
        for n_workers in worker_counts:
            for k in k_values:
                combo_records: list[ParticleRecord] = []
                combo_throughput_rows: list[dict] = []
                combo_budget_rows: list[dict] = []

                requested_max_simulations = _requested_max_simulations(
                    cfg["inference"],
                    n_workers=int(n_workers),
                    k=int(k),
                    policy=max_simulations_policy,
                )
                checkpoint_tag = f"w{int(n_workers)}_k{int(k)}"
                inference_cfg = {
                    **cfg["inference"],
                    "n_workers": int(n_workers),
                    "k": int(k),
                    "max_simulations": int(requested_max_simulations),
                    "_checkpoint_tag": checkpoint_tag,
                }
                if wall_time_limit_s is not None:
                    inference_cfg["max_wall_time_s"] = float(wall_time_limit_s)

                for base_method in cfg["methods"]:
                    method_variant = _tagged_method_name(base_method, int(k), int(n_workers))
                    for replicate, seed in enumerate(seeds):
                        if (str(n_workers), str(k), base_method, str(replicate)) in done:
                            logger.info(
                                "[scaling] --extend: skipping n_workers=%s k=%s %s replicate=%s",
                                n_workers,
                                k,
                                base_method,
                                replicate,
                            )
                            continue

                        run_start = time.time()
                        records = run_method_distributed(
                            base_method,
                            benchmark.simulate,
                            benchmark.limits,
                            inference_cfg,
                            output_dir,
                            replicate,
                            seed,
                        )
                        run_elapsed = time.time() - run_start
                        if not is_root_rank():
                            continue

                        tagged_records = _tag_records(records, method_variant)
                        true_params = _true_params_from_cfg(tagged_records, cfg["benchmark"])
                        combo_records.extend(tagged_records)

                        summary_row = _final_summary_row(
                            tagged_records,
                            base_method=base_method,
                            method_variant=method_variant,
                            k=int(k),
                            n_workers=int(n_workers),
                            replicate=int(replicate),
                            seed=int(seed),
                            requested_max_simulations=int(requested_max_simulations),
                            max_wall_time_s=wall_time_limit_s,
                            test_mode=test_mode,
                            true_params=true_params,
                        )
                        if summary_row["elapsed_wall_time_s"] <= 0 and run_elapsed > 0:
                            summary_row["elapsed_wall_time_s"] = float(run_elapsed)
                            n_sims = int(summary_row["n_simulations"])
                            summary_row["throughput_sims_per_s"] = float(n_sims / run_elapsed)
                        combo_throughput_rows.append(summary_row)
                        combo_budget_rows.extend(
                            _budget_summary_rows(
                                tagged_records,
                                base_method=base_method,
                                method_variant=method_variant,
                                k=int(k),
                                n_workers=int(n_workers),
                                replicate=int(replicate),
                                seed=int(seed),
                                requested_max_simulations=int(requested_max_simulations),
                                max_wall_time_s=wall_time_limit_s,
                                wall_time_budgets_s=wall_time_budgets_s,
                                test_mode=test_mode,
                                true_params=true_params,
                            )
                        )

                if is_root_rank():
                    _merge_and_write_combo_records(
                        output_dir,
                        n_workers=int(n_workers),
                        k=int(k),
                        records=combo_records,
                    )
                    _merge_and_write_combo_rows(
                        path=_throughput_shard_path(output_dir.data, int(n_workers), int(k)),
                        rows=combo_throughput_rows,
                        fieldnames=_THROUGHPUT_FIELDNAMES,
                        key_cols=["n_workers", "k", "base_method", "replicate"],
                        sort_fn=_sort_throughput_rows,
                    )
                    _merge_and_write_combo_rows(
                        path=_budget_shard_path(output_dir.data, int(n_workers), int(k)),
                        rows=combo_budget_rows,
                        fieldnames=_BUDGET_FIELDNAMES,
                        key_cols=["n_workers", "k", "base_method", "replicate", "budget_s"],
                        sort_fn=_sort_budget_rows,
                    )
                    aggregate_records.extend(combo_records)
                    aggregate_throughput_rows.extend(combo_throughput_rows)
                    aggregate_budget_rows.extend(combo_budget_rows)
    finally:
        closer = getattr(benchmark, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.warning("Benchmark teardown failed", exc_info=True)

    experiment_elapsed = time.time() - experiment_start
    name = cfg["experiment_name"]
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(experiment_elapsed))
    if estimate_mode and is_root_rank():
        cfg_full = load_config(args.config, test_mode=False, small_mode=False)
        full_sims = cfg_full["inference"]["max_simulations"]
        full_reps = cfg_full["execution"]["n_replicates"]
        test_sims = cfg["inference"]["max_simulations"]
        test_reps = cfg["execution"]["n_replicates"]
        sim_ratio = (full_sims * full_reps) / max(1, test_sims * test_reps)
        full_worker_counts = cfg_full.get("scaling", {}).get("worker_counts", worker_counts)
        full_k_values = cfg_full.get("scaling", {}).get("k_values", k_values)

        times_by_w: dict[int, list[float]] = {}
        for row in aggregate_throughput_rows:
            times_by_w.setdefault(int(row["n_workers"]), []).append(float(row["elapsed_wall_time_s"]))
        measured_avg = {w: sum(ts) / len(ts) for w, ts in times_by_w.items()}
        if measured_avg:
            estimated = sum(
                _interp_time(int(w), measured_avg) * sim_ratio
                for w in full_worker_counts
                for _k in full_k_values
            )

        _, _, note = compute_scaling_factor(
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        if estimated is not None:
            logger.info(
                "[%s] Estimated full run: ~%s  (%s)",
                name,
                format_duration(estimated),
                note,
            )

    if not is_root_rank():
        return

    write_timing_csv(
        output_dir.data / "timing.csv",
        name,
        experiment_elapsed,
        estimated,
        test_mode,
        run_mode,
    )
    write_timing_comparison_csv(Path(args.output_dir))
    rebuild_scaling_outputs(
        output_dir,
        cfg,
        fallback_rows=aggregate_throughput_rows,
        fallback_budget_rows=aggregate_budget_rows,
        fallback_records=aggregate_records,
    )


if __name__ == "__main__":
    main()
