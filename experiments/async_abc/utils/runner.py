"""Shared experiment execution logic used by all runner scripts."""
import argparse
import csv
import json
import logging
import math
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..benchmarks import make_benchmark
from ..inference.method_registry import method_execution_mode, run_method
from ..io.config import load_config
from ..io.paths import OutputDir
from ..io.records import ParticleRecord, RecordWriter
from .mpi import allgather, is_root_rank
from ..utils.seeding import make_seeds

logger = logging.getLogger(__name__)

_RANK_ZERO_STATUS_POLL_S = 0.2


def _propulate_test_generation_budget(test_cfg: Dict[str, Any]) -> int:
    """Return the effective Propulate test-time generation budget.

    In test mode, ``async_propulate_abc`` treats ``max_simulations`` as a total
    cross-rank budget. The estimator must mirror that behavior instead of
    assuming the raw config value is the per-rank sequential workload.
    """
    inference = test_cfg["inference"]
    test_sims = int(inference["max_simulations"])
    test_workers = max(1, int(inference["n_workers"]))
    if not inference.get("test_mode"):
        return test_sims
    return max(1, math.ceil(test_sims / test_workers))


def _method_compute_scale(method: str, cfg_full: Dict[str, Any], cfg_test: Dict[str, Any]) -> float:
    """Return the test→full compute multiplier for one inference method."""
    full_inf = cfg_full["inference"]
    test_inf = cfg_test["inference"]

    full_sims = float(full_inf["max_simulations"])
    full_workers = max(1.0, float(full_inf["n_workers"]))
    test_sims = float(test_inf["max_simulations"])
    test_workers = max(1.0, float(test_inf["n_workers"]))

    if method == "async_propulate_abc":
        return full_sims / float(_propulate_test_generation_budget(cfg_test))
    if method == "rejection_abc":
        return full_sims / test_sims
    return (full_sims / full_workers) / (test_sims / test_workers)


def _base_method_scale(cfg_full: Dict[str, Any], cfg_test: Dict[str, Any]) -> float:
    """Return the average per-method compute scale for a config."""
    methods = list(cfg_full.get("methods", [])) or list(cfg_test.get("methods", [])) or [""]
    scales = [_method_compute_scale(method, cfg_full, cfg_test) for method in methods]
    return sum(scales) / len(scales)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def compute_scaling_factor(
    config_path: Union[str, Path],
) -> Tuple[float, float, str]:
    """Estimate how much longer the full run takes compared to a test run.

    Returns
    -------
    factor : float
        Linear multiplier: ``test_elapsed * factor`` estimates the compute portion.
    extra_seconds : float
        Additional fixed time beyond the linear scaling (expected sleep for
        runtime_heterogeneity experiments; 0 for all others).
    note : str
        Parenthetical describing what the full run entails.
    """
    cfg = load_config(config_path, test_mode=False)
    test_cfg = load_config(config_path, test_mode=True)

    full_sims = cfg["inference"]["max_simulations"]
    full_workers = cfg["inference"]["n_workers"]
    full_reps = cfg["execution"]["n_replicates"]
    test_reps = test_cfg["execution"]["n_replicates"]

    factor = _base_method_scale(cfg, test_cfg) * (full_reps / test_reps)
    extra_seconds = 0.0

    if "sensitivity_grid" in cfg:
        grid = cfg["sensitivity_grid"]
        full_grid_size = 1
        for v in grid.values():
            full_grid_size *= len(v)
        test_grid = {k: v[:1] for k, v in grid.items()}
        test_grid_size = 1
        for v in test_grid.values():
            test_grid_size *= len(v)
        factor *= full_grid_size / test_grid_size
        note = f"{full_sims} sims × {full_reps} reps, {full_grid_size} grid variants"

    elif "scaling" in cfg:
        full_worker_counts = cfg["scaling"].get("worker_counts", [1])
        test_worker_counts = cfg["scaling"].get("test_worker_counts", [1])
        factor *= len(full_worker_counts) / len(test_worker_counts)
        note = f"{full_sims} sims × {full_reps} reps, {len(full_worker_counts)} worker configs"

    elif "heterogeneity" in cfg:
        het = cfg["heterogeneity"]
        mu = float(het.get("mu", 0.0))
        sigmas = het.get("sigma_levels", [het.get("sigma", 1.0)])
        # Expected sleep per sim: exp(mu + sigma^2/2); multiply by sequential sims per worker
        sims_per_sigma = full_sims * full_reps / full_workers
        extra_seconds = sum(
            math.exp(mu + s ** 2 / 2) * sims_per_sigma for s in sigmas
        )
        note = (
            f"{full_sims} sims × {full_reps} reps, "
            f"+{format_duration(extra_seconds)} sleep"
        )

    elif "sbc" in cfg:
        full_trials = int(cfg["sbc"].get("n_trials", 1))
        test_trials = int(test_cfg.get("sbc", {}).get("n_trials", 1))
        factor *= full_trials / test_trials
        note = f"{full_sims} sims × {full_trials} SBC trials, {full_workers} workers"

    elif "straggler" in cfg:
        straggler = cfg["straggler"]
        slowdown_factors = straggler.get("slowdown_factor", [1])
        base_sleep_s = float(straggler.get("base_sleep_s", 0.0))
        n_methods = len(cfg.get("methods", [1]))
        sims_per_sweep = full_sims * full_reps / full_workers * n_methods
        extra_seconds = sum(float(s) * base_sleep_s * sims_per_sweep for s in slowdown_factors)
        note = (
            f"{full_sims} sims × {full_reps} reps, {len(slowdown_factors)} slowdown levels, "
            f"+{format_duration(extra_seconds)} straggler sleep"
        )

    else:
        note = f"{full_sims} sims × {full_reps} reps, {full_workers} workers"

    return factor, extra_seconds, note


_TIMING_FIELDNAMES = [
    "experiment_name",
    "elapsed_s",
    "estimated_full_s",
    "test_mode",
    "timestamp",
]


def compute_corrected_estimate(
    elapsed: float,
    raw_results_path: Union[str, Path],
    config_path: Union[str, Path],
    extra_seconds: float = 0.0,
) -> float:
    """Return a teardown-corrected full-run timing estimate.

    The naive ``elapsed * factor`` formula over-estimates when the test run is
    dominated by fixed per-run overhead (Propulate MPI finalisation, checkpoint
    writes) rather than simulation compute.  This function separates the two:

    * **overhead_per_run** – time spent outside simulations per
      ``run_method_distributed`` call; does *not* scale with simulation count.
    * **compute_per_run** – time actually spent running simulations; scales
      linearly with ``full_sims / test_sims * test_workers / full_workers``.

    It reads ``wall_time`` (time-to-coordinator) per record from
    *raw_results_path*, groups by method, and uses each method's
    ``max(wall_time)`` as that run's compute-phase duration.

    Falls back to ``elapsed * factor + extra_seconds`` if the CSV is missing,
    empty, or lacks usable timing columns.
    """
    factor, extra, _ = compute_scaling_factor(config_path)
    fallback = elapsed * factor + (extra_seconds or extra)

    raw_results_path = Path(raw_results_path)
    if not raw_results_path.exists() or raw_results_path.stat().st_size == 0:
        return fallback

    try:
        with open(raw_results_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return fallback

        method_max_wt: Dict[str, float] = {}
        for row in rows:
            wt_str = row.get("wall_time", "")
            m = row.get("method", "")
            if wt_str and m:
                try:
                    wt = float(wt_str)
                    method_max_wt[m] = max(method_max_wt.get(m, 0.0), wt)
                except ValueError:
                    pass

        if not method_max_wt:
            return fallback

        n_runs_test = len(method_max_wt)
        total_compute_test = sum(method_max_wt.values())
        overhead_per_run = max(0.0, (elapsed - total_compute_test) / n_runs_test)
        compute_per_run = total_compute_test / n_runs_test

        cfg_full = load_config(config_path, test_mode=False)
        cfg_test = load_config(config_path, test_mode=True)
        full_reps = cfg_full["execution"]["n_replicates"]
        test_reps = cfg_test["execution"]["n_replicates"]
        replicate_scale = full_reps / test_reps

        estimated = 0.0
        for method, method_compute_test in method_max_wt.items():
            method_scale = _method_compute_scale(method, cfg_full, cfg_test)
            estimated += replicate_scale * (
                overhead_per_run + method_compute_test * method_scale
            )
        return estimated + (extra_seconds or extra)
    except Exception:
        return fallback


def write_timing_csv(
    path: Union[str, Path],
    experiment_name: str,
    elapsed_s: float,
    estimated_full_s: Optional[float],
    test_mode: bool,
) -> None:
    """Append one timing row to a CSV file, writing the header if needed."""
    path = Path(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TIMING_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "experiment_name": experiment_name,
            "elapsed_s": round(elapsed_s, 3),
            "estimated_full_s": (
                round(estimated_full_s, 3) if estimated_full_s is not None else ""
            ),
            "test_mode": test_mode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })


def find_completed_combinations(csv_path: Path, key_cols: List[str]) -> set:
    """Return the set of key tuples already present in an existing CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file to inspect.
    key_cols:
        Column names whose combined values identify a unique run
        (e.g. ``["method", "replicate"]``).

    Returns
    -------
    set of tuple
        Each element is a tuple of string values, one per key column.
        Returns an empty set if the file is missing or empty.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return {
        tuple(row[c] for c in key_cols)
        for row in rows
        if all(c in row for c in key_cols)
    }


def make_arg_parser(description: str = "") -> argparse.ArgumentParser:
    """Return a pre-configured ArgumentParser for experiment runners."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="Path to JSON experiment config.")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Root directory for results.")
    p.add_argument("--test", action="store_true",
                   help="Test mode: small budget, local max 8 workers, SLURM max 48.")
    p.add_argument("--extend", action="store_true",
                   help="Skip parameter combinations already present in existing CSVs.")
    return p


def _rank_zero_status_path(
    output_dir: OutputDir,
    method: str,
    replicate: int,
    seed: int,
) -> Path:
    """Return the per-run status file used for rank-zero coordination."""
    safe_method = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in method)
    return output_dir.logs / f"rank_zero_{safe_method}_rep{replicate}_seed{seed}.json"


def _write_rank_zero_status(path: Path, payload: Dict[str, str]) -> None:
    """Atomically publish rank-zero method status for non-root ranks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    tmp_path.replace(path)


def _wait_for_rank_zero_status(path: Path) -> Dict[str, str]:
    """Poll until root writes a terminal status for a rank-zero method."""
    while True:
        try:
            with open(path) as f:
                payload = json.load(f)
            if isinstance(payload, dict) and payload.get("kind") in {
                "ok",
                "ImportError",
                "Exception",
            }:
                return payload
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            # Root may still be replacing the status file.
            pass
        time.sleep(_RANK_ZERO_STATUS_POLL_S)


def run_experiment(
    cfg: Dict[str, Any],
    output_dir: OutputDir,
    benchmark=None,
    methods: Optional[List[str]] = None,
    csv_name: str = "raw_results.csv",
    extend: bool = False,
) -> List[ParticleRecord]:
    """Run all methods × replicates and write results to CSV.

    Parameters
    ----------
    cfg:
        Full validated config dict.
    output_dir:
        Already-ensured :class:`~async_abc.io.paths.OutputDir`.
    benchmark:
        Benchmark instance.  If ``None``, instantiated from ``cfg["benchmark"]``.
    methods:
        Method names to run.  Defaults to ``cfg["methods"]``.
    csv_name:
        Filename for the output CSV inside ``output_dir.data``.
    extend:
        When ``True``, skip ``(method, replicate)`` combinations that already
        have at least one row in the existing CSV.

    Returns
    -------
    List[ParticleRecord]
        All records produced (across all methods and replicates).
    """
    created_benchmark = benchmark is None
    if benchmark is None:
        benchmark = make_benchmark(cfg["benchmark"])
    if methods is None:
        methods = cfg["methods"]

    inference_cfg = cfg["inference"]
    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    csv_path = output_dir.data / csv_name
    done = find_completed_combinations(csv_path, ["method", "replicate"]) if extend else set()

    writer = RecordWriter(csv_path)
    all_records: List[ParticleRecord] = []

    try:
        for method in methods:
            for replicate, seed in enumerate(seeds):
                if (method, str(replicate)) in done:
                    logger.info(
                        "[runner] --extend: skipping %s replicate=%s (already done)",
                        method,
                        replicate,
                    )
                    continue
                try:
                    records = run_method_distributed(
                        method, benchmark.simulate, benchmark.limits,
                        inference_cfg, output_dir, replicate, seed,
                    )
                except ImportError as exc:
                    if is_root_rank():
                        warnings.warn(
                            f"Skipping method '{method}' (missing dependency): {exc}",
                            stacklevel=2,
                        )
                    logger.warning("[runner] skipping '%s': %s", method, exc)
                    break  # skip all replicates for this method
                if is_root_rank():
                    writer.write(records)
                    all_records.extend(records)
    finally:
        if created_benchmark:
            closer = getattr(benchmark, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.warning("Benchmark teardown failed", exc_info=True)

    return all_records


def run_method_distributed(
    name: str,
    simulate_fn,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
) -> List[ParticleRecord]:
    """Execute one inference method with rank-aware coordination."""
    execution_mode = method_execution_mode(name)
    root_rank = is_root_rank()

    if execution_mode == "rank_zero":
        status_path = _rank_zero_status_path(output_dir, name, replicate, seed)
        if root_rank:
            try:
                status_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except TypeError:
                if status_path.exists():
                    status_path.unlink()

            try:
                records = run_method(
                    name,
                    simulate_fn,
                    limits,
                    inference_cfg,
                    output_dir,
                    replicate,
                    seed,
                )
            except ImportError as exc:
                _write_rank_zero_status(
                    status_path,
                    {"kind": "ImportError", "message": str(exc)},
                )
                raise
            except Exception:
                message = traceback.format_exc()
                _write_rank_zero_status(
                    status_path,
                    {"kind": "Exception", "message": message},
                )
                raise RuntimeError(message)

            _write_rank_zero_status(status_path, {"kind": "ok", "message": ""})
            return records

        payload = _wait_for_rank_zero_status(status_path)
        kind = payload.get("kind", "Exception")
        message = payload.get("message", "")
        if kind == "ok":
            return []
        if kind == "ImportError":
            raise ImportError(message)
        raise RuntimeError(message)

    should_run_here = execution_mode == "all_ranks" or root_rank

    records: List[ParticleRecord] = []
    error_payload: Optional[Tuple[str, str]] = None

    if should_run_here:
        try:
            records = run_method(
                name,
                simulate_fn,
                limits,
                inference_cfg,
                output_dir,
                replicate,
                seed,
            )
        except ImportError as exc:
            error_payload = ("ImportError", str(exc))
        except Exception:
            error_payload = ("Exception", traceback.format_exc())

    all_errors = allgather(error_payload)
    first_error = next((payload for payload in all_errors if payload is not None), None)
    if first_error is not None:
        kind, message = first_error
        if kind == "ImportError":
            raise ImportError(message)
        raise RuntimeError(message)

    if execution_mode == "all_ranks":
        return records if root_rank else []
    return records
