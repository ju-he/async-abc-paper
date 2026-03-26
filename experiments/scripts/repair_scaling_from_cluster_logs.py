#!/usr/bin/env python3
"""Rebuild scaling throughput artifacts from cluster job logs and attempt traces."""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.seeding import make_seeds
from scaling_runner import _write_scaling_shards, rebuild_scaling_outputs

_JOB_LOG_RE = re.compile(r"abc_scaling_(?P<n_workers>\d+)-\d+\.out$")
_ROOT_FINISH_RE = re.compile(
    r"\[progress\]\[rank=0\]\[(?P<method>[^\s]+) rep=(?P<replicate>\d+)\] "
    r"elapsed=(?P<elapsed>[\d.]+)s status=finish(?P<tail>.*)"
)
_KV_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)")


def _count_attempts(trace_dir: Path) -> int:
    total = 0
    for path in sorted(trace_dir.glob("*.jsonl")):
        with open(path) as f:
            total += sum(1 for _ in f)
    return total


def _parse_job_log(log_path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    with open(log_path) as f:
        for line in f:
            match = _ROOT_FINISH_RE.search(line)
            if match is None:
                continue
            tail = {kv.group("key"): kv.group("value") for kv in _KV_RE.finditer(match.group("tail"))}
            rows[(match.group("method"), int(match.group("replicate")))] = {
                "elapsed_s": float(match.group("elapsed")),
                "tail": tail,
            }
    return rows


def _job_n_workers(log_path: Path) -> int:
    match = _JOB_LOG_RE.fullmatch(log_path.name)
    if match is None:
        raise ValueError(f"Could not parse worker count from {log_path}")
    return int(match.group("n_workers"))


def _simulation_count_for_method(
    output_dir: OutputDir,
    method: str,
    replicate: int,
    seed: int,
    n_workers: int,
    tail: dict[str, str],
) -> int:
    attempt_dir = output_dir.logs / f"{method}_rep{replicate}_seed{seed}__w{n_workers}_attempts"
    if attempt_dir.exists():
        return _count_attempts(attempt_dir)

    simulations = tail.get("simulations")
    if simulations not in (None, "", "0"):
        return int(simulations)

    records = tail.get("records")
    if records not in (None, ""):
        return int(records)

    raise RuntimeError(
        f"Could not determine simulation count for method={method} "
        f"replicate={replicate} n_workers={n_workers}"
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", help="Root output directory containing scaling/ and abc_scaling_*.out.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "configs" / "scaling.json"),
        help="Path to scaling config. Default: experiments/configs/scaling.json.",
    )
    parser.add_argument(
        "--job-logs-glob",
        default="abc_scaling_*.out",
        help="Glob for cluster job output logs relative to output_root. Default: abc_scaling_*.out",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    cfg = load_config(args.config, test_mode=False, small_mode=False)
    output_dir = OutputDir(output_root, cfg["experiment_name"]).ensure()

    seeds = make_seeds(cfg["execution"]["n_replicates"], cfg["execution"]["base_seed"])
    log_paths = sorted(output_root.glob(args.job_logs_glob))
    if not log_paths:
        raise RuntimeError(f"No job logs matched {args.job_logs_glob!r} under {output_root}")

    throughput_rows = []
    for log_path in log_paths:
        n_workers = _job_n_workers(log_path)
        parsed = _parse_job_log(log_path)
        for method in cfg["methods"]:
            for replicate, seed in enumerate(seeds):
                key = (method, replicate)
                if key not in parsed:
                    raise RuntimeError(f"Missing root-rank finish line for {key} in {log_path}")
                info = parsed[key]
                elapsed_s = float(info["elapsed_s"])
                n_simulations = _simulation_count_for_method(
                    output_dir,
                    method,
                    replicate,
                    seed,
                    n_workers,
                    info["tail"],
                )
                throughput_rows.append(
                    {
                        "n_workers": n_workers,
                        "method": method,
                        "replicate": replicate,
                        "seed": seed,
                        "n_simulations": n_simulations,
                        "wall_time_s": elapsed_s,
                        "throughput_sims_per_s": n_simulations / elapsed_s if elapsed_s > 0 else float("inf"),
                        "test_mode": False,
                    }
                )

    _write_scaling_shards(output_dir, throughput_rows)
    rebuild_scaling_outputs(output_dir, cfg, fallback_rows=throughput_rows)

    worker_counts = sorted({int(row["n_workers"]) for row in throughput_rows})
    print(
        f"Rebuilt scaling artifacts for {len(worker_counts)} worker counts "
        f"and {len(throughput_rows)} method/replicate rows under {output_dir.root}"
    )


if __name__ == "__main__":
    main()
