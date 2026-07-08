#!/usr/bin/env python3
"""Standalone re-run of a sharded experiment's finalize/merge step.

Use when the automatic finalize (triggered by the last shard to complete) OOMs:
the merge concatenates every shard's ``raw_results.csv`` and then loads the full
combined record set into memory. For a fast-simulator benchmark whose per-run
history is multi-GB (e.g. gandk ~8 GB), that in-memory load exceeds a node that
is shared by other ranks. Run this as a **1-rank job on a full (or large-memory)
node** so the merge process gets the whole node's RAM. All shard runs must have
already completed — this only re-does the merge, so it does not touch inference
results (see .plans/bug-fixes/previous-fixes.md 2026-07-08).
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from async_abc.io.config import load_config  # noqa: E402
from async_abc.utils.sharding import (  # noqa: E402
    ShardLayout,
    all_shards_completed,
    load_shard_statuses,
    shard_output_dirs,
    write_merge_done,
)
from async_abc.utils.shard_finalizers import finalize_experiment_by_name  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, help="Campaign output root (the sharded run's output-dir).")
    p.add_argument("--experiment", required=True, help="Experiment name (e.g. gandk).")
    p.add_argument("--config", required=True, help="Path to the experiment config JSON.")
    p.add_argument("--run-id", required=True, help="Shard run id (the run_YYYYMMDD_HHMMSS under _shards/<exp>/runs/).")
    p.add_argument("--num-shards", type=int, required=True, help="Number of shards in the run.")
    p.add_argument("--force", action="store_true", help="Re-finalize even if merge.done.json already exists.")
    a = p.parse_args()

    cfg = load_config(Path(a.config), test_mode=False, small_mode=False)
    layout = ShardLayout(output_root=Path(a.output_dir), experiment_name=a.experiment, run_id=a.run_id)

    if layout.merge_done_path.exists() and not a.force:
        print(f"merge already done: {layout.merge_done_path} (use --force to redo)")
        return
    if not all_shards_completed(layout, a.num_shards):
        raise SystemExit(
            f"not all {a.num_shards} shards completed for {a.experiment}/{a.run_id}; refusing to merge partial data"
        )

    shard_dirs = shard_output_dirs(layout, a.num_shards)
    statuses = load_shard_statuses(layout, a.num_shards)
    print(f"finalizing {a.experiment} run {a.run_id} ({a.num_shards} shards) → {layout.final_output_dir.data}")
    payload = finalize_experiment_by_name(cfg, layout, shard_dirs, statuses) or {}
    write_merge_done(layout, payload)
    print(f"merge complete: {layout.final_output_dir.data}")


if __name__ == "__main__":
    main()
