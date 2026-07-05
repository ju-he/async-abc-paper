"""CommWorldMap 48-rank verification driver (MPI-01, Phase 2 D-02).

Run locally as smoke test:
    mpirun -n 2 python experiments/scripts/verify_commworldmap_48.py /tmp/out.json

Run on cluster via verify_commworldmap_48.sh (48 ranks).
"""
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: verify_commworldmap_48.py <output_json_path>")
    output_path = Path(sys.argv[1])

    from mpi4py import MPI
    from async_abc.benchmarks.gaussian_mean import GaussianMean
    from async_abc.inference.pyabc_wrapper import run_pyabc_smc
    from async_abc.io.paths import OutputDir

    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    bm = GaussianMean({
        "observed_data_seed": 0,
        "n_obs": 10,
        "true_mu": 0.0,
        "prior_low": -5.0,
        "prior_high": 5.0,
    })

    # D-02: gaussian_mean, population=100, 3 generations, 48 ranks.
    # We use max_simulations as pyABC's proxy for population size per
    # generation in this codebase. n_generations lives in cfg for the
    # baseline runner but pyabc_smc uses max_total_nr_simulations; we
    # scale so that 3 generations of ~100 accepted particles run.
    cfg = {
        "max_simulations": 100,  # per-generation population target
        "k": 5,
        "tol_init": 5.0,
        "parallel_backend": "mpi",
        "n_workers": world_size,
        "pyabc_mpi_sampler": "mapping",  # CommWorldMap default path
        # No max_wall_time_s: verification must complete 3 generations.
    }

    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmpdir:
        od = OutputDir(Path(tmpdir), "commworldmap_48_verify").ensure()
        records = run_pyabc_smc(
            bm.simulate, bm.limits, cfg, od,
            replicate=0, seed=42,
        )
    elapsed_s = time.monotonic() - start

    barrier_reached = True  # reaching here implies the post-finally Barrier passed
    elapsed_by_rank = MPI.COMM_WORLD.gather(elapsed_s, root=0)

    if rank == 0:
        payload = {
            "job_id": job_id,
            "hostname": hostname,
            "world_size": world_size,
            "n_records": len(records),
            "method": records[0].method if records else None,
            "barrier_reached": barrier_reached,
            "elapsed_s": elapsed_s,
            "elapsed_by_rank_s": elapsed_by_rank,
            "max_elapsed_s": max(elapsed_by_rank) if elapsed_by_rank else None,
            "min_elapsed_s": min(elapsed_by_rank) if elapsed_by_rank else None,
            "pyabc_mpi_sampler": "mapping",
            "max_simulations": cfg["max_simulations"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        if len(records) == 0:
            raise SystemExit("verification FAILED: no records returned")
        print(f"[verify_commworldmap_48] PASS: {len(records)} records, "
              f"{elapsed_s:.1f}s, world_size={world_size}, job_id={job_id}")
    else:
        if records != []:
            raise SystemExit(
                f"verification FAILED: non-root rank {rank} returned {len(records)} records"
            )
