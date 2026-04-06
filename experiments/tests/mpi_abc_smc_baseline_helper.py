"""MPI integration helper for abc_smc_baseline wall-time MPI teardown tests.

Run via:
    mpirun -n 2 <python> <this_file> <output_json_path> [mpi_sampler] [client_max_jobs] [max_wall_time_s]

Rank 0 runs run_abc_smc_baseline with parallel_backend="mpi"; rank 1 is the
MPI worker (executor=None, falls through the with-block).

The test exercises the MPI path using the requested pyABC MPI sampler. The
default is the synchronous ``MappingSampler`` path; the legacy futures-based
path remains selectable for regression coverage.

On success rank 0 writes a JSON result to argv[1] with timing diagnostics for
all ranks, including the elapsed spread between the fastest and slowest rank.
"""
import json
import time
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    from mpi4py import MPI
    from async_abc.benchmarks.gaussian_mean import GaussianMean
    from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
    from async_abc.io.paths import OutputDir

    output_path = Path(sys.argv[1])
    mpi_sampler = sys.argv[2] if len(sys.argv) > 2 else "mapping"
    client_max_jobs_arg = sys.argv[3] if len(sys.argv) > 3 else None
    client_max_jobs = (
        int(client_max_jobs_arg)
        if client_max_jobs_arg not in (None, "", "none")
        else None
    )
    max_wall_time_s = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    bm = GaussianMean({
        "observed_data_seed": 0,
        "n_obs": 10,
        "true_mu": 0.0,
        "prior_low": -5.0,
        "prior_high": 5.0,
    })

    # Use wall-time stopping with a high generation cap so the test exercises
    # the same early-stop path as the scaling jobs, but keep the outstanding
    # queue bounded to one worker's worth of work.
    cfg = {
        "max_simulations": 200,
        "k": 5,
        "tol_init": 5.0,
        "n_generations": 1000,
        "max_wall_time_s": max_wall_time_s,
        "parallel_backend": "mpi",
        "n_workers": world_size,
        "pyabc_mpi_sampler": mpi_sampler,
    }
    if client_max_jobs is not None:
        cfg["pyabc_client_max_jobs"] = client_max_jobs

    run_start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmpdir:
        od = OutputDir(Path(tmpdir), "mpi_baseline_test").ensure()
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, replicate=0, seed=42)
    run_elapsed_s = time.monotonic() - run_start

    # If we reach here on any rank, the post-with COMM_WORLD.Barrier() was
    # passed — meaning all ranks exited the MPICommExecutor cleanly.
    barrier_reached = True
    elapsed_by_rank = MPI.COMM_WORLD.gather(run_elapsed_s, root=0)

    if rank == 0:
        result = {
            "n_records": len(records),
            "method": records[0].method if records else None,
            "barrier_reached": barrier_reached,
            "world_size": world_size,
            "pyabc_mpi_sampler": mpi_sampler,
            "client_max_jobs": client_max_jobs,
            "max_wall_time_s": max_wall_time_s,
            "elapsed_by_rank_s": elapsed_by_rank,
            "max_elapsed_s": max(elapsed_by_rank),
            "min_elapsed_s": min(elapsed_by_rank),
            "elapsed_spread_s": max(elapsed_by_rank) - min(elapsed_by_rank),
        }
        output_path.write_text(json.dumps(result))
        assert len(records) > 0, f"Expected records, got {records}"
    else:
        assert records == [], f"Expected worker rank to return no records, got {records}"
