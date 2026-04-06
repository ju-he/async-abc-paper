"""MPI integration helper for abc_smc_baseline wall-time MPI teardown tests.

Run via:
    mpirun -n 2 <python> <this_file> <output_json_path>

Rank 0 runs run_abc_smc_baseline with parallel_backend="mpi"; rank 1 is the
MPI worker (executor=None, falls through the with-block).

The test exercises the bounded-backlog MPI path. With 2 ranks (1 worker),
pyABC's ConcurrentFutureSampler is configured with client_max_jobs equal to the
worker count so wall-time stopping does not leave a deep speculative queue
behind. The run still uses the MPI backend and must tear down cleanly.

On success rank 0 writes a JSON result to argv[1] with:
  n_records   — number of ParticleRecords returned
  method      — record.method of first record ("abc_smc_baseline")
  barrier_reached — True if COMM_WORLD.Barrier() after the with-block was reached
"""
import json
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
    rank = MPI.COMM_WORLD.Get_rank()

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
        "max_wall_time_s": 0.5,
        "parallel_backend": "mpi",
        "n_workers": 2,
        "pyabc_client_max_jobs": 2,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        od = OutputDir(Path(tmpdir), "mpi_baseline_test").ensure()
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, replicate=0, seed=42)

    # If we reach here on any rank, the post-with COMM_WORLD.Barrier() was
    # passed — meaning all ranks exited the MPICommExecutor cleanly.
    barrier_reached = True

    if rank == 0:
        result = {
            "n_records": len(records),
            "method": records[0].method if records else None,
            "barrier_reached": barrier_reached,
        }
        output_path.write_text(json.dumps(result))
        assert len(records) > 0, f"Expected records, got {records}"
    else:
        assert records == [], f"Expected worker rank to return no records, got {records}"
