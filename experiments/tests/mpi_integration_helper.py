"""MPI integration helper for test_inference.py.

Run via:
    mpirun -n 2 <python> <this_file> <output_json_path> [mpi_sampler] [client_max_jobs] [max_wall_time_s]

Rank 0 runs run_pyabc_smc with parallel_backend="mpi"; rank 1 is the worker.
On success rank 0 writes a JSON result to the path given as argv[1] and exits 0.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

# Make async_abc importable from the experiments package root
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == '__main__':
    from mpi4py import MPI
    from async_abc.benchmarks.gaussian_mean import GaussianMean
    from async_abc.inference.pyabc_wrapper import run_pyabc_smc
    from async_abc.io.paths import OutputDir

    output_path = Path(sys.argv[1])
    mpi_sampler = sys.argv[2] if len(sys.argv) > 2 else "mapping"
    client_max_jobs_arg = sys.argv[3] if len(sys.argv) > 3 else None
    client_max_jobs = (
        int(client_max_jobs_arg)
        if client_max_jobs_arg not in (None, "", "none")
        else None
    )
    max_wall_time_s = float(sys.argv[4]) if len(sys.argv) > 4 else None
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    bm = GaussianMean({
        "observed_data_seed": 0,
        "n_obs": 10,
        "true_mu": 0.0,
        "prior_low": -5.0,
        "prior_high": 5.0,
    })

    cfg = {
        "max_simulations": 30,
        "k": 5,
        "tol_init": 5.0,
        "parallel_backend": "mpi",
        "n_workers": world_size,
        "pyabc_mpi_sampler": mpi_sampler,
    }
    if client_max_jobs is not None:
        cfg["pyabc_client_max_jobs"] = client_max_jobs
    if max_wall_time_s is not None:
        cfg["max_wall_time_s"] = max_wall_time_s

    run_start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmpdir:
        od = OutputDir(Path(tmpdir), "mpi_test").ensure()
        records = run_pyabc_smc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=42)
    run_elapsed_s = time.monotonic() - run_start
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
