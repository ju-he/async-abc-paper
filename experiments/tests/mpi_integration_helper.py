"""MPI integration helper for test_inference.py.

Run via:
    mpirun -n 2 <python> -m mpi4py.futures <this_file> <output_json_path>

Rank 0 runs run_pyabc_smc with parallel_backend="mpi"; rank 1 is the worker.
Workers import this module for state sync but do not execute the main block.
On success writes a JSON result to the path given as argv[1] and exits 0.
"""
import json
import sys
import tempfile
from pathlib import Path

# Make async_abc importable from the experiments package root
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == '__main__':
    from async_abc.benchmarks.gaussian_mean import GaussianMean
    from async_abc.inference.pyabc_wrapper import run_pyabc_smc
    from async_abc.io.paths import OutputDir

    output_path = Path(sys.argv[1])

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
        # n_workers omitted: worker count is determined by mpirun -n N
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        od = OutputDir(Path(tmpdir), "mpi_test").ensure()
        records = run_pyabc_smc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=42)

    result = {"n_records": len(records), "method": records[0].method if records else None}
    output_path.write_text(json.dumps(result))
    assert len(records) > 0, f"Expected records, got {records}"
