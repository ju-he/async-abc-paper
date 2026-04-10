"""MPI CommWorldMap coordination helper for test_mpi_hardening.py.

Run via:
    mpirun -n 2 <python> <this_file> <output_json_path> [scenario]

Scenarios: normal, root_exception, multi_call, double_shutdown

Each scenario exercises CommWorldMap under a specific coordination pattern
and writes a JSON result from rank 0. The subprocess returncode is the
signal of success (0) or failure (non-zero).

Per CLAUDE.md: exceptions propagate loudly; no silent swallowing.
"""
import json
import sys
from pathlib import Path

# Make async_abc importable from the experiments package root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _square(x):
    return x * x


def _inc(x):
    return x + 1


def _double(x):
    return x * 2


def _run_normal(cmap, comm):
    """Scenario: root maps, workers loop, all reach Barrier."""
    if cmap.is_root:
        try:
            result = cmap.map(_square, list(range(10)))
            assert result == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], (
                f"map result mismatch: {result}"
            )
        finally:
            cmap.shutdown()
    else:
        cmap.worker_loop()
    if comm.Get_size() > 1:
        comm.Barrier()
    return {"n_records": 10}


def _run_root_exception(cmap, comm):
    """Scenario: root raises inside try, shuts down in finally, workers exit cleanly."""
    if cmap.is_root:
        try:
            cmap.map(_square, [1, 2, 3])  # one valid map first
            raise RuntimeError("probe")
        except RuntimeError as exc:
            assert str(exc) == "probe"
        finally:
            cmap.shutdown()
        assert cmap._shutdown is True, "shutdown flag not set after finally"
    else:
        cmap.worker_loop()
    if comm.Get_size() > 1:
        comm.Barrier()
    return {"root_exception_caught": True}


def _run_multi_call(cmap, comm):
    """Scenario: root calls map multiple times before shutdown."""
    if cmap.is_root:
        try:
            r1 = cmap.map(_inc, [1, 2, 3])
            assert r1 == [2, 3, 4], f"first map: {r1}"
            r2 = cmap.map(_double, [10, 20])
            assert r2 == [20, 40], f"second map: {r2}"
            r3 = cmap.map(_square, [])  # empty iterable path
            assert r3 == [], f"empty map: {r3}"
        finally:
            cmap.shutdown()
    else:
        cmap.worker_loop()
    if comm.Get_size() > 1:
        comm.Barrier()
    return {"multi_call_maps": 3}


def _run_double_shutdown(cmap, comm):
    """Scenario: root calls shutdown() twice; second call must be a no-op."""
    idempotent = None
    if cmap.is_root:
        cmap.map(_inc, [1, 2, 3])  # establish worker_loop is active
        cmap.shutdown()
        # Second call must be a no-op: no bcast (would hang since workers
        # exited), no raise.
        cmap.shutdown()
        idempotent = cmap._shutdown is True
    else:
        cmap.worker_loop()
    if comm.Get_size() > 1:
        comm.Barrier()
    return {"shutdown_idempotent": idempotent}


_SCENARIOS = {
    "normal": _run_normal,
    "root_exception": _run_root_exception,
    "multi_call": _run_multi_call,
    "double_shutdown": _run_double_shutdown,
}

if __name__ == '__main__':
    from mpi4py import MPI
    from async_abc.inference.pyabc_sampler import CommWorldMap

    output_path = Path(sys.argv[1])
    scenario = sys.argv[2] if len(sys.argv) > 2 else "normal"

    if scenario not in _SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Valid: {list(_SCENARIOS)}"
        )

    comm = MPI.COMM_WORLD
    cmap = CommWorldMap(comm)
    scenario_fn = _SCENARIOS[scenario]
    scenario_result = scenario_fn(cmap, comm)

    if comm.Get_rank() == 0:
        payload = {
            "scenario": scenario,
            "world_size": comm.Get_size(),
            "barrier_reached": True,
            **scenario_result,
        }
        Path(sys.argv[1]).write_text(json.dumps(payload))
