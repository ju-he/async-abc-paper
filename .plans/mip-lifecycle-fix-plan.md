# Harden MPI Lifecycle for JUWELS Cluster Runs

## Analysis

### Observed cluster symptoms
From `/home/juhe/remotes/scratch/herold2/async-abc/small11/_jobs`, the jobs split into three classes:

1. Jobs still making progress
- `13649271`, `13649272`, `13649273` (`sbc` shards 2-4) were still writing fresh `async_propulate_abc` progress updates around `2026-04-03 19:46 CEST`.
- `13649333` (`scaling_bundle_1_16`) was still active shortly before that and had not yet shown a stale hang signature.

2. Jobs clearly hung after finishing computation
- `13649261` gaussian_mean
- `13649262`, `13649263` gandk
- `13649265` lotka_volterra
- `13649269`, `13649270` sbc shards 0-1
- `13649274`, `13649275` straggler

These jobs all showed the same pattern:
- their last meaningful log lines were already `status=finish` for `pyabc_smc` or `abc_smc_baseline`,
- they never logged the experiment-level `Done in ...`,
- their `.out` files then stayed silent for 16 to 56 minutes while the jobs remained `R` in `squeue`.

This strongly indicates that the inference itself finished, but the process hung during MPI teardown or a post-run collective.

3. Separate Propulate startup / transport failures
- `13649334` (`scaling_48`) completed `async_propulate_abc rep=0`, then started `rep=1`, printed only the `status=start` lines for all ranks, and never reached even the first `evaluations=1` update.
- `13649335` (`scaling_96`) was no longer running and had crashed with ParaStation/PSCOM assertions:
  - `pscom_read_get_buf: Assertion 'req->cur_data.iov_len > 0' failed.`

That points to MPI state corruption or transport instability between sequential MPI-heavy runs, not to a pure application-level infinite loop.

### Relevant code paths inspected

#### pyABC MPI path
- `experiments/async_abc/inference/pyabc_wrapper.py`
- `experiments/async_abc/inference/abc_smc_baseline.py`
- `experiments/async_abc/inference/pyabc_sampler.py`
- `experiments/async_abc/inference/method_registry.py`
- `experiments/async_abc/utils/runner.py`

Findings:
- with `n_workers > 1`, `resolve_pyabc_parallel_backend(...)` forces pyABC methods to `parallel_backend="mpi"`,
- `method_execution_mode_for_cfg(...)` upgrades those MPI-backed pyABC methods to `all_ranks`,
- both wrappers run inside:
  - `with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:`

Using the cluster venv `nastjapy_copy/.venv`, the installed stack is:
- `mpi4py 4.1.1`
- `pyabc 0.12.17`
- `propulate 1.2.2`

Inspection of `mpi4py.futures` showed:
- `MPICommExecutor.__exit__()` calls `executor.shutdown(wait=True)`
- `MPIPoolExecutor.shutdown(wait=True)` calls `pool.join()`

This matches the hang signature exactly:
- all work completed,
- all ranks emitted `status=finish`,
- then shutdown blocked waiting for MPI future pool teardown.

#### Propulate MPI path
- `experiments/async_abc/inference/propulate_abc.py`

Findings:
- each run duplicates `MPI.COMM_WORLD` via `_make_propulate_comm()`,
- the duplicate communicator is reused inside one Propulate run,
- cleanup is “best effort” and currently allows pending intra-island requests to survive as only a warning,
- if there are pending requests, the code explicitly skips communicator cleanup “to avoid freeing a busy MPI communicator.”

That explains why sequential Propulate runs in the same job can poison the next run:
- `scaling_48` finished one replicate, then the next replicate hung before first evaluation,
- `scaling_96` showed a lower-level ParaStation transport assertion,
- both are consistent with incomplete communicator/request cleanup between replicates.

### Why the selected fix direction
A more drastic option would have been to split pyABC off MPI entirely and run it as separate multicore jobs, but that changes the execution model and cluster submission shape.

The chosen direction keeps the current experiment design intact:
- keep pyABC and baseline on MPI,
- keep `all_ranks` semantics,
- keep `96`-rank scaling enabled,
- harden executor shutdown and communicator lifecycle so the current design exits cleanly.

## Plan

### Summary
Implement a repo-level MPI lifecycle fix that keeps the current MPI execution model for `pyabc_smc`, `abc_smc_baseline`, and `async_propulate_abc`, but removes the two observed failure modes:

- pyABC methods finish all work and then hang during `MPICommExecutor` teardown.
- sequential Propulate runs can reuse a dirty MPI state, causing the next replicate to stall at startup or crash with ParaStation `pscom` assertions.

The implementation should preserve the current experiment design and keep `96`-rank scaling enabled.

### Key Changes
- Add a small internal MPI-executor helper used by the pyABC wrappers.
  - Replace direct `with MPICommExecutor(...)` usage in `pyabc_wrapper.py` and `abc_smc_baseline.py`.
  - On root rank, enter the upstream `MPICommExecutor` normally but perform shutdown with `executor.shutdown(wait=False)` by default instead of relying on `MPICommExecutor.__exit__()` and its blocking `wait=True` join.
  - Keep a rollback switch in config: add `inference.mpi_executor_shutdown` with values `"nonblocking"` default and `"blocking"` for debugging/regression checks.
  - Emit explicit logs before executor enter, before shutdown request, and after shutdown request so future cluster logs show whether teardown was reached.

- Tighten Propulate communicator lifecycle in `propulate_abc.py`.
  - Keep the per-run `MPI.COMM_WORLD.Dup()` model.
  - After a completed run, treat leftover intra-island requests as a hard error instead of logging a warning and continuing into the next replicate.
  - Always perform a world-level synchronization after communicator cleanup and before returning from the run, so the next replicate never starts while another rank is still tearing down the previous duplicate communicator.
  - Log communicator creation, cleanup result, pending-request count, and post-cleanup synchronization completion.

- Preserve current MPI execution semantics.
  - Do not change `method_execution_mode_for_cfg` behavior for MPI-backed pyABC methods; they must still run as `all_ranks`.
  - Do not split pyABC off into multicore or single-rank execution.
  - Do not remove or cap `96`-rank scaling from configs or submitters.

- Improve failure behavior in runners.
  - If Propulate communicator cleanup fails, raise an explicit exception and mark the run failed instead of allowing the next replicate to hang.
  - Keep current sharding/finalization behavior unchanged; the goal is to fail fast with a clear error, not silently continue with poisoned MPI state.

### Tests and Validation
- Add unit tests for the pyABC MPI helper.
  - Assert default MPI shutdown mode is nonblocking.
  - Assert `"blocking"` config uses `wait=True`.
  - Assert worker-rank behavior remains unchanged and still returns no records.

- Extend existing inference tests.
  - Update fake `MPICommExecutor` coverage in `experiments/tests/test_inference.py` so shutdown mode is asserted explicitly.
  - Keep the existing `mpirun -n 2` helper path and run it under a timeout to ensure the wrapper exits cleanly instead of hanging after result generation.

- Add Propulate lifecycle tests.
  - Unit-test the cleanup path so pending intra-requests raise immediately.
  - Unit-test that a successful run performs communicator cleanup and final synchronization before the next run can begin.

- Cluster acceptance criteria for the fix.
  - Re-run the same `--small` submission pattern on JUWELS.
  - Success means:
    - no pyABC job remains in `R` after its last `status=finish` line,
    - `scaling_48` progresses past `async_propulate_abc rep=1` startup,
    - `scaling_96` completes without `pscom_read_get_buf` assertions,
    - each affected job writes its experiment-level `Done in ...` line and leaves the queue normally.

### Public Interface Changes
- Add one optional inference config key:
  - `mpi_executor_shutdown`: `"nonblocking"` or `"blocking"`.
  - Default: `"nonblocking"` for MPI-backed pyABC methods.
  - Scope: internal execution control only; no output schema changes.

### Assumptions
- The pyABC hang is caused by blocking executor shutdown, not by unfinished sampling work.
- The Propulate startup stall after a completed replicate is caused by incomplete communicator/request cleanup between sequential runs in the same MPI world.
- The required outcome is to keep the current MPI-based experiment design intact, including `96`-rank scaling.
