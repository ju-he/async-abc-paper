# Previous Bug Fixes

## 2026-04-05/06: 48-worker hang in abc_smc_baseline MPI path (two bugs)

### Bug 1: inter-comm teardown race (fixed first)

**Symptom:** All 48 ranks log `abc_smc_baseline rep=0 status=finish` at ~12.9s, then hang indefinitely.

**Root cause:** Workers returned `[]` from **inside** the `with MPICommExecutor(COMM_WORLD, root=0)` block. Their `__exit__` is a no-op (executor=None), so workers immediately called `allgather(COMM_WORLD)` in `run_method_distributed`. But root was still inside `MPICommExecutor.__exit__` doing `inter_comm.Disconnect()`. On ParaStation MPI, workers calling `COMM_WORLD.allgather()` while root holds the inter-communicator in `Disconnect()` deadlocks.

**Fix:** Restructured so workers fall through the `with` block instead of `return []`. Added `COMM_WORLD.Barrier()` after the block to sync all 48 ranks before any `COMM_WORLD` collective.

### Bug 2: tracker.drain() hang (fixed second)

**Symptom:** Root logs `abc_smc_baseline rep=0 status=finish` at ~2.0s, then hangs for 5+ minutes. Workers never log finish.

**Root cause:** `_FutureTracker.drain()` (called after `abc.run()` returns) blocked indefinitely in `concurrent.futures.wait(pending)`. pyABC's `ConcurrentFutureSampler` keeps up to 200 concurrent futures (`client_max_jobs=200`) across 47 workers, leaving O(100+) queued-but-not-yet-sent futures after sampling finishes. These were being processed by workers one by one before shutdown, taking many minutes on this cluster's filesystem/MPI stack.

**Fix (initial):** Removed `_FutureTracker` entirely. After `_run_abc_smc_baseline_with_sampler` returns, call `executor.shutdown(wait=True, cancel_futures=True)` directly. This cancels all queued futures immediately (via `pool.cancel()`), then waits only for the ≤n_workers futures actually in-flight. This improved the 2-rank regression test but turned out to be incomplete on the cluster.

**Files:** `experiments/async_abc/inference/abc_smc_baseline.py` (`run_abc_smc_baseline`, mpi parallel_backend path)

### Follow-up: shutdown-only fix was incomplete under cluster MPI

**Symptom:** `abc_smc_baseline` runs with wall-time stopping still showed long post-finish tails on the cluster. In some cases rank 0 logged `status=finish` quickly but workers only finished tens of seconds later; in others the SLURM step hit the allocation limit after rank 0 had already finished.

**Root cause:** The real issue was not just "queued futures waiting in Python". pyABC's `ConcurrentFutureSampler` defaults to `client_max_jobs=200`, which lets it oversubmit far beyond `n_workers`. On ParaStation MPI, by the time `abc.run()` stops, many orphan futures have already propagated far enough through `MPICommExecutor` that `cancel_futures=True` can no longer retract them. Those already-dispatched futures still need to finish and/or deliver results before communicator teardown can complete cleanly.

**Fix (attempt 2):**
- Add `pyabc_client_max_jobs` inference config knob.
- Default it to `n_workers` for MPI pyABC methods, drastically reducing speculative oversubmission.
- Reintroduce a lightweight tracked executor, but only as teardown instrumentation/safety net.
- Initially kept an explicit `executor.shutdown(wait=True, cancel_futures=True)` in `abc_smc_baseline`.
- Keep the post-`MPICommExecutor` `COMM_WORLD.Barrier()`.

**Outcome:** Helped reduce backlog but did not fix the 48-rank cluster hang.

### Follow-up: explicit shutdown caused double-teardown ownership

**Symptom:** Even after bounding `client_max_jobs`, the 48-rank scaling job still hung after rank 0 finished `abc.run()`. Cluster logs showed root `status=finish` followed by either long worker tails or a permanent stall during teardown.

**Root cause:** `mpi4py.futures.MPICommExecutor.__exit__` already calls `executor.shutdown(wait=True)` automatically on the root rank. We were also calling `executor.shutdown(wait=True, cancel_futures=True)` manually inside the `with MPICommExecutor(...)` block. That meant the same `MPIPoolExecutor` was being shut down twice: once explicitly in our code and then again by `MPICommExecutor.__exit__`. The stuck point in cluster logs is consistent with the second shutdown/join path wedging in communicator teardown on ParaStation MPI.

**Fix (current):**
- Remove the explicit inner `executor.shutdown(...)` call from `abc_smc_baseline`.
- Keep `pyabc_client_max_jobs = n_workers` and the tracked executor for diagnostics.
- Hand shutdown ownership entirely to `MPICommExecutor.__exit__`.
- Add timing/debug logs around:
  - `abc.run()` return
  - `MPICommExecutor` context exit
  - post-exit `COMM_WORLD.Barrier()`

**Files:**
- `experiments/async_abc/inference/pyabc_sampler.py`
- `experiments/async_abc/inference/abc_smc_baseline.py`
- `experiments/async_abc/inference/pyabc_wrapper.py`
- `experiments/tests/test_inference.py`
- `experiments/tests/mpi_abc_smc_baseline_helper.py`

**Outcome:** This also turned out to be incomplete. Removing the explicit inner shutdown did not eliminate the 48-rank hang.

### Follow-up: switch default MPI pyABC sampler from futures to synchronous mapping

**Symptom:** After the bounded-backlog and single-owner-shutdown changes, the single-48 scaling job still froze immediately after rank 0 logged `abc_smc_baseline status=finish`, while smaller packed jobs merely showed long tails. This ruled out our extra shutdown logic as the main cause.

**Root cause:** The persistent failure was the futures-based MPI execution model itself. `pyabc.ConcurrentFutureSampler` plus `MPICommExecutor` remained fragile on ParaStation MPI even with reduced backlog and cleaner shutdown ownership. The shared pattern in cluster logs was: rank 0 finished `abc.run()`, but worker completion still depended on asynchronous future teardown that sometimes never completed at 48 ranks.

**Fix (current default):**
- Add `pyabc_mpi_sampler` inference config with:
  - `mapping` as the default for MPI pyABC methods
  - `concurrent_futures_legacy` as an explicit opt-in fallback
- Use `pyabc.MappingSampler` with a blocking MPI `map` adapter (`executor.map`) for both `abc_smc_baseline` and `pyabc_smc`.
- Keep the legacy futures path in-tree, but warn clearly that it has shown teardown hangs on the cluster.
- Ignore `pyabc_client_max_jobs` for the mapping path because there is no speculative client-side future queue to bound.
- Keep the post-`MPICommExecutor` `COMM_WORLD.Barrier()` so all ranks complete the MPI context before moving on.

**Files:**
- `experiments/async_abc/inference/pyabc_sampler.py`
- `experiments/async_abc/inference/abc_smc_baseline.py`
- `experiments/async_abc/inference/pyabc_wrapper.py`
- `experiments/tests/test_inference.py`
- `experiments/tests/mpi_abc_smc_baseline_helper.py`
- `experiments/tests/mpi_integration_helper.py`



## 2026-04-05: 48-worker scaling job hangs at abc_smc_baseline start

**Symptom:** 48-worker scaling job hangs after `async_propulate_abc` completes. All 47 non-root ranks print `abc_smc_baseline status=start` but rank 0 never does. No error messages.

**Root cause:** `_async_archive_rows` in `convergence.py` iterates over every record (O(n^2)) to compute the quality curve. With 48 workers producing ~48k records, rank 0 gets stuck for hours in the scaling runner's inline `_final_summary_row` → `_quality_curve_by_wall_time` → `posterior_quality_curve` call. Meanwhile, ranks 1-47 proceed to `MPICommExecutor(COMM_WORLD).__enter__()` for abc_smc_baseline and block waiting for rank 0.

**Fix:** Added `max_eval_points=500` parameter to `_async_archive_rows` / `posterior_quality_curve`. When record count exceeds this cap, evaluation indices are spread uniformly via `np.linspace`, reducing complexity from O(n^2) to O(n × max_eval_points).

**Files:** `experiments/async_abc/analysis/convergence.py` (`_async_archive_rows`, `_observable_quality_rows`, `posterior_quality_curve`)

## 2026-04-05: MPI deadlock in post-loop drain (asymmetric intra_requests)

**Symptom:** sensitivity, ablation (k=10 variant), and scaling_48 jobs hang indefinitely after inference completes (~23-30s elapsed). No errors, output just stops.

**Root cause:** In `_propulate_with_wall_time_limit` (propulate_abc.py), the post-loop drain had an asymmetric branch: ranks with empty `intra_requests` (all sends pruned during hot loop) did a single `_receive_intra_island_individuals()` and advanced to a barrier, while ranks with pending sends looped on `Testall` waiting for those ranks to recv — classic MPI rendezvous deadlock.

**Fix:** Replaced the if/else drain with a collective `Allreduce` loop where ALL ranks keep calling `_receive_intra_island_individuals()` until a global MIN of `local_done` flags confirms all sends are complete.

**Files:** `experiments/async_abc/inference/propulate_abc.py` (lines 275-303)

## 2026-04-05: Scaling runner ignores test-mode wall time clamp

**Symptom:** `--test` scaling jobs run for hours because `wall_time_limit_s` (900s) comes from the scaling config, not the test-mode clamp (30s).

**Fix:** Added test-mode clamp for `wall_time_limit_s` and filtered `wall_time_budgets_s` accordingly.

**Files:** `experiments/scripts/scaling_runner.py` (after line 781)

## 2026-04-04: MPI request cleanup (c8dfef3)

**Symptom:** ablation with k=10 accumulates tens of thousands of outstanding `isend` requests, exhausting ParaStationMPI/pscom resources.

**Fix:** `_cleanup_propulate_intra_requests` now uses `Testsome` to periodically prune completed sends during the hot loop.

## 2026-04-04: MPI deadlock under high message volume (935d9d1)

**Symptom:** Propulate wall-time cleanup hangs because a single drain + Waitall deadlocks when MPI rendezvous-mode sends need matching recvs.

**Fix:** Replaced single Waitall with loop of drain-and-Testall.

## 2026-04-04: MPI hangs after pyABC sampling (a2c11d8)

**Symptom:** Hangs between Propulate replicates or after pyABC sampling.

**Fix:** Added post-run COMM_WORLD barriers and proper communicator cleanup.

## 2026-04-04: Per-iteration MPI allreduce replaced with local wall-time check (df4cd45)

**Symptom:** Collective allreduce each iteration caused synchronization overhead.

**Fix:** Each rank checks its own clock independently; no collective ops in hot loop.
