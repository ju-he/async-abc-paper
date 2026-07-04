# Previous Bug Fixes

## 2026-04-08: CommWorldMap hang on root exception + pyABC NaN weight crash

**Symptom (hangs):** Non-scaling jobs (`gandk` shard_000, `straggler` shard_001, `sbc` shards 001/004) hang indefinitely. Progress log shows repeated identical lines (e.g. `simulations=1 elapsed=20.2s`) with no advancement. All are `all_ranks` methods using CommWorldMap.

**Symptom (crash):** Scaling jobs (`scaling_48`, `scaling_bundle_1_16`) crash with `AssertionError: The population total weight nan is not normalized` from `pyabc/population/population.py:98`. Preceded by `RuntimeWarning: invalid value encountered in scalar divide` at `population.py:415`.

**Root cause (hangs):** When `abc.run()` throws on root (e.g. NaN weight error), `CommWorldMap.shutdown()` was skipped because it wasn't in a `try/finally`. Workers remained blocked in `worker_loop()` at `comm.bcast(None, root=0)` waiting for the shutdown signal. Root then hung on `allgather(error_payload)` in `run_method_distributed`, which workers never reached — double deadlock.

**Root cause (NaN weight):** With the "unify stopping criterion" change (commit 2666648), `max_wall_time_s` is the sole binding stop. When pyABC's wall-time fires mid-generation, the sampler stops collecting particles. The partial generation has particles whose weights can't be normalized (sum is 0 or NaN), causing `Population.__init__` to reject them. In the CommWorldMap path this became a hang (root throws, no shutdown); in the scaling MPICommExecutor path it was a visible crash.

**Fix 1 (hangs):** Wrapped CommWorldMap root path in `try/finally` to ensure `cmap.shutdown()` is always called, even when `_run_with_map_callable` throws. Workers now always receive the shutdown bcast and can proceed to Barrier/allgather.

**Fix 2 (NaN weight):** Wrapped `abc.run()` in both `_run_abc_smc_baseline_with_sampler` and `_run_pyabc_smc_with_sampler` to catch the specific `AssertionError` (matching "weight" and "nan" in the message). On catch, falls back to `abc.history` which contains all completed generations. Records are extracted from whatever generations succeeded — treating the interrupted generation as a graceful early stop.

**Files:** `experiments/async_abc/inference/abc_smc_baseline.py`, `experiments/async_abc/inference/pyabc_wrapper.py`

## 2026-04-08: CommWorldMap replaces MPICommExecutor for default mapping sampler

**Symptom:** Non-scaling experiments (`lotka_volterra`, `straggler`, `gaussian_mean`, etc.) hang in `MPICommExecutor.__exit__()` → `Disconnect()` after pyABC methods complete. Two failure modes: (1) at 48 ranks, even a single `Create_intercomm`/`Disconnect` cycle hangs (lotka_volterra `pyabc_smc`); (2) at 16 ranks, repeated cycles hang (straggler, 3rd `abc_smc_baseline` call). The shared-executor fix for the scaling runner doesn't help non-scaling experiments because they use `run_experiment()` / `run_method_distributed()` which creates a new `MPICommExecutor` per method call.

**Root cause:** `MPICommExecutor` uses `Create_intercomm` + `Disconnect` on `COMM_WORLD`, which is fragile on ParaStation MPI at scale. The `Disconnect` call is a collective operation on an inter-communicator that deadlocks non-deterministically at high rank counts.

**Fix:** Added `CommWorldMap` class to `pyabc_sampler.py` — a COMM_WORLD-based blocking parallel map using only `bcast`/`send`/`recv`. No inter-communicators, no `Create_intercomm`, no `Disconnect`. Root broadcasts the function, distributes work items dynamically (one-at-a-time for load balance), and collects ordered results. Workers run a loop processing batches until shutdown. Replaced the self-managed `MPICommExecutor` path in both `abc_smc_baseline.py` and `pyabc_wrapper.py` with `CommWorldMap` for the default `mapping` sampler. The `concurrent_futures` legacy sampler still uses `MPICommExecutor` (opt-in only). The scaling runner's shared `MPICommExecutor` path is unchanged (it still works since it does only one cycle).

**Key design:** All ranks enter `CommWorldMap` within their `run_method` call and all ranks exit it. This means `allgather` in `run_method_distributed` still works — no runner changes needed.

**Files:** `experiments/async_abc/inference/pyabc_sampler.py`, `experiments/async_abc/inference/abc_smc_baseline.py`, `experiments/async_abc/inference/pyabc_wrapper.py`, `experiments/tests/test_inference.py`

## 2026-04-07: Shared MPICommExecutor across scaling workloads (48-rank repeated teardown hang)

**Symptom:** `scaling_48` baseline jobs hang on the **second** `abc_smc_baseline` invocation. First baseline completes (with ~40s teardown), second deadlocks in `MPICommExecutor.__exit__()` → `Disconnect()`. Happens with both `mapping` and `concurrent_futures` samplers.

**Root cause:** Each baseline call created a new `MPICommExecutor(COMM_WORLD, root=0)`, triggering a full `Create_intercomm` + `Disconnect` cycle. ParaStation MPI at 48 ranks is fragile under repeated inter-communicator lifecycles — the first cycle works but the second deadlocks in `Disconnect`. Additionally, workers raced ahead via `if not is_root_rank(): continue` in the scaling runner with no inter-workload synchronization.

**Fix:** Restructured the scaling runner to open **one** `MPICommExecutor` per `n_workers` value, reused across all k-values and replicates. Non-MPI methods (e.g. `async_propulate_abc`) run first, then all pyABC baselines execute under the shared executor. Workers stay in the server recv loop processing work from root; root iterates over workloads. One `Create_intercomm` + one `Disconnect` total. Added `mpi_executor` parameter to `run_abc_smc_baseline` and `run_pyabc_smc` to accept an externally-managed executor, threaded through `run_method` and `run_method_distributed` via `**kwargs`.

**Follow-up (test12):** The shared executor itself worked (one `Create_intercomm`/`Disconnect` cycle), but the scaling runner hung after the first k-value completed. Root cause: `_run_workloads` called `run_method_distributed` even with the shared executor active. For `all_ranks` mode, `run_method_distributed` ends with `allgather(error_payload)` expecting all ranks to participate — but workers were trapped in `MPICommExecutor`'s server recv loop and never reached `allgather`. Fixed by calling `run_method` directly (root-only) when `mpi_executor` is provided, bypassing the `allgather` coordination that workers cannot participate in.

**Files:** `experiments/scripts/scaling_runner.py`, `experiments/async_abc/inference/abc_smc_baseline.py`, `experiments/async_abc/inference/pyabc_wrapper.py`, `experiments/async_abc/inference/method_registry.py`, `experiments/async_abc/utils/runner.py`

## 2026-04-07: Restore mapping as default pyABC MPI sampler (48-rank teardown hang)

**Symptom:** `scaling_48` baseline jobs (test10) hang indefinitely after `abc.run()` completes. Rank 0 finishes in ~4.7s but 47 workers never exit `MPICommExecutor.__exit__`. Output stops growing entirely. First baseline already showed 37s teardown delay (rank 0: 2.9s, workers: 40.5s). Second baseline never completed teardown.

**Root cause:** `pyabc.ConcurrentFutureSampler` maintains a speculative future queue via `MPIPoolExecutor`. On ParaStation MPI with 48 ranks, the async teardown of outstanding futures in `MPICommExecutor.__exit__` deadlocks. This is the same hang documented multiple times (Apr 5-6), which the `mapping` sampler was originally introduced to fix. The Apr 6 revert back to `concurrent_futures` default re-introduced the hang.

**Fix:** Changed default `pyabc_mpi_sampler` from `concurrent_futures` to `mapping` in `resolve_pyabc_mpi_sampler()`. `MappingSampler` uses blocking `executor.map()` — no speculative queue, no async teardown, clean exit. `concurrent_futures` remains available as an explicit opt-in with a warning about the known hang.

**Files:** `experiments/async_abc/inference/pyabc_sampler.py`, `experiments/tests/test_inference.py`, `experiments/tests/mpi_integration_helper.py`, `experiments/tests/mpi_abc_smc_baseline_helper.py`

## 2026-04-07: SLURM time budget too small for MPICommExecutor overhead

**Symptom:** All test-mode scaling_48 jobs (test5-test9) killed by `CANCELLED DUE TO TIME LIMIT` after completing only 2 of 8 k-values. Appeared as a "hang" but was a deterministic timeout.

**Root cause:** `submit_scaling.py`'s `_job_time_hours` budgets `wall_time_limit_s` (30s in test mode) per workload. But each `abc_smc_baseline` MPI call incurs ~50s of `MPICommExecutor` lifecycle overhead (`Create_intercomm` + `Disconnect` on ParaStation MPI with 48 workers). The overhead is 167% of the budgeted 30s per workload. This is test-mode-specific: in production, 50s overhead vs 900s wall cap is only 5.6%.

**Fix:** Added `MPI_EXECUTOR_OVERHEAD_S = 60` constant to `_job_time_hours`, so each workload is budgeted `wall_time_limit_s + mpi_overhead_s`. Test-mode budget goes from 21 min to ~53 min.

**Follow-up:** Reduced `test_k_values` from 8 to 3 (`[10, 100, 1000]`) in both `scaling.json` and `small/scaling.json`. This cuts test workloads from 16→6 and SLURM budget from ~53→~23 min. 8 k-values was excessive for pipeline validation.

**Follow-up:** Restored `wall_time_exact` stop policy for `abc_smc_baseline` in scaling. The Apr 6 revert switched baseline to `simulation_cap_approx` based on misdiagnosed MPI hangs that were actually SLURM timeouts. Both scaling methods now use fixed wall-clock budgets for apples-to-apples comparison, matching the paper's experimental design.

**Files:** `experiments/jobs/submit_scaling.py`, `experiments/configs/scaling.json`, `experiments/configs/small/scaling.json`, `experiments/scripts/scaling_runner.py`

## 2026-04-06: revert pyABC wall-time stop path in scaling, restore futures default

**Symptom:** After switching scaling to inject `max_wall_time_s` into `abc_smc_baseline` / `pyabc_smc`, cluster jobs could freeze during pyABC MPI teardown, and config loading silently inflated `n_generations` to 1000 for any wall-time-tagged run.

**Root cause:** The unstable path was live wall-time termination of pyABC MPI runs, not the steady-state futures sampler itself. Separately, config annotation/validation still treated wall-time as an execution stop policy for sync pyABC methods and auto-expanded generation budgets.

**Fix:**
- Restore `concurrent_futures` as the default `pyabc_mpi_sampler`.
- Keep `mapping` selectable, and keep `concurrent_futures_legacy` only as a deprecated compatibility alias.
- Change scaling stop policy so only `async_propulate_abc` gets live `max_wall_time_s`; pyABC methods now run under explicit simulation/generation caps and are compared at wall-time budgets in post-processing.
- Remove config-time `n_generations=1000` auto-inflation and the related warning.
- Reduce scaling execution-budget knobs to modest explicit values.

**Files:**
- `experiments/async_abc/inference/pyabc_sampler.py`
- `experiments/async_abc/inference/abc_smc_baseline.py`
- `experiments/scripts/scaling_runner.py`
- `experiments/async_abc/io/config.py`
- `experiments/configs/scaling.json`
- `experiments/configs/small/scaling.json`
- `experiments/tests/test_inference.py`
- `experiments/tests/test_runners.py`
- `experiments/tests/mpi_integration_helper.py`
- `experiments/tests/mpi_abc_smc_baseline_helper.py`

**Follow-up:** Updated stale config tests that still expected the removed
`n_generations` warning/auto-inflation, and made scaling budget summaries use
`sim_end_time` as the completion boundary with a `wall_time` fallback for older
records.

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
