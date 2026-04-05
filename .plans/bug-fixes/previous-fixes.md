# Previous Bug Fixes

## 2026-04-05: 48-worker hang after abc_smc_baseline finishes (inter-comm teardown race)

**Symptom:** 48-worker scaling job hangs indefinitely after all 48 ranks log `abc_smc_baseline rep=0 status=finish` at ~12.9s. No output from any rank thereafter.

**Root cause:** In `run_abc_smc_baseline`, workers (ranks 1–47) returned `[]` from **inside** the `with MPICommExecutor(COMM_WORLD, root=0)` block. Workers' `__exit__` is a no-op (executor=None), so workers exited immediately and called `allgather(COMM_WORLD)` in `run_method_distributed`. But root was still inside `MPICommExecutor.__exit__` doing `executor.shutdown(wait=True)` → `inter_comm.Disconnect()`. On ParaStation MPI, workers calling `COMM_WORLD.allgather()` while root holds the inter-communicator in `Disconnect()` causes a deadlock.

**Fix:** Restructured the MPI path so workers do **not** `return []` from inside the `with` block — they fall through instead. Added a `COMM_WORLD.Barrier()` after the `with` block exits, ensuring all 48 ranks have fully released the inter-communicator before any rank proceeds to `COMM_WORLD.allgather()` in `run_method_distributed`.

**Files:** `experiments/async_abc/inference/abc_smc_baseline.py` (`run_abc_smc_baseline`, mpi parallel_backend path)



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
