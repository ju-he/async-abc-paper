# Previous Bug Fixes

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
