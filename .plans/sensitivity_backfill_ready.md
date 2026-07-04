# Sensitivity backfill — READY (fire when sensitivity_000 times out)

STATUS: RESOLVED_DROP_SHARD0   <!-- USER DECISION 2026-06-27: drop shard-0, accept 3-shard coverage (replicate-sharded, so the 192-variant grid is fully covered by shards 1-3; only ~1 replicate of smoothing lost on most cells). NO backfill. Remaining manual step: run the sensitivity finalize/merge over shards 1-3 (+ shard-0's 3 partial variants) to produce top-level run_full_.../sensitivity/ + plots. shard-0 stalled on the intermittent pscom per-config MPI teardown hang (PROPULATE_SKIP_DISCONNECT not set in non-scaling jobs) — 7 configs in 24h. Original below for reference: backfill 14064734 submitted then CANCELLED. shard-000 only FINALIZED 3/192 variant CSVs in 24h (siblings: 192 each) — it was STUCK, not "144/193". My live "CSV count 99->141" metric counted TRANSIENT propulate attempt-log CSVs (cleared on --extend resume, NORMAL), not finalized variant results. No finalized data lost (3 intact + 7 checkpoint pickles). Real issue: shard-000 made ~no finalized progress after the first ~25min. A plain 1-worker --extend backfill will hit the same stall. NEEDS USER: investigate why shard-000 stalls (slow node jwc08n025? a hanging config?) and re-shard / run with more workers, OR accept 3 shards' coverage. --extend resume mechanism itself is fine. -->

## Why
`sensitivity_000` (job 14061028, run_full_20260626_1816) runs at ~half its siblings' pace and will
TIMEOUT at its 24h wall (~2026-06-27 19:05) having completed ~144/193 of its config slice. Siblings
001/002/003 are COMPLETED (193 each). Only shard-0's remaining ~49 configs need backfilling.
`sensitivity_runner.py --extend` skips already-written (variant, method, replicate) rows and runs only
the missing ones, resuming IN PLACE (same output-dir/shard-index/shard-run-id).

## Trigger
Fire when job 14061028 leaves RUNNING (TIMEOUT or otherwise) — i.e. no longer in `list_jobs`.

## Option A — auto-submit via jsc-mpc MCP (what the monitor will do)
```
submit_job(
  cluster="juwels-cluster", project="async-abc-paper", nodes=1,
  walltime="12:00:00", partition="batch",
  command="module restore sim_backend && module load ParaStationMPI && "
          "source /p/project1/tissuetwin/herold2/sim_backend/.venv/bin/activate && "
          "srun python /p/project1/tissuetwin/herold2/async-abc-paper/experiments/scripts/sensitivity_runner.py "
          "--config /p/project1/tissuetwin/herold2/async-abc-paper/experiments/configs/sensitivity.json "
          "--output-dir /p/scratch/tissuetwin/herold2/async-abc/run_full_20260626_1816 "
          "--shard-index 0 --num-shards 4 --shard-run-id run_20260626_190607 --extend")
```
Budget OK: async-abc-paper has ~92 nh remaining (soft 24); this job is ≤12 nh.

## Option B — manual fallback (exact env, one command on a login node)
A ready sbatch is staged on the cluster:
```
sbatch /p/scratch/tissuetwin/herold2/async-abc/run_full_20260626_1816/sensitivity_shard_000_backfill.sbatch
```
Prefer Option B if Option A's srun task topology doesn't match the original run.

## After backfill completes
Run the sensitivity finalize/merge to aggregate the 4 shards (3 complete + backfilled shard-0) into the
top-level `sensitivity/` dir + plots (the auto-merge only fires when all shards finish in one orchestration;
a timed-out shard means it must be run manually).

---

## SEPARATE ISSUE — gaussian_mean OOM (root-caused + FIXED in the propulate fork)
`gaussian_mean` (job 14061000) OOM'd (MaxRSS 53.96 GB / 94 GB, task 0 Killed) at the end of the run.
- FALSE LEAD (my first hypothesis): the post-run posterior-quality/plotting stage. Disproved by profiling
  the real saved history: replaying the FULL analysis+plotting over the persisted 325k-record
  raw_results.csv peaks at only **1.3 GB**. So the plotting path is NOT the cause.
- REAL CAUSE: the async method's retroactive AMIS reweighting, `ABCPMC.extract_posterior(population)`
  (propulate fork), runs on the FULL IN-MEMORY evaluated history. For the ultra-fast Gaussian simulator
  that history is millions of individuals (~1e3 sims/s/worker × 48 × 300 s ≈ 1e7); the persisted
  raw_results is heavily down-sampled (325k), which is why the saved-history replay looked cheap. Each
  snapshot's `log_mixture_density` materialises an **(n, k)** Mahalanobis matrix over all n → tens of GB
  → OOM on rank 0. This is the O(n·k) path the docstring explicitly flags; gaussian_mean is the only
  benchmark whose simulator is fast enough to hit it (compute_posterior_weights=true here, unlike scaling).
- FIX (committed `d9ab64f` on feature/async-abc + PUSHED to the cluster via `pushpropulate` rsync →
  /p/project1/tissuetwin/herold2/propulate — LIVE on JUWELS): `extract_posterior` now evaluates the
  cumulative mixture in CHUNKS over the history (`_EXTRACT_POSTERIOR_CHUNK=65536`) → peak O(chunk·k)
  instead of O(n·k); bit-identical results (new test `test_chunking_matches_unchunked`, full
  TestExtractPosterior green). File: `propulate/propulate/propagators/abcpmc.py`.
- VALIDATION (2026-06-27): 1-rep ASYNC-ONLY repro (job 14063747) COMPLETED, MaxRSS 1.97 GB (was 53.96),
  18 plots produced — the extract_posterior chunking fix is correct + necessary. BUT the FULL multi-method
  re-run (job 14063810: async+abc_smc_baseline+rejection x5 reps) STILL OOM'd: FAILED, MaxRSS 49.2 GB,
  0 plots (inference fine — raw_results + gaussian_analytic_summary written, mean abs err 0.057 — died early
  in plot_benchmark_diagnostics). => the chunking fix is INSUFFICIENT for the full run; there is a SECOND
  cause in the COMBINED multi-method/multi-rep in-memory record set fed to plotting (async + abc_smc +
  rejection x5, full un-downsampled history) and/or pyABC abc_smc in-memory history. NOTE: the async-only
  profiler never exercised the abc_smc/rejection records, so it missed this. NEXT (next session, not done):
  bound the records handed to plot_benchmark_diagnostics — compute final-state (top-k) results from the FULL
  history FIRST, then uniformly subsample the dense simulation_attempt stream per (method,replicate) before
  plotting (curves already cap at 500 eval pts; final posteriors preserved exactly; loud log). Or run
  gaussian sharded 1 rep/shard. Profiling the multi-method path needs the abc_smc/rejection records present.
- NOTE: gaussian_mean's headline validity result, **SBC calibration, already SUCCEEDED** (separate
  experiment, coverage.csv present) — only the posterior-recovery-vs-time plots are missing.
