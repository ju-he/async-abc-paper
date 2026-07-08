# Previous Bug Fixes

## 2026-07-08 — Rerun-campaign OOM: fast-sim benchmarks OOM-killed mid-run on 94 GB/48-rank batch nodes

**Symptom:** In the review rerun campaign (frozen main@0aa1237 + propulate@e148f4f), Stage-1 shard jobs
for the **fast-simulator** benchmarks (gaussian_mean, gandk, lotka_volterra, ablation) FAIL ~60 s into
the run: exactly one rank per job is `Killed` (SIGKILL: task 11/24/27/14), the rest `Terminated`, exit
15, **no Python traceback** = Linux OOM-killer. `runtime_heterogeneity` (lognormal-delay-throttled)
COMPLETED. The `--test` smoke run passed (small budget); only full mode OOMs.

**Root cause (NOT a code-logic bug):** JUWELS **`batch` nodes are only 94 GB / 48 ranks = 1.96 GB/rank**
(`scontrol show node`: RealMemory=94000; `sacct` MaxRSS ≈ 2.19 GB/task → 48×2 GB ≈ 100 GB > 94 GB →
collective OOM). The async-ABC `propulator.population` is **unbounded by design** — every rank retains
every evaluated Individual (active+inactive) to build the `raw_results` attempt history (`propulate_abc.py`:
when `max_wall_time_s` is set the generation budget is -1 and the run executes the FULL wall-time;
this doc already notes history reaches "1e5–1e6 individuals/rank" for a near-instant simulator; CPM
survives only because its slow simulator caps history at ~41k). Fast sims (gaussian ~172 evals/s/rank)
fill this to ~2.5M individuals/rank over 300 s ≈ **~6.6 GB/rank** (measured ~2.2 KB/individual incl.
Python overhead). **Amplifier:** propulate fork commit `52f44e2` ("incrementally maintained
active-population view") made the propagator ~1.54× faster → ~1.5× more individuals accumulate in the
same fixed 300 s wall-time → tips a previously-marginal footprint over the 94 GB ceiling. This is WHY
the campaign reruns (propagator changed) AND why it now OOMs.

**Scope:** only the 4 fast-sim benchmarks. Slower experiments fit: runtime_heterogeneity completed;
straggler/SBC/sensitivity are throttled or short-per-unit; scaling LV/CPM ran before at ~1 GB/rank
(w96_k192 finalized at 978k records ≈ 0.87 GB/task).

**Fix (memory workaround — preserves the frozen commit AND exact record semantics; a proper engine fix
would stream records to disk mid-run = a refactor that risks changing outputs):** spread the 48 workers
across more nodes so each rank gets more RAM. batch: **12 ranks/node × 4 nodes = 7.8 GB/rank**; or
mem192 (180 GB): 24/node × 2 nodes = 7.5 GB/rank. Add `--ntasks-per-node` control to
`submit_replicate_shards.py`. Topology change is benign for the fixed-48-worker quality benchmarks
(posterior is statistical; cross-node latency ≪ eval time). **Validation:** job 14095184 (gaussian,
4 nodes × 12/node) — result pending.

## 2026-06-25 — CPU scaling "MPI teardown hang" was extract_posterior O(n·S·k) on the analysis path

**Symptom:** The CPU `scaling` (lotka_volterra) sweep wedges on JUWELS at high worker counts ×
high archive size — specifically the **w48/w96 × k192/k1000** combos never finalize (no
`throughput_summary` shard), while every `scaling_cpm` combo and every w1/w16 and `*_k48` combo
completes. The job dies with no Python traceback, not OOM, ~minutes into the allocation, at a combo
boundary "after teardown". A standalone reproducer (`repro_pscom_teardown.{sh,py}`) showed all 48
ranks reach `_free_propulate_comm` end (`FREE end 0.00s`) and then wedge before the next combo —
**identically under ParaStation MPI and Open MPI v5.0.5** (confirmed via the `mpi4py.Get_library_version()`
marker), so it is **MPI-independent**.

**Wrong turns (kept here so they are not repeated):** diagnosed in sequence as OOM (disproved by
sacct: 1.7 GB/188 GB), the ParaStation `MPI_Comm_free`/pscom teardown (disproved: `PROPULATE_SKIP_DISCONNECT=1`
did not help and `FREE` is 0.00 s), and the post-loop intra-island drain (disproved: markers show it
finishes in ~10 s, far under the 120 s bound). Each was a guess from heuristics; the fix only came
from (a) reading the markers/`job.log` off the shared mount, and (b) a local micro-benchmark.

**Root cause:** After the timed inference loop and communicator free, `run_propulate_abc` calls
`ABCPMC.extract_posterior(population)` (the retroactive AMIS reweighting — the reported posterior) on
**every** rank. Its cost is **O(n_history · amis_snapshots · k)** and it is **not bounded by the
inference wall-time** — `log_mixture_density` builds `(n, k)` arrays for each of `amis_snapshots`
snapshots. On cheap-simulator scaling sweeps (fixed-walltime, `_stop_policy_for_method` =
`wall_time_exact`) the history reaches ~2e5–1e6 individuals. A local benchmark at k=1000 measured
~0.8 ms/particle (2k→2.7 s, 200k→160 s, ~8 GB), so the hung combos are **10–16 min of single-threaded
NumPy per combo, per rank** — overrunning the SLURM wall clock. The other ranks block at the
post-method `allgather` (`runner.py:876`) waiting for the slowest, which presents as a post-teardown
hang. k-dependence is the `·k` factor (k=48 combos finish in ~30 s even at 8e5 records; k=1000 do not);
CPM survives because its expensive simulator caps the history at ≤41k records.

**Fix:** Gate the estimator behind `inference_cfg["compute_posterior_weights"]` (default **True**, so
all posterior-quality experiments — SBC, gaussian_mean, ablation — are unchanged). The throughput
`scaling`/`scaling_cpm` configs set it **False**: they never consume `posterior_weight` (only
`analysis/sbc.py` does), and `extract_posterior` is a pure function of the saved history, so the
weights are recomputable offline from the raw CSV if ever needed. With the flag off the post-run path
is just the O(n log n) population sort + record build (~tens of seconds even at 1e6 records). No silent
fallback — when skipped, an INFO line records it (CLAUDE.md "crash loudly" compliance).

**Files:** `experiments/async_abc/inference/propulate_abc.py` (flag read + gated `extract_posterior`
block), `experiments/configs/scaling.json`, `experiments/configs/scaling_cpm.json`,
`experiments/tests/test_inference.py` (`test_compute_posterior_weights_false_skips_extract_posterior`
asserts the estimator is not invoked and `posterior_weight` stays empty).

**Note:** The layered ParaStation robustness changes from the wrong turns (per-combo process isolation
in the scaling wrappers, bounded drain via `PROPULATE_DRAIN_TIMEOUT_S`, `PROPULATE_SKIP_DISCONNECT`
default, `SCALING_ENV_SETUP` MPI-swap hook) are correct hardening and were kept, but they do **not**
address this headline hang — this fix does.

### 2026-06-25 (addendum) — the fix initially missed the `--small` config tier

**Symptom after the first fix:** the A/B repro (which sets `compute_posterior_weights` directly)
completed, but the real `--small` scaling sweep STILL wedged at exactly k>=192 (k48 finalized at ~1.5M
records; w48/w96 × {k192,k1000} were Force Terminated ~5s after the wall-time break, same 2 ranks
SIGKILLed). It looked like a *separate* teardown bug.

**Diagnosis:** captured per-rank Python tracebacks via an env-gated `faulthandler.dump_traceback_later`
in `scaling_runner` (`SCALING_FAULTHANDLER_S`) — ptrace-free, because py-spy/gdb are blocked by
`ptrace_scope` on the compute nodes (the watchdog is a sibling, not parent, of the ranks). All 48 ranks'
last frame was `extract_posterior -> run_propulate_abc:574`. So it was never a second bug — it was the
SAME extract_posterior, still running.

**Root cause of the miss:** `load_config(small_mode=True)` does not merge — it loads
`configs/small/<name>.json` **standalone** (`_resolve_small_config_path`). Every real scaling run uses
`--small`, so it reads `configs/small/scaling.json`, which did **not** have the flag. The full-tier
`configs/scaling.json` I patched is only used by non-`--small` runs. (The k48 shard's empty
`posterior_weight` that suggested the fix was active was a red herring: extract_posterior ran but raised
and was caught -> None; at k=1000 the same call is ~20x slower per the `*k` factor, so it wedges instead
of returning.)

**Fix:** add `"compute_posterior_weights": false` to `configs/small/scaling.json` and
`configs/small/scaling_cpm.json` too. Regression test `TestScalingPosteriorWeightsDisabled` asserts the
flag holds through ALL real load paths (full+small x test+no-test) so the tiers cannot silently diverge
again.

**Files:** `experiments/configs/small/scaling.json`, `experiments/configs/small/scaling_cpm.json`,
`experiments/scripts/scaling_runner.py` (faulthandler dumper), `experiments/tests/test_config.py`,
`experiments/jobs/scaling_single_combo.sh` (diagnostic harness).

**Lesson:** when gating behaviour via config, patch (and test) EVERY tier the loader can resolve —
full and `small/`. A flag present in one tier and absent in the sibling is invisible until the exact
tier that's missing it runs in production.

### 2026-06-25 (part 2) — residual 2-node teardown kill: redundant post-run build on non-root ranks

**Symptom (after extract_posterior was fixed):** with the gate working, the w48 combos finalized
cleanly at ~1.5M records, but `w96_k192` (96 ranks across 2 nodes) was **reproducibly** Force
Terminated at teardown — both replicates, two runs in a row — while w48 (1 node) had **zero** Force
Terminated. Per-step data: async_propulate ran to ~329s (past the 300s budget) but only **~26 of 96
ranks** reached `status=finish`; the other ~70 were still in the post-run, and `abc_smc_baseline` never
started.

**Root cause:** `run_method_distributed` keeps only ROOT's records in `all_ranks` mode
(`return records if root_rank else []`), but `run_propulate_abc` ran the post-run **sort + per-particle
record build on every rank**. At ~6e5–1e6 individuals/rank that O(n log n) + O(n) Python build is slow
and highly variable under 96-way CPU contention across 2 nodes, so ranks desynced: the fast ones hit
the post-method `allgather`/teardown while the slow ones were still building, and the step missed the
(~wall+40s) teardown window and was killed. 1 node stayed inside the window; 2 nodes did not.

**Fix:** gate the discarded post-run build to **root only**. `run_method_distributed` sets
`inference_cfg["_records_root_only"]=True` for `all_ranks` mode; `run_propulate_abc` returns `[]` on
non-root immediately after the existing post-`Free` `COMM_WORLD.Barrier` (so no desync into the next
Dup). Output is **identical** — root's records are exactly what was already returned — but the 95
non-root ranks now skip straight to the collective, removing both the 95x redundant work and the
desync. Testable via the `_comm_world_is_root()` seam; tests assert non-root returns `[]` and root still
builds.

**Files:** `experiments/async_abc/utils/runner.py` (set the flag for all_ranks),
`experiments/async_abc/inference/propulate_abc.py` (`_comm_world_is_root` + non-root early return),
`experiments/tests/test_inference.py` (root-only build tests).

### 2026-06-25 (part 3) — residual w96/2-node failure is a pscom SEGFAULT (not a hang)

**Symptom:** after all the above, `w96_k192` (96 ranks / 2 nodes) still reproducibly fails; w48 (1
node) never does. Every w96 "Force Terminated" is the downstream of **one rank crashing**.

**Diagnosis (direct, via the jsc-mpc MCP):** reproduced the exact combo standalone with a ptrace-free
per-rank `faulthandler` dumper (`SCALING_FAULTHANDLER_S`). Findings, all from the cluster:
- The job dies at **~180s** (not the 300s wall-time) when **`srun: error: ... task 8: Segmentation
  fault (core dumped)`** → srun Force-Terminates the step. So it is **not** a timeout, **not** the
  teardown window, **not** a deadlock (0 ranks reached `status=finish`; all 96 were still in the eval
  loop).
- **Not OOM**: sacct MaxRSS 871 MB / MaxVMSize 3.4 GB per task against a 188 GB node.
- No Python `Fatal Python error` dump despite `faulthandler.enable()` → the fault is a **C-level crash
  in the ParaStation pscom / mpi4py transport**, under the high-volume intra-island all-to-all
  messaging (each eval isends to all 95 peers) that only exists at >=2 nodes. MPI confirmed ParaStation
  MPI 5.10.0-1.
- The crashing rank's periodic dumps sat in `population.__repr__` <- `propulator.py:343` (the debug
  log f-string) inside the high-frequency receive loop.

**Status: root-caused, not yet resolved.** It is the original "pscom" concern — vindicated — but a
crash *during* the run, not at teardown. The skip-disconnect / drain-bound / per-combo-isolation /
root-only changes do not address it (they target teardown, not an in-run pscom crash).

**Mitigation shipped (propulate fork `ju-he/propulate@eb70297`, branch feature/async-abc):** gate the
per-generation debug log-string (incl. `Individual.__repr__` on every received individual) and the O(N)
`_get_active_individuals` in `_receive_intra_island_individuals` behind `log.isEnabledFor(DEBUG)`.
Removes a large per-message cost in the exact crash-site receive loop → faster drain, less pscom
backlog pressure. Likely-helpful but NOT a guaranteed fix for a C-level pscom segfault.

**Candidate real fixes (open):** (a) run w96 under OpenMPI instead of ParaStation (different transport;
the most promising sidestep — testable via the MCP); (b) pscom tuning (`PSP_*`); (c) reduce
intra-island message volume in the propulate fork (batch/throttle the all-to-all isend); (d) accept
11/12 small-grid combos. Deploy of the fork mitigation needs `cd /p/project1/tissuetwin/herold2/propulate
&& git pull` (editable install).

**Further MCP investigation (2026-06-25, continued):**
- OpenMPI venv (`/p/.../scaling_openmpi_venv`) is **half-built** — no `scaling_openmpi_env.sh`, its python
  fails with `libpython3.12.so.1.0: cannot open` (Python module not loaded), `module avail OpenMPI`
  empty under Stages/2025. Not turnkey; the swap would need the module stack reverse-engineered.
- **Transport is NOT the cause.** Forced verbs/TCP via `PSP_UCP=0` (job 14053021): the crash *changed
  mode* — no segfault, instead `mpid_irecv_done: read from socket failed ... Failure during collective`
  (a peer connection dropped mid-Bcast). UCX → segfault, TCP → socket drop ⇒ **a rank dies regardless of
  transport**. The trigger is the unbounded high-volume async messaging, not a transport bug.
- **The real culprit: no send backpressure.** `_cleanup_propulate_intra_requests` only `Testsome`s
  (retires *completed* sends). At 96 ranks each eval posts 95 `isend`s; if peers' recvs lag, outstanding
  requests grow without bound → pscom per-connection resource exhaustion → a rank crashes (~180s in).
  w48 stays under the limit; w96 (2× the fan-out) does not.

**Fix implemented (wrapper-side, no propulate-fork change):** added **send backpressure** to the eval
loop in `_cleanup_propulate_intra_requests` (propulate_abc.py). When outstanding `intra_requests` exceed
`PROPULATE_MAX_INFLIGHT_SENDS` (env, default 4096 ≈ 43/peer at 96 ranks), it runs bounded rounds of
(drain incoming via `_receive_intra_island_individuals` → `Testsome`-retire our completed sends).
Draining incoming FIRST lets peers progress and receive our sends, so the retire makes progress — all
non-blocking ⇒ cannot deadlock even if every rank backpressures at once; bounded round count ⇒ never
spins. Results unchanged; only send pacing. The ABCPMC propagator is untouched (this is the propulator's
intra-island worker sync, not the proposal). Unit-tested via an injected `_testsome` seam
(`test_intra_send_backpressure_*`).

**VALIDATED on the cluster (MCP job 14053142, 2 nodes / 96 ranks, ParaStation MPI 5.10.0-1,
`PROPULATE_MAX_INFLIGHT_SENDS=4096`):** w96_k192 — which segfaulted at ~180s on **every** prior attempt
(6/6 reps across 3 production runs + 2 MCP runs) — **completed cleanly**: `srun rc=0`, both methods
finished (`async_propulate_abc` then `abc_smc_baseline`), and all shards written
(`raw_results_w96_k192.csv` 121 MB, `throughput_summary`, `budget_summary`, the abc_smc `.db`). The
backpressure fix resolves the >=2-node pscom crash; the full 12/12 small scaling grid is now reachable.
The fix is already pulled onto the cluster. **Small scaling grid is now 12/12 complete** (MCP jobs
14057684 fresh 2-rep recompute + 14058045 finalize): w96_k192 finalized with both replicates (978,034
records); async_propulate_abc at w96/k192 = 871-941 sims/s vs abc_smc_baseline 303-563. Default cap 4096
worked first try.

**Operational gotcha for recompute:** a *previously-crashed* combo leaves a propulate checkpoint that is
RESUMED on the next run (independent of `--extend`; `--small` doesn't reset checkpoints), and resuming a
crashed-state checkpoint re-triggers the crash even with the fix. So a normal `submit_scaling --extend`
will NOT fix a crashed combo — clear `scaling/logs/propulate_*__<combo>` (and `abc_smc_baseline_*__<combo>*`)
first, then recompute fresh. Only combos that crashed under the OLD (pre-backpressure) code are affected.

### 2026-06-26 — 256-rank (6-node) import-time SEGFAULT: eager `import GPy` in the propulate fork

**Symptom:** the FULL-tier `scaling` combo `w256_k1000` (256 ranks / 6 nodes) crashes **at import time** —
all ranks log `elapsed=0.0s status=start` and then one rank takes a `Segmentation fault (core dumped)`
**before the eval loop starts** (MCP job 14058701: `task 21: Segmentation fault`, srun rc=143, no shards).
Lower rank counts (w1…w96, including the 2-node w96 backpressure validation) all run fine — the crash is
specific to the 6-node / 256-rank import.

**Wrong turn (kept so it is not repeated):** first labelled a generic "import storm" from a sloppy
`grep -rl exec_module` that matches NORMAL startup imports in *every* rank file. The user correctly pushed
back (CPM / gaussian ran fine at lower ranks). The real wedge frame only emerged from reading **each rank's
LAST `faulthandler` snapshot** (the text after the final `Timeout`) — NOT grepping for a frame that appears
in *any* snapshot.

**Root cause:** the propulate fork's `surrogate.py` did an **eager top-level `import GPy`**, so plain
`import propulate` dragged in the heavy GPy → IPython → astroid dependency tree. At 256 ranks across 6 nodes
the simultaneous heavy-import storm (import-machinery + shared-FS metadata contention over that large tree)
crashes a rank during import. The async-ABC eval loop itself was never reached, so none of the prior
teardown/backpressure fixes could have addressed it.

**Fix (propulate fork `feature/async-abc@1d8dfdc`):** make the GPy import **lazy** in `propulate/surrogate.py`
— `import propulate` no longer pulls in GPy/IPython/astroid (verified locally: GPy absent from `sys.modules`
after `import propulate`). Deployed to the cluster by `git pull` on a login node (compute nodes have no
git/internet; editable install).

**VALIDATED on the cluster (MCP job 14060522, 256 ranks / 6 nodes, ParaStation MPI 5.10.0-1,
`PROPULATE_MAX_INFLIGHT_SENDS=4096`, async-only 300s):** `w256_k1000` — which segfaulted at import on the
prior attempt — **completed cleanly**: `srun rc=0`, all 256 ranks reached `status=finish` (rank 0
`evaluations=21 records=212919`), `[scaling] Done in 5m 20s`, **no segfault / no GPy import chain** in any
trace. Shards written: `raw_results_w256_k1000.csv` (59 MB), `throughput_summary`, `budget_summary`,
`timing.csv`. Throughput **709.7 sims/s** over the full 300s budget. This confirms BOTH that the import
crash is gone AND that the 256-rank eval-loop teardown is stable at cap=4096 (the secondary concern). The
full-tier 128/256-worker scaling points are now viable.

**Note (not a bug):** `worker_utilization=0.0215` (2.1%) at w256 — expected for the cheap lotka simulator at
256 workers (workers idle between fast sims); it is part of the scaling story the paper measures, not a
stability defect. Raising the backpressure cap above 4096 for throughput fidelity at 256 ranks remains an
optional tuning knob, not a correctness fix.

## 2026-06-18 — ablation finalize crash: KeyError 'quality' in plot_ablation_amis_isolation

**Symptom:** During the JUWELS small run, `ablation` inference completed (`[ablation] Done in 11m 31s`, all 48 ranks `status=finish`) but the finalize shard exited code 1 with:
```
finalize_ablation_experiment → plot_ablation_amis_isolation (reporters.py:3600)
  quality_df.groupby("wall_time", sort=True)["quality"]
KeyError: 'Column not found: quality'
```
No ablation plots/metadata were produced; shard data in `_shards/ablation/` was intact.

**Root cause:** `plot_ablation_amis_isolation` referenced a non-existent `"quality"` column. `posterior_quality_curve` returns the metric in the `"wasserstein"` column (see `QUALITY_CURVE_COLUMNS` in `analysis/convergence.py`); the plot's own y-label is already "Wasserstein distance to truth". The bug never surfaced in tests because the only ablation test config uses variants `full_model`/`small_archive` — without a `no_amis` variant the plotter early-returns via `_skip` (line 3573) before reaching the aggregation. The real `ablation.json` has both `full_model` and `no_amis`, so the full run hits the groupby. It would have crashed the full ablation run identically.

**Fix:** `reporters.py:3600` `["quality"]` → `["wasserstein"]`. Added regression test `test_plot_ablation_amis_isolation_exports_files_when_both_variants_present` (writes both `ablation_full_model.csv` + `ablation_no_amis.csv`, asserts the plot is produced and `not skipped`) so the aggregation path is covered. Verified by reproducing the exact finalizer call against the real merged variant CSVs from the failed run — produces `ablation_amis_isolation.{pdf,png}` with no error.

**Recovery for the failed run:** re-run finalize-only on the existing shard data (no recompute needed).

**Files:** `experiments/async_abc/plotting/reporters.py`, `experiments/tests/test_plotting.py`

## 2026-04-14 — Phase 3 Plan 03: runtime_heterogeneity plot generation hang in test mode

**Symptom:** `run_all_paper_experiments.py --test` hangs for 60+ minutes after `runtime_heterogeneity` inference completes. Process at 100% CPU on main thread, producing no output, with matplotlib font files open.

**Root cause:** `runtime_heterogeneity_runner.main()` generates matplotlib gantt charts after inference. The gantt plot calls `ax.barh(...)` once per simulation record — with 32755 records produced in a 30s test run, this renders 32755 matplotlib patches, which is O(N^2) or worse in matplotlib's backend and takes hours.

**Fix:** Added `if not test_mode:` guard around all plot generation calls in `runtime_heterogeneity_runner.py`. Data outputs (raw_results.csv, timing.csv, metadata.json, runtime_performance_summary.csv, runtime_debug_summary.csv, speedup_summary.csv) are unaffected and still generated in test mode. Plots are only generated in full (non-test) runs.

**Files:** `experiments/scripts/runtime_heterogeneity_runner.py`

## 2026-04-14 — Phase 3 Plan 01: Dead code removal

- Removed `TrackedFutureExecutor` class and `MPICommExecutor` / `concurrent_futures` branches from `pyabc_sampler.py`, `pyabc_wrapper.py`, `abc_smc_baseline.py`
- `mpi_executor` kwarg dropped from `run_pyabc_smc` and `run_abc_smc_baseline` (no callers post-Phase-2 scaling_runner migration)
- `resolve_pyabc_mpi_sampler` now rejects `'concurrent_futures'` and `'concurrent_futures_legacy'` with `ValueError`
- Tests updated to match (`TestBuildPyabcSampler.test_mpi_concurrent_futures_now_raises_value_error`, `test_resolve_pyabc_mpi_sampler_rejects_*`)
- Rationale: Phase 2 D-03 migrated scaling_runner to CommWorldMap. No config or caller references these paths. Keeping dead code makes future MPI debugging harder.
- Files: `experiments/async_abc/inference/pyabc_sampler.py`, `experiments/async_abc/inference/pyabc_wrapper.py`, `experiments/async_abc/inference/abc_smc_baseline.py`, `experiments/tests/test_inference.py`

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

## 2026-06-27: gaussian_mean benchmark OOM (54 GB) — extract_posterior O(n·k) over full history (propulate fork)

**Symptom:** `gaussian_mean` production job (14061000) FAILED — sacct MaxRSS 53.96 GB / 94 GB, `task 0: Killed`
at the end of the run, after all methods (async, abc_smc_baseline, rejection) logged `status=finish`. No
plots, no timing/budget published (only the down-sampled 73 MB raw_results.csv). Only gaussian_mean failed;
gandk/lotka/CPM finalize fine.

**False lead:** the post-run posterior-quality/plotting stage. DISPROVED by profiling the real saved history —
replaying the full `plot_benchmark_diagnostics` over the persisted 325k-record raw_results.csv peaks at 1.3 GB.

**Root cause:** the async method's retroactive AMIS estimator `ABCPMC.extract_posterior(population)` runs on the
FULL IN-MEMORY evaluated history. The Gaussian simulator is fast enough that the history is millions of
individuals (~1e3 sims/s/worker × 48 × 300 s ≈ 1e7); raw_results is heavily down-sampled on persist (→325k),
which is why the saved-history replay looked cheap. Each AMIS snapshot's `log_mixture_density` built an
**(n, k)** Mahalanobis matrix over all n → tens of GB → OOM on rank 0. This is the O(n·k) path the docstring
flags; gaussian_mean is the only benchmark whose simulator is fast enough to reach it (it sets
compute_posterior_weights=true, unlike the scaling sweeps which set it false — see the 2026-06 scaling fixes).

**Fix:** `extract_posterior` now evaluates the cumulative proposal mixture in CHUNKS over the n history points
(`_EXTRACT_POSTERIOR_CHUNK = 65536` in `propulate/propulate/propagators/abcpmc.py`) → peak memory O(chunk·k)
instead of O(n·k). logsumexp is column-wise so each particle's weight is independent of chunk boundaries:
result is bit-identical (regression test `tests/test_abcpmc.py::TestExtractPosterior::test_chunking_matches_unchunked`;
full TestExtractPosterior green). **This is a propulate-FORK change** — commit+push it and `git pull` on a
JUWELS login node (compute nodes have no git) before re-running. gaussian_mean must be RE-RUN to produce the
missing plots (1-rep repro first to confirm bounded RSS). gaussian_mean's SBC calibration (separate experiment)
already succeeded; only the posterior-recovery-vs-time plots are missing.

## 2026-06-28 — gaussian_mean OOM cause 2 (multi-method plotting) FIXED + full re-run VALIDATED

**Second cause (the chunking fix alone was insufficient):** the full multi-method run still OOM'd at ~49 GB
inside `plot_benchmark_diagnostics`, which held the full un-downsampled multi-method/multi-rep history in
memory (async + abc_smc_baseline + rejection × 5 reps). The async-only profiler missed it.

**Fix:** `_subsample_history_for_plots` in `experiments/async_abc/plotting/reporters.py` (async-abc
`refactor/general@9423cc9`) — computes final-state (top-k) posteriors from the FULL history first, then
uniformly subsamples the dense simulation-attempt stream per (method, replicate) before plotting (cfg
`plots.max_history_records_for_plots`, default 200k; curves already cap at 500 eval pts; final posteriors
preserved exactly).

**Deploy (was a blocker — "deploy mechanism unknown"):** the async-abc repo has NO push alias; deploy is the
same rsync pattern as `pushpropulate`, target `/p/project1/tissuetwin/herold2/async-abc-paper`. Recipe saved
to memory `reference_asyncabc_cluster_deploy.md`.

**VALIDATION (2026-06-28):** full gaussian_mean re-run (job 14066112, both fixes live +
`PROPULATE_SKIP_DISCONNECT=1`, 1 node / 48 ranks) COMPLETED, exit 0:0, **MaxRSS 28.8 GB** (was 49–54 GB →
OOM), 16 plots produced. async recovers the analytic posterior mean to 0.002–0.013 abs err (≈ sync baseline,
≪ rejection). `quality_vs_wall_time.pdf` wired into the paper as Fig. `gaussian-recovery`. OOM closed.
