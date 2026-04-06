# Restore Futures-Default pyABC and Make Wall-Time Analysis-Only

## Summary

Revert MPI pyABC to the futures-based sampler as the default again, rename the
sampler option so the default is no longer labeled "legacy", and stop using
wall-clock as a live stopping condition for `pyabc_smc` and
`abc_smc_baseline` in scaling. Instead, run those methods with explicit small
simulation/generation budgets and evaluate them at wall-time budgets purely in
post-processing by truncating completed work to `budget_s`.

This keeps pyABC on its natural execution model, avoids the MPI teardown path
triggered by wall-time stopping, and preserves fair wall-time comparison to
`async_propulate_abc`.

## Background and Reasoning

### What the cluster runs showed

- The first suspicious behavior appeared after wall-time budgets were wired into
  scaling runs for `abc_smc_baseline` and `pyabc_smc`.
- In repeated cluster `--test` runs, both methods often appeared to finish on
  rank 0, but the SLURM job remained alive until the allocation limit.
- The failure pattern was scale-sensitive:
  - packed `1,4` jobs often showed long post-finish tails but still progressed
  - single `48`-worker jobs repeatedly froze after a rank-0 pyABC finish log
- The stuck point was consistently after `abc.run()` returned on rank 0, not in
  scaling finalization or postprocessing.

### Why the first hypotheses seemed plausible

- The initial strong suspicion was pyABC oversubmission via
  `ConcurrentFutureSampler(client_max_jobs=200)`.
- That was reasonable because `client_max_jobs > n_workers` creates a deep queue
  of speculative futures, which can leave orphan work behind when pyABC stops
  early on wall time.
- On this cluster MPI stack, `cancel_futures=True` did not appear to fully
  retract already-dispatched work.
- This led to the first mitigation:
  - add `pyabc_client_max_jobs`
  - default it to `n_workers`
  - reduce speculative backlog dramatically

### What that ruled out, and what it did not

- Bounding `client_max_jobs` helped somewhat, but the 48-rank hang remained.
- The next hypothesis was double shutdown:
  - our code called `executor.shutdown(...)`
  - `MPICommExecutor.__exit__` also calls shutdown
- That was also a reasonable suspicion, so shutdown ownership was simplified.
- But the hang still remained at 48 ranks after removing the explicit inner
  shutdown.
- That ruled out our extra shutdown logic as the primary cause.

### Why `MappingSampler` looked like the next logical step

- After backlog reduction and shutdown cleanup failed, the next suspect was the
  futures model itself.
- `MappingSampler` was attractive because it removes speculative asynchronous
  future submission while keeping pyABC’s own sampling logic.
- The implementation switched MPI pyABC defaults to `MappingSampler` via
  `executor.map`, keeping the same `MPICommExecutor` transport.

### What the MappingSampler run proved

- The `test8` cluster run showed that `pyabc_mpi_sampler=mapping` was
  definitely active in the 48-worker job.
- Despite that, the 48-worker job still froze immediately after rank 0 logged
  that `abc.run()` had returned and the code was exiting the MPI executor
  context.
- That falsified the “speculative futures backlog is the primary cause” theory.
- At that point the remaining common factor was no longer the pyABC sampler
  strategy, but `mpi4py.futures.MPICommExecutor` / `MPIPoolExecutor` under
  wall-time-triggered pyABC teardown on this cluster.

### Why not continue chasing `mpi4py.futures`

- It is possible that a direct `COMM_WORLD` master/worker implementation could
  avoid the executor issue.
- But that would be a substantial custom MPI control path and no longer the
  natural pyABC execution model.
- The user preference here is to compare against async ABC using pyABC in the
  more standard futures-backed mode, which is also conceptually closer to the
  async execution model we are comparing against.

### Why the plan changes both the sampler default and the wall-time policy

- The mapping experiment showed that switching samplers alone does not solve the
  cluster hang.
- The common trigger across the bad runs was the live wall-time stop path.
- The older pyABC futures mode had worked before wall-time-driven scaling was
  introduced, which suggests the unstable part is not steady-state execution but
  stopping pyABC at arbitrary wall-clock boundaries under MPI.
- Therefore the plan restores the natural futures-based sampler as default, but
  removes wall-time as a live stop condition for pyABC methods in scaling.

### Why wall-time remains the right comparison axis

- The fairness goal is to compare methods at equal wall-clock budgets.
- That does not require every method to terminate exactly at the budget
  boundary.
- For sync pyABC methods, it is acceptable to:
  - run to their configured simulation/generation cap
  - then evaluate only work completed before each wall-time budget
- This preserves wall-time-based comparison while avoiding the cluster failure
  mode triggered by early wall-time termination.

### Why the plan keeps budgets small instead of adding a new estimator

- The repo already has explicit budget controls:
  - `max_simulations`
  - `n_generations`
  - scaling `max_simulations_policy`
  - test/small-mode clamps
- There is currently no online estimator that predicts the required pyABC
  simulation budget from a target wall-clock budget.
- Introducing a new estimator now would add another moving part without solving
  the core cluster issue.
- The smaller and safer change is:
  - keep pyABC execution budgets explicit and modest
  - stop live wall-time interruption
  - continue comparing by post hoc wall-time truncation

## Behavior and Interface Changes

- Rename `pyabc_mpi_sampler` values to:
  - `concurrent_futures` as the default MPI pyABC sampler
  - `mapping` as the explicit fallback/debug sampler
- Support a temporary compatibility alias:
  - accept `concurrent_futures_legacy`
  - normalize it to `concurrent_futures`
  - emit a deprecation warning
- Keep `parallel_backend="mpi"` unchanged.
- Keep `pyabc_client_max_jobs`, active only for `concurrent_futures`.
- Do not use `max_wall_time_s` as a live stop control for `pyabc_smc` or
  `abc_smc_baseline` in scaling runs.
- Keep `max_wall_time_s` for methods that genuinely stop by wall clock,
  especially `async_propulate_abc`.

## Implementation Changes

- In `experiments/async_abc/inference/pyabc_sampler.py`:
  - change the default MPI sampler back to `concurrent_futures`
  - rename the current "legacy" branch to `concurrent_futures`
  - keep `mapping` selectable but non-default
  - add alias handling for `concurrent_futures_legacy`
  - move warnings so only `mapping` or the deprecated alias warns
- In `experiments/async_abc/inference/pyabc_wrapper.py` and
  `experiments/async_abc/inference/abc_smc_baseline.py`:
  - make the futures path the normal MPI path again
  - keep `pyabc_client_max_jobs = n_workers` as the default queue-depth cap
  - keep the MPI barrier/teardown safeguards already in place
- In `experiments/scripts/scaling_runner.py`:
  - `_stop_policy_for_method` (lines 589-592) currently returns `wall_time_exact`
    for `async_propulate_abc`, `abc_smc_baseline`, and `pyabc_smc`. Remove
    `abc_smc_baseline` and `pyabc_smc` from the `wall_time_exact` branch; return
    `simulation_cap_approx` for them instead. This is a 1-2 line change.
  - No other injection logic needs changing: the block at lines 830-833 already
    gates `max_wall_time_s` injection on `stop_policy == "wall_time_exact"`.
  - Continue writing `wall_time_limit_s` and `wall_time_budgets_s` into
    metadata and budget summaries — those are unaffected by this change.
- Audit scaling config `max_simulations_policy` values (`min_total`,
  `per_worker`, `k_factor`) before executing: compute what total simulation
  count `_requested_max_simulations` would produce at 48 workers and compare
  to a rough runtime estimate. If the result is clearly too large for practical
  cluster runtimes without wall-time stopping, reduce the policy values directly
  in the scaling config. No new estimation logic — just set smaller explicit
  numbers.
- In `experiments/async_abc/io/config.py`:
  - Remove the auto-inflation: `inference.setdefault("n_generations", 1000)`
    (lines 147-148 in `_annotate_mode()`). This is unconditionally a footgun —
    do not narrow, remove entirely.
  - Remove the companion warning (lines 88-97 in `_validate()`): "n_generations
    is low for a wall-time-limited run." Once the auto-inflation is gone and
    pyABC scaling runs no longer receive `max_wall_time_s`, this warning is dead
    code for the scaling path and misleading for other paths.
- In pyABC method configs and experiment configs:
  - keep pyABC budgets small and explicit
  - use configured `max_simulations` and `n_generations` as the actual
    execution cap
  - do not introduce a new dynamic estimator in this change
- For scaling specifically:
  - keep using `_requested_max_simulations(...)`, but ensure its inputs remain
    modest in both full and test configs
  - if current scaling `max_simulations_policy` is larger than needed for the
    desired wall-time budgets, reduce that policy directly in config rather
    than adding a new estimator
- For other experiments:
  - rely on existing config budgets and test/small-mode clamps
  - do not auto-inflate pyABC generation counts just because wall-time
    metadata exists

## Wall-Time Evaluation Semantics

- Wall-time remains an analysis axis for all methods.
- For pyABC methods, budget summaries should be computed from completed work
  before `budget_s`:
  - attempts: only attempts with `sim_end_time <= budget_s`
  - posterior quality: last completed observable pyABC state with
    `wall_time <= budget_s`
  - throughput/utilization by budget: only completed work up to `budget_s`
- Do not add partial-generation reconstruction in this change.
- Keep sync pyABC quality curves generation-based; wall-time truncation only
  chooses the last completed generation before the budget.
- No new analysis or reporting code is expected for the post-hoc truncation:
  confirm during implementation that existing analysis already filters attempts
  by `sim_end_time` and quality states by `wall_time` before applying budget
  thresholds. If filtering is not in place, add it; otherwise this section
  requires no code changes.

## Current Simulation-Budget Logic

- The repo does not currently estimate the needed pyABC simulation count
  online.
- Most experiments use `config["inference"]["max_simulations"]` directly,
  subject to existing test/small-mode clamps.
- Scaling uses `_requested_max_simulations(...)` in
  `experiments/scripts/scaling_runner.py`, which picks the max of:
  - configured `max_simulations`
  - `max_simulations_policy.min_total`
  - `max_simulations_policy.per_worker * n_workers`
  - `max_simulations_policy.k_factor * k`
- The runtime estimation logic in
  `experiments/async_abc/utils/runner.py` is reporting-only; it does not set
  inference budgets.
- "Keep it small" in this change means keeping the configured pyABC budgets and
  scaling policy inputs modest, not adding a new estimation mechanism.

## Tests

- Sampler tests:
  - MPI default resolves to `concurrent_futures`
  - `mapping` remains selectable
  - deprecated `concurrent_futures_legacy` alias normalizes correctly with
    warning
- Scaling runner tests:
  - `async_propulate_abc` remains `wall_time_exact`
  - `pyabc_smc` and `abc_smc_baseline` no longer receive `max_wall_time_s`
    from scaling
  - budget summaries still use `wall_time_budgets_s`
- Config tests:
  - no automatic `n_generations=1000` injection tied to pyABC wall-time
    analysis
  - no warning expecting high `n_generations` for scaling pyABC runs
- Analysis/reporting tests:
  - pyABC budget metrics are derived from completed work before the time
    threshold
- Cluster validation:
  - rerun `--test` scaling jobs for `48` and packed `1,4`
  - confirm pyABC methods complete normally and budget summaries still align to
    the requested wall-time thresholds

## Assumptions

- `concurrent_futures` is the replacement name for the default MPI pyABC
  sampler.
- "Small way" means using explicit modest config budgets and existing
  test/small-mode clamps, not adding a new runtime-based simulation estimator.
- For scaling, if current `max_simulations_policy` is too large for practical
  runs, it should be reduced in config as part of this change.
- `concurrent_futures` + natural termination (i.e. pyABC stopping at its
  configured simulation/generation cap, not at a wall-time boundary) has NOT
  been explicitly validated at 48 ranks on this cluster after the backlog-cap
  and single-owner-shutdown fixes. The cluster `--test` validation is the gate
  for this assumption.
- Fallback: if the cluster `--test` run reveals that `concurrent_futures` still
  hangs at 48 ranks even with natural termination, revert the sampler default
  back to `mapping`. The wall-time policy change (removing live `max_wall_time_s`
  from pyABC configs) remains valid regardless of which sampler is default.
