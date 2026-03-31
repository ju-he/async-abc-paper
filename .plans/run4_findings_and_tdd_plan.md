# Run4 Findings And TDD Plan

Date: 2026-03-31

Scope:
- Results inspection for `/home/juhe/remotes/scratch/herold2/async-abc/run4`
- Current cluster job state assessment
- Root-cause analysis for why async is not clearly better
- Multi-phase TDD plan to fix the measurement, comparison, and implementation issues

## Executive Summary

The main conclusion is not that Propulate is fundamentally wrong. The stronger conclusion is:

1. Several comparisons are not apples-to-apples because the methods often consume very different realized simulation budgets.
2. Some summary metrics are currently invalid or misleading.
3. There is likely a pyABC/MPI finalization or coordination bug causing some jobs to remain `running` after inference has already finished.
4. Async already looks better in some fairer wall-time comparisons, especially for smaller and medium archive sizes.
5. Some experiments do not currently answer the intended scientific question because the analysis contract is incomplete.

## Current Run State

Assessment was based on:
- shard `status.json`
- stdout file mtimes and tails
- shard-local `raw_results.csv`
- generated summary tables and plot data

### Clearly Still Running

- `13630012` (`sensitivity`)
- `13630006`, `13630010` (`runtime_heterogeneity`)
- `13630105` (`scaling`)

Evidence:
- fresh stdout updates around `21:18` to `21:19` CEST
- sensitivity logs show repeated `finish -> start` cycles, which is expected because each shard iterates over a large hyperparameter grid

Relevant files:
- [sensitivity_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/sensitivity_runner.py)
- [sensitivity.json](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/sensitivity.json)

### Finished Or Stale But Still Listed By SLURM

- `13629987` (`lotka_volterra`)
- `13630109` (`scaling 96`)

Evidence:
- shard status for `lotka_volterra` is already `completed`
- scaling-96 stdout ends with `Done in 3h 14m 46s`

### Very Likely Hung After Inference Finished

- `13629978`, `13629979` (`gaussian_mean`)
- `13629981`, `13629982` (`gandk`)
- `13630008` (`runtime_heterogeneity`)

Evidence:
- shard status still says `running`
- stdout stopped hours earlier
- stdout ends exactly at `pyabc_smc ... status=finish`
- shard-local `raw_results.csv` contains only `async_propulate_abc` rows while the pyABC SQLite DB exists

Interpretation:
- useful inference appears to have completed
- the likely failure point is record materialization, MPI cleanup, or outer coordination/finalization

### Suspicious, But Less Certain

- `13630111`, `13630113`, `13630115` (`scaling 128/192/256`)

Evidence:
- no stdout movement for a long time
- tail still shows only `status=start`

Interpretation:
- may be hung early in an all-ranks phase
- evidence is weaker than for the clearly finished-then-stuck jobs

## Job Abort Recommendation From The Inspection

Definite abort set:
- `13629987`
- `13629978`
- `13629979`
- `13629981`
- `13629982`
- `13630008`

Keep running:
- `13630012`
- `13630006`
- `13630010`
- `13630105`

Watch but do not immediately kill:
- `13630111`
- `13630113`
- `13630115`

## Findings By Experiment

## 1. Scaling

Most relevant outputs:
- [throughput_summary.csv](/home/juhe/remotes/scratch/herold2/async-abc-paper/home/juhe/remotes/scratch/herold2/async-abc/run4/scaling/data/throughput_summary.csv)
- [budget_summary.csv](/home/juhe/remotes/scratch/herold2/async-abc-paper/home/juhe/remotes/scratch/herold2/async-abc/run4/scaling/data/budget_summary.csv)
- [scaling_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/scaling_runner.py)

Key observation:
- async is often better at equal wall-time budget for `k=48` and often `k=192`
- async is not consistently better in final-stop summaries

Concrete examples from `quality_at_budget__all_methods_all_k__T60_data.csv`:
- `k=48, w=16`: baseline `0.437744`, async `0.305681`
- `k=48, w=32`: baseline `0.458228`, async `0.310796`
- `k=192, w=16`: baseline `0.531286`, async `0.399113`

But the final summaries are not fair because realized attempts differ sharply.

Examples from `throughput_summary.csv`:
- `k=48, w=16`: async uses about `3.09x` as many attempts as baseline
- `k=1000, w=16`: async uses about `0.25x` as many attempts as baseline

Implication:
- "final quality" is being compared at different effective compute budgets
- this alone can invert the apparent winner

### Scaling-Specific Measurement Bug

`worker_utilization` is currently not trustworthy.

In [scaling_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/scaling_runner.py#L418), utilization is computed from every timed record:

- both `simulation_attempt` rows
- and `population_particle` rows

For synchronous methods this double-counts work intervals and can produce values above `1.0`, which is impossible.

## 2. Runtime Heterogeneity

Relevant files:
- [runtime_heterogeneity.json](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/runtime_heterogeneity.json)

Current state:
- some shards are clearly still active
- one shard already marked itself completed on disk
- one shard looks finished-then-stuck

Interpretation:
- this experiment still contains useful signal
- the final high-variance synchronous baseline stage can become extremely slow
- this is the kind of regime where async should help, but the run is not yet cleanly finished

Important caveat:
- pyABC progress output under MPI is misleading here
- repeated `simulations=1` does not mean only one attempt has happened

## 3. Sensitivity

Relevant files:
- [sensitivity_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/sensitivity_runner.py#L161)
- [sensitivity.json](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/sensitivity.json#L26)

This is not hanging.

Each shard runs the full Cartesian grid:
- `k`: 3 values
- `perturbation_scale`: 3 values
- `scheduler_type`: 3 values
- `tol_init_multiplier`: 4 values

Total:
- `3 * 3 * 3 * 4 = 108` variants per shard

So repeated "restart" patterns in the stdout are expected.

## 4. Straggler

Relevant output:
- [throughput_vs_slowdown_summary.csv](/home/juhe/bwSyncShare/Code/async-abc-paper/home/juhe/remotes/scratch/herold2/async-abc/run4/straggler/data/throughput_vs_slowdown_summary.csv)

This experiment supports the async mechanism, but the headline metric is not a fair winner metric.

Observed confound:
- baseline finishes after roughly `1300` attempts
- async is forced to use `10000` attempts

So the raw throughput plot is showing different stopping points, not matched budgets.

Interpretation:
- good mechanism demo
- weak as a direct claim that async is globally better

## 5. Cellular Potts

Relevant file:
- [cellular_potts.json](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/cellular_potts.json)

Current issue:
- quality-vs-time style plots were skipped because the true parameter names do not match the inferred parameter names expected by analysis

Config currently uses:
- `true_division_rate_normalized`
- `true_motility_normalized`

But analysis expects `true_<param_name>` matching the actual parameter columns.

Interpretation:
- this is an analysis contract issue
- the experiment currently cannot support the intended async-vs-sync quality claim

## 6. SBC

Relevant output:
- [coverage.csv](/home/juhe/bwSyncShare/Code/async-abc-paper/home/juhe/remotes/scratch/herold2/async-abc/run4/sbc/data/coverage.csv)

This does not show a fundamental correctness failure in async.

Examples:
- `gaussian_mean`, `mu`, nominal `0.5`: baseline `0.753333`, async `0.536667`
- `gaussian_mean`, `mu`, nominal `0.8`: baseline `0.946667`, async `0.806667`

Interpretation:
- async is not obviously invalid
- calibration differs, but the evidence does not point to Propulate being fundamentally broken

## 7. Ablation

Relevant files:
- [ablation_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/ablation_runner.py)
- [ablation.json](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/ablation.json)

This is useful for tuning async itself, but it does not compare async vs sync.

Reason:
- the ablation config only runs `async_propulate_abc`

## Root Cause Analysis

## A. Wrong Comparison Axis

This is the biggest issue.

The current analysis often compares:
- different realized numbers of attempts
- different realized posterior sample counts
- different stopping conditions

That means "final quality" is not a fair direct comparison.

## B. Stop Rules Are Not Symmetric

Relevant files:
- [propulate_abc.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/propulate_abc.py#L71)
- [abc_smc_baseline.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/abc_smc_baseline.py#L122)

Propulate:
- converts `max_simulations` into a generation budget based on worker count in total-simulation mode

Baseline:
- uses pyABC with `max_total_nr_simulations` and fixed `n_generations`

These are not equivalent operational budgets.

## C. Logging And Summary Semantics Are Polluted

Known issues:
- invalid `worker_utilization`
- misleading pyABC MPI progress
- missing Cellular Potts quality comparison

These issues distort interpretation even when the underlying run may be fine.

## D. Likely Implementation Bug Is In The pyABC/MPI Path, Not In Propulate

Relevant files:
- [pyabc_sampler.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/pyabc_sampler.py)
- [pyabc_wrapper.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/pyabc_wrapper.py)
- [runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/utils/runner.py)

The pyABC path uses `ConcurrentFutureSampler` on top of MPI.

The hung-job pattern strongly suggests:
- pyABC finished inference
- then record extraction, MPI cleanup, or outer gather/finalization did not complete cleanly

This explains the `gaussian_mean`, `gandk`, and one `runtime_heterogeneity` shard much better than a Propulate algorithm bug.

## Multi-Phase TDD Plan

The guiding principle is:
- first make the failures reproducible
- then fix metrics and logging
- then fix the MPI hang
- only then retune algorithm budgets

## Phase 1. Freeze The Current Failures As Regression Tests

Add a `tests/regression/` slice with fixtures derived from the current `run4` outputs.

Initial failing tests:
- shard status says `running` while stdout ends at `pyabc_smc ... status=finish`
- scaling summary never reports `worker_utilization > 1.0`
- summaries always expose realized attempts and posterior sample counts
- Cellular Potts quality analysis fails loudly when true-parameter mapping is missing

Done when:
- these tests fail on the current implementation for the intended reasons

## Phase 2. Fix The Metrics Layer First

Start in:
- [scaling_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/scaling_runner.py)

Tests:
- utilization is computed only from `simulation_attempt` records
- synchronous methods with both attempt and population rows do not double-count
- all utilization values lie in `[0, 1]`

Implementation:
- derive utilization from attempt records only
- keep population-particle timing out of active-time accounting
- surface `realized_attempts` and `posterior_samples` explicitly

Done when:
- regenerated scaling summaries no longer contain impossible utilization values

## Phase 3. Make Head-To-Head Comparisons Fair By Construction

Start in:
- [propulate_abc.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/propulate_abc.py)
- [abc_smc_baseline.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/abc_smc_baseline.py)
- scaling and straggler analysis scripts

Tests:
- every summary row includes realized attempts, posterior samples, final tolerance, and wall time
- equal-wall-time and equal-realized-attempt summaries are generated from the same raw data
- plots labeled as method comparisons must state the comparison axis

Implementation:
- define three official comparison modes:
  - equal wall time
  - equal realized attempts
  - time to target quality
- keep "final quality at each method's own stopping rule" only as a secondary diagnostic

Done when:
- the analysis can separately answer:
  - who is better at equal wall time?
  - who is better per realized attempt?
  - who reaches target quality faster?

## Phase 4. Repair pyABC Progress Reporting

Start in:
- [pyabc_wrapper.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/pyabc_wrapper.py)
- [abc_smc_baseline.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/abc_smc_baseline.py)

Tests:
- progress counters are monotone
- progress output cannot remain at `simulations=1` when traced attempts are much larger
- finish logs match traced realized attempts

Implementation:
- drive progress from traced or aggregated attempt counts, not a root-local callback count
- distinguish submitted, completed, and accepted work
- include realized attempts and accepted particles in finish logs

Done when:
- runtime and scaling logs can be interpreted without having to inspect raw CSVs

## Phase 5. Isolate And Fix The MPI Finalization Hang

Start in:
- [pyabc_sampler.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/pyabc_sampler.py)
- [pyabc_wrapper.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/pyabc_wrapper.py)
- [runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/utils/runner.py)

Tests:
- MPI integration test for `gaussian_mean` where pyABC finishes and shard status becomes `completed`
- MPI integration test for `gandk` with the same guarantee
- timeout-based regression test for the "finished inference but shard still running" signature
- all ranks reach the post-run synchronization point

Implementation:
- add rank-aware logging around:
  - `abc.run()`
  - history extraction
  - record materialization
  - `allgather`
  - shard status writes
- ensure the MPI-backed pyABC sampler is fully torn down before outer gather/finalization
- if necessary, move record extraction to rank 0 after clean sampler shutdown
- ensure failures mark the shard as failed rather than leaving it forever `running`

Done when:
- the currently stuck `gaussian_mean` and `gandk` jobs can be reproduced and then complete cleanly

## Phase 6. Fix The Cellular Potts Analysis Contract

Start in:
- [cellular_potts.json](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/cellular_potts.json)
- the analysis code that resolves true parameter values

Tests:
- true parameters are resolved for every inferred Cellular Potts parameter
- quality-vs-time plots are generated rather than skipped

Implementation:
- either rename config keys to match parameter names
- or add an explicit mapping layer in analysis
- validate this mapping before plotting

Done when:
- Cellular Potts produces the same quality artifacts as the simpler benchmarks

## Phase 7. Add Experiment-Level Acceptance Tests

Goal:
- verify the scientific claims at a small deterministic scale

Tests:
- scaling generates equal-wall-time quality summaries for representative `k`
- runtime heterogeneity excludes incomplete shards from claims
- straggler reports clearly label whether the axis is equal wall time or equal attempts

Implementation:
- use small deterministic fixtures in CI
- keep full cluster reruns as manual validation

Done when:
- reruns cannot silently recreate the current ambiguity

## Phase 8. Revisit Algorithm Budgets Only After Instrumentation Is Trustworthy

This phase is intentionally last.

Tests:
- each `k` configuration produces a minimum acceptable posterior sample count for both methods
- large-`k` runs warn or fail if total simulations are too small relative to archive size

Implementation:
- add a budget sanity rule for large `k`
- consider scaling total simulations with `k`
- clearly label underpowered regimes

Done when:
- large-`k` behavior no longer mixes algorithm limits with budget starvation

## Recommended Execution Order

Run phases in this order:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7
8. Phase 8

Reason:
- reproducibility first
- measurement before interpretation
- interpretation before implementation tuning

## Practical Next Step

The highest-leverage next action is:
- implement Phase 1 and Phase 2 together

Reason:
- they will immediately remove the most misleading metrics
- and they create the regression harness needed to safely debug the MPI hang afterward
