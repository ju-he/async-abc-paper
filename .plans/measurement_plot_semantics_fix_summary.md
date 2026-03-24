# Measurement And Plot Semantics Fix Summary

## Goal

Fix the experiment pipeline so benchmark conclusions are based on method-correct measurements and plots instead of mixed or contaminated states.

The concrete goals were:

- replace the old "minimum tolerance rows" heuristic with a canonical final posterior reconstruction API
- isolate pyABC/ABC-SMC sweep conditions so one condition cannot reuse another condition's database
- repair misleading plot semantics in posterior, corner, archive/tolerance, runtime heterogeneity, and straggler outputs
- fix missing async tolerance exports and improve metadata provenance
- make the small `lotka_volterra` target quality nontrivial

## Plan Summary

### 1. Canonical final-state reconstruction

Introduce one shared helper used by reporters and SBC.

- `async_propulate_abc`: reconstruct final archive from all records in one replicate using `epsilon_final = min(non-null tolerance)`, keep `loss < epsilon_final`, sort by `(loss, wall_time, step)`, cap by `k`
- `abc_smc_baseline` / `pyabc_smc`: use the final generation population per replicate
- `rejection_abc`: use accepted posterior records per replicate, capped by `k`
- pooled plots concatenate per-replicate final states after reconstructing them independently

### 2. Plot semantic repairs

- posterior and corner plots should use the canonical final-state API and export per-method sample counts
- `archive_evolution` should become a tolerance-vs-attempts plot, separated by method and replicate
- `tolerance_trajectory` should deduplicate repeated state points and report missing tolerance instrumentation
- benchmark diagnostics should stop emitting the legacy `quality_vs_time` alias by default
- sensitivity heatmaps should represent all configured dimensions explicitly instead of averaging one away

### 3. Sweep isolation and runtime measurement fixes

- pyABC DB paths should include method, replicate, seed, and `_checkpoint_tag`
- runtime heterogeneity should compare a unified `utilization_loss_fraction`
  - async: worker idle fraction
  - sync: barrier overhead fraction
- straggler outputs should separate async worker Gantt views from sync generation/barrier timelines

### 4. Provenance and tolerance fixes

- async raw records should always export an effective tolerance once one is known
- git hash metadata should be resolved from the repo root, not the caller's cwd
- small-mode `lotka_volterra` should use a stricter `target_wasserstein`

## Implemented Changes

### Final-state analysis

Added [final_state.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/analysis/final_state.py).

- added `FinalStateResult`
- added `base_method_name`
- added `final_state_results`
- added `final_state_records`
- implemented method-specific final-state reconstruction

Exported these helpers from [analysis/__init__.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/analysis/__init__.py).

### Reporter and plotting changes

Updated [reporters.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/plotting/reporters.py).

- `_final_population()` now delegates to canonical final-state reconstruction
- `plot_posterior()` and `plot_corner()` now use final reconstructed states and write `sample_counts` metadata
- `plot_archive_evolution()` now writes tolerance vs simulation attempts, with one series per method/replicate, while keeping the `archive_evolution.*` filename
- `plot_tolerance_trajectory()` now relies on deduplicated state points and records `missing_tolerance_methods`
- `plot_idle_fraction()` and `plot_idle_fraction_comparison()` now use unified utilization-loss summaries with `measurement_method`
- `plot_quality_vs_wall_time_diagnostic()` no longer emits the deprecated alias unless explicitly requested
- `plot_benchmark_diagnostics()` now emits only the non-legacy benchmark diagnostics by default
- added `plot_generation_timeline()` for sync generation/barrier views
- `plot_worker_gantt()` now marks async-only worker timelines more explicitly
- `plot_sensitivity_summary()` now preserves the `scheduler_type` and `tol_init_multiplier` dimensions instead of averaging them away

Updated [trajectory.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/analysis/trajectory.py).

- `tolerance_over_wall_time()` now deduplicates repeated state points
- added `tolerance_over_attempts()`

Updated [common.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/plotting/common.py).

- `sensitivity_heatmap()` now supports 4D faceting
- exported long-form CSV data with:
  - `tol_init_multiplier`
  - `scheduler_type`
  - `k`
  - `perturbation_scale`
  - `value`

### SBC changes

Updated [sbc_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/sbc_runner.py).

- `_posterior_samples()` now uses canonical final-state reconstruction
- trial record extraction now passes `archive_size`
- rank-histogram binning now uses the maximum sample count per group instead of a fragile first-row assumption

Updated [shard_finalizers.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/utils/shard_finalizers.py).

- aligned rank histogram binning with SBC runner
- straggler finalization now emits separated async worker and sync generation timeline outputs

### Sweep isolation

Updated [abc_smc_baseline.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/abc_smc_baseline.py) and [pyabc_wrapper.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/pyabc_wrapper.py).

- DB filenames now include:
  - method
  - replicate
  - seed
  - sanitized `_checkpoint_tag`
- stale `.db`, `-wal`, and `-shm` files at that exact path are removed before starting a non-extended run

This prevents different straggler/heterogeneity conditions from reusing the same pyABC database.

### Async tolerance export

Updated [propulate_abc.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/propulate_abc.py).

- record export now carries forward the current effective tolerance when an individual record does not expose its own tolerance

This is intended to stop async `lotka_volterra` exports from producing all-null tolerance columns.

### Metadata provenance

Added [git.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/utils/git.py).

- added `find_repo_root()`
- added repo-root-based `get_git_hash()`

Updated [export.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/plotting/export.py) and [metadata.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/utils/metadata.py) to use the shared helper.

### Small config adjustment

Updated [lotka_volterra.json](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/small/lotka_volterra.json).

- changed small-mode `analysis.target_wasserstein` from `50.0` to `0.5`

### Straggler output split

Updated [straggler_runner.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/straggler_runner.py).

- worst-case straggler plotting now separates:
  - async worker Gantt output
  - sync generation timeline output

## Test Updates

Updated [test_plotting.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/tests/test_plotting.py).

- added final-state reconstruction regression coverage
- added 4D sensitivity heatmap coverage
- added tagged DB path coverage
- added runtime heterogeneity export coverage for `measurement_method`
- added sync generation timeline coverage
- added tolerance-trajectory deduplication coverage
- updated benchmark-diagnostic expectations to match removal of the default legacy alias

Updated [test_sbc.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/tests/test_sbc.py).

- added a regression test for `_posterior_samples()` using reconstructed async final archives

Updated [test_runners.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/tests/test_runners.py).

- updated the gaussian runner expectation to check the non-legacy benchmark diagnostics instead of `quality_vs_time.pdf`

## Verification

Verified with the requested venv:

```bash
nastjapy_copy/.venv/bin/pytest \
  experiments/tests/test_plotting.py \
  experiments/tests/test_sbc.py \
  experiments/tests/test_runners.py::TestGaussianMeanRunner::test_creates_phase3_plots \
  -q
```

Result:

- `66 passed in 39.86s`

## Remaining Work

This change set fixes the measurement/plotting plan above, but I did not finish a full-suite cleanup.

When I ran the full test suite earlier, the remaining failures were outside this measurement/plotting slice:

- scaling-factor expectations in `experiments/tests/test_config.py`
- a shard-count expectation in `experiments/tests/test_sharding.py`

Those are separate from the implemented plan and were not addressed here.
