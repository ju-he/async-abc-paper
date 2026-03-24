# Run 1 Audit: Replicate-Sharded Experiments

Date: 2026-03-24
Audited results root: `/home/juhe/remotes/scratch/herold2/async-abc/run1`
Audited code commit: `d93223b7b5361586c9674f695d751153395dfe89`

Scope:
- Included: `gaussian_mean`, `gandk`, `lotka_volterra`, `cellular_potts`, `runtime_heterogeneity`, `ablation`
- Excluded per request: `sbc`
- Not assessed here because still ongoing per request: `sensitivity`, `straggler`
- `scaling` was not part of `submit_replicate_shards.py --experiments all` at this commit, so it is out of scope for this submission audit

## Executive Summary

None of the six audited experiments completed as an intended full paper run.

What is actually merged into the experiment root directories is the earlier `--test` run only:
- every audited experiment has `data/metadata.json` with `run_mode: "test"` and `completed_replicates: [0]`
- every audited experiment has a test-mode shard run with `merge.done.json`
- none of the later full-mode shard runs have `merge.done.json`

The main execution issue is systematic:
1. The full run treated the earlier test replicate `0` as already completed, so it only submitted replicates `1..4`.
2. Those later full-mode shard runs did not finish cleanly.
3. Because the full-mode runs never merged, the visible plots and CSVs at the experiment root are test-only artifacts, not paper artifacts.

There are also real plotting/reporting problems independent of the failed runs:
1. PNG export failed everywhere: every plot metadata file has `"png": null`.
2. Benchmark posterior and corner plots collapse to one method only, because the reporter uses the single global minimum tolerance across all methods.
3. `runtime_heterogeneity` has a broken Gantt plot with negative times for `sigma > 0`.
4. `runtime_heterogeneity` and `cellular_potts` are missing configured/expected plots.

## Global Findings

### 1. Full runs did not continue cleanly after the test run

For all six audited experiments:
- the test run produced `replicate 0`
- the subsequent full run only submitted `replicates 1,2,3,4`
- the root-level merged outputs were never updated past the test run

This is not what we want for a paper run. The test artifact should not count as the completed full replicate `0`.

### 2. Full-mode failures by experiment

| Experiment | Full shard outcome |
| --- | --- |
| `gaussian_mean` | all 4 full shards hit SLURM time limit (`00:30:00`) |
| `gandk` | all 4 full shards hit SLURM time limit (`01:05:11`) |
| `lotka_volterra` | all 4 full shards hit SLURM time limit (`00:30:00`) |
| `cellular_potts` | shards 0 and 3 completed; shards 1 and 2 terminated during `pyabc_smc` with nonzero exit |
| `runtime_heterogeneity` | all 4 full shards hit SLURM time limit (`02:35:35`) |
| `ablation` | all 4 full shards hit SLURM time limit (`03:10:06`) |

Note: several failed full shards are still recorded as `"state": "running"` in `status.json`. Those status files are stale after termination.

### 3. Root plots are test-mode only

For all six experiments:
- only `replicate 0` is present in the root-level merged CSVs
- the plots therefore summarize a single tiny-budget test run
- these figures are diagnostic at best, not valid paper figures

### 4. PNG export is broken everywhere

All plot metadata files have `"png": null`, despite the checklist in `.plans/ressources/experiment_code_checklist.md` requiring PDF and PNG outputs.

## Per-Experiment Assessment

### `gaussian_mean`

Did the jobs run as intended?
- No.
- Test shard merged successfully.
- Full run only submitted replicates `1..4`, and all four full shards hit the `00:30:00` time limit.
- Root outputs are still test-only.

Visual plot inspection:
- The plots are internally readable as tiny diagnostics.
- `quality_vs_time` and `tolerance_trajectory` clearly come from a single replicate and tiny budget.
- Posterior and corner plots are not method comparisons; they show one method only.

Alignment with intended paper:
- Partial at best in plot type, not in evidentiary value.
- The paper wants posterior correctness and convergence comparisons. A single test replicate with ~100-budget settings is not sufficient.

Plotting/reporting errors:
- PNG export missing.
- `posterior_mu_data.csv` matches `abc_smc_baseline` only, so the posterior/corner reporter is collapsing across methods incorrectly.

### `gandk`

Did the jobs run as intended?
- No.
- Test shard merged.
- Full run only submitted replicates `1..4`, and all four full shards hit the `01:05:11` time limit.
- Root outputs are test-only.

Visual plot inspection:
- The plot set is structurally sensible, but clearly sparse and test-budget limited.
- Posterior marginals and the corner plot again show one posterior cloud, not a method comparison.
- `quality_vs_time` shows highly unstable single-replicate diagnostics.

Alignment with intended paper:
- Not paper-ready.
- The intended paper comparison needs proper posterior recovery and convergence evidence over full runs and multiple replicates.

Plotting/reporting errors:
- PNG export missing.
- Posterior/corner plots again collapse to one method only.

### `lotka_volterra`

Did the jobs run as intended?
- No.
- Test shard merged.
- Full run only submitted replicates `1..4`, and all four full shards hit the `00:30:00` time limit.
- Root outputs are test-only.

Visual plot inspection:
- The posterior plots are nearly flat/uniform.
- The tolerance schedule is essentially stuck at `1e6`.
- These are exactly what we would expect from an underpowered test run, not from a successful benchmark inference run.

Alignment with intended paper:
- No.
- The paper intends posterior recovery on this model. These figures do not show that.

Plotting/reporting errors:
- PNG export missing.
- Posterior/corner plots are not method comparisons.

### `cellular_potts`

Did the jobs run as intended?
- No.
- Test shard merged.
- Full run only submitted replicates `1..4`.
- Of those four full shards, two completed and two terminated during `pyabc_smc`; no final merge happened.
- Root outputs are still test-only.

Visual plot inspection:
- The posterior and corner plots are extremely sparse, with effectively only five accepted points.
- The tolerance plot is also clearly test-budget limited.
- These do not look like full inference results.

Alignment with intended paper:
- No.
- The paper intends this benchmark as the realistic workload case. The visible outputs are too sparse and incomplete to support that claim.

Plotting/reporting errors:
- PNG export missing.
- `quality_vs_time` is enabled in the config but absent from outputs.
- More generally, the visible plot set is incomplete relative to the configured benchmark diagnostics.

### `runtime_heterogeneity`

Did the jobs run as intended?
- No.
- Test shard merged.
- Full run only submitted replicates `1..4`, and all four full shards hit the `02:35:35` time limit.
- Root outputs are test-only.

Visual plot inspection:
- The only produced figure is `worker_gantt`.
- That Gantt plot is effectively broken: most bars are off to the far left, and the x-axis spans negative wall-clock time.

Alignment with intended paper:
- No.
- The paper wants computational-performance evidence under heterogeneous runtimes.
- The config requests `idle_fraction`, `throughput_over_time`, `idle_fraction_comparison`, and `gantt`, but only the Gantt plot exists.
- Even that Gantt plot is not trustworthy in its current form.

Plotting/reporting errors:
- PNG export missing.
- `worker_gantt_data.csv` contains negative `sim_start_time` / `sim_end_time` values for `sigma0.5`, `sigma1.0`, `sigma1.5`, and `sigma2.0`.
- The resulting Gantt figure is visually wrong.
- Requested summary plots are missing entirely.

### `ablation`

Did the jobs run as intended?
- No.
- Test shard merged.
- Full run only submitted replicates `1..4`, and all four full shards hit the `03:10:06` time limit.
- Root outputs are test-only.

Visual plot inspection:
- The ablation bar chart only has visible bars for `full_model` and `small_archive`.
- The other four variants are `NaN`, so the figure is visibly incomplete.

Alignment with intended paper:
- No.
- An ablation figure with four missing variants is not usable for the paper.

Plotting/reporting errors:
- PNG export missing.
- The bar chart silently includes missing variants as empty/NaN categories instead of flagging the run as incomplete.

## Concrete Plotting/Reporting Bugs

### A. Posterior/corner plots drop all but one method

Evidence:
- `gaussian_mean/plots/posterior_mu_data.csv` matches `abc_smc_baseline` only.
- The plot data files do not contain a `method` column.
- In `plotting/reporters.py`, `_final_population(records)` selects records at the single global minimum tolerance, not per method.

Impact:
- Benchmark posterior and corner plots are not comparing inference methods.
- This does not align with the intended paper narrative.

### B. Runtime heterogeneity Gantt plot has invalid negative times

Evidence from `runtime_heterogeneity/plots/worker_gantt_data.csv`:
- `async_propulate_abc__sigma0.0`: positive times around `0.005 .. 0.108`
- `async_propulate_abc__sigma0.5`: negative times around `-31.276 .. -31.173`
- `async_propulate_abc__sigma1.0`: negative times around `-31.841 .. -31.738`
- `async_propulate_abc__sigma1.5`: negative times around `-32.245 .. -32.142`
- `async_propulate_abc__sigma2.0`: negative times around `-32.654 .. -32.551`

Impact:
- The Gantt figure is visually wrong and not interpretable.

### C. PNG generation failed across the board

Evidence:
- every `*_meta.json` has `"png": null`
- job logs emit warnings like `Could not rasterize ... keeping PDF only`

Impact:
- The output does not satisfy the reproducibility checklist in `.plans/ressources/experiment_code_checklist.md`.

### D. Missing configured plots

`cellular_potts`:
- config enables `quality_vs_time`
- no `quality_vs_time` plot exists

`runtime_heterogeneity`:
- config enables `idle_fraction`, `throughput_over_time`, `idle_fraction_comparison`, `gantt`
- only `worker_gantt` exists

## Bottom Line

For the six audited experiments, the visible outputs do not represent successful paper runs.

The main reason is execution failure of the full-mode shard runs after the earlier test submission. On top of that, there are real plotting/reporting issues that should be fixed even after rerunning:
- do not let test outputs satisfy full-run replicate completion
- fix posterior/corner reporting so it compares methods correctly
- fix runtime-heterogeneity time normalization
- restore PNG export
- ensure configured plots are actually emitted, or fail loudly when prerequisites are missing
