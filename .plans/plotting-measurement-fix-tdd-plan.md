# Plotting, Measurement, And Summary Plot TDD Plan

## Status

- [x] Phase 0: lock the reproduction surface
- [x] Phase 1: add audit and guardrails
- [x] Phase 2: fix measurement semantics
- [x] Phase 3: split paper summaries from diagnostics
- [x] Phase 4: upgrade existing summary experiments
- [x] Phase 5: add config/interface defaults
- [x] Phase 6: validate and sign off

## Summary

- Reproduce the known runtime heterogeneity, straggler, Lotka-Volterra, and summary-plot issues with tests first.
- Repair recorded measurement semantics before changing paper-facing visuals.
- Use `nastjapy_copy/.venv/bin/python -m pytest ...` for all validation.
- Keep replicate-level figures as diagnostics, but make canonical plot names emit paper-facing summaries.

## Phase 0: Reproduction

### Goals

- Add focused regression tests for:
  - runtime heterogeneity finalize/plot aggregation stability
  - straggler throughput semantics
  - pyABC loss/tolerance semantics
  - benchmark summary plotting behavior
  - silent skips for missing true-parameter quality plots

### Notes

- Existing plotting tests already cover mixed runtime measurement rows. Extend them to enforce grouped aggregation behavior and summary-vs-diagnostic outputs.
- Existing sharding/runner tests already cover straggler and runtime heterogeneity smoke paths. Add targeted regressions rather than broad new integration suites first.

## Phase 1: Audit And Guardrails

### Deliverables

- Benchmark audit export with method/replicate validity signals.
- Explicit skip metadata for invalid paper-facing plots.
- Quality/threshold plots must only emit when `true_params` are available and the audit allows them.

### Data to export

- `method`
- `replicate`
- `final_tolerance`
- `tolerance_monotone`
- `wall_time_span`
- `attempt_count_span`
- `final_posterior_size`
- `fallback_or_extinction_fraction`
- `has_true_params`
- `paper_quality_plots_allowed`
- `paper_threshold_plots_allowed`
- `invalid_reason`

## Phase 2: Measurement Semantics

### Deliverables

- Fix runtime heterogeneity aggregation by grouping over `measurement_method`.
- Fix straggler throughput to use active simulation span as the canonical denominator.
- Preserve `elapsed_wall_time_s` as a diagnostic field.
- Stop using epsilon as a synthetic fallback loss for pyABC-derived records.
- Re-audit Lotka after the recording fix and gate any retuning behind an explicit pilot diagnostic.

## Phase 3: Paper Summaries Vs Diagnostics

### Deliverables

- Canonical outputs become paper summaries:
  - `archive_evolution`
  - `tolerance_trajectory`
  - `quality_vs_wall_time`
  - `quality_vs_attempt_budget`
  - `quality_vs_posterior_samples`
  - `time_to_target_summary`
  - `attempts_to_target_summary`
- Replicate-level outputs move to:
  - `archive_evolution_diagnostic`
  - `tolerance_trajectory_diagnostic`
  - `quality_vs_wall_time_diagnostic`
  - `quality_vs_attempt_budget_diagnostic`
  - `quality_vs_posterior_samples_diagnostic`
  - `time_to_target_diagnostic`
  - `attempts_to_target_diagnostic`

### Summary defaults

- Mean + pointwise 95% confidence intervals.
- Right-continuous interpolation / forward-fill for step-like curves.
- Tolerance aggregated in `log10(epsilon)` space and back-transformed.
- Threshold summaries ignore crossings before `analysis.min_particles_for_threshold`.

## Phase 4: Existing Summary Experiments

### Deliverables

- Sensitivity heatmap layout cleanup.
- Ablation mean + 95% CI bars.
- SBC coverage plot with Wilson/binomial intervals.
- Runtime heterogeneity paper summaries with uncertainty.
- Straggler throughput-vs-slowdown paper summaries with uncertainty.

## Phase 5: Config Defaults

### Defaults to add

- `plots.emit_paper_summaries: true`
- `plots.emit_diagnostics: true`
- `analysis.ci_level: 0.95`
- `analysis.min_particles_for_threshold: k if present else 100`

## Phase 6: Validation

### Validation order

1. Focused regression tests
2. `experiments/tests/test_plotting.py`
3. `experiments/tests/test_sharding.py`
4. `experiments/tests/test_runners.py`
5. Artifact-based replot/finalize checks against saved cluster outputs

### Commands

```bash
nastjapy_copy/.venv/bin/python -m pytest experiments/tests/test_plotting.py
nastjapy_copy/.venv/bin/python -m pytest experiments/tests/test_sharding.py
nastjapy_copy/.venv/bin/python -m pytest experiments/tests/test_runners.py
```

## Progress Log

- 2026-03-25: Created implementation plan file and aligned it to the confirmed venv path `nastjapy_copy/.venv`.
- 2026-03-25: Added benchmark audit export and skip-metadata support for invalid paper plots via `plot_audit.csv`, `plot_audit_summary.json`, and `_meta.json` skip records.
- 2026-03-25: Fixed extension compatibility normalization so new plotting/analysis defaults do not break `--add-replicates` shard submission against older metadata.
- 2026-03-25: Fixed runtime heterogeneity aggregation by separating `worker_idle` and `barrier_overhead` by `(sigma, base_method, measurement_method, replicate)` and faceting worker Gantt diagnostics by method.
- 2026-03-25: Fixed straggler throughput semantics to export and prefer `active_wall_time_s` while preserving `elapsed_wall_time_s` as a diagnostic field; summary plot now shows mean + CI.
- 2026-03-25: Removed epsilon-as-loss fallback from pyABC-derived records and wired `tol_init` through both pyABC SMC wrappers.
- 2026-03-25: Converted benchmark canonical plots to paper summaries and preserved replicate-level diagnostics under explicit `*_diagnostic` names.
- 2026-03-25: Added threshold guardrail `analysis.min_particles_for_threshold` to convergence logic and made threshold plots skip with explicit metadata when the target is not reached under that guard.
- 2026-03-25: Upgraded sensitivity heatmap layout, ablation summaries, SBC coverage summaries, and runtime/straggler paper plots to include uncertainty-aware summary outputs.
- 2026-03-25: Added regression coverage in `test_analysis.py`, `test_plotting.py`, `test_runners.py`, `test_sbc.py`, and confirmed `test_sharding.py` remains green under the new defaults.
- 2026-03-25: Validation completed for `experiments/tests/test_analysis.py`, `experiments/tests/test_plotting.py`, `experiments/tests/test_sharding.py`, `experiments/tests/test_runners.py`, and `experiments/tests/test_sbc.py` with `nastjapy_copy/.venv/bin/python -m pytest ...`.
- 2026-03-25: Added retry support for `--finalize-only` after shard batches that failed during finalization, which allowed the saved `runtime_heterogeneity` batch `run_20260324_222442` to publish a top-level output tree successfully.
- 2026-03-25: Added `replot.py` coverage for SBC so saved `sbc_ranks.csv` and `coverage.csv` can regenerate paper/diagnostic plots without rerunning inference.
- 2026-03-25: Added Lotka-specific `tol_init` diagnostics and tightened the audit to mark pathologically fallback-dominated runs invalid for paper-facing quality/threshold plots.
- 2026-03-25: Artifact validation against `/home/juhe/remotes/scratch/herold2/async-abc/small2` confirmed:
  - `runtime_heterogeneity` finalization now succeeds and emits top-level plots/data plus `plot_audit.csv`.
  - `sbc` replot now refreshes `coverage_table_data.csv` with `n_trials`, `empirical_coverage_ci_low`, and `empirical_coverage_ci_high`.
  - `lotka_volterra` now emits `lotka_tol_init_diagnostic.json`/`.csv`, with `recommended_tol_init ~= 407.69` and `pathological_fallback = true`.
  - `lotka_volterra` paper quality plots now skip with `skip_reason = plot_audit_failed` because all method/replicate rows are flagged `pathological_fallback_or_extinction`.

## Remaining Follow-up

- Optional operational next step: rerun the Lotka-Volterra benchmark with a calibrated `tol_init` near the diagnostic recommendation (`~408` rather than `500000`) and compare extinction/fallback rates before accepting any Lotka paper figure.
