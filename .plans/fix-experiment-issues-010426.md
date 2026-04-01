  ---
  Multi-Phase TDD Plan: Fixing Experiment Issues

  Test runner: nastjapy_copy/.venv/bin/python -m pytest experiments/tests/ -x -q
  Baseline: 534 passed, 1 skipped

  ---
  Phase 1: Wall-Time Stopping for rejection_abc (+ config validation)

  Goal: Make rejection_abc support max_wall_time_s so all 4 methods can be stopped on wall time.

  Tests first (in test_inference.py):
  - test_rejection_abc_respects_max_wall_time_s: Create a slow simulate_fn (sleeps 0.05s), set max_wall_time_s=0.2, verify the method stops before exhausting max_simulations
  - test_rejection_abc_without_wall_time_runs_full_budget: Ensure backward compatibility when max_wall_time_s is absent

  Implementation (rejection_abc.py):
  - Extract max_wall_time_s from inference_cfg.get("max_wall_time_s")
  - Add time.time() - run_start >= max_wall_time_s check inside the while loop at line 60
  - Change stopping condition to: while sim_count < max_sims and len(accepted) < k and (max_wall_time_s is None or elapsed < max_wall_time_s)

  Also in this phase — validate scheduler_type and benchmark.name:
  - Test: test_invalid_scheduler_type_raises and test_invalid_benchmark_name_raises in test_config.py
  - Impl: Add checks in config.py:_validate() against VALID_SCHEDULER_TYPES and VALID_BENCHMARK_NAMES

  ---
  Phase 2: n_generations Safety Net for Sync Baselines

  Goal: Ensure n_generations is always high enough to never be the binding constraint when wall time is the intended stopping criterion.

  Tests first (in test_config.py):
  - test_config_with_wall_time_warns_low_n_generations: When max_wall_time_s is set and n_generations <= 10, emit a warning
  - test_config_wall_time_sets_default_high_n_generations: When max_wall_time_s is set and n_generations is absent, default to 1000 instead of 5

  Tests (in test_inference.py):
  - test_abc_smc_baseline_uses_high_n_generations_with_wall_time: With max_wall_time_s=0.5 and n_generations=1000, verify the run stops on time, not on generation cap

  Implementation:
  - In abc_smc_baseline.py line 258: When max_wall_time_s is set, default n_generations to 1000 instead of 5
  - In config.py:_validate(): Add a warning when max_wall_time_s is set but n_generations is suspiciously low (< 50)

  ---
  Phase 3: Add max_wall_time_s to All Benchmark Configs + Test-Mode Clamping

  Goal: Every benchmark experiment uses wall-time as the primary stopping criterion.

  Tests first (in test_config.py):
  - test_test_mode_clamps_max_wall_time_s: Verify test mode reduces max_wall_time_s to a small value (e.g. 30s)
  - test_all_benchmark_configs_have_wall_time: Parametrized test over all 4 benchmark configs verifying max_wall_time_s is present

  Implementation:
  - Update schema.py:_TEST_MODE_OVERRIDES_TEMPLATE to add "max_wall_time_s": 30 under clamp.inference
  - Update configs:
    - gaussian_mean.json: add "max_wall_time_s": 300, set "n_generations": 1000
    - gandk.json: add "max_wall_time_s": 600, set "n_generations": 1000
    - lotka_volterra.json: add "max_wall_time_s": 600, set "n_generations": 1000
    - cellular_potts.json: add "max_wall_time_s": 3600, set "n_generations": 1000
    - Update small/ variants proportionally
  - Update straggler.json, sbc.json, ablation.json, sensitivity*.json similarly (add wall time, raise n_generations)

  ---
  Phase 4: Equalize Checkpoint Granularity in Convergence Analysis

  Goal: Make quality curves and time-to-threshold comparisons fair by using time-based checkpoints for all methods.

  Tests first (in test_analysis.py):
  - test_time_based_checkpoints_equal_density_across_methods: Create async records (100 events) and sync records (5 generations) over the same wall-time span. With checkpoint_mode="time_uniform" and checkpoint_count=20, verify both
  methods get exactly 20 checkpoints
  - test_time_based_sync_checkpoints_use_locf: Between generations, the sync state should be the previous generation's population (last-observation-carried-forward)
  - test_time_based_async_checkpoints_subset_of_full: Time-based checkpoints should produce a subset of the all-events reconstruction, with the same Wasserstein values at matching times
  - test_posterior_quality_curve_checkpoint_mode_parameter: Verify the new checkpoint_mode parameter is threaded through

  Implementation (convergence.py):
  - Add checkpoint_mode: str = "native" parameter to posterior_quality_curve() and _observable_quality_rows()
  - When checkpoint_mode="time_uniform", both _async_archive_rows and _sync_generation_rows emit checkpoints at the same evenly-spaced time grid
  - For async: at each checkpoint time t, build the archive from the prefix of events with wall_time <= t (same logic as now, but only at grid points — reduces O(N²) to O(K×N))
  - For sync: at each checkpoint time t, use the most recent completed generation as the state (LOCF)
  - Default "native" preserves backward compatibility

  ---
  Phase 5: Fix Benchmark Bugs

  Goal: Fix the prior mismatch in Gaussian Mean, the unnormalized summary stats in Lotka-Volterra, and the fragile extinction fallback.

  5a: Gaussian Mean prior mismatch

  Test first (in test_benchmarks.py):
  - test_analytic_posterior_uses_uniform_prior: Verify analytic_posterior_mean() uses the same Uniform prior as the ABC inference. For a symmetric setup (prior_low=-5, prior_high=5, true_mu=0), this is trivially observed_mean, but for an
  asymmetric prior it should differ.

  Implementation (gaussian_mean.py):
  - Add analytic_posterior_mean_uniform() that computes the correct posterior under the Uniform prior (which is just the MLE truncated to the prior bounds for a flat prior — simply observed_mean clipped to [prior_low, prior_high])
  - Deprecate or document analytic_posterior_mean() as being under a Gaussian prior, not the ABC prior

  5b: Lotka-Volterra summary statistics normalization

  Test first (in test_benchmarks.py):
  - test_lotka_volterra_summary_stats_balanced_scale: Verify all 6 summary statistics contribute meaningfully to the distance. Generate two parameter sets that differ only in the log-CV dimension; verify the distance changes by a
  non-trivial amount (>1% of total)
  - test_lotka_volterra_normalized_stats_backward_compatible_with_flag: Old behavior preserved via a normalize_stats config flag

  Implementation (lotka_volterra.py):
  - Add a normalize_stats config option (default True for new runs)
  - When enabled, normalize each summary statistic by the observed value: (sim_stat - obs_stat) / (|obs_stat| + epsilon) — making the distance unitless and balanced
  - When disabled (backward compat), use current raw Euclidean

  5c: Lotka-Volterra extinction retry

  Test first (in test_benchmarks.py):
  - test_lotka_volterra_retries_multiple_times_on_extinction: Mock _gillespie to return extinction for seeds 42 and 43 but succeed for 44. Verify the benchmark initializes successfully.
  - test_lotka_volterra_raises_after_max_retries: If all retries fail, raise a clear error

  Implementation (lotka_volterra.py):
  - Replace the single retry with a loop of up to max_retries=10, incrementing the seed each time
  - Raise RuntimeError if all retries produce extinction

  ---
  Phase 6: Code Quality Fixes

  Goal: Eliminate the worst code duplication and dead code.

  6a: Extract shared pyABC code

  Test first:
  - test_shared_pyabc_db_suffix_matches_old_behavior: Verify the extracted _db_suffix() and _prepare_db_path() produce the same results as the current duplicated versions

  Implementation:
  - Create experiments/async_abc/inference/_pyabc_common.py
  - Move _db_suffix(), _prepare_db_path(), and the shared model/sampler/transition setup into it
  - Update pyabc_wrapper.py and abc_smc_baseline.py to import from _pyabc_common

  6b: Fix dead code in method_registry.py

  Test first:
  - test_method_execution_mode_unknown_raises: Verify method_execution_mode("nonexistent") raises KeyError with a clear message

  Implementation (method_registry.py):
  - Remove unreachable return "rank_zero" on line 124
  - Simplify the guard at line 118 (remove the always-true if name not in METHOD_REGISTRY since it's guaranteed at that point)

  6c: Fix target_wasserstein in small Lotka-Volterra config

  Implementation: Change configs/small/lotka_volterra.json target_wasserstein from 0.5 to 50.0

  6d: Remove unused import math from sensitivity.py

  6e: Harmonize <= to < in rejection_abc.py line 66

  Test first (in test_inference.py):
  - test_rejection_abc_uses_strict_less_than: Verify a particle with loss == tol_init is NOT accepted (consistent with all other methods)

  ---
  Phase 7: Documentation & Metric Transparency

  Goal: Document the Wasserstein-to-point-mass metric, the archive reconstruction semantics, and the state_kind distinction.

  Tests first (in test_analysis.py):
  - test_quality_row_includes_state_kind: Verify the state_kind column is present and correct ("archive_reconstruction" vs "generation_population" vs "accepted_prefix")
  - test_wasserstein_documented_in_quality_curve_output: Verify the returned DataFrame includes a docstring-accessible description (via function docstring)

  Implementation:
  - Add comprehensive docstrings to _wasserstein_to_true_params(), _async_archive_rows(), _sync_generation_rows()
  - Ensure state_kind is visible in plot legends via reporters.py
  - Document in convergence.py module docstring that Wasserstein is W1-to-point-mass (mean absolute deviation from truth in 1D)

  ---
  Execution Order & Dependencies

  Phase 1 (wall-time for rejection_abc + validation)
      |
  Phase 2 (n_generations safety net)
      |
  Phase 3 (configs + test-mode clamping)  -- depends on 1 & 2
      |
  Phase 4 (checkpoint equalization)       -- independent of 1-3
      |
  Phase 5a,5b,5c (benchmark bugs)        -- independent, can parallelize
      |
  Phase 6a-6e (code quality)             -- independent
      |
  Phase 7 (documentation)               -- last, after all code changes

  Each phase: write failing tests → implement → verify all 534+ tests still pass.
