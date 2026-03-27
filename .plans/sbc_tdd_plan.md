# SBC Fixes — Multi-Phase TDD Plan

## Context

The SBC experiment has four categories of problems identified in the critical review:
1. **Importance weights ignored** — SBC ranks use unweighted top-k archive particles, biasing results
2. **Only trivial 1D benchmark** — Gaussian mean SBC validates nothing scientifically interesting
3. **Rank histogram statistically underpowered** — 100 trials / 101 bins ≈ 1 count per bin
4. **Silent trial dropout** — convergence failures are silently dropped, no metadata record

Each phase follows TDD: write failing tests first, then implement until passing.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Weighted SBC ranks and coverage | ✅ Done |
| 2 | Multi-benchmark SBC (g-and-k) | ✅ Done |
| 3 | Statistical improvements (histogram bins + n_trials) | ✅ Done |
| 4 | Trial dropout diagnostics | ✅ Done |

---

## Phase 1 — Weighted SBC ranks and coverage

### What changes

**`experiments/async_abc/analysis/sbc.py`**
- Add `compute_rank_weighted(posterior_samples, weights, true_value, *, seed=None) -> int`
  - Normalise `weights`; use `np.random.default_rng(seed)` to resample `len(samples)` indices with replacement; compute rank in resampled array via `searchsorted`
  - If all weights are None / zero, fall back to `compute_rank`
- Update `sbc_ranks(trials)` to call `compute_rank_weighted` when the trial dict contains `"posterior_weights"`; fall back to `compute_rank` otherwise
- Update `empirical_coverage(trials, coverage_levels)` similarly: when `"posterior_weights"` is present, use the **same seed-derived resampled samples** (or a CDF-based computation) for the interval bounds

**`experiments/scripts/sbc_runner.py`**
- Update `_posterior_samples` to return `(samples, weights)` instead of only `samples`; or add a companion `_posterior_weights(records, *, archive_size)` helper
- Update `_extend_trial_records` to store `"posterior_weights"` alongside `"posterior_samples"` in each trial dict
  - For async: weights are `r.weight`; None entries → 1.0 fallback
  - For SMC: same — final-generation particle weights
- Update `_write_trial_records_jsonl` to serialise `posterior_weights` as a float list
- Update the jsonl loader in the finalizer to deserialise `posterior_weights`

**`experiments/async_abc/utils/shard_finalizers.py`** (`finalize_sbc_experiment`)
- Load and pass `posterior_weights` through the same merge/recompute path

### Tests to write first (`experiments/tests/test_sbc.py`)

```
test_compute_rank_weighted_uniform_weights_matches_unweighted
  — weights all 1.0 → same rank as compute_rank (within resampling noise)

test_compute_rank_weighted_concentrated_weight
  — single particle gets weight 1.0, rest 0.0 → rank is always 0 or L depending on position

test_compute_rank_weighted_none_weights_fallback
  — weights=None → falls back to compute_rank, no error

test_sbc_ranks_uses_posterior_weights_when_present
  — trial dict has "posterior_weights" → ranks differ from unweighted version when weights are non-uniform

test_sbc_ranks_falls_back_without_posterior_weights
  — trial dict has no "posterior_weights" → same result as before (backward compat)

test_empirical_coverage_weighted_calibrated_posterior
  — construct trials with known posterior + weights such that weighted coverage ≈ nominal;
    unweighted version of same data produces different (biased) coverage

test_trial_records_jsonl_roundtrip_with_weights
  — write trial records with posterior_weights, read back, check arrays match

test_posterior_samples_returns_weights
  — _posterior_samples returns weights matching r.weight from records
```

**Pass criteria**: all 8 new tests pass; all existing test_sbc tests still pass.

---

## Phase 2 — Multi-benchmark SBC (extend to g-and-k)

### Design decision

Add a `"benchmarks"` list inside the `"sbc"` config block. The runner loops over benchmarks; all outputs gain a `"benchmark"` column. The single `"benchmark"` top-level key is kept for backward compatibility and used as the first entry if `"sbc.benchmarks"` is absent.

### Config changes

**`experiments/configs/sbc.json`** — add `sbc.benchmarks`:
```json
{
  "benchmark": { "name": "gaussian_mean", ... },  ← kept for compat
  "sbc": {
    "n_trials": 100,
    "coverage_levels": [0.5, 0.8, 0.9, 0.95],
    "benchmarks": [
      { "name": "gaussian_mean", "n_obs": 100, "sigma_obs": 1.0,
        "prior_low": -5.0, "prior_high": 5.0,
        "inference_overrides": {}
      },
      { "name": "gandk", "n_obs": 200,
        "inference_overrides": {
          "max_simulations": 20000, "k": 200, "tol_init": 2.0
        }
      }
    ]
  }
}
```

**`experiments/configs/small/sbc.json`** — add `sbc.benchmarks` with reduced budgets:
```json
"benchmarks": [
  { "name": "gaussian_mean", ..., "inference_overrides": {} },
  { "name": "gandk", "n_obs": 100, "inference_overrides": { "max_simulations": 2000, "k": 50 } }
]
```

### Runner changes (`sbc_runner.py`)

- Extract benchmark list: `_resolve_benchmark_configs(cfg) -> list[dict]` that returns either `[cfg["benchmark"]]` (compat) or the benchmarks list
- Outer loop over benchmarks; inner loops over methods × trials (unchanged)
- Tag each trial record with `"benchmark": benchmark_name`
- Output CSVs (`sbc_ranks.csv`, `coverage.csv`) gain a `benchmark` column
- `_write_trial_records_jsonl` serialises `"benchmark"` field

### Plot changes (`plotting/sbc.py`)

- `plot_rank_histogram` — facet rows by `(benchmark, method)` instead of `method` alone; title per panel includes benchmark name
- `plot_coverage_table` — one line per `(benchmark, method)` combination; legend includes benchmark prefix

### Finalizer changes (`shard_finalizers.py`)

- `finalize_sbc_experiment` passes `benchmark` column through merge and recomputation

### Tests to write first

```
test_sbc_config_benchmarks_list_parsed
  — load sbc.json with benchmarks list; assert both gaussian_mean and gandk extracted

test_sbc_runner_loops_over_both_benchmarks
  — mock run_method_distributed; assert calls for both gaussian_mean and gandk trials

test_sbc_ranks_csv_has_benchmark_column
  — end-to-end in test mode; assert "benchmark" column exists in sbc_ranks.csv

test_sbc_coverage_csv_has_benchmark_column
  — same as above for coverage.csv

test_sbc_trial_records_jsonl_has_benchmark_field
  — check jsonl output includes "benchmark" in each row

test_plot_rank_histogram_multibenchmark_facets
  — ranks_df with two benchmark values → figure has one panel per (benchmark, method)

test_plot_coverage_table_multibenchmark_lines
  — coverage_df with two benchmarks → one line per benchmark × method combination

test_single_benchmark_compat_no_benchmarks_key
  — config without sbc.benchmarks key → falls back to cfg["benchmark"]; output has benchmark column
```

**Pass criteria**: all 8 new tests pass; test_sbc_full_config_treats_abc_smc_baseline_as_all_ranks_under_mpi still passes.

---

## Phase 3 — Statistical improvements (histogram bins + n_trials)

### Changes

**`experiments/async_abc/plotting/sbc.py`**
- `plot_rank_histogram`: replace `bins=min(max(bins, 5), 30)` with `bins=int(group["n_samples"].max()) + 1`; this gives `L+1` bins as required by the SBC literature
- Add a uniform reference line: `ax.axhline(len(group) / bins, color="grey", ls="--", lw=0.8, label="uniform")`
- Keep the `squeeze=False` grid but increase default panel height slightly: `figsize=(6, max(3.5, 3.2 * len(methods)))`

**`experiments/configs/sbc.json`**
- `sbc.n_trials`: 100 → **300** (improves coverage CI half-width from ±0.044 to ±0.025; rank histogram gets ~3 counts per bin with L=100)
- If this is too expensive, document that n_trials=100 is a known limitation and leave as-is pending compute budget decision

**`experiments/configs/small/sbc.json`**
- `sbc.n_trials`: 50 → **100** (small mode is already reduced, increasing to 100 keeps it reasonable)

### Tests to write first

```
test_rank_histogram_bins_equals_n_samples_plus_one
  — ranks_df with n_samples=50 → figure uses 51 bins (no cap)

test_rank_histogram_uniform_reference_line_present
  — check axes has at least 2 lines after plot_rank_histogram call (histogram + reference)

test_rank_histogram_large_n_samples_no_cap
  — ranks_df with n_samples=200 → uses 201 bins, not capped at 30
```

**Pass criteria**: 3 new tests pass; existing test_sbc_plot_metadata_is_complete still passes.

---

## Phase 4 — Trial dropout diagnostics

### Changes

**`experiments/scripts/sbc_runner.py`**
- `_extend_trial_records`: when `samples.size == 0`, log a warning with `trial_idx`, `method`, `benchmark`; return `False` to signal dropout
- Accumulate per-benchmark, per-method dropout count in a dict `trial_dropouts`
- Pass dropout info to `write_metadata` via `extra`

**`experiments/async_abc/utils/write_metadata`** (wherever it writes): ensure `extra` dict is serialised into `metadata.json` — this likely already works.

**Output**: `metadata.json` gains field `"trial_dropouts": {"gaussian_mean": {"async_propulate_abc": 0, "abc_smc_baseline": 0}, ...}`

### Tests to write first

```
test_extend_trial_records_empty_samples_logs_warning
  — patch logger; call _extend_trial_records with empty records; assert logger.warning called

test_extend_trial_records_returns_false_on_dropout
  — function returns False when samples are empty

test_metadata_includes_trial_dropout_counts
  — run sbc_runner in test mode; read metadata.json; assert "trial_dropouts" key present

test_dropout_count_nonzero_when_method_returns_empty
  — mock run_method_distributed to return []; run runner; metadata shows dropout count > 0
```

**Pass criteria**: 4 new tests pass; test_sbc_runner_test_mode still passes.

---

## Summary of file changes

| File | Phase |
|------|-------|
| `async_abc/analysis/sbc.py` | 1 (weighted ranks/coverage) |
| `scripts/sbc_runner.py` | 1 (weights in trial records), 4 (dropout logging) |
| `async_abc/utils/shard_finalizers.py` | 1 (weights in finalizer), 2 (benchmark column) |
| `configs/sbc.json` | 2 (benchmarks list), 3 (n_trials) |
| `configs/small/sbc.json` | 2 (benchmarks list), 3 (n_trials) |
| `async_abc/plotting/sbc.py` | 2 (multibenchmark facets), 3 (bin fix) |
| `tests/test_sbc.py` | all phases |

## Verification

1. `python -m pytest experiments/tests/test_sbc.py -v` — all existing + new tests pass
2. `python experiments/scripts/sbc_runner.py --config experiments/configs/sbc.json --output-dir /tmp/sbc_test --test` — exits 0, produces `sbc_ranks.csv` with `benchmark` column, `coverage.csv` with both `gaussian_mean` and `gandk` rows, `metadata.json` with `trial_dropouts`
3. `python -m pytest experiments/tests/ -v` — full test suite passes (no regressions)
4. Visually inspect `rank_histogram.png`: panels for both benchmarks × methods, `L+1` bins, uniform reference line
5. Visually inspect `coverage_table.png`: 4 lines (2 benchmarks × 2 methods), 95% Wilson bands

## Implementation order within each phase

Within each phase: write all tests first (they should fail), then implement until they pass, then verify no regressions.
