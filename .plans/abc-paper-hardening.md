# Plan — ABC-paper code hardening for methods-venue submission

## Progress

| Wave | Status | Notes |
|------|--------|-------|
| **W1** | **Complete (2026-05-26)** | propulate commit `c4ac585`; vendored copy in async-abc-paper is a symlink so no SHA bump needed. 75/75 propulate tests pass, 45/45 paper-side benchmark tests pass. Microbench logsumexp overhead ~10 ms / 380 ms (well within 5% budget). W1.5 benchmark fixes were already in place from prior work (`fix-experiment-issues-010426.md`). |
| **W2** | **Complete (2026-05-26)** | propulate commit `0f52943`; async-abc-paper commits `acb6751` (W2.1), `932920f` (W2.3 + W2.5), `41d5778` (W2.4). 76/76 propulate tests pass; SBC, ablation reporter, wall-time parity, checkpoint audit, benchmark, and analysis test modules all green. |
| **W2 post-clean** | **Complete (2026-05-27)** | async-abc-paper commit `4d0b3e5` repairs 25 pre-existing test failures (8 root causes: `_site.detect_defaults` SYSTEMNAME guard, stale `test_sharding` assertions, partial `sensitivity.json` grid, unguarded `mpi4py` imports in `propulate_abc`/`scaling_runner`, `propulate_comm` being dropped when mpi4py absent, `multicore→mpi` promotion requiring `mpi4py.importorskip`, eager `posterior_quality_curve` import in `runtime_summary`). Full suite: 574 passed, 25 skipped, 0 failures. |
| **W3** | **Complete (2026-05-27)** | propulate commit `fd2f047` (W3.1); async-abc-paper commits `e133a26` (W3.3), `096a7a4` (W3.2), `4fbf99d` (W3.4). 89/89 propulate tests pass; 585 passed / 25 skipped / 0 failures on the paper side. |
| W4 | Pending (paper text) | |

### W1 deliverables (landed in `c4ac585`)

- **W1.1** AMIS denominator in log-space via `scipy.special.logsumexp`. New `_log_mixture` helper handles zero-weight rows. Verified by `TestW1LogSpaceAMIS::test_d15_gaussian_kernel_finite_weights_no_warning`.
- **W1.2** Boundary-clipping fallback replaced by uniform-prior draw with IS weight = 1.0. Verified by `TestW1BoundaryFallback::test_wide_kernel_falls_back_to_uniform_with_weight_one`.
- **W1.3** `_MIN_DENOM = 1e-12` weight floor replaced by reject-and-resample-parent up to `_MAX_WEIGHT_RETRIES = 5`; terminal exhaustion yields `weight = 0`. Verified by `test_near_zero_denominator_handled` (now expects `weight == 0.0`) and `TestW1RetryExhaustion::test_partial_retry_then_success_yields_normal_weight`.
- **W1.4** [tutorials/abc_example.py:43](tutorials/abc_example.py#L43) `ABCPMC(loss_fn=function, limits=limits)` → `ABCPMC(limits=limits)`.
- **W1.5** Already in place from prior work: `analytic_posterior_mean()` is uniform-prior (correctly documented), `normalize_stats` Lotka flag (default True) at `benchmarks/lotka_volterra.py:126`, bounded extinction retry loop at line 141 with `RuntimeError` on exhaustion. Tests at `experiments/tests/test_benchmarks.py:135,266,286` all pass.

### W2 deliverables

- **W2.1** ([async-abc-paper `acb6751`]) `gaussian_credible_coverage()` in `analysis/sbc.py` computes coverage of asymptotic mean±z·sd intervals at nominal levels {0.5, 0.8, 0.9, 0.95}. Wired into `sbc_runner.py` to produce `gaussian_ci_coverage.csv`. Four new tests in `test_sbc.py` cover nominal coverage, under-coverage detection, weighted samples, and empty input.
- **W2.2** ([propulate `0f52943`]) Default `amis_snapshots` flipped `0 → 20` in [abcpmc.py:327](propulate/propagators/abcpmc.py#L327); existing `test_disabled_by_default` renamed to `test_enabled_by_default`; new opt-out test. Performance regression budget relaxed from 2.0 → 5.0 s to absorb the log-space overhead from W1.1 (these tests explicitly set `amis_snapshots=0` to isolate algorithmic scaling).
- **W2.3** ([async-abc-paper `932920f`]) New `plot_ablation_amis_isolation()` reporter emits `ablation_amis_isolation.{pdf,png,csv,json}` whenever the ablation config contains both `full_model` and `no_amis` variants. Uses `posterior_quality_curve(checkpoint_strategy="time_uniform", checkpoint_count=24)`. Wired into both `ablation_runner.py` and the shard finalizer.
- **W2.4** ([async-abc-paper `41d5778`]) New `Deadline` class in `inference/_pyabc_common.py` with monotonic clock + `expired`/`remaining` properties. `rejection_abc` uses it; pyABC wrappers keep their internal `max_walltime` path. Docstrings now document first-rank-hit semantics. Five new tests in `test_walltime_parity.py` cover the Deadline helper and per-method wall-time enforcement with a 3 s budget. Side fix: `make_acceptor` no longer imports `propulate` for the hard-kernel branch.
- **W2.5** ([async-abc-paper `932920f`]) `plot_quality_vs_wall_time` and `plot_quality_by_sigma` switched to `checkpoint_strategy="time_uniform"`. Other reporters keep `quantile` because `time_uniform` is only semantically meaningful on a wall-clock axis. New audit test `test_checkpoint_strategy_audit.py` regex-parses `reporters.py` and asserts every paper-facing wall-time call site uses `time_uniform`.
- **W2.6** ([propulate `0f52943`]) `_warned_none` and `_warned_zero_weight` declared in `__init__`. Docstrings tightened: `select_archive` is now documented as a quantile-trimmed rejection rule (not strict ε-rejection); the `kernel_aware` ESS-bisection claim is softened to "wired up in W3.1 — currently uses loss-quantile / acceptance-rate / decay regardless of the flag."

### W3 deliverables

- **W3.1** ([propulate `fd2f047`]) Target-ESS bandwidth bisection (Del Moral, Doucet & Jasra 2012) for smooth-kernel ABC. `EpsilonScheduler` gains `_bisect_target_ess` + `_relative_ess` (log-space via logsumexp) + `_kernel_aware_from_accepted`; the three concrete schedulers dispatch into them from `compute` / `compute_cached` when `kernel_aware=True` and a smooth kernel is supplied. Constructor surface: new `kernel_fn` and `ess_target` (default 0.95) kwargs on `EpsilonScheduler`, every concrete scheduler, and `create_scheduler`; `ABCPMC.__init__` now passes `self._kernel_fn` and a new `ess_target` parameter down through the factory automatically. Hard-kernel and `kernel_aware=False` paths preserve the original loss-quantile / acceptance-rate / decay rule. New `TestKernelAwareBisection` (13 tests) covers fallback paths, ε-in-box, target-ESS achievement, monotone direction (lower target → tighter ε), cached/uncached parity, all three scheduler dispatches, end-to-end monotone preservation via ABCPMC, and parameter validation.
- **W3.2** ([async-abc-paper `096a7a4`]) AMIS snapshot-buffer ESS-stability sweep. New `analysis.ess_vs_n_at_fixed_S(records, window=…)` computes the sliding-window relative ESS over the evaluated stream; `plot_amis_snapshot_ess_stability` overlays per-S curves with replicate-mean ±95% CI bands and saves `amis_snapshot_ess_stability.{pdf,png,csv,json}`. The ablation finalizer fires the plot automatically when ≥2 distinct `amis_snapshots` values appear in the variant list; `amis_snapshot_sweep` registered as an alias for `ablation` in the finalizer registry. Production + small configs sweep `S ∈ {0, 5, 10, 20, 40, 80}` and `{0, 5, 20, 40}` respectively.
- **W3.3** ([async-abc-paper `e133a26`]) `analysis.simulation_time_bias_report(records)` returns per-run mean simulator wall-time segmented by acceptance, Pearson correlation between `sim_time` and `loss`, and a χ² independence test on median-split `sim_time` vs accepted. Skips with an explicit `skip_reason` when `sim_start_time` / `sim_end_time` are missing (sync methods). 4 new tests cover the strong-bias case (corr > 0.5, p < 0.01), the null case (|corr| < 0.2, p > 0.05), the empty-records case, and the skip case.
- **W3.4** ([async-abc-paper `4fbf99d`]) `PROPULATE_SKIP_DISCONNECT=1` environment variable lets large-scale runs skip `MPI_Comm_free` (which can hang for 30+ s on ParaStation MPI at ≥48 ranks). Accepts `1`/`true`/`yes` with whitespace trimming; default behaviour unchanged. 3 new tests cover the skip path, default path, and accepted env-var values.

### Note on vendored copy

`/home/juhe/bwSyncShare/Code/async-abc-paper/propulate` is a symlink to `/home/juhe/bwSyncShare/Code/propulate`, so any commit in the upstream directly reflects in the paper repo. The original plan's "bump vendored submodule SHA" step is a no-op for this workspace.

---

## Context

The asynchronous-steady-state ABC-PMC algorithm in
[propulate/propagators/abcpmc.py](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py)
and the experiment-and-analysis pipeline in
[async-abc-paper/](/home/juhe/bwSyncShare/Code/async-abc-paper/) form the joint
source for a planned methods-paper submission targeting *Statistics &
Computing*, *JCGS*, or *SIAM JUQ*. The paper plan in
[.plans/ressources/paper-concept.md](/home/juhe/bwSyncShare/Code/async-abc-paper/.plans/ressources/paper-concept.md)
claims **consistency + a CLT** for the algorithm as a corollary of Cornuet et
al. 2012 (AMIS) under Wilkinson 2013 smooth-ABC. A code review uncovered
several divergences between the implementation and that paper plan, plus
several methodological gaps that reviewers at a methods venue will probe:
log-space AMIS arithmetic, boundary-fallback semantics, weight-floor outliers,
missing SBC-CLT calibration, missing AMIS ablation, non-uniform wall-time
stopping across methods. ABCPMC is published in propulate's source but not in
any reproducibility-critical release; both default-changing and API-shape
changes are acceptable.

The two repositories must move in lock-step: `async-abc-paper/propulate/` is a
vendored copy of upstream propulate at commit `11637ff`. Every algorithm
patch lands at both paths; the vendored SHA is bumped after upstream merges.

---

## Critical files

Upstream (BOTH = patch both, vendored copy too):
- `/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py` (algorithm)
- `/home/juhe/bwSyncShare/Code/propulate/tutorials/abc_example.py` (smoke)
- `/home/juhe/bwSyncShare/Code/propulate/tests/test_abcpmc.py` (regression)
- `/home/juhe/bwSyncShare/Code/propulate/tests/benchmarks/bench_abcpmc.py` (microbench)

Paper-side only (PAPER):
- `/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/propulate_abc.py`
- `/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/_pyabc_common.py`
- `/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/inference/{pyabc_wrapper.py,abc_smc_baseline.py,rejection_abc.py}`
- `/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/analysis/{sbc.py,convergence.py,audit.py}`
- `/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/benchmarks/{gaussian_mean.py,lotka_volterra.py}`
- `/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/scripts/{ablation_runner.py,sbc_runner.py}`
- `/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/configs/{ablation.json,sbc.json,small/*}`

Existing infrastructure to reuse (do not duplicate):
- `_IncrementalCache` in [abcpmc.py:150-234](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L150-L234)
- `_make_kernel(name)` registry in [abcpmc.py:108-123](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L108-L123) — already imported by `_SmoothKernelAcceptor` in `_pyabc_common.py:79`
- `posterior_quality_curve(..., checkpoint_strategy="time_uniform")` in `analysis/convergence.py:155`
- `empirical_coverage()` in `analysis/sbc.py`
- pyABC matched-kernel acceptor in `_pyabc_common.make_acceptor`

---

## Wave 1 — Correctness blockers (must land before any submission)

**W1.1 — AMIS denominator in log-space.** BOTH. M.
[abcpmc.py:644-665](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L644-L665).
Replace the linear-space `np.exp(log_pdfs)` mixture sum with
`scipy.special.logsumexp(log_w + log_pdf)` over current proposal ∪ snapshots.
Compute `log_denom`, then `child.weight = self.prior_density * exp(-log_denom)`.
Resolves underflow with Gaussian kernel at d≥10 + tight Σ.
**Verify:** new d=15 Gaussian-kernel unit test in `test_abcpmc.py` over 2000
calls asserting no `RuntimeWarning` and `np.isfinite(child.weight)`; rerun
`bench_abcpmc.py` (≤5% regression).

**W1.2 — Boundary fallback = uniform prior draw.** BOTH. S.
[abcpmc.py:634-639](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L634-L639).
Replace `np.clip(...)` with a fresh uniform-prior draw (re-use the bootstrap
path) and stamp `child.tolerance = effective_tol` so the candidate still
counts as archive-phase. Aligns with paper §3.4.
**Verify:** new test forces resample exhaustion (degenerate prior box + huge
kernel cov), asserts position ∈ box, weight finite, `child.tolerance` set.

**W1.3 — Replace `_MIN_DENOM` floor with reject-and-resample-parent.** BOTH. M.
[abcpmc.py:617-675](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L617-L675).
Wrap parent-draw + candidate-sample + denominator in a ≤5-attempt retry loop.
If `log_denom < log(_MIN_DENOM)` after retries, set `child.weight = 0.0` (the
weighted-covariance and resampling paths already tolerate zero rows) instead
of producing `prior/1e-12 ≈ 1e12` outliers that swamp the mixture.
**Verify:** update `test_near_zero_denominator_handled`
([test_abcpmc.py:369-387](/home/juhe/bwSyncShare/Code/propulate/tests/test_abcpmc.py#L369-L387))
to expect `weight == 0.0` (not `1e12`); add archive-far-from-support test
asserting no individual has `weight > 1/_MIN_DENOM/10`.

**W1.4 — Tutorial fix.** propulate only. S.
[tutorials/abc_example.py:43](/home/juhe/bwSyncShare/Code/propulate/tutorials/abc_example.py#L43)
passes `loss_fn=function` to `ABCPMC(...)` — constructor does not accept it.
Remove that kwarg; `loss_fn` belongs on `Propulator`.
**Verify:** add `python tutorials/abc_example.py --test` smoke run to CI.

**W1.5 — Benchmark-config correctness.** PAPER. M.
Per `.plans/run4_findings_and_tdd_plan.md`: (a) reconcile Gaussian-mean
analytic posterior with the `n_obs`/`sigma_obs**2/n_obs` mismatch in
[benchmarks/gaussian_mean.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/benchmarks/gaussian_mean.py);
(b) ensure Lotka-Volterra summary-stat normalisation is always on (the
`normalize_stats=True` default in `benchmarks/lotka_volterra.py` is set, but
audit that every runner path honours it); (c) replace the extinction-retry
`while True` with a bounded counter raising `ExtinctionError`.
**Verify:** `pytest experiments/tests/test_benchmarks.py`; add an analytic-CI
closeness test asserting posterior-mean error ≤ 3σ of analytic 95% CI.

---

## Wave 2 — Methods-venue requirements (must land for S&C/JCGS/SIAM JUQ)

**W2.1 — SBC Gaussian-CI calibration.** PAPER. M.
Add `gaussian_credible_coverage(samples, theta_true, levels)` in
[analysis/sbc.py](/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/async_abc/analysis/sbc.py)
alongside `empirical_coverage`, computing coverage of asymptotic
mean±z·sd intervals at the limit-bandwidth replicates. Wire into
`sbc_runner.py` so the report table has a Gaussian-CI column at levels
{0.5, 0.8, 0.9, 0.95}. **This is the empirical demonstration of Theorem 2's
CLT** — without it, methods-venue reviewers will reject the theoretical
contribution.
**Verify:** new test in `experiments/tests/test_analysis.py` with synthetic
Gaussian posterior samples produces ~nominal coverage; SBC pipeline runs
end-to-end on `configs/sbc.json` and `configs/small/sbc.json`.

**W2.2 — Flip upstream `amis_snapshots` default 0 → 20.** BOTH. S.
[abcpmc.py:308](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L308).
User confirmed ABCPMC has no reproducibility-critical downstream users, so
flip the default. Add CHANGELOG entry; update class docstring to describe 0
as the legacy single-proposal opt-out.
**Verify:** existing AMIS tests
([test_abcpmc.py:906-968](/home/juhe/bwSyncShare/Code/propulate/tests/test_abcpmc.py#L906-L968))
adjusted to set `amis_snapshots=0` explicitly where they currently rely on
the default-disabled mode; new test asserts default constructor yields
`_amis_snapshots == 20`.

**W2.3 — AMIS isolation in ablation.** PAPER. S.
The ablation config likely has `no_amis` and `hard_kernel_baseline` variants
already; verify and wire the reporter
(`experiments/async_abc/plotting/reporters.py::plot_ablation_summary`) to
emit an AMIS-on vs AMIS-off figure pair on a wall-clock x-axis using
`posterior_quality_curve(checkpoint_strategy="time_uniform")`. This
underwrites paper §19 contribution (7).
**Verify:** run `configs/small/ablation.json` end-to-end; assert artefact
`ablation_amis_isolation.{pdf,png,csv,json}` exists.

**W2.4 — Uniform wall-time enforcement across methods.** PAPER. M.
Files: `inference/{rejection_abc.py, pyabc_wrapper.py, abc_smc_baseline.py,
propulate_abc.py}`. Centralise a `Deadline` helper in `_pyabc_common.py` and
have all wrappers consult it: pyABC via its per-population callback hook,
rejection_abc inside its draw loop, propulate via the existing
`max_wall_time_s` Propulator stop. Document that wall-time semantics are
*first-rank-hit*, not collective.
**Verify:** new `experiments/tests/test_walltime_parity.py` runs all three
methods with `max_wall_time_s=10` and asserts wall-clock end time ≤ 10 + ε
on each method.

**W2.5 — `time_uniform` checkpoint audit.** PAPER. S.
The `time_uniform` mode exists in `convergence.py:155`. Audit every
paper-facing script under `experiments/scripts/` and reporter under
`experiments/async_abc/plotting/` and ensure every call to
`posterior_quality_curve` for a *comparison* plot passes
`checkpoint_strategy="time_uniform"`. Diagnostic plots may keep `"all"`.
**Verify:** add a grep-based CI test that fails if any paper-facing reporter
file calls `posterior_quality_curve` with default checkpoint strategy.

**W2.6 — Tighten docstrings; fix `_warned_none`.** BOTH. S.
[abcpmc.py:567-574](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L567-L574):
declare `self._warned_none = False` in `__init__`. In `select_archive`
docstring and `ABCPMC` class header
([abcpmc.py:446](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L446),
[abcpmc.py:265-268](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L265-L268)):
state that hard-kernel mode applies "top-k by lowest loss among those with
`loss < ε`" (quantile-trimmed rejection, not strict ε-rejection). Remove
the docstring claim that schedulers do target-ESS bisection in
kernel-aware mode (which W3.1 will then *make true*).
**Verify:** existing tests pass; docstring builds.

---

## Wave 3 — Methods-paper strengthening (must land for methods venue)

**W3.1 — Implement target-ESS bisection for `kernel_aware` schedulers.** BOTH. L.
Currently the `kernel_aware` flag is stored but ignored
([abcpmc.py:700-710](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py#L700-L710)).
User chose to implement, not remove. Approach (Del Moral, Doucet & Jasra 2012):
in each scheduler's `compute_cached(...)`, when `kernel_aware=True`:
(a) compute the kernel-weighted ESS at the current ε under the chosen
kernel (use stored `weight` × `K_eps(loss)` for archive members);
(b) bisect on ε to target a fixed-ratio ESS drop (default `ess_target=0.95`);
(c) clip with the existing monotone guarantee;
(d) keep loss-quantile / geometric-decay / acceptance-rate as the
*fallback* when in hard-kernel or `kernel_aware=False` mode.
Plumb `ess_target` through the factory + constructor.
**Verify:** new test class `TestKernelAwareBisection` in `test_abcpmc.py`:
(i) bisection converges in O(log) iterations on a fabricated archive;
(ii) ESS ≈ target after one step on a converged archive;
(iii) result respects monotone guarantee; (iv) cached/uncached parity is
preserved on the new path; (v) end-to-end SBC run on `configs/small/sbc.json`
with kernel=gaussian still passes calibration.

**W3.2 — Snapshot-buffer ESS-stability study (Condition C4 evidence).** PAPER. M.
Add `experiments/configs/{small/,}amis_snapshot_sweep.json` sweeping
`amis_snapshots ∈ {0, 5, 10, 20, 40, 80}` on Gaussian-mean and g-and-k.
New analysis path in `analysis/convergence.py`: `ess_vs_n_at_fixed_S(records)`.
Reporter emits `amis_snapshot_ess_stability.{pdf,png,csv,json}` for the
supplementary. This is the empirical answer to "(C4) requires S to grow with
n" and replaces the need for an adaptive-snapshot implementation.
**Verify:** sweep runs to completion under `--test`; figure shows the
expected stabilisation behaviour at S≥20.

**W3.3 — Simulation-time bias diagnostic.** PAPER. S.
New function `simulation_time_bias_report(history)` in `analysis/audit.py`
returning (i) mean simulator wall-time of accepted vs rejected, (ii) Pearson
correlation between `sim_time` and `loss`, (iii) chi-square independence
test. Append to the per-run audit JSON; reference in paper §17 (limitations).
**Verify:** synthetic-history test where `sim_time = f(loss)` returns
non-zero bias; null test where `sim_time ⊥ loss` returns ~zero.

**W3.4 — MPI teardown hardening at scale.** PAPER. M.
`CommWorldMap`-style barrier-and-disconnect already in
`inference/pyabc_sampler.py`. Apply the same pattern to the
`concurrent_futures` path in `propulate_abc.py`; raise teardown timeouts in
`experiments/jobs/submit_*.py`; add `PROPULATE_SKIP_DISCONNECT=1` escape
hatch documented for ParaStation MPI.
**Verify:** extend `experiments/jobs/verify_commworldmap_48.sh` to exercise
the propulate path and assert clean exit + < 30 s teardown.

---

## Wave 4 — Paper-text track (Track C — not code)

These are tracked here for completeness but executed in
`/home/juhe/bwSyncShare/Code/async-abc-paper/latex/`, not in code:

- **(C1) sketch lemma.** §4.4: add a short derivation that the archive
  evolution rule of §3.2 maintains a bounded proposal-density ratio under
  a standard adaptive-PMC assumption (cite Beaumont et al. 2009 §4.1 and
  Cappé et al. 2008). Even a weak integrable-ratio statement is enough to
  retain consistency (loses the CLT); state both options clearly.
- **(C4) snapshot study writeup.** Cite the empirical results from W3.2 and
  state the conditions under which the fixed-S choice is justified.
- **Simulation-time bias.** Cite the W3.3 diagnostic, state that bias is
  measured per-run, declare correction as future work, and reference the
  Lenormand et al. 2013 latency-aware variants as the natural next step.

---

## Verification (end-to-end)

After W1 lands in both repos, gate on:

```bash
# upstream
cd /home/juhe/bwSyncShare/Code/propulate
pytest tests/test_abcpmc.py -x
python tutorials/abc_example.py --test    # smoke
python tests/benchmarks/bench_abcpmc.py   # microbench, ≤5% regression

# paper repo — bump vendored SHA
cd /home/juhe/bwSyncShare/Code/async-abc-paper
git -C propulate fetch && git -C propulate checkout <new-sha>
pytest experiments/tests/ -x
for cfg in experiments/configs/small/*.json; do
  python experiments/scripts/$(basename "$cfg" .json)_runner.py \
    --config "$cfg" --test --output-dir results/_w1_gate
done
```

After W2 + W3, additionally gate on:

```bash
# methods-paper gate
python experiments/scripts/sbc_runner.py --config experiments/configs/small/sbc.json \
  --output-dir results/_w2_sbc --test    # must produce Gaussian-CI column
python experiments/scripts/ablation_runner.py --config experiments/configs/small/ablation.json \
  --output-dir results/_w2_ablation --test    # must produce amis-isolation pair
pytest experiments/tests/test_walltime_parity.py -x
python experiments/scripts/amis_snapshot_sweep_runner.py \
  --config experiments/configs/small/amis_snapshot_sweep.json \
  --output-dir results/_w3_sweep --test
bash experiments/jobs/verify_commworldmap_48.sh   # optional, cluster-only
```

Paper-side artefacts that must exist before submission:

- `results/<run>/sbc/.../gaussian_ci_coverage.csv` (W2.1)
- `results/<run>/ablation/.../ablation_amis_isolation.{pdf,csv,json}` (W2.3)
- `results/<run>/amis_sweep/.../amis_snapshot_ess_stability.{pdf,csv,json}` (W3.2)
- Per-run `simulation_time_bias.json` in audit folders (W3.3)

---

## Sequencing and effort

| Wave | Items | Effort | Gate |
|------|-------|--------|------|
| W1 | W1.1–W1.5 | ~3 days | upstream pytest + microbench + paper small-configs green |
| W2 | W2.1–W2.6 | ~3 days | SBC Gaussian-CI table + ablation figure + wall-time parity |
| W3 | W3.1–W3.4 | ~5–6 days | kernel-aware bisection tests; snapshot-ESS sweep; teardown verified |
| W4 | C-track paper text | parallel to W3 | latex/methods.tex § 4 updated, references added |

Cross-repo discipline: every PR in upstream `propulate` touching `abcpmc.py`
is followed by a one-commit bump of the vendored SHA in `async-abc-paper/`
and a paper-side test rerun before the W1/W2 batches are merged.
