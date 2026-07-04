# Critical Review: Async Steady-State ABC-SMC Paper, Experiments & Implementation

## Context

This is a *review*, not an implementation task. The user asked for a thorough, critical
assessment of (a) the paper concept in `.plans/ressources/paper-concept.md`, (b) the
experiment settings in `async-abc-paper/experiments/`, and (c) the async ABC implementation
in `/home/juhe/bwSyncShare/Code/propulate`, plus whether the visualization code produces
publication-ready figures. The sections below are findings + a prioritized remediation
backlog. Nothing here has been edited in the repos.

Sources verified directly: `paper-concept.md`, `propulate/propagators/abcpmc.py`,
`propulate/.plans/algorithm-review.md`, `.plans/bug-fixes/previous-fixes.md`, and the
experiment config/runner inventory.

---

## Overall verdict

The engineering is careful and the paper framing is sophisticated (AMIS + smooth-kernel
ABC corollary is a legitimate, publishable angle). **The single biggest risk is a
methodological mismatch between what the theory claims and what the code does.** The paper
positions the method as a *posterior approximator* with an AMIS consistency + CLT guarantee,
but the implementation has an **optimiser-leaning core** (top-`k`-by-loss archive) that
undermines the central claim. This must be reconciled before submission. The experiment
settings are mostly reasonable but have several thinness/consistency issues. The figure
pipeline is solid on plumbing but not yet dialed to journal publication standard.

---

## 1. Concept ⇄ implementation mismatch (highest priority)

### 1.1 Optimiser vs. posterior approximator (propulate `abcpmc.py`, issue A5)
The archive is **top-`k` by lowest loss** (`get_archive`, abcpmc.py:604–617). As low-loss
particles accumulate, the top-`k` collapses to a tight cluster around the best-fit region,
the weighted covariance shrinks, and the proposal kernel narrows toward a point mass. That
is correct behaviour for an *optimiser*, wrong for a *posterior sampler* — which is exactly
what the paper claims to be (§3, §4, "ABC-PMC"). This directly threatens:
- **Condition (C1)** of Theorem 1 (bounded proposal-density ratio `c ≤ q/π ≤ C`). A kernel
  that collapses to a tight cluster violates the *lower* bound `c` over most of the support —
  the proposal puts ~zero mass far from the mode while the prior does not. The paper even
  admits C1 is "the strongest condition… not proved." A collapsing top-`k` kernel makes it
  empirically false, not just unproven.
- The **CLT** (Theorem 2), which inherits C1.

**Recommendation:** Add the `archive_mode = "top_k" | "all_accepted"` switch from A5 and run
the *validity* benchmarks (Gaussian, g-and-k, SBC) in `all_accepted` mode. Keep `top_k` only
for the optimiser-style/HPC throughput story if desired, but the posterior-quality and SBC
claims should use `all_accepted`. Without this, an informed reviewer will reject the
consistency claim. This is the central decision flagged in `algorithm-review.md`'s
"Cross-cutting design decision."

### 1.2 Mixed weighting regimes (issue A1, still present)
Prior-phase children carry `weight = 1.0`; archive-phase children carry importance weights of
order `10³–10⁶`. Both can coexist in the top-`k` mixture with no smooth handover, and in the
*async* setting a slow-evaluated prior-phase particle can re-enter the archive *after*
archive-phase particles are established. The discriminator already exists (`tolerance is None`);
the fix (archive-phase-only kernel once `k` archive-phase particles exist) is low-risk. This
matters for the "smooth handover / no phase discontinuity" claim in the concept (§3 "Smooth
kernels remove the prior-vs-archive phase discontinuity") — that claim is **not currently
true** at the weighting level.

### 1.3 Monotone-tolerance broken by island migration (issue A2, still present)
The documented monotone-tightening invariant is reconstructed as `min(ind.tolerance)` over the
**active** subset. Migration/pollination set `active = False`; when the running-min individual
emigrates, reconstructed tolerance *widens* silently. The paper's bandwidth-schedule condition
(C3: monotone non-increasing `ε_n`) is then violated in exactly the island-model regime the
paper leans on for asynchrony. Fix is a single decreasing `_tolerance_floor` float (A2). Low
effort, directly supports a stated theorem condition.

### 1.4 Simulation-time bias (issue A6) — acknowledged but unmitigated
The paper §17 honestly lists this as a limitation and the heterogeneity experiment *measures*
it (`simulation_time_bias_report`). That is the right posture. But note the tension: §1.1's
top-`k` archive *amplifies* simulation-time bias (fast regions both over-arrive AND survive
top-`k`), so `all_accepted` mode also softens A6. Worth a sentence connecting the two.

### 1.5 Status of A3/A4 (resolved — do not re-flag)
`algorithm-review.md` predates the W1 hardening. The current code (abcpmc.py:672–768) already
replaced the A3 weight-blow-up with a 5-retry→`weight=0` scheme and the A4 boundary-clip with a
uniform-prior fallback. These are *fixed*; the stale review doc should be annotated to say so to
avoid future confusion.

---

## 2. Experiment settings — reasonable? (mostly, with caveats)

### Verified inventory
13 configs (11 registered in `run_all_paper_experiments.py`, 2 unregistered). Four benchmarks
(Gaussian, g-and-k 4D, Lotka-Volterra 4D, realistic simulator workload), validity (SBC), HPC (straggler,
runtime heterogeneity, scaling, scaling_realistic), method-analysis (sensitivity 144-grid, ablation,
amis_snapshot_sweep). Baselines: `async_propulate_abc` vs `abc_smc_baseline` (kernel-matched
sync) and `rejection_abc` (small problems). `pyabc_smc` exists but is no longer in active
configs.

### Issues to address
- **Replicate count = 5 everywhere.** For headline speedup/quality-CI claims with high-variance
  HPC timing, 5 replicates gives wide confidence bands and weak significance. Consider ≥10 for
  the headline heterogeneity/scaling figures (cheap benchmarks at least).
- **Two unregistered experiments.** `scaling_realistic` and `amis_snapshot_sweep` have full configs +
  submit scripts but are absent from `EXPERIMENT_REGISTRY` and from `test_all.sh` (hardcoded
  list). They won't run via the main orchestrator. Either register them or document why they're
  out-of-band. `amis_snapshot_sweep` is the *direct evidence for the paper's core AMIS claim*
  (it sweeps `S = 0,5,10,20,40,80` to justify the default `S = 20` and the "empirically tight at
  fixed S" assertion in §4.4/§17) — it should not be a second-class citizen.
- **Scheduler confound.** Lotka-Volterra + both scaling experiments use `geometric_decay`;
  Gaussian/g-and-k use `acceptance_rate`. Since scheduler type is *also* a sensitivity-analysis
  variable, using different schedulers per benchmark in the main comparisons mildly confounds the
  async-vs-sync story. Document the per-model tuning rationale explicitly, or hold scheduler fixed.
- **pyABC dropped from active configs.** The concept (§1, §2.1, contribution 3) makes
  "apples-to-apples vs **pyABC** with matched `K_ε`" a headline contribution, but the live
  baseline is the in-house `abc_smc_baseline`, and `previous-fixes.md` documents a long tail of
  pyABC MPI-teardown deadlocks. Either (a) get at least the *non-scaling* pyABC comparison working
  for the validity benchmarks (the matched-acceptor claim needs real pyABC numbers), or (b) soften
  the contribution language to "a kernel-matched synchronous ABC-SMC baseline" and demote pyABC to
  a single-node external-validity check. As written, the paper promises pyABC evidence the configs
  don't currently produce.
- **The realistic simulator workload is gated on the simulation backend** (stub raises ImportError). The headline HPC use case
  is non-reproducible without the external dependency — fine for the authors, but state it and
  ship the reference-data + a synthetic fallback so reviewers can run *something*.
- **Test-mode realistic-workload scaling wall-time = 1800 s** (same as production) defeats the smoke-test purpose
  (`small/scaling_realistic.json`). Minor, but it has bitten the team before (see the runtime_heterogeneity
  test-mode hang in `previous-fixes.md`).
- **MPI teardown fragility is a standing risk.** `previous-fixes.md` shows repeated ParaStation
  `Disconnect`/`Create_intercomm` deadlocks, and the recent `PROPULATE_SKIP_DISCONNECT` escape hatch
  (commit `4fbf99d`) is a symptom. None of this biases results, but it threatens the *completeness*
  of the scaling sweep (256-core runs are where it bites). Budget for it.

### Reasonable as-is (no change needed)
SBC `n_replicates = 1` (correct — 100 trials provide the stochasticity); straggler 16 workers;
heterogeneity per-replicate `stable_seed`; per-model tolerance scales (LV `tol_init = 5e5` is
distance-scale-appropriate); wall-time-bound realistic workload. Seeding is deterministic per replicate; the one
gap is no explicit per-MPI-rank seed in the generic path (the ABC tutorial seeds `seed + rank`,
but confirm every runner does likewise, else rank RNG streams may correlate).

---

## 3. Visualization — publication-ready? (close, not yet)

### Strengths (verified)
PDF (vector) + PNG dual output; colourblind-safe palettes (`tab10`, `viridis`); per-replicate
aggregation with t-distribution 95% CIs; `frameon=False` legends; axis labels with units;
per-figure `_data.csv` + `_meta.json` provenance (git hash, timestamp); an **audit gate** that
records an explicit skip instead of emitting a misleading figure. This is well above typical
research-code standard.

### Gaps blocking "publication-ready"
- **No central style.** No `.mplstyle` / `rcParams` block. Fonts fall to matplotlib defaults
  (~10 pt DejaVu Sans) and vary across plots (`"small"` vs `8` vs default). For Springer Nature
  `sn-jnl`, set serif (or the journal's font), and fix sizes (≈8 pt tick, 9 pt label, 9–10 pt
  legend) once, globally.
- **Figure widths not tied to column widths.** Sizes are ad hoc (`(5,3.5)`, `(6,4)`, `(9,3.5)`).
  Target the journal's text widths (single ≈ 3.4 in / 88 mm, double ≈ 7.0 in / 180 mm) so fonts
  render at the intended pt size after LaTeX scaling. Right now a `(9,3.5)` figure scaled into a
  3.4-in column shrinks fonts ~2.6×.
- **Several paper-facing plots are PNG-only** (`posterior_quality_plot`, `threshold_summary_plot`,
  `corner_plot`, `tolerance_trajectory_plot`, gantt). Raster figures in a vector-capable journal
  look second-rate; route paper figures through PDF. 150 DPI PNG is fine for diagnostics, low for
  print.
- **`fig.savefig(..., bbox_inches="tight")` + fixed figsize** means final dimensions are not
  deterministic — the tight bbox crops vary, so two figures nominally `(6,4)` won't share margins
  or font scale on the page. For a coherent figure set, prefer `constrained_layout` + explicit
  size, drop `bbox_inches="tight"` for paper figures.
- **Embedded titles** on most plots — journals want captions, not in-axes titles. Make titles
  suppressible for paper output.
- **LaTeX figures are all placeholders** (`sn-article.tex:250–274`); `figures/{benchmark,overview,
  runtime}/` are empty. So no figure has actually been round-tripped into the paper yet — the true
  publication-readiness test (does it look right at column width in the PDF) hasn't been run.

---

## 4. Prioritized remediation backlog

**P0 — methodological (do before generating final results)**
1. Add `archive_mode="all_accepted"` (A5) and use it for all validity/SBC/posterior-quality runs;
   reconcile the optimiser-vs-approximator framing in §3/§4. *(propulate abcpmc.py)*
2. Fix the monotone-tolerance floor under deactivation (A2) — supports condition C3. *(abcpmc.py)*
3. Archive-phase-only kernel mixture (A1) — supports the "no phase discontinuity" claim. *(abcpmc.py)*
4. Decide pyABC's fate: get the non-scaling matched-acceptor comparison running, or soften the
   contribution language. *(experiments configs + inference wrappers)*

**P1 — experiment completeness/robustness**
5. Register `amis_snapshot_sweep` and `scaling_realistic`; sync `test_all.sh` with the registry.
6. Raise replicates (≥10) for headline heterogeneity/scaling figures on cheap benchmarks.
7. Document or remove the per-benchmark scheduler confound.
8. Annotate the stale `propulate/.plans/algorithm-review.md` to mark A3/A4 as resolved by W1.

**P2 — figure polish (do once, before final figure generation)**
9. Add a shared `.mplstyle` (fonts, sizes, line widths, palette) and a `journal_figsize()` helper
   keyed to `sn-jnl` column widths.
10. Route all paper-facing figures to PDF; switch to `constrained_layout`; make titles suppressible.
11. Round-trip 2–3 real figures into `sn-article.tex` and check them at column width in the built PDF.

---

## Verification plan

- **A1/A2/A5 fixes:** the `algorithm-review.md` §Verification section already specifies the minimal
  tests (construct mixed-phase archive; deactivate running-min; multi-modal toy for top_k vs
  all_accepted). Run the existing `propulate/tests/test_abcpmc.py` cache-equivalence tests
  (the cleanest correctness anchor) after each change. Use the venv at
  `sim_backend_venv/.venv/bin/python`.
- **SBC re-validation:** after `all_accepted`, re-run the SBC experiment; rank histograms should be
  flatter and coverage closer to nominal than under `top_k`. This is the empirical proof the
  posterior claim now holds.
- **pyABC baseline:** run a single-node `gaussian_mean` with `pyabc_smc` enabled and confirm the
  matched-`K_ε` acceptor produces comparable posteriors to `async_propulate_abc`.
- **Figures:** build `sn-article.tex` with real figures substituted for 2–3 placeholders; inspect
  font sizes and dimensions at column width in the compiled PDF.
