# Rerun campaign — final analysis/writing phase report (2026-07-15)

Compute matrix complete (`rerun_20260707`). This phase regenerated all figures and
re-derived every quoted number. Commits on `campaign-tooling`: `623456c` (figures),
`e5bdf52` (paper text/numbers), plus the earlier tooling commits.

## What was done
- **All 15 paper figures regenerated** from `rerun_20260707` via the shared
  `paper_style` module (Type-42 fonts verified 0/15, print-width, Okabe–Ito).
  Every generator now reads committed vendored CSVs (`experiments/data/paper_figures/`)
  by default and re-derives with `--refresh`. Paper compiles clean (no undefined
  refs, no overfull boxes).
- **parameter_bias compute run** (was missing from the 11-group campaign): jobs
  14107933-936; also fixed the missing shard-finalizer/completed-replicates registry
  entries for it.
- **sensitivity Wasserstein summary** computed on-cluster over the 64×2.26 GB combos
  (the sharded finalize had fallen back to the legacy tolerance metric).
- **lv_timing** reduced to a 2-way productive-vs-coordination split — the frozen
  commit lacks the per-arrival phase-timing instrumentation (removed after `7f4be1f`),
  so the old 3-way sim/proposal/coordination split is unreproducible.

## CLAIM-LEVEL changes the new data forced (need author sign-off)
These are not digit swaps — the qualitative story changed. Each is now reflected in
the figures and text; flagging for review:

1. **g-and-k posterior recovery is no longer an async win.** New final Wasserstein:
   async 0.086 vs sync 0.074 (comparable, async marginally higher). Only Lotka–Volterra
   remains a clear async win (0.31 vs 0.35). §6 recovery paragraph + fig caption + abstract
   framing updated ("markedly lower on the two fast simulators" → "lower on LV, comparable
   on g-and-k and CPM").
2. **LV strong scaling: async peaks on one node then DECLINES, and the fair baseline
   OVERTAKES it beyond 3 nodes.** Old text claimed async "still leads at every scale"
   (erodes 4×→2×). New: async 6240@48w (1 node) → 1729@288w; sync fair 2952@48 → 6034@288.
   Crossover between 144 and 192 workers. §7.1, Table 2, scaling caption, Discussion
   honest-boundary all rewritten. Also **async is now faster than sync at a single worker**
   (149 vs 105) — the old "single-worker async slower" caveat is removed.
3. **Straggler collapse is far larger than "≈3×".** sync 8800→60 sims/s (control→20×,
   >2 orders of magnitude); async ~3900 flat. Abstract + §6 + caption updated. Factor-0
   control added (II.8.5).
4. **CPM async advantage is ≈5× at 384 workers** (was ≈3×). Abstract + §7.3 + caption.
5. **Ablation: removing AMIS is neutral on the uniform-runtime Gaussian target**
   (no_amis 0.072 ≈ full 0.073); only the hard kernel degrades (0.076). Old claim
   "removing either ingredient degrades" was reframed: AMIS's reweighting matters under
   runtime–parameter coupling (§6), not on a uniform target — a coherent, honest story.
6. **SBC baseline over-confidence is now statistically resolvable.** At 1000 trials
   MC error ≈0.007–0.016 << the 0.09–0.14 gap, so "comparable to the gap → don't
   over-interpret" (the old 100-trial hedge) is replaced: the async method is *better*
   calibrated on this benchmark. Table 3 + text updated.
7. **Sensitivity scheduler axis dropped** (inert under the smooth kernel → held fixed at
   acceptance_rate, not swept). W range 0.08–0.15. Caption/text updated.
8. **param-bias metric** is now posterior-mean error to the analytic posterior (honest,
   resolves the §4.1 "Wasserstein to analytic posterior" mislabel); AMIS "empirically
   cancels" softened to "provides an additional correction" (review II.4, Limitation iv).

## Deferred / not done (optional polish)
- II.1.b granularity-disclosure sentence in §5 (the matched-ε claim is now true in code;
  the extra prose about per-generation vs per-arrival granularity is optional).
- Orphan-PDF housekeeping (II.7.b.6): 13 unreferenced figures/*.pdf still present.
- Data staging dir `/home/juhe/async-abc-rerun-staging` is a local mirror (transient).
