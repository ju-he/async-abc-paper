# Cellular Potts screening — results

**Run 2026-09-18** against `.plans/HANDOFF_cpm_sweep_2026-09-18.md`. Four screening corpora,
**9,316 fresh CPM evaluations**, 0.44 node hours of the 24 authorised. Driver
`experiments/scripts/diag_cpm_screening.py`, launcher `experiments/jobs/cpm_screening.sh`.
Reports under `experiments/data/cpm_screening/*/screening_report*.json`, rendered tables in
`experiments/data/cpm_screening/logs/*.md`.

---

## The headline: the handoff's premise does not survive a designed experiment

The handoff's plan rests on one measurement — signal-to-noise 0.30, "Monte Carlo noise in ρ at
fixed θ is 3.4× the entire systematic variation over the parameter space" — and concludes that
**no choice of parameters survives that**, so the protocol must be fixed first.

Measured on a fresh space-filling design under the **shipped** protocol, the **shipped** prior,
the **shipped** feature-space model and the **shipped** reference simulation:

| | within-θ | between-θ | SNR |
|---|---|---|---|
| handoff, from the stored campaign | 0.053 | 0.016 | **0.30** |
| this screen, 2,056 designed evaluations | 0.141 | 0.533 | **3.78** |

The two are not in conflict; they measure different things. The pipeline reproduces the campaign's
discrepancy almost exactly — median ρ by `division_rate` prior quantile comes out
4.22 / 2.95 / 0.96 / 1.34 / 0.64 / 0.60 against the campaign's stored 3.54 / 1.26 / 0.554 / 0.334 /
0.264 / 0.245, the same monotone collapse to a flat floor, uniformly ~2.4× in scale. The within-θ
noise matches after the same factor (0.141 vs 0.053). What differs is the **between**-θ term: the
handoff's 0.016 is the spread of cell medians over the prior *cells that the converged sampler
visited at least 20 times*, which is by construction the flat region the sampler had already
collapsed onto. It is a correct statement about the posterior region and not a statement about the
signal available across the prior.

**So the noise floor is not the binding constraint. The prior is** — which is what the handoff told
us not to start with.

---

## Phase A — protocol. Three of the four candidate factors do nothing.

50³ corpus, equal-weight ρ, robust (MAD-based) scales throughout:

| factor | result |
|---|---|
| **snapshot averaging** (1 → 3 → 6 snapshots) | SNR 3.49 → 1.88 → 1.95. **Hurts.** |
| **radial bins** (32 → 16 → 8) | SNR 3.49 → 3.79 → 3.15. **No effect.** |
| **replicate seeds** (k = 1 → 2 → 4) | SNR 3.49 → 3.92 → 5.60; within-θ 0.064 → 0.040. **Works, as √k.** |
| **domain 50³ → 80³** | SNR 3.49 → 2.67. **No gain, at ~2.5× the cost.** |

Snapshot averaging fails because the snapshots are not repeat measurements of one state: the cluster
grows from a median of 15 cells at t=250 to 60 at t=500, so averaging mixes a growth trajectory into the
summary and adds between-time variance to the within-θ term. Only replicate *seeds* are repeat
measurements.

The domain result is the flatly contradicted one. The handoff's reasoning was that `r95 = 17.3` in a
50³ box (half-width 25) means "the cluster is already pressed against the walls, which is what caps
growth". At 80³ the cluster is the **same size** — median r95 17.4 in both, median 63 cells in both —
but wall contact goes from 27.7% of evaluations to **0%**. Growth is capped by the division condition
and the volume constraint, not by the box. The one expensive protocol change buys nothing.

---

## The real defects, in order of size

**1. Equal block weighting throws away most of the signal.** Under the shipped model, `feature_weights`
is empty, so each of the 7 blocks gets 1/7 of the distance budget. Their individual SNRs:

| block | dims | within-θ | between-θ | SNR |
|---|---|---|---|---|
| log_n | 1 | 0.078 | 0.972 | **12.45** |
| log_r95 | 1 | 0.102 | 0.920 | **9.03** |
| radial_fa_equal_volume | 10 | 1.090 | 1.353 | 1.24 |
| pair_correlation_gofr | 10 | 1.294 | 1.264 | 0.98 |
| radial_s2_equal_volume | 10 | 1.143 | 0.942 | 0.82 |
| radial_density_profile_equal_volume | 10 | 1.297 | 1.005 | 0.78 |
| dbscan_gaslike_fraction | 1 | 0.000 | 0.038 | dead |

Two 1-dimensional scalars carry essentially all of it; the four 10-dimensional curve blocks sit at
SNR ≈ 1 and collectively spend 4/7 of the budget on noise. Dropping to the two scalars raises SNR
from 3.78 to **5.00** at k=1 and from 7.86 to **13.49** at k=4 — a free 1.7× before any simulation
changes.

**2. `dbscan_gaslike_fraction` is dead, and it is fixable.** Confirmed: within-θ scatter exactly zero,
scaler `median=0.0, scale=1.0` (the RobustScaler IQR-zero fallback), `block_norms = 0.0` silently
rewritten to 1.0 at `feature_scaling.py:1816`. It consumes 1/7 of the budget and carries nothing.
It is dead because the *range* never straddles the gas–liquid transition, not because the feature is
useless: in round 2, with adhesion widened to `J_cc ∈ [40, 400]` and `J_cl ∈ [20, 200]`, the same
block comes alive (within-θ 0.922, SNR 0.42). Recalibrate it or drop it; do not leave it in.

**3. About one evaluation in five is degenerate and is scored anyway.** 19.0% (round 1), 25.9%
(round 2) and 19.6% (80³) of evaluations end below 20 cells, where a 32-bin radial profile has under
one cell per bin and g(r) is meaningless. `CellularPotts.simulate` returns a finite discrepancy for
these exactly as for a good run — there is no cell-count guard. They are the source of the heavy
right tail that makes every non-robust statistic unusable: an RMS-based version of this screen read
a single degenerate level in the adhesion sweep as a 25σ "transition". Under the shipped prior the
rate is only 3.7%, so this is a hazard introduced by widening the priors, not a pre-existing bug in
the shipped campaign — but any reparameterisation must add the guard.

---

## Phase B/C — parameters

Two numbers per parameter, because they answer different questions. **Identifiability** is the
handoff's ratio (between-θ spread / within-θ noise, both robust) along the parameter's own response
direction in a noise-whitened summary space; below ~1 no campaign will find it. **Responsive window**
is the shortest contiguous sub-range carrying 80% of the monotone response — prior mass outside it
is mass the sampler cannot resolve.

### Under the shipped configuration (the paper's current setup)

| parameter | shipped prior | identifiability | responsive window | share of prior |
|---|---|---|---|---|
| motility | [0, 10000] linear | **13.36** | [357, 6786] | 64% |
| division_rate | [6e-5, 0.6] linear | **0.40** | [0.0215, 0.193] | 29% |

`|cos|` between their response directions: **0.89**.

That is the whole story of the flat CPM posterior, and it is not a noise story:

- `division_rate` scores **0.40**, below the identifiability threshold, because its response is
  confined to the bottom 29% of a linear prior and 67% of the prior mass sits above 0.193 where the
  response has saturated. The posterior running to the prior edge is the correct answer to the
  question actually asked.
- `motility` is individually *well* identified (13.4) — the handoff's "shallow interior optimum,
  well ~5% deep" understates it, because a 5% change in ρ is large against a within-θ noise of 0.14.
- But the two move the summary along nearly the same direction (0.89), both through `log_n` and
  `log_r95`, so jointly the pair is close to degenerate: one well-determined combination, one
  unconstrained.

### Six-parameter screen, reparameterised priors (round 1, 50³, protocol b32_s1_k4)

| parameter | prior | identifiability | carried by | responsive window | share of prior |
|---|---|---|---|---|---|
| division_rate | [0.002, 0.2] log | **17.69** | log_n 56%, log_r95 21% | [0.00236, 0.0328] | 57% |
| surface_lambda | [0.25, 8] log | **8.36** | log_n 53%, log_r95 21% | [0.464, 5.52] | 71% |
| recalc_time | [2, 60] log | **3.70** | log_n 28%, g(r) 21% | [8, 33] | 42% |
| motility | [100, 4000] log | **3.50** | log_n 58%, log_r95 18% | [722, 2694] | 36% |
| persistence | [0, 0.99] linear | **0.00** | g(r) 90% | [0.035, 0.106] | 7% |
| adhesion_cc | [40, 200] linear | **0.00** | g(r) 76% | [57, 91] | 21% |

Round 2 (protocol b16_s1_k2; refined priors, `adhesion_cc` widened to [40, 400], `adhesion_cl` = J(cancer, medium) added
over [20, 200], `persistence` on a log prior) reproduces the ranking and settles the tail:
division_rate 7.18, motility 4.18, surface_lambda 3.36, persistence 1.54, adhesion_cc 0.65,
adhesion_cl 0.00, recalc_time 0.00.

**Reparameterising `division_rate` works.** Log-uniform on [0.002, 0.2] moves identifiability from
0.40 to 17.7 and the responsive window from 29% to 57% of the prior. This was the handoff's
suggestion and the data support it.

**Adhesion does not drive anything here, and the handoff's physics for it is right but its range was
not.** Surface tension against the medium is γ = J_cl − J_cc/2 = 151 − 51.5 = 99.5 at the template
values, and stays positive for every J_cc below 302 — the [40, 200] screen never leaves the cohesive
regime, which is why it moved the summary by 2σ. Round 2 crossed γ = 0 from both sides
(J_cc to 400, J_cl to 20) and adhesion **still** scores 0.65 and 0.00. The two adhesion parameters are
0.98 confounded with each other, exactly as CPM theory requires (only ΔH/T enters the Boltzmann
acceptance, and only the γ combination enters the tension) — so they must never both be inferred.

**Everything that works, works through population size.** `log_n` and `log_r95` carry 21–79% of every
identifiable parameter's response direction. The confounding matrix is the consequence:

| | division_rate | motility | surface_lambda | recalc_time |
|---|---|---|---|---|
| **division_rate** | 1.00 | 0.87 | 0.96 | 0.76 |
| **motility** | 0.87 | 1.00 | 0.88 | 0.70 |
| **surface_lambda** | 0.96 | 0.88 | 1.00 | 0.81 |
| **recalc_time** | 0.76 | 0.70 | 0.81 | 1.00 |

All four sit within 0.70–0.96 of each other. **There is no second identifiable direction in this
summary space.** That, not the noise floor, is what stops the benchmark carrying a multi-parameter
inference claim.

---

## Recommendation

**A one-parameter CPM inference claim is supportable now; a two-parameter one is not, and no protocol
change in the handoff's list makes it so.**

If the benchmark is to carry an inference claim, the defensible configuration is:

1. **Infer `division_rate` alone**, log-uniform on **[0.002, 0.2]** (physical probability per cell per
   timestep; λ = p·T over T = 500 spans 1 to 100 expected division attempts). Identifiability 17.7,
   responsive over 57% of the prior, reference value 0.03 sitting mid-prior rather than in the bottom
   2%. Hold everything else at the template values.
2. **Four replicate seeds per evaluation** (`DistanceMetric.calculate_distance_replicates` already
   implements the feature averaging). Within-θ noise 0.064 → 0.040 at 4× the cost. Keep one snapshot
   at t=500, 32 bins, and the 50³ box — none of those are worth changing.
3. **Reweight the blocks**: `feature_weights = {log_n: 0.5, log_r95: 0.5}`, or at minimum drop
   `dbscan_gaslike_fraction`. Free, and worth 1.7× on SNR.
4. **Add a cell-count guard** to `CellularPotts.simulate`: below ~20 cells the curve summaries are
   noise, and returning `inf` there is both honest and what the ABC archive logic already expects.

For a genuine two-parameter claim the summary space has to change, not the protocol: every parameter
that moves this feature set moves it along the population-size axis. That needs a summary that is
size-invariant by construction — the existing `radial_*_equal_volume` blocks were presumably meant to
be exactly that, and at SNR ≈ 1 they are not delivering it. Screening a size-normalised feature set is
the next experiment, and it is a feature-engineering problem, not a sweep.

If the benchmark stays scaled-only, §5.2 and §6.2 of the paper need no change, but the stated reason
should be corrected: it is not that Monte Carlo noise swamps the signal, it is that the summary
statistics resolve only one direction in parameter space.

---

## Cost and housekeeping

- 4 jobs, 1 node each: 4:53 + 13:20 + 4:16 + 3:58 = **0.44 node hours** of the 24 authorised
  (plus two sub-minute `devel` probe jobs).
- 9,316 evaluations, **0 simulation failures**.
- Scratch inodes: **net +4 from this work**. Per-evaluation directories are removed by the driver —
  **0 stray `eval_*` directories** from these four jobs, whose peak footprint was 212 files. The four
  corpora were then archived to `cpm_screen_{50,80,r2,shipped}.tar.gz` under
  `/p/scratch/tissuetwin/herold2/async-abc/` (24 MB each) and the directories removed.
  The project-wide counter did move over the same day, 3,015,969 (09:21) to 3,405,552 (17:21), but
  that is someone else's +390k: this campaign never held more than 212 files. Unrelated, and
  pre-existing: 18 `eval_*` directories survive under `old/run1/` and `small_20260618_191443/` from
  March and June runs.
  No attempt traces are written — `_attempt_trace.py` is reached only from `pyabc_wrapper` and
  `abc_smc_baseline`, and this driver uses neither.
- `keep_eval_dirs` is not set to `true` anywhere in the repo (checked). `_cleanup_combo_artifacts`
  in `scaling_runner.py:630` still has **no production caller** — unchanged by this work, and still
  worth wiring in before the next scaling campaign.

## Reproducing

```
experiments/jobs/submit.sh --ntasks=48 --nodes=1 --time=00:40:00 \
    experiments/jobs/cpm_screening.sh <scratch_out> --blocksize 50 \
    --n-anchors 10 --anchor-seeds 24 --oat-levels 14 --oat-seeds 6 \
    --n-lhs 400 --lhs-seeds 4 --ref-seeds 48

python experiments/scripts/diag_cpm_screening.py --mode analyze --out <local_copy>
```

The four corpora used here were generated with `--design-seed 20260918` (rounds 1 and the 80³ run),
`20260919` (round 2, with `--prior persistence=0.002:0.3:log --prior division_rate=0.002:0.06:log
--prior motility=300:3000:log`) and `20260920` (the shipped-prior control, with
`--parameters division_rate motility --prior division_rate=0.00006:0.6:lin --prior motility=0:10000:lin`).

---

# Addendum — do the sibling campaign's features help?

The spheroid inference campaign in `../nastjapy/inference-campaign/` reached the *same diagnosis
independently*, on a different model, and built features against it. Its canonical log is
`../nastjapy/.planning/ressources/inference_execution_log.md`.

**Its F67 is this screen's result in different words:** "`r95/r50` is very nearly a deterministic
decreasing function of cell count … `N`, `tail` and `r95/r50` carry on the order of one effective
dimension, not three … an NPE trained on (N, tail, gas, r95/r50) would return approximately the
prior for every mechanics dimension. **⇒ this, not coverage, is now the binding obstacle**, and it
is a feature-design problem, not a compute problem." Its F10 is the same for the blocks this paper
uses: "the size-normalized radial profiles (radial_fa/s2, density) are motility-blind —
pcorr(feature, motility | division_rate) ~ 0".

Two things it hands over.

## 1. The screening statistic, which this screen was missing

F69/F71 score a feature by **partial rank correlation with log N removed from both sides** — does
it see the parameter *beyond what population size already explains*. That is a sharper instrument
than the response-direction geometry used above, and it changes the answer. Applied to this paper's
corpora (1,600 LHS evaluations, 50³):

| parameter | best \|partial ρ\|, shipped blocks | best \|partial ρ\|, + campaign features | via |
|---|---|---|---|
| motility | 0.22 | **0.41** | `surface_roughness` |
| division_rate | 0.23 | **0.31** | `surface_roughness` |
| recalc_time | 0.32 | 0.32 | `radial_density_profile` |
| surface_lambda | 0.21 | 0.21 | `log_r95` |
| persistence | 0.22 | 0.20 | `radial_s2` |
| adhesion_cc | 0.09 | 0.08 | — |

## 2. Two features, already implemented, already registered

`invasion_ratio` (r95/r50) and `surface_roughness` (CV of the outer-shell radii) are in nastjapy's
`FEATURE_FUNCTIONS` and need nothing but cell positions, so adding them to the benchmark is a
`distance_metric_params.json` edit. F71: "**Two features carry most of the new signal and neither
was in the gate:** `surface_roughness` (0.51 motility / 0.47 leader_fraction) and
`pair_correlation_gofr__0`." Re-run of the round-1 50³ design with them extracted (2,392
evaluations, 4.5 min, corpus `campaign_features/`) puts them above every curve block:

| block | SNR |
|---|---|
| log_n / log_r95 | 33.5 / 15.6 |
| **invasion_ratio** | **2.33** |
| **surface_roughness** | **1.72** |
| radial_fa / radial_s2 / radial_density / g(r) | 1.53 / 1.06 / 1.05 / 0.97 |

**Two of this screen's findings are independently confirmed by the campaign's data.**
`radial_s2_equal_volume` is dead weight — its ten PCs "max out at |0.19| against every parameter"
there, SNR 1.06 here. And `dbscan_gaslike_fraction` is not intrinsically dead: it is the campaign's
single *best* feature for leader motility (partial 0.73). Both benchmarks kill it the same way, by
calibrating it over a range that never crosses the transition it measures.

## But adding them is not sufficient, and the reason is the metric

Added alongside the size blocks at equal weight, they change almost nothing: division_rate↔motility
confounding 0.88 → 0.84, motility identifiability 3.37 → 3.23, and the equal-weight discrepancy SNR
slightly *worse* (3.03 → 2.76) because two more blocks each take a share of the budget. The
information is real but invisible at 1/9 of the budget next to a block at SNR 33.

Removing the size blocks from the summary space shows what is underneath:

| summary space | division_rate ↔ motility | motility identifiability |
|---|---|---|
| size blocks only (`log_n`, `log_r95`) | **1.00** | 4.22 |
| shipped 7, equal weight | 0.88 | 3.37 |
| shipped 7 + the two campaign features | 0.84 | 3.23 |
| shipped 7, size blocks removed | **0.24** | 0.84 |
| that, + `invasion_ratio` + `surface_roughness` | 0.26 | **1.62** |

Two cells carry the argument. With only the size blocks every parameter is *perfectly* degenerate
(|cos| = 1.00) — the one-direction result in its purest form. With size removed the confounding
collapses to 0.24, so the curve blocks do hold a separable direction; and it is there, and only
there, that the campaign's two features earn their keep, roughly doubling motility from 0.84 to
**1.62**, across the identifiability threshold.

**So: they carry real information, but whether a metric can use it was left open here.** It is
settled in the next section, and the answer is no — the size-removed row above is a mirage, and is
corrected there.

## What does not transfer

- **The growth curve.** The campaign's nano test put `growth_model_r` at pcorr 0.64 for division
  given motility where single-snapshot features sat at ~0. Extracted here over the *full* 10-frame
  trajectory from t=50, it reaches **0.14** (and `growth_model_K` 0.15, a raw log-N slope 0.24).
  Measured, not inferred from a truncated window.
- **Adhesion, persistence, recalc_time** stay unidentifiable under every combination tried.
- **`shape_anisotropy` and `log_n_trajectory` are traps.** `shape_anisotropy` is PC1/PC3 and a
  near-degenerate smallest axis drives its block norm to 6×10⁸, after which it swamps every other
  block; `log_n_trajectory` begins at log(4) on the four seeded spheroids, so its early entries are
  coarsely discretised and heavy-tailed, and it captured 84–96% of the whitened response direction
  of parameters whose identifiability is 0. Both are off by default in `--extra-blocks`.

## The scale caveat, from the campaign's own data

The campaign ran this experiment at ~148 cells ("nano") and concluded: "**motility recovery stays
~0.25 of range (≈uninformed) for every feature set** … motility remains data-ceiling-limited at 148
cells even with trajectory dynamics — confirming it needs the **macro** scale (developed invasion
morphology)." This benchmark sits at a median of 60 cells, below that. Motility reaching only 1.62
even with the right features is consistent with their ceiling, and it is the honest reason to expect
a two-parameter CPM claim to stay marginal rather than become comfortable.

**Net effect on the recommendation above:** the one-parameter `division_rate` claim is unchanged and
still the safe deliverable. A `division_rate` + `motility` claim moves from "not supportable" to
"marginal, and contingent on a reweighted metric that includes `invasion_ratio` and
`surface_roughness`" — worth one more post-processing pass on the committed corpus before any
decision, and cheap, since it needs no simulation.


---

# Addendum 2 — the weighting question, settled: drop five of the seven blocks

The open question above was which weighting exploits the campaign's features. Answering it needed a
statistic that does not depend on estimating the noise at the reference, where rho is a near-zero
squared distance whose robust scale is badly determined — the flaw that sank the first attempt.

**The statistic: how many parameter directions does the discrepancy surface resolve?** Around the
reference, rho is quadratic in theta, so the eigenvectors of its Hessian are the directions a
sampler can and cannot see. Fit that surface using only the top-r curvature directions and ask how
well it predicts **held-out thetas** (5-fold, folds by theta so replicates never straddle).
Directions that are real improve held-out prediction; directions that are noise do not. No noise
estimate is required, because cross-validation supplies one.

Run on the 400-theta LHS stratum, four replicate seeds per evaluation:

`experiments/scripts/diag_cpm_resolved_directions.py`, on the 400-theta LHS stratum of the
`campaign_features` corpus, four replicate seeds per evaluation:

| block weighting | held-out R² at r=1 | at r=2 | at r=3 | directions resolved |
|---|---|---|---|---|
| equal, 9 blocks (what the paper ships) | 0.204 | 0.205 | 0.199 | **1** |
| **`log_n` + `log_r95` only** | 0.799 | **0.825** | 0.822 | **2** |
| scalars + campaign features at 25% | 0.758 | 0.773 | 0.771 | 2 |
| scalars + campaign features at 50% | 0.608 | 0.608 | 0.609 | **1** |
| size blocks removed | 0.068 | 0.080 | 0.073 | 2 |

Same ordering at one seed per evaluation (0.114 / **0.685** / 0.677 / 0.555 / 0.035); full tables in
`experiments/data/cpm_screening/logs/resolved_directions_k{1,4}.md`.

**Three conclusions, and the first two correct claims made earlier in this document.**

1. **The size-removed comparison in Addendum 1 is an artefact.** Dropping `log_n`/`log_r95` does
   drop the division_rate↔motility confounding from 0.88 to 0.24 — but that space predicts the
   discrepancy at held-out R² **0.08**, against 0.83 with the size scalars alone. The directions
   stop being parallel because the signal disappears, not because it separates. A geometry measured
   in a space with essentially no signal says nothing, and "the blocker is the weighting, not the
   feature list" was drawn partly from that row.

2. **Adding the campaign's features to the metric monotonically degrades it** — 0.825 → 0.773 →
   0.608 as their share goes 0% → 25% → 50%, and at 50% the second direction is lost. Their partial
   correlation (0.41 for motility) is real, but `surface_roughness` sits at SNR 1.72 and
   `invasion_ratio` at 2.33 against `log_n` at 33.5: mixing them in costs more in added noise than
   they contribute in signal. **The answer to "will the campaign's features help?" is no, for this
   benchmark's discrepancy** — not because the features are bad, but because this benchmark's cell
   counts make everything except the two size scalars noise-dominated.

3. **But the weighting question has a winner, and it is better than this document's first
   recommendation.** `log_n` + `log_r95` alone resolves **two** directions where the shipped
   equal weighting resolves one, at four times the held-out predictive power (0.825 vs 0.202). The
   two directions are division_rate (leading eigenvector −0.97 on it) and motility (+0.88).

**Why a 2-D summary beats a 43-D one, concretely.** The parameters' *leading* response directions
are nearly parallel — both move `log_n` hard, which is the |cos| = 0.88 reported above. What
separates them is the **ratio** of `log_n` to `log_r95`: partial rank correlation of `log_r95` with
motility, after removing `log_n`, is +0.21. That is a modest signal living in a two-dimensional
ratio, and burying it under 41 further coordinates at SNR ≈ 1 is what the shipped equal weighting
does. Removing them does not add information; it stops spending the budget on noise.

## Revised recommendation

This supersedes the one-parameter recommendation above.

1. **Infer `division_rate` and `motility` jointly**, log-uniform on **[0.002, 0.2]** and
   **[100, 4000]**. Two parameters, two resolved directions.
2. **Set `feature_weights = {"log_n": 0.5, "log_r95": 0.5}`** and drop the other five blocks from
   the CPM distance. This is the single highest-value change in this document: 1 → 2 resolvable
   directions, held-out R² 0.205 → 0.825, discrepancy SNR 2.35 → 14.56.
3. **Four replicate seeds per evaluation** (held-out R² 0.685 → 0.825), one snapshot at t=500,
   32 bins, 50³ box.
4. **Add the cell-count guard** to `CellularPotts.simulate`, unchanged from above.

The expected posterior is a correlated but proper two-parameter posterior, not a flat one. Whether
the correlation is tight enough to be worth reporting as an inference claim is a judgement call the
figure will settle — and it is now a cheap experiment, because none of this needs a new simulator
configuration, only a distance-metric edit and a rerun.