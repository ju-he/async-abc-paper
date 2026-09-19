# Proposed Cellular Potts experiment setup

**Written 2026-09-19.** Follows `.plans/cpm_screening_results_2026-09-18.md` (read its two addenda —
the second corrects the first). Four further screening corpora, **21,312 evaluations, 0.97 node hours**.
Every number below is measured, not projected. Configuration artefacts are written and validated;
nothing here needs a decision before it can be run.

---

## The proposal in one line

**Infer `division_rate` alone, log-uniform on [0.001, 0.2], with a two-block distance
(`log_n`, `log_r95`) and four replicate seeds.** Forecast posterior: **91% contraction, bias +0.006,
coverage 92%** at 5% acceptance. Motility is held fixed at 1400 because it cannot be identified —
demonstrated below, not assumed.

---

## What was screened

| corpus | design | evaluations |
|---|---|---|
| `cpm_two_t501` | division_rate × motility, 501 steps | 5,360 |
| `cpm_two_t1001` | the same at 1001 steps | 5,360 |
| `cpm_one_t501` | division_rate alone, motility fixed at 1400 | 5,232 |
| `cpm_two_k16` | division_rate × motility, 16 replicate seeds | 5,360 |

All at 50³, one snapshot, truth placed at the log-midpoint of the measured responsive window. Read
by `diag_cpm_posterior_forecast.py`, which is new here: the LHS stratum **is** a draw from the prior
and every draw carries its discrepancy, so rejection ABC is just "keep the smallest rho" and the
accepted subset **is** the posterior of that setup. No modelling, no further simulation. It is a
lower bound on what the adaptive sampler achieves per simulation; the comparison between setups is
the point.

---

## Why motility is out

This is the load-bearing negative result, so it is shown four ways.

**1. The posterior for motility is the prior — at every setting tried.** Contraction (1 − sd_post/sd_prior)
at 5% acceptance:

| setup | division_rate | motility |
|---|---|---|
| 501 steps, k=1 | 72% | **−7%** |
| 501 steps, k=4 | 78% | **−13%** |
| 1001 steps, k=4 | 87% | **−3%** |
| 16 replicate seeds | 84% | **−1%** |

Negative contraction is not a rounding artefact: the accepted set lies along a diagonal ridge across
the prior square, which puts more marginal mass near the edges than uniform.

**2. More replicate seeds sharpen the ridge instead of shortening it.** The posterior correlation
between the two parameters at 2% acceptance goes **0.67 (k=1) → 0.82 (k=4) → 0.94 (k=16)**. If the
degeneracy were Monte Carlo noise, more seeds would break it; instead they reveal it more exactly.
No amount of sampling effort identifies motility.

**3. A longer run makes it worse, not better.** At 1001 steps the cluster reaches a median of 269
cells (from 52), the within-theta noise on `log_n` drops 6× to 0.0044, and division_rate's
contraction improves to 87%. Motility's stays at −3%, and the confounding rises to **|cos| = 1.00**,
exactly degenerate. The reason is visible in the same run: 269 cells against a domain capacity of
~250 means the population has saturated the box, at which point everything is `log_n` and the two
parameters move it identically. `radial_density_profile` and `dbscan_gaslike_fraction` both go dead
there for the same reason.

**4. It is a property of the summary, not of the sampler.** Motility's response is monotone across
its whole prior — 26 noise sigmas from M=112 to M=3564 — and its identifiability score is 4.15. It is
not weak. It is parallel: `|cos|` with division_rate is 0.86, both acting through `log_n` (49%) and
`log_r95` (25%). A strong response along an axis another parameter already owns buys nothing.

---

## The recommended setup, and the evidence for each choice

### 1. One inferred parameter: `division_rate`, log-uniform on [0.001, 0.2], truth 0.009

Log-uniform is not cosmetic. The responsive window is **[0.0012, 0.017]** — 50% of a log-uniform
prior on [0.001, 0.2] and **8% of a linear one**. A linear prior reproduces the original failure
exactly: the posterior runs to the edge because most of the prior is saturated. The truth at 0.009
sits at u = 0.415, mid-prior.

*This required a code change*: `normalize_cpm_param`/`denormalize_cpm_param` were linear-only, so a
log-uniform prior was not expressible. They now honour an optional `"scale": "log"` in the
parameter-space JSON, defaulting to linear.

### 2. Distance over `log_n` and `log_r95` only, weighted 0.5 / 0.5

| weighting | contraction | bias |
|---|---|---|
| equal over 7 blocks (shipped) | 66% | **+0.063** |
| `log_n` + `log_r95` | **91%** | **+0.006** |

The shipped metric is not merely less precise, it is **biased**: its posterior median sits 0.06 of
the prior range from the truth, an offset larger than the two-block posterior's entire standard
deviation. Two further benefits fall out. The two scalars are well defined at any cell count, while a
32-bin g(r) on 12 cells is not — so dropping the curve blocks also removes the degenerate-simulation
hazard that made ~30% of evaluations meaningless. And nothing but `log_n`/`log_r95` is extracted, so
each evaluation is cheaper.

### 3. Four replicate seeds per evaluation

Contraction at 2% acceptance, at matched total simulation cost: **78% (k=1) → 82% (k=4) → 83% (k=16)**.
Four is where the curve flattens; sixteen costs 4× per evaluation for one further point.

### 4. Protocol unchanged: 50³, 501 steps, one snapshot at t=500

Every alternative was measured and rejected in the previous round (snapshot averaging hurts, bin count
is irrelevant, 80³ gains nothing) or here (1001 steps saturates the box).

### 5. A four-seed reference, and a tolerance floor at ~5% acceptance

A reference is one realisation of noise like any other evaluation. Measured over 12 independent
four-seed references:

| acceptance | contraction | bias (mean ± between-reference sd) | coverage of the 90% interval |
|---|---|---|---|
| 20% | 78% | +0.003 ± 0.022 | 100% |
| 10% | 87% | −0.002 ± 0.021 | 100% |
| **5%** | **91%** | **−0.003 ± 0.020** | **92%** |
| 2% | 93% | −0.003 ± 0.019 | 92% |
| 1% | 93% | −0.004 ± 0.021 | **75%** |

Below ~5% acceptance the posterior concentrates on the reference's own noise realisation: the last
two points of contraction cost 17 points of coverage. The between-reference bias scatter (±0.021) is
by then as large as the posterior's own standard deviation (0.020), which is exactly the condition
for undercoverage. **Stop the tolerance schedule at ~5%.**

---

## BLOCKER before this can be run — replicate averaging is not implemented

`n_replicates_per_evaluation: 4` is present in both configs and **nothing reads it**.
`CellularPotts.simulate` runs one simulation and calls `DistanceMetric.calculate_distance`;
the four-seed feature averaging that every forecast in this document assumes is not wired in.
nastjapy already provides `DistanceMetric.calculate_distance_replicates`, which averages the raw
feature arrays across replicates — the same operation `diag_cpm_screening.py` performs offline — so
the change is to run `k` simulations per evaluation with distinct seeds and pass their directories to
that method instead.

It is the first task of the next session, and it is not optional: at one seed per evaluation the
forecast drops from 91%/64% contraction to roughly 85%/16%, and the reference in
`experiments/data/cpm_reference_proposed/` is a four-seed set generated to match.

## Artefacts, written and validated

| file | what |
|---|---|
| `experiments/configs/cellular_potts_division_only.json` | the experiment |
| `experiments/assets/cellular_potts/parameter_space_division_only.json` | log-uniform prior; motility fixed at 1400 |
| `experiments/assets/cellular_potts/sims_feature_space_model_size.json` | two blocks, weights 0.5/0.5, fitted on this setup's own 4,800-simulation training corpus |
| `experiments/assets/cellular_potts/distance_metric_params_size.json` | extracts only the two blocks |
| `experiments/data/cpm_reference_division_only/` | four-seed reference at the truth |

Verified end to end by constructing `CellularPotts` from the config: prior maps u=0.415 → 0.009, four
references load, two blocks at 0.5/0.5, motility held at 1400.

Two further code changes were needed and are covered by tests:

* **Fixed parameters.** "Motility is not inferred" and "motility is held at 1400" are different
  experiments — the shipped template sets `motilityamount[9] = 50` — and only the second is
  reproducible. The parameter-space JSON now takes a `"fixed"` section carrying path and value, applied
  to every simulation.
* **A bug.** Multi-seed reference containers were documented as supported and were not: the container
  branch of `_collect_reference_paths` was dead code. Written up in `.plans/bug-fixes/previous-fixes.md`.

---

## What this does and does not license

**It licenses** a recovery claim: a known truth, a posterior concentrating on it with 91% contraction
and correct coverage at a stated tolerance, on a real multiscale simulator. That is a genuine
inference result and it is what the benchmark was missing.

**It does not license** a multi-parameter claim. The paper should say plainly that CPM carries a
one-parameter inference claim, and why the second parameter is excluded — because every parameter
that moves this summary moves it along the population-size axis. That is a more useful statement
than the current scoping and it is now backed by 21,000 measured evaluations.

**The honest caveat.** A one-parameter posterior is a weaker demonstration than a two-parameter one,
and a reviewer may reasonably ask whether a benchmark with one identifiable parameter is worth its
compute in a scaling paper. The counter is that CPM's role is the expensive, realistic simulator, and
a correct one-parameter recovery on it is worth more than a flat two-parameter one. If the answer is
that it is not worth it, the fallback is unchanged: keep CPM systems-only and correct the stated
reason, which is not that noise swamps the signal.

**If a two-parameter claim is wanted**, the requirement is now precise: a summary statistic that
responds to motility *without* responding to population size. The sibling nastjapy campaign's
`surface_roughness` and `invasion_ratio` are the right idea and were measured (Addendum 1); they carry
real information, and at this benchmark's ~60 cells they are too noisy to use. That is a
feature-engineering problem at a larger cell count, not a sweep.

## Cost

4 jobs, 1 node each: 9:15 + 23:48 + 8:41 + 16:42 = **0.97 node hours**, 21,312 evaluations, 0
failures. Running total for the CPM screening work: **1.4 of the 24 authorised node hours**. Corpora
archived to `/p/scratch/tissuetwin/herold2/async-abc/cpm_*.tar.gz`; scratch left with 9 archive files
and no directories.

---

# Addendum — other parameter combinations, including adhesion at high motility

**Asked 2026-09-19:** were other pairs explored, e.g. division and adhesion at a fixed and
probably rather high motility? Fairly: **no.** Every adhesion screen to that point held motility at
its centre. Pooled over every corpus that varied it, motility reached at most 3830 with a median of
~650, against a shipped prior ceiling of 10000. The regime the question names had never been
visited, and the hypothesis behind it is sound — at low motility a cluster stays cohesive whatever
J is, so adhesion can only express itself once cells have the motile energy to work against the
surface tension.

Seven further screens, **31,096 evaluations, 0.76 node hours**, all at 50³ with the protocol above.

## The interaction is real, and it is what destroys the model

The same six mechanics parameters screened at motility held **low** and **high**:

| | motility = 500 | motility = 6000 |
|---|---|---|
| median cells at t=500 | 49 | **13** |
| evaluations below 20 cells | 27% | **74%** |
| adhesion_cl identifiability | 0.00 | **1.18** |
| adhesion_cc identifiability | 0.00 | 0.73 |
| division_rate identifiability | 14.9 | 11.0 |

So adhesion does become visible at high motility — and the mechanism that reveals it is the
mechanism that evaporates the cluster. High motility plus weak adhesion disperses the population
before it can grow, and three quarters of the prior lands where the summaries mean nothing. There is
no window in which adhesion is both visible and the model measurable. (Note the collapse is the
*interaction*, not motility alone: pooled over corpora at template adhesion, median cell count falls
only from 71 to 38 across the whole motility range.)

## Pairs screened, and what each posterior does

Forecast by `diag_cpm_posterior_forecast.py`, four replicate seeds, 5% acceptance, twelve
independent references, two-scalar weighting except where noted:

| setup | first parameter | second parameter | confounding |
|---|---|---|---|
| **`division_rate` alone** | **91%** | — | — |
| `division_rate` × `motility` | 78% | **−13%** | 0.86 |
| `division_rate` × `adhesion_cl`, M = 2500 | 90% | **6%** | **0.05** |
| `division_rate` × `adhesion_cl`, prior narrowed to [110, 230] | — | identifiability **0.00** | 0.05 |
| `division_rate` × `surface_lambda`, M = 1400 | 78% | **7%** | 0.17 |
| `motility` × `adhesion_cl`, division fixed | **45%** | **1%** | 0.10 |
| `adhesion_cl` alone at its transition, everything else fixed | **1%** | — | — |

Contraction is 1 − sd_post/sd_prior; the first column is the parameter named first.

**Adhesion is genuinely orthogonal — and that is not enough.** `adhesion_cl` sits at |cos| 0.05 to
division_rate and 0.10 to motility, the only thing found in this whole campaign that is off the
population-size axis, and it is carried by `radial_fa_equal_volume` (86%) rather than by the
scalars. But orthogonal and weak is still unusable: 6% contraction beside division, 1% on its own.
No weighting rescues it — `radial_fa` alone, and three mixed weightings, all leave it at −3% to 2%
while costing division_rate 30 to 90 points.

**Narrowing its prior made it worse, and that located the real transition.** [110, 230] centred on
the template J_cl = 151 took identifiability from 1.73 to 0.00. The weak signal in the wide prior was
coming from the bottom end, which is where the physics says it should: the surface tension is
γ = J_cl − J_cc/2 = J_cl − 51.5, zero at J_cl ≈ 51, not 151. The motility × adhesion screen had
independently put the responsive window at [37, 71], bracketing it. So the fairest possible test is
adhesion alone, prior [25, 110] straddling the transition, motility and division both fixed — and it
gives **1% contraction**. That is the answer: adhesion is not identifiable from these summaries at
this scale, at the transition or away from it, at low motility or high.

**`motility` is identifiable once `division_rate` is fixed** — 45%, against −13% when both are free.
Its failure really is the ridge and nothing else.

## What this changes in how the screen is read

Comparing identifiability against posterior contraction across eleven configurations calibrates the
score, and the threshold quoted from the handoff was far too generous:

| identifiability | measured contraction |
|---|---|
| ≳ 10 | 85–93% |
| 3–4 | 45% if unconfounded, 0% if confounded with a stronger parameter |
| ≲ 2 | 0–7% |

"A parameter scoring below 1 will not be identified" is true but nearly vacuous. **A parameter needs
roughly 10, and low confounding, before its posterior contracts usefully.** Every earlier table in
these documents that flagged a score of 1–4 as promising should be read with that in mind.

## Verdict

The one-parameter proposal stands, now tested against seven further candidate configurations rather
than assumed. If a second reported parameter is wanted for presentation, `division_rate × adhesion_cl`
at M = 2500 is the least damaging pair — division keeps 90% against 91% alone, where motility costs
it 13 points and surface_lambda 13 — but adhesion's own 6% is not an inference result and should not
be presented as one.

**Cost:** 7 jobs, 1 node each, 6:49 + 5:37 + 6:15 + 6:43 + 6:53 + 7:29 + 5:59 = **0.76 node hours**.
Running total for all CPM screening: **2.2 of the 24 authorised node hours**, 52,408 evaluations,
zero failures. Corpora archived; scratch left with 16 archives and none of my directories.

---

# Addendum 2 — other routes to a 2-D inference, and one that works

**Asked:** what other options are there? The failure so far was always the same shape — every
parameter moves the summary along the population-size axis — so the routes out are the ways of
breaking that, and they are worth listing before choosing:

| route | what it would mean | status |
|---|---|---|
| **A. A size-invariant summary** | a statistic whose null does not depend on n | the campaign's `f_radial`/`omega` are built this way; untested here |
| **B. Measure motion, not arrangement** | MSD from the snapshots already written | **tested — works as a block (SNR 4.3), does not break the ridge** |
| **C. Use per-cell data** | `Volume`, `Surface`, `Polarity`, `MotilityDir` are in every CSV and no block reads them | untested |
| **D. A parameter that moves a different observable** | something acting on packing, not count | **tested — `cell_volume`, and it works** |
| **E. More cells** | 80³ *and* longer together, or a much bigger box | untested; 80³ alone and t=1001 alone both fail |
| **F. Reparameterise along the ridge** | report a tight combination and a loose one | honest but still one effective dimension |

Two were tested, at 9,984 evaluations and 0.3 node hours.

## B. MSD works as a statistic and still does not rescue motility

`msd` needs only cell positions matched by CellID across snapshots — which the protocol already
writes — so it costs nothing. It is the only block in any of these screens that measures motion
rather than inferring it from a static arrangement, and it is immediately the **third-strongest block
of all**:

| block | SNR |
|---|---|
| log_n / log_r95 | 30.1 / 20.7 |
| **msd** | **4.28** |
| g(r) / non_gaussian_parameter / radial_density / radial_s2 / radial_fa | 1.82 / 1.77 / 1.57 / 1.11 / 0.88 |

But motility's identifiability with `msd` in the summary is 2.44 and its confounding with
division_rate is still 0.82. MSD measures how far cells move, and in this model cells that divide
more also move more, so it lands on the same axis. **A better statistic does not fix a degenerate
parameterisation** — which is the same lesson as the campaign's features in Addendum 1, now with a
statistic that is size-independent by construction rather than merely intensive.

## D. `cell_volume` gives the second direction — this is the result

Target cell volume (`CellsInSilico.volume.default.value`, a standard CPM parameter meaning cell
size) sets how much space a cell occupies, so it moves the cluster radius **at fixed cell count**.
That is the one direction in `(log_n, log_r95)` that division and motility both leave alone. Screened
against division_rate with motility fixed at 1400:

| | identifiability | confounding with division_rate |
|---|---|---|
| division_rate | 22.6 | — |
| **cell_volume** | **26.3** | **0.07** |
| *(motility, for contrast)* | *2.4* | *0.82* |

It is the first parameter in this entire campaign that is **both** strongly identifiable **and**
orthogonal. adhesion was orthogonal but weak (1.7); motility is strong but parallel (0.86). Both
conditions are met here for the first time.

**The posterior**, four replicate seeds, sixteen independent references, two-scalar weighting:

| acceptance | ESS | division_rate contraction | coverage | cell_volume contraction | coverage | posterior corr |
|---|---|---|---|---|---|---|
| 10% | 120 | 85% | 94% | 16% | 100% | +0.13 |
| 5% | 60 | 88% | 94% | 38% | 100% | +0.09 |
| **2%** | **24** | **91%** | **94%** | **64%** | **100%** | **+0.04** |
| 1% | 12 | 91% | 94% | 78% | 94% | −0.26 |

Both parameters contract, the posterior is essentially uncorrelated, and coverage holds — where
division × motility gave 78% and **−13%** at a correlation of +0.78.

**Why the two-scalar distance still wins.** `cell_volume`'s response direction is carried mostly by
`radial_density_profile` (86%), so one would expect that block to be needed — yet weighting it in
gives 36% against the two scalars' 64%. The reason is the physical reading of the two resolved
directions found earlier: `log_n` is the count and the **ratio** of `log_r95` to `log_n` is the
packing. division moves the count, `cell_volume` moves the packing. The pair of scalars spans both
and is far less noisy than the density block, which reaches the same information the long way round.
This also explains, after the fact, why `scalars_only` resolved exactly two directions.

## Revised recommendation: a two-parameter setup

This supersedes the one-parameter recommendation. Both configurations are written and validated;
the one-parameter one is kept as the conservative fallback.

1. **Infer `division_rate` and `cell_volume`**, log-uniform on [0.001, 0.2] and [200, 1200], truths
   0.009 and 500. `motility` fixed at 1400.
2. **Distance over `log_n` and `log_r95` at 0.5/0.5**, four replicate seeds, 50³, one snapshot at
   t=500 — all unchanged.
3. **Tolerance floor at ~2% acceptance**, not 5%: with two constrained directions the posterior no
   longer over-concentrates on the reference's noise realisation, so coverage holds where the
   one-parameter setup lost it.

`experiments/configs/cellular_potts_two_param.json` and
`experiments/assets/cellular_potts/parameter_space_division_volume.json`.

## A bug this turned up, and a correction to what was committed yesterday

`generate_cpm_reference.py` ignored the parameter-space file's `fixed` section, so the reference
committed with the one-parameter proposal was generated at the template's `motilityamount[9] = 50`
while every evaluation would have run at 1400 — **the observed data came from a different model than
the simulations compared against it**. It also denormalised `--true-params` through the module
default limits rather than the file's own `physical_range` and `scale`, so any custom parameter space
was silently mapped through the shipped division/motility ranges. Both are fixed and tested
(`TestGenerateCPMReferenceHonoursTheParameterSpace`), the reference is regenerated as
`experiments/data/cpm_reference_proposed/` (verified: `motilityamount[9] = 1400`), and the wrong one
is deleted. The screening results are unaffected — `diag_cpm_screening.py` applies fixed parameters
to its reference stratum like any other evaluation, so every forecast above was self-consistent.

## What is still untested

**A** (size-invariant summaries: the campaign's `f_radial`, `omega`, `outer_frac`) and **C** (per-cell
`Volume`/`Surface` distributions, which no block currently reads and which bear directly on
`surface_lambda` and `temperature`) would be the next places to look for a *third* direction. **E**
(80³ with a longer run, or a substantially bigger box) is the expensive route to the cell counts the
sibling campaign says the mechanics parameters need. None is required for the two-parameter claim.

**Cost:** 2 jobs, 8:52 + 8:49 = **0.30 node hours**. Running total for all CPM screening: **2.5 of
the 24 authorised**, 62,392 evaluations, zero failures.

---

# Addendum 3 — routes A and C: better statistics, no third direction

The two routes flagged as untested in Addendum 2, both implemented and screened (4,792
evaluations, 0.26 node hours, `cpm_custom`). Six new blocks, all **intensive by construction** —
proportions, coefficients of variation, dimensionless ratios — which is precisely the property the
shipped feature set lacks:

* **Route C, per-cell data.** `cell_shape_index` (Surface / Volume^(2/3), mean and CV),
  `cell_volume_dispersion` (CV of cell volume), `motility_order` (polar and nematic order of the
  per-cell motility directions). Every CellInfo CSV carries `Volume`, `Surface`, `MotilityDir` and
  `Polarity`; no shipped block reads any of them. These are the only statistics here that look at a
  *cell* rather than at an arrangement of cells.
* **Route A, size-invariant structure.** `aggregation_omega` (Σn_i²/Σn_i over contact components,
  over N), `outer_fraction`, `radial_variance_fraction` (the sibling campaign's estimator, whose
  null median is free of component size).

## They are good statistics

Screened against six parameters at motility 1400:

| block | SNR |
|---|---|
| log_n / log_r95 | 29.2 / 18.3 |
| **cell_shape_index** | **6.90** |
| msd | 4.96 |
| **aggregation_omega** | **2.56** |
| **outer_fraction** | **2.23** |
| cell_volume_dispersion | 1.34 |
| *every shipped curve block* | *≤ 1.29* |

`cell_shape_index` is the third-strongest block of all and beats the four 10-dimensional curve
blocks by a factor of five. And it does what it was built for: **`surface_lambda` goes from
identifiability 1.30 to 6.14**, carried by `msd` (41%) and `cell_shape_index` (27%).

## And there is still no third direction

| parameter | identifiability | confounding with division_rate |
|---|---|---|
| division_rate | 21.9 | — |
| **cell_volume** | **14.0** | **0.10** |
| surface_lambda | 6.1 | **0.81** |
| persistence | 2.0 | **0.00** |
| recalc_time | 1.8 | 0.68 |
| temperature | 0.0 | 0.85 |

`surface_lambda` is now visible but parallel — it changes cell shape, which changes how cells pack,
which changes the count, so it lands back on the size axis. `persistence` is *perfectly* orthogonal
(0.00) and far too weak. Nothing clears both bars. Forecast on the six-parameter corpus confirms it:
division_rate 56%, cell_volume 47%, and everything else between −5% and 11%.

**The general shape of the result, now from three independent directions of attack.** Better
features (the sibling campaign's, Addendum 1), better observables (MSD, Addendum 2), and better
statistics (these) all improve what can be *seen* without adding what can be *separated*. In this
model at ~50 cells every mechanism except cell size ultimately expresses itself through how many
cells there are. `cell_volume` works because it is the one knob that changes the cluster's radius
without changing its count.

**No change to the recommendation.** `division_rate` + `cell_volume`, two-scalar distance, four
seeds, 2% tolerance floor. The new blocks are not needed: `scalars_only` scores 21.9 against 4.4 for
an equal weighting over all fourteen. They are worth keeping in the diagnostic because they are the
best non-scalar statistics measured here, and because a future attempt at a third parameter should
start from them rather than from the shipped curves.

**A bug caught by the loud-failure guard.** The first run of this screen failed all 4,792
evaluations with `TypeError: '<' not supported between 'str' and 'int'`: the new per-cell hook bound
its DataFrame to `frame`, shadowing the loop's integer frame index, so the next bin count passed a
DataFrame in as `timestep_range` and nastjapy iterated its column names. The driver refused to
report a screen rather than producing a partial corpus. Fixed and re-run clean.

**Cost:** 2 jobs (one failed, one clean), 7:16 + 8:08 = **0.26 node hours**. Running total for all
CPM screening: **2.7 of 24**, 71,976 evaluations.
