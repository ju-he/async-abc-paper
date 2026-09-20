# Raising the CPM parameter count, and fixing what wastes the budget

**Written 2026-09-19**, after `.plans/cpm_two_param_validation_2026-09-19.md`. Two threads that turn
out to be the same problem: what limits posterior accuracy per simulation, and what it would take for
CPM's posterior comparison to be a comparison the method can win.

---

## Why the two threads are one

Decomposing time-to-accuracy for the reported estimator:

```
evaluations-to-accuracy  ~  log2(tol_init / eps*)  x  evaluations-per-halving
wall-clock-to-accuracy   ~  that  /  throughput        <- where barrier-free execution wins
```

The reported posterior's spread is set by the bandwidth `eps`, which walks down from `tol_init` at a
bounded rate (`max_tighten_factor` per search, one search per `bisect_interval` calls). **Measured on
the validation replicate:**

| arrivals | eps actually used | k-th best loss (where the rule says it goes) | ratio |
|---|---|---|---|
| 500 | 10.0 | 0.0300 | 333x |
| 2,000 | 10.0 | 0.0011 | 8,751x |
| 5,000 | 1.25 | 0.0002 | 7,270x |
| 13,115 | 0.0041 | 0.0001 | 68x |

The bandwidth never leaves `tol_init` for 3,000 evaluations and is still 68x too loose at the end. On
a cheap simulator with millions of evaluations this is invisible. **On an expensive simulator the
transient is the whole run** — which is precisely the regime the method targets, so this is not a
side issue.

Raising the parameter count makes the simulator more expensive (bigger box), which makes the
transient *worse*. Fixing the transient is therefore a prerequisite for raising the parameter count,
not an alternative to it.

---

## Lever 1 — the reporting bandwidth. Free, retroactive, and large

Re-reporting the **same stored history** at different `eps_final` (no new simulation):

| `eps_final` | ESS | division_rate | cell_volume | coverage |
|---|---|---|---|---|
| as run (0.0041) | 1222 | 90% | **58%** | both |
| 0.001 | 1025 | 92% | 73% | both |
| 0.0002 | 370 | 92% | **82%** | both |
| 0.0001 | 186 | 92% | 81% | both |

The accuracy was already in the evaluated history; the reporting bandwidth discarded 24 points of it
on the weak parameter. Coverage holds throughout, so this is not over-concentration.

**The rule to adopt:** report at the k-th order statistic of the evaluated losses. It is truth-free,
data-driven, needs no new simulation, and is exactly what the scheduler's own docstring says the
bandwidth "equilibrates near" — it simply never gets there within an expensive run's budget. The same
rule applies to every arm, which is what keeps the comparison fair.

## Lever 2 — `tol_init`. Sweep in flight

`tol_init: 10.0` is ~1000x the prior's *median* discrepancy of 0.459, so 4-5 halvings are spent
walking down from an arbitrary constant. Jobs 14262028/29/30 sweep 1.0 / 0.1 / 0.01 against the
shipped 10.0, async only, 2 replicates each (6 node hours), everything else held. `tol_init` is
shared by both arms of the matched comparison by construction.

## Lever 3 — `bisect_interval`, `max_tighten_factor`, `min_tol`

On the propagator, previously unreachable from an experiment config; now exposed (unset = the old
default, so every earlier run reproduces). Not yet swept.

---

## The fairness problem that runs the other way

`rejection_abc` is `rank_parallel`: each rank takes a different *replicate*, so with 5 replicates on
48 ranks **43 ranks idle**. The baseline is being given 1/48th of the machine.

Rejection ABC is embarrassingly parallel, and on a two-parameter target it is a strong method. Job
14262032 measures the fair version — 13,000 prior draws on 48 ranks, matching the async replicate's
evaluation count and wall clock. From the 2,000-draw corpus it should land near 89% / ~62%.

If so: **the reporting fix is load-bearing for the claim.** Without it the async arm merely ties a
fairly-resourced rejection baseline; with it, 92%/82% against ~89%/62%. That is a better paper than
one where the baseline was quietly hobbled — but it has to be stated that way.

---

## Raising the parameter count — Route E

**The blocker is the summary, not the parameter list.** The two-scalar distance resolves *exactly two*
directions: `log_n` is the count, `log_r95`/`log_n` is the packing. division_rate moves the first,
cell_volume the second. A third parameter needs a third resolvable direction.

Of the six routes in the proposal's Addendum 2, three were tested and failed (better campaign
features, MSD, per-cell statistics — all improve what can be *seen*, none add what can be
*separated*). **Route E — more cells — is the one never tested**, and the proposal names it as the
missing ingredient: the mechanics features "carry real information, and at this benchmark's ~60 cells
they are too noisy to use. That is a feature-engineering problem at a larger cell count, not a sweep."

Two candidates are **already orthogonal and fail only the strength bar**, which is exactly what more
cells fixes:

| candidate | identifiability | confounding with division_rate |
|---|---|---|
| `adhesion_cl` | 1.2-1.7 | **0.05** |
| `persistence` | 2.0 | **0.00** |
| *(`surface_lambda`)* | *6.1* | *0.81 — parallel, no use* |

There is also a physical reason to expect the geometry itself to improve. At 50³ the population
**saturates the box** — 269 cells against a capacity of ~250 — and saturation is the stated mechanism
that collapses every response onto the count axis. At 80³ capacity is ~1,000, so nothing saturates.

**Job 14262055** is the reconnaissance screen: 80³, t=1001, `write_every` 100 and `extract_from` 500
(the same six analysed snapshots, same relative window, as the t=501 screens — only box and duration
change), five parameters with motility fixed at 1400, `adhesion_cl` prior straddling its measured
transition at J_cl ~ 51, `division_rate` widened downward because doubling the duration moves its
responsive window down. ~3,100 simulations, ~1 node hour.

**What would make it a success:** any parameter reaching identifiability >= 10 with confounding
<= 0.3, and `diag_cpm_resolved_directions.py` reporting three directions rather than two.

**Why this is the right move for the paper and not a detour.** Rejection ABC's acceptance falls like
eps^d, so its cost to a fixed posterior grows exponentially in dimension while an adaptive sampler's
does not. Two parameters is where rejection is strongest; three or four is where the method's
advantage is real rather than argued. A bigger box is simultaneously a more expensive, more realistic
HPC workload — the thing CPM is in the paper to be.

**The cost, and the catch.** 80³ with twice the duration is ~8x per simulation (~30 s against 3.7 s),
so a 1-hour 48-rank replicate falls from ~13,000 evaluations to ~1,600 — which is *below* the
bandwidth transient measured above. A production run at this size needs either the transient fixed
(levers 1-3), more nodes (CPM already scales to 384 workers at ~97% efficiency), or both. Levers
first, then size.

---

## Order of work

1. Read the `tol_init` sweep and the fair-rejection baseline (both in flight).
2. Implement the order-statistic reporting rule and re-report every stored history with it.
3. Read the Route-E screen: is there a third direction at ~10x the cell count?
4. If yes, design the 3-4 parameter setup and size it against the fixed transient.
   If no, CPM stays a two-parameter recovery claim and the honest framing is that its posterior
   comparison is not where the method's advantage lies — the 4-D g-and-k is.

---

# Addendum — what the rejection fix actually changed (gaussian_mean, job 14262064)

The arm now spends its budget instead of stopping at k accepted:

| | before (threshold at shared `tol_init` 5.0) | after (`best_k`) |
|---|---|---|
| evaluations used, of 20,000 | ~100 | **19,505-19,888** |
| tolerance reached | 5.0 (admits 99.1% of the prior) | **~0.025** |
| posterior sd, 5 replicates | ~2.89 (**the prior**, uniform on [-5, 5]) | **0.091-0.107** |

So the baseline went from a prior sampler to a genuine ABC posterior concentrated near the truth.
The measured tolerance (~0.025) matches the 0.0286 predicted from 20,000 prior draws beforehand,
which is a useful check on the `best_k` selection.

**This weakens, in our own direction, every comparative statement the paper makes against rejection
ABC.** The text quotes the asynchronous arm at 0.028 against "rejection ABC's 1.x" as a reference the
other arms are "far below"; against a rejection arm that now sits at sd ~0.09 around the truth the
gap is a few-fold rather than the roughly forty-fold those numbers imply. The direction still favours
the method on this benchmark, but the numbers have to be re-derived rather than re-used, and the
honest framing is much more modest. Re-deriving them belongs with the paper rewrite, not here.

---

# Addendum — Route E, first screen (job 14262057): mis-designed, but informative

3,108 evaluations at 80³ / t=1001, 0.68 node hours, zero failures.

**The two-parameter setup gets stronger at this size**, which is worth having on its own:

| parameter | identifiability at 50³/t=501 | at 80³/t=1001 | confounding with division_rate |
|---|---|---|---|
| division_rate | 21.9 | **30.1** | — |
| cell_volume | 14.0 | **23.9** | **0.08** |

**But the third-direction candidates went the wrong way**, and not because Route E failed:

| candidate | identifiability 50³ → 80³ | confounding 50³ → 80³ |
|---|---|---|
| `adhesion_cl` | 1.7 → 1.75 | **0.05 → 0.84** |
| `persistence` | 2.0 → 1.13 | **0.00 → 0.70** |

**The screen did not test what it was meant to.** Median cell count came out at **76**, barely above
the 50³/t=501 baseline of ~49 — because the `division_rate` prior was widened *downward*
([0.0002, 0.2] log) to avoid saturation at the doubled duration, which exactly cancels the point of
the bigger box. A bigger box does not create cells; the division rate and the duration do. With half
the prior mass below 5 cells and a tail reaching 2,174, `log_n` dominated every response direction
(between-theta 0.979 against within-theta 0.017, SNR 56) and dragged both weak parameters onto the
count axis. `adhesion_cl` is now carried by `log_n` (37%) where at 50³ it was carried by
`radial_fa_equal_volume` (86%) — the signature of exactly that artifact.

**The corpus calibrates the re-run for free.** Cells at t=1000 against `division_rate`, with fill
fraction against the box capacity:

| division_rate | median cells | p90 | median fill |
|---|---|---|---|
| [0.002, 0.004) | 29 | 52 | 0.03 |
| [0.004, 0.008) | 78 | 122 | 0.07 |
| **[0.008, 0.016)** | **234** | 476 | **0.19** |
| **[0.016, 0.05)** | **388** | 769 | **0.36** |
| [0.05, 0.2) | 659 | 1398 | 0.62 (p90 0.91 — saturating) |

So the band that gives many cells without filling the box is **[0.008, 0.05]**: 234-388 cells
against the 50³ baseline's ~49, at 19-36% fill. Job **14262079** re-screens there
(`division_rate` [0.006, 0.04], `cell_volume` narrowed to [200, 800] so the worst corner — highest
rate, largest cells — stays near 55% fill rather than saturating).

**Route E is therefore still open, not refuted.** The first screen tested a low-cell-count regime
with a badly conditioned prior; the second tests the regime the route is actually about.

---

# Results — the two levers land in the same place, and the fair baseline settles the positioning

## Lever 2: `tol_init` (jobs 14262028/29/30, 6 node hours, 2 replicates each)

| `tol_init` | eps the schedule reached | ratio to the k-th order statistic | division_rate | cell_volume | ESS | coverage |
|---|---|---|---|---|---|---|
| **10.0 (shipped)** | 0.0041 | **68x** | 90% | **58%** | 1222 | both |
| 1.0 | 0.00062 / 0.00046 | 9x / 6x | 92% / 91% | 77% / 76% | 1053 / 460 | both |
| **0.1** | 0.00018 / 0.00013 | **2x / 2x** | 93% / 91% | **81% / 84%** | 351 / 267 | both |
| 0.01 | 0.00054 / 0.00013 | 9x / 2x | 91% / 91% | 79% / 82% | 633 / 273 | both |

Lowering `tol_init` from the shipped 10.0 to 0.1 takes the weak parameter from 58% to **81-84%**
contraction with coverage intact, purely by letting the schedule reach its own documented
equilibrium inside the budget: the ratio to the k-th order statistic falls from 68x to 2x. 0.01 is
not better than 0.1 and is less consistent, so the sweet spot is about **1/5 of the prior's median
discrepancy** (0.459) — which is a rule that transfers to other benchmarks.

**The two levers are redundant, and that is the strongest evidence the mechanism is understood.**
Fixing the bandwidth *online* (`tol_init` 0.1 → 81-84%) and fixing it *retroactively* (report at the
order statistic → 82%) reach the same answer from opposite directions.

**Prefer `tol_init` as the primary fix.** It is an *a priori* configuration choice justified by the
prior-predictive discrepancy scale, which can be measured before any run and without reference to the
truth. Choosing a reporting bandwidth after the fact is more contestable, however principled the rule.
Keep `reported_eps_rule: order_statistic` as the robustness check, not the headline.

Note also that division_rate barely moves (90 → 91-93%): the entire gain is on the parameter that was
under-resolved, which is where extra bandwidth resolution should show up and nowhere else.

## The fair rejection baseline (job 14262032, 13,000 draws on 48 ranks, budget-matched)

| | tolerance at k=100 | division_rate | cell_volume | ESS |
|---|---|---|---|---|
| fair rejection ABC | 0.0029 | 91% | 67% | 100 |
| async **as the paper reports it** | 0.0041 | 90% | **58%** | 1222 |
| async at `tol_init` 0.1 | 0.00013 | 91% | **84%** | 267 |

**A fairly-resourced rejection ABC beats the asynchronous arm as currently configured.** It loses to
it once the bandwidth is set sensibly. The fix is therefore load-bearing for the CPM posterior claim,
and the claim must not be made without it.

**The cleanest statement of the advantage is in the tolerance, not the contraction.** At a matched
budget the adaptive sampler's k-th order statistic is **6.1e-05 against rejection's 2.9e-03 — 48x
tighter**. That is the per-simulation efficiency claim stated directly; the contraction gap follows
from it, and unlike the contraction it does not depend on how the posterior is reported.

## What this means for the paper

The CPM posterior comparison is winnable and honest, but only with the bandwidth fixed. Report the
48x tolerance ratio as the primary quantity. And the direction of travel for a decisive margin is
still dimension, since rejection's cost to a fixed posterior grows like eps^-d and we are already 48x
ahead in eps.

---

# Route E, properly tested (job 14262079): no third direction, and now we know why

3,108 evaluations at 80³ / t=1001 with `division_rate` in the calibrated band. **Median 313 cells
against the 50³ baseline's ~49** — 6.4x — at 24% fill (p90 51%), so nothing saturates. This is the
test the first screen failed to be.

| parameter | identifiability 50³ → 80³ | confounding with division_rate |
|---|---|---|
| `cell_volume` | 14.0 → **46.2** | 0.15 |
| `division_rate` | 21.9 → 15.6 | — |
| `surface_lambda` | 6.1 → **10.5** | **0.81 → 0.98** |
| `persistence` | 2.0 → 2.2 | 0.00 → 0.91 |
| `adhesion_cl` | 1.7 → 2.2 | 0.05 → 0.48 |

**`surface_lambda` cleared the strength bar exactly as predicted and is still useless.** More cells
made it visible (6.1 → 10.5, comfortably past the ≳10 threshold) and simultaneously drove its
confounding with `division_rate` from 0.81 to **0.98** — perfectly parallel. `adhesion_cl` and
`persistence` lost the orthogonality that made them candidates at all.

**More cells sharpens the coupling rather than breaking it.** The saturation hypothesis was wrong:
nothing saturates here, and the responses are *more* aligned, not less, because with a larger
population the growth dynamics dominate the summary even more completely. Anything that changes how
cells pack changes how fast the cluster grows, and at 313 cells that chain is tighter than at 49.

**Route E is now tested and fails.** All four routes to a third direction are closed: better campaign
features, MSD, per-cell statistics, and now cell count. The campaign's general result holds and is
now established at 6.4x the population: *every mechanism in this model except cell size ultimately
expresses itself through how many cells there are.* `cell_volume` works because it is the one knob
that changes the cluster's radius without changing its count.

**CPM is a two-parameter benchmark. That is final** — not for want of screening, but as a property of
the model at this scale.

**What the screen does deliver.** The two-parameter setup is far stronger at 80³: `cell_volume`
identifiability **46.2 against 14.0**, and it is now carried by `radial_density_profile` (94%) rather
than borrowed from the scalars. With an 8x more expensive simulation that is a better *systems*
benchmark and a better two-parameter posterior — it is just not a third parameter.

**So the decisive-margin-by-dimension route on CPM is closed.** It lives on the 4-D g-and-k instead,
where rejection ABC's cost to a fixed posterior grows like eps^-d and we are already 48x ahead in eps
on a 2-D target.

---

# The comparison that actually works — and the one that does not

Matched-budget tolerances on the 4-D g-and-k, streamed from the 2026-07-07 production history
(first 50,000 arrivals per replicate, k=100 — the same quantity `best_k` rejection reports at):

| method | k-th best loss at 50,000 evaluations |
|---|---|
| `rejection_abc` (fixed, `best_k`, job 14262065) | 0.46 |
| **`async_propulate_abc`** | **0.060 - 0.068** |
| `abc_smc_baseline` (synchronous) | **0.030 - 0.052** |
| *`rejection_abc` as the paper ran it* | *9.9 — it made only 100 draws* |

(The production history confirms the prior-sampling defect directly: its `rejection_abc` arm wrote
**500 rows in total**, 100 per replicate, out of a 50,000 budget.)

## Against rejection ABC the advantage is consistent across dimension

A tolerance ratio is not dimension-free — acceptance scales like eps^d — so the comparable quantity
is how many rejection draws would be needed to match:

| benchmark | dim | eps advantage | simulations rejection would need |
|---|---|---|---|
| Cellular Potts | 2 | 48x | ~48² ≈ **2,300x** |
| g-and-k | 4 | 7.4x | ~7.4⁴ ≈ **3,000x** |

Two to three orders of magnitude, on both, from two independent measurements. **This is the claim to
make**, and it is robust to the bandwidth question because the k-th order statistic describes where
the sampler put its draws, not the bandwidth it reported at.

## Against the synchronous baseline, per-evaluation efficiency is not where we win

On 4-D the synchronous baseline reaches a *tighter* tolerance per evaluation than the asynchronous
arm (0.030-0.052 against 0.062). The paper already concedes this ("roughly twice as inaccurate on the
four-dimensional one"); this confirms it at a matched budget rather than at a matched wall clock.

**The strategic error to avoid is conflating the two comparisons.** The method beats *rejection ABC*
by ~3 orders of magnitude in simulations-to-a-tolerance, in both dimensions. It beats the
*synchronous baseline* on **throughput**, not on per-evaluation efficiency — which is the paper's
actual thesis and where the barrier-removal evidence is strongest. Presenting both as one
"posterior quality" comparison is what makes the case look weaker than it is.

**Open, and cheap to close.** The g-and-k async numbers come from a run with `tol_init: 10.0`. If the
bandwidth transient degrades *sampling* and not merely reporting within a 50,000-evaluation prefix,
0.062 is pessimistic and the async-vs-sync gap on 4-D may narrow or reverse. One g-and-k job at
`tol_init` set from its prior-predictive scale settles it, and it should be settled before any of
this reaches the paper.

---

# The quantity that decides whether throughput is worth anything

Throughput is instrumental. What matters is posterior quality per node-second:

    quality per node-second  =  per-evaluation efficiency  x  evaluations per node-second

and the asynchronous method *loses* the first term on 4-D while winning the second. Whether it nets
out is set by **how fast tolerance improves with budget** — the exponent of eps against n. Measured
by the k-th order statistic over prefixes of real runs:

| | exponent, early | exponent, late | rejection's n^(-1/d) | 2x throughput buys |
|---|---|---|---|---|
| **CPM, 2-D, expensive** | -2.6 | **-1.1** | -0.50 | **2.1x** better eps |
| **g-and-k, 4-D, cheap** | -0.31 | **-0.18** | -0.25 | **1.13x** better eps |

**On CPM throughput converts.** The curve is steeper than rejection's, so the advantage *widens* with
compute: a 2-5x throughput gain becomes a 2-9x tolerance gain.

**On g-and-k it does not.** At -0.18 it takes **18x the budget to halve eps**, and rejection's -0.25
is *steeper*, so the 7.4x constant-factor lead would erode with more compute rather than grow. At
matched wall clock (600s) the synchronous baseline reaches eps ~ 0.026 against the asynchronous arm's
~0.031. On that benchmark, more throughput does not buy a better posterior.

The flat exponent is very likely the paper's own ESS ceiling seen from another angle: the proposal
concentrates onto the archive and stops exploring, so extra draws add little new information.

**What to do with it.** The paper reports throughput (§5.3) and posterior quality (§5.2) in separate
sections and never connects them. This exponent is the connection, and it should be reported per
benchmark next to the throughput numbers, because it is what says whether a systems gain is a science
gain. It also *explains* the existing results instead of excusing them: the 4-D case is weak because
the curve is flat there, and CPM should be strong because it is not.

**Caveats.** The CPM exponent is one replicate over a 26x budget range and is flattening (-2.6 →
-1.1); the production run (14262214) gives five replicates and pins it. If it keeps flattening toward
-0.5 the throughput argument weakens on CPM too. Cross-method record counts (3.3M baseline against
1.8M asynchronous in 600s) may not be comparable units — pyABC may log rejected proposals — so no
throughput claim is made from them; only the eps-at-matched-wall-clock comparison stands.

Confirmed separately: the g-and-k bandwidth transient does **not** bind (jobs 14262212/13, matched
50,000 evaluations at `tol_init` 10.0 and 1.3 give the same k-th order statistic, ratio 1x both), so
0.062 is the sampler's real per-evaluation efficiency on 4-D and the transient is CPM-specific.

## The CPM exponent, pinned on seven replicates

The `tol_init` sweep produced six more asynchronous replicates, so the exponent need not rest on one:

| `tol_init` | replicates | exponent (1k-12k) | late segment |
|---|---|---|---|
| 10.0 | 1 | -1.77 | -0.94 |
| 1.0 | 2 | -1.65, -1.52 | -1.28, -1.16 |
| 0.1 | 2 | -1.64, -1.69 | -1.02, -0.82 |
| 0.01 | 2 | -1.47, -1.39 | -1.35, -1.06 |
| **all** | **7** | **-1.59 +/- 0.13** | **-1.09 +/- 0.19** |

Against rejection ABC's n^-0.50 at d=2, the advantage **widens by about 1.5x per doubling of
compute** rather than staying a constant factor. Every replicate lands between -1.39 and -1.77, so
the steep curve is not a one-run artefact.

**`tol_init` does not affect the sampling exponent** — a 1000x range moves it within noise. Combined
with the same result on g-and-k (jobs 14262212/13), this settles it: **the bandwidth transient is
purely a reporting defect.** The sampler always found the target efficiently; only the reported
posterior discarded it. That is worth stating precisely in the paper, because it separates a
configuration mistake from an algorithmic limitation.

Incidental: at `tol_init` 0.01 the sampler is genuinely better *early* (eps 0.0022 at n=1000 against
0.005 at 10.0, 2.3x tighter) because the kernel concentrates from the start; it washes out by
n=12,000. On a budget-limited expensive run that early advantage is worth something.

---

# Archive size, and a correction to the simulation-ratio claim

## The two benchmarks are in different regimes, and I assumed the wrong one

`eps` was measured throughout as the k-th order statistic, so both its k- and n-scaling are
measurable rather than assumed. Measured:

| | rejection eps vs n | eps vs k | regime |
|---|---|---|---|
| **CPM** | **n^-1.01** | k^+0.99 | **noise-dominated** — positive density at rho=0, effective dim 1 |
| **g-and-k** | **n^-0.24** | k^+0.28 | **geometry-dominated** — d=4, exactly as theory predicts |

**Correction.** Earlier in this document the tolerance ratio was converted to a simulation ratio
through acceptance ~ eps^d, giving "~2,300x" for CPM. That conversion assumes geometry-dominated
scaling. CPM is not: eps ~ k/n, so the simulation ratio *equals* the tolerance ratio. The correct
figure is **~53x, not 2,300x**. The g-and-k figure stands (measured n^-0.24 gives ~4,600x against
the ~3,000x estimated). Measure the baseline's scaling; do not infer it from dimension.

**Also retract** the suggestion that the g-and-k advantage "erodes with more compute". The exponent
gap is 0.24 against 0.18, so closing a 7.4x lead takes e^33 more compute. True in sign, irrelevant
in practice.

## The advantage is robust to k

At a matched 12,000-evaluation budget on CPM:

| k | async eps | rejection eps | advantage |
|---|---|---|---|
| 10 | 6.6e-06 | 3.6e-04 | 54x |
| 30 | 2.1e-05 | 1.1e-03 | 51x |
| 100 | 6.7e-05 | 3.6e-03 | 53x |
| 300 | 2.0e-04 | 9.8e-03 | 49x |
| 1000 | 7.7e-04 | 3.4e-02 | 44x |

Stable over two decades of k, so no comparison here is a k=100 artefact. The async exponent steepens
mildly with k (-1.38 at k=10 to -1.75 at k=1000).

## Where k does bite — and it contradicts the paper's own advice

§Limitations says *"Raising k raises the effective sample proportionally and is the obvious lever."*
That is half the story. ESS scales with k, but the reported bandwidth is the k-th order statistic and
so loosens as k^(1/d_eff):

* **CPM (d_eff ~ 1): doubling k doubles eps.** Particles are bought one-for-one with resolution. The
  "obvious lever" is close to free of net benefit here, and may be harmful.
* **g-and-k (d = 4): eps ~ k^0.28.** Doubling k costs 21% in bandwidth for 100% more ESS — a genuine
  bargain, which is presumably the case the advice was written from.

So the advice is benchmark-dependent and the paper states it unconditionally. Two further effects
need a run rather than arithmetic: k sets the proposal breadth (so it changes where the sampler
draws, not merely how it reports), and `bisect_interval` defaults to k, so raising k lengthens the
bandwidth transient. Jobs **14262269** (k=30) and **14262270** (k=300) sweep it at `tol_init` 0.1
against the validated k=100.

---

# Does throughput convert? On CPM, yes — measured (job 14261956, the as-shipped control)

Five replicates per method, each arm given the same 3600s wall clock, so "end of run" *is* matched
compute. Truncating the faster arm to the slower one's evaluation count separates the two factors.

| | evaluations in 3600s | eps (k=100) |
|---|---|---|
| **`async_propulate_abc`** | **13,266** | **6.3e-05** |
| `abc_smc_baseline` | 7,957 | 2.4e-04 |
| async truncated to 7,956 | 7,956 | 1.2e-04 |
| `rejection_abc` (as shipped) | 100 | — (prior sampler) |

**At equal compute the asynchronous arm reaches a 3.8x tighter tolerance**, decomposing as

    3.8x  =  1.67x (throughput)  x  2.2x (per-evaluation efficiency)

**This is the opposite of g-and-k**, where the asynchronous arm is 1.2-2x *worse* per evaluation and
throughput does not convert. On the expensive 2-D simulator both factors point the same way. It is
the first time the two have been measured together rather than reported in separate sections.

**Caveat on what is comparable.** `eps` — the k-th order statistic of each method's own draws — is
the quantity that compares across methods. Posterior *contraction* is not available for the
synchronous arm here: pyABC records carry no AMIS `posterior_weight`, so that column is empty.

**The control also shows the bandwidth defect is worse than one replicate suggested.** Across five
replicates at the shipped `tol_init: 10.0`, mean `cell_volume` contraction is **13%**, not the 58%
the single validation replicate gave — the defect is erratic as well as harmful. Against the 81-84%
measured at `tol_init` 0.1, the fix is worth roughly **70 points**, not the 24 quoted earlier from
one run. The fixed production run (14262214) gives five replicates at 0.1 for the matched comparison.

## Archive size, measured (jobs 14262269/70) — the paper's advice is backwards here

**Reporting k**, one fixed 13,115-evaluation history re-reported at the k-th order statistic:

| k | eps | ESS | division_rate | cell_volume | coverage |
|---|---|---|---|---|---|
| 10 | 6.6e-06 | 14 | 92% | **85%** | both |
| 100 | 6.1e-05 | 123 | 92% | 81% | both |
| 1000 | 6.7e-04 | 925 | 92% | 75% | both |
| 3000 | 2.5e-03 | 1383 | 91% | **65%** | both |

ESS tracks k almost exactly (confirming ESS ~ k); contraction on the weak parameter degrades
monotonically; coverage holds throughout.

**Sampler k**, compared at a fixed reporting k=100 so only the sampler differs:

| sampler k | eps at k=100 | eps at its own k | ESS | division_rate | cell_volume |
|---|---|---|---|---|---|
| **30** | **4.6e-05 / 5.3e-05** | 1.4e-05 | 72-102 | **93-94%** | **82-83%** |
| 100 | 7.8e-05 / 6.3e-05 | 6.3e-05 | 267-351 | 91-93% | 81-84% |
| 300 | 9.7e-05 / 9.2e-05 | 3.1e-04 | 1846-2110 | 75% | **6-7%** |

**Raising k is actively harmful on this benchmark, through three compounding mechanisms:**

1. **The sampler gets worse.** A larger archive is a broader proposal: eps is 1.6x looser at k=300
   than at k=30 *at the same budget*. Arithmetic could not have predicted this one.
2. **The reported bandwidth loosens as k^~1** in this noise-dominated regime — a 5x looser eps.
3. **`bisect_interval` defaults to k**, so the bandwidth transient is 3x longer at k=300.

Net: `cell_volume` contraction 83% -> 81-84% -> **6-7%** while ESS climbs 85 -> 300 -> 2000. **You buy
effective particles and destroy the posterior they estimate.** §Limitations' *"raising k raises the
effective sample proportionally and is the obvious lever"* is backwards here; **k=30 is at least as
good as k=100 on every quality measure**, at a third of the ESS.

**Confound, stated.** The sweep varied k, which implicitly varied `bisect_interval` too, so the k=300
collapse mixes all three mechanisms. They are separable now that `bisect_interval` is exposed, and
worth separating before the paper states a rule. The practical conclusion stands for anyone raising
k the way the harness currently does.

**What to check next:** coverage at small k (ESS 72-102 is a noisy estimate even if the target is
sharper), and whether the same ordering holds on 4-D g-and-k, where eps ~ k^0.28 makes the bandwidth
penalty five times smaller and the advice may well be right.

---

# Five-replicate confirmation (jobs 14261956 control, 14262214 fixed)

Asynchronous arm, five replicates each, everything identical but `tol_init`:

| | division_rate | cell_volume | ESS | eps (k=100) | coverage |
|---|---|---|---|---|---|
| control, `tol_init` 10.0 | 84% +/- 6% | **13% +/- 14%** | 760 | 6.3e-05 | 5/5 |
| **fixed, `tol_init` 0.1** | **92% +/- 1%** | **81% +/- 1%** | 352 | 7.6e-05 | 5/5 |

**+68 points on the weak parameter, and the between-replicate spread collapses from +/-14% to +/-1%.**
The shipped setting was erratic as well as wrong, which is why a single validation replicate showed
58% where the five-replicate mean is 13%. Coverage holds 5/5 in both.

**eps is essentially identical between the two arms** (6.3e-05 against 7.6e-05, within 20%). Same
sampling, radically different reported posterior — the final confirmation that the bandwidth defect
was purely a reporting problem and the sampler always found the target. The fixed configuration also
gives a far more reproducible posterior, which matters for anything reported with error bars.

---

# CORRECTION — the cross-method comparisons were counting the wrong rows

**The defect.** `abc_smc_baseline` emits **two** record kinds: `population_particle` (accepted, so
pre-filtered to small losses) and `simulation_attempt`. The asynchronous arm emits only attempts.
Every cross-method comparison above that counted raw CSV rows therefore mixed accepted particles into
the baseline's "budget" and flattered it. Measured on the CPM control: baseline replicate 0 has 8,077
rows but a maximum `attempt_count` of 5,777.

**What changes.** Counting only `record_kind == 'simulation_attempt'`:

| benchmark | dim | sim cost | per-simulation | throughput | net at equal wall clock |
|---|---|---|---|---|---|
| gaussian_mean | 1 | ~us | async **1.13x worse** | async **3.64x worse** | **~4x worse** |
| g-and-k | 4 | ~4 ms | async **1.50x better** | async 2.78x worse | **~1.9x worse** |
| lotka_volterra | 4 | ~1 s | async **1.52x better** | async 1.12x worse | **~1.4x better** |
| **Cellular Potts** | 2 | ~3.7 s | async **2.51x better** | async **2.35x better** | **~7.06x better** |

Two earlier claims in this document are wrong and are corrected here:

* "the asynchronous arm is 1.2-2x *worse* per evaluation on g-and-k" — **it is 1.50x better**.
* "3.8x tighter at equal compute on CPM (1.67x throughput x 2.2x per-evaluation)" — **it is 7.06x
  (2.35x throughput x 2.51x per-simulation)**.

## The mechanism is simulator cost, not dimension

The throughput column is **monotone in cost per simulation**: 3.64x worse at microseconds, 2.78x at
4 ms, 1.12x at ~1 s, 2.35x *better* at 3.7 s. The asynchronous method pays a roughly fixed
per-arrival cost — rebuild the proposal, compute the weight, update the archive — which dominates a
4 ms simulation and vanishes against a 3.7 s one. **The crossover is around 1-3 s per simulation**,
and every benchmark falls on the side its cost predicts.

The *sampler* meanwhile is better per simulation on three of four benchmarks and only marginally
worse on the 1-D analytic one. **The losses are not a sampling deficiency; they are coordination
overhead spent on simulations too cheap to justify it.** That is the paper's own thesis with a
measured crossover instead of an assertion, and it reframes g-and-k and gaussian_mean as the correct
side of a known boundary rather than results to be excused.

**Method note for anything downstream:** always filter to `record_kind == 'simulation_attempt'` before
comparing methods on budget. The mixed-row defect silently favours whichever arm reports accepted
particles.

---

# FINAL — the CPM production comparison, both arms correctly configured

Five replicates per method, 3600s wall clock each, counting only
`record_kind == 'simulation_attempt'`:

| | throughput | per-simulation | **equal wall clock** |
|---|---|---|---|
| control, `tol_init` 10.0 | 2.35x | 2.51x | **7.06x** |
| **fixed, `tol_init` 0.1** | 2.37x | 1.44x | **4.09x** |

**Fixing `tol_init` helps the synchronous baseline more than the asynchronous arm** — the baseline's
eps improves 4.5e-04 -> 3.1e-04 while the asynchronous arm's barely moves (6.3e-05 -> 7.6e-05, within
replicate noise). That is exactly what should happen: `tol_init` is shared by construction, the
matched-epsilon baseline starts from it too, and the arm that was suffering more from a bad value
gains more from a good one.

**So the number to report is 4.09x, not 7.06x.** The larger figure comes from a configuration that
handicapped the baseline, and using it would be the same class of error as the rejection-ABC defect.

**Two distinct quantities, both needed:**

* **eps** measures where the samplers draw. Asynchronous is **4.09x** ahead at equal wall clock
  (2.37x throughput x 1.44x per-simulation).
* **The reported posterior** is what the paper's quality claim rests on, and there the bandwidth fix
  is not optional: `cell_volume` contraction 13% +/- 14% -> **81% +/- 1%**, coverage 5/5 both ways.

The fix is therefore necessary for our own claim *and* narrows the sampling gap. Both belong in the
write-up; reporting only the first would be dishonest, and reporting only the second would understate
the method.
