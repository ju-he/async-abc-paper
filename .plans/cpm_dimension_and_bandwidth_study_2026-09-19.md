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
