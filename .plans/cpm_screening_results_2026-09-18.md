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
- Scratch inodes: baseline 3,015,969 at the start. Per-evaluation directories are removed by the
  driver — **0 stray `eval_*` directories** after four jobs. The four corpora (212 files) were
  archived to `cpm_screen_{50,80,r2,shipped}.tar.gz` under
  `/p/scratch/tissuetwin/herold2/async-abc/` and the directories removed: **net +4 inodes**.
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
