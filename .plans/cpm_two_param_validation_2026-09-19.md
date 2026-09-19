# Validating the two-parameter CPM setup

**Written 2026-09-19.** Follows `.plans/HANDOFF_cpm_experiment_2026-09-19.md` and the proposal it
points at, `.plans/cpm_setup_proposal_2026-09-19.md`. Job 1 (replicate averaging) is done; job 2
(run it and check the forecast) is measured below.

---

## Job 1 — replicate averaging, and a second thing the forecast assumed

`n_replicates_per_evaluation: 4` sat in both proposed configs and nothing read it. An evaluation now
runs `k` simulations at the same theta and scores them through
`DistanceMetric.calculate_distance_replicates`, which averages the raw feature arrays — averaging
features, not distances, is what divides the within-theta noise by `k`.

The replicate seeds are a deterministic function of the evaluation's own seed. The first one **is**
that seed, so `k=1` reproduces the previous behaviour exactly and the shipped `cellular_potts.json`
is untouched; the rest are SHA-256 derived rather than consecutive integers. A replicate that fails
fails the whole evaluation — an evaluation averaged over fewer replicates carries more noise than
the tolerance schedule was set for, which is why the offline screening drops incomplete units too.

**A second gap, not in the handoff.** `DistanceMetric` averages the distance to *each* reference
directory. Every forecast was computed against the *average of the four references*, which is a
different and half-as-noisy observation. Scoring the four reference directories themselves as a
candidate measures the gap exactly:

| reference read as | d(4-replicate candidate) | d(1-replicate candidate) |
|---|---|---|
| one replicate-averaged observation | **0.000** | 0.023 |
| four independent observations | 0.012 | 0.035 |

The 0.012 is an irreducible pedestal under *every* candidate. Both proposed configs now set
`average_reference_replicates: true`; the flag defaults off, so nothing else in the repo changes.

Tests: 30 new, suite 774 passed / 10 skipped.

---

## The forecast holds — measured on the production code path, with no sampler involved

`diag_cpm_prior_corpus.py` (new) draws the prior and scores every draw through the shipped
benchmark — same `CellularPotts` object, feature-space model, replicate averaging and averaged
reference the experiment runs with. Rejection ABC is then "keep the smallest rho", exactly as in
`diag_cpm_posterior_forecast.py`, but with the production discrepancy rather than one fitted inside
the diagnostic. **2000 draws, 8000 simulations, zero failures, 0.21 node hours.**

| acceptance | tolerance | division_rate | (forecast) | cell_volume | (forecast) | corr |
|---|---|---|---|---|---|---|
| 20% | 0.1168 | 77% | — | 4% | — | +0.11 |
| 10% | 0.0431 | **84%** | 85% | **16%** | 16% | +0.08 |
| 5% | 0.0201 | **89%** | 88% | **31%** | 38% | +0.24 |
| **2%** | **0.0081** | **89%** | 91% | **58%** | 64% | +0.09 |
| 1% | 0.0037 | 89% | 91% | 66% | 78% | +0.25 |

Contraction is 1 − sd_post/sd_prior. The truth is inside the 90% interval for both parameters at
every acceptance level. Bias at 2%: −0.020 on division_rate, +0.061 on cell_volume, both small
against the prior range.

The forecast is confirmed within its own sampling noise — at 2% acceptance the posterior has 40
members, on which the standard deviation carries about ±11%. The agreement at 10% and 5%, where the
sample is 200 and 100, is close to exact. **`cell_volume` is a real second direction**, and
`division_rate` contracts as forecast.

### Two numbers nothing in the repo knew

* **14.9 s per evaluation** (median, four simulations, 48 ranks, 50³) — 3.7 s per simulation.
* **2% acceptance is a tolerance of 0.0081**, 5% is 0.0201, and the prior's *median* discrepancy is
  0.459. Every CPM config sets `tol_init: 10.0`, which is a thousand times above that median — so
  `rejection_abc`, which uses `tol_init` as a fixed threshold and stops at `k` accepted, accepts
  every draw it makes and terminates after ~100 evaluations. **Its reported CPM posterior is the
  prior.** That is a pre-existing property of the shipped `cellular_potts.json` as well, and it is
  worth knowing before the method comparison is read; see the open question below.

---

## The real async ABC run — the setup delivers, and one premise of the handoff does not

One method (`async_propulate_abc`), one replicate,
`experiments/configs/cellular_potts_two_param_validate.json`. **13,115 evaluations, 52,460
simulations, zero failures, 1.05 node hours.**

A note on budget, because the config does not say what it does: `max_simulations: 2000` is **inert**
for this method. `propulate_abc` sets the generation budget to unlimited whenever `max_wall_time_s`
is set (`propulate_abc.py:530`), deliberately — the wall clock is meant to be the binding criterion
— so a replicate runs the full 3600 s and evaluates ~13,000 times, not 2000. That is also why the
full config costs about **10.5 node hours, not the ~4 the handoff estimated**.

### The reported posterior matches the forecast

| | division_rate | cell_volume | correlation | coverage |
|---|---|---|---|---|
| forecast (screening corpus) | 91% | 64% | +0.04 | 94% / 100% |
| rejection ABC, production path, 2% | 89% | 58% | +0.09 | both covered |
| **async ABC, reported posterior** | **90%** | **58%** | +0.12 | **both covered** |

Bias −0.024 and +0.054 of the prior range; ESS 1222 of 12,827. **The two-parameter claim holds.**

### And the premise that rejection ABC is a lower bound on the sampler is false — for the estimator the paper reports

Replaying `extract_posterior` over prefixes of the same run separates the sampler from the reporting
rule, and they are nowhere near each other:

| evaluations | reported AMIS posterior | top-100 archive |
|---|---|---|
| 500 | −0% / 4% | **87%** / 31% |
| 2,000 | 3% / 3% | **94% / 76%** |
| 5,000 | 12% / 3% | **95% / 86%** |
| 10,000 | 89% / 46% | 94% / 86% |
| 13,115 | **90% / 58%** | 93% / 87% |

(division_rate / cell_volume.) The sampler has found the target within **500** evaluations — its
archive is already at 87% on division_rate, better than rejection ABC manages with 2000 draws. The
*reported* estimator is still the prior at 5,000 and only switches on between 5,000 and 10,000.
At the matched 2000-evaluation budget the comparison is 3% reported against 89% for rejection ABC
and 94% for the run's own archive.

Nothing is wrong: this is the paper's own §Limitations effect — the reported posterior's spread is
set by the bandwidth the ESS-retention schedule has reached, not by what the archive knows — and
this is the first time it has been measured on a CPM target that is actually identified. Two
consequences worth carrying into the write-up:

* **The handoff's "rejection ABC is a lower bound on what the adaptive sampler achieves per
  simulation" is true of the archive and false of the reported estimator.** Any forecast-to-run
  comparison has to say which of the two it means.
* **The over-concentrated archive under-covers, as the paper predicts.** At 5,000 and 10,000
  evaluations the top-100 archive's 90% interval *excludes* the true division_rate (bias −0.025
  against sd 0.015–0.018), while the reported posterior covers it at every budget. The proposal's
  warning about stopping the tolerance schedule at ~2% acceptance is the same effect seen from the
  other side.

---

## Open question for job 3 — `rejection_abc` on CPM is a prior sampler

`rejection_abc` uses `tol_init` as a **fixed** threshold and stops at `k` accepted. Every CPM config
sets `tol_init: 10.0`; the prior's median discrepancy is 0.459 and 5% acceptance is at 0.0201. So it
accepts every draw, terminates after ~100 evaluations, and its posterior is the prior — and it is in
the paper's CPM posterior figure (`sn-article.tex:605`) as one of three arms. That was harmless when
nothing on CPM was identified. It will not be harmless in a figure where the other two arms
contract.

There is no per-method inference override in the harness (`tol_init` is shared, and the async method
needs it loose as a *starting* bandwidth), so the options are:

1. **Report it as measured and say what it is** — one honest sentence, no code change.
2. **Drop it from the CPM config**, as `lotka_volterra.json` already does.
3. **Add per-method inference overrides** and give it `tol_init: 0.02`. It would then spend its full
   hour and accept ~12 of ~240 draws — a genuine ABC posterior at a matched wall-clock budget. This
   is the real fix and it touches the whole campaign's config handling.

The production run includes all three methods, so the data supports any of them.

---

## Cost

| job | what | node hours |
|---|---|---|
| 14261876 | devel smoke test (`--test`) | 0.04 |
| 14261882 | prior corpus, 2000 draws | 0.21 |
| 14261878 | async validation, 1 replicate (cancelled after it) | 1.05 |
| 14261956 | production: 3 methods x 5 replicates | ~10.5 (estimated) |

Against 4 node days authorised. Scratch left with no evaluation directories.
