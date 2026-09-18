# Handoff — Cellular Potts parameter & settings sweep

**Written 2026-09-18.** Start a fresh session with this file. Budget authorised by the user:
**24 node hours**, not necessarily in one go.

---

## The task in one line

Find a Cellular Potts configuration — *settings first, then parameters* — in which ABC can actually
identify something, so the benchmark can carry an inference claim instead of being scaled-only.

## Why the obvious framing is wrong, and what was already measured

Do not start by picking new parameters. The binding constraint is not the parameter choice.

Measured on the 170,228 stored asynchronous evaluations of `rerun_20260707/cellular_potts`
(diagnosis is committed in this repo; the numbers below are reproducible from that CSV):

**1. `division_rate` is a one-sided cliff, flat over 80% of its prior.** Median loss by prior
quantile `u` (physical `p = 0.00006 + u·0.59994`):

| u | p | median loss |
|---|---|---|
| 0–0.01 | 0.00006–0.006 | 3.54 |
| 0.01–0.02 | 0.006–0.012 | 1.26 |
| 0.02–0.05 | 0.012–0.030 | 0.554 |
| 0.05–0.10 | 0.030–0.060 | 0.334 |
| 0.10–0.20 | 0.060–0.120 | 0.264 |
| **0.2–1.0** | **0.12–0.60** | **0.251 → 0.238** |

The data say "enough divisions happened" and nothing more: bounded below, unbounded above. That is
why the posterior runs to the prior edge and why the 0.05 reference sits below all 500 reported
draws.

**2. `motility` has a shallow interior optimum** at `M ≈ 1000–1500` (u ≈ 0.10–0.15), rising to a
median loss of 0.98 above `M = 5000`. The well is ~5% deep.

**3. The killer — signal-to-noise ≈ 0.30.** Over 3,456 cells of the prior (0.5% × 0.5%) with ≥20
evaluations each:

- within-cell loss sd (repeat draws at essentially the same θ): **0.053**
- between-cell spread of the cell medians, across the whole box: **0.016**

Monte Carlo noise in ρ at fixed θ is **3.4× the entire systematic variation over the parameter
space**. No choice of parameters survives that — a rerun would produce another flat posterior.

**4. Why the noise is that large.** ~55 cells (`log_n` median 4.007), feeding **32-bin** radial
profiles → 1.7 cells per bin, from **one seed** and **one snapshot** at t=500. And `r95 = 17.3` in a
50³ box (half-width 25): the cluster is already pressed against the walls, which is what caps growth
and makes division rate unidentifiable from above.

**5. A dead feature.** `dbscan_gaslike_fraction` has scaler `median=0.0, scale=1.0` — the
RobustScaler fallback when IQR = 0, i.e. it was constant across the training sims — and
`block_norms = 0.0`, which `feature_scaling.py:1816` silently rewrites to 1.0 (`0.0 or 1.0`). It
consumes 1/7 of the distance budget and carries nothing. Recalibrate it over a motility range that
straddles the gas–liquid transition, or drop it.

---

## Current configuration (for reference)

`experiments/assets/cellular_potts/sim_config.json`:

- Geometry `blocksize [50,50,50]`, `timesteps 501`, `CellInfo` writer `steps 500` (**one snapshot**)
- Cell type 9 = cancer; `volume.default 500` (domain capacity ~250 cells, actual ~55)
- `volume.lambda[9] = 7.5`, `surface.lambda[9] = 1` (types 7,8 use 5.625)
- `adhesion.matrix[9] = [0,151,0,0,0,0,0,0,0,103,0,...]` → J(9,liquid)=151, J(9,9)=103
- `temperature = 50`
- `orientation`: `persistentRandomWalk`, `persistenceMagnitude 0.834`, `recalculationtime 15`,
  `motilityamount[9] = 50`
- `DefineFunctions`: `division_cond_cancer() = ( volume >= 0.9 * volume0 ) & ( rnd() <= 0.0004723 )`
- filling: 4 spheroids, radius 5, centre 25, celltype 9

`parameter_space_division_motility.json` — the two currently inferred:

| name | path | normalised | physical |
|---|---|---|---|
| division_rate | `define_functions.division_cond_cancer[1]` | [0,1] | [0.00006, 0.6] |
| motility | `CellsInSilico.orientation.motilityamount[9]` | [0,1] | [0, 10000] |

Feature space (`sims_feature_space_model.json`), 7 blocks / 43 dims, `feature_weights = {}` so each
block gets 1/7 after block-norm division:

| block | type | dims |
|---|---|---|
| log_r95 | RobustScaler1D | 1 |
| log_n | RobustScaler1D | 1 |
| radial_fa_equal_volume | CurvePCAScaler | 10 |
| radial_s2_equal_volume | CurvePCAScaler | 10 |
| radial_density_profile_equal_volume | CurvePCAScaler | 10 |
| dbscan_gaslike_fraction | RobustScaler1D | 1 (dead, see above) |
| pair_correlation_gofr | CurvePCAScaler | 10 |

---

## Proposed design

### Phase A — settings, i.e. kill the noise floor (do this first)

Fix θ at the reference point and vary only the **protocol**, measuring the within-θ sd of each
feature block and of the total discrepancy. Candidate factors, in descending expected leverage:

1. **Snapshot averaging — nearly free.** The sim already runs 501 steps and writes only at 500. Set
   `Writers.CellInfo.steps` to emit at e.g. 300/350/400/450/500 and average the summaries over the
   last few. ~5 quasi-independent samples at zero extra simulation cost.
2. **Fewer radial bins.** 32 → 8 in `distance_metric_params.json` (`n_bins` for the three radial
   blocks and `pair_correlation_gofr`). 4× the cells per bin, also free.
3. **Replicate seeds per evaluation**, k ∈ {1, 2, 4}. Cuts noise by √k at k× the cost; use only if
   1+2 are not enough.
4. **Domain size** 50³ → 80³ (the one expensive factor). Gets the cluster off the walls and lets
   `log_n` stay monotone in division rate instead of saturating. This is what makes division rate
   identifiable *from above* at all.

Target: get SNR from 0.30 to ≥1.5. Report the achieved within-θ sd per block for each protocol.

### Phase B — parameters, with the winning protocol

Your seven summaries measure the spatial structure of a cohesive population. Three blocks (g(r),
radial density, gaslike fraction) are driven by **cohesion**; two (radial FA, S2) by **shape and
directional order**. Nothing currently inferred drives those five. Candidates:

- **adhesion J(9,9)** (currently 103) — the strongest driver of g(r), density steepness, gaslike
  fraction
- **surface λ(9)** (currently 1, vs 5.625 for types 7/8) — drives FA and S2, which presently have
  no driver at all
- **motility amount**, log-uniform over M ∈ [100, 4000] where the response actually lives
- **persistenceMagnitude** (0.834) or **recalculationtime** (15) — directional correlation, which is
  what S2/FA encode; separates "fast and random" from "slower and persistent", which motility amount
  alone cannot express
- **division**, only if reparameterised: use λ = p·T (expected division attempts per cell). The
  measured transition is λ ≈ 6 → 60, so log-uniform over p ∈ [0.002, 0.2] puts it mid-prior instead
  of in the bottom 2%.

**Identifiability trap to avoid:** adhesion and temperature are degenerate in CPM — only ΔH/T enters
the Boltzmann acceptance, so static structure cannot separate them. Fix T = 50 and infer J, or
parameterise J/T explicitly. Do not put both in.

### Phase C — the screening metric

For each candidate parameter and each feature block, report

    identifiability = (between-θ sd of the block mean over the prior range)
                      / (within-θ sd of the block at fixed θ)

plus the same for the total discrepancy, and the saturation window (where the response goes flat).
A parameter scoring < 1 will not be identified by a campaign, however long it runs. This is the
check that would have caught the current pair.

---

## Cost

A CPM evaluation is **~5 s** (9.33 sims/s across 48 workers, from the paper's own throughput).
So:

- a 2,000-point screening design at 50³ ≈ **4 minutes on one node**
- the same at 80³ (~4× cost) ≈ 15 minutes

**24 node hours is far more than screening needs.** Spend it on breadth — more design points, more
replicate seeds, more protocol variants — not on long runs. Keep individual jobs small; the MCP
enforces **2 node-hours per job and 8 per rolling 24 h session** (`remaining_budget` on cluster
`juwels-cluster`), so plan several small submissions rather than one large one, or have the user
submit directly.

---

## PREREQUISITE — do this before anything else

There is **no live SSH ControlMaster**, so no cluster tool will work. Ask the user to run, in their
own terminal:

    ssh -fN -o ControlMaster=auto -o ControlPath=/run/user/1000/jsc-mpc/cm-%C -o ControlPersist=4h juwels

Then `mcp__jsc-mpc__run_on_login` etc. become available. Cluster name is `juwels-cluster` (not
`juwels`); the other configured cluster is `jupiter`.

Deploy recipe and production job command: memory `reference_asyncabc_cluster_deploy.md` (rsync, no
push alias). Scratch is also mounted locally at `/home/juhe/remotes/scratch/herold2/async-abc`,
which is how the diagnosis above was done without SSH — reads are ~12 MB/s, so `awk` the columns you
need into a local cache before doing anything iterative.

---

## CLEANUP — this is a hard requirement, not housekeeping

Scratch hit **98.1% of its inode soft limit** in August (3,925,347 / 4,000,000). The sweep must not
refill it. Two distinct leaks:

**1. Per-evaluation simulation directories.** `CellularPotts._cleanup_eval_dir`
(`experiments/async_abc/benchmarks/cellular_potts.py:555`) zips `.vtk/.csv/.vti` and removes the
directory — but only when `keep_eval_dirs` is false. **Verify `keep_eval_dirs` is not set to true in
any sweep config** (`cellular_potts.py:504`, default `False`). At 2,000+ evaluations a leak here is
thousands of directories.

**2. Attempt traces — the one that actually caused the crisis.**
`experiments/async_abc/inference/_attempt_trace.py:57` opens one `worker_<rank>_pid_<pid>.jsonl` per
MPI rank per run. A 384-rank run leaves 383 files in one directory; these were **73% of the folder's
inodes** (196,171 files, 2,758 GB). `scaling_runner.py:630` defines `_cleanup_combo_artifacts()`
which removes them — **and it has no production caller**. Check whether that is still true before
submitting, and either wire it in or archive after each job.

Do **not** simply delete traces: they are not a strict subset of the CSVs (measured deficit 7.7% of
rows on `scaling_cpm w384_k1000`, and the conversion drops `pid` and `param_key`). Archive with
`tar | pigz -6` (measured 4.7× compression) as in `.plans/scratch_inode_recovery_plan.md`, which has
the full procedure and the restore path.

Budget an inode check before and after: `jutil project dataquota -p tissuetwin`.

---

## What "done" looks like

1. A protocol (snapshots / bins / seeds / domain) with a **measured** within-θ noise floor and the
   SNR it achieves, against the current 0.30.
2. A ranked table of candidate parameters by the Phase-C identifiability score, with saturation
   windows in physical units.
3. A recommended parameter set and prior ranges, with the evidence for each.
4. Scratch inode usage no higher than when you started.
5. The screening script committed under `experiments/scripts/` (convention: `diag_*.py` for
   diagnostics that produce cited numbers — and give it a real default path, not a session
   scratchpad; two committed diagnostics were already broken that way and had to be repaired in
   `24e3fae`).

## Context you may want

- `.plans/review_remediation_2026-09-17.md` — the review remediation this came out of, scope (a)
- `.plans/reviews/codex-gpt56sol-2026-09-17.md` — the external review
- `.plans/scratch_inode_recovery_plan.md` — the inode procedure, measured numbers
- The paper now scopes CPM as systems-only and says why (§5.2, §6.2); if this sweep succeeds, that
  scoping is what gets revisited.
