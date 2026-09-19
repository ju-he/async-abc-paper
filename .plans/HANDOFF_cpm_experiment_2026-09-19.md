# Handoff — build and run the better CPM experiment

**Written 2026-09-19. PARTLY SUPERSEDED the same day: jobs 1 and 2 are done.** Read
`.plans/cpm_two_param_validation_2026-09-19.md` first — replicate averaging is wired and the
forecast is confirmed on a real run (90%/58% contraction, truth covered). Only **job 3**, the
paper rewrite, is still open, and the validation document lists three measured facts that change
how it should be written. Two claims below are now known to be wrong: the ~4 node hour estimate
(it is ~10.5, because `max_simulations` is inert under a wall-time cap) and "rejection ABC is a
lower bound on what the adaptive sampler achieves per simulation" (true of the archive, false of
the reported estimator).

## Where things stand

The screening campaign is finished: 71,976 evaluations, 2.7 of the 24 authorised node hours, zero
failures, scratch clean (19 `.tar.gz` archives, no directories). Working tree clean at `e36bac7`.

**Read `.plans/cpm_setup_proposal_2026-09-19.md` — all of it, including its three addenda.** The
conclusion moved several times as better statistics were applied; the last statement of each
question is the right one, and each addendum says plainly what it corrects. `.plans/cpm_screening_results_2026-09-18.md`
is the measurement record behind it. `.plans/HANDOFF_cpm_sweep_2026-09-18.md` is SUPERSEDED.

**The proposed experiment.** Infer `division_rate` and `cell_volume`, log-uniform on [0.001, 0.2] and
[200, 1200], truths 0.009 and 500, `motility` fixed at 1400, distance over `log_n` and `log_r95` at
0.5/0.5, four replicate seeds, 50³, one snapshot at t=500, tolerance floor ~2% acceptance.
Forecast: division_rate 91% contraction, cell_volume 64%, posterior correlation +0.04, coverage
94%/100%. Config `experiments/configs/cellular_potts_two_param.json`, validated end to end;
`cellular_potts_division_only.json` is the conservative fallback.

## The three jobs, in order

**1. Wire up replicate averaging. This is a blocker.** `n_replicates_per_evaluation: 4` is in both
configs and nothing reads it — `CellularPotts.simulate` runs one simulation and calls
`DistanceMetric.calculate_distance`. Run `k` simulations per evaluation with distinct seeds and pass
their directories to nastjapy's `DistanceMetric.calculate_distance_replicates`, which averages the
raw feature arrays (the same operation `diag_cpm_screening.py` does offline). Every forecast assumes
it; at one seed the prediction drops to ~85%/16%. Needs tests, and the seed derivation must be
deterministic in the evaluation's own seed.

**2. Run the experiment and check the forecast.** Everything so far is forecast from screening
corpora via rejection ABC, which is a lower bound on what the adaptive sampler achieves per
simulation. A real async ABC run is the validation. Budget roughly 4 node hours for the full
config (3 methods × 5 replicates × 2000 evaluations × 4 seeds); consider one method first.
The number to check is contraction against the prior, not the loss curve.

**3. Decide what the paper says.** If the posterior matches the forecast, CPM carries a
two-parameter inference claim and §5.2/§6.2 need rewriting. If it does not, the fallback is the
one-parameter config. Either way the *stated reason* in the paper needs correcting: it is not that
Monte Carlo noise swamps the signal — a designed experiment measures SNR 3.78 where the campaign's
own evaluations gave 0.30, because that 0.30 was measured over the region the sampler had already
collapsed onto.

## What not to re-litigate

Screened and settled, with numbers in the proposal: motility (0.82–0.86 confounded with
division_rate, posterior = prior at every setting including 16 replicate seeds and 1001 timesteps);
adhesion both sides, at low and high motility, wide and narrow priors, at its actual transition
(1% contraction); surface_lambda, persistence, recalc_time, temperature; the sibling nastjapy
campaign's `surface_roughness`/`invasion_ratio`; MSD; and six new intensive blocks including
per-cell shape. All improve what can be *seen*; none add what can be *separated*.

**Calibration for reading any of these tables:** identifiability ≳10 *and* low confounding gives
~90% posterior contraction; 3–4 gives 45% if unconfounded and 0% if not; ≲2 gives nothing. The old
"below 1 will not be identified" threshold is far too generous.

## Tools

`experiments/scripts/diag_cpm_screening.py` — simulate/analyse; priors, centre and fixed parameters
all CLI-overridable, so a new screen needs no code edit.
`diag_cpm_resolved_directions.py` — how many directions the discrepancy resolves, by held-out
prediction. Use this for confounding questions, not per-parameter scores.
`diag_cpm_posterior_forecast.py` — reads the posterior off a corpus with no new simulation.

Corpora are archived at `/p/scratch/tissuetwin/herold2/async-abc/cpm_*.tar.gz`; unpack one next to
its `protocol.json` so `<dir>/corpus/rank_*.jsonl` exists. Deploy and job recipes: memory
`reference_asyncabc_cluster_deploy`. Local analysis needs pandas < 3 (the repo `.venv` has 3.0.5,
which nastjapy's DataHandler cannot load reference CSVs under); build a scratch venv if needed.
