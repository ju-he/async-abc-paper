# Remediation plan — codex GPT-5.6-Sol review, 2026-09-17

Source: `.plans/reviews/codex-gpt56sol-2026-09-17.md` (Reject; systems result stands,
inference claims do not).

Findings verified locally before writing this plan are marked **[verified]**; the rest are
taken from the review and still need confirmation.

---

## 0. RESOLVED (2026-09-18): the raw records survive and are reachable locally

Scratch is mounted at `/home/juhe/remotes/scratch/herold2/async-abc` — no SSH needed. Every
campaign's per-particle `data/raw_results.csv` is there, `posterior_weight` column included:
`run_full_20260626_1816/{gaussian_mean,gandk,lotka_volterra,cellular_potts,scaling}`,
`twin2_20260729/straggler_*`, `cpmtwin_20260729`, `heterotwin_20260730`, `rerun_20260707/*`.
Sizes run from 4 MB to 11 GB (the `scaling` and `kfrontier` ones), so the big passes still want a
large-memory machine, but **Tier B is recomputation, not re-running.**

The original gating text is kept below for the record.

## 0b. The original gating unknown (now answered)

Everything in Tier B is "recompute, no new simulation" **if and only if** the per-particle
`data/raw_results.csv` files survived the 2026-08-14 scratch cleanup. That cleanup tarred the
`*_attempts/` jsonl traces (196k inodes) and explicitly left the 12,951 `*.csv` alone
(`.plans/scratch_inode_recovery_plan.md` §2), so they *should* be intact — but this has not
been checked since.

No live SSH ControlMaster right now, so I could not verify. Run once in your own terminal:

    ssh -fN -o ControlMaster=auto -o ControlPath=/run/user/1000/jsc-mpc/cm-%C -o ControlPersist=4h juwels

then check, per campaign root:

    find /p/scratch/tissuetwin/herold2/async-abc -maxdepth 4 -name raw_results.csv -printf '%s %p\n'

`posterior_weight` is a persisted column of `ParticleRecord`
(`experiments/async_abc/io/records.py:84,129`) **[verified]**, so a surviving `raw_results.csv`
carries the reported estimator's weights and every Tier-B metric can be rebuilt post hoc.
Caveat: the scaling/throughput sweeps set `compute_posterior_weights=False`
(`propulate_abc.py:552-554`) **[verified]** — those runs have no weights, but they are
systems-only and need none.

MCP budget is 8 node-h per rolling 24 h, 2 per job. Analysis passes fit; campaign reruns do not
— those go through your own sbatch.

---

## 0c. Progress log (2026-09-18) — scope (a) chosen

**Tier A — done.** A1 twin-quality claim (`6aded59`), A2/A3 statelessness (`6cd304c`), A4 scaling
caption + effective simulating ranks and A5 look-ahead related work (`f97be57`), A6 truncnorm and
B4 thinning (`c2d2a9d`). Paper compiles, 57 pages, no undefined refs.

**Two findings that were not in the review:**

* *The point-mass metric has a computable floor.* For the straggler configuration the analytic
  posterior has sd $0.1$, so an exactly correct posterior scores $0.1\sqrt{2/\pi} = 0.0798$ — and
  every arm scored 0.069–0.085, i.e. at or *below* the score of a perfect answer. Below, because a
  point-mass target rewards an over-concentrated archive. That is a much sharper statement than
  "the wrong column was selected", and it is now in §6.1, the Table 1 caption, and a test.
* *`raw_results.csv` did not record the field needed to rebuild the reported estimator.*
  `tolerance` is the running-minimum trajectory for the plots; during the prior phase it carries
  `tol_init` rather than `None`, and `None` is exactly how `extract_posterior` identifies bootstrap
  draws. New `proposal_tolerance` column records the stamped value verbatim (`930e32c`).

**The legacy-data question is closed, and better than expected.** Prior draws carry `weight`
exactly 1.0 (proposal = prior) in a contiguous leading run, so the boundary is *read off* rather
than fitted: 113/115/112/113/115 across five replicates of a `k=100`, `W=16` run, each reproducing
that replicate's own stored `posterior_weight` at max$|\Delta w| = 1.3\times10^{-16}$ (`f809c6a`).
It is `k + O(W)`, not `k` — the ranks in flight when the archive filled. **Every campaign on disk
is exactly reconstructible; no re-run is needed for Tier B.**

**Tier B in flight.** `async_abc/analysis/reported_posterior.py` replays `extract_posterior` over
wall-clock prefixes and scores against a reference posterior; replay is bit-identical to the live
computation (asserted). `GandK.reference_posterior_samples` gives the g-and-k reference on the
right target — p(θ | s_obs), not the full-data posterior — via the asymptotic quantile law,
validated against 40k simulated datasets (`e0eef7c`). `make_reported_recovery_fig.py` written;
gaussian panel generating.

**Scratch is mounted locally** at `/home/juhe/remotes/scratch` — the 5.6 GB gaussian
`raw_results.csv` reads at ~12 MB/s, so cache the needed columns with `awk` first (8 min) and work
from the local copy.

### Tier B status (2026-09-18, later)

Done: `reported_posterior.py` (replay, bit-identical, `930e32c`/`f809c6a`); g-and-k reference on
p(θ | s_obs) via the asymptotic octile law, validated against 40k simulated datasets (`e0eef7c`);
`make_reported_recovery_fig.py` with accuracy and ESS panels; §5.2, §6.2, limitations, abstract and
conclusion rewritten (`054ab16`).

**Two findings that change what the paper is about:**

1. *ESS is set by k, not by the budget* (`c7b874e`). 237–303 effective particles out of 1.1–1.3M
   evaluated, flat across a 13× growth of the history; ESS/k ∈ [1.9, 3.6] over a controlled sweep;
   reproduces serially. Mechanism: reporting at the tightest bandwidth reached, while that
   bandwidth is set by ESS-retention bisection against the archive. This is now the paper's main
   caveat and it reframes the flat accuracy curve, the CLT rate-condition failure, and the
   M ≈ 400–800 reported-support result (= 4–8k).
2. *Lotka–Volterra is 98% extinct* (`ae2e7ee`). 2.6% survival at the true parameter, 1.7% over the
   prior, observed data itself survival-conditioned. Throughput unaffected; posterior recovery
   carries ~2% of nominal budget. LV is now scoped like CPM.

**Open in Tier B:** the g-and-k accuracy sentence (`%%GANDK-ACCURACY%%` in `sn-article.tex`) — that
panel is generating. CPM/hetero twin quality (B3), the vendor-or-delete sweep (B5), data
availability (B6).

**LV reference posterior: recommend NOT building it.** Synthetic-likelihood MCMC needs M surviving
datasets per θ; at 1.7% survival that is ~60× the g-and-k cost, for a benchmark whose posterior
claim is now scoped away anyway.

**Practical notes.** Replay costs ~38 µs per 1000 records at d=1. Cache the campaign columns with
`awk` first — include `record_kind`: the baseline writes one `simulation_attempt` row per evaluation
(4.9M) alongside its `population_particle` rows (5,300), and pooling them scores an accumulated
cloud of attempts instead of the reported posterior. Do not monitor these jobs with
`until ! pgrep -f "<script>.py"` — the wait loop's own command line matches the pattern, so it never
exits.

## 1. The scope decision that sizes everything else

Two coherent papers can be built from what exists. Pick before doing Tier C.

**(a) Systems paper with honest inference diagnostics.** Contribution = barrier-free execution
and its throughput/utilisation wins. Posterior claims shrink to: calibration (SBC) where it can
be run, and distance-to-*reference-posterior* only on the benchmarks where a reference is
obtainable. CPM stays the systems showcase with no posterior-quality claim. Theory stays but is
explicitly scoped to the idealised estimator and the count-limited execution mode.

**(b) Method paper.** Requires all of Tier C: matched-budget twins, reference posteriors on
every benchmark, pre-registered reporting support, outcome-dependent-runtime experiment, and
the proposal-order→arrival-order transfer either proved or replaced by a theory-valid mode.

Recommendation: **(a), with the Tier-C items that are cheap (matched-budget twin, post-hoc cost
benchmark, pre-registered M/k/S) folded in.** The evidence already on disk supports (a) today;
(b) is another full campaign and the theory gap is not obviously closable.

---

## 2. Tier A — text/code fixes, no compute

| # | Issue | Action |
|---|---|---|
| A1 | "Unchanged posterior quality throughout" (§1 contrib. 5, §6.1) | **[verified]** `make_twin_tables.py:91-94` emits only `final_quality_wasserstein`, which sits at 0.071–0.085 for *every* arm including twins whose weighted estimator is 1.15 / 1.81. Emit all three columns; rewrite the claim (see §2.1 below). |
| A2 | "Stateless" (abstract, §3.1 `sn-article.tex:91`, §5 `:284`) | **[verified]** `abcpmc.py:1146-1160` builds the IS denominator from `self._snapshots`; the resulting `child.weight` is what `_build_proposal:880-892` uses for the mixture weights — and the denominator also gates the proposal-time rejection retry loop. Snapshots are **proposal-affecting**, not performance state. Narrow the claim to "the *reported estimator* is a pure function of the evaluated history" (this is true and is what `extract_posterior` replay delivers) and say plainly that the snapshot buffer and the two throttles are carried state a restart resets. |
| A3 | Kill-and-resume paragraph (`:812`) | Same root cause: it attributes the clean-vs-resumed difference to RNG re-seeding and MPI arrival order only. Add the snapshot-buffer reset as a third, *statistical*, source. |
| A4 | Scaling config vs caption (Fig. 5, Table 4) | `make_scaling_combined_fig.py:18-23,47-71` uses k=100 at 48/96 workers and k=W only above; caption says k=W throughout. Fix the caption. Also decide whether the pyABC x-axis should be *effective simulator workers* (`pyabc_sampler.py:151-196` — root coordinates, does not simulate); if so, every pyABC point shifts by one rank. |
| A5 | Related work (§2.4) | Add and engage Schälte et al., *A wall-time minimizing parallelization strategy for ABC* (PLOS ONE 2023) — look-ahead starts next-generation simulations before the current one drains, with preliminary-vs-final proposals and reweighting. Narrow the novelty claim to per-arrival update; do not claim ABC frameworks universally retain a draining barrier. |
| A6 | `gaussian_mean.py:70-81` | **[verified]** analytic posterior mean under a bounded uniform prior is the truncated-normal mean, not the clipped observed mean; the sampler at `:83-101` already uses the correct law. Swap in `scipy.stats.truncnorm.mean`, then diff every number that touches it (expected to be negligible at the current truth, but confirm). |
| A7 | Appendix B time-to-posterior | Report simulation / online coordination / retrospective weighting / total separately. Throughput stays a valid systems metric but must not be presented as end-to-end inference speed. (The 20-min number itself needs A-tier text *and* a C-tier measurement — see C4.) |

### 2.1 What the straggler-twin table should say instead

The committed data (median over 5 replicates) **[verified]**:

| arm | 20× slowdown: plain W1 | weighted W1 | analytic W1 | n_sims |
|---|---|---|---|---|
| async | 0.076 | 0.089 | 0.009 | 958,661 |
| twin_coarse | 0.071 | 1.146 | 1.067 | 3,200 |
| twin_fine | 0.075 | 1.808 | 1.729 | 3,200 |

Two honest readings, and the paper must distinguish them:

1. **At fixed wall-clock, the barrier costs inference quality, not just throughput.** This is a
   *stronger* systems claim than the one currently made, and the data support it.
2. **But it is confounded**: the twin completed 300× fewer evaluations. Separating "barrier per
   se" from "fewer simulations" needs C1.

Either way "unchanged posterior quality throughout" must go — it was never the interesting
claim.

---

## 3. Tier B — recompute from existing raw records, no new simulation

Precedent for exactly this kind of pass: `experiments/scripts/regen_quality_artifacts.py`
(post-hoc rebuild from `raw_results.csv`, no simulation, mem192 node for the big ones).

* **B1 — replace the recovery metric.** **[verified]** `convergence.py:3-9,90-150` measures W1
  from the *unweighted top-k archive* to a Dirac at the true parameter (in 1D: mean absolute
  deviation from truth). It rewards a wrongly-collapsed posterior and never evaluates the
  reported estimator. Build a weighted-prefix curve: at each checkpoint t, replay
  `extract_posterior` over records with `wall_time <= t` and score the weighted sample against a
  *reference posterior*. `posterior_quality_curve` has no weight path, so this is new code —
  but `_build_proposal` is already shared between the live proposal and history replay, so the
  replay machinery exists. Relabel every existing curve as a top-k point-concentration
  diagnostic; keep it only as a secondary panel.
* **B2 — reference posteriors.** gaussian_mean / gaussian_mean_nd: analytic, already
  implemented (`_final_quality_wasserstein_analytic`) **[verified]**. g-and-k: exact-likelihood
  MCMC via numerical inversion of the quantile function — laptop-scale, no cluster. Lotka–
  Volterra: no tractable likelihood → one long-run gold-standard ABC-SMC at much tighter ε (one
  expensive job, reused forever). Cellular Potts: no reference is feasible → **do not make a
  posterior-quality claim**; keep the corner plot + SBC and frame CPM as systems-only.
* **B3 — CPM and hetero twin quality.** **[verified]** `twin_cpm_raw.csv` has
  `final_quality_wasserstein` 0/56 populated; `twin_hetero_raw.csv` has no quality column at
  all. Recompute both from raw records, or drop the quality sentences.
* **B4 — the thinning bug.** **[verified]** `reporters.py:3878` rebinds `records` to the
  subsample before every downstream posterior/quality call, contradicting its own docstring
  ("Posterior/quality use the full history"). Top-k by loss is always retained, which is
  precisely why the unweighted metric never showed it — but the *weighted full-history*
  estimator on >200k-record histories (fast Gaussian, straggler) was computed on thinned data.
  Fix: keep `full_records` and `plot_records` separate; regression test asserting equality with
  thinning on/off; then audit which vendored artifacts flowed through this path and regenerate
  them.
* **B5 — vendor the untraceable numbers.** Six clusters have no committed artifact: multimodal
  SBC (85.7% / 86.7% / ESS 0.99 / max weight 0.011 / coverage 0.50-0.96), d=32 coverage and the
  doubled-budget d=8 k=50 run (`ksweep_summary.csv` stops at d=16 **[verified]**), the
  r-diagnostics (ζ̂=0.022, 1.1% TV, 832/33,748), the param-bias variants (error ≤0.025, ESS
  0.88, 15% widening, ≤0.013 movement), the hetero-twin posterior errors (0.003-0.034), and the
  20-min/0.3 GB retrospective pass. For each: locate the run on scratch and vendor its CSV into
  `experiments/data/paper_figures/` like everything else, or delete the claim. No middle option.
* **B6 — data availability.** Replace "will be deposited upon publication" with an actual
  deposit: immutable raw outputs + a one-command regeneration pipeline. `make_twin_tables.py:33-38`
  hard-codes private scratch paths — that has to go.

---

## 4. Tier C — new runs

* **C1 — matched-budget twin (do this one regardless of scope choice).** Re-run the straggler
  twin arms at a *fixed evaluation count* for both arms instead of fixed wall-clock. This is the
  single experiment that turns A1's confounded result into a clean claim. Cheap: the twin
  already tops out at ~3,200 evaluations at 20×, so cap both arms there (plus one higher rung).
  Also report the five hung jobs as time-to-result/failure outcomes rather than retrying them
  away.
* **C2 — record/replay control (stronger, more work).** Record the async proposal stream and
  replay it under the barrier schedule. Removes the "barrier changes the arrival history and
  therefore the proposals" objection to `abcpmc_barrier.py:119-133` entirely. Only worth it for
  scope (b).
* **C3 — outcome-dependent completion time.** The current coupling study adds a delay keyed on
  θ *after* evaluation, so a θ-only ratio r(θ) absorbs it by construction. The theory's real
  exposure is runtime depending on the trajectory/discrepancy (plausible for CPM), which
  distorts p(ρ|θ) and cannot be absorbed. Run a simulator whose runtime is a function of ρ.
  Report the result honestly whichever way it goes; if it breaks, scope the theorem to θ-only
  coupling and say so explicitly.
* **C4 — post-hoc cost benchmark.** Time and profile the retrospective pass at each history
  size, commit the log. Analysis-only, fits the MCP budget, closes A7 and one of B5's items.
* **C5 — pre-registered reporting support.** §6.2 currently picks M≈400-800 *after* seeing
  which value crosses nominal coverage — that shows coverage is tunable, not that the estimator
  is calibrated. Fix M, k, S on a calibration split before evaluation, then report held-out SBC
  across dimensions and models.

---

## 5. Tier D — theory scoping

* **D1 — proposal order vs arrival order.** The proofs use proposal order; the implementation
  consumes an arrival-ordered log (`propulate_abc.py:727-749`) and the appendix already states
  the transfer is unproved. Cleanest honest move: make the count-limited / drain-to-completion
  configuration an explicit **theory-valid mode**, show empirically that it matches the deadline
  mode, and present the deadline mode as the engineering default outside the theorem.
* **D2 — r = 1, positive limiting bandwidth, asynchronous filtration.** Currently assumed or
  unverified. Either commit the diagnostics (B5) and present them as evidence-for-assumptions,
  or downgrade every downstream statement to explicitly conditional. In particular
  `sn-article.tex:595` ("the estimator it reports is consistent") cannot stand as written.

---

## 6. Suggested order

1. §0 SSH check — decides whether Tier B is recomputation or re-running.
2. A1 + A2 + A3 + A6 (a day; removes the two claims that make this a reject).
3. B4 (the thinning bug is a live correctness defect, and it gates trusting any Tier-B output).
4. B1 + B2 on gaussian_mean and g-and-k (both references are cheap) — this is the proof of
   concept that the honest metric can be reported at all.
5. C1 matched-budget twin.
6. B5 vendor-or-delete sweep, B3, B6.
7. Scope decision (§1) → either stop here and write (a), or continue into C2/C3/C5 for (b).
8. D1 + D2 last, once it is clear which experiments exist to back them.
