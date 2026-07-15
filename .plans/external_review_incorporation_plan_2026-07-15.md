# External peer-review incorporation plan (2026-07-15)

**Source:** `.plans/reviews/external-peer-review-2026-07-15.md` (recommends major revision).
**Cross-references:** internal review `.plans/paper_review_2026-07-05.md` (Part I/II) and the
completed rerun campaign (`.plans/campaign_final_report_2026-07-15.md`). Paper source:
`latex/sn-article-template/sn-article.tex`. Test venv: `nastjapy_copy/.venv/bin/python`.

This plan covers **Tier 0 + Tier 1** in actionable detail (mostly text + one small
experiment). Tier 2/3 are previewed at the end and will get their own plan once the
positioning question is settled.

## Framing note (why Tier 0 is cheap)

The abstract, contribution 5 (tex:50), and conclusion (tex:353) **already** state the honest
position: *the idealized full-mixture estimator inherits AMIS's guarantees; the practical
sliding-buffer/top-k estimator lies outside those guarantees and is validated empirically by
SBC.* The method section, theory firewall, and appendix, however, still contradict this by
claiming the **reported** posterior "evaluates exactly this object … over the full evaluated
history" and is "bit-identical" (tex:115, 171, 399). **Verified against the code:** the reported
posterior uses `n_proposals = amis_snapshots = 20` snapshots (`abcpmc.py:1318-1322`,
`propulate_abc.py:543`), i.e. a bounded draw-proportional mixture of m=20 reconstructed
proposals + a defensive prior floor — *not* the full n≈10⁷ history. So Tier 0 = make the body
consistent with the abstract's existing hedge. This also resolves external concern 2's fork:
the O(nk) complexity is **correct** precisely *because* m=S=20 is a fixed constant
(O(n·m·k)=O(nk)); what is wrong is only the "exact / full history" wording.

---

## TIER 0 — Reposition the theory honestly (text only, no new math)

Addresses external concern **1** (CLT/consistency not established) by scoping the claims to
what we can defend, rather than proving a new theorem. Aligns with internal §1.2a, §2.1, II.8.7.

**T0.1 — Downgrade Theorems 1–2 to a scoped Proposition + fix the citation.**
- §4 (`sec:theory`): keep the consistency/CLT statement as characterizing the **idealized**
  full-mixture estimator, but (a) state it as a *Proposition* inheriting the AMIS rationale, not
  a self-contained theorem; (b) cite the actual consistency result — Marin, Pudlo & Sedki,
  "Consistency of adaptive importance sampling and recycling schemes," *Bernoulli* 25(3), 2019
  (arXiv:1211.2548), for **modified** AMIS — and note that the original AMIS convergence
  (`cornuet2012amis`) is heuristic. Add the bib entry `marin2019consistency`.
- Proof sketch (tex:169): "The result follows from \cite[Thm.~1--2]{cornuet2012amis}" → cite
  marin2019consistency for the recycling-scheme consistency, with the caveat that its scheme
  restricts adaptation; if the mapping to Conditions 1–4 is not clean, say so explicitly.

**T0.2 — Fix the augmented-space formulation (concern 1, first point).**
- The proof sketch treats samples as θ_i∼q_i and inserts K_ε(ρ_i) afterward. Reformulate on the
  augmented space (θ,ρ) with proposal q_i(θ)·p(ρ|θ) and target ∝ π(θ)K_ε(ρ); since K_ε is
  bounded this is a short correction (one or two sentences in the proof sketch) but it must be
  stated correctly. Reference the Wilkinson smooth-likelihood target `\cite{wilkinson2013}`.

**T0.3 — Make the CLT-rate assumption explicit instead of implied.**
- Condition 3 currently assumes only ε_n→ε_∞. A CLT centered at π_{ε_∞} needs a rate such as
  √n{π_{ε_n}(f)−π_{ε_∞}(f)}→0. **Add this as an explicit clause of Condition 3** (so the theorem
  is correctly conditional), and note that a schedule as slow as 1/log n would violate it — i.e.
  the theorem holds *given* a fast-enough ε-schedule, which our runs satisfy. Do **not** claim
  it follows from monotone convergence.
- Condition 4: note it is a *consistency* assumption for the denominator; for the CLT it needs a
  rate too. Since the **reported** estimator uses the bounded m=20 mixture (see T1.1), reframe
  Condition 4 as "the assumption, validated empirically by SBC, that the bounded snapshot mixture
  tracks the cumulative denominator once the proposal sequence stabilizes."

**T0.4 — Remove the incorrect "PMC lacks CLTs" claim (concern 1 / additional).**
- tex:169: "The CLT yields confidence intervals … which generational ABC-PMC does not provide
  without extra machinery." SMC central-limit theory is extensive (Del Moral; Chopin;
  Douc–Moulines). Delete or soften to: "which typical generational ABC-PMC implementations do
  not expose per run."

**T0.5 — Note the defensive-prior alignment (sets up Tier 3).**
- The reviewer's suggested fix (persistent defensive component q_n^{def}=δπ+(1−δ)q_n so the
  Condition-1 density-ratio bound holds) is *close to what the code already does* (the prior
  floor 0.5/(m+1) in `extract_posterior`, `abcpmc.py:1362-1371`). In the theory-vs-implementation
  paragraph (tex:171) mention this component explicitly and flag "formalizing the estimator with
  this defensive mixture, giving a complete theorem with rates, is deferred (Tier 3 / future
  work)." This turns a gap into a scoped, honest limitation.

---

## TIER 1 — Text-rigor fixes + one small experiment

### T1.1 — Estimator exactness + complexity (concern 2; internal II.2, never landed)

The single most important body edit. Make the reported estimator's description match the code
(m=20 snapshots + prior floor), which also makes the O(nk) claim correct-by-construction.

- **tex:115, weight object (iii):** replace "$\bar q_n$ the cumulative proposal mixture over the
  \emph{full} evaluated history … all reported posteriors and quality metrics are computed from
  it. The bounded buffer in (ii) is the online approximation of the cumulative denominator that
  (iii) evaluates \emph{exactly} after the run." →
  a draw-proportional deterministic mixture of m≤S=20 history-reconstructed proposals **plus a
  defensive prior component** (mass floored at 0.5/(m+1)); the retroactive step evaluates *this
  bounded object*, the Condition-4 approximation to the full cumulative mixture. Also correct the
  over-claim "all reported posteriors and quality metrics are computed from it" — the
  Wasserstein-vs-time quality curves use the top-k archive (async) / final population (sync), not
  the estimator (internal §4.2); the estimator drives the reported posterior densities and SBC.
- **tex:171:** "evaluates exactly this object retroactively" → "evaluates a bounded-m (m≤S=20)
  draw-proportional approximation of this object with a defensive prior floor." Keep the two
  honest caveats already there (buffer, Condition 1).
- **tex:399 (appendix):** "over the full evaluated history … bit-identical to the unchunked
  computation" → the denominator uses m≤S=20 reconstructed proposals; state complexity
  **O(n·m·k)=O(nk) with m=S=20 fixed** (this is why it is O(nk), not O(n²k)). *Keep*
  "bit-identical" but fix the reason: each history point's denominator is evaluated
  independently within one chunk (no reduction *across* chunks), so chunking changes memory, not
  the value — the reviewer's "log-sum-exp not associative" objection does not apply because there
  is no cross-chunk reduction.
- **Add a short pseudocode/paragraph** in App A for the retroactive denominator: the m snapshots
  are `picks = archive_idx[::step][:m]` with `step = max(1, |archive|//m)`; draw-proportional
  weights; prior floor 0.5/(m+1). State the snapshot count (m=20) explicitly.
- **Measurement (one number, from existing runs):** report the measured `extract_posterior`
  wall-time and peak memory on the largest history (gaussian-mean, n≈10⁷). Either read it from a
  campaign run's logs or re-run `extract_posterior` once on a saved gaussian history. Add one
  sentence to §5/App A and, if easy, an end-to-end time-to-posterior figure (post-processing is a
  bounded fraction of the timed budget).
- **tex:174:** "so it never dominates wall-clock or memory" → "so it stays a bounded fraction of
  wall-clock and memory (measured in App~\ref{app:implementation})" (removes the absolute claim).

### T1.2 — One unambiguous weight definition (concern 3; internal §2.3)

- **tex:115, weight object (ii):** "assigned online and used only for the effective-sample-size
  diagnostic" is false — the stored weight also feeds adaptation (Eq. for W̃, `abcpmc.py:880-891`),
  parent selection, and the ESS bandwidth search (internal §2.3). Reword: "assigned online; it
  enters proposal adaptation through Eq.~\eqref{eq:steady-state-proposal} and the
  effective-sample-size diagnostic, but never the reported posterior."
- **Add a consolidated "stored fields + weight lifecycle" element:** either expand
  Alg.~\ref{alg:async-abc} or add a small table listing every stored field (θ, ρ, proposal-time
  ε-stamp, proposal-time weight) and, for each of the three weights, where it is computed and
  where it is consumed. This lets a reader trace proposal reconstruction and crash-replay without
  ambiguity (the reviewer's explicit ask).

### T1.3 — Statelessness / reproducibility / absolute wording (concern 7; internal §2.2, II.8.1)

- **tex:85:** "exactly reproducible" → "reproducible up to MPI arrival order"; add a clause that
  the online AMIS buffer and the two scheduler throttles are per-instance *performance* state,
  while the reported estimator and the archive are pure functions of history (which is what
  crash recovery requires).
- **tex:174:** "identical whether computed online or after a crash/restart" — scope to the
  *estimator*: after restart the exact per-arrival trajectory need not be reproduced (MPI order),
  only a valid one; the reported estimator is invariant to arrival order (state this, it is the
  load-bearing claim).
- The full crash-recovery *guarantee* is backed by the kill-and-resume experiment in **Tier 2**;
  Tier 1 only scopes the wording so it is defensible without it.

### T1.4 — Calibration wording (concern 6; already partly done)

- SBC paragraph (tex:267): it already says "better … not exactly nominal." Add the explicit
  residual: "the asynchronous method still under-covers slightly at the two highest levels (0.87
  vs 0.90 and 0.92 vs 0.95, several binomial standard errors at 1000 trials), so we report it as
  *substantially better calibrated than the matched baseline, with residual upper-level
  under-coverage*, not as exactly nominal."

### T1.5 — Benchmark descriptions (additional; App B)

- Expand App B with, per benchmark (gaussian-mean, g-and-k, Lotka–Volterra, Cellular Potts): the
  prior, the summary statistics, the discrepancy/normalization, how the observed dataset is
  generated (seed), and how the reference posterior/true parameters are defined. All of this is
  in the configs + `benchmarks/*.py`; it is a transcription task, not new work. Replace every
  "standard" with the actual specification.

### T1.6 — Systems configuration (additional)

- Add a short "Computing environment" paragraph (App or §5): JUWELS partition (batch, 48 physical
  cores/node, ~94 GB; mem192 for the memory-heavy reruns), interconnect, ParaStationMPI +
  `Stages/2025` toolchain, Python/pyABC 0.12.17/propulate-fork versions, `--ntasks-per-node`
  process placement. Sourced from `reference_asyncabc_cluster_deploy` + the module env. Note the
  scaling replicate counts (LV 5, CPM 3–5) already shown; add IQR/CI at every scaling point
  (fig bands already do this — state it).

### T1.7 — Related work breadth (additional; supports the novelty claim)

- Expand the related-work paragraph (tex:80) beyond paige2014cascade/murray2016anytime/pyABC/
  ABCpy to cover: asynchronous/streaming ABC, anytime Monte Carlo, adaptive importance sampling
  (AMIS lineage incl. Marin–Pudlo–Sedki), off-barrier/parallel SMC, and dynamic master–worker
  ABC. A short literature sweep + 5–10 citations. Keep the hedged novelty claim ("to our
  knowledge, the first per-arrival ABC proposal update") but make it defensible against this
  broader set.

### T1.8 — Table 1 over-generalization (additional)

- `tab:method-comparison` (tex:134-141) frames rows as "Classical ABC-SMC" universally. Reframe
  caption + intro sentence to "typical ABC-SMC/PMC implementations (e.g. the pyABC baseline used
  here)"; soften rows that are not intrinsic to ABC-SMC (smooth kernels exist; not all variants
  carry the same mutable state). Compare against the *specific* configured baseline, not the
  whole family.

### T1.9 — The one small experiment: AMIS on/off *under coupling* + drain (concern 5)

The reviewer's sharpest empirical gap, and it directly tests the ablation reframing we adopted
in the campaign (AMIS neutral on the uniform target; expected to matter under runtime–parameter
coupling). We have the infrastructure live (`parameter_bias` just ran).

- **No-AMIS arm:** add a `parameter_bias_no_amis.json` config = `parameter_bias.json` with
  `amis_snapshots: 0` (the exact mechanism the ablation "no_amis" variant uses,
  `ablation.json:45-52`). Run the same coupling sweep (48 w, 60 s, 5 reps, σ∈{0,0.5,1,2}), same
  submit recipe as parameter_bias (jobs 14107933-936; delay-throttled → packed 48/node batch,
  4 shards, ~10 min/shard). Small compute (~a few node-hours).
- **Report more than the mean:** for AMIS vs no-AMIS across σ, report bias in the posterior
  **mean, variance, and quantiles**, and coverage — not just posterior-mean error. Add these
  columns to the analysis (`make_param_bias_fig.py` / the analytic summary).
- **Drain-after-deadline variant (the decisive censoring test):** re-run one coupling setting
  where, after the wall-clock deadline, all in-flight simulations are allowed to finish and the
  posterior is recomputed. If the posterior moves, that difference *is* the deadline-censoring
  effect (informative censoring the AMIS denominator does not correct). This needs a small runner
  flag ("drain in-flight before finalize"); scope it as a targeted addition to the
  heterogeneity/param-bias runner.
- **Text:** rewrite the param-bias/ablation/Limitation-(iv) discussion from the results. If AMIS
  helps under coupling → the reframed claim is confirmed with the *right* experiment. If it does
  not → state that plainly and lean on "raw archive robustness at these coupling levels," and
  keep the censoring caveat. Either way this replaces the current unsupported "weights cancel it"
  lineage with a measured statement.

---

## Order of operations (Tier 0 + Tier 1)

1. **Text-only, no compute (can all land first):** T0.1–T0.5, T1.1 (except the measurement),
   T1.2, T1.3, T1.4, T1.5, T1.6, T1.7, T1.8. Recompile after each cluster of edits; keep the
   abstract/body consistent.
2. **One measurement:** T1.1 post-processing time/memory on a gaussian history (read from logs or
   one local `extract_posterior` run) → fold the number into App A + §5.
3. **One experiment (parallelizable with step 1):** T1.9 no-AMIS-under-coupling sweep + the drain
   variant → new analysis columns → rewrite param-bias/ablation/Limitation (iv) from the data.
4. **Verify:** full test suite green (`pytest experiments/tests -q`); recompile
   (`pdflatex → bibtex → pdflatex ×2`) with no undefined refs / overfull boxes; grep for removed
   absolute wording ("exactly", "bit-identical" reason, "never dominates", "PMC lacks CLTs");
   claim re-check that the body now matches the abstract's honest framing.

## Tier 2 / Tier 3 preview (separate plan after the positioning call)

- **Tier 2 (moderate compute, defends the core causal claim):**
  - **Barrierized twin (concern 4):** a batched version of *our* algorithm (identical proposals,
    deterministic kernel weights, archive, estimator; updates only after batches of N) as the
    principal systems control, so the systems delta is attributable to the barrier alone. Likely a
    propagator/runner variant → frozen-commit change → re-validate + targeted reruns. Highest-value
    new experiment; biggest cost.
  - **Kill-and-resume experiment (concern 7):** actual crash mid-run + resume; show the reported
    estimator matches. Backs the crash-recoverable claim empirically.
  - **Multidim / multimodal SBC (concern 6):** at least one nonlinear multidimensional and one
    multimodal target (top-k archive is vulnerable to mode loss); add ESS / max-normalized-weight
    / weight-tail diagnostics.
- **Tier 3 (hard / venue-dependent):**
  - A complete theorem with explicit CLT rate and asymptotic variance for the *actual* algorithm,
    formalized with the defensive prior mixture δπ+(1−δ)q_n (concern 1). Only if targeting a
    theory venue; otherwise Tier 0's honest scoping stands.

## Open decisions blocking Tier 2/3 (need author input)

1. **Positioning / target venue:** systems-first (Tier 0 scoping suffices) vs theory-heavy
   (attempt the Tier 3 theorem).
2. **Appetite for the barrierized twin** (reopens the campaign + frozen commit) vs weakening the
   "synchronization-only" causal language in text.
3. **Scope of added validation:** multimodal/multidim SBC + kill-resume now, or defer.
