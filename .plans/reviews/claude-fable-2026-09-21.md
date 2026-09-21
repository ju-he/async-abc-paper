# Referee report — Statistics and Computing

**Manuscript:** *Asynchronous, Generation-Free ABC: Streaming AMIS Reweighting for Heterogeneous Simulator Workloads on HPC* (58 pp. incl. appendices; source `latex/sn-article-template/sn-article.tex`, compiled 2026-09-21)

**Referee:** independent (no prior contact with this manuscript). Line numbers below refer to `sn-article.tex`. Where I checked a number against the vendored data I say so; the files are under `experiments/data/paper_figures/`, `experiments/data/diagnostics/` and the generator scripts under `experiments/scripts/`. The compile log is clean (no undefined references or citations; all 25 cited keys resolve in the `.bib`).

---

## 1. Summary

The paper proposes an ABC sampler with no generations: the proposal mixture is rebuilt after every arriving evaluation from a sliding top-*k* archive, every evaluated particle is weighted against a draw-proportional mixture of past proposals (a streaming form of AMIS with a defensive prior floor), and the reported posterior is a retroactive estimator recomputed from the evaluated history at the tightest bandwidth reached. The sampler is implemented as a propagator in Propulate's barrier-free island model over MPI. The authors claim (C1) that a generation barrier costs the *straggler factor* E[max_W T]/μ of the workload and that this cost is predictable from the asynchronous run's own timing, verified against a "barrierized twin" of their own sampler (within 1% for an injected persistent straggler, "a few per cent" on a Cellular Potts tissue simulator, over 1.2×–400×); (C2) that on a matched-kernel pyABC baseline the extra simulations convert into a tighter tolerance (2.4× the simulations, 1.6× per simulation, 4.1× tighter at equal wall clock on Cellular Potts 50³); (C3) that the advantage switches on at a few milliseconds per simulation and is lost to per-arrival coordination below that; and (C4) that the reported estimator is consistent (Theorem 1, to a tilted target π^r), calibrated by SBC in 1-D, on a bimodal target and (conservatively) in 4-D, recovers a two-parameter Cellular Potts posterior with the truth covered in every replicate, and is limited by the starting bandwidth and by an effective-sample-size ceiling of a few multiples of *k*. A CLT for damped-adaptation variants and a decomposition of the fidelity ratio *r* are given in appendices, with the candid finding that the CLT's rate condition is measurably violated on the Cellular Potts run.

---

## 2. Major issues

### M1. The headline "within 1% / a few per cent" for C1 is assembled from three different instruments and quotes only the easy cases

The abstract (l. 42), contribution 2 (l. 57) and the conclusion (l. 503) state that the barrier's cost, "predicted from the asynchronous run's timing alone", matches "a barrierized twin of our sampler within 1% under an injected straggler and a few per cent on a Cellular Potts tissue simulator, from 1.2× to 400×". Checking this against `fig_predictor/predictor_rows.csv` and `make_predictor_fig.py`:

* **The persistent-straggler agreement is nearly tautological.** The slow worker carries a deterministic post-evaluation delay of 0.1 s × {5, 10, 20}; the "prediction" is 16/(2.000 s + 3.6 ms) (l. 288). With a deterministic 2 s delay on one of 16 ranks and a barrier every 16 evaluations, the twin's throughput *must* be 16/2.004. This verifies that the twin was implemented correctly; it says nothing about the predictor's power on a random workload. It is nonetheless the only case quoted "within 1%".
* **The lognormal-heterogeneity points are not predicted "from the asynchronous run's timing alone".** Table 4's caption (l. 323) says "*Predicted* is from the injected law", and l. 288 explains why: a prediction from the truncated asynchronous sample "undershoots by 2–4× at the two largest spreads". So on the one workload where the runtime law is random and realistic, the predictor as advertised fails, and the number reported comes from the known injected law instead. Even so, predicted/measured is 0.84 at σ = 1.5 and 1.19 at σ = 2 (Table 4: 14.0 vs 16.6; 35.4 vs 29.6). These 16–20% misses never reach the abstract, introduction or conclusion, which mention only the straggler and the tissue simulator.
* **On Cellular Potts 50³ the twin was 1.9–2.8× slower than the asynchronous arm, and the paper attributes only 1.2× of that to the barrier** (Table 5, l. 341). The remaining 1.5–2.3× is the twin's simulations "taking longer … which a barrier cannot cause … we do not attribute it, and we do not count it." This factorisation is post hoc and the exclusion is not justified: the candidate mechanisms the authors themselves name — "synchronised I/O bursts when *W* simulations start together" — are *consequences of the barrier* (barrier-free execution staggers starts). The vendored `tab_twin_cpm/twin_cpm_decomposition.csv` makes this worse, not better: at 48 workers the twin's per-replicate mean simulation duration is 23.9, 19.5, 8.3, 6.8, 6.7 s against 5.1–5.2 s on every asynchronous replicate, with twin duration CVs of 0.26–1.02 against 0.09–0.11, while the *within-generation* straggler factor stays at 1.19–1.24. A large overall CV with a small within-generation straggler factor means the slowdown is correlated across a whole generation — exactly what synchronised starts would produce. If that is what is happening, the barrier's true cost on this workload is the measured 1.9×, not 1.2×, and the "prediction" is off by 40–60%. The authors must either explain the duration inflation (e.g. re-run the twin with staggered starts, or instrument I/O) or stop claiming the tissue simulator as a validation of the predictor.
* **The 80³ point is not a twin measurement at all.** The predictor figure's caption (l. 285) and the script docstring say the 80³ point is "against the matched pyABC baseline"; `predictor_rows.csv` carries it as a single hand-entered row (`T_async` = NaN, measured 2.01, predicted 1.90). Yet the abstract, l. 57 ("a real tissue simulator at two sizes") and l. 280 ("For every configuration on which we ran the barrierized twin — … the Cellular Potts simulator at four worker counts and two sizes") present it as a twin measurement. It is also 5–10% off (1.90 predicted vs 2.01 utilisation ratio / 2.1 throughput ratio in the text, l. 343), not "a few per cent".
* **The count of excluded points is wrong.** l. 280–281: "Twelve configurations lie inside the model's stated domain … Two configurations lie outside the domain and are drawn open". There are 15 configurations (5 straggler + 5 heterogeneity + 4 CPM 50³ + 1 CPM 80³); `make_predictor_fig.py` defines `OUTSIDE = {("straggler", 0.0), ("straggler", 1.0), ("cpm50", 384.0)}` — three — and the figure indeed shows three open markers. The straggler 1× point (predicted/measured = 0.42) is silently dropped from the prose. The 12-point range 0.84–1.19 and median 0.996 do check out, but only after excluding three of fifteen points, one of them without saying so.

Taken together: the claim that is actually supported is "the straggler factor computed from the injected or measured per-evaluation law predicts the twin's slowdown to within ±20% on the synthetic workloads, and to within ~10% of the *utilisation* ratio on the tissue simulator after discarding an unexplained 1.5–2.3× duration effect". That is a useful result, but it is not what the abstract says.

### M2. The C2 headline numbers (2.4× / 4.1×) are, by the paper's own analysis, mostly not the barrier's

l. 343: "At 50³ the same comparison against pyABC gives 2.3×, of which only 1.2× is the barrier: the rest is pyABC's fixed per-generation overhead … We therefore quote 2.0× for the Cellular Potts systems gain, because it is the number attributable to the mechanism the paper is about." The abstract (l. 42), contribution 3 (l. 58), §7.2 (l. 396) and the conclusion (l. 503) nonetheless headline 2.4× simulations and 4.1× tolerance at equal wall clock — the numbers that include ~2× of pyABC implementation overhead that "a barrier cannot cause". The "2.0×" appears once and is never used. Either the headline is the barrier-attributable gain (then use the twin, or the 80³ run, and say so) or it is a comparison of two software packages (then say that, and do not present it under "Removing [the barrier] converts into a tighter tolerance").

Two further points weaken C2 on the flagship benchmark:

* **The per-simulation advantage on Cellular Potts is shrinking with *n*.** Fig. 6(d)'s legend gives the last-decade slopes as *n*^−2.21 for the synchronous baseline and *n*^−1.44 for the asynchronous arm; the synchronous curve visibly closes on the asynchronous one at the right end of the panel. Extrapolating, the 1.6× per-simulation ratio at *n* = 5,400 would reach parity at roughly *n* ≈ 10,000. The text (l. 371) mentions only that both adaptive samplers steepen beyond rejection ABC's rate; it does not mention that the baseline steepens faster. "At least as efficient per simulation" is therefore a statement about one budget.
* **No uncertainty is attached to any ratio in Table 6.** The g-and-k per-simulation ratio of 1.06× is used to support "at least the baseline's on every benchmark but the one-dimensional analytic one" (l. 58); with five replicates and no replicate range on the ratio, 1.06 is not distinguishable from 1.

### M3. The matched synchronous baseline is handicapped in ways the paper documents but does not correct for

The pyABC baseline is root-driven (rank 0 does not simulate; l. 786), uses a population of 100 on 47 simulating ranks for the main C2/C3 comparisons (Table 10), and — as the authors note — carries a fixed per-generation overhead that is a large fraction of a 13 s evaluation. A population of 100 on 47 workers forces ≥3 rounds per generation with the last round 13% full, so the baseline idles by construction even on a homogeneous simulator (48% utilisation at σ = 0 in Table 4). The scaling study fixes this with population = *W* (only above 100 workers); the headline comparisons do not. The closest related method, look-ahead scheduling (Alamoudi et al. 2024, by the pyABC authors), is discussed at length (l. 92–94) but not run, although it ships in pyABC. A referee for this journal will expect at least one of: (i) C2 reported against the barrierized twin (the clean instrument the authors built), (ii) the pyABC baseline with a population that is a multiple of the worker count, or (iii) a look-ahead pyABC run on at least one benchmark.

### M4. The reported estimator collapses under batched proposals, and the paper under-reports it

Table 3's lower block shows the twin's *reported* posterior at *W*₁ = 0.74 (10×) and 1.73 (20×) against 0.008–0.070 for the asynchronous re-run at the same evaluation counts and seeds; the paper frames this as "What the barrier does to the estimator" (l. 367). But the twin is, by the authors' construction, "identical in everything but a collective barrier"; a barrier does not change the evaluated history's validity. What collapses is therefore the *estimator* when proposals arrive in batches from a frozen archive. The vendored `tab_twin_hetero/twin_hetero_raw.csv` shows this is not confined to the straggler case: the twin's `full_history_w1_to_analytic` is 0.15–0.23 at σ = 0 (no heterogeneity at all; 21,120 evaluations, *W* = 48), 0.8–1.5 at σ = 0.5 and 2.0–2.4 at σ ≥ 1, while its top-*k* archive stays at 0.01–0.03 throughout. None of this appears in the manuscript. It matters for three reasons: (a) it is direct evidence about the fidelity ratio *r* — under a piecewise-constant proposal path the *m* ≤ 21 snapshot quadrature (factor (b) of §A.2) can miss most of the path's variation, which is exactly the mechanism the theory says to check; (b) the "ESS" of these collapsed posteriors is high (1,300–4,000 of 21,120), so the effective-sample-size diagnostic the paper relies on does not detect the failure; (c) it bounds where the estimator can be used (any batched or partially synchronous deployment). The paragraph at l. 367 should be rewritten as an estimator-robustness result, report the heterogeneity-twin posteriors, and connect them to §A.2(b).

### M5. Internal inconsistencies about which Cellular Potts configuration is which

* Fig. 5's caption (l. 364) and Table 2 give the 50³ evaluation cost as 13 s (production run: 12.9–13.4 s in `cpm_production_replicates.csv`). The twin/scaling campaign that supplies every 50³ point of the predictor figure and Table 5 has asynchronous simulations of **5.1 s** (l. 341, and `twin_cpm_decomposition.csv`), with CV 0.10 rather than the 0.05 of Table 2. Two different Cellular Potts configurations are both called "Cellular Potts 50³"; the scaling caption's "13 s per evaluation" is wrong for that campaign; and the reader cannot tell whether the C1 measurements were made on the two-parameter, four-replicate problem the rest of the paper describes.
* l. 591 and Table 9 refer to "the Cellular Potts production history (33,748 asynchronous evaluations across 48 workers …)" with ESS fraction 0.213 and 832 archive changes. The production run of Table 8 has 12,900 evaluations and ESS 352 (fraction 0.027). These cannot be the same run. All of §A.2–A.3's measurements of *r* and of the stabilisation rate are on an unidentified history.
* l. 343 quotes the 50³ throughput ratio against pyABC as 2.3×; Table 6 and the abstract say 2.4× (2.39 in `crossover.csv`); the scaling campaign gives 2.43.

### M6. The theory is claimed for "the estimator this paper reports, computed by the algorithm this paper runs" (l. 170), but its hypotheses are not shown to hold for those runs

* Assumption 5(i) — the fidelity condition ‖q̄*_n/q̄_n − r‖_∞ → 0 — is the whole content of the sampler's contribution and is assumed, not derived from the algorithm. Two of its four factors are estimated on one history each; the other two are "bounded, not measured"; and under asynchrony factor (d) additionally carries an un-modelled scheduling effect (l. 589). The paper's own Appendix says (l. 538) that for a fixed proposal the tilted-limit statement "would not merit a theorem". What is genuinely new is the uniform martingale SLLN over a compact parameter class, which is standard technique. Calling this "We prove consistency of the reported estimator" in the abstract, without the qualifier "to a tilted target under an unverified fidelity assumption", overstates it. The Limitations section is honest; the abstract and conclusion are not.
* Assumption 5(i) also needs ε_n ↓ ε_∞ > 0. Table 9 records that the reported runs "leave `min_tol` unset (zero is accepted), so neither part is *established*". So even the consistency theorem's premises are not verified on the reported runs.
* l. 613: the implementation's post-hoc estimator consumes "a rank's *arrival*-ordered log", the proofs index by *proposal* time, and "we do not prove that the theorems transfer between the two". Different ranks therefore hold different histories of the same run and could report different posteriors. This contradicts §3.1's framing (l. 99: "the estimator is identical whether it is formed online or afterwards") and the "pure function of the evaluated history" claim in the abstract, neither of which carries the caveat.
* §2.3 (l. 87) states "AMIS is consistent and satisfies a CLT under regularity conditions" citing Cornuet et al.; Appendix A (l. 538) states that Cornuet et al.'s "convergence argument is heuristic". One of these must go.

### M7. The ablation and sensitivity studies use a metric the paper itself disowns

§F.6 (l. 942): the Wasserstein distance to a point mass at the truth "is floored at the posterior's own spread — on the Gaussian configuration an exactly correct posterior scores 0.1√(2/π) = 0.080 … and it ranks a collapsed archive above a correct one. We retain it only as a concentration diagnostic and report none of its values as posterior quality". Then §F.7 reports "the full method attains a final Wasserstein-to-truth of 0.073 (CI [0.067, 0.079]); the hard-indicator kernel raises it to 0.076 … dropping AMIS leaves it essentially unchanged (0.072 …)" and §F.8 measures "posterior quality (Wasserstein distance to the true μ, lower is better)" with "no failure region". Every ablation value lies *below* the 0.080 floor of a correct posterior, i.e. in the regime where the metric rewards over-concentration, so the ordering among variants carries no information about quality (and "slow decay" at 0.069 is the "best" precisely because it is the most collapsed). Either re-score both studies against the analytic posterior (the *W*₁ already used in Fig. 8 and Table 3) or drop them.

### M8. "Stores nothing between calls" versus a carried snapshot buffer

§3.4 (l. 129): "the buffer 𝒮 is itself a deterministic function of ℋ_n, so the propagator stores nothing between calls"; Table 1: "Archive state: none; reconstructed per call". §3.1 (l. 101): "the propagator does carry state between calls — the online AMIS snapshot buffer enters the denominator of every candidate it proposes"; Appendix D (l. 790): "the online AMIS snapshot buffer is rebuilt empty on restart … the post-restart proposal sequence is drawn from a different … adaptation trajectory". Algorithm 1 itself says "periodically snapshot q_n into 𝒮". These are contradictory statements about the paper's central design property. The honest version (state is carried in the sampler; only the *reported estimator* is history-reconstructed) should be the one stated everywhere, including Table 1.

### M9. C4 is delivered with caveats that the abstract omits

* In 4-D (g-and-k) the reported estimator is *worse* than the baseline: *W*₁ 0.069 vs 0.029 at equal wall clock (l. 438), with ESS "7 to 2987 … across replicates", and it over-covers (0.62/0.91/0.96/0.99 for *A*). The abstract's "(iv) … verify its calibration" and the conclusion's "consistent, calibrated and replayable" do not mention that on the only multi-dimensional benchmark with a reference posterior the method is less accurate than the comparator and not calibrated.
* Table 7 reports the synchronous baseline's SBC only on the 1-D Gaussian, where it under-covers badly (0.35/0.63/0.75/0.83). The vendored diagnostics (`sbc_extra_sbc1000_20260729.md`) contain the baseline's g-and-k and bimodal coverages — near-nominal for *A* (0.54/0.82/0.91/0.95) and *g*, and 0.45/0.82/0.90/0.95 on the bimodal target. Omitting the rows where the baseline looks good, while keeping the one where it looks bad, is selective reporting.
* "similarly for the others" / "the other three behave alike" (l. 420, Table 7 caption): the diagnostics give *B* = 0.538/0.830/0.914/0.960 and *k* = 0.563/0.833/0.922/0.957 — close to nominal, not conservative like *A*.
* On the 80³ run, rejection ABC with 1,000 prior draws contracts 78%/53% against 50%/15% for the asynchronous estimator at ~1,000 evaluations (l. 477). This is the regime "the method is for" (l. 473), and the adaptive sampler is beaten by rejection on contraction; the text does not discuss it, nor report coverage there (the vendored table shows rejection *misses* the truth on the division rate at 80³ — `covered_division_rate = False` — which would help the authors' case and is also unreported).

### M10. The per-simulation-efficiency "monotonicity in cost" (C3) has no mechanism and rests on four confounded points

l. 408: "Both [ratios] are monotone in that cost". The throughput ratio plausibly is, since the per-arrival coordination cost (~3.6–8 ms, `predictor_rows.csv` notes) is a fixed overhead. But the *per-simulation* ratio (0.78, 1.06, 1.30, 1.6) is a statistical property of the sampler on four different inference problems (dimensions 1, 4, 4, 2; different summaries; one with 98% extinction). Nothing in the paper explains why per-draw efficiency should depend on how long a simulation takes, and with four heterogeneous benchmarks the ordering is as consistent with "depends on the problem" as with "rises with cost". The crossover figure and Discussion rule should be restricted to the throughput ratio, and the worker-count dependence of the boundary (l. 410: the baseline overtakes beyond three nodes even at 4 ms) should be part of the rule.

### M11. Numerical contradictions that a reader will trip over

(Details in §4.) The most consequential:

* "Enlarging the archive tenfold moves that crossover in to a single node" (l. 410; repeated l. 854 and Table 12 caption) is contradicted by Table 14 ("1000 … 3 nodes") and by Table 12 itself (k = 1000 at 48 workers: 3804 sims/s vs the baseline's 2952).
* Table 8's caption calls ε₀ = 10 "a thousand times the prior-predictive median discrepancy"; l. 473 gives that median as 0.46, so ε₀ = 10 is ~22× it (and 100× the production value). The Discussion (l. 486) repeats "a thousand times too large".
* Fig. 2's caption and l. 288 say the asynchronous arm "holds ≈3200–3900"; Table 3 has 3112 at 1× and the figure's own data (`straggler_throughput.csv`) has 2645 at 1× — the figure and the table come from different campaigns with different asynchronous throughputs (2645 vs 3112 at 1×; 3734 vs 3241 at 5×), which the paper does not say.

---

## 3. Minor issues

**Clarity and terminology**

1. "Tolerance", "bandwidth" and ε are used interchangeably; "archive size" and "population" likewise; "simulation" and "evaluation" are conflated on Cellular Potts (Table 2 says an evaluation is four simulations, then Table 6/8 count "sims"; the per-unit cost is 13 s in one campaign and 5.1 s in another — see M5).
2. The heterogeneity study's 1 s base evaluation time is never stated in the design (§6.5 says only "multiplied by a lognormal factor"; a 0.1 ms simulation times a lognormal factor would not take 1 s); it surfaces at l. 320.
3. `amis_interval` (l. 122) is never given a value. The online ring buffer (size *S*, sampled every `amis_interval` calls) and the retroactive draw-proportional snapshot set are different mixtures; say so at first mention.
4. §3.2 describes the schedule as "at most one halving per search and one search per *k* arrivals"; Appendix C describes three schedulers (quantile, geometric, acceptance-rate) in detail and then says that under a smooth kernel all of them are replaced by ESS-retention bisection. Since every reported run uses a smooth kernel, the three rules are irrelevant and the one actually used (bisection target 0.95, halving cap, search interval = *k*) is described in one sentence. Invert the emphasis.
5. l. 213 says "no *deterministic* bandwidth floor is needed" while Table 9 says a positive limit is not established; the row's status label "not satisfied" contradicts its own text ("neither part is *established* — not that either is disproved").
6. §3.3 gives Σ_n = s·Cov; Appendix C gives Σ_n = s[Cov + λ_n I]. Use one.
7. The predictor caption (l. 285) speaks of "a 4 ms simulator with no straggler"; the Gaussian mean costs 0.1 ms per simulation, 4 ms is the per-*evaluation* time including overhead.
8. Table 10 lists *k* = 200 for the g-and-k SBC rows while the text says "*k* = 100 throughout" and the Discussion recommends *k* = 100; the SBC text never mentions *k* = 200.
9. l. 420: "doubling the simulation budget leaves it unchanged" — in the vendored diagnostics the doubled-budget run (`sbc_gandk_2x`, 500 trials) is of the *archive* estimator (0.848/0.890 vs 0.847/0.890), not the reported full-history one. Say which.
10. l. 420: the buffer "saturating at *S* = 5 in every dimension" — Table 13 (right) has *S* = 20 markedly worse than *S* = 5 at *d* = 8 (+0.055 vs +0.006) and *d* = 16 (+0.028 vs +0.010). The default *S* = 20 is not supported as better than 5 by this sweep.
11. Table 3's caption: "the last two rows differ only in whether workers wait" — the last two rows are two twin granularities; the intended contrast is the asynchronous re-run row against the twin rows.
12. l. 288: "the three granularities agree to three significant figures" from 5× up; at 5× they read 31.6 / 31.8 / 31.8.
13. Table 8's caption says "medians otherwise", but the ESS column reports means (760, 352 are the means; the medians are 645, 303).
14. l. 438: "7 to 2987 at the same wall clock" — 7 is at 150 s and 2987 at 600 s in `reported_recovery_per_replicate.csv`; at 600 s the range is 44–2987.
15. l. 475: "moves the cell-volume contraction monotonically from 85% to 78%" — `rereport.csv` gives 85.3, 84.2, 84.7, 82.4, 77.8: the endpoints are right, the path is not monotone.
16. l. 396: "doubling the asynchronous arm's 481 simulations to 988 tightened its tolerance 7.7×" — 7.7 is replicate 0 only; the median is 8.4.
17. l. 341: the utilisation ratio "equals the straggler factor of the twin's own generations (1.21–1.29)" — 1.18–1.21 vs 1.21–1.29 is agreement to ~7%, not equality.
18. §7.3's "rule" (parity at ~4 ms) and the Discussion's "Use it above a few milliseconds" omit the worker count, which the same subsection shows decides the boundary on Lotka–Volterra.
19. Appendix D's instrumented Lotka–Volterra run uses 128 and 256 workers ("3 nodes", "6 nodes"), which are not fully packed (144, 288), contrary to the "fully packed" rule stated for the scaling runs.
20. l. 782: "the 3 × 10⁷-particle Gaussian history" — no described run produces a history that long (the asynchronous Gaussian arm reaches 1.3 M, the synchronous 4.6 M).
21. The straggler twin is faster at 1× (65.3) than at 0× (48.7) with a barrier every *W* (Table 3); unexplained.
22. The Gaussian-mean recovery curve (Fig. 8a) *rises* from ~0.009 to 0.016 over the last third of the run while the history grows; the text ("sits at 0.008–0.016 … at every checkpoint") reads this as flat. It looks like the reporting rule trading bandwidth against ESS, and deserves a sentence.
23. Data availability: "35 CSV files" — there are 37 under `experiments/data/paper_figures/`.
24. The kill-and-resume numbers (l. 790) have no vendored data and could not be checked.
25. The rejection-ABC rate "*n*⁻¹" on Cellular Potts (l. 371) is asserted without a rejection curve in Fig. 6; for a 2-D discrepancy the *k*-th order statistic scales like *n*^{−1/2} unless the noise floor dominates — say which regime is meant.

**Figures and tables**

26. Fig. 1: three open markers, caption says two (M1).
27. Fig. 5(b): an unexplained second marker ("×") at 192 and 384 workers for the baseline; the caption does not say what it is.
28. Fig. 6(c): the caption says the extinction plateau "is cut off", but the spikes to 10³ for *n* < 5000 are drawn; either cut it off or describe it.
29. Fig. 8(b) shows the asynchronous *W*₁ spiking to 0.3 at 150 s with ESS ≈ 7; the text mentions "intermittent weight degeneracy" only in passing. A run-to-run failure of this size in 4-D deserves its own diagnosis.
30. Table 2 lists a single 300 s budget for the Gaussian mean; the heterogeneity and parameter-coupled studies use 60 s.
31. Table 14's "worst |Δcov|" for *k* = 50 is 0.250 (consistent with Table 13) while the vendored `kfrontier_summary.csv` says 0.271; the two files disagree.

**Style and length**

32. The main text runs to 28 pages and the appendices to 30. Much of the prose is argumentative rather than expository ("that is a quantity, not a slogan", "the honest caveat", "we state it here rather than leave it to be discovered"). Statistics and Computing readers will want the results stated once, plainly, with their uncertainty; a good deal of the rhetorical framing can go, and with it perhaps a quarter of the length.
33. The theory section (§4) reads as a commentary on its own assumptions; the actual theorem and corollary occupy six lines. Consider moving the "one channel" discussion to the appendix and stating the theorem with its assumptions compactly.

---

## 4. Numbers-consistency audit

Each row: the claim, the two (or more) locations, whether they agree, and the source I checked where applicable.

| # | Quantity | Location A | Location B | Verdict |
|---|---|---|---|---|
| 1 | CPM 50³ simulations ratio | Abstract l. 42, l. 58, l. 396, l. 503: 2.4× | Table 6: 2.39×; `crossover.csv` 2.388 | agree |
| 2 | Same, as quoted in C1 | l. 343: "gives 2.3×" | Table 6 2.39×; scaling campaign 2.43 | **disagree** (2.3 vs 2.4) |
| 3 | "We therefore quote 2.0×" | l. 343 | never used elsewhere; abstract uses 2.4× | **inconsistent framing** |
| 4 | CPM per-simulation ratio | Abstract 1.6× | Table 6 1.6×; CSV 1.5999 | agree |
| 5 | CPM equal-wall-clock ratio | Abstract 4.1× | Table 6 4.09×; CSV 4.088; l. 396 (7.6e-5 vs 3.1e-4) | agree |
| 6 | Straggler prediction | Abstract "within 1%" | Table 3: 102/103, 202/202, 401/402; CSV 0.995–0.999 | agree (but see M1) |
| 7 | CPM prediction accuracy | Abstract/l. 57 "a few per cent … at two sizes" | Table 5: pred 1.26–1.32 vs meas 1.18–1.21 (7–9%); 80³ 1.90 vs 2.01/2.1 (5–10%) | **understated** |
| 8 | Heterogeneity prediction | Not in abstract/intro/conclusion | Table 4: 14.0 vs 16.6 (−16%), 35.4 vs 29.6 (+19%) | **omitted from headline** |
| 9 | In-domain predictor stats | l. 280: 12 points, 0.84–1.19, median 0.996 | `predictor_rows.csv` medians: 12 in-domain, 0.840–1.194, median 0.996 | agree |
| 10 | Excluded predictor points | l. 281 and Fig. 1 caption: "two" | `make_predictor_fig.py` `OUTSIDE` has 3; figure shows 3 open markers | **disagree** |
| 11 | 80³ predictor point | l. 280 "configurations on which we ran the twin … two sizes"; abstract "twin" | Fig. 1 caption and script: against pyABC; `predictor_rows.csv` single hand-entered row | **disagree** |
| 12 | Range of measured costs | Abstract "1.2× to 400×"; conclusion "two and a half orders" | Table 5 1.18–1.21; Table 3 402 | agree |
| 13 | Straggler async throughput | l. 288 and Fig. 2 caption "≈3200–3900" | Table 3: 3112 at 1×; `straggler_throughput.csv`: 2645 at 1× | **disagree** (two campaigns) |
| 14 | Twin throughputs, straggler | l. 288: 48.7→8.0, 163→8.0 | Table 3: 48.7/8.0, 162.8/8.0; twin raw CSV | agree |
| 15 | 16/2.004 = 7.98 | l. 288 | arithmetic | agree |
| 16 | "buys 3.3× at the control" | l. 288 | 162.8/48.7 = 3.34 | agree |
| 17 | Heterogeneity table entries | Table 4 | `predictor_rows.csv`, `twin_hetero_raw.csv` medians (47.6/42.1/28.9/17.6/11.4; 44.8/15.1/4.37/1.06/0.37; 45.1/19.8/6.50/1.75/0.45; 23.1/18.9/…; ratios) | agree |
| 18 | E[max₄₈] lognormal σ=2 | l. 320: 147 s vs mean 7.4 | e² = 7.39; 48/(11.4/35.4) = 149 s | agree |
| 19 | Twin CPM decomposition | Table 5 | `twin_cpm_decomposition.csv` (ratios 1.91/1.87/1.86/2.82; util 1.18–1.21; straggler factor 1.21/1.28/1.21/1.29; durations 6.7–36.2 s vs 5.08–5.33 s) | agree |
| 20 | CPM 50³ per-evaluation cost | Table 2, Fig. 5 caption: 13 s; production CSV 12.9–13.4 s | l. 341 and twin/scaling CSV: 5.1 s | **disagree** (two configurations) |
| 21 | CPM 50³ runtime CV | Table 2: 0.05; production CSV 0.045–0.054 | l. 341 "0.10 on those runs"; scaling CSV 0.09–0.11 | agree only if two configurations are acknowledged |
| 22 | Parallel efficiency 1→8 nodes | l. 343: 97% | 71.98/(8×9.33) = 0.96 | agree |
| 23 | Table 8 entries | Table 8 | `cpm_production_replicates.csv` (13,300/12,900/5,400; 99.8/42.8%; ε; 84±6, 13±14, 92±1, 81±1, 91±1, 72±4, 91, 67) | agree; ESS 760/352 are means, caption says medians |
| 24 | ε₀ = 10 control ratio | l. 396: 7.1× (4.5→3.1e-4 vs 6.3→7.6e-5) | CSV medians 4.46e-4/6.31e-5 = 7.07 | agree |
| 25 | 80³ ratios | l. 396: 2.1×, 2.1×, 17× | CSV medians 990/466 = 2.12; 0.0503/0.0244 = 2.06; 0.0495/0.00292 = 16.9 | agree |
| 26 | 80³ "481 → 988 tightened 7.7×" | l. 396 | replicate 0: 7.71; median 8.4 | partial |
| 27 | 80³ utilisation/throughput vs prediction | l. 343: 98% vs 49%, 2.1×, pred 1.90 | CSV util 0.975/0.484; predictor row measured 2.01 (utilisation ratio) | agree; figure and text use different "measured" definitions |
| 28 | 80³ contraction | l. 477: 50/15, 52/11, 78/53 | CSV means 0.500/0.145, 0.517/0.106, 0.777/0.529 | agree; rejection `covered_division_rate=False` unreported |
| 29 | ε₀ = 10 vs prior-predictive median | Table 8 caption, l. 486: "a thousand times" | l. 473: median 0.46 → 22× | **disagree** |
| 30 | Transient ratios 321× and 2× | l. 473 | `tol_init.csv` medians 320.7, 2.03 | agree |
| 31 | Re-report k = 10…1000: 85%→78% monotone | l. 475 | `rereport.csv` 85.3/84.2/84.7/82.4/77.8 | endpoints agree; not monotone |
| 32 | k = 300 never left ε₀, 7% | l. 475 | `k.csv` eps_reported 0.1; 0.062/0.069 | agree |
| 33 | k = 30: one of two replicates misses truth | l. 475 | `k.csv` covered False (rep 0) | agree |
| 34 | 93%/85% on replicate 0 | l. 473 | `rereport.csv` k = 100: 0.929/0.847 | agree |
| 35 | Table 6 all entries | Table 6 | `crossover.csv` | agree |
| 36 | CPM slopes | l. 371 "both adaptive samplers steepen" | Fig. 6(d) legend: sync −2.21, async −1.44 | agree, but the sync-steeper-than-async fact is omitted (M2) |
| 37 | SBC table entries | Table 7 | `sbc_extra_sbc1000_20260729.md`, `sbc_extra_concern6_20260718.md` (`sbc_gandk` archive 0.847/0.890) | agree |
| 38 | "similarly for the others" | l. 420, Table 7 caption | *B* 0.538/0.830/0.914/0.960; *k* 0.563/0.833/0.922/0.957 | **disagree** |
| 39 | Baseline SBC on g-and-k/bimodal | absent from Table 7 | diagnostics: *A* 0.540/0.822/0.912/0.954; bimodal 0.452/0.820/0.901/0.947 | **omitted** |
| 40 | Both modes retained 86% | l. 420 | diagnostics 0.862 | agree |
| 41 | SBC g-and-k archive size | text: k = 100 throughout | Table 10: k = 200 | **not stated in text** |
| 42 | Recovery numbers | l. 438: 0.008–0.016; 0.019–0.028; 0.011; 0.029/0.069/0.47 | `reported_recovery.csv` | agree |
| 43 | Gaussian ESS "237–303" | l. 438 | CSV medians 238–296 | approx. agree |
| 44 | "7 to 2987 at the same wall clock" | l. 438 | per-replicate: 7 at 150 s, 2987 at 600 s | **disagree** (different checkpoints) |
| 45 | Scaling table | Table 12 | `scaling_combined.csv`, `kfrontier_summary.csv` | agree |
| 46 | k = 1000 crossover | l. 410, l. 854, Table 12 caption: "single node" | Table 14: 3 nodes; data: 3804 > 2952 at 48 workers | **disagree** |
| 47 | k-frontier throughput/percentages | Table 14 | `kfrontier_summary.csv` | agree |
| 48 | k-frontier worst |Δcov| for k = 50 | Table 14: 0.250 | `kfrontier_summary.csv`: 0.271 | disagree (table consistent with Table 13) |
| 49 | k/S sweep entries (spot-checked 8 cells) | Table 13 | `ksweep_summary.csv` | agree |
| 50 | d = 8, k = 50 coverage 0.29 at 0.50 | l. 961 | CSV 0.289 | agree |
| 51 | Buffer "saturating at S = 5" | l. 420 | Table 13: d = 8 S = 5 +0.006 vs S = 20 +0.055 | loose |
| 52 | LV timing 55/25/8/2% | l. 410, Fig. 10 caption | `lv_timing.csv` | agree |
| 53 | Hetero quality/idle captions | Figs. 7, 9 | `quality.csv`, `sims.csv`, `idle.csv` | agree |
| 54 | Param-coupled numbers | l. 845 | `throughput.csv`, `error.csv` | agree |
| 55 | Ablation numbers | l. 952 | `ablation_comparison.csv` | agree (but see M7) |
| 56 | Sensitivity range 0.09–0.15 | l. 961 | `sensitivity_heatmap.csv` | agree |
| 57 | r-diagnostics: ζ̂ 0.022/0.042, r ∈ [0.92, 1.80], TV 1.1%/0.26%, bounds 2.3%/4.4%, 0.5% narrower | l. 229, l. 591 | `r_denominator_mismatch_*.json` | agree |
| 58 | "Production history" 33,748 evaluations, ESS fraction 0.213 | l. 591, Table 9 | Table 8 production: 12,900 evaluations, ESS 352 | **disagree** (different run) |
| 59 | Retroactive pass cost | l. 782 | `retrospective_pass_cost.json` | agree |
| 60 | 3 × 10⁷-particle history | l. 782 | no such run described | **unsupported** |
| 61 | Data-availability counts | l. 522: 35 CSV, 27 scripts | 37 CSV, 27 scripts | CSV count off |
| 62 | AMIS consistency status | §2.3 "consistent … CLT" | App. A "heuristic" | **contradiction** |
| 63 | Propagator state | §3.4 "stores nothing"; Table 1 "none" | §3.1, App. D: snapshot buffer carried | **contradiction** |
| 64 | Theory covers the run estimator | §4 l. 170 | App. B l. 613 "we do not prove that the theorems transfer" | **contradiction** |
| 65 | Twin heterogeneity reported posterior | absent | `twin_hetero_raw.csv`: W₁ 0.15–0.23 (σ = 0), 2.0–2.4 (σ ≥ 1) | **omitted** (M4) |

---

## 5. Recommendation

**Major revision.**

The underlying work is substantial and, in its appendices, unusually honest: a working generation-free ABC sampler on a real HPC island model, a clean instrument (the barrierized twin) for isolating the barrier, a genuinely useful straggler-factor predictor, a real tissue simulator, and a data release that lets a referee re-derive nearly every number (I did, and the vast majority check). The problem is the distance between what the appendices show and what the abstract, introduction and conclusion say. The headline C1 accuracy is quoted from the trivial case and hides ±20% misses and an unexplained 1.5–2.3× factor; the headline C2 gain is one the paper itself attributes mostly to pyABC overhead; the theory is claimed for the reported estimator while its premises are, by the authors' own table, unverified or violated on the reported runs; the reported estimator's collapse under batched proposals is present in the vendored data and half-reported; and there are a dozen internal contradictions (three vs two excluded points, 5.1 s vs 13 s, 33,748 vs 12,900 evaluations, "single node" vs "3 nodes", "a thousand times" vs 22×, the disowned metric in the ablation, "stores nothing" vs a carried buffer). None of this requires new cluster campaigns to fix, with the possible exception of a twin re-run to settle the Cellular Potts duration effect; most of it is re-framing, re-scoring, and stating which run is which.

**The three changes that would most improve the paper:**

1. **Restate C1 with the instruments and the misses in view.** Separate the persistent-straggler (deterministic, twin), lognormal-heterogeneity (injected law, twin, ±20%), and Cellular Potts (utilisation ratio, twin at 50³ only; pyABC at 80³) results; count the excluded points correctly; put the heterogeneity misses into the abstract; and either explain or measure the twin's 1.5–2.3× duration inflation on Cellular Potts (a twin re-run with staggered starts or on a quiet filesystem would settle it) — until then, do not claim the tissue simulator validates the predictor to "a few per cent". Identify the 50³ campaign used for C1 (5.1 s, CV 0.10) as distinct from the production benchmark (13 s, CV 0.05), and identify the 33,748-evaluation history used in §A.2–A.3.

2. **Make C2's headline the barrier-attributable gain and give it an uncertainty.** Report the equal-*n* and equal-wall-clock tolerance ratios against the barrierized twin at 50³ (the clean comparator the authors already have), or lead with the 80³ figure the paper itself prefers (2.0×), and confine the 2.4×/4.1× pyABC numbers to a clearly labelled software comparison. Add replicate ranges to every ratio in Table 6, state that the Cellular Potts per-simulation advantage is budget-dependent given the measured slopes (−2.21 vs −1.44), and run — or explain why not — a pyABC configuration with population a multiple of the worker count, and ideally pyABC's look-ahead mode, on one benchmark.

3. **Bring the estimator's fragility into the main text and re-score the appendix studies.** Report the heterogeneity-twin reported posteriors (W₁ 0.15–2.4 while the archive is at 0.01–0.03), reframe l. 367 as a robustness result for the estimator rather than "what the barrier does", connect it to the snapshot-quadrature factor (b) of §A.2, and state in the abstract that the 4-D g-and-k recovery is worse than the baseline's and over-covers. Re-score the ablation and sensitivity studies against the analytic posterior (or delete them), include the baseline's g-and-k and bimodal SBC rows, and fix the "stores nothing between calls" / consistency-for-the-run-estimator statements so that the main text matches the appendices.
