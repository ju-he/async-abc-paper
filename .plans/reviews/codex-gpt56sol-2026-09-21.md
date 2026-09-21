# External review — codex CLI, GPT-5.6-Sol (xhigh reasoning), 2026-09-21

Target: `latex/sn-article-template/sn-article.tex` at commit 07364d5 (branch `campaign-tooling`), with the eight main-text figures attached as images. Reviewer had read-only access to the whole repository (vendored data, generator scripts, venv) and was asked for a referee report with a numbers-consistency audit. Tokens used: 455,794.

---

# Referee report

## 1. Summary

The manuscript proposes a generation-free ABC sampler implemented in Propulate. It maintains a sliding top-\(k\) archive, adapts a smooth-kernel mixture proposal after individual arrivals, stores proposal-time importance weights based on a finite snapshot buffer, and reports a retrospective full-history estimator using an approximate deterministic-mixture AMIS denominator. The paper makes four empirical claims: C1, synchronization costs are predicted by a workload’s “straggler factor”; C2, asynchronous throughput translates into a tighter ABC tolerance; C3, the advantage begins at simulations costing a few milliseconds; and C4, the reported posterior is calibrated, recovers a two-parameter Cellular Potts example, and is principally limited by initial bandwidth and archive size. It also proves convergence to an \(r\)-tilted smooth-ABC target under fidelity, filtration, and bandwidth-stabilization assumptions, plus a CLT under stronger rate conditions.

## 2. Major issues

### 2.1 The manuscript does not prove consistency of the implemented asynchronous estimator

The abstract says, “We prove consistency of the reported estimator” (line 42), and the theory section opens with “computed by the algorithm this paper runs” (line 170). Neither statement is established.

First, Theorem 1 converges to \(\pi_{\epsilon_\infty}^{r}\), not to the intended smooth-ABC posterior. Exact consistency requires \(r\equiv1\) in Corollary 1 (lines 215–226), but \(r\equiv1\) is neither proved nor empirically bounded. Only two components of \(r\) are partially compared with another approximate denominator; the truncation-normalizer error and the asynchronous completion/order component remain unmeasured. The reported 1.1% total-variation result is therefore not a bound on the error between the implemented estimator and the intended posterior.

Second, essential assumptions do not hold or are not established for the reported runs:

- The adapted filtration is explicitly assumed under asynchrony (lines 196, 758).
- The runs leave `min_tol` unset, so \(\epsilon_\infty>0\), required even for Theorem 1, is not established (line 762). The empirical tolerance curves continue to decrease, making convergence to zero plausible.
- The CLT rate condition is violated for the examined configuration (lines 593–599, 761).
- Most decisively, Appendix line 613 states that the proofs use proposal order while the post-hoc estimator uses an arrival-ordered log, and that “we do not prove that the theorems transfer between the two.” This directly contradicts the main-text claim.

There is also a provenance problem with the empirical theory diagnostics. The current Cellular Potts inference uses division rate and cell volume, with motility fixed (lines 825 and 830). However, `diag_denominator_mismatch_cpm.py` and `diag_proposal_drift.py` analyze `{"division_rate", "motility"}`. Thus the 1.1% denominator result and the stabilization/confounding argument concern an earlier parameterization, not the posterior experiment reported in the paper. This also explains the contradiction between line 830, which says the two retained directions are nearly orthogonal (\(|\cos|=0.07\)), and lines 597–599, which say the Cellular Potts parameters are confounded and form a ridge.

The authors should either prove a result for the actual arrival-ordered, completion-time-selected estimator and verify its assumptions, or state much more narrowly that they prove conditional convergence to an unknown tilted target for an idealized ordering. The abstract, introduction, conclusion, and C4 claim must not call the present result unqualified consistency.

### 2.2 The algorithm and barrierized twin are not described as implemented

The state claims are internally contradictory. Line 101 correctly says the running sampler carries a snapshot buffer and scheduler throttles. Lines 129 and 134 then say the buffer is a deterministic function of history and the update “carries no state.” Appendix line 790 states that the buffer is rebuilt empty after restart and therefore changes all subsequent proposal-time weights and the adaptation trajectory. Since stored proposal-time weights enter later proposal mixtures, this is algorithmic state, not merely an inconsequential cache.

The bandwidth schedule is also misstated. Line 109 says it moves toward a bandwidth with \(k\) particles below it. Appendix lines 770–776 and the implementation use the default extra margin \(\mu=k\), so tightening is gated by \(k+\mu=2k\), not \(k\). This matters directly to the bandwidth-transient argument and to comparisons with the \(k\)-th order statistic.

More seriously, the “barrierized twin” does not implement the generational behavior attributed to it. Line 232 says the barrier ensures “a generation of \(W\) proposals is drawn from one archive state,” and line 367 explains its posterior collapse by saying each generation contains \(W\) copies of one frozen proposal. Yet `abcpmc_barrier.py` explicitly says that history truncation to a batch boundary is “deliberately NOT implemented”; it inserts a barrier before `__call__` while retaining rank-local asynchronous histories. The Propulate loop receives messages after a worker’s own evaluation, so merely entering a collective does not prove that all ranks subsequently use an identical archive.

The twin is still a useful synchronization instrument, but it is not the frozen-archive generational algorithm described in the manuscript. The posterior-collapse interpretation in line 367 is therefore unsupported. The authors should record and compare the archive/proposal identifiers used by every rank at each barrier, or implement a true synchronized snapshot/all-gather generation and clearly distinguish it from the synchronization-only twin.

### 2.3 C1 conflates distinct notions of “straggler factor” and does not consistently predict from asynchronous timing alone

The factor \(\mathbb E[\max_{i\le W}T_i]/\mathbb E[T]\) applies to exchangeable or identically distributed batch runtimes under appropriate renewal assumptions. The persistent-worker experiment is not such a workload: one fixed rank is slow while the others repeatedly complete short jobs. At \(20\times\), the paper gives approximately 2.004 s for the slow worker and 3.6 ms for a fast evaluation. The mean duration of one fixed-assignment 16-worker batch is then approximately
\[
(15\times0.0036+2.004)/16 \approx 0.129\ {\rm s},
\]
so \(\max/\mathrm{mean}\approx15.6\), not \(401\). The reported \(401\times\) is the measured asynchronous aggregate rate divided by \(16/2.004\), a legitimate systems ratio but not the straggler factor as defined in the abstract and introduction.

The same distinction appears in the lognormal study. Line 320 gives \(\mathbb E[\max_{48}T]\approx147\) s and \(\mathbb E[T]\approx7.4\) s at \(\sigma=2\), implying a straggler factor around \(19.9\), whereas Table 4 reports a predicted ratio of \(35.4\). The latter uses the finite-budget asynchronous completion rate, which is strongly affected by censoring of long jobs. It is not \(\mathbb E[\max]/\mathbb E[T]\).

Nor are all predictions derived from asynchronous timing alone:

- For lognormal heterogeneity, lines 288 and 320 acknowledge that the empirical asynchronous sample is tail-censored and the prediction uses the known injected lognormal law.
- The predictor-generation script likewise samples from the parametric injected law.
- The \(80^3\) Cellular Potts point is a single hard-coded value from a study log and is measured against pyABC, not the twin.
- The \(50^3\) points use the utilization ratio as the measured quantity, whereas the straggler and heterogeneity points use throughput ratios.

Thus Figure 1 combines different estimands and comparators on one equality plot. In addition, its code marks three points as outside the model—straggler \(0\times\), straggler \(1\times\), and Cellular Potts at 384 workers—while the caption says there are two.

Finally, the dismissal of the twin’s 1.5–2.3-fold longer simulation durations as something “a barrier cannot cause” (line 341) is too strong. Synchronized starts can cause synchronized I/O and filesystem contention; indeed this is one of the manuscript’s proposed explanations. That would be a secondary consequence of the barrier, not an unrelated effect that can simply be divided out.

C1 needs a precise estimand for IID runtime heterogeneity, fixed worker heterogeneity, finite wall-clock censoring, and synchronization-induced contention. Predictions should be generated by one documented procedure from the available asynchronous record, with uncertainty intervals and without substituting the known injected law unless the claim is correspondingly narrowed.

### 2.4 C2 does not isolate conversion of barrier removal into statistical efficiency

The pyABC comparator differs from the proposed method in more than synchronization: proposal adaptation occurs per generation rather than per arrival; its reported estimator is a final population rather than a full-history deterministic mixture; it uses classical population weights; it reserves one rank as a dispatcher; and it closes generations on early finishers and discards latecomers. The paper acknowledges most of these differences individually, but still titles C2 “Throughput converts into a tighter tolerance” and repeatedly attributes the comparison to removing the barrier.

The Cellular Potts \(50^3\) results demonstrate the attribution problem. The proposed method is 2.39 times faster than pyABC, but the barrierized twin indicates only about 1.2 times of that is barrier waiting. The manuscript attributes the rest to pyABC overhead. The headline \(2.4\times\) simulation gain therefore cannot be presented as the consequence of removing the barrier.

Likewise, line 371 says throughput and per-simulation efficiency “multiply into the net effect.” Numerically, \(2.388\times1.600=3.82\), not the reported equal-wall-clock ratio 4.09; for Lotka–Volterra, \(0.963\times1.300=1.25\), not 1.30. Nonlinear tolerance curves and separate median aggregation prevent the proposed decomposition.

A convincing C2 evaluation should include the same sampler with and without synchronization at equal wall time—or use the measured tolerance-versus-\(n\) curves to predict and then validate that comparison—across several benchmarks. It should also compare with the closest look-ahead/asynchronous ABC method, not only discuss it. The best-\(k\) rejection sampler is useful for order-statistic efficiency but is not a posterior baseline.

### 2.5 C3’s “few milliseconds” boundary is not identified by the experiment

Figure 4 connects four different inference problems at approximately 0.127 ms, 1.71 ms, 4.15 ms, and 13.5 s. Across these points, dimension, discrepancy geometry, proposal behavior, runtime variability, extinction, baseline acceptance, and history size all change. Monotonicity across four heterogeneous benchmarks does not identify simulation cost as the cause, and interpolation between g-and-k and Lotka–Volterra cannot establish a crossover near 4 ms.

There is an important internal warning: on the \(4\) ms Lotka–Volterra problem, Table 6 gives a throughput ratio of 0.963 over the 600 s production run, whereas the 180 s scaling experiment at the same 48 workers and \(k=100\) gives \(6240/2952=2.11\). A supposed fixed cost boundary changes from slightly unfavorable to more than twofold favorable on nominally the same benchmark. This suggests strong dependence on campaign configuration, run length, or history-growth overhead, none of which Figure 4 controls.

Related inconsistencies are:

- The Cellular Potts scaling data show 5.1 s mean evaluations at 48 workers, while the caption calls the workload a 13 s evaluation.
- The stated 97% efficiency is correct only from one node (48 workers) to eight nodes; relative to the dotted line anchored at one worker, efficiency at 384 workers is about 84%.
- Lines 410, 854, and the scaling-table caption say \(k=1000\) moves the crossover “in to a single node,” but Table 10 and `kfrontier_summary.csv` put the first synchronous overtake at 144 workers, i.e. three nodes.

C3 needs a controlled experiment that varies only simulator cost—for example by adding calibrated delays to the same target, seeds, runtime distribution, worker count, and archive size—and repeats this over worker counts and run lengths.

### 2.6 C4 does not establish calibration or posterior recovery for the reported estimator

The one-dimensional Gaussian SBC result is convincing, but the broader C4 wording is not.

For g-and-k, the reported coverage \(0.62/0.91/0.96/0.99\) at nominal \(0.50/0.80/0.90/0.95\) is materially conservative, not calibrated. More importantly, the committed SBC diagnostic reports nonuniform rank histograms for three of four parameters: \(A\), \(g\), and \(k\) have chi-square \(p<0.01\), while only \(B\) passes at conventional levels. The main table shows only \(A\), and “similarly for the others” conceals meaningful parameter-specific behavior. The explanation that this is a support effect “not a weighting failure” is not established by the support sweep.

The multimodal result is also weaker than stated: both modes are retained in only 86.2% of trials, and the mode containing the truth in 87.1%. Central interval coverage can appear nominal even when one mode is lost.

The Cellular Potts study has no reference posterior. Containment of the fixed synthetic truth in five algorithmic reruns of the same observed dataset is not posterior calibration and does not demonstrate recovery of the posterior distribution. Moreover, the rejection arm has only one replicate, despite Table 8 saying “five replicates per method.”

Several supplementary studies score an object other than the estimator defined in Equation 8:

- Figure 3’s heterogeneity errors come from `gaussian_analytic_summary.csv`, which is the unweighted top-\(k\) archive, not the full-history AMIS estimator.
- Line 845 calls an “unweighted top-\(k\)” mean the “reported” posterior, then calls an AMIS-weighted archive the estimator covered by the theory. Equation 8 instead sums over the full evaluated history.
- The ablation uses Wasserstein distance from an archive to a point mass, a diagnostic the manuscript itself rejects as rewarding posterior collapse (lines 270 and 941–942).

The starting-bandwidth interpretation also overreaches. Line 473 says “The sampler’s draws are the same in both rows,” but the two rows are separate runs and \(\epsilon\) enters both proposal weights and covariance, so changing \(\epsilon_0\) changes the sampling trajectory. The similar \(100\)-th order statistics show similar achieved discrepancies, not identical draws. In addition, \(\epsilon_0=10\) is \(10/0.46\approx22\) times the stated prior-predictive median, not “a thousand times” as claimed in Table 8 and line 486. Truth is covered in all control runs as well, so the abstract’s “once the starting bandwidth is set…” condition pertains to contraction, not truth coverage.

C4 should be rebuilt around the actual full-history estimator: rank-based SBC in multiple dimensions, multimodal mode-retention diagnostics, parameter-dependent runtimes, and sufficient replicated datasets. If Cellular Potts remains a truth-recovery illustration rather than a calibrated posterior benchmark, it should be described that way.

### 2.7 The provenance and reproducibility trail is not yet reliable enough

Line 522 says there are 35 per-figure CSV files; there are currently 37. More importantly, it says all 27 generator scripts have a `--refresh` route and that every number can be checked from the repository alone. This is not true:

- `make_predictor_fig.py` has no refresh path and explicitly exits if one is requested.
- The \(80^3\) predictor value is hard-coded from a cluster job log.
- `make_twin_tables.py` and several diagnostic scripts rely on personal absolute scratch paths.
- The raw records required to regenerate the summaries are not deposited.
- The current `kfrontier_summary.csv` reports a worst deviation of 0.271 for \(k=50\), whereas Table 12 reports 0.250. The generator accidentally combines the main and doubled-budget variants when aggregating.

Before acceptance, the authors should provide a versioned public archive and a single manifest mapping each figure/table/text number to the exact run configuration, raw data, aggregation rule, and generator invocation.

## 3. Minor issues

1. “Simulation” and “evaluation” are used interchangeably. One Cellular Potts evaluation contains four simulator replicates, yet tables and axes label evaluations as “simulations/s.” Define and use one unit consistently.

2. In dimensions above one, the reported \(W_1\) is actually a 100-projection sliced Wasserstein distance on unstandardized coordinates. It should be named accordingly. The g-and-k coordinates have different scales, so the distance is dominated by the broader coordinates unless parameters are standardized.

3. Line 119 says smooth kernels make every evaluated particle contribute continuously to the proposal. Only top-\(k\) particles enter the archive, and membership changes discontinuously at the \(k\)-th rank; the bootstrap-to-archive transition also remains discontinuous.

4. The claim that the snapshot buffer “saturates at \(S=5\) in every dimension” is too strong. At \(d=8\), mean signed coverage deviation changes from \(0.006\) at \(S=5\) to \(0.055\) at \(S=20\) and \(0.074\) at \(S=50\). At \(d=4\), \(S=0\) is closer to zero than \(S=5\).

5. Table 8 says “medians otherwise,” but ESS values 760 and 352 are means. Their corresponding medians are approximately 645 and 303.

6. Figure 1’s caption says two open-marker configurations, while the plotting script marks three.

7. Figure 5’s Cellular Potts caption should distinguish the 5.1 s scaling configuration from the 13.5 s production configuration.

8. “Truth covered in every replicate” should not be phrased as coverage when the dataset is fixed and only algorithmic seeds vary.

9. The g-and-k reference is an asymptotic-octile approximation sampled by MCMC, not an exact posterior. Its Monte Carlo and approximation errors should be quantified.

10. The sensitivity and ablation studies are one-dimensional and use a point-mass diagnostic. They do not justify general robustness of posterior calibration.

11. The manuscript is 58 pages and repeats many caveats and claims across the introduction, results, discussion, limitations, and appendices. It would benefit from a substantially shorter main paper and a separate supplement.

12. The submission still contains placeholder authors, emails, affiliation, acknowledgements, funding, and contributions. If this is not an anonymized-review requirement, these must be completed.

13. Correct “in to” to “into” in the scaling discussion and caption.

## 4. Numbers-consistency audit

The following lists every quantitative claim I cross-checked against another manuscript location, the committed figure/table data, or the supplied generators.

### C1 and systems measurements

- **Predictor-domain summary (line 280):** committed predictor rows give 12 in-domain points, predicted/measured median 0.996 and range 0.840–1.194. **Agrees.**

- **Measured range 1.2–400 (abstract; lines 280, 503):** the in-domain minimum is approximately 1.18 and the maximum 401.8. **Agrees after rounding.**

- **Persistent-straggler throughput (Table 3):** async medians are 3888, 3112, 3241, 3218, 3205; fine-twin medians are 48.7, 65.3, 31.6, 15.9, 7.98. **Agrees.**

- **Persistent-straggler ratios:** measured ratios at \(5,10,20\times\) are 102.5, 202.5, and 401.8; predicted values are 102.1, 201.9, and 401.4. **Supports the “within 1%” statement.**

- **Straggler posterior values (lines 367 and Table 3):** at 6304 evaluations the matched async \(W_1\) is 0.00831 versus 0.737 for the fine twin; at 3200 it is 0.0696 versus 1.729. **Agrees.**

- **Lognormal throughput table:** medians 47.6/42.1/28.9/17.6/11.4 async and 44.8/15.1/4.37/1.06/0.370 fine twin match Table 4. **Agrees.**

- **Lognormal measured and predicted ratios:** 1.06, 2.8, 6.6, 16.6, 29.6 measured and 1.00, 2.77, 6.34, 14.0, 35.4 predicted. **Agrees with the table.**

- **Lognormal “straggler factor” at \(\sigma=2\):** line 320 gives \(147/7.4=19.9\), while the reported predicted async/twin ratio is 35.4. **Does not agree with the manuscript’s identification of the ratio as \(\mathbb E[\max]/\mathbb E[T]\).**

- **Cellular Potts \(50^3\) factorization (Table 5):** throughput ratios 1.91/1.87/1.86/2.82, utilization ratios 1.18/1.19/1.21/1.21, and duration ratios 1.63/1.59/1.52/2.33 match the committed decomposition. **Agrees up to median-rounding.**

- **Cellular Potts predictor accuracy:** predicted/measured utilization ratios at 48/96/192 workers are 1.067, 1.088, and 1.084; the \(80^3\) point is 0.945. **The data support roughly 5–9% error, not uniformly “a few percent.”**

- **Cellular Potts \(80^3\):** committed production summaries give async/sync throughputs 0.274/0.134, ratio 2.05, and utilization 97.6%/49.3%; the predictor row is 1.90 against 2.01. **Approximately agrees with 2.1 measured and 1.9 predicted, but the predictor row has only one aggregate observation and is not a twin comparison.**

- **Open-marker count in Figure 1:** plotting code marks straggler \(0\times\), straggler \(1\times\), and CPM-384 as outside. **Disagrees with the caption’s “two configurations.”**

### C2 and C3

- **Gaussian matched-budget row:** throughput 0.284, per-simulation ratio 0.775, equal-wall ratio 0.242. **Agrees with Table 6.**

- **g-and-k matched-budget row:** 0.458, 1.063, 0.879. **Agrees.**

- **Lotka–Volterra matched-budget row:** 0.963, 1.300, 1.300. **Agrees.**

- **Cellular Potts matched-budget row:** 12,896 versus 5400 simulations, throughput 2.388, per-simulation 1.600, equal-wall 4.088. **Agrees with the 2.4/1.6/4.1 headline.**

- **Claim that the factors multiply:** \(2.388\times1.600=3.82\), not 4.09; \(0.963\times1.300=1.25\), not 1.30. **Does not agree literally.**

- **Cellular Potts rejection comparison:** \(2.943\times10^{-3}/7.629\times10^{-5}=38.6\). **Agrees with “39 times looser.”**

- **Control-bandwidth comparison:** \(4.458\times10^{-4}/6.311\times10^{-5}=7.06\). **Agrees with 7.1.**

- **Heterogeneity Figure 3:** async archive-mean error is 0.00697–0.01081 and baseline error rises from 0.0533 to 0.284; completed simulations match 2832→684 and 1375→118. **Agrees numerically, but these are unweighted archive statistics, not the reported full-history estimator.**

- **Lotka–Volterra timing:** productive fractions are 25.3% at 48 workers, 7.8% at the three-node instrument, and 2.0% at six nodes. **Agrees with line 410.**

- **Lotka–Volterra scaling:** async throughput 6240 at 48 workers and 1729 at 288; synchronous 2952 and 6034. **Agrees with Table 10.**

- **Lotka–Volterra 48-worker comparison across studies:** production gives 0.963 async/sync, while scaling gives \(6240/2952=2.11\). **Internally inconsistent unless the configuration/run-length dependence is explained.**

- **Cellular Potts node scaling:** \(71.985/(8\times9.332)=0.964\). **Supports 97% efficiency from one node to eight; relative to one worker it is only about 84%.**

- **Cellular Potts scaling cost:** decomposition data give approximately 5.1 s per evaluation, while the benchmark and caption say 13 s. **Does not agree.**

- **Archive-size throughput:** \(k=100\) gives 6240 versus 6684 at \(k=50\), a 6.6% difference. **Agrees with “within 7% of fastest.”**

- **\(k=1000\) crossover:** at one node async remains ahead, 3804 versus 2952; the first overtake is at 144 workers/three nodes. **Disagrees with “moves … in to a single node”; agrees with Table 12’s three-node entry.**

### C4 and posterior results

- **Gaussian SBC:** 0.510/0.802/0.895/0.941 versus nominal 0.50/0.80/0.90/0.95. **Agrees after rounding.**

- **Bimodal SBC:** 0.502/0.823/0.907/0.959 and both-mode retention 0.862. **Agrees.**

- **g-and-k \(A\) coverage:** 0.615/0.911/0.960/0.988. **Agrees with 0.62/0.91/0.96/0.99.**

- **g-and-k “similarly for the others”:** the other parameters are also conservative, but committed rank tests reject uniformity for three of four parameters. **Only partly agrees; “calibrated” is unsupported.**

- **Reference-posterior recovery:** Gaussian final medians are 0.0161 async, 0.0276 sync, 0.0112 rejection; g-and-k 0.0685, 0.0285, 0.470. **Agrees with line 438 after rounding.**

- **Gaussian ESS range:** checkpoint medians range 237.8–296.2. **Does not agree exactly with the stated 237–303.**

- **g-and-k ESS extrema:** the minimum 7.48 occurs at 150 s and the maximum 2986.7 at 600 s. No single checkpoint spans both values. **Disagrees with “7 to 2987 at the same wall clock across replicates.”**

- **Cellular Potts contractions:** means are 91.7%/80.7% async, 91.1%/72.2% sync, and 91.4%/67.0% rejection. **Agrees with 92/81, 91/72, and 91/67.**

- **Cellular Potts ESS:** 351.9 async and 65.2 sync are means; medians are 303.0 and 65.6. **The numerical values agree with the means but not with the caption’s “medians otherwise.”**

- **Truth containment:** all five fixed async and all five sync runs contain the truth in both intervals; the rejection arm contains it but has one run. **The containment claim holds, but “five replicates per method” does not.**

- **Initial bandwidth relative to prior-predictive median:** \(10/0.46=21.7\). **Disagrees with “a thousand times.”**

- **Bandwidth transient:** the median reported-bandwidth/order-statistic ratio at \(\epsilon_0=10\) is 320.7; at \(\epsilon_0=0.1\) it is 2.03. **Agrees with 321 and 2.**

- **Re-reporting replicate 0:** contraction is 92.9%/84.7% at report \(k=100\). **Agrees with 93%/85%.**

- **Archive re-report sweep:** cell-volume contraction declines from 85.3% at report \(k=10\) to 77.8% at \(k=1000\), with both truths retained. **Agrees.**

- **Running \(k=300\):** cell-volume contraction is 6.2% and 6.9% in the two runs; reported bandwidth remains 0.1. **Agrees with approximately 7% and failure to leave \(\epsilon_0\).**

- **Running \(k=30\):** cell-volume contraction is 83.3% and 82.0%, and one of two division-rate intervals misses truth. **Agrees.**

- **Cellular Potts \(80^3\) weak posterior:** means are approximately 50.0%/14.5% async, 51.7%/10.6% sync, 77.7%/52.9% rejection. **Agrees with 50/15, 52/11, 78/53.**

- **Snapshot “saturation at \(S=5\)”:** at \(d=8\), signed deviation is 0.006 at \(S=5\), 0.055 at \(S=20\), and 0.074 at \(S=50\). **Does not support saturation without degradation.**

- **\(k\)-frontier calibration:** Table 11 gives worst signed-deviation magnitude 0.250 for \(k=50\) and 0.064 for \(k=100\). **Agrees with the main sweep, but the committed `kfrontier_summary.csv` reports 0.271 for \(k=50\) because it mixes the doubled-budget variant.**

### Appendices and reproducibility

- **Denominator diagnostic:** committed JSON gives \(\widehat\zeta=0.02212\), \(r\in[0.9199,1.8028]\), TV distance 1.0615%; Gaussian TV is 0.2644%. **Agrees numerically with lines 591 and 229, but the Cellular Potts diagnostic uses the old division-rate/motility model.**

- **Retrospective-pass cost:** fitted cost is 49.32 s per million particles, maximum linear-fit residual 7.91%, extrapolated 8.22 min at \(10^7\) and 24.66 min at \(3\times10^7\). **Agrees with line 782.**

- **Ablation:** full 0.0732, hard kernel 0.0756, no AMIS 0.0719. **Agrees, though the metric is the disfavored point-mass diagnostic.**

- **Sensitivity range:** committed cells span 0.0917–0.1527. **Agrees approximately with 0.09–0.15.**

- **Parameter-coupled runtime:** async throughput 1881→2014→1570→1146 and archive-mean errors about 0.007–0.013; weighted-archive ESS fractions around 0.88 versus 0.99, maximum weights around 0.03 versus 0.01, and drain shifts at most 0.0125. **Agrees with line 845 after rounding, but again concerns archive-based rather than full-history reporting.**

- **Data inventory:** 37 CSV files and 27 `make_*.py` scripts are present. **The script count agrees; the claimed 35 CSV files does not.**

## 5. Recommendation

**Major revision.**

The idea is interesting and the manuscript contains unusually candid diagnostics, but the central claims currently exceed what the theory and experiments establish. The three changes that would most improve the paper are:

1. **Make the theory and algorithm match the actual asynchronous estimator.** Resolve proposal-order versus arrival-order reconstruction, stateful snapshot behavior, completion-time selection, and the positive-bandwidth condition; prove or quantitatively bound fidelity to the intended ABC posterior.

2. **Redesign C1–C3 as causal systems experiments.** Use a genuine synchronized twin with verified common proposal states, a controlled simulator-cost sweep on one fixed inference problem, consistent throughput/utilization estimands, and a closest asynchronous/look-ahead ABC comparator.

3. **Rebuild C4 and the audit trail around the exact reported estimator.** Run full rank-based SBC and multimodal diagnostics on the full-history estimator, update all Cellular Potts diagnostics to the current parameterization, add replicated datasets where posterior recovery is claimed, and publish the raw-data/configuration manifest needed to regenerate every number.
