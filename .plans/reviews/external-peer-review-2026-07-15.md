# Peer-review report

## Overall assessment

**Recommendation: Major revision / reject-and-resubmit.**

The paper addresses an important and practically consequential problem: synchronization waste in ABC-SMC when simulator runtimes are heterogeneous. The proposed combination of single-arrival adaptation, history-reconstructed state, smooth ABC kernels, and AMIS-style reweighting is interesting. The systems results are also potentially strong, especially the straggler experiment and the Cellular Potts scaling study. The paper deserves credit for reporting the unfavorable Lotka–Volterra multi-node result rather than presenting asynchrony as universally superior.

However, the central theoretical claims are not currently justified, the exact posterior estimator appears computationally inconsistent with its stated complexity, and the experimental baseline does not isolate synchronization as cleanly as claimed. These are substantive issues rather than matters of exposition.

## Strengths

1. **Important target regime.** Barrier-induced idle time is a genuine bottleneck for expensive, parameter-dependent simulators.

2. **Compelling systems evidence.** The persistent-straggler result clearly illustrates the failure mode of generation barriers. The Cellular Potts study provides a realistic workload and shows a substantial throughput gap at scale.

3. **Honest characterization of limits.** The decline beyond one node for the inexpensive Lotka–Volterra simulator is informative and strengthens the systems narrative.

4. **Good experimental breadth.** The paper includes calibration, posterior recovery, artificial heterogeneity, parameter-dependent runtime, scaling, sensitivity, and ablation studies.

5. **Clear overall presentation.** The figures are legible, and the distinction among computational throughput, utilization, and posterior quality is generally well organized.

## Major concerns

### 1. The consistency and CLT results are not established as stated

Section 4 presents the results as corollaries of AMIS theory, but several additional arguments are required.

First, each observation consists of both a parameter and a stochastic simulator output. The natural importance-sampling construction is therefore on an augmented space such as \((\theta,\rho)\), with proposal density \(q_i(\theta)p(\rho\mid\theta)\). The current proof sketch treats the samples essentially as \(\theta_i\sim q_i\) and inserts \(K_\epsilon(\rho_i)\) afterward. This can likely be repaired because the kernel is bounded, but it needs to be formulated correctly.

More seriously, the CLT is centered at the limiting target \(\pi_{\epsilon_\infty}\), while Condition 3 only assumes \(\epsilon_n\to\epsilon_\infty\). A CLT centered at the limiting target normally requires a rate such as

\[
\sqrt n\{\pi_{\epsilon_n}(f)-\pi_{\epsilon_\infty}(f)\}\to 0,
\]

or an equivalent condition. Monotonic convergence and \(\epsilon_n-\epsilon_{n+1}\to0\) do not imply this. A schedule converging as slowly as \(1/\log n\), for example, would violate the required scale.

Condition 4 has a similar problem: convergence of an approximate denominator to the full mixture without a rate may support consistency, but it is generally insufficient for the stated CLT. Moreover:

- If the reported estimator uses the exact full cumulative mixture, Condition 4 is unnecessary or tautological.
- If it uses the fixed \(S=20\) buffer, uniform convergence to the cumulative mixture does not follow without a separate proposal-stabilization result.
- The paper acknowledges that Condition 1 does not hold for the implemented top-\(k\) proposal.

Thus the theorems currently describe neither the implemented algorithm nor a fully specified alternative. A practical resolution would be to add a persistent defensive prior component,

\[
q_n^{\mathrm{def}}=\delta\pi+(1-\delta)q_n,
\]

establish uniform covariance bounds, and give a complete theorem for that actual algorithm. The CLT should include explicit rates and a precise asymptotic variance. The statement that generational ABC-PMC lacks CLTs should also be removed or qualified; standard SMC central-limit theory is extensive.

### 2. The claimed cost of the exact retrospective estimator appears incorrect

The paper defines

\[
\bar q_n(\theta_i)=\frac1n\sum_{j=1}^{n}q_j(\theta_i),
\]

where each \(q_j\) is itself a mixture with up to \(k\) components. Evaluating this denominator for all \(n\) particles appears to require, naively,

\[
O(n^2k)
\]

kernel-component evaluations, not the stated \(O(nk)\).

Chunking 65,536 evaluation points reduces peak memory, but it does not remove the sum over all proposal times. If only \(M\) proposal snapshots are retained, the cost is \(O(nMk)\), but that would not be the full cumulative mixture over every proposal unless \(M=n\). The claim that histories of order \(10^7\) particles are processed exactly makes this discrepancy especially consequential.

The manuscript needs:

- exact pseudocode for constructing the retrospective denominator;
- the number of distinct proposals or snapshots in that denominator;
- correct time and storage complexity;
- measured post-processing time and memory at the largest runs;
- end-to-end time-to-posterior results that include this step.

At present, either the complexity statement is wrong or the implemented denominator is not the estimator defined in Equation 6.

### 3. The three weight definitions are internally inconsistent

The history in Equation 3 stores \(w_i=\pi/q_{\tau_i}\). Equation 4 then uses \(w_jK_\epsilon(\rho_j)\) to construct the next proposal. Equation 5 instead defines a proposal-time weight using a sliding mixture denominator. The “three weight objects” paragraph subsequently says that the Equation 5 weight is used **only** for the ESS diagnostic.

This leaves several unresolved possibilities:

- Does proposal adaptation use \(\pi/q_{\tau_i}\), the sliding-buffer weight, or a retrospectively updated cumulative-mixture weight?
- What does “corrected at every subsequent use” mean operationally?
- If Equation 5 is only diagnostic, why is the stored \(w_j\) central to Equation 4?
- Can the complete sequence of proposals \(q_j\) be reconstructed from the saved history without also reconstructing every historical weight version?

This is not merely a notation issue: the answer determines the proposal sequence, the retrospective denominator, the theoretical assumptions, and whether crash recovery reproduces the same computation. The paper should provide one unambiguous algorithm containing every weight update and every stored field.

### 4. The baseline does not cleanly isolate synchronization

The synchronous method is said to use a probabilistic acceptor that retains a sample with probability \(K_\epsilon(\rho)\). The asynchronous estimator, in contrast, appears to retain every evaluated particle and use the deterministic factor \(K_\epsilon(\rho)\) in its importance weight.

Using the same kernel does not make these statistically identical:

- the synchronous baseline introduces an additional Bernoulli rejection step;
- its generation length depends on the resulting acceptance probability;
- the asynchronous method extracts information from every simulation;
- the proposal and importance-weighting mechanisms differ substantially.

Consequently, the inference-per-wall-clock comparison reflects more than barrier removal. The most convincing control would be a **barrierized twin** of the proposed algorithm: identical proposals, deterministic kernel weights, archive, and posterior estimator, but with updates performed only after batches of size \(N\). A synchronous deterministic-kernel importance-sampling baseline would also help.

The present comparison remains useful as an implementation comparison against pyABC, but claims that it attributes the difference specifically to synchronization should be weakened.

### 5. AMIS reweighting does not by itself correct runtime-dependent completion bias

At a fixed wall-clock deadline, particles with long parameter-dependent runtimes are more likely to remain unfinished. This is informative censoring. Dividing by the proposal density corrects proposal adaptation; it does not generally correct selection according to completion time.

The parameter-coupled-runtime experiment is valuable, but the conclusion is too strong. A stable posterior mean in a one-dimensional capped-delay experiment does not establish that runtime bias has been corrected. In particular, the most relevant ablation—**AMIS versus no AMIS under runtime–parameter coupling**—is absent. The no-AMIS experiment is instead conducted on a uniform-runtime target, where the paper itself expects little difference.

The authors should:

- run the coupling experiment with and without AMIS;
- report bias in means, variances, quantiles, and coverage, not only posterior mean;
- test symmetric, multimodal, and more severe runtime relationships;
- distinguish submitted, completed, and still-running particles;
- repeat the analysis after draining all jobs submitted before the deadline.

If draining changes the posterior, the difference directly measures deadline censoring. A formal correction would require completion probabilities or an anytime-sampling argument, not merely the AMIS proposal denominator.

### 6. Calibration evidence is promising but overstated

The asynchronous coverage values are \(0.50,0.78,0.87,0.92\) at nominal levels \(0.50,0.80,0.90,0.95\). These are clearly better than the synchronous baseline, but the deviations at the two highest levels are several binomial standard errors with 1,000 trials. Thus “well calibrated” should be replaced by something like “substantially better calibrated than the matched baseline, with residual upper-level undercoverage.”

A stronger validation should include:

- confidence intervals or formal uniformity tests for SBC ranks;
- SBC on at least one nonlinear, multidimensional problem;
- effective sample size, maximum normalized weight, and weight-tail diagnostics;
- a multimodal example, because a top-\(k\) archive is particularly vulnerable to mode loss;
- sensitivity of calibration to \(k\), \(S\), covariance jitter, and the defensive-mixture weight if added.

The current hyperparameter study is confined to the one-dimensional Gaussian benchmark and is insufficient to support broad robustness claims.

### 7. “Stateless,” “exactly reproducible,” and crash-recoverable need more precise definitions

A candidate generator cannot literally be a pure function of history unless its randomness is also derived deterministically from that history. A conventional mutable pseudorandom-number-generator state contradicts the pure-function claim. Furthermore, asynchronous MPI arrival order is generally nondeterministic and can change after restart, especially when simulations were in flight during failure.

The paper should specify:

- whether the history is an ordered event log;
- how candidate random keys are generated;
- whether proposal IDs and in-flight candidates are persisted;
- how duplicate or late results are handled after restart;
- whether recovery reproduces the identical future trajectory or merely a valid trajectory;
- results from an actual kill-and-resume experiment.

The Appendix documents replicate-level seeds, but that does not by itself establish deterministic per-arrival replay.

## Additional comments

- **Table 1 overgeneralizes ABC-SMC.** Classical ABC-SMC is not intrinsically restricted to hard kernels, mutable software state, or one particular proposal implementation. The table should compare this method with the specific baseline used in the experiments.

- **Benchmark descriptions are insufficient.** The priors, summaries, discrepancy normalization, observation generation, and reference-posterior construction for g-and-k, Lotka–Volterra, and Cellular Potts must be provided. Calling them “standard” is not enough for reproduction.

- **Strong-scaling fairness needs clarification.** The synchronous population changes with worker count, while the asynchronous archive is mostly fixed. This alters both the statistical algorithm and the distribution of the generation maximum. Results should be shown across several batch/population sizes.

- **Report full systems configuration.** Include CPU model, interconnect, process placement, MPI configuration, software versions, and communication volume. Cellular Potts scaling has only three to five replicates; uncertainty should be shown at every point.

- **Avoid unsupported absolute wording.** Phrases such as “bit-identical,” “exactly reproducible,” and “never dominates wall-clock” require evidence. Floating-point log-sum-exp reductions are not generally associative at the bit level.

- **The related-work discussion is too short for the novelty claim.** The claim of being the first per-arrival ABC proposal-update method needs a broader treatment of asynchronous ABC, anytime Monte Carlo, adaptive importance sampling, off-barrier SMC, and dynamic master-worker ABC.

- **The submission is not archival-ready.** Author names, affiliations, funding, contributions, code, and data are still placeholders or future promises. Code and raw scaling data should ideally be available during review.

## Required revision path

A publishable revision should, at minimum:

1. Replace the proof sketch with a correct theorem for a precisely defined algorithm, preferably using a defensive prior mixture and appropriate CLT rate conditions.
2. Resolve the \(O(nk)\) versus apparent \(O(n^2k)\) estimator-complexity issue and report end-to-end costs.
3. Give a single, internally consistent definition of all online and retrospective weights.
4. Add a barrierized version of the same algorithm as the principal systems control.
5. Directly test runtime-dependent censoring and the role of AMIS using targeted ablations.
6. Expand calibration to a nontrivial multidimensional or multimodal example.
7. Release sufficient code, configurations, and data to reproduce the central results.

The core systems idea is promising, and the empirical results justify continued development. In its current form, however, the paper’s theoretical guarantees and “synchronization-only” causal interpretation are not sufficiently supported.
