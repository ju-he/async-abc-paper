# Figure review — codex CLI, GPT-5.6-Sol (xhigh reasoning), 2026-09-23

Target: the two algorithm-illustration drafts `latex/figure-drafts/tikz/a8_algorithm_ring.tex` (ours) and `a9_pmc_ring.tex` (textbook ABC-PMC), with their panel scripts and `_toy2d.py`, at the state of 2026-09-23 (uncommitted, branch `campaign-tooling`). Reviewer had read-only access to the repository and the propulate checkout, the 220-dpi renders of both rings attached as images, and the brief in `.plans/reviews/figure-brief-a8a9-2026-09-23.md`. Token usage not reported by the CLI.

---

# Independent review: A8 and A9 algorithm illustrations

## Overall assessment

A8 contains a useful geometric explanation of proposal construction and is worth developing for the Method section. Before publication, however, it must distinguish the proposal-time weight from the reported posterior weight and represent the delayed, asynchronous return of a candidate correctly.

A9 should not enter the paper in its present form. Its “draw all \(M\), simulate all \(M\), wait for the slowest” sequence is neither textbook draw-until-\(N\)-accepted ABC-PMC nor the pyABC baseline used in the experiments. A smaller cadence comparison would be more accurate and effective.

## 1. Accuracy findings

### A8: generation-free method

1. **Panel 6 and the missing reported estimator — blocking.**  
   **Figure:** labels \(w^\star=\pi(\theta^\star)/\bar q_n^{\mathrm{on}}(\theta^\star)\) simply as “weight,” stores it in the history, and ends the loop without showing any other weight.  
   **Paper/code:** this is the proposal-time core weight. It influences later proposals but is not the reported particle weight. The reported estimator reweights every history member as
   \[
   W_{i,n}\propto \frac{\pi(\theta_i)K_{\epsilon_n}(\rho_i)}{\bar q_n(\theta_i)},
   \]
   with a different, history-reconstructed denominator containing draw-proportional snapshots and a defensive prior component ([paper §3.4](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/sn-article.tex:98), [reported estimator](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/sn-article.tex:139), [implementation](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py:1217)).  
   **Likely misreading:** \(w^\star\) is the posterior weight. That directly undermines the “three weights” paragraph. Label it “stored proposal-time weight—not reported,” and add a small history-to-reported-estimator branch or put the reported-weight equation prominently in the caption.

2. **Closing arrow and appended tuple — should fix.**  
   **Figure:** the \(6\to1\) arrow immediately simulates and appends \((\theta^\star,\rho^\star,\epsilon_n,n,w^\star)\) ([A8 TeX](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/tikz/a8_algorithm_ring.tex:41)).  
   **Paper/code:** the propagator returns an unevaluated candidate; an external worker simulates it, potentially while many later proposals and completions occur. The history stores a proposal-time index \(\tau^\star\), precisely because proposal and completion order differ.  
   Replace \(n\) in the tuple by \(\tau^\star\), and render the return as “dispatch to a free worker; later completion appends to \(\mathcal H\).” Multiple faint in-flight candidates would prevent the ring being read as a serial sampler.

3. **Panel 2 makes a smooth bandwidth look like a hard acceptance boundary — should fix.**  
   **Figure:** a dashed ellipse labeled \(\epsilon_n\) encloses the blue archive, visually matching A9’s hard threshold.  
   **Paper/code:** for the smooth method, \(\epsilon_n\) is the scale of \(K_{\epsilon_n}(\rho)\), not an acceptance boundary. The archive is independently the top \(k\) by discrepancy, and every archive member contributes continuously through the kernel weight ([paper lines 88–96](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/sn-article.tex:88)).  
   Use graded kernel shading or label the ellipse explicitly as a “toy level set \(\rho=\epsilon_n\), not an acceptance region.” The current parallel with A9 falsely suggests that the main difference is merely threshold selection.

4. **The tolerance toy is not a faithful miniature of the implemented scheduler — should fix.**  
   **Figure documentation:** says every quantity comes from a faithful miniature whose scheduler targets the discrepancy containing \(2k\) particles.  
   **Toy:** `sched_eps` returns the \(2k\)-th order statistic directly on every call ([`_toy2d.py:127`](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/_toy2d.py:127)).  
   **Paper/code:** smooth-kernel scheduling uses a kernel-weighted ESS-retention search, gated by at least \(2k\) particles below the current bandwidth, capped at a factor-of-two tightening, and normally run once per \(k\) local calls ([Appendix C](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/sn-article.tex:694), [scheduler implementation](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py:1431)).  
   The formula \(\epsilon_n=\min(\epsilon_{\rm hist},\epsilon_{\rm sched})\) is correct, but the generated numerical state is only a scheduler surrogate. Either implement the actual rule or call the run “illustrative,” not “faithful.”

5. **Panel 3’s formula is correct, but the toy covariance differs materially from the implementation — should fix.**  
   **Figure:** \(\Sigma_n=s\,\widehat{\mathrm{Cov}}_{\tilde W}(A_n)\).  
   **Toy:** computes \(\sum_j W_j(\theta_j-\bar\theta)(\theta_j-\bar\theta)^\top\), without the implementation’s weighted bias correction, and adds a fixed \(10^{-4}I\) ([`_toy2d.py:70`](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/_toy2d.py:70), [`_toy2d.py:123`](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/_toy2d.py:123)). The actual code multiplies by \(1/(1-\sum W_j^2)\) for normalized weights and uses scale-aware jitter ([implementation](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py:825)). For this illustrated state, the missing correction is about \(1.13\).  
   This does not invalidate the conceptual panel, but it contradicts the “every quantity is what the algorithm computes” claim.

6. **Panels 4–6 use an internally inconsistent toy proposal density — should fix in the toy or disclose.**  
   **Toy:** samples by rejecting perturbations outside the unit box, but `Proposal.pdf` evaluates the untruncated Gaussian mixture without component normalizers ([draw](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/_toy2d.py:100), [density](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/_toy2d.py:92)). Thus its draws and the density used in \(w^\star\) are not the same distribution.  
   **Implementation:** evaluates per-component in-box masses ([implementation](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py:928)). Appendix C acknowledges that the correlated normalizer is approximate, but it is not simply omitted.  
   Truncation is legitimately too detailed for the drawing; the toy should nevertheless either be internally consistent or explicitly be described as qualitative.

7. **Bootstrap chord terminates at the wrong point in the loop — should fix.**  
   **Figure:** the dashed \(1\to5\) chord says \(\theta^\star\sim\pi,\ w^\star=1\), after which the visible path continues through panel 6. The source comments disagree over whether it bypasses panels 2–4 or 2–6 ([A8 TeX](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/tikz/a8_algorithm_ring.tex:45)).  
   **Paper/code:** bootstrap uses a prior draw with weight one and does not construct \(q_n\) or evaluate the AMIS denominator.  
   Route the chord directly to dispatch/simulation, bypassing panels 2–6.

8. **Snapshot-buffer notation — minor.**  
   **Figure:** “\(\mathcal S\leftarrow q_n\)” looks like replacement.  
   **Code:** \(q_n\) is pushed into a bounded FIFO ring after the current candidate has been weighted; the oldest entry is evicted when full ([implementation](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py:1177)).  
   Write “push \(q_n\) into \(\mathcal S\) every `amis_interval` calls” and identify gray contours as stored snapshots. The equal-weight online denominator itself is correct, and correctly has no prior floor.

9. **Toy history claim — minor.**  
   The README says every toy particle stores \((\theta,\rho,\epsilon,\tau,w)\), but `run_async` has no \(\tau\) field and stamps bootstrap tolerances as \(+\infty\), whereas the implementation uses `tolerance=None` for bootstrap draws. This does not alter the displayed steady-state panels, but the documentation overstates fidelity.

The remaining A8 elements are sound at Method-level abstraction: the history tuple, top-\(k\) archive formula, adaptation weights \(w_jK_{\epsilon_n}(\rho_j)\), mixture \(q_n\), parent-and-perturb construction, ordinary clockwise arrows, and “one arrival, one update” center text. The fixed illustrated perturbation is a legitimate realized draw and is not clipped in this run.

### A9: textbook ABC-PMC

10. **Panel 1 omits the discrepancies required by panel 2 — should fix.**  
    **Figure:** \(P_t=\{(\theta_i,\omega_i)\}_{i\le N}\), then computes a quantile of \(\rho_i\).  
    **Method:** the population record used for adaptation must include or be associated with \(\rho_i^{(t)}\).  
    Use \(P_t=\{(\theta_i^{(t)},\rho_i^{(t)},\omega_i^{(t)})\}_{i=1}^N\). “Earlier generations discarded” is otherwise a fair characterization of the final-population estimator.

11. **Threshold ellipses are only noiseless toy level sets — minor.**  
    The toy acceptance decision uses noisy simulated discrepancies, whereas the ellipse is generated from the noiseless parameter-space discrepancy. In the illustrated final generation, one accepted candidate lies outside that geometric ellipse because its simulated discrepancy is below threshold. Label the ellipse as a toy/noiseless level set; it is not an exact acceptance region for stochastic ABC.

12. **Panel 5’s “all \(M\) draws before simulation” is incorrect — blocking.**  
    **Figure:** \(\theta_1^\star,\ldots,\theta_M^\star\sim q_t\), “simulate all \(M\),” with \(M=36\) known before acceptance. The Python panel comment explicitly says every candidate is drawn before any is simulated ([A9 panels](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/a9_pmc_panels.py:62)).  
    **Textbook method and toy code:** draw a candidate, simulate it, test it, and continue until the \(N\)-th acceptance; \(M\) is the resulting random total and cannot be known in advance. `run_pmc` itself implements this sequential stopping rule correctly ([`_toy2d.py:181`](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/_toy2d.py:181)).  
    Replace panel 5 with a repeated propose–simulate–accept loop, annotated “\(M\) proposals in total, stopping at \(N\) accepted.”

13. **The barrier is misplaced and overclaimed — blocking.**  
    **Figure:** places a barrier between the already-complete set of \(M\) proposals and acceptance, saying “wait for the slowest.”  
    **Textbook dependency:** the generation boundary is that \(P_{t+1}\) must be complete and weighted before \(q_{t+1}\) can be formed. Candidate evaluations within a generation may be scheduled dynamically; waiting for the slowest of a predetermined set of all eventual proposals is not intrinsic. The paper’s actual pyABC baseline explicitly discards latecomers rather than waiting for them ([§5.2](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/sn-article.tex:207)).  
    Put the dependency marker on the \(P_{t+1}\to\) next-generation transition. Say “next proposal waits for a complete population,” not “simulate all \(M\); wait for the slowest,” unless the figure is explicitly renamed as a fixed-batch barrierized variant.

14. **Panel 6’s weight lacks normalization and generation indices — should fix.**  
    **Figure:** \(\omega_i=\pi(\theta_i)/q_t(\theta_i)\).  
    **Toy/code:** computes these values and then normalizes them ([`_toy2d.py:190`](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/py/_toy2d.py:190)).  
    Use
    \[
    \omega_i^{(t+1)}\propto
    \frac{\pi(\theta_i^{(t+1)})}{q_t(\theta_i^{(t+1))}},
    \qquad \sum_i\omega_i^{(t+1)}=1.
    \]
    The apparent \(q_t\) versus \(q_{t-1}\) discrepancy is only indexing: for a transition \(t\to t+1\), \(q_t\) is the single previous proposal.

15. **The figure is not visibly distinguished from the experimental baseline — blocking if included in this paper.**  
    **Figure:** title says only “generation-staged ABC-PMC.”  
    **Paper:** the primary comparator is pyABC with the same smooth kernel and bandwidth schedule, not this hard-threshold textbook scheme; its scheduling also does not match the depicted fixed barrier.  
    Any caption must begin with something like: “Schematic textbook hard-threshold ABC-PMC for conceptual comparison; not the matched pyABC baseline used in §5.2.”

Panels 3 and 4 are otherwise correct: current-population weights determine \(2\widehat{\mathrm{Cov}}_\omega\), and the new population is proposed from the resulting single previous-population mixture. The ordinary \(1\to4\) arrows, closing label \(P_{t+1}\), and “one generation, one update” center text are conceptually correct after moving the generation dependency to the closing transition.

## 2. Omissions from the real algorithm

The following are legitimate omissions from a Method-level illustration: log-space normalization and underflow fallback; covariance jitter; Cholesky and proposal memoization; capped in-box retries and rare prior fallback; migration-triggered cache resets; scheduler cache/throttling details; approximate truncated-Gaussian normalizers; and chunked bounded-memory posterior extraction. These belong in Appendix C.

Two omissions are not harmless if A8 is the main explanatory figure:

- the distinction between proposal-time and reported posterior weights;
- the fact that dispatch, simulation, and arrival are asynchronous and may occur out of proposal order.

Scheduler cadence and box truncation need at most one caption sentence, provided the toy is no longer called an exact or faithful miniature.

## 3. Reader understanding and redundancy

A8 does help. Its best feature is that the same parameter-space cloud visibly becomes a top-\(k\) archive, weighted covariance, mixture, parent, and candidate. That makes the proposal machinery substantially easier to understand than Algorithm 1 alone.

The most likely first-read errors are:

1. interpreting the dashed \(\epsilon_n\) ellipse as a hard acceptance region;
2. treating \(w^\star\) as the reported posterior weight;
3. reading the ring as a serial simulate-and-return loop;
4. not knowing what point size, gray contours, white points, and dashed ellipses encode.

A corrected A9 could help with positions 1, 3, 5, and 6: history versus population, recomputed versus carried weights, one proposal versus a population, and pre-simulation versus post-acceptance weighting. Position 4 is largely redundant because both methods form a Gaussian mixture. Position 2 currently hides rather than explains the smooth-kernel/hard-threshold distinction.

There is no close duplicate among the paper’s current empirical figures. Among the drafts, however:

- A1 already explains workers, history, the propagator call, and the reported estimator.
- A4 explains the three weights more accurately and explicitly.
- GA3 and GA1 explain stream-versus-generation cadence and barriers more directly.
- Appendix Table `method-comparison` already provides the property-by-property classical comparison.

Thus A8 adds unique geometric value. A9 mostly duplicates GA3/GA1 and the comparison table while being less accurate.

## 4. Ranked improvements

### Essential before publication

1. **Either rebuild A9’s propose–simulate–accept workflow and move the generation dependency, or omit A9.** Explicitly state that it is not the experimental pyABC baseline.

2. **In A8, rename \(w^\star\) “proposal-time core weight—not reported” and add the reported-weight equation as a small branch from the full history.**

3. **Replace A8’s immediate closing arrow with dispatch/in-flight/later-arrival semantics and store \(\tau^\star\), not an ambiguous \(n\).** Route the bootstrap chord directly to dispatch.

4. **Distinguish smooth bandwidth from hard acceptance visually.** A8 should show graded \(K_{\epsilon_n}(\rho)\); reserve a boundary for A9’s hard threshold, with a stochastic-toy caveat.

5. **Redesign at native Springer width.** A8 is \(486.6\) pt wide and A9 \(492.7\) pt; fitting either into 372 pt reduces 8 pt labels to about 6.1 pt and 7 pt notes to about 5.3 pt. The side-by-side wrapper is \(867.3\) pt wide, reducing them to approximately 3.4 and 3.0 pt. That is unusable regardless of raster DPI.

### Strongly recommended

6. Move most equations into a numbered key beneath the graphics. Retain only the discriminating formulas in the panels:

   - A8: \(\tilde W_j^{(n)}\), \(q_n\), the one-draw construction, and \(w^\star\);
   - A9: quantile threshold, \(2\operatorname{Cov}\), \(q_t\), and normalized \(\omega_i^{(t+1)}\).

7. Add a compact visual legend: gray intensity \(=\rho\), blue/orange point size \(=\) mixture/population weight, gray contours \(=\) stored proposals, dashed contour \(=\) toy level set.

8. Increase contrast. Several `black!60` annotations and pale points will become marginal after reduction and in print.

9. Correct the toy covariance/proposal density or soften the provenance claim to “qualitative toy realization.” Quantitative faithfulness is unnecessary for this figure; accurate labeling is necessary.

### Optional simplification

A better main-text form would combine A8’s useful geometry with a small cadence comparison:

- a five-panel horizontal or \(2\times3\) flow: history → archive/weights → proposal → one candidate/pre-weight → worker/in-flight return;
- a side branch from history to the reported estimator;
- underneath, a two-row timeline: “ours: every completion triggers one proposal” versus “textbook PMC: fill \(N\) accepted particles before the next proposal.”

That would replace the entire A9 ring, avoid redrawing an almost identical proposal panel, and fit naturally at 372 pt with 8 pt text.

## 5. Verdicts

**A8:** Include in the main Method section, preferably immediately after the “three weights” paragraph and before Algorithm 1, but only after the blocking weight distinction, asynchronous return, bootstrap path, and width problems are corrected. Its parameter-space sequence offers real explanatory value not present in the current paper figures.

**A9:** Do not include the current ring. After correction, it could serve as an appendix schematic, but the main text would benefit more from a compact cadence inset or comparison table. If retained anywhere, it must be identified as textbook hard-threshold ABC-PMC and explicitly separated from the matched pyABC experimental baseline.

Token usage: not available to me.