# Round proposals (merged)

- round: 02
- date: 2026-10-06

### P-001
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Both the per-simulation ratio and the throughput ratio rise with the cost of a simulation, and their product, the ratio at equal wall clock, exceeds one on Lotka--Volterra and reaches $4.1\times$ on Cellular Potts.
```

**After**
```text
The product of the per-simulation ratio and the throughput ratio, the ratio at equal wall clock, exceeds one on Lotka--Volterra and reaches $4.1\times$ on Cellular Potts.
```

**Rationale**
Removes a claim re-derived one subsection later ("Both ratios increase with simulation cost", 6.3) and keeps 6.2 on the equal-wall-clock conversion it is about; every number and reference is unchanged.

### P-002
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
pyABC is what a practitioner would run instead, and the same experiments score it (Fig.~\ref{fig:barrier-pyabc}, Appendix~\ref{app:moved}).
```

**After**
```text
The same experiments score pyABC (Fig.~\ref{fig:barrier-pyabc}, Appendix~\ref{app:moved}).
```

**Rationale**
The practitioner framing is stated once in 5.2 and is the kept version; the 6.1 paragraph needs only the pointer to the figure, which stays with its references intact.

### P-003
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Assumption~\ref{ass:stab} holds the remaining conditions. For consistency, \eqref{eq:fidelity} is the whole of the sampler's contribution: the sampler need not take any particular form, but whatever it does must be asymptotically reflected in the denominator, up to a ratio $r$ that the results carry.
```

**After**
```text
Assumption~\ref{ass:stab} collects the remaining conditions. For consistency, \eqref{eq:fidelity} is the only one that concerns the sampler: whatever form the sampler takes, its draw mixture must be asymptotically reflected in the denominator, up to the ratio $r$.
```

**Rationale**
Replaces "holds" and "carry" with the specific relations (the assumption collects conditions; the draw mixture is reflected in the denominator) and drops the "the whole of the sampler's contribution" restatement; the condition itself is unchanged.

### P-004
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The present scheme, however, is adaptive at every draw, so the argument runs through a martingale array rather than an i.i.d.\ one, and the available AMIS results do not cover it: the convergence argument of \citet{cornuet2012amis} is heuristic, and \citet{marin2019consistency} restrict the adaptation so that the weighting stage decouples, which a single-arrival scheme does not do.
```

**After**
```text
The present scheme, however, is adaptive at every draw, so the argument runs through a martingale array rather than an i.i.d.\ one. The available AMIS results do not cover it: the convergence argument of \citet{cornuet2012amis} is heuristic, and \citet{marin2019consistency} restrict the adaptation so that the weighting stage decouples.
```

**Rationale**
Splits the clause stack and drops the trailing "which a single-arrival scheme does not do", which the preceding sentence ("adaptive at every draw") already implies and Section 4 already states; both citations stay.

### P-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
On the five Cellular Potts production replicates (about $12{,}900$ asynchronous evaluations each across $48$ workers, two correlated dimensions, so that (c) is live; Appendix~\ref{app:cpm}) both factors are active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**After**
```text
The five Cellular Potts production replicates have about $12{,}900$ asynchronous evaluations each across $48$ workers and two correlated dimensions, so that (c) is live (Appendix~\ref{app:cpm}). Both factors are active there: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**Rationale**
The setting gets its own sentence and the result sentence starts with its subject; all numbers, the reference and the (c) side-condition are preserved.

### P-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The posterior estimate lies $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share, with posterior means within $0.012$ standard deviations, marginal widths within $1\%$ (narrower in nine of ten, the tail-suppressing direction that (a) predicts), and the effective sample size unchanged.
```

**After**
```text
The posterior estimate lies $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share. Posterior means agree within $0.012$ standard deviations and marginal widths within $1\%$, narrower in nine of ten, which is the tail-suppressing direction that (a) predicts; the effective sample size is unchanged.
```

**Rationale**
One result per sentence; the parenthetical interpretation becomes a relative clause and loses nothing.

### P-007
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Of contribution (d), the two events a stored history records, a rejection fallback and an exhausted redraw, occurred neither in the $12{,}000$ draws of that history nor in the $63{,}699$ archive-phase draws of the five Cellular Potts replicates; a redraw that succeeded on a later attempt leaves no trace, so this is a necessary condition for the event, not a bound on its probability.
```

**After**
```text
Of contribution (d), the two events a stored history records, a rejection fallback and an exhausted redraw, occurred neither in the $12{,}000$ draws of that history nor in the $63{,}699$ archive-phase draws of the five Cellular Potts replicates. A redraw that succeeded on a later attempt leaves no trace, so this is a necessary condition for the event, not a bound on its probability.
```

**Rationale**
Splits at the semicolon so the caveat reads as its own statement; the contrast "a necessary condition, not a bound" is scientific and kept verbatim.

### P-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Fitting $\|q_\tau-q_\infty\|_\infty\sim\tau^{-b}$ gives $b=0.50$--$0.71$ per replicate over the full range ($0.59$ pooled) and $b=0.25$--$0.72$ ($0.48$ pooled) when the fit is restricted to the first half, away from the end where using the terminal proposal as a stand-in for $q_\infty$ biases $b$ upward.
```

**After**
```text
Fitting $\|q_\tau-q_\infty\|_\infty\sim\tau^{-b}$ gives $b=0.50$--$0.71$ per replicate over the full range ($0.59$ pooled). Restricted to the first half, away from the end where the terminal proposal standing in for $q_\infty$ biases $b$ upward, the fit gives $b=0.25$--$0.72$ ($0.48$ pooled).
```

**Rationale**
One fit per sentence; the reason for restricting the range stays attached to the fit it explains, and no value changes.

### P-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
A tightening bandwidth does not move it: running a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in, moves $b$ by less than $0.07$, and both runs satisfy $b>\tfrac12$ at $b=0.61$--$0.76$.
```

**After**
```text
A tightening bandwidth does not move it. We ran a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in; the two runs differ in $b$ by less than $0.07$, and both satisfy $b>\tfrac12$ at $b=0.61$--$0.76$.
```

**Rationale**
Puts the experiment in a sentence with a finite verb and the result after it; the pinned-bandwidth description and both numbers are preserved.

### P-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The rule of \S\ref{sec:experiments-metrics}, about a fifth of the prior-predictive median, recovers the posterior, and re-reporting a stored history at the $k$-th order statistic of its own discrepancies, at no simulation cost, reaches a comparable posterior ($93\%/85\%$ on replicate $0$) and is the check we apply.
```

**After**
```text
The rule of \S\ref{sec:experiments-metrics}, about a fifth of the prior-predictive median, recovers the posterior. Re-reporting a stored history at the $k$-th order statistic of its own discrepancies costs no simulations and reaches a comparable posterior ($93\%/85\%$ on replicate $0$); it is the check we apply.
```

**Rationale**
The aside becomes a predicate ("costs no simulations") and each finding gets its own sentence; the reference and the numbers stay.

### P-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The asynchronous posterior-mean error to the analytic posterior stays ${\approx}0.01$ at every coupling, comparable to pyABC, which discards latecomers (Fig.~\ref{fig:param-bias}b), with or without AMIS ($S{=}0$), for the unweighted archive and for the AMIS-reweighted estimator \eqref{eq:posterior-estimator}, and in a steeper regime (base delay $0.5$\,s, cap $8$\,s).
```

**After**
```text
The asynchronous posterior-mean error to the analytic posterior stays ${\approx}0.01$ at every coupling, comparable to pyABC, which discards latecomers (Fig.~\ref{fig:param-bias}b). This holds with or without AMIS ($S{=}0$), for the unweighted archive and for the AMIS-reweighted estimator \eqref{eq:posterior-estimator}, and in a steeper regime (base delay $0.5$\,s, cap $8$\,s).
```

**Rationale**
The result stands alone and the robustness conditions follow in their own sentence; every condition, number and reference is kept.

### P-012
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Contribution (b) is where the estimator's cost meets its fidelity. With $m$ fixed, the $m$-point mixture tracks the $n$-point one only if the proposal path settles, the error being $V_n/m$.
```

**After**
```text
With $m$ fixed, the $m$-point mixture tracks the $n$-point one only if the proposal path settles, the error being $V_n/m$.
```

**Rationale**
The paragraph still makes sense without the opener (test 4), and the cost-versus-fidelity point is made twice elsewhere in the same subsection; the technical sentence that follows is unchanged.

### P-013
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The raw archive shows no measurable mean bias at these couplings; AMIS neither creates nor removes one here, and a correction in the manner of anytime Monte Carlo \citep{murray2021anytime} is future work.
```

**After**
```text
AMIS neither creates nor removes a mean bias here, and a correction in the manner of anytime Monte Carlo \citep{murray2021anytime} is future work.
```

**Rationale**
Removes a restated result so the paragraph ends on the two things not yet said (AMIS is neutral; a correction is future work); the citation stays.

### P-014
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)
- allow-token-change: yes
- token-note: deletes a duplicated caveat; the "6" is "step 6", and the caveat with the step reference stays in 3.5

**Before**
```text
The one exception is a small buffer of past proposals that the running sampler keeps for the weight of step 6; it affects which parents are chosen next, not the estimate.
```

**After**
```text

```

**Rationale**
3.5 is where the kept snapshots are defined and is the natural home of the caveat; the overview paragraph closes on "the same whether it is formed during the run or afterwards", which still holds since the buffer is per-worker and not shared.

### P-015
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The proofs index by \emph{proposal} time, whereas the implementation's post-hoc estimator consumes a rank's \emph{arrival}-ordered log and reconstructs the $q_{\tau_s}$ from prefixes of it; the two agree for a sequential run and need not otherwise, and the discrepancy is one more contributor to $r$.
```

**After**
```text
The implementation's post-hoc estimator, by contrast, consumes a rank's \emph{arrival}-ordered log and reconstructs the $q_{\tau_s}$ from prefixes of it; the two orderings agree for a sequential run and need not otherwise, and the discrepancy is one more contributor to $r$.
```

**Rationale**
Drops the repeated clause and names what "the two" refers to; the mathematical content (the arrival-order discrepancy contributes to $r$) is untouched.

### P-016
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The \emph{per-step drift} $\|q_{\tau+1}-q_\tau\|_\infty$ decays as $\tau^{-0.7}$ to $\tau^{-0.95}$, and the archive's membership changes only $825$--$874$ times in about $12{,}900$ draws, tracking $k\log n$ at a ratio between $0.54$ and $0.92$ across the whole run.
```

**After**
```text
The \emph{per-step drift} $\|q_{\tau+1}-q_\tau\|_\infty$ decays as $\tau^{-0.7}$ to $\tau^{-0.95}$. The archive's membership changes only $825$--$874$ times in about $12{,}900$ draws, tracking $k\log n$ at a ratio between $0.54$ and $0.92$ across the whole run.
```

**Rationale**
One result per sentence, as the round-1 verifiers asked for Appendix A; nothing else changes.

### P-017
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
On an expensive simulator the whole run can be spent inside the bandwidth transient, and a starting value twenty times the prior-predictive median cost most of the contraction on the weakly identified parameter; about a fifth of that median transferred, and re-reporting a stored history at the order statistic of its own discrepancies is the check (\S\ref{sec:results-posterior}).
```

**After**
```text
On an expensive simulator the whole run can be spent inside the bandwidth transient, and a starting value twenty times the prior-predictive median cost most of the contraction on the weakly identified parameter; about a fifth of that median recovered it, and re-reporting a stored history at the order statistic of its own discrepancies is the check (\S\ref{sec:results-posterior}).
```

**Rationale**
Names the relation the result section reports (the rule recovers the contraction) instead of "transferred"; the rule title and reference are unchanged.

### P-018
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The calibration tests of \S\ref{sec:results-posterior} fix $k$ and $S$ per benchmark, which leaves open whether the two efficiency knobs must be retuned as the dimension grows.
```

**After**
```text
The calibration tests of \S\ref{sec:results-posterior} fix $k$ and $S$ per benchmark, which leaves open whether the two settings must be retuned as the dimension grows.
```

**Rationale**
Matches the term used in the Conclusion ("Two settings bound what it delivers") and in Table C.3's caption; the reference is unchanged.

### P-019
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The departure from the ideal is carried as $r$ inside the statements, then decomposed and partly measured.
```

**After**
```text
The departure from the ideal enters the statements as $r$, which is then decomposed and partly measured.
```

**Rationale**
States how $r$ relates to the theorems (it enters them) rather than "is carried"; the decomposed-and-partly-measured claim is unchanged.

### P-020
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Particles are indexed by \emph{proposal} time, not arrival time. This indexing is why the martingale structure survives asynchronous execution: a worker forms its proposal from the history it has received at that moment, which is part of $\mathcal{F}_{i-1}$, so the conditional density $\tilde q_i$ of the emitted candidate is $\mathcal{F}_{i-1}$-measurable and
```

**After**
```text
Particles are indexed by \emph{proposal} time, not arrival time, so the martingale structure survives asynchronous execution: a worker forms its proposal from the history it has received at that moment, which is part of $\mathcal{F}_{i-1}$, so the conditional density $\tilde q_i$ of the emitted candidate is $\mathcal{F}_{i-1}$-measurable and
```

**Rationale**
Removes the cleft and keeps the causal chain and the display equation that completes the sentence exactly as they are.

Another round: no. After these, what remains is the author's own register (dense, number-bound, plain about failures); a further pass would start trading compact sentences for flatter ones, which round 1 showed gets reverted.

### P-021
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-005
- status: rejected
- reason: duplicate of P-005 (Claude version: conservative, per round-1 verification)

**Before**
```text
On the five Cellular Potts production replicates (about $12{,}900$ asynchronous evaluations each across $48$ workers, two correlated dimensions, so that (c) is live; Appendix~\ref{app:cpm}) both factors are active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**After**
```text
The five Cellular Potts production replicates contain about $12{,}900$ asynchronous evaluations each across $48$ workers (Appendix~\ref{app:cpm}). Their two correlated dimensions make (c) live, and both measured factors are active. The bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$. Contribution (a) is $\widehat\zeta=0.011$, and contribution (b) is $0.024$--$0.028$. Together they give $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**Rationale**
This gives the configuration and each result its own sentence. It preserves every number, qualifier, and reference while removing the compressed result chain.

### P-022
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-006
- status: rejected
- reason: duplicate of P-006 (Claude version)

**Before**
```text
The posterior estimate lies $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share, with posterior means within $0.012$ standard deviations, marginal widths within $1\%$ (narrower in nine of ten, the tail-suppressing direction that (a) predicts), and the effective sample size unchanged.
```

**After**
```text
The posterior estimate lies $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share. The posterior means are within $0.012$ standard deviations. The marginal widths are within $1\%$ and are narrower in nine of ten, the tail-suppressing direction that (a) predicts. The effective sample size is unchanged.
```

**Rationale**
Each diagnostic becomes independently scannable. The direction predicted by (a) remains attached to the width result it qualifies.

### P-023
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-016
- status: rejected
- reason: duplicate of P-016 (Claude version)

**Before**
```text
The \emph{per-step drift} $\|q_{\tau+1}-q_\tau\|_\infty$ decays as $\tau^{-0.7}$ to $\tau^{-0.95}$, and the archive's membership changes only $825$--$874$ times in about $12{,}900$ draws, tracking $k\log n$ at a ratio between $0.54$ and $0.92$ across the whole run.
```

**After**
```text
The \emph{per-step drift} $\|q_{\tau+1}-q_\tau\|_\infty$ decays as $\tau^{-0.7}$ to $\tau^{-0.95}$. The archive's membership changes $825$--$874$ times in about $12{,}900$ draws. Across the whole run, this turnover tracks $k\log n$ at a ratio between $0.54$ and $0.92$.
```

**Rationale**
The rewrite reports one measurement per sentence and then states the comparison. It removes only packing, not the interpretation.

### P-024
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-008
- status: rejected
- reason: duplicate of P-008 (Claude version)

**Before**
```text
Fitting $\|q_\tau-q_\infty\|_\infty\sim\tau^{-b}$ gives $b=0.50$--$0.71$ per replicate over the full range ($0.59$ pooled) and $b=0.25$--$0.72$ ($0.48$ pooled) when the fit is restricted to the first half, away from the end where using the terminal proposal as a stand-in for $q_\infty$ biases $b$ upward.
```

**After**
```text
Fitting $\|q_\tau-q_\infty\|_\infty\sim\tau^{-b}$ over the full range gives $b=0.50$--$0.71$ per replicate ($0.59$ pooled). Restricting the fit to the first half gives $b=0.25$--$0.72$ ($0.48$ pooled). This restriction avoids the end, where using the terminal proposal as a stand-in for $q_\infty$ biases $b$ upward.
```

**Rationale**
The two empirical ranges and the reason for the restricted fit become distinct. The upward-bias caveat remains unchanged in force.

### P-025
- file: latex/tmlr/tmlr-article.tex
- category: heading
- severity: M
- source: codex
- status: rejected
- reason: author decision: question headings in Appendix A stay (a heading rename was reverted in round 1 for the same reason)

**Before**
```text
\paragraph{Do the reported runs meet the stabilization conditions?} The proposal sequence is reconstructible from a stored history, so we assess Assumption~\ref{ass:stab}(ii) empirically.
```

**After**
```text
\paragraph{Empirical stabilization diagnostic.} We reconstruct the proposal sequence from each stored history and assess Assumption~\ref{ass:stab}(ii) empirically.
```

**Rationale**
The new heading names the content directly. The sentence also replaces an abstract capability with the concrete analysis performed.

### P-026
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Assumption~\ref{ass:stab}(ii) needs $b>\tfrac12$, at which the aggregate $n^{-1/2}\sum_{i\le n}\|q_i-q_\infty\|_\infty$ vanishes like $n^{1/2-b}$: the production runs sit at that threshold.
```

**After**
```text
Assumption~\ref{ass:stab}(ii) needs $b>\tfrac12$. Under this condition, the aggregate $n^{-1/2}\sum_{i\le n}\|q_i-q_\infty\|_\infty$ vanishes like $n^{1/2-b}$. The production runs sit at that threshold.
```

**Rationale**
The rewrite separates the mathematical condition from the empirical finding. It preserves the deliberately plain statement that the runs sit at the threshold.

### P-027
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$ (which does not by itself give $r\equiv1$, since the other three factors remain), and the proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

**After**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$. This does not by itself give $r\equiv1$, since the other three factors remain. The proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

**Rationale**
The caveat and proof condition become prominent without changing their logic. This is a conservative split of mathematical prose.

### P-028
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The implementation ties them, $\delta=0.5/(m{+}1)$, so growing $m$ there would send $\delta\to0$, and with it the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded} that makes the Lindeberg step trivial and fixes the constants in the law of large numbers; the rate conditions would have to be redone.
```

**After**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$. Growing $m$ would therefore send $\delta\to0$ and remove the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}. That bound makes the Lindeberg step trivial and fixes the constants in the law of large numbers. The rate conditions would have to be redone.
```

**Rationale**
The dependency chain is easier to follow as four short statements. Every logical consequence and the lemma reference remain intact.

### P-029
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
On a one-dimensional Gaussian-mean history where the exact $n$-point mixture is computable, (a) and (b) together give $\widehat\zeta=0.042$ and a total-variation distance of $0.26\%$ from the reference that removes (a) and (b); there too the estimate is $0.5\%$ narrower.
```

**After**
```text
On a one-dimensional Gaussian-mean history, the exact $n$-point mixture is computable. Contributions (a) and (b) together give $\widehat\zeta=0.042$. The total-variation distance from the reference that removes (a) and (b) is $0.26\%$, and the estimate is $0.5\%$ narrower.
```

**Rationale**
The availability of the exact reference is setup, not another result. Separating it makes the two reported discrepancies easier to compare.

### P-030
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-007
- status: rejected
- reason: duplicate of P-007 (Claude version)

**Before**
```text
Of contribution (d), the two events a stored history records, a rejection fallback and an exhausted redraw, occurred neither in the $12{,}000$ draws of that history nor in the $63{,}699$ archive-phase draws of the five Cellular Potts replicates; a redraw that succeeded on a later attempt leaves no trace, so this is a necessary condition for the event, not a bound on its probability.
```

**After**
```text
For contribution (d), a stored history records two events: a rejection fallback and an exhausted redraw. Neither occurred in the $12{,}000$ draws of that history or in the $63{,}699$ archive-phase draws of the five Cellular Potts replicates. A redraw that succeeded on a later attempt leaves no trace. The recorded absence is therefore a necessary condition for the event, not a bound on its probability.
```

**Rationale**
The rewrite separates the observation from its measurement limitation. It keeps the important qualification that the event count is not a probability bound.

### P-031
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The bound \eqref{eq:tilt-bound} is loose against these direct measurements by an order of magnitude on both ($3.6$--$4.0\%$ against $0.2$--$0.5\%$ on Cellular Potts, $4.4\%$ against $0.26\%$ on the one-dimensional history), as sup-norm-free bounds valid for every $f$ generally are.
```

**After**
```text
The bound \eqref{eq:tilt-bound} is loose against both direct measurements by an order of magnitude. On Cellular Potts it gives $3.6$--$4.0\%$ against $0.2$--$0.5\%$; on the one-dimensional history it gives $4.4\%$ against $0.26\%$. Sup-norm-free bounds valid for every $f$ are generally this loose.
```

**Rationale**
Each benchmark receives its own comparison, followed by the interpretation. The rewrite preserves the scope of the explanation and all reported values.

### P-032
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-009
- status: rejected
- reason: duplicate of P-009 (Claude version)

**Before**
```text
A tightening bandwidth does not move it: running a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in, moves $b$ by less than $0.07$, and both runs satisfy $b>\tfrac12$ at $b=0.61$--$0.76$.
```

**After**
```text
A tightening bandwidth does not move it. We ran a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in. The two settings move $b$ by less than $0.07$. Both runs satisfy $b>\tfrac12$ at $b=0.61$--$0.76$.
```

**Rationale**
The setup, comparison, and result become separate statements. The short opening contrast remains scientifically meaningful because the next sentences test it directly.

### P-033
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The measurements above locate the practical knee: on the Gaussian-mean history the posterior estimate is within $3\times10^{-4}$ in total variation of the exact $n$-point mixture by $m=50$, so a $2.5\times$ more expensive post-hoc pass removes factor (b) almost entirely.
```

**After**
```text
On the Gaussian-mean history, the posterior estimate is within $3\times10^{-4}$ in total variation of the exact $n$-point mixture by $m=50$. At that point, a $2.5\times$ more expensive post-hoc pass removes factor (b) almost entirely.
```

**Rationale**
The result and its computational implication remain, but the announced “practical knee” is removed. Cost language is retained because it quantifies an actual computational tradeoff.

### P-034
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The slow worker's recorded evaluation time ($2.000$\,s at $20\times$) plus the $3.6$\,ms per-evaluation overhead measured on the fast workers is the generation time, so the twin cannot exceed $16/2.004=7.98$ simulations per second, and the measured asynchronous rate over that gives $401\times$ against a measured $402\times$; at $5\times$ and $10\times$ the agreement is the same.
```

**After**
```text
The generation time is the slow worker's recorded evaluation time ($2.000$\,s at $20\times$) plus the $3.6$\,ms per-evaluation overhead measured on the fast workers. The twin therefore cannot exceed $16/2.004=7.98$ simulations per second. Dividing the measured asynchronous rate by that ceiling gives $401\times$, against a measured $402\times$. The agreement is the same at $5\times$ and $10\times$.
```

**Rationale**
The calculation is now presented in its actual sequence. No quantity or interpretation changes.

### P-035
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
There is no population to keep consistent and no shared mutable state, so workers never wait for one another: each updates from whatever history it has received, a crashed run is recovered by replaying its log, and the posterior estimate is the same whether it is formed during the run or afterwards.
```

**After**
```text
There is no population to keep consistent and no shared mutable state. Workers therefore never wait for one another. Each updates from whatever history it has received. A crashed run is recovered by replaying its log, and the posterior estimate is the same whether it is formed during the run or afterwards.
```

**Rationale**
The operational properties become individually visible without turning them into a symmetrical list. The causal relation between history reconstruction and barrier-free execution remains explicit.

### P-036
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
On the Gaussian mean the asynchronous estimator is closer to the analytic posterior than pyABC's final population at every checkpoint, and a fairly budgeted rejection sampler is as close; in one dimension with that many draws, rejection is enough.
```

**After**
```text
On the Gaussian mean, the asynchronous estimator is closer to the analytic posterior than pyABC's final population at every checkpoint. A rejection sampler given the same simulation budget is as close. In one dimension with that many draws, rejection is enough.
```

**Rationale**
The replacement specifies what “fairly” means and separates the negative result from the comparison. It preserves the direct conclusion that rejection suffices here.

### P-037
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The evidence points to a reporting-support effect rather than a weighting failure: the top-$k$ archive alone would under-cover ($0.85/0.89$ at the $0.90/0.95$ levels), the full history over-covers, and coverage rises monotonically through nominal as the support grows from one to the other.
```

**After**
```text
The top-$k$ archive alone would under-cover ($0.85/0.89$ at the $0.90/0.95$ levels). The full history over-covers. Coverage rises monotonically through nominal as the support grows from one to the other. This pattern points to a reporting-support effect rather than a weighting failure.
```

**Rationale**
The empirical findings now precede the interpretation they support. This also gives the paragraph the interpretive ending required by the subsection claim.

### P-038
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Every larger archive is dominated by it on \emph{both} axes: $k=200$ is $5\%$ slower and nearly three times worse calibrated ($0.180$), $k=400$ is $14\%$ slower and $0.195$, and $k=800$ costs $31\%$ of throughput and is still slightly worse calibrated than $k=100$ ($0.074$).
```

**After**
```text
Every larger archive is dominated by it on \emph{both} axes. At $k=200$, the sampler is $5\%$ slower and nearly three times worse calibrated ($0.180$). At $k=400$, it is $14\%$ slower with a deviation of $0.195$. At $k=800$, it costs $31\%$ of throughput and remains slightly worse calibrated than $k=100$ ($0.074$).
```

**Rationale**
One archive size is reported per sentence, which makes the non-monotone calibration values easier to inspect. The unfavorable results and both-axis conclusion remain explicit.

I would run one more round after these edits, limited to checking paragraph rhythm and newly adjacent sentence openings; the remaining substantive prose is already precise enough that broader rewriting would risk flattening it.
