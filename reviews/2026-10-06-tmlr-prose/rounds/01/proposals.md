# Round proposals (merged)

- round: 01
- date: 2026-10-06

### P-001
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: H
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)
- allow-token-change: yes
- token-note: deletes a duplicated sentence; \ref{ass:filtration} is still cited in item (d) two paragraphs below The sentence on the filtration gap under asynchronous execution is stated twice within the same subsection, once after the displayed decomposition of $r$ and once inside item (d), with near-identical wording.

**Before**
```text
Under asynchronous execution the first factor also carries the gap between the density the propagator emitted and the conditional law of Assumption~\ref{ass:filtration}, so it is partly a property of the schedule.
```

**After**
```text

```

**Rationale**
Same caveat re-derived twice (accepted pattern); item (d) already names "(asynchronously) the filtration gap" in its title and closes with the fuller version, so the overview paragraph loses nothing and the reader meets the point once.

### P-002
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Three things are not routine. The scheme is adaptive at every draw, so the argument runs through a martingale array rather than an i.i.d.\ one, and the available AMIS results do not cover it: the convergence argument of \citet{cornuet2012amis} is heuristic, and \citet{marin2019consistency} restrict the adaptation so that the weighting stage decouples, which a single-arrival scheme does not do. The prior floor in the \emph{denominator} bounds every weight by $1/\delta$ with no moment or density-ratio condition. This makes the conditional Lindeberg step trivial and lets the method use a local archive proposal at all, for which the usual two-sided ratio bound fails. And the departure from the ideal is carried as $r$ inside the statements, then decomposed and partly measured.
```

**After**
```text
The present scheme, however, is adaptive at every draw, so the argument runs through a martingale array rather than an i.i.d.\ one, and the available AMIS results do not cover it: the convergence argument of \citet{cornuet2012amis} is heuristic, and \citet{marin2019consistency} restrict the adaptation so that the weighting stage decouples, which a single-arrival scheme does not do. The prior floor in the \emph{denominator} bounds every weight by $1/\delta$ with no moment or density-ratio condition. This makes the conditional Lindeberg step trivial and lets the method use a local archive proposal at all, for which the usual two-sided ratio bound fails. The departure from the ideal is carried as $r$ inside the statements, then decomposed and partly measured.
```

**Rationale**
Meta-commentary and a mirrored list; "however" carries the contrast with "the elementary observation" directly, the three points remain in the same order with the same content, and the citations are untouched.

### P-003
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The estimate is only as good as the bandwidth the schedule reached, and the order-statistic re-report is the check.
```

**After**
```text

```

**Rationale**
Punchline closer plus duplicated caveat; the paragraph ends on the measurement ("brings them to $0.010$--$0.015$"), which is the evidence the slogan summarizes.

### P-004
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)
- allow-token-change: yes
- token-note: deletes a duplicated caveat; \ref{sec:experiments-baselines} is still cited in the Results preamble The Results preamble two paragraphs earlier already defines pyABC as "the baseline with the same kernel and schedule", and 5.2 states it in full; this closing sentence re-derives the caveat a third time.

**Before**
```text
pyABC runs with the same kernel and bandwidth schedule (\S\ref{sec:experiments-baselines}), so none of this is a kernel or schedule difference.
```

**After**
```text

```

**Rationale**
Same caveat stated in several places (accepted pattern); the paragraph ends on the utilization and simulation-count numbers, which is where the C1 comparison against pyABC should land.

### P-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
\item Consistency of the estimator up to a ratio, measuring how faithfully its denominator represents the proposals used, that we decompose and partly measure (\S\ref{sec:theory}; proofs and a central limit theorem in the appendix); calibration checks in one and four dimensions and under multimodality, recovery of a two-parameter Cellular Potts posterior, and the two settings that bound the posterior, the starting bandwidth and the archive size (C4, \S\ref{sec:results-posterior}).
```

**After**
```text
\item Consistency of the estimator up to a ratio that measures how faithfully its denominator represents the proposals used; we decompose that ratio and measure part of it (\S\ref{sec:theory}; proofs and a central limit theorem in the appendix). Calibration checks in one and four dimensions and under multimodality, recovery of a two-parameter Cellular Potts posterior, and the two settings that bound the posterior, the starting bandwidth and the archive size (C4, \S\ref{sec:results-posterior}).
```

**Rationale**
Several results packed into one sentence with the main relation buried; two sentences keep every item, every reference and the C4 label.

### P-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The Cellular Potts benchmark is the real simulator, and its runtime spread is a property of the inference problem rather than of the machine: how fast cells divide and how large they grow determine how many cells the lattice holds, which sets both the summaries and the runtime (correlation $0.99$ between runtime and cell count), and the spread grows with the lattice because at $80^3$ the box no longer bounds the cell count.
```

**After**
```text
The Cellular Potts benchmark is the real simulator, and its runtime spread is a property of the inference problem rather than of the machine. How fast cells divide and how large they grow determine how many cells the lattice holds, and the cell count fixes both the summaries and the runtime (correlation $0.99$ between runtime and cell count). The spread grows with the lattice, because at $80^3$ the box no longer bounds the cell count.
```

**Rationale**
"X determines Y which sets Z" chain (accepted pattern); three sentences give one mechanism each and keep the scientific contrast (problem rather than machine) and the correlation.

### P-007
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The barrier's cost on this workload is therefore $1.2\times$, the ceiling its runtime spread sets, and the ceiling rises with the lattice: at $80^3$, where an evaluation costs $170$\,s and the runtime coefficient of variation is $0.23$, the predicted cost is $1.9\times$ and the gap measured against pyABC $2.1\times$.
```

**After**
```text
The barrier's cost on this workload is therefore $1.2\times$, as its runtime spread predicts, and it rises with the lattice: at $80^3$, where an evaluation costs $170$\,s and the runtime coefficient of variation is $0.23$, the predicted cost is $1.9\times$ and the gap measured against pyABC $2.1\times$.
```

**Rationale**
Replaces the "sets a ceiling" metaphor with the relation the paragraph has just established (the utilization ratio matches the prediction); numbers and the pyABC comparison unchanged.

### P-008
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: claude
- status: rejected
- reason: superseded by P-037, which rewrites the same sentence and the "What differs is" cleft before it

**Before**
```text
On a simulator that affords millions of evaluations the transient is negligible, which is why the cheap runs never showed it; it binds in the regime the method is designed for.
```

**After**
```text
On a simulator that affords millions of evaluations the transient is negligible, which is why the cheap runs never showed it; on an expensive simulator, the regime the method is designed for, it can take up the whole run.
```

**Rationale**
Vague verb "binds" replaced by what the preceding sentence measured ($62$ calls per worker without the schedule moving); the claim is the one already made in the Discussion rule, neither strengthened nor softened.

### P-009
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The $k$ decision is therefore a floor to clear: go far enough above the small-archive failure region and stop, since further enlargement adds no calibration and costs throughput monotonically.
```

**After**
```text
The archive size therefore needs only to clear a threshold: above the small-archive failure region, further enlargement adds no calibration and costs throughput monotonically.
```

**Rationale**
Product-like vocabulary and a closing slogan; the After keeps the conclusion (threshold, not trade-off) and the two consequences in formal register.

### P-010
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Three comparators appear, each for a different question, and each keeps one name throughout.
```

**After**
```text
We use three comparators, each for a different question.
```

**Rationale**
Meta-commentary; the three \emph'd names that follow and the Results preamble establish the terminology without an announcement.

### P-011
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The decisive row is the last: because all state is a pure function of the history, there is nothing to synchronize between workers, which fits the method to the asynchronous island model of \S\ref{sec:experiments-baselines}.
```

**After**
```text
Because all state is a pure function of the history (last row), there is nothing to synchronize between workers, which fits the method to the asynchronous island model of \S\ref{sec:experiments-baselines}.
```

**Rationale**
Replaces the narrated pointer with a parenthetical reference; the mechanism and the cross-reference stay.

### P-012
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Every quantity the sampler uses is recomputed from the history it is handed; this rule lets workers run without ever waiting for one another. There is no population to keep consistent and no shared mutable state, so each worker updates from whatever history it has received, a crashed run is recovered by replaying its log, and the posterior estimate is the same whether it is formed during the run or afterwards.
```

**After**
```text
Every quantity the sampler uses is recomputed from the history it is handed. There is no population to keep consistent and no shared mutable state, so workers never wait for one another: each updates from whatever history it has received, a crashed run is recovered by replaying its log, and the posterior estimate is the same whether it is formed during the run or afterwards.
```

**Rationale**
Removes the restated consequence and keeps every listed property; the rule is stated once and its consequences follow it.

### P-013
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Three findings fixed the setup. Of the seven summary blocks the shipped configuration weighted equally, two scalars, the log cell count and the log $95$th-percentile radius, carry nearly all of the signal (signal-to-noise $12$ and $9$ against about $1$ for the four ten-dimensional curve blocks), so the discrepancy is taken over those two, equally weighted. One evaluation is the feature-average of four replicate simulations at the same parameters, because a single realization is too noisy for the discrepancy to order nearby parameters. And the two summaries resolve exactly two directions (how many cells there are, and how large the cluster is at a given count), so the inferred parameters are the two that move them independently: the division rate (count) and the target cell volume (radius at fixed count), whose responses are near-orthogonal ($|\cos|=0.07$).
```

**After**
```text
Of the seven summary blocks the shipped configuration weighted equally, two scalars, the log cell count and the log $95$th-percentile radius, carry nearly all of the signal (signal-to-noise $12$ and $9$ against about $1$ for the four ten-dimensional curve blocks), so the discrepancy is taken over those two, equally weighted. One evaluation is the feature-average of four replicate simulations at the same parameters, because a single realization is too noisy for the discrepancy to order nearby parameters. The two summaries resolve exactly two directions (how many cells there are, and how large the cluster is at a given count), so the inferred parameters are the two that move them independently: the division rate (count) and the target cell volume (radius at fixed count), whose responses are near-orthogonal ($|\cos|=0.07$).
```

**Rationale**
Meta announcement removed and the "And"-opener dropped; the three findings, their numbers and their order are unchanged.

### P-014
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Two configurations of the benchmark appear in this paper.
```

**After**
```text

```

**Rationale**
Redundant with the paragraph title; the paragraph then opens directly on the production configuration.

### P-015
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Snapshots are used in two places.
```

**After**
```text

```

**Rationale**
Narrates the text; the titled paragraphs carry the structure, and the preceding sentence ("any snapshot can be rebuilt from the history alone") is the better end to the paragraph.

### P-016
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Barrier removal delivers simulations reliably and predictably; converting all of them into posterior accuracy is limited by the reporting rule, which we adopted for its simplicity: report at the tightest bandwidth reached, chosen by ESS retention against the archive.
```

**After**
```text
Barrier removal delivers simulations predictably; converting all of them into posterior accuracy is limited by the reporting rule, which we adopted for its simplicity: report at the tightest bandwidth reached, chosen by ESS retention against the archive.
```

**Rationale**
Drops the adverb that no result supports and keeps the one that C1 does.

### P-017
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The proposal path's total variation is therefore $O(\log n)$, and the snapshot clause is comfortably met. The \emph{distance to the limit} is the binding one.
```

**After**
```text
The proposal path's total variation is therefore $O(\log n)$, and the snapshot clause is met. The \emph{distance to the limit} is the harder condition.
```

**Rationale**
Intensifier removed without softening (the $O(\log n)$ rate is the evidence) and the relation named directly; the following sentences supply the exponents.

### P-018
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
It holds with room to spare for a point-identified target under the same top-$k$ rule and bandwidth schedule and fails on a ridge; the Cellular Potts target, which resolves two directions but the second less sharply than the first (Appendix~\ref{app:cpm}), lands between the two.
```

**After**
```text
It holds with a margin for a point-identified target under the same top-$k$ rule and bandwidth schedule and fails on a ridge; the Cellular Potts target, which resolves two directions but the second less sharply than the first (Appendix~\ref{app:cpm}), lies between the two.
```

**Rationale**
Formal register for the same three-way ordering; the exponents quoted in the preceding sentences are untouched.

### P-019
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
What the paper reports is the $m=21$ end, and \S\ref{sec:method-amis} gives its cost.
```

**After**
```text
The paper reports the $m=21$ end; \S\ref{sec:method-amis} gives its cost.
```

**Rationale**
Cleft removed; content and cross-reference unchanged.

### P-020
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
This is the boundary of the asynchronous advantage on a cheap, homogeneous simulator, and it lifts once the simulator carries real per-evaluation cost (Fig.~\ref{fig:scaling}, right).
```

**After**
```text
This is the boundary of the asynchronous advantage on a cheap, homogeneous simulator; it disappears once an evaluation has real cost (Fig.~\ref{fig:scaling}, right).
```

**Rationale**
Names the relation (the boundary disappears) instead of "lifts" and "carries"; the figure reference stays.

### P-021
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
This appendix carries the central limit theorem and its per-run confidence intervals, a bound on how far the tilt by $r$ can move the target, and the decomposition and partial measurement of $r$ on stored runs.
```

**After**
```text
This appendix contains the central limit theorem and its per-run confidence intervals, a bound on how far the tilt by $r$ can move the target, and the decomposition and partial measurement of $r$ on stored runs.
```

**Rationale**
Vague verb replaced by the plain one; the list of contents is unchanged.

### P-022
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The posterior estimate sits $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share, with posterior means within $0.012$ standard deviations, marginal widths within $1\%$ (narrower in nine of ten, the tail-suppressing direction that (a) predicts), and the effective sample size unchanged.
```

**After**
```text
The posterior estimate lies $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share, with posterior means within $0.012$ standard deviations, marginal widths within $1\%$ (narrower in nine of ten, the tail-suppressing direction that (a) predicts), and the effective sample size unchanged.
```

**Rationale**
Aligns the appendix with the verb Section 4 uses for the same number; nothing else changes.

### P-023
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Each particle's denominator is evaluated independently within a single chunk (there is no log-sum-exp reduction \emph{across} chunk boundaries), so chunking changes memory, not the value, and the result is bit-identical to the unchunked computation.
```

**After**
```text
Each particle's denominator is evaluated independently within a single chunk (there is no log-sum-exp reduction \emph{across} chunk boundaries), so the result is bit-identical to the unchunked computation.
```

**Rationale**
Removes the "X, not Y" restatement and keeps the precise claim (bit-identical); the memory statement is made in the preceding sentence.

### P-024
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: rejected
- reason: superseded by P-035, which rewrites the whole paragraph including this pointer

**Before**
```text
The lower panels show the reason. The posterior's effective sample size stays at $238$--$296$ at every checkpoint of the Gaussian run while the history grows from $10^5$ to $1.3\times10^6$ particles, and a controlled sweep puts $\mathrm{ESS}/k$ between $1.9$ and $3.6$ for $k\in\{50,100,200\}$.
```

**After**
```text
The lower panels give the reason: the posterior's effective sample size stays at $238$--$296$ at every checkpoint of the Gaussian run while the history grows from $10^5$ to $1.3\times10^6$ particles, and a controlled sweep puts $\mathrm{ESS}/k$ between $1.9$ and $3.6$ for $k\in\{50,100,200\}$.
```

**Rationale**
Personified-evidence pointer folded into the result; all numbers unchanged.

### P-025
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- source: claude
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The asynchronous estimator is the tightest and carries the largest effective sample ($303$ against pyABC's $66$ of $100$, medians).
```

**After**
```text
The asynchronous estimator is the tightest and has the largest effective sample ($303$ against pyABC's $66$ of $100$, medians).
```

**Rationale**
Vague verb replaced; the comparison and its numbers stay.

Another round: yes, one more and then stop. The items above are scattered single-sentence fixes rather than a structural problem, and what remains after them is a short list of vague verbs the author may want to decide on as a policy (the remaining "binds" in 6.4 and Appendix F, "sits at the threshold" in Limitations and Appendix A, "turns on" in the abstract), plus the long paragraphs of Appendix A, which a second pass could split once the duplications here are gone.

### P-026
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-005
- status: rejected
- reason: duplicate of P-005 (Claude version preferred: two sentences, list-item register kept)

**Before**
```text
\item Consistency of the estimator up to a ratio, measuring how faithfully its denominator represents the proposals used, that we decompose and partly measure (\S\ref{sec:theory}; proofs and a central limit theorem in the appendix); calibration checks in one and four dimensions and under multimodality, recovery of a two-parameter Cellular Potts posterior, and the two settings that bound the posterior, the starting bandwidth and the archive size (C4, \S\ref{sec:results-posterior}).
```

**After**
```text
\item Consistency of the estimator up to a ratio that measures how faithfully its denominator represents the proposals used; we decompose and partly measure that ratio (\S\ref{sec:theory}; proofs and a central limit theorem in the appendix). We test calibration in one and four dimensions and under multimodality, and recover a two-parameter Cellular Potts posterior. We also identify the two settings that bound the posterior: the starting bandwidth and the archive size (C4, \S\ref{sec:results-posterior}).
```

**Rationale**
The revision separates the theoretical result, empirical validation, and limitations without changing their scope. It removes a sentence whose main clause is buried beneath two lists.

### P-027
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
With the starting bandwidth set from the prior-predictive discrepancies, the method recovers a two-parameter tissue posterior tighter than pyABC's on a larger effective sample, and that effective sample size, a few multiples of the archive size, is the remaining limit.
```

**After**
```text
With the starting bandwidth set from the prior-predictive discrepancies, the method recovers a two-parameter tissue posterior tighter than pyABC's on a larger effective sample. That effective sample size remains limited to a few multiples of the archive size.
```

**Rationale**
The negative result deserves its own direct sentence. The revision preserves both the comparison with pyABC and the stated ESS limitation.

### P-028
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The claims need three kinds of benchmark: targets with a reference posterior, so that calibration and recovery can be scored; runtime laws known by construction, so that the barrier prediction can be tested exactly; and a real simulator of the class described in \S\ref{sec:background}, where the method is meant to be used. Table~\ref{tab:benchmarks} lists the five configurations.
```

**After**
```text
We evaluate calibration and recovery on targets with reference posteriors, test the barrier prediction under runtime laws known by construction, and measure performance on a real simulator of the class described in \S\ref{sec:background}. Table~\ref{tab:benchmarks} lists the five configurations.
```

**Rationale**
This states what each benchmark class measures instead of narrating the construction of the argument. The three experimental roles remain intact.

### P-029
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-006
- status: rejected
- reason: duplicate of P-006 (Claude version preferred: keeps "how fast cells divide and how large they grow")

**Before**
```text
The Cellular Potts benchmark is the real simulator, and its runtime spread is a property of the inference problem rather than of the machine: how fast cells divide and how large they grow determine how many cells the lattice holds, which sets both the summaries and the runtime (correlation $0.99$ between runtime and cell count), and the spread grows with the lattice because at $80^3$ the box no longer bounds the cell count.
```

**After**
```text
The Cellular Potts benchmark is the real simulator. Its runtime spread arises from the inference problem rather than the machine. The division rate and cell size determine how many cells the lattice holds, and the cell count determines both the summaries and the runtime (correlation $0.99$ between runtime and cell count). The runtime spread grows with the lattice because at $80^3$ the box no longer bounds the cell count.
```

**Rationale**
The causal chain becomes explicit rather than being nested inside one sentence. All mechanisms and the reported correlation are preserved.

### P-030
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Three weightings appear, and only the last is reported: the \emph{archive weights} $\tilde W_j$ shape the next proposal, the \emph{parent weight} $w^\star$ stored with each particle is used only inside the running sampler, and the \emph{posterior weights} $W_{i,n}$ are computed afterwards over the whole history. Sections~\ref{sec:method-history} to \ref{sec:method-weights} follow the figure's steps; \S\ref{sec:method-algorithm} assembles them.
```

**After**
```text
The \emph{archive weights} $\tilde W_j$ shape the next proposal. The \emph{parent weight} $w^\star$ stored with each particle is used only inside the running sampler. Only the \emph{posterior weights} $W_{i,n}$, computed afterwards over the whole history, are reported. Sections~\ref{sec:method-history} to \ref{sec:method-weights} follow the figure's steps; \S\ref{sec:method-algorithm} assembles them.
```

**Rationale**
The revision keeps the necessary distinction among all three weights but removes the conspicuous numbered-list template. Short declarative sentences also make the different roles easier to retrieve.

### P-031
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The estimate $\widehat\pi_n$ of \eqref{eq:posterior-estimator} divides each particle's weight by $\bar q_n$, a fixed-size mixture of reconstructed proposals, whereas the particles were drawn from the sequence of all proposals the run used. The question is whether the denominator is the density the sample was drawn from.
```

**After**
```text
The estimate $\widehat\pi_n$ of \eqref{eq:posterior-estimator} divides each particle's weight by $\bar q_n$, a fixed-size mixture of reconstructed proposals, whereas the particles were drawn from the sequence of all proposals the run used. Consistency therefore depends on how well $\bar q_n$ represents the density from which the pooled sample was drawn.
```

**Rationale**
The replacement states the mathematical dependency rather than narrating the question posed by the section. It preserves the substantive mismatch that motivates the theorem.

### P-032
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: codex
- duplicate-of: P-010
- status: rejected
- reason: duplicate of P-010 (replacement preferred over deletion: keeps "each for a different question")

**Before**
```text
Three comparators appear, each for a different question, and each keeps one name throughout.
```

**After**
```text

```

**Rationale**
The pyABC, twin, and rejection-ABC definitions are self-contained without this announcement. Deleting it removes redundant meta-commentary.

### P-033
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: codex
- status: rejected
- reason: author decision: "the method a practitioner would run" is the baseline's stated role, kept

**Before**
```text
\emph{pyABC} is pyABC \citep{klinger2018pyabc,schalte2022pyabc} with its acceptor replaced by a probabilistic-rejection acceptor using the same smooth kernel and the same bandwidth schedule as the asynchronous sampler (Appendix~\ref{app:implementation}); it is the method a practitioner would run, and the comparator for C2 and C3.
```

**After**
```text
\emph{pyABC} is pyABC \citep{klinger2018pyabc,schalte2022pyabc} with its acceptor replaced by a probabilistic-rejection acceptor using the same smooth kernel and the same bandwidth schedule as the asynchronous sampler (Appendix~\ref{app:implementation}). It is the comparator for C2 and C3.
```

**Rationale**
“The method a practitioner would run” sounds product-oriented and adds no experimental definition. The replacement retains the matched configuration and its exact role.

### P-034
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The asynchronous sampler pays a roughly fixed cost per arrival: it rebuilds the $\mathcal{O}(k)$ proposal, weights the arrival against the kept snapshots, and exchanges the arrival with every other worker. Set against a growing per-simulation cost, that fixed cost determines where the method becomes advantageous. Figure~\ref{fig:crossover} plots the throughput and per-simulation ratios of Table~\ref{tab:matched-eps} against the cost of one simulation. Both rise with it, but only the throughput ratio has a mechanism; the per-simulation ratio is a statistical property of four inference problems that differ in dimension, summaries and noise, so the rule rests on the throughput ratio alone. Throughput parity is reached at about $4$\,ms per simulation, and the net advantage at equal wall clock turns positive between $2$ and $4$\,ms.
```

**After**
```text
Each arrival incurs a roughly fixed coordination cost: rebuilding the $\mathcal{O}(k)$ proposal, weighting the arrival against the kept snapshots, and exchanging it with every other worker. This cost becomes less important as the simulation cost increases. Figure~\ref{fig:crossover} plots the throughput and per-simulation ratios of Table~\ref{tab:matched-eps} against the cost of one simulation. Both ratios increase with simulation cost. Only the throughput ratio has a common mechanism across benchmarks; the per-simulation ratio also depends on dimension, summaries and noise in the four inference problems. We therefore base the boundary on throughput: parity is reached at about $4$\,ms per simulation, and the net advantage at equal wall clock turns positive between $2$ and $4$\,ms.
```

**Rationale**
The revision separates the systems mechanism from problem-specific statistical efficiency. It preserves the measured crossover and the reason the rule uses throughput.

### P-035
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-024
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The lower panels show the reason. The posterior's effective sample size stays at $238$--$296$ at every checkpoint of the Gaussian run while the history grows from $10^5$ to $1.3\times10^6$ particles, and a controlled sweep puts $\mathrm{ESS}/k$ between $1.9$ and $3.6$ for $k\in\{50,100,200\}$. The estimate is reported at the tightest bandwidth reached, and that bandwidth is chosen by ESS retention against the archive (\S\ref{sec:method-tolerance}), so the effective support stays at a few multiples of $k$ however many particles are evaluated. In one dimension that support is small and stable; in four it is small and erratic ($7$ to $2987$ across checkpoints and replicates), and accuracy follows it. The four-dimensional deficit is consistent with intermittent weight degeneracy rather than bias, and the ceiling binds when the target is identified well enough for the bandwidth to tighten.
```

**After**
```text
The effective sample sizes explain this reversal. The posterior's effective sample size stays at $238$--$296$ at every checkpoint of the Gaussian run while the history grows from $10^5$ to $1.3\times10^6$ particles. A controlled sweep puts $\mathrm{ESS}/k$ between $1.9$ and $3.6$ for $k\in\{50,100,200\}$. The estimate is reported at the tightest bandwidth reached, which is chosen by ESS retention against the archive (\S\ref{sec:method-tolerance}). Its effective support therefore stays at a few multiples of $k$ however many particles are evaluated. In one dimension that support is small and stable; in four it is small and erratic ($7$ to $2987$ across checkpoints and replicates), and accuracy follows it. The four-dimensional deficit is consistent with intermittent weight degeneracy rather than bias. The limitation appears once the target is identified well enough for the bandwidth to tighten.
```

**Rationale**
The evidence–interpretation relation is stated directly, and the numerical result no longer shares a sentence with the sweep. The unfavorable four-dimensional result remains explicit.

### P-036
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The cell volume separates them: $81\%\pm1\%$ for the asynchronous estimator, $72\%\pm4\%$ for pyABC's last generation with its population weights, and $67\%$ for the best $100$ rejection draws.
```

**After**
```text
For cell volume, contraction is $81\%\pm1\%$ for the asynchronous estimator, $72\%\pm4\%$ for pyABC's last generation with its population weights, and $67\%$ for the best $100$ rejection draws.
```

**Rationale**
The replacement names the measured relation immediately. It preserves every method, number, and weighting qualification.

### P-037
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-008
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
What differs is the bandwidth the estimate is reported at: after the run's $62$ calls per worker the schedule had not moved from $\epsilon_0$, which ended $321\times$ above the tolerance the draws supported, against $2\times$ at $\epsilon_0=0.1$. On a simulator that affords millions of evaluations the transient is negligible, which is why the cheap runs never showed it; it binds in the regime the method is designed for.
```

**After**
```text
The runs differ in the bandwidth used for reporting. After the run's $62$ calls per worker, the schedule had not moved from $\epsilon_0$, which ended $321\times$ above the tolerance the draws supported, against $2\times$ at $\epsilon_0=0.1$. The transient is negligible on a simulator that affords millions of evaluations, which is why the cheap runs never showed it. It remains consequential in the regime the method is designed for.
```

**Rationale**
The revision foregrounds the actual variable and separates the cheap-simulator explanation from the expensive-simulator limitation. No interpretation is softened.

### P-038
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
The effective sample size is a few multiples of $k$, but raising $k$ has costs: re-reporting one history at the order statistic for $k$ from $10$ to $1000$ loosens the final bandwidth and lowers the cell-volume contraction from $85\%$ to $78\%$, running with $k=300$ leaves the schedule at $\epsilon_0$ for the whole hour, and on the cheap benchmarks $k=100$ is Pareto-optimal for throughput and calibration (Appendix~\ref{app:moved}).
```

**After**
```text
The effective sample size is a few multiples of $k$, but raising $k$ has costs. Re-reporting one history at the order statistic for $k$ from $10$ to $1000$ loosens the final bandwidth and lowers the cell-volume contraction from $85\%$ to $78\%$. Running with $k=300$ leaves the schedule at $\epsilon_0$ for the whole hour. On the cheap benchmarks, $k=100$ is Pareto-optimal for throughput and calibration (Appendix~\ref{app:moved}).
```

**Rationale**
Each experiment now receives its own sentence, so the reader can distinguish post-hoc re-reporting, an actual run, and the cross-benchmark frontier. The negative outcomes remain unchanged.

### P-039
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: codex
- status: rejected
- reason: author decision: the Discussion rule titles stay as imperatives (Claude reviewer: leave alone)

**Before**
```text
\emph{Use it above a few milliseconds per simulation.} Below that the per-arrival coordination costs more than a barrier does, and a generation-based sampler with a population sized to the machine is the better tool; above it, removing the barrier recovers a cost that grows with the worker count and with the spread of simulation times (\S\ref{sec:results-boundary}).
```

**After**
```text
\emph{Applicability threshold.} Below a few milliseconds per simulation, the per-arrival coordination costs more than a barrier does, and a generation-based sampler with a population sized to the machine is the better tool. Above that threshold, removing the barrier recovers a cost that grows with the worker count and with the spread of simulation times (\S\ref{sec:results-boundary}).
```

**Rationale**
The revision retains the practical recommendation while giving it a technical heading. Splitting the scientific contrast makes both regimes explicit.

### P-040
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: codex
- status: rejected
- reason: author decision: the Discussion rule titles stay; the cost language is about compute allocations

**Before**
```text
\emph{Predict the gain before paying for it.} The barrier's cost is $W/\mathbb{E}[\max_{i\le W}D_i]$ against the asynchronous throughput, computable from the run's own per-evaluation timing, with two cautions: the timing sample must be longer than the tail of the runtime distribution, and a persistently slow worker contributes its own evaluation time rather than its wall-clock span (\S\ref{sec:results-barrier}).
```

**After**
```text
\emph{Estimate the barrier cost from run timings.} The barrier's cost is $W/\mathbb{E}[\max_{i\le W}D_i]$ against the asynchronous throughput and is computable from the run's own per-evaluation timing. The timing sample must be longer than the tail of the runtime distribution, and a persistently slow worker contributes its own evaluation time rather than its wall-clock span (\S\ref{sec:results-barrier}).
```

**Rationale**
The heading now states the operation rather than advertising its value through a payment metaphor. The two cautions are retained and made easier to locate.

### P-041
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: H
- source: codex
- duplicate-of: P-016
- status: rejected
- reason: duplicate of P-016 (minimal edit preferred; the codex rewrite restates the limit in stronger terms)

**Before**
```text
The effective-sample-size ceiling is where the method can still improve. Barrier removal delivers simulations reliably and predictably; converting all of them into posterior accuracy is limited by the reporting rule, which we adopted for its simplicity: report at the tightest bandwidth reached, chosen by ESS retention against the archive. Further work on this method belongs in that rule rather than in the scheduler.
```

**After**
```text
The reporting rule is the main remaining limit on effective sample size. Barrier removal reliably and predictably increases the number of simulations, but reporting at the tightest bandwidth reached, chosen by ESS retention against the archive, prevents all of them from contributing to posterior accuracy. Further work should therefore focus on the reporting rule rather than the scheduler.
```

**Rationale**
The revision preserves the interpretation and research direction without presenting them as successive punchlines. It also replaces the announcement of “simplicity” with the actual reporting rule.

### P-042
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Consistency holds up to the fidelity ratio $r$, of which two factors are measured and two bounded, with the adapted filtration under asynchronous execution assumed (\S\ref{sec:theory}); the central limit theorem is proved for damped-adaptation variants, and the production runs sit at the threshold of its rate condition rather than clearly inside it (Appendix~\ref{app:theory-more}).
```

**After**
```text
Consistency holds up to the fidelity ratio $r$, of which two factors are measured and two bounded. The adapted filtration under asynchronous execution is assumed (\S\ref{sec:theory}). The central limit theorem is proved for damped-adaptation variants, and the production runs sit at the threshold of its rate condition rather than clearly inside it (Appendix~\ref{app:theory-more}).
```

**Rationale**
Each limitation becomes independently visible, which is important for review. The revision neither strengthens nor softens any theoretical claim.

### P-043
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: codex
- duplicate-of: P-002
- status: rejected
- reason: duplicate of P-002 (P-002 covers the announcement, the "And" closer and the contrast)

**Before**
```text
Three things are not routine.
```

**After**
```text

```

**Rationale**
The following sentences identify each departure from existing AMIS theory without this announcement. Deletion removes a conspicuous rhetorical template and no mathematical content.

### P-044
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: codex
- duplicate-of: P-002
- status: rejected
- reason: duplicate of P-002 (P-002 covers this passage; "trivial" is standard for the Lindeberg step)

**Before**
```text
The prior floor in the \emph{denominator} bounds every weight by $1/\delta$ with no moment or density-ratio condition. This makes the conditional Lindeberg step trivial and lets the method use a local archive proposal at all, for which the usual two-sided ratio bound fails. And the departure from the ideal is carried as $r$ inside the statements, then decomposed and partly measured.
```

**After**
```text
The prior floor in the \emph{denominator} bounds every weight by $1/\delta$ with no moment or density-ratio condition. This bound supplies the conditional Lindeberg condition and permits a local archive proposal, for which the usual two-sided ratio bound fails. The statements carry the departure from the ideal through $r$, which is then decomposed and partly measured.
```

**Rationale**
The revision states what the bound establishes without announcing the calculation’s ease. It also removes the staged “And” closer while preserving the role of $r$.

### P-045
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Everything specific to the code enters through $r$. To attribute it, walk from the density actually sampled from down to the function actually divided by, changing one thing at a time.
```

**After**
```text
We decompose $r$ by inserting intermediate mixtures between the density actually sampled from and the function used in the denominator, changing one implementation feature at a time.
```

**Rationale**
The replacement states the mathematical construction directly. It removes both the abstract “everything enters” phrasing and the instructional metaphor.

### P-046
- file: latex/tmlr/tmlr-article.tex
- category: heading
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
\paragraph{Measured, in part.} Two of the four factors, the prior floor and the snapshot quadrature, can be estimated rather than bounded, because $\widehat\pi_n$ is a post-hoc pass: recomputing \eqref{eq:snapshot-denominator} at a much larger $m$ and at the observed prior share gives a reference against which those factors are evaluated at every particle.
```

**After**
```text
\paragraph{Empirical estimates of the prior-floor and snapshot-quadrature contributions.} These two contributions can be estimated rather than bounded because $\widehat\pi_n$ is a post-hoc pass. Recomputing \eqref{eq:snapshot-denominator} at a much larger $m$ and at the observed prior share gives a reference against which they are evaluated at every particle.
```

**Rationale**
The new heading identifies the measured quantities. The prose then explains the procedure without re-announcing the entire decomposition.

### P-047
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Assumption~\ref{ass:stab}(ii) is the one place where we can do better than assume, because the proposal sequence is reconstructible from a stored history. Rebuilding $q_\tau$ along the five Cellular Potts production replicates and comparing against each run's terminal proposal separates two rates that behave differently.
```

**After**
```text
The proposal sequence is reconstructible from a stored history, so we assess Assumption~\ref{ass:stab}(ii) empirically. Rebuilding $q_\tau$ along the five Cellular Potts production replicates and comparing against each run's terminal proposal separates two rates that behave differently.
```

**Rationale**
The replacement states the evidential basis and the action taken. It avoids commentary about how this part of the paper compares with its other assumptions.

### P-048
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- source: codex
- status: rejected
- reason: author decision (2026-09-22 humanizer pass): "Identifiability does." kept deliberately

**Before**
```text
Identifiability does. Re-running the serial benchmark with its two parameters confounded (observing only $\theta_1+\theta_2$, so the target is a ridge rather than a point) collapses the exponent to $b=0.00$--$0.29$: on a ridge the top-$k$ archive slides along the flat direction indefinitely, so the proposal never settles.
```

**After**
```text
Identifiability affects the stabilization rate. Re-running the serial benchmark with its two parameters confounded (observing only $\theta_1+\theta_2$, so the target is a ridge rather than a point) collapses the exponent to $b=0.00$--$0.29$: on a ridge the top-$k$ archive slides along the flat direction indefinitely, so the proposal never settles.
```

**Rationale**
The revision names the affected quantity rather than relying on the preceding paragraph to complete “does.” The negative exponent result and its mechanism remain unchanged.

### P-049
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: L
- source: codex
- status: applied
- reason: batch triage 2026-10-06 (all items; user choice)

**Before**
```text
Decoupling the two is a one-line change to the post-hoc pass (Appendix~\ref{app:theory}, Remark~\ref{rem:growing-m}); the retroactive pass then costs $\mathcal{O}(n\,m_n\,k)$ instead of $\mathcal{O}(nk)$, and the literal cumulative mixture is the endpoint $m_n=n$ at $\mathcal{O}(n^2k)$.
```

**After**
```text
The post-hoc pass can decouple the two (Appendix~\ref{app:theory}, Remark~\ref{rem:growing-m}). The retroactive pass then costs $\mathcal{O}(n\,m_n\,k)$ instead of $\mathcal{O}(nk)$, and the literal cumulative mixture is the endpoint $m_n=n$ at $\mathcal{O}(n^2k)$.
```

**Rationale**
The replacement retains the implementation option and both complexity statements. It removes the unsupported emphasis on how easy the change is.

### P-050
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- source: codex
- status: rejected
- reason: author decision (2026-09-22 humanizer pass): "The posterior does not follow." kept deliberately

**Before**
```text
The posterior does not follow.
```

**After**
```text

```

**Rationale**
Deleting the slogan removes personification and a redundant paragraph opener. The following sentence supplies the exact posterior result and comparisons.

I would run another round after these changes because the appendices still contain lower-value clause stacking and repeated interpretation, but the dominant AI-like templates would first need to be removed so the remaining issues can be judged in context.
