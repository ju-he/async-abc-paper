# Round 01 evaluation (Claude reviewer)

File reviewed: latex/tmlr/tmlr-article.tex (read in full, lines 1-984).

## Scores

**AI-likeness: 4/10.** The main text has already shed most of the generated register: sentences are short, every claim is tied to a table or figure, and the negative results (the Gaussian-mean loss, the 4-D ESS erraticism, the rate exponent at the threshold) are stated flat. What remains is a thin but consistent layer of signposting-by-count ("Three things are not routine", "Three findings fixed the setup", "Three comparators appear", "Two configurations ... appear", "Snapshots are used in two places"), a family of vague verbs doing the work of specific relations (carries, sits, binds, lifts, lands, turns on), and a handful of paragraph-closing slogans that restate what the numbers just said.

**Readability: 6/10.** The structure is clear (claims in headings, one claim per subsection, metrics defined before use) and the Results paragraphs are claim-first. Readability is lost to sentence packing: contribution 4 folds two contributions and a relative clause into one list item, the Cellular Potts runtime paragraph chains "determine ... which sets ... and ... grows ... because" through one sentence, and Appendix A's prose paragraphs run to 15-20 lines with several results per sentence.

Passages that most drive the AI-likeness score:

1. Appendix A, "Relation to existing AMIS theory": "Three things are not routine. ... And the departure from the ideal is carried as $r$ inside the statements, then decomposed and partly measured." A counted announcement, a mirrored list, and an "And"-opened punchline.
2. Appendix A, "What $r$ is for this implementation": the same sentence about asynchronous execution and the filtration gap is written out twice within nine lines (the paragraph after the displayed decomposition, and item (d)), each time with "carries the gap" and "a property of the schedule".
3. Appendix F, archive-size paragraph: "The $k$ decision is therefore a floor to clear: go far enough above the small-archive failure region and stop, since further enlargement adds no calibration and costs throughput monotonically." Informal imperative, closing slogan.

## What to leave alone

- Section headings 6.1-6.4 and the paragraph titles inside Results ("The prediction holds", "What the twin does not isolate", "The schedule must move"): they carry the claims and are fixed by the author.
- "The advantage turns on at a few milliseconds per simulation" in the abstract and contribution 3: "turns on" is on the vague-verb list, but it mirrors the fixed heading of 6.3 and the phrase is the paper's own motif; changing it in two places and not the heading would split the terminology.
- The scientific contrasts: "slowest rather than average", "the ceiling its runtime spread sets" aside, "a property of the inference problem rather than of the machine", "containment, not coverage", "a diagnostic rather than an asymptotic statement", "changes the spread, not the mean", "two-dimensional as a property of the model, not for want of screening", "unit peak, not unit integral", "a necessary condition for the event, not a bound on its probability". Each names the alternative a reader would otherwise assume.
- The negative results and limitations as written: the Gaussian-mean loss ("the one on which it is worse is the one-dimensional analytic target"), the 4-D order reversal and "small and erratic" ESS, "bounded, not measured, so the accounting of $r$ is partial", "sit at the threshold of its rate condition rather than clearly inside it", the unattributed duration ratio in Table 8, the $1/W$ flattering that "is not corrected". These are the most human paragraphs in the paper.
- The short fragments "Identifiability does." and "The posterior does not follow." They are the author's short-declarative habit, not a template, and each is immediately cashed out with numbers.
- "Three weightings appear, and only the last is reported" (3.1): counted signposting, but the clause "only the last is reported" is load-bearing disambiguation between $\tilde W_j$, $w^\star$ and $W_{i,n}$ and I would keep it.
- The benchmark-rationale triad in 5.1 ("targets with a reference posterior, so that ...; runtime laws known by construction, so that ...; and a real simulator ..."): parallel, but each clause states a different requirement and the parallelism helps a reader map the five rows of Table 1 to it.
- Appendix B (Proofs) in its entirety: the qualifiers ("must not be read as", "That is the substantive clause", "No deterministic bandwidth floor is required") are mathematical, not rhetorical, and "trivial" is standard usage for the Lindeberg step.
- The Conclusion's return to the opening framing and the Discussion's four italic rule titles, including "Predict the gain before paying for it": the cost language is about compute allocations and is doing work.
- The repetition of the fidelity caveat across abstract, contributions, Section 4, Limitations and Conclusion: each is a one-clause reference, not a re-derivation, which is the pattern the author accepts.
- Captions: the encodings are complete and I propose nothing that would remove one.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: H
- status: proposed
- reason: The sentence on the filtration gap under asynchronous execution is stated twice within the same subsection, once after the displayed decomposition of $r$ and once inside item (d), with near-identical wording.

**Before**
```text
Under asynchronous execution the first factor also carries the gap between the density the propagator emitted and the conditional law of Assumption~\ref{ass:filtration}, so it is partly a property of the schedule.
```

**After**
```text
```

**Rationale**
Same caveat re-derived twice (accepted pattern); item (d) already names "(asynchronously) the filtration gap" in its title and closes with the fuller version, so the overview paragraph loses nothing and the reader meets the point once.

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: "Three things are not routine" announces a count instead of making the contrast, and the third item opens with "And", the punchline template.

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

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- status: proposed
- reason: Closing slogan that restates the paragraph; the same point ("re-reporting at the order statistic is the check") is already made in 6.4 and in the Discussion rule.

**Before**
```text
The estimate is only as good as the bandwidth the schedule reached, and the order-statistic re-report is the check.
```

**After**
```text
```

**Rationale**
Punchline closer plus duplicated caveat; the paragraph ends on the measurement ("brings them to $0.010$--$0.015$"), which is the evidence the slogan summarizes.

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason: The Results preamble two paragraphs earlier already defines pyABC as "the baseline with the same kernel and schedule", and 5.2 states it in full; this closing sentence re-derives the caveat a third time.

**Before**
```text
pyABC runs with the same kernel and bandwidth schedule (\S\ref{sec:experiments-baselines}), so none of this is a kernel or schedule difference.
```

**After**
```text
```

**Rationale**
Same caveat stated in several places (accepted pattern); the paragraph ends on the utilization and simulation-count numbers, which is where the C1 comparison against pyABC should land.

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Contribution 4 packs the theory contribution and the four posterior results into one sentence, with the relative clause "that we decompose and partly measure" separated from "ratio" by a participial phrase.

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

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: A "determine ... which sets ... and grows ... because" chain carrying three mechanisms and a correlation in one sentence.

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

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
- reason: "the ceiling its runtime spread sets, and the ceiling rises" is the sets-a-ceiling idiom repeated within one clause; the specific relation is that the spread predicts the cost.

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

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
- reason: "it binds in the regime the method is designed for" is a vague verb plus a punchline; the specific statement is the one the Discussion makes, that the transient can take the whole run.

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

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- status: proposed
- reason: Informal imperative ("go far enough ... and stop") and the metaphor "a floor to clear" close the paragraph as a slogan.

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

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: "Three comparators appear ... and each keeps one name throughout" narrates the text's own naming discipline; the names are then fixed again in the Results preamble.

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

### C-011
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: "The decisive row is the last" narrates the table and uses an intensifier where the row's content can be stated directly.

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

### C-012
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason: The first two sentences both say that the recompute-from-history rule means workers never wait; the second then lists the consequences.

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

### C-013
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: L
- status: proposed
- reason: "Three findings fixed the setup" announces a count, and the third finding opens with "And", the list-punchline template.

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

### C-014
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- status: proposed
- reason: The paragraph is titled "Two configurations." and its first sentence says the same thing.

**Before**
```text
Two configurations of the benchmark appear in this paper.
```

**After**
```text
```

**Rationale**
Redundant with the paragraph title; the paragraph then opens directly on the production configuration.

### C-015
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: L
- status: proposed
- reason: Signpost sentence; the two uses follow immediately as \emph-titled paragraphs ("The parent weight.", "The posterior weights.").

**Before**
```text
Snapshots are used in two places.
```

**After**
```text
```

**Rationale**
Narrates the text; the titled paragraphs carry the structure, and the preceding sentence ("any snapshot can be rebuilt from the history alone") is the better end to the paragraph.

### C-016
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "reliably and predictably" is a doublet; predictability is C1, reliability is not a measured property here.

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

### C-017
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "comfortably" is an intensifier and "the binding one" a vague verb for the condition that is harder to satisfy.

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

### C-018
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "with room to spare" is informal and "lands between" a vague verb in a sentence that states a quantitative ordering.

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

### C-019
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "What the paper reports is ..." is a cleft.

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

### C-020
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "it lifts" and "the simulator carries real per-evaluation cost" are two vague verbs in one clause.

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

### C-021
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "carries" for "contains" in the appendix opener.

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

### C-022
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "sits ... in total variation from" where the main text (Section 4) says "lies".

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

### C-023
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- status: proposed
- reason: "changes memory, not the value" and "bit-identical to the unchunked computation" say the same thing in one clause.

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

### C-024
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
- reason: "The lower panels show the reason." is a one-sentence figure pointer; joining it to the finding states the reason instead of announcing it.

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

### C-025
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "carries the largest effective sample" uses the vague verb for a plain "has".

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
