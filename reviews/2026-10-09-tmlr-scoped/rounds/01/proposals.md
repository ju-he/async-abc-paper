# Round proposals (merged)

- round: 
- date: 2026-10-09

### P-001
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: claude
- status: applied
- reason: both reviewers proposed it; Claude's wording (batch rule)

**Before**
```text
The event concerns only the bandwidth at which the estimate is reported, $\epsilon_n$ in \eqref{eq:posterior-estimator}: the bandwidths the sampler proposes with enter the proofs only through $\tilde q_i$, on which Theorem~\ref{thm:consistency} places no condition.
```

**After**
```text
The event concerns only the bandwidth $\epsilon_n$ at which the estimate is reported in \eqref{eq:posterior-estimator}; the bandwidths the sampler uses to build its proposals enter the proofs only through $\tilde q_i$, on which Theorem~\ref{thm:consistency} places no condition.
```

**Rationale**
"the bandwidths the sampler proposes with enter the proofs" is a garden-path sentence: "proposes with enter" reads as a typo on first pass. The After states the same distinction (reporting bandwidth versus proposal bandwidths) with the same reference and no change of claim.

### P-002
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: H
- source: claude
- status: applied
- reason: both reviewers proposed it; Claude's wording (batch rule)

**Before**
```text
The implementation also offers quantile, geometric-decay and acceptance-rate rules (\texttt{scheduler\_type}); they act only under the hard kernel, and with a smooth kernel every choice reduces to the rule above, so the option has no effect in any reported run. That search runs on a rank's first call, then once per \texttt{bisect\_interval} ($=k$) of its own calls, and again whenever the rank's cached history is rebuilt after an out-of-order arrival; asynchronous arrivals cause such rebuilds and lockstepped ones do not, which is why the barrierized twin's bandwidth tightens less often than the asynchronous sampler's at equal call counts (\S\ref{sec:results-barrier}).
```

**After**
```text
The search runs on a rank's first call, then once per \texttt{bisect\_interval} ($=k$) of its own calls, and again whenever the rank's cached history is rebuilt after an out-of-order arrival; asynchronous arrivals cause such rebuilds and lockstepped ones do not, which is why the barrierized twin's bandwidth tightens less often than the asynchronous sampler's at equal call counts (\S\ref{sec:results-barrier}). The implementation also offers quantile, geometric-decay and acceptance-rate rules (\texttt{scheduler\_type}); they act only under the hard kernel, and with a smooth kernel each reduces to the rule above, so the option has no effect in any reported run.
```

**Rationale**
The newly inserted sentence about the alternative rules sits between the ESS search and "That search runs ...", so "That search" now points at the alternative rules. Moving the aside to the end restores the antecedent and keeps every clause, including the hard-kernel restriction.

### P-003
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: both reviewers proposed it; Claude's wording (batch rule)

**Before**
```text
A floor of $10^{-4}\epsilon_0$ guarantees this without binding on the g-and-k and Cellular Potts runs, but not on the two near-instantaneous benchmarks; and even where the limits apply they describe the run's continuation, since no reported run reached its floor (\S\ref{sec:theory}).
```

**After**
```text
A floor of $10^{-4}\epsilon_0$ guarantees this; it would have clipped the two near-instantaneous benchmarks but none of the g-and-k and Cellular Potts runs. Even where the limits apply, they describe the run's continuation, since no reported run reached its floor (\S\ref{sec:theory}).
```

**Rationale**
"guarantees this without binding on X, but not on Y" lets "not" attach to "guarantees", which suggests the floor fails to guarantee the event on Y. That reverses the meaning. The After states which runs the floor would clip and splits the three-part semicolon chain, keeping the limitation and the continuation caveat intact.

### P-004
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: rejected
- reason: single reviewer, severity below H (batch rule); author may promote

**Before**
```text
The bandwidth a finite budget reaches is not an asymptotic $\epsilon_\infty$: no run reached its floor, and the effective-sample-size ceiling (\S\ref{sec:theory}) is the consequence.
```

**After**
```text
No run reached its floor, so the limits describe each run's continuation (\S\ref{sec:theory}).
```

**Rationale**
The fixed-$k$ / ESS-ceiling argument is derived in full in Section 4 and restated in Section 8. The status table needs only the verdict and the pointer. The After keeps the fact (no run reached its floor) and the consequence for the limit statements without a third derivation of the ceiling.

### P-005
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- source: claude
- status: rejected
- reason: single reviewer, severity below H (batch rule); author may promote

**Before**
```text
The event concerns the reporting bandwidth only; the sampler's own bandwidths enter through $\tilde q_i$, on which no condition is placed. A strictly positive
```

**After**
```text
A strictly positive
```

**Rationale**
This repeats the second sentence of Section 4's "Which runs the limits describe" nearly word for word. The cell's remaining sentences (the clip argument and the per-benchmark bandwidths) are the status evidence the table exists for. Drop this if the author wants the table rows fully self-contained.

### P-006
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: claude
- status: applied
- reason: both reviewers proposed it; Claude's wording (batch rule)

**Before**
```text
The argument has four steps. The prior floor bounds every weight (Lemma~\ref{lem:bounded}).
```

**After**
```text
The prior floor bounds every weight (Lemma~\ref{lem:bounded}).
```

**Rationale**
This is a counted announcement followed by exactly four parallel sentences, a typical generated scaffold. The four sentences already read as a roadmap without the count, and the lemma references carry the structure.

### P-007
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- source: claude
- status: applied
- reason: both reviewers proposed it; Claude's wording (batch rule)

**Before**
```text
This lets the method use a local archive proposal at all, for which the usual two-sided ratio bound fails. The departure from the ideal enters the statements as $r$, which is then decomposed and partly measured.
```

**After**
```text
This lets the method use a local archive proposal at all, for which the usual two-sided ratio bound fails.
```

**Rationale**
The closer restates the appendix's first sentence ("decomposes $r$ and measures part of it") as a paragraph-ending summary, and the next subsection opens by doing exactly that. The paragraph ends naturally on the weight-bound point, which is its content. "at all" is kept.

### P-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: rejected
- reason: single reviewer, severity below H (batch rule); author may promote

**Before**
```text
A floor at $10^{-4}\epsilon_0$ puts the run under the limits of \S\ref{sec:theory} and never binds on the g-and-k and Cellular Potts runs.
```

**After**
```text
A floor at $10^{-4}\epsilon_0$ brings the run within the limit statements of \S\ref{sec:theory} and would not have bound on any g-and-k or Cellular Potts run reported here.
```

**Rationale**
"puts the run under the limits" can be read as "keeps it below some limits". Also, "never binds" states as a general fact what is a counterfactual about runs that had no floor set. The After is exact on both points and claims no more or less than Table~\ref{tab:assumption-status}.

### P-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: rejected
- reason: single reviewer, severity below H (batch rule); author may promote

**Before**
```text
The following lemma supplies a filtration from the execution model that adds the durations, the worker identities and the delivery randomness, and under which the emitted density is the conditional law of the candidate.
```

**After**
```text
The following lemma derives from the execution model a filtration that adds the durations, the worker identities and the delivery randomness, and under which the emitted density is the conditional law of the candidate.
```

**Rationale**
In the Before, "that adds the durations" first attaches to "the execution model". Placing "from the execution model" before the object makes the filtration the subject of both relative clauses, and "derives" matches the wording of Assumption~\ref{ass:filtration}.

### P-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: applied
- reason: both reviewers proposed it; Claude's wording (batch rule)

**Before**
```text
In the same hour the asynchronous sampler completes $2.4\times$ the simulations of pyABC, of which \S\ref{sec:results-barrier} attributes $1.2\times$ to the barrier at this lattice size and most of the rest to pyABC's per-generation overhead; it is $1.6\times$ more efficient per simulation, read at pyABC's count of $5{,}400$ where pyABC's curve is the steeper (Fig.~\ref{fig:eps-curves}d), and ends $4.1\times$ tighter at equal wall clock.
```

**After**
```text
In the same hour the asynchronous sampler completes $2.4\times$ the simulations of pyABC, of which \S\ref{sec:results-barrier} attributes $1.2\times$ to the barrier at this lattice size and most of the rest to pyABC's per-generation overhead. It is $1.6\times$ more efficient per simulation, read at pyABC's count of $5{,}400$ where pyABC's curve is the steeper (Fig.~\ref{fig:eps-curves}d), and ends $4.1\times$ tighter at equal wall clock.
```

**Rationale**
The added attribution clause makes this sentence carry two decompositions (throughput, then efficiency) joined by a semicolon. Splitting at the semicolon keeps every number and the compact phrasing, and the reader sees the throughput attribution finish before the per-simulation figure starts.

### P-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: rejected
- reason: single reviewer, severity below H (batch rule); author may promote

**Before**
```text
An $\mathcal{F}_\infty$-measurable $q$, such as the run's final denominator $\bar q_n$, is \emph{not} admissible here; the proof below handles it through uniformity instead. The case $\phi=\pi h$ is the one used below.
```

**After**
```text
An $\mathcal{F}_\infty$-measurable $q$, such as the run's final denominator $\bar q_n$, is \emph{not} admissible here; the proof below handles it through uniformity instead and uses only the case $\phi=\pi h$.
```

**Rationale**
Two consecutive sentences end on "below". The second is a short add-on that merges cleanly into the first, and the math and the admissibility caveat stay unchanged.

### P-012
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- source: claude
- status: rejected
- reason: caption prose only; the figure environment is untouched (author to confirm captions are in scope)

**Before**
```text
Cellular Potts $50^3$ worker utilization, earlier parameterization (fraction of worker wall-clock spent inside the simulator) versus worker count ($1800$\,s budget; error bars are $\pm1$ s.d.); pyABC uses a population equal to the worker count (Fig.~\ref{fig:scaling}, right).
```

**After**
```text
Cellular Potts $50^3$ worker utilization (fraction of worker wall-clock spent inside the simulator) versus worker count, earlier parameterization ($1800$\,s budget; error bars are $\pm1$ s.d.); pyABC uses a population equal to the worker count (Fig.~\ref{fig:scaling}, right).
```

**Rationale**
The inserted "earlier parameterization" now sits between "utilization" and its parenthetical definition, so the definition reads as glossing the parameterization. Moving the tag after "versus worker count" restores the attachment and changes no content.

Another round: probably not on this scope. The added text is already low on AI-likeness, and after C-001 to C-004 the remaining issues are content decisions (the out-of-scope notes above), not prose. A verification pass on the applied diff is enough.

### P-013
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- source: codex
- duplicate-of: P-006
- status: rejected
- reason: The five-sentence proof roadmap narrates an itemized sequence that the proof immediately executes, producing the strongest mirrored-template effect in the scoped additions.

**Before**
```text
The argument has four steps. The prior floor bounds every weight (Lemma~\ref{lem:bounded}). For a fixed bandwidth and denominator, the weighted average minus its predictable compensator is an average of bounded martingale differences and vanishes (Lemma~\ref{lem:uslln}). The class of bandwidths and denominators is compact and the weights are Lipschitz in its parameters, so this convergence is uniform over the class. Uniformity allows the run-dependent pair $(\epsilon_n,\bar q_n)$ to be substituted, after which Lemma~\ref{lem:condmean} identifies the compensator as the tilted integral.
```

**After**
```text
The prior-floor bound (Lemma~\ref{lem:bounded}) and the uniform martingale law of large numbers (Lemma~\ref{lem:uslln}) give uniform convergence over the compact class of bandwidths and denominators. This permits substitution of the run-dependent pair $(\epsilon_n,\bar q_n)$, after which Lemma~\ref{lem:condmean} identifies the compensator as the tilted integral.
```

**Rationale**
The revision states the proof mechanism instead of announcing and then enumerating four steps. It preserves every lemma reference and the reason uniformity is needed.

### P-014
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-001
- status: rejected
- reason: “The bandwidths the sampler proposes with” is malformed, and the sentence obscures the important distinction between reporting and proposal bandwidths.

**Before**
```text
The event concerns only the bandwidth at which the estimate is reported, $\epsilon_n$ in \eqref{eq:posterior-estimator}: the bandwidths the sampler proposes with enter the proofs only through $\tilde q_i$, on which Theorem~\ref{thm:consistency} places no condition.
```

**After**
```text
The event concerns only the reporting bandwidth $\epsilon_n$ in \eqref{eq:posterior-estimator}; the bandwidths used to form proposals enter the proofs only through $\tilde q_i$, on which Theorem~\ref{thm:consistency} places no condition.
```

**Rationale**
This corrects the syntax and names the two bandwidth roles directly. The theorem’s scope and all notation remain unchanged.

### P-015
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: H
- source: codex
- duplicate-of: P-003
- status: rejected
- reason: The limitations section re-derives the floor condition from the theory section while combining two distinct limitations in one semicolon-heavy sentence.

**Before**
```text
A floor of $10^{-4}\epsilon_0$ guarantees this without binding on the g-and-k and Cellular Potts runs, but not on the two near-instantaneous benchmarks; and even where the limits apply they describe the run's continuation, since no reported run reached its floor (\S\ref{sec:theory}).
```

**After**
```text
A floor of $10^{-4}\epsilon_0$ keeps the reporting bandwidth away from zero without binding on the g-and-k and Cellular Potts runs, but it does not cover the two near-instantaneous benchmarks. No reported run reached this floor, so even where the limits apply, they describe the run's continuation (\S\ref{sec:theory}).
```

**Rationale**
The revision retains both unfavorable qualifications but separates benchmark coverage from the finite-run caveat. It also replaces the vague “guarantees this” with the precise condition being guaranteed.

### P-016
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: rejected
- reason: The sentence chains floor status, scheduler behavior, kernel support, and the effective-sample-size consequence with three equally weighted clauses.

**Before**
```text
No reported run reached its floor, so each is still a fixed-$k$ estimate: the scheduler holds the bandwidth until $2k$ evaluated particles lie inside it (\S\ref{sec:method-tolerance}), only $O(k)$ particles carry appreciable kernel mass, and the effective-sample-size ceiling of \S\ref{sec:results-posterior} is the visible consequence.
```

**After**
```text
Because no reported run reached its floor, each remains a fixed-$k$ estimate: the scheduler holds the bandwidth until $2k$ evaluated particles lie inside it (\S\ref{sec:method-tolerance}), leaving only $O(k)$ particles with appreciable kernel mass and producing the effective-sample-size ceiling of \S\ref{sec:results-posterior}.
```

**Rationale**
The revision makes the scheduler-to-support-to-ESS relation explicitly causal instead of presenting a mirrored list. It preserves the fixed-$k$ limitation and both evidence pointers.

### P-017
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: rejected
- reason: The sentence joins an asymptotic result, its temporal interpretation, and the finite-budget evidence in one long closing chain.

**Before**
```text
Once the floor is reached, the average weight converges to a positive constant and the effective sample size grows linearly in $n$; the limits describe that continuation, and the measurements above and the calibration checks of \S\ref{sec:results-posterior} quantify the error at the budgets used.
```

**After**
```text
Once the floor is reached, the average weight converges to a positive constant and the effective sample size grows linearly in $n$, so the limits describe the continuation from that point. At the reported budgets, the measurements above and the calibration checks of \S\ref{sec:results-posterior} quantify the error.
```

**Rationale**
The two sentences distinguish the asymptotic continuation from the finite-budget evidence. The revision preserves the deliberate warning that the reported runs are not themselves asymptotic.

### P-018
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-002
- status: rejected
- reason: The scheduler sentence combines cadence, cache invalidation, execution mode, and the observed twin difference in a single deeply nested construction.

**Before**
```text
That search runs on a rank's first call, then once per \texttt{bisect\_interval} ($=k$) of its own calls, and again whenever the rank's cached history is rebuilt after an out-of-order arrival; asynchronous arrivals cause such rebuilds and lockstepped ones do not, which is why the barrierized twin's bandwidth tightens less often than the asynchronous sampler's at equal call counts (\S\ref{sec:results-barrier}).
```

**After**
```text
Each rank runs this search on its first call, every \texttt{bisect\_interval} ($=k$) calls thereafter, and whenever an out-of-order arrival rebuilds its cached history. Such rebuilds occur under asynchronous but not lockstepped arrivals, so the barrierized twin tightens its bandwidth less often at equal call counts (\S\ref{sec:results-barrier}).
```

**Rationale**
The revision separates the search cadence from the execution-mode mechanism. It preserves the causal explanation for the measured difference between the twin and the asynchronous sampler.

### P-019
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-010
- status: rejected
- reason: Four distinct quantitative comparisons are packed into one sentence, making it difficult to distinguish throughput, barrier attribution, per-simulation efficiency, and equal-wall-clock efficiency.

**Before**
```text
In the same hour the asynchronous sampler completes $2.4\times$ the simulations of pyABC, of which \S\ref{sec:results-barrier} attributes $1.2\times$ to the barrier at this lattice size and most of the rest to pyABC's per-generation overhead; it is $1.6\times$ more efficient per simulation, read at pyABC's count of $5{,}400$ where pyABC's curve is the steeper (Fig.~\ref{fig:eps-curves}d), and ends $4.1\times$ tighter at equal wall clock.
```

**After**
```text
In the same hour the asynchronous sampler completes $2.4\times$ the simulations of pyABC; \S\ref{sec:results-barrier} attributes $1.2\times$ to the barrier at this lattice size and most of the rest to pyABC's per-generation overhead. At pyABC's count of $5{,}400$, where pyABC's curve is the steeper (Fig.~\ref{fig:eps-curves}d), the asynchronous sampler is $1.6\times$ more efficient per simulation; at equal wall clock, it ends $4.1\times$ tighter.
```

**Rationale**
The revision groups the throughput decomposition separately from the two tolerance comparisons. Every value, attribution, and figure reference remains intact.

### P-020
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- source: codex
- duplicate-of: P-007
- status: rejected
- reason: The sentence restates both the appendix introduction and the next subsection’s purpose without adding a condition, result, or evidence pointer.

**Before**
```text
The departure from the ideal enters the statements as $r$, which is then decomposed and partly measured.
```

**After**
```text

```

**Rationale**
The preceding sentences already explain why $r$ arises, and the following subsection immediately decomposes and measures it. Deleting the slogan-like closer removes genuine repetition without affecting the argument.

### P-021
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- source: codex
- status: rejected
- reason: The sentence announces that an example will show a difficulty instead of beginning the example.

**Before**
```text
A two-worker example shows the difficulty.
```

**After**
```text
Consider an execution with two workers.
```

**Rationale**
The replacement moves directly into the construction while retaining a useful transition. The substantive example and its role in motivating the filtration remain unchanged.

### P-022
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- source: codex
- duplicate-of: P-002
- status: rejected
- reason: “The implementation also offers” sounds product-like, and the sentence expresses the same no-effect conclusion twice through “every choice reduces” and “the option has no effect.”

**Before**
```text
The implementation also offers quantile, geometric-decay and acceptance-rate rules (\texttt{scheduler\_type}); they act only under the hard kernel, and with a smooth kernel every choice reduces to the rule above, so the option has no effect in any reported run.
```

**After**
```text
The quantile, geometric-decay and acceptance-rate options (\texttt{scheduler\_type}) apply only under the hard kernel; all smooth-kernel settings use the rule above, so \texttt{scheduler\_type} does not affect any reported run.
```

**Rationale**
The revision states the configuration behavior directly and removes the product-style “offers” construction. It preserves the important claim that the option was inert in all reported experiments.

I would not run another round after these edits, because the remaining scoped prose is technically specific and its residual repetition is functional.
