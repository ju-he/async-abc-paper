# Evaluation (Claude), round 01, scoped to text added 2026-10-09

## Scores

- **AI-likeness: 3/10.** The added text is mostly dense technical exposition tied to specific numbers, assumptions and runs, and it largely avoids the slogan and triad patterns. What remains is the floor caveat restated in five places (Section 4, Discussion, Limitations, the assumption-status row and the hyperparameter row), plus two scaffolding sentences in the appendices: a counted proof roadmap and a paragraph closer that repeats the appendix's opening.
- **Readability: 6/10.** The logic of the floor argument holds up, but several sentences make the reader parse them twice. Examples are "the bandwidths the sampler proposes with enter the proofs", "without binding on ... but not on ...", where the "not" attaches to the wrong verb, and an inserted sentence in the bandwidth-schedule paragraph that leaves "That search" without its antecedent. Long semicolon chains (the Cellular Potts production sentence, the Limitations floor sentence) add to the load.

Passages that most drive the AI-likeness score:
1. The floor/fixed-$k$ caveat appears in full or near-full form in Section 4 ("Which runs the limits describe"), in Section 8, in the status-table row ("The event concerns the reporting bandwidth only ... The bandwidth a finite budget reaches is not an asymptotic $\epsilon_\infty$: no run reached its floor, and the effective-sample-size ceiling ... is the consequence."), in the Discussion rule and in the hyperparameter row.
2. "The argument has four steps." followed by exactly four parallel declaratives (Appendix B, proof of Theorem 1).
3. "The departure from the ideal enters the statements as $r$, which is then decomposed and partly measured." (Appendix A, closing the AMIS-relation paragraph, restating the appendix's first sentence).

## What to leave alone

- **"Which runs the limits describe" (Section 4), apart from its second sentence.** The paragraph says plainly which benchmarks are covered and which are not, and it gives the Gaussian-mean/Lotka--Volterra exclusion as a negative result. The kept items ("exactly the path", "as they were executed", "Being covered does not make a budget asymptotic.") carry content: they separate path identity from asymptotic validity. The `% TODO(2026-10-09)` comment is untouched.
- **The two-worker example in Appendix B (Setting).** "A two-worker example shows the difficulty." introduces a concrete counterexample, which is the most readable addition in the round. The example itself is concrete and short.
- **The Section 4 sentence on why the proof is self-contained, and the longer Appendix A version.** These overlap, but the main text gives the reason and the appendix adds Cornuet et al. and the weight-bound argument, so the main text summarizes and the appendix gives detail. This is not a re-derived caveat.
- **Appendix A opening ("This appendix relates ... decomposes $r$ and measures part of it").** This is ordinary section navigation.
- **Theorem title "tracking at the realized bandwidth", assumption title "fidelity", and the contribution-list edit.** These are precise and neutral.
- **"Does the proposal path settle?" paragraph.** Its first sentence partly restates factor (b), but it ties the measurement to the heading's question. Cutting it would leave the paragraph starting on a number.
- **Hyperparameter row for `min_tol`.** It is a table cell that has to be self-contained. The repetition there is acceptable.
- **The Section 6.1/6.2 attribution edits.** Apart from the one sentence-length issue in C-010, "most of the rest to pyABC's per-generation overhead" is an informative attribution and should stay.
- **Out of scope, flagged for the author (no added text, so no proposal):** (i) Section 4 still says "Assumption~\ref{ass:stab} collects the remaining conditions", which is plural, but after the CLT removal the assumption holds one condition. (ii) Proposition~\ref{prop:censoring} keeps "$o(n^{-1/2})$ whenever $W=o(\sqrt n)$" with no CLT left to give that scale its meaning. Either keep it as a plain rate or say what it is compared against. Both are content decisions, not style ones. (iii) In "Does the proposal path settle?", a per-step drift of $\tau^{-0.7}$ does not by itself sum to $O(\log n)$. The $O(\log n)$ conclusion rests on the membership-change count, and the paragraph could make that link explicit.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason:

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

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: H
- status: proposed
- reason:

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

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason:

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

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason:

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

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- status: proposed
- reason:

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

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason:

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

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- status: proposed
- reason:

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

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
- reason:

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

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
- reason:

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

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
- reason:

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

### C-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
- reason:

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

### C-012
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
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
