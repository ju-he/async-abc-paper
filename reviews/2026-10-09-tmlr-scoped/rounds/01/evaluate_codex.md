## Scores

**AI-likeness: 6/10.** The scoped additions repeatedly use explicit roadmaps, multi-stage causal chains, and closing sentences that restate a paragraph’s purpose. Their technical specificity and candid limitations prevent the prose from sounding uniformly generated, but the recurring templates remain conspicuous.

**Readability: 7/10.** Most claims are precise, evidence-bound, and supported by useful examples. Readability falls where single sentences carry a condition, mechanism, benchmark exception, and implication, and where the positive-bandwidth caveat is re-derived across several sections.

The three passages that most drive the AI-likeness score are:

- “The argument has four steps. The prior floor bounds every weight... Uniformity allows the run-dependent pair...”
- “No reported run reached its floor, so each is still a fixed-$k$ estimate... Once the floor is reached...”
- “The prior floor in the \emph{denominator} bounds every weight... This lets the method... The departure from the ideal enters the statements as $r$...”

## What to leave alone

- Keep the scientific contrasts in the AMIS discussion: fixed versus adaptive proposals, martingale arrays rather than i.i.d.\ arguments, and the published adaptation restrictions all carry technical information.
- Keep the unfavorable coverage statements for Gaussian mean and Lotka--Volterra, the fixed-budget qualification, and “Being covered does not make a budget asymptotic.” These are substantive limitations, not rhetorical pessimism.
- Keep “exactly the path,” “as they were executed,” and “at all.” Each identifies a specific equivalence or feasibility condition rather than serving as emphasis.
- Keep the quantitative bandwidth evidence in Table~\ref{tab:assumption-status}. Although it overlaps the main text, the table is the appropriate consolidated audit of which assumptions hold.
- Keep the substantive two-worker execution example. It makes the filtration problem concrete; only its meta-commentary opener needs revision.
- Keep the theorem and assumption titles, the italic Discussion rule titles, and the “Measured, in part.” heading. They are accurate navigation and conform to the author’s stated decisions.
- Keep the negative posterior and scaling results, including the effective-sample-size ceiling and the loss on near-instantaneous simulators. They are unusually specific and human-sounding.
- Keep the conclusion’s return to the generation-barrier framing. It closes the manuscript’s argument without introducing a new slogan.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- status: proposed
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

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
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

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: H
- status: proposed
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

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
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

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
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

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
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

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
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

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- status: proposed
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

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
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

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
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