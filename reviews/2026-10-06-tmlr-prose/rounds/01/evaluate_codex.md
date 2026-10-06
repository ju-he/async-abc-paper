## Scores

- AI-likeness: 8/10. The manuscript repeatedly announces distinctions, enumerates symmetrical mechanisms, and closes paragraphs with compressed slogans after the evidence has already made the point. The strongest signals are the recurring “three/four things” scaffolds, personified evidence, economic metaphors, and claim–mechanism–implication–punchline sequences.
- Readability: 6/10. The technical definitions, notation, and empirical findings are generally precise, but many sentences combine the setup, result, mechanism, qualification, and interpretation. The main claims remain recoverable, yet the prose makes readers unpack too many clause-stacked sentences and repeated summaries.

The three passages that most drive the AI-likeness score are:

1. “Three weightings appear, and only the last is reported: the \emph{archive weights} $\tilde W_j$ shape the next proposal, the \emph{parent weight} $w^\star$ stored with each particle is used only inside the running sampler, and the \emph{posterior weights} $W_{i,n}$ are computed afterwards over the whole history.”
2. “The results give a practitioner four rules,” followed by the four parallel imperatives “Use it,” “Predict the gain,” “Set the starting bandwidth,” and “Keep the archive small.”
3. “The effective-sample-size ceiling is where the method can still improve. Barrier removal delivers simulations reliably and predictably; converting all of them into posterior accuracy is limited by the reporting rule, which we adopted for its simplicity: report at the tightest bandwidth reached, chosen by ESS retention against the archive. Further work on this method belongs in that rule rather than in the scheduler.”

## What to leave alone

- Keep the C1–C4 subsection headings unchanged. They state the tested claims directly and provide useful navigation through a long systems-and-methods evaluation.
- Keep the distinction between bandwidth and tolerance, and between $S$ and $m$. These distinctions prevent genuine technical ambiguity and are reinforced consistently.
- Keep the scientific contrasts involving the slowest simulation, communication cost, matched simulation budgets, discrepancy definitions, and the twin. These contrasts identify actual experimental alternatives rather than rhetorical foils.
- Keep the unfavorable Gaussian-mean throughput result, the four-dimensional g-and-k recovery loss and erratic ESS, the stabilization exponents near or below the required threshold, and the archive-size calibration failure. These are specific, plainly reported negative results.
- Keep the Limitations section’s substantive claims: partial measurement of $r$, the assumed asynchronous filtration, the restricted central limit theorem, and uncorrected completion-time selection. The qualifications match the evidence.
- Keep the formal assumptions, theorem statements, equations, and the logical qualifications in Appendix B. Their density is appropriate for proof prose, and most repetition there records distinctions needed by the argument.
- Keep captions that fully specify markers, line types, columns, aggregation, and exceptional configurations. That detail is necessary for figures and tables to stand alone.
- Keep the final conclusion paragraph’s return to the generation-barrier framing. It closes the paper coherently and preserves the author’s intended structure.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The fourth contribution packs the theorem, diagnostics, posterior experiments, and two limitations into one heavily nested item.

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

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The abstract ends by combining a favorable recovery result and the principal limitation in one sentence.

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

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- status: proposed
- reason: The paragraph narrates what “the claims need” and then presents a balanced triad instead of introducing the benchmarks directly.

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

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: One sentence combines the source of runtime variation, two causal mechanisms, a correlation, and a lattice-size effect.

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

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- status: proposed
- reason: “Three weightings appear” announces a symmetrical list instead of defining the quantities directly.

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

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: H
- status: proposed
- reason: The second sentence poses a rhetorical question immediately after the precise mismatch has already been stated.

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

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: The sentence merely announces a list whose following paragraphs already distinguish the comparators and their purposes.

**Before**
```text
Three comparators appear, each for a different question, and each keeps one name throughout.
```

**After**
```text

```

**Rationale**
The pyABC, twin, and rejection-ABC definitions are self-contained without this announcement. Deleting it removes redundant meta-commentary.

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
- reason: The sentence combines configuration, an unsupported practitioner persona, and experimental role.

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

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: This paragraph repeatedly moves from mechanism to contrast to rule, while one sentence embeds four sources of cross-benchmark variation.

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

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The paragraph personifies the lower panels, then ends with the vague claim that a “ceiling binds.”

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

### C-011
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
- reason: An abstract noun is made to “separate” the methods before the actual comparison is given.

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

### C-012
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: “What differs” delays the subject, and the following sentence combines a general regime, an explanation of earlier results, and a limitation.

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

### C-013
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Three distinct archive-size experiments and their interpretations are compressed into one colon-led sentence.

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

### C-014
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
- reason: The imperative “Use it” gives the recommendation a product-like tone and buries the measured applicability boundary in a semicolon.

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

### C-015
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
- reason: “Predict the gain before paying for it” is an economic slogan attached to an otherwise precise estimator.

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

### C-016
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: H
- status: proposed
- reason: The Discussion closes with three slogan-like claims that repeat the preceding ESS results and personify where future work “belongs.”

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

### C-017
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The first Limitations sentence combines four separate qualifications before reaching the stabilization result.

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

### C-018
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: The appendix announces a numbered novelty list whose contents are already stated immediately afterward.

**Before**
```text
Three things are not routine.
```

**After**
```text

```

**Rationale**
The following sentences identify each departure from existing AMIS theory without this announcement. Deletion removes a conspicuous rhetorical template and no mathematical content.

### C-019
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: The proof discussion announces that a step is “trivial” and ends with a sentence-initial additive punchline.

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

### C-020
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: “Walk from” narrates the reader’s movement through the argument instead of stating the decomposition.

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

### C-021
- file: latex/tmlr/tmlr-article.tex
- category: heading
- severity: M
- status: proposed
- reason: The heading “Measured, in part” is vague, and the sentence below repeats the four-factor framing before naming what is measured.

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

### C-022
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: M
- status: proposed
- reason: The paragraph says this is “the one place” where the authors can do better than assume, which narrates the strength of the text rather than the available measurement.

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

### C-023
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- status: proposed
- reason: The isolated “Identifiability does” is a dramatic contrast with a vague verb; the next sentence already supplies the specific effect.

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

### C-024
- file: latex/tmlr/tmlr-article.tex
- category: meta
- severity: L
- status: proposed
- reason: Calling the implementation change “one-line” announces simplicity without contributing to the technical tradeoff.

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

### C-025
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- status: proposed
- reason: This fragment is fully restated by the next sentence, which reports that posterior-mean error does not increase with coupling.

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