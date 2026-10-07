## Scores

**AI-likeness: 3/10.** The manuscript is evidence-bound, technically specific, and unusually candid about failures, unfavorable comparisons, and unproved conditions; those qualities make most of it read as authored scholarship. The remaining generated feel comes from a few comprehensive inventory sentences, repeated claim-to-implication sequences, and polished paragraph closers that compress several distinct qualifications into one rhetorical unit.

**Readability: 6/10** (10 = easiest to read). The empirical sections are navigable, and their quantitative claims are consistently attached to tables, figures, or citations. Section 3 and Appendices A–B contain several clause-stacked transitions, one grammatically incomplete assumption clause, and proof sentences whose controlling subject or logical dependency becomes clear only at the end.

The three passages that most drive the AI-likeness score are:

1. “A martingale analysis of the estimator showing that it targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when the ratio tends to one; we derive from the execution model the adaptedness that asynchronous execution requires, and decompose the ratio and measure part of it…”

2. “Barrier removal delivers simulations predictably; converting all of them into posterior accuracy is limited by the reporting rule, which we adopted for its simplicity: report at the tightest bandwidth reached, chosen by ESS retention against the archive. Further work on this method belongs in that rule rather than in the scheduler.”

3. “Its estimator targets the smooth-ABC posterior up to a partly measured fidelity ratio, is calibrated in one dimension and conservative in four, and is replayable from the log. What it delivers is bounded by the starting bandwidth and the archive size…”

## What to leave alone

- Keep the direct barrier framing in the Introduction and the Conclusion. It gives the paper a coherent systems question, and the final return to that framing is appropriate rather than formulaic.

- Keep the negative results and boundary cases: the loss on the Gaussian benchmark, pyABC overtaking at scale, the four-dimensional recovery deficit, the effective-sample-size ceiling, the failure on a ridge, and the lack of an asymptotic claim for the production configuration. These are among the manuscript’s strongest and most human passages.

- Keep scientifically necessary contrasts, including slowest versus average runtime, communication versus arithmetic, the barrierized twin versus pyABC, and discrepancy versus a metric. They distinguish actual mechanisms rather than inventing rhetorical foils.

- Keep “Identifiability does.”, “The posterior does not follow.”, “the method a practitioner would run”, the “Measured, in part.” heading, the question-form Appendix A headings, the Discussion’s imperative rule titles, the Appendix F “lifts” sentence, and the final paragraph’s barrier return, as directed.

- Keep the sentence about the buffer retained for the weight of step 6 and the current separation between proposal-time indexing and the following filtration explanation. Both prevent real antecedent errors.

- Keep the prior-floor/Lindeberg passage, including “trivial.” That word has a precise proof-theoretic meaning here and is not promotional vocabulary.

- Keep the ordinary section navigation and most theorem-proof rhythm. Appendix B does not need a general stylistic shortening pass; only the specific transitions below impede the argument.

- Keep the limited repetition of the fidelity-ratio and positive-bandwidth caveats in the abstract, theory, Limitations, and Conclusion. Each occurrence serves a different synopsis level; the problem is sentence construction, not the recurrence of the qualifications.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The central-limit clause of Assumption 5 is grammatically incomplete and leaves the reader to infer the governing verb.

**Before**
```text
\emph{(ii, central limit theorem)} $\epsilon_\infty>0$ and $r$ deterministic, a stabilization rate and limits of the proposal and the denominator, stated in Appendix~\ref{app:theory-more}; clause (ii) implies clause (i).
```

**After**
```text
\emph{(ii, central limit theorem)} Under the additional conditions that $\epsilon_\infty>0$ and $r$ is deterministic, Appendix~\ref{app:theory-more} states a stabilization rate and limits of the proposal and the denominator. Clause (ii) implies clause (i).
```

**Rationale**
The revision supplies the missing verb and makes the additional conditions modify the correct statement. It preserves the logical relation between clauses (ii) and (i).

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The main theoretical result and its limiting corollary are packed into one sentence with three levels of qualification.

**Before**
```text
We prove that the estimate tracks the smooth-ABC posterior at the realized bandwidth tilted by $r_n$, with an error that vanishes whatever the sampler does, and that it converges to the posterior tilted by the limit $r$ of $r_n$ when that limit exists, so that $r\equiv1$ gives the posterior itself.
```

**After**
```text
We prove that the estimate tracks the smooth-ABC posterior at the realized bandwidth, tilted by $r_n$, with an error that vanishes whatever the sampler does. If $r_n$ converges to $r$, the estimate converges to the posterior tilted by $r$; when $r\equiv1$, this limit is the posterior itself.
```

**Rationale**
The split separates finite-run tracking from the additional hypothesis needed for a limiting result. The connective “If” makes the dependency explicit without weakening either claim.

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: H
- status: proposed
- reason: The paragraph’s operational conclusion—that the production run is outside the asymptotic regime—arrives only after a long description of the bandwidth schedule.

**Before**
```text
\paragraph{Which runs the limits describe.} Every asymptotic statement above holds on $\{\epsilon_\infty>0\}$. A run with a positive bandwidth floor (\texttt{min\_tol} in Appendix~\ref{app:implementation}) is on that event by construction. The production configuration sets no floor: its scheduler holds the bandwidth until $2k$ evaluated particles lie inside it (\S\ref{sec:method-tolerance}), so $\epsilon_n$ tracks a fixed-rank order statistic of the discrepancies, which tends to the infimum of their support, zero for a stochastic simulator. The effective-sample-size ceiling of \S\ref{sec:results-posterior} is the signature of this regime: on $\{\epsilon_\infty>0\}$ with a settled ratio the average weight converges to a positive constant and the effective sample size grows linearly in $n$, whereas with a bandwidth tied to the $2k$-th order statistic only $O(k)$ particles ever carry appreciable kernel mass. The production run is thus a fixed-$k$ estimate at a bandwidth that shrinks with the budget, and we make no asymptotic claim for it; shrinking-bandwidth asymptotics are those of ABC with a vanishing tolerance and are left to future work. For that run the theory supplies the accounting of where error enters; the measurements above and the calibration checks of \S\ref{sec:results-posterior} quantify it at the budgets used.
```

**After**
```text
\paragraph{Which runs the limits describe.} Every asymptotic statement above holds on $\{\epsilon_\infty>0\}$. A run with a positive bandwidth floor (\texttt{min\_tol} in Appendix~\ref{app:implementation}) is on that event by construction. The production configuration, however, sets no floor. Its scheduler holds the bandwidth until $2k$ evaluated particles lie inside it (\S\ref{sec:method-tolerance}), so $\epsilon_n$ tracks a fixed-rank order statistic of the discrepancies and tends to the infimum of their support, zero for a stochastic simulator. The production run is therefore a fixed-$k$ estimate at a bandwidth that shrinks with the budget, and we make no asymptotic claim for it.

The effective-sample-size ceiling of \S\ref{sec:results-posterior} is the signature of this regime. On $\{\epsilon_\infty>0\}$ with a settled ratio, the average weight converges to a positive constant and the effective sample size grows linearly in $n$; with a bandwidth tied to the $2k$-th order statistic, only $O(k)$ particles ever carry appreciable kernel mass. Shrinking-bandwidth asymptotics are those of ABC with a vanishing tolerance and are left to future work. For the production run, the theory instead supplies the accounting of where error enters; the measurements above and the calibration checks of \S\ref{sec:results-posterior} quantify it at the budgets used.
```

**Rationale**
The production configuration’s status now appears immediately after the condition it violates. The second paragraph then explains the observed effective-sample-size behavior and the theory’s remaining finite-budget role.

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: H
- status: proposed
- reason: The Limitations paragraph combines three independent limitations—the fidelity ratio, the CLT rate, and deadline length bias—before any one of them is fully resolved.

**Before**
```text
The limit statements hold up to the fidelity ratio $r$, of which two factors are measured and two bounded, and only on runs whose bandwidth stays away from zero; the production configuration sets no bandwidth floor, so we make no asymptotic claim for it (\S\ref{sec:theory}). The central limit theorem is proved for damped-adaptation variants, and the production runs sit at the threshold of its rate condition rather than clearly inside it (Appendix~\ref{app:theory-more}). Keeping every evaluated particle over-represents parameter regions that simulate quickly, the length bias of deadline-limited sampling \citep{murray2021anytime}. Under the execution model the direct part of that effect is confined to the at most $W$ candidates in flight at the deadline, an $O(W/n)$ perturbation (Proposition~\ref{prop:censoring}); the rest acts through the proposal sequence, which the weighting corrects up to $r$. A runtime-coupled study found no measurable shift of the posterior mean at the couplings tested, and we do not correct for the boundary term (Appendix~\ref{app:moved}).
```

**After**
```text
The limit statements hold only on runs whose bandwidth stays away from zero and only up to the fidelity ratio $r$, of which two factors are measured and two bounded. The production configuration sets no bandwidth floor, so we make no asymptotic claim for it (\S\ref{sec:theory}). The central limit theorem is proved for damped-adaptation variants, and the production runs sit at the threshold of its rate condition rather than clearly inside it (Appendix~\ref{app:theory-more}).

Keeping every evaluated particle over-represents parameter regions that simulate quickly, the length bias of deadline-limited sampling \citep{murray2021anytime}. Under the execution model, the direct part of that effect is confined to the at most $W$ candidates in flight at the deadline, an $O(W/n)$ perturbation (Proposition~\ref{prop:censoring}). The remaining effect acts through the proposal sequence, which the weighting corrects up to $r$. A runtime-coupled study found no measurable shift of the posterior mean at the couplings tested, and we do not correct for the boundary term (Appendix~\ref{app:moved}).
```

**Rationale**
The paragraph break separates asymptotic limitations from deadline bias, while the sentence split distinguishes the controlled direct effect from the proposal-mediated effect. All limitations remain explicit and unchanged in strength.

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The Conclusion opens with the cost definition, prediction method, and validation result in one sentence.

**Before**
```text
A generation barrier costs the ratio of the barrier-free throughput to what a generation allows, and that cost can be predicted from the asynchronous run's timing alone; the prediction matched a barrierized twin of our sampler on synthetic runtime laws and on a real tissue simulator.
```

**After**
```text
A generation barrier costs the ratio of the barrier-free throughput to the throughput a generation allows, and this cost can be predicted from the asynchronous run's timing alone. The prediction matched a barrierized twin of our sampler on synthetic runtime laws and on a real tissue simulator.
```

**Rationale**
The validation now follows the definition as a separate result. The edit retains the Conclusion’s return to the barrier framing.

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The abstract joins a theorem, its exact-fidelity condition, and an empirical calibration comparison in a single sentence.

**Before**
```text
We prove that the estimator targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when that ratio tends to one, and find it calibrated where a pyABC baseline with the same kernel and schedule under-covers.
```

**After**
```text
We prove that the estimator targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when that ratio tends to one. In calibration experiments, the estimator is calibrated where a pyABC baseline with the same kernel and schedule under-covers.
```

**Rationale**
The theoretical and empirical claims now have separate agents and evidence types. The revision preserves the exact-consistency condition and the unfavorable pyABC result.

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: Contribution C4 combines the theorem, execution model, fidelity accounting, proofs, calibration, recovery, and two tuning limitations in two heavily stacked sentences.

**Before**
```text
\item A martingale analysis of the estimator showing that it targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when the ratio tends to one; we derive from the execution model the adaptedness that asynchronous execution requires, and decompose the ratio and measure part of it (\S\ref{sec:theory}; proofs, a deadline-censoring bound and a central limit theorem in the appendix). Calibration checks in one and four dimensions and under multimodality, recovery of a two-parameter Cellular Potts posterior, and the two settings that bound the posterior, the starting bandwidth and the archive size (C4, \S\ref{sec:results-posterior}).
```

**After**
```text
\item A martingale analysis showing that the estimator targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when the ratio tends to one (\S\ref{sec:theory}). From the execution model, we derive the adaptedness that asynchronous execution requires; we also decompose the ratio and measure part of it (proofs, a deadline-censoring bound and a central limit theorem in the appendix). C4 additionally includes calibration checks in one and four dimensions and under multimodality, recovery of a two-parameter Cellular Potts posterior, and the two settings that bound the posterior, the starting bandwidth and the archive size (\S\ref{sec:results-posterior}).
```

**Rationale**
The revision separates the theorem, supporting analysis, and empirical scope without reducing them to slogan-like fragments. Every component and cross-reference remains present.

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: Assumption B3 and its consequence introduce the scheduling variables, independence condition, filtration, and two conditional laws without a pause.

**Before**
```text
\emph{(B3)} Which worker proposes next, when, and which earlier results it has received by then are determined by earlier proposal times, durations and worker identities, together with an exogenous source of randomness (message delivery) independent of the streams in (B1) and (B2). Lemma~\ref{lem:adapted} derives from this the filtration $(\mathcal{F}_i)$ the proofs use, under which $\theta_i\mid\mathcal{F}_{i-1}\sim\tilde q_i$ with $\tilde q_i$ the proposal density the worker emits, and $\rho_i\mid\mathcal{F}_{i-1}\vee\sigma(\theta_i)\sim p(\cdot\mid\theta_i)$.
```

**After**
```text
\emph{(B3)} Earlier proposal times, durations and worker identities, together with an exogenous source of randomness (message delivery), determine which worker proposes next, when it proposes, and which earlier results it has received. The exogenous source is independent of the streams in (B1) and (B2). Lemma~\ref{lem:adapted} derives from this model the filtration $(\mathcal{F}_i)$ used in the proofs. Under that filtration, $\theta_i\mid\mathcal{F}_{i-1}\sim\tilde q_i$, with $\tilde q_i$ the proposal density the worker emits, and $\rho_i\mid\mathcal{F}_{i-1}\vee\sigma(\theta_i)\sim p(\cdot\mid\theta_i)$.
```

**Rationale**
The scheduling rule, independence condition, and derived conditional laws become distinct logical steps. The revised wording also gives “that filtration” an immediate antecedent.

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The explanation after the tracking theorem delays the two observable conditions until after two intervening qualifications.

**Before**
```text
The estimate thus behaves as if its denominator were the density the sample was drawn from, up to the tilt $r_n$, whatever the proposal path; its two conditions, a bandwidth away from zero and an average weight that does not collapse, are observable from a run. Passing to a limit needs the ratio to settle:
```

**After**
```text
Whatever the proposal path, the estimate behaves as if its denominator were the density the sample was drawn from, up to the tilt $r_n$. Its two conditions, a bandwidth away from zero and an average weight that does not collapse, are observable from a run. A limiting result additionally requires the ratio to settle:
```

**Rationale**
The theorem’s interpretation, observable conditions, and extra limiting condition now appear in their dependency order. The edit removes no qualification.

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Two pieces of notation and the event restricting all asymptotic statements are introduced at the end of an already dense theory overview.

**Before**
```text
Throughout, $\epsilon_\infty:=\lim_n\epsilon_n$, which exists because the schedule never loosens and may be zero; the asymptotic statements hold on the event $\{\epsilon_\infty>0\}$, and the closing paragraph of this section says which configurations put a run there. $L_\epsilon(\theta)=\mathbb{E}[K_\epsilon(\rho)\mid\theta]$ is the smooth ABC likelihood of \eqref{eq:abc-target} and $L^{(2)}_\epsilon(\theta)=\mathbb{E}[K_\epsilon(\rho)^2\mid\theta]$ its second-moment counterpart, used by the central limit theorem.
```

**After**
```text
Throughout, $\epsilon_\infty:=\lim_n\epsilon_n$; this limit exists because the schedule never loosens, but it may be zero. The asymptotic statements hold on the event $\{\epsilon_\infty>0\}$, and the closing paragraph of this section identifies which configurations put a run there. We write $L_\epsilon(\theta)=\mathbb{E}[K_\epsilon(\rho)\mid\theta]$ for the smooth ABC likelihood of \eqref{eq:abc-target} and $L^{(2)}_\epsilon(\theta)=\mathbb{E}[K_\epsilon(\rho)^2\mid\theta]$ for its second-moment counterpart, used by the central limit theorem.
```

**Rationale**
The reading cue “We write” makes the notation introduction explicit, while the split isolates the event on which the results apply. No definition or condition changes.

### C-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The sentence describing the CLT’s scope embeds the empirical status of the production schedule inside the theorem’s applicability condition.

**Before**
```text
Theorem~\ref{thm:clt} covers variants whose adaptation is frozen after a finite burn-in or damped fast enough for the rate condition \eqref{eq:rate}; whether the data-driven schedule our experiments run meets that condition is measured in \S\ref{sec:theory-r}, where the production runs sit at its threshold.
```

**After**
```text
Theorem~\ref{thm:clt} covers variants whose adaptation is frozen after a finite burn-in or damped fast enough for the rate condition \eqref{eq:rate}. Section~\ref{sec:theory-r} assesses the data-driven schedule used in our experiments and finds that the production runs sit at the threshold of that condition.
```

**Rationale**
The theorem’s formal scope is separated from the diagnostic evidence about the implemented schedule. The production runs remain described plainly as threshold cases.

### C-012
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The noise-free variance comparison and the interpretation of its gap compete for the same main clause.

**Before**
```text
Because $L^{(2)}_\epsilon\ge L^2_\epsilon$ by Jensen, \eqref{eq:clt-variance} is never smaller than the variance of the corresponding \emph{noise-free} importance sampler that could evaluate $L_{\epsilon_\infty}$ exactly; the gap $L^{(2)}_\epsilon-L^2_\epsilon=\mathrm{Var}(K_\epsilon(\rho)\mid\theta)$ is the price of running one simulation per proposed parameter.
```

**After**
```text
Because $L^{(2)}_\epsilon\ge L^2_\epsilon$ by Jensen, \eqref{eq:clt-variance} is never smaller than the variance of the corresponding \emph{noise-free} importance sampler that could evaluate $L_{\epsilon_\infty}$ exactly. The gap $L^{(2)}_\epsilon-L^2_\epsilon=\mathrm{Var}(K_\epsilon(\rho)\mid\theta)$ is the price of running one simulation per proposed parameter.
```

**Rationale**
The split preserves the logical connective while giving the variance comparison and its interpretation separate emphasis.

### C-013
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The explanation of the tilt identity states both the operative quantity and scale invariance after a colon, which makes the second fact easy to miss.

**Before**
```text
The identity says that the tilt moves the target through the variation of $r$ over posterior mass, not through its distance from one: $\pi^{cr}_{\epsilon_\infty}=\pi^r_{\epsilon_\infty}$ for every constant $c>0$.
```

**After**
```text
The tilt moves the target through the variation of $r$ over posterior mass, not through its distance from one. In particular, multiplying $r$ by any constant $c>0$ leaves the tilted target unchanged: $\pi^{cr}_{\epsilon_\infty}=\pi^r_{\epsilon_\infty}$.
```

**Rationale**
The scientific contrast is retained because it distinguishes variation from irrelevant scaling. The second sentence makes the displayed invariance read as the reason for that distinction.

### C-014
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Three intermediate mixtures are introduced in one semicolon chain, making it difficult to retain which implementation feature changes at each step.

**Before**
```text
Let $\bar q^{(1)}=\tfrac1n\sum_{i\le n}q_i$ use the \emph{nominal} archive proposals (the truncated mixtures the propagator intends, with exact normalizing constants and without the rejection cap or the underflow redraws); let $\bar q^{(2)}$ replace those constants by the implementation's product-of-marginals values; and let $\bar q^{(3)}$ be the $m$-point draw-proportional version of $\bar q^{(2)}$ carrying the observed bootstrap share $\nu_n$.
```

**After**
```text
Let $\bar q^{(1)}=\tfrac1n\sum_{i\le n}q_i$ use the \emph{nominal} archive proposals: the truncated mixtures the propagator intends, with exact normalizing constants and without the rejection cap or the underflow redraws. We obtain $\bar q^{(2)}$ by replacing those constants with the implementation's product-of-marginals values, and $\bar q^{(3)}$ by taking the $m$-point draw-proportional version of $\bar q^{(2)}$ carrying the observed bootstrap share $\nu_n$.
```

**Rationale**
The first mixture receives a complete definition before the two transformations are introduced. The second sentence retains the sequential relationship between $\bar q^{(1)}$, $\bar q^{(2)}$, and $\bar q^{(3)}$.

### C-015
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The capped-rejection and underflow-redraw mechanisms are distinct departures from the nominal proposal but are joined in one sentence.

**Before**
```text
A parent's perturbation is rejection-sampled into the box for at most $992$ attempts, so component $j$ is realized with probability $W_j[1-(1-Z_j)^{992}]$ and the residual is emitted as a prior draw; and a candidate whose proposal-time denominator falls below $10^{-12}$ is redrawn up to five times, the last one being emitted regardless.
```

**After**
```text
A parent's perturbation is rejection-sampled into the box for at most $992$ attempts, so component $j$ is realized with probability $W_j[1-(1-Z_j)^{992}]$ and the residual is emitted as a prior draw. Separately, a candidate whose proposal-time denominator falls below $10^{-12}$ is redrawn up to five times, the last one being emitted regardless.
```

**Rationale**
“Separately” identifies the second mechanism as another implementation departure rather than a consequence of the rejection cap. Every threshold and probability remains unchanged.

### C-016
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The asynchrony contribution moves from proposal reconstruction to a no-further-gap claim and then to the $W$-particle boundary without marking those steps.

**Before**
```text
Under asynchronous execution this factor additionally includes the difference between the proposal each candidate came from, formed on the prefix its worker held, and the snapshots the post-hoc pass rebuilds from an arrival-ordered log. The emitted density is the conditional law of the candidate (Lemma~\ref{lem:adapted}), so no further gap arises; and prefixes of equal length in the two orders differ in at most $W$ candidates, those in flight at the time, so each rebuilt snapshot differs from the proposal in force at that index by a boundary of at most $W$ particles.
```

**After**
```text
Under asynchronous execution, this factor additionally includes the difference between the proposal each candidate came from, formed on the prefix its worker held, and the snapshots the post-hoc pass rebuilds from an arrival-ordered log. The emitted density is the conditional law of the candidate (Lemma~\ref{lem:adapted}), so no further gap arises at that step. Prefixes of equal length in the two orders differ in at most $W$ candidates, those in flight at the time; therefore, each rebuilt snapshot differs from the proposal in force at that index by a boundary of at most $W$ particles.
```

**Rationale**
The added antecedent “at that step” limits the no-gap claim to the conditional law. The final sentence then states the separate reconstruction boundary and its consequence.

### C-017
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The Cellular Potts fidelity diagnostic combines why factor (c) is active, the bootstrap share, the floor, two component estimates, their total, and the range of $r$ in one sentence.

**Before**
```text
The five Cellular Potts production replicates (Appendix~\ref{app:cpm}) have about $12{,}900$ asynchronous evaluations each across $48$ workers and two correlated dimensions, so that (c) is live, and both factors are active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**After**
```text
The five Cellular Potts production replicates (Appendix~\ref{app:cpm}) have about $12{,}900$ asynchronous evaluations each across $48$ workers and two correlated dimensions, so that (c) is live. Both measured factors are also active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**Rationale**
The first sentence establishes why the unmeasured normalizer factor matters; the second reports the measured factors. No diagnostic value or qualification changes.

### C-018
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The fixed-floor requirement, loss of the weight bound, role of that bound, and need for a new rate argument are nested into one causal chain.

**Before**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$, so growing $m$ there would send $\delta\to0$ and with it the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}, which makes the Lindeberg step trivial and fixes the constants in the law of large numbers; the rate conditions would have to be redone.
```

**After**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$, so growing $m$ there would send $\delta\to0$ and remove the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}. That bound makes the Lindeberg step trivial and fixes the constants in the law of large numbers; without it, the rate conditions would have to be redone.
```

**Rationale**
The split makes the lost property explicit before describing the proof steps that rely on it. It retains the accepted use of “trivial.”

### C-019
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The filtration proof names three random objects, their inputs, and their measurability in one sentence whose conclusion arrives late.

**Before**
```text
By (B3) and induction on $i$, the time $T_i$, the worker $w_i$ and the set $A_i\subset\{1,\dots,i-1\}$ of candidates whose results have reached $w_i$ by $T_i$ are functions of $(T_j,D_j,w_j)_{j<i}$ and of $\mathcal{E}$, hence $\mathcal{F}_{i-1}$-measurable.
```

**After**
```text
By (B3) and induction on $i$, the time $T_i$, the worker $w_i$ and the set $A_i\subset\{1,\dots,i-1\}$ of candidates whose results have reached $w_i$ by $T_i$ are functions of $(T_j,D_j,w_j)_{j<i}$ and of $\mathcal{E}$. They are therefore $\mathcal{F}_{i-1}$-measurable.
```

**Rationale**
The proof’s functional-dependence claim is completed before its measurability consequence. The connective “therefore” preserves the inference.

### C-020
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The post-hoc reconstruction sentence buries the antecedent of “which” beneath the lemma invocation and the $W$-particle bound.

**Before**
```text
The post-hoc estimator consumes a rank's \emph{arrival}-ordered log and reconstructs the $q_{\tau_s}$ from prefixes of it; by the last clause of the lemma each reconstructed snapshot differs from the proposal in force at that index by a boundary of at most $W$ particles, which is the asynchrony contribution to $r$ (\S\ref{sec:theory-r}(d)).
```

**After**
```text
The post-hoc estimator consumes a rank's \emph{arrival}-ordered log and reconstructs the $q_{\tau_s}$ from its prefixes. By the last clause of the lemma, each reconstructed snapshot differs from the proposal in force at that index by a boundary of at most $W$ particles. This boundary is the asynchrony contribution to $r$ (\S\ref{sec:theory-r}(d)).
```

**Rationale**
“This boundary” replaces a buried relative-clause antecedent. The proof now moves cleanly from reconstruction to the bound and then to its interpretation.

### C-021
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The normalizer argument contains the factor definition, interval geometry, lower bound, product bound, and smoothness conclusion in one sentence.

**Before**
```text
The implementation's constant is the \emph{product of per-axis} in-box masses, which is where the box form of $\Theta$ in Assumption~\ref{ass:class} enters: each factor is $\Phi((\mathrm{hi}_d-\mu_{jd})/\sigma_d)-\Phi((\mathrm{lo}_d-\mu_{jd})/\sigma_d)$, and since $\mu_{jd}$ lies in $[\mathrm{lo}_d,\mathrm{hi}_d]$ that interval contains a half-width $w_d/2$ on one side of $\mu_{jd}$, giving a factor at least $\Phi(w_d/2\sigma_+)-\tfrac12>0$; the product over coordinates, $\prod_{d}\bigl(\Phi(w_d/2\sigma_+)-\tfrac12\bigr)$, is therefore bounded below by a constant depending only on the box and $\sigma_+$, and is smooth in $(\mu_j,\Sigma)$ with bounded derivatives there.
```

**After**
```text
The implementation's constant is the \emph{product of per-axis} in-box masses, which is where the box form of $\Theta$ in Assumption~\ref{ass:class} enters. Each factor is $\Phi((\mathrm{hi}_d-\mu_{jd})/\sigma_d)-\Phi((\mathrm{lo}_d-\mu_{jd})/\sigma_d)$. Since $\mu_{jd}$ lies in $[\mathrm{lo}_d,\mathrm{hi}_d]$, that interval contains a half-width $w_d/2$ on one side of $\mu_{jd}$, giving a factor at least $\Phi(w_d/2\sigma_+)-\tfrac12>0$. The product over coordinates, $\prod_{d}\bigl(\Phi(w_d/2\sigma_+)-\tfrac12\bigr)$, is therefore bounded below by a constant depending only on the box and $\sigma_+$ and is smooth in $(\mu_j,\Sigma)$ with bounded derivatives there.
```

**Rationale**
The revision exposes the proof sequence: definition, per-axis bound, then product-level conclusion. The formulas and the dependence of the bound remain unchanged.

### C-022
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The warning after the main consistency display has two competing subjects—$\bar\Gamma_n$ and the inadmissible conditional expectation—and ends with the object controlled by the lemma.

**Before**
```text
The middle term is the function $\bar\Gamma_n(\cdot)$ evaluated at $\widehat p_n$ and must not be read as $\mathbb{E}_{i-1}[\psi_{\widehat p_n}(\xi_i)]$: $\widehat p_n$ depends on draws after $i$, is not $\mathcal{F}_{i-1}$-measurable, and that conditional expectation is not what Lemma~\ref{lem:uslln} controls.
```

**After**
```text
The middle term is the function $\bar\Gamma_n(\cdot)$ evaluated at $\widehat p_n$; it must not be read as $\mathbb{E}_{i-1}[\psi_{\widehat p_n}(\xi_i)]$. The reason is that $\widehat p_n$ depends on draws after $i$ and is not $\mathcal{F}_{i-1}$-measurable, so Lemma~\ref{lem:uslln} does not control that conditional expectation.
```

**Rationale**
The prohibited reading is stated before its measurability explanation. The second sentence makes the lemma’s scope the conclusion of the argument rather than a trailing qualification.

### C-023
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The deadline-bias explanation combines the direct censored-set effect and the indirect proposal-path effect in a single multiply embedded sentence.

**Before**
```text
The length bias of deadline-limited sampling (\S\ref{sec:limitations}) thus acts on the estimate in two ways: directly through the at most $W$ censored candidates, which the bound controls, and indirectly through the proposal sequence, whose archive is formed from arrived particles and so leans toward regions that simulate quickly; the second effect is part of $\tilde q_i$ and is corrected by the importance weighting up to $r$.
```

**After**
```text
The length bias of deadline-limited sampling (\S\ref{sec:limitations}) thus acts on the estimate in two ways. The bound controls the direct effect through the at most $W$ censored candidates; the indirect effect acts through the proposal sequence, whose archive is formed from arrived particles and so leans toward regions that simulate quickly. This second effect is part of $\tilde q_i$ and is corrected by the importance weighting up to $r$.
```

**Rationale**
The two routes are stated before their different treatments are described. “This second effect” supplies an explicit antecedent for the final correction claim.

### C-024
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The confidence-interval proof introduces an algebraic expansion and disposes of its cross terms in the same sentence, obscuring which convergence fact is being used.

**Before**
```text
In the numerator, $(f-\widehat\pi_n(f))^2=g^2+2g(\pi^r_{\epsilon_\infty}(f)-\widehat\pi_n(f))+(\pi^r_{\epsilon_\infty}(f)-\widehat\pi_n(f))^2$, and $W_{i,n}\le1/\delta$ with $\widehat\pi_n(f)\to\pi^r_{\epsilon_\infty}(f)$ a.s.\ (Corollary~\ref{cor:tilt}), so the cross terms vanish and it suffices to treat $\tfrac1n\sum_iW^2_{i,n}g(\theta_i)^2$.
```

**After**
```text
In the numerator, $(f-\widehat\pi_n(f))^2=g^2+2g(\pi^r_{\epsilon_\infty}(f)-\widehat\pi_n(f))+(\pi^r_{\epsilon_\infty}(f)-\widehat\pi_n(f))^2$. Because $W_{i,n}\le1/\delta$ and $\widehat\pi_n(f)\to\pi^r_{\epsilon_\infty}(f)$ a.s.\ (Corollary~\ref{cor:tilt}), the cross terms vanish, and it suffices to treat $\tfrac1n\sum_iW^2_{i,n}g(\theta_i)^2$.
```

**Rationale**
The algebra is completed before the boundedness and convergence argument is applied. The proof’s logical connective is made explicit without changing any notation.

I would run one final focused readability round after these edits, limited to the compiled Section 3 and Appendices A–B, because changing sentence and paragraph boundaries around displays can expose new antecedent or transition problems.