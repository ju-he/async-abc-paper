# Round 2 evaluation (Claude, independent reviewer)

File reviewed: latex/tmlr/tmlr-article.tex (984 lines, read in full).

## Scores

**AI-likeness: 3/10.** The main text now reads as one author's prose: short declarative sentences, every number tied to a table or figure, negative results stated flat (the Gaussian-mean loss, the 7-to-2987 ESS swing, the rate exponent at the threshold). What remains is structural rather than lexical: the same caveat restated in three or four places with slightly different wording, a few slogan openers in Appendix A, and a handful of vague verbs ("holds", "carries", "transferred", "knobs") that stand in for the specific relation.

**Readability: 7/10.** Sections 1-6 are easy to follow; the Method and Results subsections state one thing per sentence and the Discussion rules are compact. Appendix A ("Measured, in part." and "Do the reported runs meet the stabilization conditions?") and the parameter-coupled paragraph of Appendix D still pack three to five results into single sentences with the main clause after a parenthetical, and 6.2/6.3 and 5.2/6.1 each say the same thing twice within a page.

Passages that most drive the AI-likeness score:

1. The fidelity-ratio caveat is derived four times: Section 4 intro ("it is the only way the sampler enters the theorem"), the post-assumption paragraph ("the whole of the sampler's contribution ... up to a ratio $r$ that the results carry"), the "What $r$ is" paragraph ("Everything specific to the implementation enters through $r$"), and Table B.1 ("The one place the sampler enters"). Each restatement is phrased as a fresh insight.
2. Appendix A, "Measured, in part.": the sentence beginning "On the five Cellular Potts production replicates (about $12{,}900$ asynchronous evaluations ... so that (c) is live; Appendix~C)" holds the setting, a side-condition, the result and two sub-results before its main verb; the paragraph continues in the same shape for five more sentences.
3. Near-verbatim previews: "a crashed run is recovered by replaying its log" (Introduction and 3.1), the snapshot-buffer caveat (3.1 and 3.5), "the method a practitioner would run" (5.2) followed by "pyABC is what a practitioner would run instead" (6.1), and the Marin et al. restriction clause repeated word for word in Section 4 and Appendix A.

## What to leave alone

- Appendix B (proofs): every "which matters", "That is the substantive clause", "This is where ... is used" earns its place as mathematical signposting. I propose only one cleft merge there (C-020) and one duplicate clause (C-015); the logic and qualifiers stay.
- Scientific contrasts: "a property of the inference problem rather than of the machine" (5.1), "a reporting-support effect rather than a weighting failure" (6.4), "containment, not coverage" (6.4), "two-dimensional as a property of the model, not for want of screening" (Appendix C), "a statement about the algorithm as implemented, not about an idealized variant" (Remark B.1), "a decomposition of one quantity rather than four independent errors" (A.1). Each names a real alternative a reviewer would raise.
- The three-sentence weights paragraph in 3.1 ("The archive weights ... The parent weight ... Only the posterior weights ..."): parallel on purpose, three objects that must be told apart, and it replaced a meta-sentence.
- Negative results and limitations as written: the Gaussian-mean loss in Table 3 and 6.3, the erratic 4-D ESS, "the production runs sit at that threshold", "It remains a binding limitation in the regime the method is designed for", the unattributed duration ratio in Table D.4.
- Policy verbs: "binds"/"binding", "sits at the threshold", "turns on" (abstract, contribution 3, heading 6.3), per the author's decision.
- Previously decided items: "Identifiability does.", "The posterior does not follow.", the four italic Discussion rule titles, "it is the method a practitioner would run" in 5.2, the Appendix F "lifts" sentence, the "Measured, in part." heading, "delivers simulations predictably", the 5.1 triad and the abstract's last two sentences.
- Introduction paragraph 2 versus 3.1 paragraph 2: both say workers never wait and a crashed run is replayed from its log. An introduction previews; I would not cut either, and I flag it only as a pattern.
- Captions: "The barrier starves pyABC as $\sigma$ grows" (Fig. D.1) and the closing interpretation of Fig. D.4 are vivid but carry the encoding's meaning; the "therefore" sentence in the Fig. D.4 caption contains a \ref, so I leave it.
- "Lotka--Volterra is an adversarial simulator for a scaling study", "the practical knee", the Pareto paragraph's "one might expect ... that expectation does not hold": ordinary technical register, and the expectation is a real one.
- "The asynchronous sampler itself scales almost linearly" (6.1): "itself" is contrastive with the twin here, not emphasis.
- The Discussion's four rules re-derive 6.1-6.4 in brief; that is what a rules paragraph is for.
- Table B.1 restates Appendix A findings cell by cell; a status table is meant to be read alone.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason: 6.2 states that both ratios rise with simulation cost; 6.3 opens by stating the same from the figure. The rising-with-cost result is C3's content and belongs in 6.3.

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

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason: 5.2 already says pyABC "is the method a practitioner would run" (kept by author decision); 6.1 repeats it as the paragraph opener.

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

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason: Third statement of the "sampler enters only through the fidelity ratio" caveat in Section 4, with two vague verbs ("holds the remaining conditions", "a ratio $r$ that the results carry").

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

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason: The clause "restrict the adaptation so that the weighting stage decouples, which a single-arrival scheme does not do" appears verbatim in Section 4 and again here; the sentence is also three clauses long.

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

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Main clause buried after a four-part parenthetical; setting, side-condition and three numbers in one sentence (Appendix A, flagged by both round-1 verifiers).

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

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Four results and a nested parenthetical interpretation in one sentence (Appendix A, "Measured, in part.").

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

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Two different points (an observation and a caveat on what it proves) joined by a semicolon into one long sentence (Appendix A).

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

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Two fits and the reason for the second one are stacked into one sentence (Appendix A, stabilization paragraph).

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

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: A 50-word gerund subject ("running a ... benchmark ..., once with ..., and once with ...") before the verb "moves" (Appendix A).

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

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Two separate findings (the rule works; re-reporting is the check) and an aside ("at no simulation cost") in one sentence (6.4, starting bandwidth).

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

### C-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: One sentence carries the result, the comparator, a figure pointer, and four robustness conditions (Appendix D, parameter-coupled runtime).

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

### C-012
- file: latex/tmlr/tmlr-article.tex
- category: punchline
- severity: M
- status: proposed
- reason: A slogan opener that restates what item (b) above already said ("Fixing $m$ therefore trades cost against fidelity") and what the paragraph heading announces.

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

### C-013
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- status: proposed
- reason: The paragraph's first sentence already reports the unweighted archive's error at ≈0.01 for every coupling; the closing sentence restates it before the AMIS point.

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

### C-014
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- status: proposed
- reason: The snapshot-buffer caveat is stated in the 3.1 overview and again, in the same words ("changes which parents are chosen, not the estimate"), in 3.5 where the buffer is defined.

**Before**
```text
The one exception is a small buffer of past proposals that the running sampler keeps for the weight of step 6; it affects which parents are chosen next, not the estimate.
```

**After**
```text
```

**Rationale**
3.5 is where the kept snapshots are defined and is the natural home of the caveat; the overview paragraph closes on "the same whether it is formed during the run or afterwards", which still holds since the buffer is per-worker and not shared.

### C-015
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: L
- status: proposed
- reason: "Particles are indexed by proposal time" is stated two sentences earlier in the same paragraph (Appendix B, Setting).

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

### C-016
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
- reason: Two measured rates in one sentence (Appendix A, stabilization paragraph); the second carries its own interpretation.

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

### C-017
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "transferred" is a vague verb for what 6.4 states plainly ("recovers the posterior").

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

### C-018
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "knobs" is product vocabulary in an otherwise technical paragraph; the manuscript calls $k$ and $S$ "settings" everywhere else.

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

### C-019
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: L
- status: proposed
- reason: "is carried as $r$ inside the statements" uses the vague verb "carry" for a specific relation (Appendix A, opening paragraph).

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

### C-020
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: L
- status: proposed
- reason: "This indexing is why ..." is a cleft; Appendix B's clefts are fair game per the brief, and the two sentences state cause and effect that one sentence can carry.

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
