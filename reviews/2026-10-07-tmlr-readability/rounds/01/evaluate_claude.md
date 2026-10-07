# Round 01, reviewer: Claude (fresh subagent), 2026-10-07

File in scope: `latex/tmlr/tmlr-article.tex` (read in full). Focus as requested: Section 3, the Limitations paragraph, the Conclusion, Appendix A and Appendix B.

## Scores

**AI-likeness: 3/10.** The prose has a recognisable authorial voice: short deliberate sentences ("Identifiability does."), unfavorable numbers stated without cushioning, and contrasts that almost always carry a scientific distinction. What remains of the generated register is a handful of "is where" clefts and a refrain ("the only way the sampler enters", "the only place the sampler's proposal path enters", "Everything specific to the implementation enters through r") that recurs across Section 3, Appendix B and the status table.

**Readability: 6/10.** The main text through Section 6 reads cleanly; the theory section and the two theory appendices are where a reader has to parse twice. The specific failure modes are a verbless assumption clause, three or four sentences past 55 words that hang two independent points on a semicolon, pronouns whose antecedent is a different noun from the nearest one, and symbols reused for unrelated objects ($w_i$ as worker index and as parent weight; $w_n$ where the text means $\lambda_n$; $r$ as an integer index inside the proof of the uniform SLLN).

The three passages that most drive the AI-likeness score:

1. Section 3, opening paragraph: "it absorbs everything the implementation does, and it is the only way the sampler enters the theorem", restated in the assumptions paragraph ("is the only place the sampler's proposal path enters"), in Remark B.5 ("statements about the algorithm as implemented, not about an idealized variant"), and in the status table ("The one place the sampler enters for Corollary 2"). The same reassurance four times reads as template.
2. The "is where" cleft cluster: "This step is where the rates are used" (proof of Theorem 3), "which is where the box form of $\Theta$ in Assumption 2 enters" (Appendix B class paragraph), "The effective-sample-size ceiling is where the method can still improve" (Discussion).
3. Appendix A, the decomposition paragraph: "the factors are a decomposition of one quantity rather than four independent errors" followed by "(a) ... is the one contribution available in closed form", "(b) ... this controls the third factor above, not the fidelity of the whole", "$Z_j\approx10^{-3}$ is the scale below which that residual becomes appreciable, not a threshold above which it vanishes": four "X, not Y" closers in one page. Each is individually informative; the density is the pattern.

## What to leave alone

- Every item on the author's standing list (the two short lines, the italic Discussion rules, "the method a practitioner would run", the Appendix F "lifts" sentence, "Measured, in part.", the question-form Appendix A headings, "binds"/"sits at the threshold"/"turns on", "the weight of step 6", the barrier framing in the final Conclusion paragraph, the proposal-time sentence before the display in Appendix B, "trivial" for the Lindeberg step).
- The Conclusion: four sentences, each bound to a result, no restatement. Nothing to do.
- The Limitations paragraph apart from one antecedent (C-010): the plain statement "we make no asymptotic claim for it" and "the production runs sit at the threshold of its rate condition rather than clearly inside it" are the most human-sounding lines in the paper.
- The contrasts in the theory that carry content: "a property of the execution, not of the proposal path" (Assumption 4 versus Assumption 5), "unit peak, not unit integral", "through the variation of $r$ over posterior mass, not through its distance from one" (this is the mathematical content of the identity), "a diagnostic rather than an asymptotic statement", "containment, not coverage". All scientific.
- The "price of running one simulation per proposed parameter" after Theorem 3: cost language that names a variance term; keep.
- The refrain in passage 1 above: I did not propose cutting any instance, because each occurrence sits in a different scope (theorem, assumption, remark, table row) and a reader entering at that point needs it. I flag it so the author can decide whether the Remark B.5 instance is the one to drop.
- The "is where" clefts: proof-text rhythm is out of scope and the Discussion one is a one-line topic sentence; not worth a proposal.
- The long sentence in Assumption 4 (B3) and the "class used below" paragraph in Appendix B: dense but each clause is doing work, and a split would not help a reader who is following the measure theory.
- The Appendix B sentence after Proposition B.6 that repeats the Limitations caveat (direct part through at most $W$ in-flight candidates, indirect part through the proposal sequence): the appendix is the primary statement and the Limitations paragraph cross-references it; the apparent duplication is the intended reference structure.
- Terminology: "asynchronous sampler", "pyABC", "twin", "generation barrier", "fidelity ratio", "archive" used consistently; no "arm" anywhere.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: Assumption 5(ii) is a verbless list; the reader cannot tell which items are hypotheses and which is the pointer to the appendix.

**Before**
```text
\emph{(ii, central limit theorem)} $\epsilon_\infty>0$ and $r$ deterministic, a stabilization rate and limits of the proposal and the denominator, stated in Appendix~\ref{app:theory-more}; clause (ii) implies clause (i).
```

**After**
```text
\emph{(ii, central limit theorem)} In addition, $\epsilon_\infty>0$ and $r$ are deterministic, the proposal and the denominator have limits, and they converge at a stabilization rate stated in Appendix~\ref{app:theory-more}; clause (ii) implies clause (i).
```

**Rationale**
A sentence that cannot be parsed on first reading; the After gives each hypothesis a verb and keeps the pointer and the implication between the clauses unchanged.

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: One sentence of about 120 words spanning a display; three logical steps (apply the lemma, substitute and compute, pass to the limit) are chained by a semicolon and an "and".

**Before**
```text
Apply Lemma~\ref{lem:uslln} on $P_{a}$ with $a=\epsilon_\infty$ (deterministic by Assumption~\ref{ass:stab}(ii), so no localization is needed) to the class $\psi_{(\epsilon,p_q)}(\theta,\rho)=\pi(\theta)^2g(\theta)^2K_\epsilon(\rho)^2/q_{p_q}(\theta)^2$, which is bounded by $\|g\|^2_\infty/\delta^2$ and Lipschitz in $(\epsilon,p_q)$ by the argument of \eqref{eq:app-class} with $2L_K(\epsilon_\infty)$ in place of $L_K$; the substitution of $\widehat p_n$ used in \eqref{eq:app-consistency-main}, followed by Lemma~\ref{lem:condmean} with $\ell=2$, $\phi=\pi^2g^2$ (a fixed function under Assumption~\ref{ass:stab}(ii)), gives
$\tfrac1n\sum_iW^2_{i,n}g^2(\theta_i)=\int\pi^2g^2L^{(2)}_{\epsilon_n}\,\bigl(\bar q^\star_n/\bar q_n\bigr)\,\bar q_n^{-1}+o(1)$,
and $\epsilon_n\to\epsilon_\infty$, $\|\bar q^\star_n/\bar q_n-r\|_\infty\to0$ (clause (i), implied by (ii)) and $\|\bar q_n-\bar q_\infty\|_\infty\to0$ (the first line of \eqref{eq:rate}) send this to $\int\pi^2g^2L^{(2)}_{\epsilon_\infty}r/\bar q_\infty=\int\pi^2g^2L^{(2)}_{\epsilon_\infty}\tilde q_\infty/\bar q^2_\infty=v^2$ a.s., using $r=\tilde q_\infty/\bar q_\infty$.
```

**After**
```text
Apply Lemma~\ref{lem:uslln} on $P_{a}$ with $a=\epsilon_\infty$ (deterministic by Assumption~\ref{ass:stab}(ii), so no localization is needed) to the class $\psi_{(\epsilon,p_q)}(\theta,\rho)=\pi(\theta)^2g(\theta)^2K_\epsilon(\rho)^2/q_{p_q}(\theta)^2$, which is bounded by $\|g\|^2_\infty/\delta^2$ and Lipschitz in $(\epsilon,p_q)$ by the argument of \eqref{eq:app-class} with $2L_K(\epsilon_\infty)$ in place of $L_K$. The substitution of $\widehat p_n$ used in \eqref{eq:app-consistency-main}, followed by Lemma~\ref{lem:condmean} with $\ell=2$, $\phi=\pi^2g^2$ (a fixed function under Assumption~\ref{ass:stab}(ii)), gives
$\tfrac1n\sum_iW^2_{i,n}g^2(\theta_i)=\int\pi^2g^2L^{(2)}_{\epsilon_n}\,\bigl(\bar q^\star_n/\bar q_n\bigr)\,\bar q_n^{-1}+o(1)$.
Then $\epsilon_n\to\epsilon_\infty$, $\|\bar q^\star_n/\bar q_n-r\|_\infty\to0$ (clause (i), implied by (ii)) and $\|\bar q_n-\bar q_\infty\|_\infty\to0$ (the first line of \eqref{eq:rate}) send this to $\int\pi^2g^2L^{(2)}_{\epsilon_\infty}r/\bar q_\infty=\int\pi^2g^2L^{(2)}_{\epsilon_\infty}\tilde q_\infty/\bar q^2_\infty=v^2$ a.s., using $r=\tilde q_\infty/\bar q_\infty$.
```

**Rationale**
Only two punctuation changes ("; the substitution" to ". The substitution"; "$+o(1)$, and" to "$+o(1)$. Then"); every formula is untouched, and the three steps of the argument now start at sentence boundaries.

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: "both factors are active" has no antecedent in its sentence; the reader must go back to "Two of the four factors" two sentences earlier and past "(c) is live" to recover that "both" means (a) and (b).

**Before**
```text
The five Cellular Potts production replicates (Appendix~\ref{app:cpm}) have about $12{,}900$ asynchronous evaluations each across $48$ workers and two correlated dimensions, so that (c) is live, and both factors are active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**After**
```text
The five Cellular Potts production replicates (Appendix~\ref{app:cpm}) have about $12{,}900$ asynchronous evaluations each across $48$ workers and two correlated dimensions, so that (c) is live. Both measured factors, (a) and (b), are active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**Rationale**
Names the antecedent and separates the "(c) is live" aside from the measurement it was interrupting; the numbers and their attribution are unchanged.

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: $w_n$ is the parent-weight symbol of \eqref{eq:history}; the quantity floored at $\delta$ in \eqref{eq:snapshot-denominator} is $\lambda_n=\max(\nu_n,\delta)$. A reader checking the claim against the display finds no $w_n$ there. Author to confirm that $\lambda_n$ is the intended symbol; this is a one-symbol notation fix, not a change to any equation.

**Before**
```text
When genuine bootstrap draws carry less than a $\delta$ share of the history, $w_n=\delta>\nu_n$ and the denominator holds more prior mass than the sample does.
```

**After**
```text
When genuine bootstrap draws carry less than a $\delta$ share of the history, $\lambda_n=\delta>\nu_n$ and the denominator holds more prior mass than the sample does.
```

**Rationale**
A symbol used for a quantity it was not defined for; the After uses the symbol the display defines, and nothing else moves.

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "which exists because the schedule never loosens and may be zero" reads on first pass as "the schedule ... may be zero"; the subject of "may be zero" is the limit.

**Before**
```text
Throughout, $\epsilon_\infty:=\lim_n\epsilon_n$, which exists because the schedule never loosens and may be zero; the asymptotic statements hold on the event $\{\epsilon_\infty>0\}$, and the closing paragraph of this section says which configurations put a run there.
```

**After**
```text
Throughout, $\epsilon_\infty:=\lim_n\epsilon_n$, which exists because the schedule never loosens, and which may be zero; the asymptotic statements hold on the event $\{\epsilon_\infty>0\}$, and the closing paragraph of this section says which configurations put a run there.
```

**Rationale**
A misattached clause; repeating "which" pins "may be zero" to the limit. Nothing else changes.

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: A 25-word compound subject ("A central limit theorem ..., with a per-run variance estimator, and the decomposition ...") before the verb "are in"; the reader does not know the sentence is a pointer until its end.

**Before**
```text
A central limit theorem for variants whose adaptation is damped, with a per-run variance estimator, and the decomposition and partial measurement of $r$ are in Appendix~\ref{app:theory-more}; the proofs are in Appendix~\ref{app:theory}.
```

**After**
```text
Appendix~\ref{app:theory-more} gives a central limit theorem for variants whose adaptation is damped, a per-run variance estimator, and the decomposition and partial measurement of $r$; the proofs are in Appendix~\ref{app:theory}.
```

**Rationale**
Front-loads the point (where the material is) and turns the three items into a plain list; same content, same cross-references.

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The sentence that states the main result is 54 words with four subordinate clauses ("with an error that", "that it converges", "when that limit exists", "so that"); the finite-run and limit statements are two different claims and deserve two sentences.

**Before**
```text
We prove that the estimate tracks the smooth-ABC posterior at the realized bandwidth tilted by $r_n$, with an error that vanishes whatever the sampler does, and that it converges to the posterior tilted by the limit $r$ of $r_n$ when that limit exists, so that $r\equiv1$ gives the posterior itself.
```

**After**
```text
We prove that the estimate tracks the smooth-ABC posterior at the realized bandwidth tilted by $r_n$, with an error that vanishes whatever the sampler does. When $r_n$ has a limit $r$, the estimate converges to the posterior tilted by $r$, so that $r\equiv1$ gives the posterior itself.
```

**Rationale**
Splits at the boundary between Theorem 1 and Corollary 2, which is how the section is organised; the "so that" connective and both claims are preserved.

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "its two conditions" follows "the estimate" and "the sample" as candidate antecedents; the conditions belong to the theorem.

**Before**
```text
The estimate thus behaves as if its denominator were the density the sample was drawn from, up to the tilt $r_n$, whatever the proposal path; its two conditions, a bandwidth away from zero and an average weight that does not collapse, are observable from a run.
```

**After**
```text
The estimate thus behaves as if its denominator were the density the sample was drawn from, up to the tilt $r_n$, whatever the proposal path; the theorem's two conditions, a bandwidth away from zero and an average weight that does not collapse, are observable from a run.
```

**Rationale**
A buried antecedent; one noun replaces the pronoun and the sentence otherwise stands.

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: 58 words contrasting two regimes across a "whereas"; the reader loses the first regime by the time the second arrives.

**Before**
```text
The effective-sample-size ceiling of \S\ref{sec:results-posterior} is the signature of this regime: on $\{\epsilon_\infty>0\}$ with a settled ratio the average weight converges to a positive constant and the effective sample size grows linearly in $n$, whereas with a bandwidth tied to the $2k$-th order statistic only $O(k)$ particles ever carry appreciable kernel mass.
```

**After**
```text
The effective-sample-size ceiling of \S\ref{sec:results-posterior} is the signature of this regime: on $\{\epsilon_\infty>0\}$ with a settled ratio the average weight converges to a positive constant and the effective sample size grows linearly in $n$. With a bandwidth tied to the $2k$-th order statistic, by contrast, only $O(k)$ particles ever carry appreciable kernel mass.
```

**Rationale**
A split that keeps the contrastive connective ("by contrast") and the scientific contrast between the two regimes; no claim moves.

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "the boundary term" is not named anywhere in the Limitations paragraph; the preceding sentence calls it "the direct part of that effect ... an $O(W/n)$ perturbation". A reader has to guess that "boundary" is the same object.

**Before**
```text
A runtime-coupled study found no measurable shift of the posterior mean at the couplings tested, and we do not correct for the boundary term (Appendix~\ref{app:moved}).
```

**After**
```text
A runtime-coupled study found no measurable shift of the posterior mean at the couplings tested, and we do not correct for the $O(W/n)$ term (Appendix~\ref{app:moved}).
```

**Rationale**
Reuses the name the paragraph itself gave the term two sentences earlier; the limitation is stated exactly as before.

### C-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "bounded above because ... and below because $\tilde q_\infty$ is" requires the reader to supply "bounded" twice and to work out that the subject is $r$, not the display. (This Before is the tail of the sentence that follows the display; the display itself is untouched.)

**Before**
```text
which is \eqref{eq:fidelity} with $r=\tilde q_\infty/\bar q_\infty$, bounded above because a uniform limit of densities on a compact box is bounded and below because $\tilde q_\infty$ is.
```

**After**
```text
which is \eqref{eq:fidelity} with $r=\tilde q_\infty/\bar q_\infty$; $r$ is bounded above because a uniform limit of densities on a compact box is bounded, and bounded below because $\tilde q_\infty$ is.
```

**Rationale**
Names the subject and makes the elided "bounded" explicit; the two reasons are unchanged.

### C-012
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "it is exactly $1$" immediately follows the formula for $r_{\mathrm{floor}}-1$, so the nearest antecedent of "it" is the wrong quantity (which is $0$, not $1$, in that case).

**Before**
```text
With $u=q_\infty/\pi$ the limiting proposal-to-prior ratio, this contributes
$r_{\mathrm{floor}}=\bigl(\nu_\infty+(1-\nu_\infty)u\bigr)/\bigl(\delta+(1-\delta)u\bigr)$, so $r_{\mathrm{floor}}-1=(\delta-\nu_\infty)(u-1)/(\delta+(1-\delta)u)$: it is exactly $1$ when $\nu_\infty\ge\delta$, below $1$ wherever the proposal is thinner than the prior, and $r_{\mathrm{floor}}\to1$ as $\delta\to0$, i.e.\ as $m$ grows.
```

**After**
```text
With $u=q_\infty/\pi$ the limiting proposal-to-prior ratio, this contributes
$r_{\mathrm{floor}}=\bigl(\nu_\infty+(1-\nu_\infty)u\bigr)/\bigl(\delta+(1-\delta)u\bigr)$, so $r_{\mathrm{floor}}-1=(\delta-\nu_\infty)(u-1)/(\delta+(1-\delta)u)$: $r_{\mathrm{floor}}$ is exactly $1$ when $\nu_\infty\ge\delta$, below $1$ wherever the proposal is thinner than the prior, and $r_{\mathrm{floor}}\to1$ as $\delta\to0$, i.e.\ as $m$ grows.
```

**Rationale**
Replaces one pronoun with its referent; the formulas are untouched.

### C-013
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "the third factor above" forces the reader to count positions in a display whose factors are labeled (d), (c), (b), (a) from left to right; the label is the cue the paragraph itself uses.

**Before**
```text
The $m$-point mixture stands in for the $n$-point one, with error at most $V_n/m$ where $V_n=\sum_{i<n}\|q_{i+1}-q_i\|_\infty$ is the total variation of the \emph{reconstructed} proposal path; this controls the third factor above, not the fidelity of the whole.
```

**After**
```text
The $m$-point mixture stands in for the $n$-point one, with error at most $V_n/m$ where $V_n=\sum_{i<n}\|q_{i+1}-q_i\|_\infty$ is the total variation of the \emph{reconstructed} proposal path; this controls factor (b), not the fidelity of the whole.
```

**Rationale**
A cross-reference by label replaces a positional count; the contrast (one factor, not the whole) is scientific and stays.

### C-014
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: $W_j$ and $Z_j$ appear in a formula without being named; $Z_j$ is only explained by its use three sentences later, and $W_j$ never.

**Before**
```text
A parent's perturbation is rejection-sampled into the box for at most $992$ attempts, so component $j$ is realized with probability $W_j[1-(1-Z_j)^{992}]$ and the residual is emitted as a prior draw; and a candidate whose proposal-time denominator falls below $10^{-12}$ is redrawn up to five times, the last one being emitted regardless.
```

**After**
```text
A parent's perturbation is rejection-sampled into the box for at most $992$ attempts, so component $j$, with mixture weight $W_j$ and in-box mass $Z_j$, is realized with probability $W_j[1-(1-Z_j)^{992}]$ and the residual is emitted as a prior draw; and a candidate whose proposal-time denominator falls below $10^{-12}$ is redrawn up to five times, the last one being emitted regardless.
```

**Rationale**
Notation introduced with a reading cue at first use; the formula and the numbers are unchanged.

### C-015
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "narrower in nine of ten" has no noun; the reader has to reconstruct that the ten are the marginals (five replicates, two parameters). Author to confirm the denominator.

**Before**
```text
Posterior means agree within $0.012$ standard deviations and marginal widths within $1\%$, narrower in nine of ten, which is the tail-suppressing direction that (a) predicts; the effective sample size is unchanged.
```

**After**
```text
Posterior means agree within $0.012$ standard deviations and marginal widths within $1\%$, narrower in nine of the ten marginals, which is the tail-suppressing direction that (a) predicts; the effective sample size is unchanged.
```

**Rationale**
Supplies the missing noun; the count and the interpretation are as before.

### C-016
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: 70 words carrying two independent points (the bound is loose; the identity says what is being estimated) on one semicolon, with a parenthetical of four numbers in the middle.

**Before**
```text
The bound in \eqref{eq:tilt-bound} is loose against both direct measurements by an order of magnitude ($3.6$--$4.0\%$ against $0.2$--$0.5\%$ on Cellular Potts, $4.4\%$ against $0.26\%$ on the one-dimensional history), as sup-norm-free bounds valid for every $f$ generally are; the identity in \eqref{eq:tilt-bound} says what the direct measurements estimate, namely the centered absolute moment of the removed factors under the reference, divided by twice their mean.
```

**After**
```text
The bound in \eqref{eq:tilt-bound} is loose against both direct measurements by an order of magnitude ($3.6$--$4.0\%$ against $0.2$--$0.5\%$ on Cellular Potts, $4.4\%$ against $0.26\%$ on the one-dimensional history), as sup-norm-free bounds valid for every $f$ generally are. The identity in \eqref{eq:tilt-bound} says what the direct measurements estimate: the centered absolute moment of the removed factors under the reference, divided by twice their mean.
```

**Rationale**
One split at the semicolon and "namely" replaced by a colon; every number and both references stay.

### C-017
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "(not $r\equiv1$ itself, since the other three factors remain)" reads as if $r\equiv1$ were something being removed; the intended meaning is that removing factor (b) does not yield $r\equiv1$.

**Before**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$ (not $r\equiv1$ itself, since the other three factors remain), and the proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

**After**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$ (it does not give $r\equiv1$, since the other three factors remain), and the proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

**Rationale**
A parenthetical that could not be parsed is rewritten as a clause with a verb; also drops an "itself" used for emphasis. The claim is identical.

### C-018
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "send $\delta\to0$ and with it the uniform weight bound $1/\delta$" says the bound goes to zero; it goes to infinity, i.e. is lost. The reader stops to resolve the contradiction.

**Before**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$, so growing $m$ there would send $\delta\to0$ and with it the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}, which makes the Lindeberg step trivial and fixes the constants in the law of large numbers; the rate conditions would have to be redone.
```

**After**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$, so growing $m$ there would send $\delta\to0$ and lose the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}, which makes the Lindeberg step trivial and fixes the constants in the law of large numbers; the rate conditions would have to be redone.
```

**Rationale**
"lose" states what happens to the bound; the rest of the sentence, including the two roles of the bound, is unchanged.

### C-019
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: $w_i$ was defined in \S3.2 as the parent weight stored in the history and is used in that sense throughout the method section; the lemma reuses it as a worker index without warning, and the filtration $\sigma((\theta_j,\rho_j,D_j,w_j)_{j\le i},\mathcal{E})$ then looks as if it contained the parent weights.

**Before**
```text
Let Assumption~\ref{ass:filtration} hold, let $w_i$ be the worker that proposes candidate $i$ and $D_i$ the duration of its simulation, and let $\mathcal{F}_i:=\sigma\bigl((\theta_j,\rho_j,D_j,w_j)_{j\le i},\,\mathcal{E}\bigr)$, where $\mathcal{E}$ is the $\sigma$-field of the exogenous scheduling randomness of (B3).
```

**After**
```text
Let Assumption~\ref{ass:filtration} hold, let $w_i$ be the worker that proposes candidate $i$ (in this lemma and its proof $w_i$ is a worker index, not the parent weight of the history) and $D_i$ the duration of its simulation, and let $\mathcal{F}_i:=\sigma\bigl((\theta_j,\rho_j,D_j,w_j)_{j\le i},\,\mathcal{E}\bigr)$, where $\mathcal{E}$ is the $\sigma$-field of the exogenous scheduling randomness of (B3).
```

**Rationale**
A reading cue of the kind the manuscript already uses for $\ell$ in Lemma B.4 ("an exponent, not the fidelity ratio $r$"); renaming the symbol would touch the proof and the status table, so the cue is the minimal fix.

### C-020
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: $r$ is the fidelity ratio everywhere else in the paper; here it is an integer index, in a proof the reader reaches directly after a page about $r$.

**Before**
```text
Intersecting the almost-sure events over the countable family $t=1/r$, $\eta=1/j$ with $r,j\in\mathbb{N}$, gives the claim.
```

**After**
```text
Intersecting the almost-sure events over the countable family $t=1/r$, $\eta=1/j$ with $r,j\in\mathbb{N}$ (here $r$ is an integer index, not the fidelity ratio), gives the claim.
```

**Rationale**
Same device as Lemma B.4's parenthetical for $\ell$; the alternative, renaming the index, is also acceptable but changes two symbols.

### C-021
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: "write $A_n=a_n+o(1)$ and $B_n=b_n+o(1)$ for the two weighted averages and the two integrals" leaves the reader to guess which letters are the averages and which the integrals; the next sentence depends on it.

**Before**
```text
For \eqref{eq:tracking-ratio}, apply \eqref{eq:tracking} with $h=f$ and with $h\equiv1$, and write $A_n=a_n+o(1)$ and $B_n=b_n+o(1)$ for the two weighted averages and the two integrals.
```

**After**
```text
For \eqref{eq:tracking-ratio}, apply \eqref{eq:tracking} with $h=f$ and with $h\equiv1$, and write $A_n=a_n+o(1)$ and $B_n=b_n+o(1)$, with $A_n,B_n$ the two weighted averages and $a_n,b_n$ the two integrals.
```

**Rationale**
Notation given a reading cue at the point of introduction; nothing else in the proof changes.

### C-022
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: A 60-word sentence hanging off a display, with "using ..., and where ..." chaining two justifications and a definition; "where $R$" arrives long after $R$ appeared in the display. (This Before is the tail of the sentence that follows the display; the display itself is untouched.)

**Before**
```text
on the event $\{1/j\le\epsilon_\infty\}$ of the localization above, using $|L_{\epsilon_n}-L_{\epsilon_\infty}|\le L_K(1/j)|\epsilon_n-\epsilon_\infty|$ (Assumption~\ref{ass:kernel} and Jensen; every $\epsilon_n$ and $\epsilon_\infty$ lies in $[1/j,\epsilon_0]$ there) and $L_{\epsilon_\infty}\le1$, and where $R=r_++1$ bounds $\bar q^\star_n/\bar q_n$ for all large $n$ by \eqref{eq:fidelity}, with no structural bound on $\tilde q_i$ needed.
```

**After**
```text
on the event $\{1/j\le\epsilon_\infty\}$ of the localization above, using $|L_{\epsilon_n}-L_{\epsilon_\infty}|\le L_K(1/j)|\epsilon_n-\epsilon_\infty|$ (Assumption~\ref{ass:kernel} and Jensen; every $\epsilon_n$ and $\epsilon_\infty$ lies in $[1/j,\epsilon_0]$ there) and $L_{\epsilon_\infty}\le1$. Here $R=r_++1$ bounds $\bar q^\star_n/\bar q_n$ for all large $n$ by \eqref{eq:fidelity}, with no structural bound on $\tilde q_i$ needed.
```

**Rationale**
One split so that the definition of $R$ is its own sentence; the justifications and references are as before.

### C-023
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Two "because" clauses in one sentence, the second explaining the first; the reader has to work out which "because" governs which claim.

**Before**
```text
Because the convergence in Theorem~\ref{thm:clt} is stable, the pair $(\sqrt n(\widehat\pi_n(f)-\pi^r_{\epsilon_\infty}(f)),\widehat\sigma_n)$ converges jointly to $(\sigma_f\mathcal{Z},\sigma_f)$, because stable convergence is equivalent to joint convergence with every $\mathcal{F}_\infty$-measurable variable and $\sigma_f$ is one.
```

**After**
```text
Because the convergence in Theorem~\ref{thm:clt} is stable, the pair $(\sqrt n(\widehat\pi_n(f)-\pi^r_{\epsilon_\infty}(f)),\widehat\sigma_n)$ converges jointly to $(\sigma_f\mathcal{Z},\sigma_f)$: stable convergence is equivalent to joint convergence with every $\mathcal{F}_\infty$-measurable variable, and $\sigma_f$ is one.
```

**Rationale**
The second "because" becomes a colon, which is what it was doing; the argument is unchanged.

### C-024
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- status: proposed
- reason: The status-table row re-derives the order-statistic argument of the "Which runs the limits describe" paragraph almost verbatim ("holds the bandwidth until $2k$ particles lie inside it, so $\epsilon_n$ tracks an order statistic that tends to the infimum of the discrepancy support, zero for a stochastic simulator; the effective-sample-size ceiling is the visible consequence") while already pointing to \S3 for it.

**Before**
```text
The runs reported here leave \texttt{min\_tol} unset (zero is accepted), and their scheduler holds the bandwidth until $2k$ particles lie inside it, so $\epsilon_n$ tracks an order statistic that tends to the infimum of the discrepancy support, zero for a stochastic simulator; the effective-sample-size ceiling is the visible consequence (\S\ref{sec:theory}).
```

**After**
```text
The runs reported here leave \texttt{min\_tol} unset (zero is accepted); their scheduler then tracks an order statistic of the discrepancies that tends to zero for a stochastic simulator, with the effective-sample-size ceiling as the visible consequence (\S\ref{sec:theory}).
```

**Rationale**
Same caveat stated once in full (\S3) and referenced here; the row keeps its verdict, the mechanism in one clause, and the cross-reference.

### C-025
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Contribution 4 is a 65-word sentence that switches from a noun phrase ("A martingale analysis ... showing that") to a finite clause ("we derive ...") across a semicolon; the reader loses the list structure of the enumeration.

**Before**
```text
\item A martingale analysis of the estimator showing that it targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when the ratio tends to one; we derive from the execution model the adaptedness that asynchronous execution requires, and decompose the ratio and measure part of it (\S\ref{sec:theory}; proofs, a deadline-censoring bound and a central limit theorem in the appendix).
```

**After**
```text
\item A martingale analysis of the estimator showing that it targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when the ratio tends to one. We derive from the execution model the adaptedness that asynchronous execution requires, and decompose the ratio and measure part of it (\S\ref{sec:theory}; proofs, a deadline-censoring bound and a central limit theorem in the appendix).
```

**Rationale**
A split at the semicolon where the grammar already changes; the item keeps all three sub-claims and its references.

---

Another round: no, not for readability. These 25 items exhaust the sentences in Sections 3, 7, 8 and Appendices A and B that a reader has to parse twice; what remains is proof density that a split would not improve, and the refrain and cleft patterns noted under Scores, which the author can judge in one pass from this list without a further reviewer round.
