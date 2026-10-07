# Round proposals (merged)

- round: 01
- date: 2026-10-07

### P-001
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: claude
- status: applied
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

### P-002
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: claude
- status: applied
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

### P-003
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: claude
- status: applied
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

### P-004
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: claude
- status: applied
- reason: accepted; notation fix verified against the snapshot-denominator display: the floored prior weight is lambda_n, and w_n occurs nowhere else; AUTHOR TO CONFIRM | reviewer note: $w_n$ is the parent-weight symbol of \eqref{eq:history}; the quantity floored at $\delta$ in \eqref{eq:snapshot-denominator} is $\lambda_n=\max(\nu_n,\delta)$. A reader checking the claim against the display finds no $w_n$ there. Author to confirm that $\lambda_n$ is the intended symbol; this is a one-symbol notation fix, not a change to any equation.

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

### P-005
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-007
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-012
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-013
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-014
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-015
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
- reason: accepted; the ten are five replicates x two parameters; AUTHOR TO CONFIRM the noun | reviewer note: "narrower in nine of ten" has no noun; the reader has to reconstruct that the ten are the marginals (five replicates, two parameters). Author to confirm the denominator.

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

### P-016
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-017
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-018
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-019
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-020
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-021
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-022
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-023
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-024
- file: latex/tmlr/tmlr-article.tex
- category: duplication
- severity: M
- source: claude
- status: applied
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

### P-025
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: claude
- status: applied
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

### P-026
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-001
- status: rejected
- reason: duplicate of P-001 (Claude version keeps the hypotheses as the subject; the codex version makes the appendix the subject) | reviewer note: The central-limit clause of Assumption 5 is grammatically incomplete and leaves the reader to infer the governing verb.

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

### P-027
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-007
- status: rejected
- reason: duplicate of P-007 (Claude version: one split, same connectives) | reviewer note: The main theoretical result and its limiting corollary are packed into one sentence with three levels of qualification.

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

### P-028
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: H
- source: codex
- duplicate-of: P-009
- status: rejected
- reason: duplicate of P-009 (whole-paragraph restructure; the single split of P-009 is the minimal fix) | reviewer note: The paragraph’s operational conclusion—that the production run is outside the asymptotic regime—arrives only after a long description of the bandwidth schedule.

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

### P-029
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: H
- source: codex
- duplicate-of: P-010
- status: rejected
- reason: duplicate of P-010 (paragraph split of Limitations overlaps the P-010 sentence; may be re-proposed on its own next round) | reviewer note: The Limitations paragraph combines three independent limitations—the fidelity ratio, the CLT rate, and deadline length bias—before any one of them is fully resolved.

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

### P-030
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
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

### P-031
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
- reason: applied, then amended after verification: both verifiers rated the applied wording regressed (tautological 'In calibration experiments, the estimator is calibrated') and both proposed 'Empirically, it is calibrated where ...'; that wording is now in the file | The abstract joins a theorem, its exact-fidelity condition, and an empirical calibration comparison in a single sentence.

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

### P-032
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- duplicate-of: P-025
- status: rejected
- reason: duplicate of P-025 (Claude version: split at the semicolon only, references stay where they were) | reviewer note: Contribution C4 combines the theorem, execution model, fidelity accounting, proofs, calibration, recovery, and two tuning limitations in two heavily stacked sentences.

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

### P-033
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- source: codex
- status: applied
- reason: applied, then amended after verification: Claude verifier regressed (the applied text dropped 'by then'); restored | Assumption B3 and its consequence introduce the scheduling variables, independence condition, filtration, and two conditional laws without a pause.

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

### P-034
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-008
- status: rejected
- reason: duplicate of P-008 (Claude version: one noun replaces the pronoun; no reordering) | reviewer note: The explanation after the tracking theorem delays the two observable conditions until after two intervening qualifications.

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

### P-035
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-005
- status: rejected
- reason: duplicate of P-005 (Claude version: minimal fix of the misattached clause) | reviewer note: Two pieces of notation and the event restricting all asymptotic statements are introduced at the end of an already dense theory overview.

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

### P-036
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-037
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-038
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: reverted
- reason: reverted after verification: codex verifier regressed (paraphrases the equation and inverts the explanatory order); Claude verifier neutral | The explanation of the tilt identity states both the operative quantity and scale invariance after a colon, which makes the second fact easy to miss. | reverted after verification

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

### P-039
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-040
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-014
- status: rejected
- reason: duplicate of P-014 (Claude version adds the missing reading cue for W_j and Z_j) | reviewer note: The capped-rejection and underflow-redraw mechanisms are distinct departures from the nominal proposal but are joined in one sentence.

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

### P-041
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-042
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-003
- status: rejected
- reason: duplicate of P-003 (Claude version names the antecedent explicitly) | reviewer note: The Cellular Potts fidelity diagnostic combines why factor (c) is active, the bootstrap share, the floor, two component estimates, their total, and the range of $r$ in one sentence.

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

### P-043
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- duplicate-of: P-018
- status: rejected
- reason: duplicate of P-018 (Claude version: one verb change) | reviewer note: The fixed-floor requirement, loss of the weight bound, role of that bound, and need for a new rate argument are nested into one causal chain.

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

### P-044
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-045
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-046
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-047
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-048
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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

### P-049
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- source: codex
- status: applied
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
