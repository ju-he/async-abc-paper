## Scores

- AI-likeness: 5/10. The manuscript is technically specific, candid about failures, and substantially less formulaic than typical generated prose. The score is driven by Appendix A’s clause-stacked result reporting, symmetrical factor-by-factor treatment, and occasional slogan-like interpretations.
- Readability: 7/10. The main argument, terminology, and experimental comparisons are clear. Readability falls in the appendices when several numerical findings, qualifications, and interpretations occupy one sentence.

The three passages that most drive the AI-likeness score are:

> On the five Cellular Potts production replicates (about $12{,}900$ asynchronous evaluations each across $48$ workers, two correlated dimensions, so that (c) is live; Appendix~\ref{app:cpm}) both factors are active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.

> The \emph{per-step drift} $\|q_{\tau+1}-q_\tau\|_\infty$ decays as $\tau^{-0.7}$ to $\tau^{-0.95}$, and the archive's membership changes only $825$--$874$ times in about $12{,}900$ draws, tracking $k\log n$ at a ratio between $0.54$ and $0.92$ across the whole run.

> Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$ (which does not by itself give $r\equiv1$, since the other three factors remain), and the proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.

## What to leave alone

- Keep the C1--C4 subsection headings. They state the tested claims directly and provide useful navigation.
- Keep the abstract’s final two sentences and contribution 3 unchanged. “Turns on” is consistent with the fixed C3 heading and names a measured boundary.
- Keep “binds” and “binding,” “sits at the threshold,” and “sits on that plateau.” These verbs identify an active constraint or a measured regime rather than supplying decorative imagery.
- Keep the Gaussian-mean loss, four-dimensional ESS instability, parameter-coupled-runtime null result, and scaling reversal stated plainly. These unfavorable results give the manuscript a credible empirical voice.
- Keep the distinction among bandwidth, tolerance, snapshot count $S$, and reconstructed-proposal count $m$. The repetitions that enforce these distinctions prevent substantive ambiguity.
- Keep the proof logic and its qualifiers in Appendix B. Its long sentences usually encode dependencies or assumptions that would become less precise if compressed.
- Keep the four italicized Discussion rules and “Barrier removal delivers simulations predictably.” Their imperative form is deliberate, and the latter cleanly separates the systems result from the posterior limitation.
- Keep the conclusion’s return to the opening barrier framing. It is appropriate structural closure rather than a redundant punchline.
- Keep the captions’ detailed encodings. They are dense, but their marker, line, panel, and column definitions are necessary for independent interpretation.

## Proposals

### C-001
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The sentence combines experimental configuration, factor activation, four numerical results, and their combined range.

**Before**
```text
On the five Cellular Potts production replicates (about $12{,}900$ asynchronous evaluations each across $48$ workers, two correlated dimensions, so that (c) is live; Appendix~\ref{app:cpm}) both factors are active: the bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$, so (a) contributes $\widehat\zeta=0.011$ and (b) $0.024$--$0.028$, together $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**After**
```text
The five Cellular Potts production replicates contain about $12{,}900$ asynchronous evaluations each across $48$ workers (Appendix~\ref{app:cpm}). Their two correlated dimensions make (c) live, and both measured factors are active. The bootstrap draws make up $1.1\%$ of each history against a floor of $\delta=2.4\%$. Contribution (a) is $\widehat\zeta=0.011$, and contribution (b) is $0.024$--$0.028$. Together they give $0.035$--$0.039$ with $r\in[0.24,1.41]$.
```

**Rationale**
This gives the configuration and each result its own sentence. It preserves every number, qualifier, and reference while removing the compressed result chain.

### C-002
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: Four distinct diagnostics are currently subordinated to one total-variation result.

**Before**
```text
The posterior estimate lies $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share, with posterior means within $0.012$ standard deviations, marginal widths within $1\%$ (narrower in nine of ten, the tail-suppressing direction that (a) predicts), and the effective sample size unchanged.
```

**After**
```text
The posterior estimate lies $0.2$--$0.5\%$ in total variation from the reference at $m=400$ and the observed prior share. The posterior means are within $0.012$ standard deviations. The marginal widths are within $1\%$ and are narrower in nine of ten, the tail-suppressing direction that (a) predicts. The effective sample size is unchanged.
```

**Rationale**
Each diagnostic becomes independently scannable. The direction predicted by (a) remains attached to the width result it qualifies.

### C-003
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The sentence conflates proposal drift, archive turnover, and agreement with a theoretical rate.

**Before**
```text
The \emph{per-step drift} $\|q_{\tau+1}-q_\tau\|_\infty$ decays as $\tau^{-0.7}$ to $\tau^{-0.95}$, and the archive's membership changes only $825$--$874$ times in about $12{,}900$ draws, tracking $k\log n$ at a ratio between $0.54$ and $0.92$ across the whole run.
```

**After**
```text
The \emph{per-step drift} $\|q_{\tau+1}-q_\tau\|_\infty$ decays as $\tau^{-0.7}$ to $\tau^{-0.95}$. The archive's membership changes $825$--$874$ times in about $12{,}900$ draws. Across the whole run, this turnover tracks $k\log n$ at a ratio between $0.54$ and $0.92$.
```

**Rationale**
The rewrite reports one measurement per sentence and then states the comparison. It removes only packing, not the interpretation.

### C-004
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: Results from two fitting ranges and the bias of the terminal-proposal proxy are packed together.

**Before**
```text
Fitting $\|q_\tau-q_\infty\|_\infty\sim\tau^{-b}$ gives $b=0.50$--$0.71$ per replicate over the full range ($0.59$ pooled) and $b=0.25$--$0.72$ ($0.48$ pooled) when the fit is restricted to the first half, away from the end where using the terminal proposal as a stand-in for $q_\infty$ biases $b$ upward.
```

**After**
```text
Fitting $\|q_\tau-q_\infty\|_\infty\sim\tau^{-b}$ over the full range gives $b=0.50$--$0.71$ per replicate ($0.59$ pooled). Restricting the fit to the first half gives $b=0.25$--$0.72$ ($0.48$ pooled). This restriction avoids the end, where using the terminal proposal as a stand-in for $q_\infty$ biases $b$ upward.
```

**Rationale**
The two empirical ranges and the reason for the restricted fit become distinct. The upward-bias caveat remains unchanged in force.

### C-005
- file: latex/tmlr/tmlr-article.tex
- category: heading
- severity: M
- status: proposed
- reason: The heading narrates a question the text is about to answer instead of naming the analysis.

**Before**
```text
\paragraph{Do the reported runs meet the stabilization conditions?} The proposal sequence is reconstructible from a stored history, so we assess Assumption~\ref{ass:stab}(ii) empirically.
```

**After**
```text
\paragraph{Empirical stabilization diagnostic.} We reconstruct the proposal sequence from each stored history and assess Assumption~\ref{ass:stab}(ii) empirically.
```

**Rationale**
The new heading names the content directly. The sentence also replaces an abstract capability with the concrete analysis performed.

### C-006
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The rate requirement, its consequence, and the empirical verdict currently form a colon-driven punchline.

**Before**
```text
Assumption~\ref{ass:stab}(ii) needs $b>\tfrac12$, at which the aggregate $n^{-1/2}\sum_{i\le n}\|q_i-q_\infty\|_\infty$ vanishes like $n^{1/2-b}$: the production runs sit at that threshold.
```

**After**
```text
Assumption~\ref{ass:stab}(ii) needs $b>\tfrac12$. Under this condition, the aggregate $n^{-1/2}\sum_{i\le n}\|q_i-q_\infty\|_\infty$ vanishes like $n^{1/2-b}$. The production runs sit at that threshold.
```

**Rationale**
The rewrite separates the mathematical condition from the empirical finding. It preserves the deliberately plain statement that the runs sit at the threshold.

### C-007
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: A central qualifier is buried inside a long sentence joining an asymptotic result to its proof condition.

**Before**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$ (which does not by itself give $r\equiv1$, since the other three factors remain), and the proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

**After**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$. This does not by itself give $r\equiv1$, since the other three factors remain. The proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

**Rationale**
The caveat and proof condition become prominent without changing their logic. This is a conservative split of mathematical prose.

### C-008
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: H
- status: proposed
- reason: The sentence chains an implementation choice, two consequences, and a required theoretical revision.

**Before**
```text
The implementation ties them, $\delta=0.5/(m{+}1)$, so growing $m$ there would send $\delta\to0$, and with it the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded} that makes the Lindeberg step trivial and fixes the constants in the law of large numbers; the rate conditions would have to be redone.
```

**After**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$. Growing $m$ would therefore send $\delta\to0$ and remove the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}. That bound makes the Lindeberg step trivial and fixes the constants in the law of large numbers. The rate conditions would have to be redone.
```

**Rationale**
The dependency chain is easier to follow as four short statements. Every logical consequence and the lemma reference remain intact.

### C-009
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The one-dimensional check reports three distinct findings in a single sentence.

**Before**
```text
On a one-dimensional Gaussian-mean history where the exact $n$-point mixture is computable, (a) and (b) together give $\widehat\zeta=0.042$ and a total-variation distance of $0.26\%$ from the reference that removes (a) and (b); there too the estimate is $0.5\%$ narrower.
```

**After**
```text
On a one-dimensional Gaussian-mean history, the exact $n$-point mixture is computable. Contributions (a) and (b) together give $\widehat\zeta=0.042$. The total-variation distance from the reference that removes (a) and (b) is $0.26\%$, and the estimate is $0.5\%$ narrower.
```

**Rationale**
The availability of the exact reference is setup, not another result. Separating it makes the two reported discrepancies easier to compare.

### C-010
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The observed event counts and the limitation of those counts should not share a semicolon-heavy sentence.

**Before**
```text
Of contribution (d), the two events a stored history records, a rejection fallback and an exhausted redraw, occurred neither in the $12{,}000$ draws of that history nor in the $63{,}699$ archive-phase draws of the five Cellular Potts replicates; a redraw that succeeded on a later attempt leaves no trace, so this is a necessary condition for the event, not a bound on its probability.
```

**After**
```text
For contribution (d), a stored history records two events: a rejection fallback and an exhausted redraw. Neither occurred in the $12{,}000$ draws of that history or in the $63{,}699$ archive-phase draws of the five Cellular Potts replicates. A redraw that succeeded on a later attempt leaves no trace. The recorded absence is therefore a necessary condition for the event, not a bound on its probability.
```

**Rationale**
The rewrite separates the observation from its measurement limitation. It keeps the important qualification that the event count is not a probability bound.

### C-011
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Two benchmark comparisons and their general explanation are compressed into a parenthetical closer.

**Before**
```text
The bound \eqref{eq:tilt-bound} is loose against these direct measurements by an order of magnitude on both ($3.6$--$4.0\%$ against $0.2$--$0.5\%$ on Cellular Potts, $4.4\%$ against $0.26\%$ on the one-dimensional history), as sup-norm-free bounds valid for every $f$ generally are.
```

**After**
```text
The bound \eqref{eq:tilt-bound} is loose against both direct measurements by an order of magnitude. On Cellular Potts it gives $3.6$--$4.0\%$ against $0.2$--$0.5\%$; on the one-dimensional history it gives $4.4\%$ against $0.26\%$. Sup-norm-free bounds valid for every $f$ are generally this loose.
```

**Rationale**
Each benchmark receives its own comparison, followed by the interpretation. The rewrite preserves the scope of the explanation and all reported values.

### C-012
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The experiment’s two settings, effect size, and pass criterion are crowded into one sentence.

**Before**
```text
A tightening bandwidth does not move it: running a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in, moves $b$ by less than $0.07$, and both runs satisfy $b>\tfrac12$ at $b=0.61$--$0.76$.
```

**After**
```text
A tightening bandwidth does not move it. We ran a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in. The two settings move $b$ by less than $0.07$. Both runs satisfy $b>\tfrac12$ at $b=0.61$--$0.76$.
```

**Rationale**
The setup, comparison, and result become separate statements. The short opening contrast remains scientifically meaningful because the next sentences test it directly.

### C-013
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The cost comparison and the empirical fidelity result are joined as a manufactured payoff.

**Before**
```text
The measurements above locate the practical knee: on the Gaussian-mean history the posterior estimate is within $3\times10^{-4}$ in total variation of the exact $n$-point mixture by $m=50$, so a $2.5\times$ more expensive post-hoc pass removes factor (b) almost entirely.
```

**After**
```text
On the Gaussian-mean history, the posterior estimate is within $3\times10^{-4}$ in total variation of the exact $n$-point mixture by $m=50$. At that point, a $2.5\times$ more expensive post-hoc pass removes factor (b) almost entirely.
```

**Rationale**
The result and its computational implication remain, but the announced “practical knee” is removed. Cost language is retained because it quantifies an actual computational tradeoff.

### C-014
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: The generation-time calculation, throughput ceiling, measured ratio, and additional settings are packed into one sentence.

**Before**
```text
The slow worker's recorded evaluation time ($2.000$\,s at $20\times$) plus the $3.6$\,ms per-evaluation overhead measured on the fast workers is the generation time, so the twin cannot exceed $16/2.004=7.98$ simulations per second, and the measured asynchronous rate over that gives $401\times$ against a measured $402\times$; at $5\times$ and $10\times$ the agreement is the same.
```

**After**
```text
The generation time is the slow worker's recorded evaluation time ($2.000$\,s at $20\times$) plus the $3.6$\,ms per-evaluation overhead measured on the fast workers. The twin therefore cannot exceed $16/2.004=7.98$ simulations per second. Dividing the measured asynchronous rate by that ceiling gives $401\times$, against a measured $402\times$. The agreement is the same at $5\times$ and $10\times$.
```

**Rationale**
The calculation is now presented in its actual sequence. No quantity or interpretation changes.

### C-015
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: A single causal sentence carries four operational consequences.

**Before**
```text
There is no population to keep consistent and no shared mutable state, so workers never wait for one another: each updates from whatever history it has received, a crashed run is recovered by replaying its log, and the posterior estimate is the same whether it is formed during the run or afterwards.
```

**After**
```text
There is no population to keep consistent and no shared mutable state. Workers therefore never wait for one another. Each updates from whatever history it has received. A crashed run is recovered by replaying its log, and the posterior estimate is the same whether it is formed during the run or afterwards.
```

**Rationale**
The operational properties become individually visible without turning them into a symmetrical list. The causal relation between history reconstruction and barrier-free execution remains explicit.

### C-016
- file: latex/tmlr/tmlr-article.tex
- category: vocabulary
- severity: M
- status: proposed
- reason: “Fairly budgeted” is vague and mildly promotional where the experimental control can be named directly.

**Before**
```text
On the Gaussian mean the asynchronous estimator is closer to the analytic posterior than pyABC's final population at every checkpoint, and a fairly budgeted rejection sampler is as close; in one dimension with that many draws, rejection is enough.
```

**After**
```text
On the Gaussian mean, the asynchronous estimator is closer to the analytic posterior than pyABC's final population at every checkpoint. A rejection sampler given the same simulation budget is as close. In one dimension with that many draws, rejection is enough.
```

**Rationale**
The replacement specifies what “fairly” means and separates the negative result from the comparison. It preserves the direct conclusion that rejection suffices here.

### C-017
- file: latex/tmlr/tmlr-article.tex
- category: structure
- severity: M
- status: proposed
- reason: The sentence leads with an interpretation and then compresses three pieces of evidence into a colon-driven list.

**Before**
```text
The evidence points to a reporting-support effect rather than a weighting failure: the top-$k$ archive alone would under-cover ($0.85/0.89$ at the $0.90/0.95$ levels), the full history over-covers, and coverage rises monotonically through nominal as the support grows from one to the other.
```

**After**
```text
The top-$k$ archive alone would under-cover ($0.85/0.89$ at the $0.90/0.95$ levels). The full history over-covers. Coverage rises monotonically through nominal as the support grows from one to the other. This pattern points to a reporting-support effect rather than a weighting failure.
```

**Rationale**
The empirical findings now precede the interpretation they support. This also gives the paragraph the interpretive ending required by the subsection claim.

### C-018
- file: latex/tmlr/tmlr-article.tex
- category: readability
- severity: M
- status: proposed
- reason: Three archive-size results are presented with identical grammar inside one long sentence.

**Before**
```text
Every larger archive is dominated by it on \emph{both} axes: $k=200$ is $5\%$ slower and nearly three times worse calibrated ($0.180$), $k=400$ is $14\%$ slower and $0.195$, and $k=800$ costs $31\%$ of throughput and is still slightly worse calibrated than $k=100$ ($0.074$).
```

**After**
```text
Every larger archive is dominated by it on \emph{both} axes. At $k=200$, the sampler is $5\%$ slower and nearly three times worse calibrated ($0.180$). At $k=400$, it is $14\%$ slower with a deviation of $0.195$. At $k=800$, it costs $31\%$ of throughput and remains slightly worse calibrated than $k=100$ ($0.074$).
```

**Rationale**
One archive size is reported per sentence, which makes the non-monotone calibration values easier to inspect. The unfavorable results and both-axis conclusion remain explicit.

I would run one more round after these edits, limited to checking paragraph rhythm and newly adjacent sentence openings; the remaining substantive prose is already precise enough that broader rewriting would risk flattening it.