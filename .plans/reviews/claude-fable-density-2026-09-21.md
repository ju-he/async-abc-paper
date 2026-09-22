Target: `latex/sn-article-template/sn-article.tex`, branch `campaign-tooling`, commit 9837a81.
Editorial pass (redundancy / density / defensiveness), Claude Fable 5.1, 2026-09-21.

# A. Verdict

The main text (abstract through conclusion) is 8,670 body words plus 853 words of captions; it could reasonably drop to about 6,500-7,000 body words without losing a claim, a qualification that honesty needs, or a number the reader must have. The biggest problem is **redundancy**, and its specific form is instructive: several review rounds have been absorbed by adding the same caveat or attribution at every place a referee might look, so the paper now carries the earlier-configuration disclosure in five main-text places, the straggler-factor-versus-throughput-ratio distinction in six, the 1.2x/2.0x attribution of the Cellular Potts gain in four, the "estimator is a pure function of the history, but the running sampler carries a buffer" pair in five, the bandwidth-schedule mechanism ("once per k calls, halves at most") in three, the four-node placement caveat in five, and every practitioner rule twice (once at the end of the results subsection that earns it, once in the Discussion). The Limitations section is almost entirely a third statement of qualifications the results already make. Density is the second problem and is concentrated in four paragraphs (lines 238, 277, 330, 311) that recite table cells in prose, plus a theory section that displays and discusses an assumption clause whose theorem lives in the appendix. Defensiveness is real but is mostly the *same* passages as the redundancy: the repeated items are caveats, and a handful of passages explain at length why a number is *not* quoted. Cutting the repeats fixes most of the tone by itself; the remaining defensive tics are a dozen "rather than X" contrast tails and three references to earlier drafts.

# B. Findings

Word savings are estimates for the main text unless marked (appendix). Line numbers verified against the file with `grep -n`.

## Redundancy

**R1. The Limitations section restates the results' own qualifications.**
- Location: lines 410-417, §8.
- Quote: "\emph{Theory.} Consistency holds up to a fidelity ratio $r$ between the density..."
- Diagnosis: All five bullets repeat, often sentence for sentence, qualifications already made where they belong: the Theory bullet (412) is line 188 again (partial accounting of r, CLT rate missed for a reason in the inference problem, filtration assumed, floor unset); the systems bullet (414) repeats the earlier-configuration disclosure (188, 235, 277, 285), the twin duration inflation (277), the calibration-instrument paragraph (195) and the Lotka-Volterra scoring note (193); the baselines bullet (415) repeats 216, 223 and 318; the reported-bandwidth bullet (416) is the last sentence of 393. Two items are not limitations at all: "the $2.1\times$ predicted for $384$ workers ... is not measured" (the text already says "predicted"), and "The snapshot buffer is fixed at $S=20$" while Appendix F sweeps $S\in\{0,5,20,50\}$ (Table 14).
- Recommendation: Rewrite the section as one paragraph of about 170 words that lists each limitation once with a pointer and no restated numbers. Proposed text:
  > Consistency holds up to the fidelity ratio $r$; two of its four factors are measured at about one per cent in total variation, two are bounded, and the adapted filtration under asynchronous execution is assumed (§4, Table 9). The central limit theorem covers damped-adaptation variants, not the configuration we run, whose target is too weakly identified for its rate condition. Keeping every evaluated particle over-represents parameter regions that simulate quickly; a runtime-coupled study found no measurable shift of the posterior mean at the couplings tested, and we do not correct for it (Appendix F). The synchronous baseline runs a population of 100 on 47 simulating ranks, and is scored uniformly weighted on the two campaigns whose pyABC histories were not kept; a population sized to the machine was run only in the scaling study. The $50^3$ twin and scaling campaigns and the theory diagnostics used an earlier configuration of the Cellular Potts benchmark (Appendix E). The cost boundary of §6.3 belongs to the placement and budget it was measured under, and the reported posterior is only as good as the bandwidth the schedule reached (§6.4).
- Saving: ~330 words.

**R2. Headline numbers appear in the abstract, the contributions list, the results, and the conclusion.**
- Location: lines 57-59 (§1 contributions), line 420 (§9); source of truth is line 42 (abstract) and §6.
- Quote (57): "A barrierized twin of that sampler, and the measurement that its slowdown is predicted..."
- Diagnosis: Contributions 2 and 3 reproduce the abstract's "within $1\%$ ... $20\%$ ... $10\%$ ... $1.2\times$ to $400\times$" and "$2.1$--$2.4\times$ ... $1.6\times$ ... $4.1\times$" verbatim; the conclusion repeats the first triple in words ("within one per cent ... twenty ... ten"). A contributions list should say what was done and where, not re-run the abstract.
- Recommendation: Rewrite the four contributions without numbers, one line each:
  > 1. A generation-free ABC sampler combining smooth-kernel ABC with streaming AMIS reweighting, whose reported estimator is a replayable function of the evaluated history (§3). 2. A barrierized twin of that sampler, and the measurement that the barrier's cost is predicted from the asynchronous run's timing alone across two and a half orders of magnitude (§6.1). 3. The measurement that the gain converts into a tighter tolerance against a kernel-matched pyABC baseline on a tissue simulator, and that the advantage turns on at a few milliseconds per simulation (§6.2, §6.3). 4. Consistency of the reported estimator up to a fidelity ratio (§4; proofs and a central limit theorem in the appendix), its calibration, and the two settings that bound what it delivers (§6.4).
  In the conclusion (420), drop the "within one per cent ... twenty ... ten" clause and "across two and a half orders of magnitude"; keep "predictable from the asynchronous run's timing alone".
- Saving: ~120 words.

**R3. The straggler factor versus the general throughput ratio is explained six times.**
- Location: lines 50 (§1), 221 (§5.3), 238 (§6.1), 274 (§6.1), 400 (§7), 420 (§9).
- Quote (50): "In general the cost is the ratio of the barrier-free throughput to..."
- Diagnosis: This distinction was added in response to a review (digest item 1) and now appears wherever the straggler factor is mentioned: the intro sentence, the full definition in Metrics, a 45-word parenthetical in the straggler paragraph ("this is the throughput ratio of §5.3, not the straggler factor of the sixteen workers' runtimes, which would be $15.6$..."), a clause in the heterogeneity paragraph ("so both ratios exceed the straggler factor proper, $19.9$"), the Discussion's first rule, and the conclusion's first sentence. One careful statement suffices.
- Recommendation: Keep line 221 as the single definition. At 50, cut "In general the cost is the ratio ... runs at $W/\mu$." At 238, cut the parenthetical from "(this is the throughput ratio" to "never wait)". At 274, cut "so both ratios exceed the straggler factor proper, $19.9$". At 400, cut "which for identically distributed runtimes is the straggler factor of the workload and". At 420, cut "--- the straggler factor of the workload when runtimes are identically distributed ---".
- Saving: ~140 words.

**R4. "The estimator is a pure function of the history; the running sampler carries a buffer" is stated five times in §3.**
- Location: lines 76, 78, 106, 108, 111 (§3.1, §3.4, §3.5); also Table 1 caption (690) and Appendix D (727).
- Quote (78): "That is a property of the estimator, not of the running sampler..."
- Diagnosis: Line 76 states the property; 78 states the exception and then restates the property ("The archive, the bandwidth and the reported weights are therefore reconstructed from history rather than carried"); 106 restates both ("the running sampler carries the buffer between calls (§3.1), and the reported estimator rebuilds its own snapshots"); 108 restates the property ("the archive, the adaptation weights and the reported posterior are pure functions of them"); 111 restates both again ("depends on no state carried between calls (the running sampler's snapshot buffer and scheduler throttle are the exceptions of §3.1)"). This is the "stores nothing" correction from the last round applied everywhere at once.
- Recommendation: Keep 76 and the first sentence of 78 (through "(Appendix~\ref{app:protocol})"). Cut the last sentence of 78 ("The archive, the bandwidth and the reported weights are therefore..."). At 106, cut "the running sampler carries the buffer between calls (§3.1), and". At 108, cut "Each stored record holds the fields of \eqref{eq:history}; the archive, the adaptation weights and the reported posterior are pure functions of them." At 111, rewrite as: "Every quantity it uses is reconstructed from the history $\mathcal{H}_n$ passed in; until the history holds at least $k$ members it emits uniform prior draws to seed the archive."
- Saving: ~110 words.

**R5. The 1.2x-barrier / 2.0x-attributable decomposition of the Cellular Potts gain is made four times.**
- Location: lines 279 (§6.1), 311 (§6.2), 285 (Fig. 5 caption), 954 (Fig. 12 caption, appendix).
- Quote (311): "by the factorisation of \S\ref{sec:results-barrier} about $1.2\times$ of this is the barrier..."
- Diagnosis: 279 makes the decomposition and says "We therefore quote $2.0\times$"; 311 repeats it in a 50-word parenthetical ending "which is why $2.0\times$ is the figure we attribute to the barrier"; both captions repeat it again.
- Recommendation: Keep 279. At 311, replace the parenthetical with "($99.8\%$ worker utilisation against $42.8\%$; §6.1 attributes $1.2\times$ of this to the barrier)". At 285, cut "the gap to pyABC is mostly its per-generation overhead, the barrier's share being $1.2\times$ at $50^3$ and a predicted $2.1\times$ at eight nodes at the $80^3$ spread (§6.1)". At 954, cut from "At this size, however" to "made at $80^3$".
- Saving: ~90 words (plus ~40 appendix).

**R6. The bandwidth-schedule mechanism is explained three times in the main text.**
- Location: lines 86 (§3.2), 391 (§6.4), 393 (§6.4); echoed at 404 and 416.
- Quote (391): "The schedule walks down from $\epsilon_0$ at a bounded rate --- each rank searches once per $k$..."
- Diagnosis: §3.2 already says "each rank's scheduler searches once per $k$ of its own calls, and a search may at most halve the bandwidth"; 391 repeats it word for word; 393 repeats it a third time ("because it tightens by at most a halving once per $k$ calls on each rank").
- Recommendation: Keep 86. At 391, rewrite the sentence as "The schedule tightens at the bounded rate of §3.2, so on an expensive simulator that transient is the whole run: ...". At 393, cut "because it tightens by at most a halving once per $k$ calls on each rank and" and keep "under lockstep [the twin] takes fewer search steps (Appendix C)".
- Saving: ~60 words.

**R7. Each practitioner rule is stated at the end of the results subsection and again in the Discussion.**
- Location: 320 vs 400 (§6.3 / §7), 391 vs 404 (§6.4 / §7), 395 vs 406 and 420 (§6.4 / §7 / §9).
- Quote (320): "The rule this gives a practitioner is simple. Above a few milliseconds per simulation..."
- Diagnosis: 320's last two sentences are the Discussion's first rule verbatim ("a generation-staged sampler with a population sized to the machine is the better tool"). 391's "The rule that transfers is to set $\epsilon_0$ from ... about a fifth of it" is rule 3. 395's "The two settings together bound what the estimator delivers --- the starting bandwidth decides ..., the archive size ..." is rule 4 and is repeated once more at 420 ("What bounds it is the reporting rule: ...").
- Recommendation: Let the Discussion own the rules. At 320, cut from "The rule this gives a practitioner is simple." to the end of the paragraph. At 391, keep the finding ("$\epsilon_0$ set to about a fifth of the prior-predictive median discrepancy, $0.46$ here, recovers the posterior") but cut "The rule that transfers is to set $\epsilon_0$ from a quantity measurable before the run and without the truth". At 395, cut "The two settings together bound ... after the fact." At 420, shorten to "What bounds it is the reporting rule (§7)."
- Saving: ~130 words.

**R8. The four-node placement caveat appears five times in the main text and twice in one appendix paragraph.**
- Location: 294 (Table 5 caption), 318 (§6.3), 325 (Fig. 4 caption), 415 (§8), 764 (Appendix D, twice in the same paragraph).
- Quote (318): "The production campaign of the cheap benchmarks ran its $48$ ranks twelve per node on four nodes..."
- Diagnosis: The caveat is load-bearing once (it is why the 4 ms boundary is not a constant). Table 5's caption, Fig. 4's caption, the Limitations bullet and Appendix D each restate it; Appendix D line 764 says it twice, once with the `srun` flags.
- Recommendation: Keep one sentence at 318 (see D6 for the compression). At 294, cut "The three cheap benchmarks ran $48$ ranks as twelve per node on four nodes, Cellular Potts on one node (Appendix~\ref{app:protocol})". At 325, cut "$48$ ranks placed twelve per node on four nodes". At 415, covered by R1. At 764, delete the first mention ("The production campaign of the three cheap benchmarks ... fully packed.") and keep the second, which has the flags.
- Saving: ~70 words (plus ~50 appendix).

**R9. The earlier-configuration disclosure is repeated at every Cellular Potts number.**
- Location: 188 (§4), 235 (Fig. 1 caption), 277 (§6.1), 285 (Fig. 5 caption), 414 (§8); and 508, 510, 677, 678, 734, 827, 954 in the appendices.
- Quote (277): "on the benchmark's earlier single-simulation configuration ($5.1$\,s per evaluation, division rate and motility; Appendix~\ref{app:cpm})"
- Diagnosis: The disclosure must exist (see D), but the paper never introduces it in §5.1, where the benchmark is defined, so each later mention re-explains it ("single-simulation", "$5.1$ s", "division rate and motility").
- Recommendation: Add one sentence at the end of line 193: "The $50^3$ twin and scaling campaigns, and the theory diagnostics of Appendix A, ran an earlier two-parameter configuration of this benchmark (division rate and motility, $5.1$\,s per single-simulation evaluation; Appendix~\ref{app:cpm})." Then reduce every later mention to the tag "(earlier configuration)": at 188 keep "(earlier configuration, Appendix E)"; at 277 cut "($5.1$\,s per evaluation, division rate and motility; Appendix~\ref{app:cpm}), not the production setup of Table~\ref{tab:benchmarks}"; at 285 keep "earlier configuration"; at 414 covered by R1.
- Saving: ~40 words net in main text.

**R10. The matched-budget tolerance metric is defined three times.**
- Location: 219 (§5.3), 291 (§6.2), 294 (Table 5 caption).
- Quote (291): "the tolerance $\epsilon_{(100)}(n)$ at which exactly $100$ of a method's own draws would be accepted after $n$ simulations (\S\ref{sec:experiments-metrics})"
- Diagnosis: §5.3 defines $\epsilon_{(k)}(n)$ and both readings (equal $n$, equal wall clock); §6.2's first sentence re-defines it in full and then cites §5.3; Table 5's caption defines "Per-simulation" and "Equal wall clock" a third time.
- Recommendation: At 291, rewrite the opening as "Table~\ref{tab:matched-eps} reads that off every benchmark as $\epsilon_{(100)}(n)$ (§5.3; curves in Fig.~\ref{fig:eps-curves})". At 294, cut "Per-simulation: $\epsilon$ of the synchronous arm over the asynchronous arm at the synchronous arm's own simulation count. Equal wall clock: the same ratio at the end of both runs. Both rise with the cost of a simulation." (the column headers plus §5.3 carry it).
- Saving: ~70 words.

**R11. Fig. 1's text and caption list the same configurations and the same three open markers.**
- Location: 230 (§6.1) and 235 (caption).
- Quote (230): "For every configuration on which we ran the barrierized twin --- a persistently slow worker at five slowdowns..."
- Diagnosis: The text enumerates the configurations, says no synchronous data enter the prediction, and names the three open markers; the caption enumerates the configurations again with worker counts and names the open markers again with a pointer back to the text. Also "predicted from the asynchronous run's timing alone" is at 42, 57, 230, 235, 247, 420.
- Recommendation: Make the caption the short one: "Barrier cost predicted from the asynchronous arm's timing (§5.3) against the cost measured with the barrierized twin, for the straggler, heterogeneity and Cellular Potts campaigns (Table 2; the $50^3$ points use the utilisation ratio and the earlier configuration, the $80^3$ point is measured against pyABC). Medians with replicate ranges; dashed line is equality; open markers are outside the model's domain." Cut "No synchronous or twin data enter the prediction." and "The rest of this subsection walks through the three workloads." from 230.
- Saving: ~50 words.

**R12. The scoring rule for reported posteriors is given in §5.3, in §6.4, in the Fig. 4 caption and in the Limitations.**
- Location: 223, 356, 361, 415.
- Quote (361): "the asynchronous method's retroactive AMIS estimator replayed over the prefix completed by each checkpoint, the synchronous baseline's last completed generation (uniformly weighted; no history was kept)..."
- Diagnosis: §5.3 states how each method is scored, including the uniform weighting where histories were not kept; the caption restates it; 356 restates the uniform-weighting reason in a parenthetical; 415 states it again.
- Recommendation: At 361, replace the enumeration with "Each method is scored on the posterior it reports (§5.3)". At 356, cut "(uniformly weighted; its history was not kept)". 415 covered by R1.
- Saving: ~50 words.

**R13. Background says "we take AMIS to its streaming limit" twice in consecutive paragraphs, and the AMIS-consistency remark is made three times.**
- Location: 70 and 72 (§2); also 129 (§4) and 455 (Appendix A).
- Quote (72): "Our proposal-adaptation machinery descends from the adaptive-importance-sampling line..."
- Diagnosis: Line 70 ends "Our method takes its streaming limit: stage size one, single-arrival-driven adaptation." Line 72 begins with the same statement and cites the same two papers again. Line 70's "it is consistent under regularity conditions \cite{marin2019consistency}, the convergence argument in \cite{cornuet2012amis} being heuristic" is repeated at 129 and 455.
- Recommendation: Cut the first sentence of 72 through "single-arrival limit." and start the paragraph at "Parallel and off-barrier SMC has been studied". At 70, cut ", the convergence argument in \cite{cornuet2012amis} being heuristic" (129 and Appendix A carry the point where it matters).
- Saving: ~50 words.

**R14. The "calibration instrument, not a production run" paragraph is repeated in the Limitations.**
- Location: 195 (§5.1) and 414 (§8).
- Quote (195): "The Cellular Potts benchmark is a \emph{calibration instrument}, and we state its role so that it is not read as a production run."
- Diagnosis: 414 restates "cost seconds to minutes per evaluation against hours in production, infer two parameters on one synthetic dataset with a known truth, and establish the method and the predictor rather than a production posterior". The framing at 195 ("we state its role so that it is not read as") is referee-facing.
- Recommendation: Keep 195, rewritten as two sentences: "Production simulations of this model take hours on many cores; ours take seconds to minutes and infer two parameters from synthetic data with a known truth. The benchmark therefore tests whether the method recovers a posterior it should recover and measures the barrier's cost on a real workload; a production project takes from it the predictor of §5.3, which needs only its own run's timing." Cut the repeat in 414 (R1).
- Saving: ~60 words.

**R15. The twin's fixed-simulation-count versus wall-clock mechanics are explained three times.**
- Location: 216 (§5.2), 238 (§6.1), 247 (Table 3 caption).
- Quote (247): "the asynchronous arm is wall-limited, the twin runs to a fixed simulation count; \emph{predicted} is $W/\mathbb{E}[\max D]$ from the asynchronous timing..."
- Recommendation: Keep 216. At 238, cut "which must run to a fixed simulation count,". At 247, cut from "Throughput in simulations per second" to "dominates." and replace with "Throughput in simulations per second; predicted as in §5.3." Also cut "; \texttt{make\_twin\_rereport.py}" (script names do not belong in captions).
- Saving: ~50 words.

**R16. "13,000 prior draws on all 48 ranks" is repeated five times.**
- Location: 216, 311, 364, 369 (Fig. 6 caption), 373 (Table 8 caption).
- Recommendation: Keep at 216 (the definition of the comparator) and in Table 8's caption. At 311, shorten to "A fairly resourced rejection sampler (Table~\ref{tab:cpm-production}) reaches $2.9\times10^{-3}$, $39\times$ looser." At 364, "the best $100$ rejection draws"; at 369, "the $100$ best rejection draws".
- Saving: ~30 words.

**R17. §3.4 pre-announces the "three weights" paragraph that immediately follows.**
- Location: 106 (last sentence) and 108 (§3.4).
- Quote (106): "Equation~\eqref{eq:amis-weight} is the proposal-time importance weight; the \emph{posterior} estimator reported in..."
- Recommendation: Cut the last sentence of 106; the paragraph at 108 says it properly.
- Saving: ~30 words.

**R18. The asynchronous-filtration caveat is made five times in the main text.**
- Location: 129, 132 (twice), 155 (Assumption 4), 188, 412.
- Quote (132): "That such a conditional density exists with respect to a filtration the proofs can use is itself an assumption once execution is asynchronous..."
- Diagnosis: The assumption statement (155) already says "Under sequential execution this holds by construction; under asynchronous execution it is a genuine assumption (Appendix B)". Lines 129 and 132 say it before the reader reaches the assumption, twice in 132 ("which is why Assumption 4 states it rather than leaving it implicit" and "Assumption 4 is where that is confronted rather than hidden"); 188 and 412 repeat it after.
- Recommendation: At 129, cut "; the proofs index particles by proposal time and the pass by arrival time, and Appendix~\ref{app:theory} states what closing that gap assumes". At 132, cut both sentences from "That such a conditional density exists" to "confronted rather than hidden." At 188, keep the parenthetical (it explains what factor (d) carries). 412 covered by R1.
- Saving: ~90 words.

**R19. The snapshot-denominator recipe is described in §3.4, §4 and Appendix C.**
- Location: 108, 143, 719.
- Quote (143): "the draw-proportional deterministic mixture \cite{veach1995,owen2000safe} of $m\le S{+}1=21$ history-reconstructed proposals with a prior component whose mass is the observed bootstrap share $\nu_n$ floored at $\delta=0.5/(m{+}1)$, exactly as the implementation builds it (Appendix~\ref{app:implementation})"
- Recommendation: §4 (143) is the right home because the theorem uses the recipe. At 108, shorten to "where $\bar q_n$ is the $m$-snapshot denominator with a defensive prior component that §4 specifies". At 143, cut ", exactly as the implementation builds it".
- Saving: ~35 words.

**R20. Signposting that repeats the headings.**
- Location: 226 (§6 intro), 230 (§6.1).
- Quote (226): "The four subsections take the four claims in turn: the barrier's cost and its prediction (C1)..."
- Diagnosis: The four subsection titles, which follow immediately, say the same thing with the same labels. "The rest of this subsection walks through the three workloads" (230) announces three paragraph headings the reader sees on the same page.
- Recommendation: Cut both sentences.
- Saving: ~45 words.

**R21. Crash recovery by log replay is stated in §1 and again in §3.1 with the same words.**
- Location: 52 and 76.
- Quote (52): "and a crashed run is recovered by replaying its log rather than by checkpointing sampler state"
- Recommendation: Keep it in §3.1 (76); at 52, cut the clause and end the sentence at "Propulate optimization engine \cite{taubert2023propulate}".
- Saving: ~15 words.

**R22. The Cellular Potts runtime CV "from 0.05 to above 0.2" is given three times.**
- Location: 50 (§1), 193 (§5.1), 400 (§7); Table 2 also lists it.
- Recommendation: Keep 193 in words ("the runtime spread rises with the box because runtime tracks cell count"); at 50, end the sentence at "across a prior"; at 400, cut "as the Cellular Potts model's did from $0.05$ to above $0.2$".
- Saving: ~35 words.

**R23. (appendix) The Cellular Potts setup is specified in Appendix D and again in Appendix E.**
- Location: 762 (Appendix D, "Cellular Potts:" entry) and 767, 769 (Appendix E).
- Diagnosis: Priors $[0.001,0.2]$ and $[200,1200]$, truths $0.009$ and $500$, the two summaries and the four-replicate averaging are stated in 193, 762, 767 and 769.
- Recommendation: In 762, replace the Cellular Potts entry with "\emph{Cellular Potts:} see Appendix~\ref{app:cpm}." In 769, cut the recap of the production configuration ("division rate and target cell volume, four replicate simulations per evaluation, $13$\,s at $50^3$ and $170$\,s at $80^3$").
- Saving: ~110 words (appendix).

**R24. (appendix) The k-frontier conclusion is stated three times in Appendix F.**
- Location: 860 (last three sentences), 886-888, 924 (Table 14 caption).
- Quote (860): "The archive size is therefore a systems cost as well as a statistical knob --- but, put on the same axis..."
- Recommendation: Keep 886-888 and the table; at 860 cut from "The archive size is therefore a systems cost" to the end of the paragraph; at 924 cut "Only $k{=}50$ and $k{=}100$ are Pareto-optimal; every larger archive is both slower \emph{and} worse calibrated than $k{=}100$."
- Saving: ~110 words (appendix).

**R25. References to earlier drafts of this paper.**
- Location: 223 (§5.3), 590 (Appendix B remark), 948 and 958 (Appendix F).
- Quote (223): "We no longer report the distance from an unweighted archive to a point mass at the truth..."
- Diagnosis: A reader of the submitted version has no earlier version; "we no longer", "earlier drafts' condition ... was missing", "Earlier versions of this work scored", "predate the reported estimator" are referee-facing history. The substantive point (a point-mass distance rewards a collapsed posterior) is worth one sentence in Appendix F.
- Recommendation: At 223, cut the last sentence. At 590, cut "It is also the content that earlier drafts' condition ... does not deliver it." At 948, rewrite as "A Wasserstein distance from the unweighted archive to a point mass at the truth is floored at the posterior's own spread ($0.080$ on the Gaussian configuration) and ranks a collapsed archive above a correct one; we use it only as a concentration diagnostic (ablation and sensitivity below)." At 958, cut "This study and the sensitivity grid that follows predate the reported estimator and".
- Saving: ~40 words main text, ~60 appendix.

## Density

**D1. Assumption 5(ii) and the CLT commentary sit in the main text while the CLT is in the appendix.**
- Location: 161-170 (Assumption 5(ii) with the displayed rate condition \eqref{eq:rate}), and 172 from "The central limit theorem asks more" to "no matter how run-dependent $\tilde q_\infty$ is."
- Quote (161): "\emph{(ii, central limit theorem)} In addition $\epsilon_\infty$ and $r$ are deterministic..."
- Diagnosis: Theorem 2 and Corollary 3 are in Appendix A, but their assumption clause, a two-line display, and ~110 words explaining why consistency permits random $r$ and the CLT does not, are in §4. A reader of the main text needs Assumption 5(i) only.
- Recommendation: Move clause (ii), display \eqref{eq:rate} and the sentence after it to Appendix A, immediately before Theorem 2 (as "Assumption 5(ii)"). At 172, cut from "The central limit theorem asks more" to the end of the paragraph and replace with one sentence: "The central limit theorem in Appendix~\ref{app:theory-more} needs in addition a stabilisation rate and deterministic limits." Also cut at 172: "The Lipschitz constant is stated per compact sub-interval because it degrades as $\epsilon\downarrow0$; no \emph{deterministic} bandwidth floor is needed, since the proof localizes over $[1/j,\epsilon_0]$ and intersects countably many almost-sure events (Appendix~\ref{app:theory})" (this is proof mechanics; line 570 has it in full).
- Saving: ~220 words plus one display.

**D2. The straggler paragraph recites Table 3 and Fig. 2.**
- Location: 238 (§6.1).
- Quote: "The asynchronous arm, wall-limited, holds $3200$--$3900$ simulations per second at every slowdown; the twin..."
- Diagnosis: 330 words. Throughputs at every slowdown for three arms and two granularities, all predicted-versus-measured pairs (401/402, 102/103, 202/202), the $3.3\times$ granularity sub-result and "agree to three significant figures" are all readable from Table 3. What the prose must carry is the one-line prediction and the instructive failure at $0\times$/$1\times$.
- Recommendation: Rewrite the paragraph as roughly: "One of $16$ workers carries a permanent post-evaluation delay of $0.1$\,s scaled by $0$--$20$ (Table~\ref{tab:twin}). The prediction is a one-line calculation: the slow worker's evaluation time ($2.000$\,s at $20\times$) plus the $3.6$\,ms per-evaluation overhead measured on the fast workers is the generation time, so the twin cannot exceed $16/2.004=7.98$ simulations per second, and the measured asynchronous rate over that gives $401\times$ against a measured $402\times$; at $5\times$ and $10\times$ the agreement is the same. At $0\times$ and $1\times$ the prediction fails, instructively: with no straggler to wait for, the collective itself costs about $0.3$\,s per barrier on a $4$\,ms workload, which the model does not contain, and coarsening the barrier to one per $112$ evaluations amortises it. From $5\times$ up the barrier's cost no longer depends on how often it is taken." (~130 words.)
- Saving: ~190 words.

**D3. The calibration paragraph repeats Table 7 cell by cell.**
- Location: 330 (§6.4).
- Quote: "On the one-dimensional Gaussian mean it tracks the nominal level at every level ($0.51/0.80/0.90/0.94$); the synchronous baseline under-covers..."
- Diagnosis: Nine four-tuples of coverage values appear in the prose and in the table. The prose should carry the pattern (nominal in 1-D where the baseline under-covers; both modes kept; conservative on $A$ and $g$ in 4-D; the support explanation) and leave the values to the table.
- Recommendation: Replace the numbers with pattern statements: "On the Gaussian mean it tracks the nominal level at every level while the synchronous baseline under-covers at every level, an ordering that the Monte Carlo error of ${\pm}0.01$ resolves. On a bimodal target ... the archive retains both modes in $86\%$ of trials and coverage is nominal. On g-and-k the estimator is conservative on $A$ and $g$ and near nominal on $B$ and $k$, with $B$'s rank histogram the only uniform one at the $1\%$ level; the baseline is near nominal on $A$ and $g$ and under-covers $B$ and $k$." Keep the reporting-support sentence and the "(0.85/0.89)" archive-only values, which are the argument.
- Saving: ~110 words.

**D4. The real-simulator paragraph recites Table 12.**
- Location: 277 (§6.1).
- Quote: "the utilisation ratio is $1.18$--$1.21\times$ at every worker count --- the asynchronous arm at $99$--$99.8\%$, the twin at $82$--$85\%$ --- and equals..."
- Recommendation: Rewrite the middle as "Throughput is utilisation over mean simulation time, and the two factors separate: the utilisation ratio is $1.2\times$ at every worker count and matches both the straggler factor measured inside the twin's own generations and the prediction from the asynchronous arm's runtime distribution (Table~\ref{tab:twin-cpm}). The remaining $1.5$--$2.3\times$ is the twin's simulations taking longer, uncorrelated with the parameters, which we do not attribute and do not count; the utilisation ratio is therefore a lower bound on this workload." Cut the parenthetical "(synchronised I/O bursts would be a secondary cost of the barrier, filesystem load none of its doing)".
- Saving: ~80 words.

**D5. Two passages explain at length why a number is not quoted.**
- Location: 311 (§6.2), from "The control run with the shipped starting bandwidth gives $7.1\times$" to "we do not use it", and from "At $80^3$ the same run gives" to "not a property of the method".
- Quote: "The control run with the shipped starting bandwidth gives $7.1\times$ at equal wall clock rather than $4.1\times$, and the difference is a warning about matched comparisons..."
- Diagnosis: ~140 words justify not quoting $7.1\times$ and not quoting $17\times$. The first is a pre-emptive answer to a question no reader will ask (the control row is in Table 8 with its tolerance; the reader can see it). The second is worth one clause.
- Recommendation: Cut the $7.1\times$ passage entirely. Replace the $80^3$ passage with: "At $80^3$ the asynchronous arm completes $2.1\times$ the simulations and is $2.1\times$ more efficient per simulation; at a one-hour budget both arms are still on the steep early part of their curves, so we do not quote an equal-wall-clock ratio there."
- Saving: ~100 words.

**D6. The placement explanation in §6.3 carries six throughput numbers.**
- Location: 318 (§6.3), from "That boundary belongs to the placement" to "fixes it for one."
- Quote: "on one packed node the same Lotka--Volterra configuration completes $6240$ simulations per second on the asynchronous arm against $2952$..."
- Recommendation: Rewrite as: "That boundary belongs to the placement and budget it was measured under: the cheap benchmarks ran $48$ ranks spread over four nodes, where every arrival crosses the network, and on one packed node the same Lotka--Volterra configuration gives a throughput ratio of $2.1$ instead of $0.96$ (Table~\ref{tab:scaling}). The crossover moves down with tighter placement." (The 840-to-3550 within-run rise is a diagnostic detail; move it to Appendix D if wanted.)
- Saving: ~60 words.

**D7. Fig. 2 duplicates Fig. 1 and Table 3, and its caption reconciles it with another campaign.**
- Location: 240-244 (Fig. 2 and caption), 243 in particular.
- Quote (243): "This is the $300$\,s wall-limited campaign; the twin campaign's asynchronous arm in Table~\ref{tab:twin} agrees within $15\%$ except at $1\times$ ($2645$ here against $3112$ there)."
- Diagnosis: Fig. 2 shows async-versus-pyABC throughput against slowdown on a different campaign from Table 3; C1 rests on Fig. 1 and Table 3. Its caption spends 40 words reconciling two campaigns' numbers, which is exactly the kind of cross-check that belongs in an appendix (it was added in response to digest item 13).
- Recommendation: Move Fig. 2 to Appendix F (next to the heterogeneity figures) with a one-line caption; keep the sentence in 238 that the pyABC baseline falls from ${\approx}8800$ to ${\approx}60$. Drop the reconciliation sentence or move it to Appendix D.
- Saving: ~90 caption words in the main text plus one float.

**D8. The Propulate call signature in §3.1.**
- Location: 78.
- Quote: "For an efficient implementation we leverage the existing Propulate framework, whose propagator exposes a single call \texttt{\_\_call\_\_(inds) -> Individual}..."
- Diagnosis: The method section does not need the Python signature or "without persistent archive state or assimilation callbacks"; the one fact the reader needs (the propagator receives the whole history and emits one candidate) is already implied by line 76 and stated in Algorithm 1.
- Recommendation: Replace with "The Propulate propagator interface \cite{taubert2023propulate}, which hands the whole evaluated history to each call and takes one candidate back, is exactly this contract." Move the signature to Appendix C if wanted.
- Saving: ~45 words.

**D9. §3.4 carries tuning prose for S and a cost argument for the post-hoc pass.**
- Location: 106 ("Setting $|\mathcal{S}|=0$ recovers ... keeping each update cheap.") and 108 ("The reported estimator is a one-time pass of cost ... a bounded fraction of the run.").
- Quote (106): "Setting $|\mathcal{S}|=0$ recovers single-current-proposal weighting; increasing $S$ enriches..."
- Recommendation: At 106, compress the two sentences to "$|\mathcal{S}|=0$ recovers single-proposal weighting; we use $S=20$, which captures most of the variance reduction of the full cumulative mixture at a modest per-call cost (Appendix F sweeps $S$)." At 108, compress to "The reported estimator is a one-time $\mathcal{O}(nk)$ pass over the history, run after the timed budget and excluded from every throughput measurement (Appendix~\ref{app:implementation})."
- Saving: ~60 words.

**D10. A normalisation remark inside Assumption 2.**
- Location: 153 (§4).
- Quote: "Nothing requires $\bar q_n$ to integrate to one: the proofs use only positivity, boundedness and Lipschitz dependence on the parameters..."
- Diagnosis: This is a remark about the proof, not part of the assumption; Appendix B says it at 532 and 570, and Table 9 (674) says it again.
- Recommendation: Cut from the assumption; keep the appendix statements.
- Saving: ~45 words.

**D11. Record-counting minutiae in the metric definition.**
- Location: 219 (§5.3).
- Quote: "count only records of completed simulations (the synchronous baseline also writes one record per accepted particle, and mixing the two flatters it), and order records by completion time"
- Recommendation: Move the parenthetical to Appendix D (protocol); keep "completed simulations only, ordered by completion time".
- Saving: ~20 words.

**D12. Benchmark paragraph repeats Table 2's cost and CV columns.**
- Location: 193 (§5.1), last sentence.
- Quote: "We run it at two sizes: a $50^3$ box, where one evaluation costs $13$\,s, and an $80^3$ box with twice the duration, where it costs $170$\,s and the runtime coefficient of variation rises from $0.05$ to $0.13$--$0.23$..."
- Recommendation: "We run it at two sizes (Table~\ref{tab:benchmarks}); the runtime spread rises with the box because runtime tracks cell count ($r=0.99$) and cell count is no longer bounded at $80^3$."
- Saving: ~30 words.

**D13. Table 5's caption carries eight replicate ranges as running text.**
- Location: 294.
- Quote: "Replicate ranges, pairing replicates by index (benchmarks in table order): per-simulation $0.77$--$1.12$, $1.02$--$1.15$..."
- Recommendation: Put the ranges in the table as parenthetical sub-entries under each ratio, or as two extra columns; the caption then needs only "replicate ranges in parentheses".
- Saving: ~45 caption words.

**D14. Fig. 5's caption restates §6.1 and §6.3.**
- Location: 285.
- Quote: "per-arrival coordination grows with the worker count, throughput peaks on one node and the baseline overtakes beyond three (\S\ref{sec:results-boundary})..."
- Recommendation: "Strong scaling at a fixed wall-clock budget against a synchronous baseline with population $\max(100,W)$. Left: Lotka--Volterra ($4$\,ms, $180$\,s; medians and inter-quartile ranges). Right: Cellular Potts $50^3$, earlier configuration ($5.1$\,s, $1800$\,s; log--log), with linear scaling dotted." Let §6.1 and §6.3 say what the panels show.
- Saving: ~55 caption words.

**D15. The $80^3$ posterior sub-result.**
- Location: 395 (§6.4), last sentence.
- Quote: "At $80^3$, with about $1000$ evaluations an hour, every method reports a weak posterior ($50\%/15\%$ asynchronous, $52\%/11\%$ baseline, $78\%/53\%$..."
- Recommendation: "At $80^3$, with about $1000$ evaluations an hour, no method reports a useful posterior, and we use that size for the systems measurement only." (If the authors want the numbers, Table 8 could gain a row.)
- Saving: ~25 words.

**D16. Two justifications of comparator design choices that the reader does not need.**
- Location: 216 (§5.2).
- Quote: "not a fixed threshold, which on an expensive simulator admits every draw" and "the wall-time-limited baseline uses a fixed population and generation count, which compares more cleanly than stopping rules on $\epsilon$"
- Diagnosis: Both explain an alternative the paper did not use. The first is a residue of the rejection-baseline fix; the second is a design note.
- Recommendation: Cut both clauses.
- Saving: ~35 words.

**D17. The in-run CV parenthetical at $80^3$.**
- Location: 279 (§6.1).
- Quote: "($0.13$ on the asynchronous one, which concentrates as it runs; the prediction uses the arm's own in-run distribution, not a prior-wide screen)"
- Recommendation: Cut; Table 2 gives the CV range and §5.3 says the prediction uses the run's own timing.
- Saving: ~25 words.

**D18. The $k\ge400$ crossover sub-result in §6.3.**
- Location: 320.
- Quote: "--- at three for $k\ge400$, since the per-arrival proposal reconstruction is $\mathcal{O}(k)$"
- Recommendation: Cut; Appendix F and Table 14 carry it and §6.4 already says the archive size is a systems cost.
- Saving: ~15 words.

**D19. Declarations carry repository inventory.**
- Location: 439 (Data availability), 445 (Code availability).
- Quote (439): "($37$ CSV files under \texttt{experiments/data/paper\_figures/}), the generator that produces each one from the campaign output (\texttt{experiments/scripts/make\_*.py}, $27$ of them, most with a \texttt{-{}-refresh} path..."
- Recommendation: Data: "The repository contains the summary data behind every figure and table, the scripts that regenerate each from the campaign output, and the committed output of every diagnostic quoted in the text; the raw per-particle records (${\approx}30$\,GB for the Gaussian-mean campaign alone) and per-rank timing logs will be deposited in a public archive upon publication." Code: cut the parenthetical listing what the test suite pins.
- Saving: ~90 words.

**D20. (appendix) Ablation and sensitivity report a metric the paper says is uninformative, at length.**
- Location: 957-973 (Appendix F, two subsections, two figures).
- Quote (958): "values below $0.080$ do not distinguish a correct posterior from a collapsed one, so the two studies show that no ingredient removal changes the archive's concentration, and nothing about posterior quality."
- Diagnosis: ~700 words and two figures whose own opening sentence limits their content to "no ingredient removal changes the archive's concentration"; the sensitivity paragraph then spends ~150 words scoping what "no failure region" may not be taken to mean. If the studies say nothing about posterior quality, they do not earn two figures.
- Recommendation: Replace both subsections with a three-sentence note (no figures): "An ablation (hard kernel; no AMIS) and a hyperparameter grid (perturbation scale, initial tolerance, archive size) on the Gaussian-mean benchmark, scored by the concentration diagnostic above, show that no ingredient removal or setting in the grid changes the archive's concentration ($0.07$--$0.15$ throughout). Removing AMIS is neutral on this uniform-runtime target, consistent with the parameter-coupled study. Calibration, which these diagnostics cannot see, is the archive-size sweep above." Keep the data in the repository.
- Saving: ~600 words and two floats (appendix).

**D21. (appendix) Protocol paragraphs that over-explain small effects.**
- Location: 723 (effective simulating ranks) and 727 (kill-and-resume), Appendix D.
- Quote (723): "We do not correct for it because the correction shrinks monotonically with scale while the gap we report grows with scale..."
- Recommendation: 723: "pyABC's rank $0$ dispatches and does not simulate, so the baseline has $W-1$ simulating ranks; we report the allocated $W$ for both arms, which flatters the asynchronous arm by $1/W$ ($2\%$ at $W=48$) and is not corrected." 727: halve it; cut the "kill landed during a checkpoint write" anecdote and "Three things prevent the replay, and they are worth separating".
- Saving: ~170 words (appendix).

**D22. (appendix) The repository bug-log paragraph.**
- Location: 771 (Appendix E).
- Quote: "Two properties of the stored records are documented in the repository's bug log and matter for reproduction."
- Diagnosis: Column-swap and dropped-weight defects in the record writer, both fixed, belong in the repository's changelog. The only fact the paper needs (pyABC weights unavailable for the older campaigns) is already at 223.
- Recommendation: Cut the paragraph.
- Saving: ~80 words (appendix).

## Defensiveness

**F1. The Limitations section over-enumerates.** See R1; two of its items are not limitations (a stated prediction being a prediction; a fixed $S$ that Appendix F in fact sweeps), and "the twin deadlocks intermittently at $\ge192$ ranks and needed retries" (414) is an engineering note that affects no reported number. Cut all three.

**F2. "Rather than X" contrast tails in §4 read as a reply to a referee.**
- Location: 129 ("rather than beside it as a list of caveats"), 132 ("rather than leaving it implicit"; "confronted rather than hidden"), 172 ("carry rather than assume away"), 188 ("assumed rather than enforced").
- Quote (129): "and it carries the implementation's departures from the ideal inside the statement, as a single ratio $r$, rather than beside it as a list of caveats."
- Diagnosis: Each says "we are being honest about this" instead of just being honest about it. One such sentence (129's) is a fair statement of the paper's bargain; four is a tone.
- Recommendation: Keep 129's clause. At 132, cut both sentences (R18). At 172, "up to a ratio $r$ that the results carry." At 188, "so the positive limit $\epsilon_\infty>0$ is assumed for them (Table 9)."
- Saving: ~30 words; the gain is tone.

**F3. "We state its role so that it is not read as a production run."** Line 195; see R14. Rewrite without the meta-statement.

**F4. Pre-empting a referee's arithmetic.**
- Location: 291 (§6.2).
- Quote: "(not their literal product, since the curves are not power laws over the range and each ratio is a median over replicates)"
- Diagnosis: "Compound into" already avoids the claim that they multiply; the parenthetical answers a review comment (digest item 3) rather than informing the reader.
- Recommendation: Cut.
- Saving: ~20 words.

**F5. The 15.6 parenthetical in the straggler paragraph.** Line 238; see R3. It exists to forestall a "your abstract's definition gives 15.6, not 401" objection. Once §5.3 defines the predicted quantity, it is unnecessary.

**F6. "Less contestable fix."**
- Location: 391 (§6.4).
- Quote: "but an a priori setting is the less contestable fix"
- Recommendation: "and is the check we apply; setting $\epsilon_0$ before the run is the fix."
- Saving: ~5 words; tone.

**F7. The conclusion hedges its own recommendation.**
- Location: 420 (§9).
- Quote: "we think generation-free execution is the right model, and we have tried to give the practitioner what adopting it needs"
- Diagnosis: "We think" and "we have tried to give" undercut a sentence whose content is the paper's thesis. Note also that this sentence says "consistent up to a \emph{measured} fidelity ratio" while the abstract and §4 say "partly measure"; the conclusion should not claim more than the abstract.
- Recommendation: "For simulator-based inference on machines where simulations are expensive and unequal, generation-free execution is the right model, and the practitioner adopting it has what the choice needs: a prediction of the gain from their own timing, and the two settings on which the reported posterior depends." Change "measured" to "partly measured".
- Saving: ~10 words; tone and consistency.

**F8. "We do not correct for it and do not claim the reweighting cancels it."**
- Location: 413 (§8).
- Diagnosis: A double disclaimer after a null result. The null result is the information.
- Recommendation: Covered by R1's rewrite ("and we do not correct for it").

**F9. The selection-effect study in Appendix F ends with the same double disclaimer, and "honest" appears as a label twice in the appendices.**
- Location: 851 ("We accordingly report the raw archive as robust here rather than claiming AMIS corrects the over-representation, and treat a formal censoring correction as future work"), 508 ("The bound \eqref{eq:tilt-bound} is honest but loose"), 860 ("This is the honest boundary of the asynchronous advantage").
- Recommendation: 851: "The raw archive is robust at these couplings; AMIS neither creates nor removes a mean bias here, and a censoring correction is future work." 508: "The bound is loose against these direct measurements". 860: "This is the boundary of the asynchronous advantage on a cheap, homogeneous simulator".
- Saving: ~30 words (appendix); tone.

**F10. Appendix A's "what is and is not new" paragraph argues with an imagined referee.**
- Location: 455.
- Quote: "stated for a fixed proposal it would not merit a theorem. ... We do not claim to weaken the hypotheses of any existing AMIS theorem; we claim a different bargain..."
- Diagnosis: The paragraph is a good statement of the contribution's shape, but it opens by conceding the theorem is "elementary" for a fixed proposal and closes with a "we do not claim" disclaimer. The three substantive points (adaptive at every draw; the floor makes Lindeberg trivial; $r$ is inside the statement) are what to keep.
- Recommendation: Cut the first sentence after "The $r$-tilted limit ... tilted target" and the final "We do not claim ... condition on the algorithm." sentence; keep the three points.
- Saving: ~60 words (appendix).

**F11. The CLT non-coverage of the reported configuration is asserted four times in the appendices.**
- Location: 472, 512, 516, 678 (Appendix A and Table 9); plus 188 and 412 in the main text.
- Quote (512): "We therefore do not claim Theorem~\ref{thm:clt} for the configuration these experiments run: its rate condition is not merely unverified there but measurably violated."
- Diagnosis: This is a load-bearing caveat (see D) but one statement per document part suffices: once in §4 (188), once at the theorem (472), once in Table 9.
- Recommendation: At 512, keep the first sentence; at 516, cut "We therefore state Theorem~\ref{thm:clt} for the damped- or frozen-adaptation variants when the target is weakly identified, and note that" and end the paragraph at the "room to spare" observation. Cut from 412 (R1).
- Saving: ~40 words (appendix).

**F12. "Two qualifications keep this from being read too broadly" and the sensitivity scoping paragraph.**
- Location: 888 and 967 (Appendix F).
- Quote (967): "``No failure region'' should therefore be read as a statement about one-dimensional point quality only."
- Diagnosis: 888 is fine for an appendix (it explains why the throughput column is an upper bound). 967 spends ~150 words scoping a study the paper has already said is uninformative about posterior quality; see D20.
- Recommendation: 888: keep. 967: covered by D20.

**F13. Fig. 2's caption reconciles campaigns; Table 3's caption names a script.** Lines 243 and 247; see D7 and R15. Both are traces of review responses.

**F14. The Lotka-Volterra extinction subsection announces its own candour.**
- Location: 976 (Appendix F).
- Quote: "One property of the Lotka--Volterra benchmark bounds what its statistical numbers can mean, and we state it here rather than leave it to be discovered."
- Recommendation: Start at "The simulator has an absorbing state". Halve the paragraph: the survival fractions, the survival-conditioned target, and "throughput unaffected, posterior not scored" are the content.
- Saving: ~100 words (appendix).

**F15. The "accepted" disclaimer in Appendix C contradicts the bullets that follow.**
- Location: 707.
- Quote: "We avoid the word ``accepted'' here precisely because acceptance is probabilistic under a smooth kernel."
- Diagnosis: The three bullets immediately below use "accepted" four times.
- Recommendation: Cut the parenthetical from "(This bandwidth-membership count" to "smooth kernel.)" and write "within the bandwidth" in the bullets, or keep "accepted" and drop the disclaimer.
- Saving: ~60 words (appendix).

# C. Section-by-section summary

Counts are from a script over the section boundaries (LaTeX commands, math, tables and algorithm bodies stripped; captions counted separately). Main text body 8,670 + captions 853 = 9,523.

| Section | Lines | Current (body + captions) | Target | Main lever |
|---|---|---|---|---|
| Abstract | 42 | 245 | 225 | Keep the numbers here; trim (iv) |
| 1 Introduction | 48-60 | 559 | 420 | General-ratio sentence and CV numbers out (R3, R22); crash clause out (R21); contributions without numbers (R2) |
| 2 Background | 62-72 | 249 | 200 | Merge 70/72 (R13) |
| 3 Method | 73-127 | 892 (232/139/80/355/86) | 700 | One pure-function statement (R4); Propulate call out (D8); §3.4 tuning and cost prose (D9, R17, R19) |
| 4 Theory | 128-189 | 1,069 | 760 | Assumption 5(ii) and CLT commentary to App. A (D1); filtration once (R18); proof mechanics out (D1, D10); contrast tails (F2) |
| 5.1 Benchmarks | 192-214 | 400 (349 + 51) | 330 | Table-2 numbers out (D12); instrument paragraph compressed (R14); add the earlier-configuration sentence (R9) |
| 5.2 Baselines | 215-217 | 298 | 250 | Design-alternative clauses out (D16); rejection detail once (R16) |
| 5.3 Metrics | 218-224 | 509 | 440 | "No longer report" out (R25); record parenthetical to App. D (D11); keep 221 as the single definition of the predicted quantity |
| 6.0 Results intro | 225-227 | 67 | 30 | Signposting out (R20) |
| 6.1 C1 | 228-288 | 1,363 (1,015 + 348) | 980 | Table-3/12 numbers out of prose (D2, D4); Fig. 2 to appendix (D7); captions (R11, R15, D14); attribution once (R5); 15.6 parenthetical out (R3) |
| 6.2 C2 | 289-315 | 664 (563 + 101) | 470 | Metric re-definition out (R10); attribution dup out (R5); 7.1x and 17x explanations out (D5); literal-product parenthetical out (F4); caption ranges into table (D13) |
| 6.3 C3 | 316-327 | 576 (515 + 61) | 420 | Rule to Discussion (R7); placement to one sentence (D6, R8); caption (R8) |
| 6.4 C4 | 328-396 | 1,498 (1,210 + 288) | 1,150 | SBC numbers out of prose (D3); mechanism once (R6); 393 compressed; takeaway dup out (R7); captions (R12, R16) |
| 7 Discussion | 397-409 | 418 | 380 | Keep the four rules and 408; drop straggler re-definition and CV numbers (R3, R22) |
| 8 Limitations | 410-418 | 503 | 170 | Rewrite as one paragraph (R1, F1) |
| 9 Conclusion | 419-421 | 213 | 140 | No numbers, no re-definition, no hedge (R2, R3, F7) |
| **Main text** | | **9,523** | **~7,070** | |
| App. A theory-more | 452-519 | 2,290 | 2,050 | F10, F11; receives Assumption 5(ii) |
| App. B proofs | 520-684 | 2,281 | 2,200 | R25 (590); otherwise leave |
| App. C implementation | 685-720 | 986 | 880 | F15; trim the benchmark numbers in 719 |
| App. D protocol | 721-765 | 1,157 | 900 | D21; R8 (764); R23 (762) |
| App. E CPM setup | 766-772 | 590 | 420 | R23 (769); D22 (771) |
| App. F additional results | 773-977 | 3,981 (2,673 + 1,308) | 2,900 | D20 (ablation/sensitivity); R24; F9, F14; caption 954 (R5) |

# D. Caveats that must stay (criterion 3 counterbalance)

These are the qualifications without which a claim in the paper would misrepresent what was measured. Each should appear **once** in the main text, at the location given, and may be echoed in a table or the appendix; the recommendations above remove the copies, not the originals.

1. **Consistency is up to the fidelity ratio $r$, of which two factors are measured and two bounded** (line 188, and Table 9). The theorem's statement is literally "up to $r$"; dropping the partial-measurement caveat would turn a tilted-target result into an exact one. Keep the abstract's "decompose and partly measure" and make the conclusion match it (F7).
2. **The central limit theorem does not cover the configuration the experiments run** (line 188, once; Appendix A at the theorem). Contribution 4 advertises a CLT; the reader must learn in the main text that it is for damped-adaptation variants and why (a weakly identified target, not asynchrony).
3. **The adapted filtration under asynchronous execution is an assumption** (Assumption 4, line 155, and Appendix B). It is the one place the theory meets the asynchrony the paper is about.
4. **The $50^3$ twin, scaling and theory-diagnostic runs used an earlier configuration of the Cellular Potts benchmark** (once in §5.1 per R9; tag in captions). Data provenance; a reader comparing Table 8 with Table 12 needs it.
5. **Only about $1.2\times$ of the $2.4\times$ Cellular Potts throughput gain is the barrier; the rest is pyABC's per-generation overhead, and $2.0\times$ is the attributable figure** (line 279). Without it the C2 headline would credit the mechanism with the comparator's overhead.
6. **The predictor's three out-of-domain configurations and the per-instrument misses** (lines 230, 238, 274). The prediction claim is credible because the paper says where it fails ($0\times$/$1\times$: collective latency; $384$ ranks: contention tail) and by how much (16-19% at the largest spreads).
7. **The twin's simulation-duration inflation is unattributed, so the utilisation ratio is a lower bound** (line 277, one sentence). It is why the Cellular Potts twin ratio is $1.2\times$ and not $1.9\times$.
8. **The C3 boundary is a property of the placement and budget it was measured under** (line 318, one sentence). Otherwise "4 ms" reads as a constant of the method.
9. **On the Gaussian mean the asynchronous method is worse per draw ($0.78\times$) and $4\times$ looser at equal wall clock** (lines 291, 318). This negative result is what gives C3 its content; it must not be softened.
10. **g-and-k coverage is conservative on $A$ and $g$ with non-uniform rank histograms for three of four parameters, and the bimodal target keeps both modes in 86% of trials** (line 330). "Calibrated in one dimension and conservative in four" (conclusion) is the honest summary and depends on this.
11. **Cellular Potts recovery is on one synthetic dataset: containment, not coverage** (line 364). One clause.
12. **The effective sample size of the reported posterior is a few multiples of $k$ regardless of history length** (line 356). This is a finding, not a caveat; it is the paper's stated headroom (408) and should stay prominent.
13. **The reported posterior is only as good as the bandwidth the schedule reached; on short histories and under lockstep it stalls** (line 393, compressed). It explains the twin's lower block in Table 3 and the $\epsilon_0$ rule.
14. **Completion-time selection: every-particle reporting over-represents fast regions; the runtime-coupled study found no shift at the couplings tested and no correction is applied** (one sentence in Limitations per R1).
15. **The synchronous baseline's population of 100 on 47 ranks idles by construction, and its Gaussian/g-and-k posteriors are scored uniformly weighted because histories were not kept** (line 216 and one sentence in Limitations). These are handicaps of the comparator that favour the proposed method; they must be disclosed once.
16. **Lotka-Volterra sends 98% of simulations to extinction and is not scored for posterior quality** (line 193; Appendix F).
17. **The two cautions on the predictor** (line 402: the timing sample must outlast the runtime tail; a persistent straggler contributes its own evaluation time). These are the only place the Discussion adds something the results do not, and a practitioner applying rule 2 needs them.

# E. Top 10 edits by payoff

1. [ ] Rewrite §8 Limitations as the one-paragraph version in R1 (drop the two non-limitations and the deadlock note). (~330 words)
2. [ ] Move Assumption 5(ii), display \eqref{eq:rate} and the CLT commentary at 172 to Appendix A; cut the proof-mechanics sentence at 172 and the normalisation remark in Assumption 2 (D1, D10). (~265 words, one display)
3. [ ] Strip the numbers from contributions 2-4 (57-59) and from the conclusion (420); the abstract carries them. Fix "measured" to "partly measured" in the conclusion and drop "we think / we have tried" (R2, F7). (~130 words)
4. [ ] Rewrite the straggler paragraph (238) to the ~130-word version in D2, including removal of the 15.6 parenthetical; cut the Table-12 numbers from 277 (D4). (~270 words)
5. [ ] Let the Discussion own the practitioner rules: cut the rule statements at 320, 391 and 395 (R7); cut the straggler re-definition and CV numbers from 400 (R3, R22). (~160 words)
6. [ ] State the general-ratio versus straggler-factor point once, at 221: cut from 50, 238, 274, 400, 420 (R3). (~140 words)
7. [ ] Reduce the pure-function/running-sampler pair to §3.1: cut from 78 (last sentence and the Propulate call), 106, 108, 111 (R4, D8, D9, R17, R19). (~250 words)
8. [ ] In §6.2: remove the metric re-definition (291), the attribution duplicate and the two "why we do not quote" passages (311), the literal-product parenthetical (291); move Table 5's ranges into the table (R10, R5, D5, F4, D13). (~280 words)
9. [ ] Replace the SBC values in 330 with pattern statements (D3); compress 393 to the twin-stall finding plus the re-report result (R6); shorten the Fig. 4 and Table 8 captions (R12, R16). (~230 words)
10. [ ] Move Fig. 2 to Appendix F; cut the campaign-reconciliation sentence and the script name from captions 243/247; shorten the Fig. 1 and Fig. 5 captions to what the text does not say; put the earlier-configuration sentence in §5.1 and reduce later mentions to a tag (D7, R11, R15, D14, R9). (~250 words, one float)

Bonus (appendix): replace the ablation and sensitivity subsections with the three-sentence note in D20 and drop their two figures; cut the bug-log paragraph (D22) and halve kill-and-resume (D21). (~850 words, two floats)
