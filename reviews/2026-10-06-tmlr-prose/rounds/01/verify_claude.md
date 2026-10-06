# Round 01 verification (Claude)

- reviewer: Claude (fresh verifier, no prior context)
- date: 2026-10-06
- inputs: proposals.md (37 applied items), round01.diff, tmlr-article.tex for context

## Verdicts
| id | verdict | reason (one sentence) |
|---|---|---|
| P-001 | changed-message | Token rule only: a `\ref{ass:filtration}` was removed; the caveat survives verbatim in item (d) ("Under asynchronous execution this factor additionally carries the gap ... Assumption~\ref{ass:filtration}, which is a property of the schedule") and the edit was pre-authorized, so the prose itself is improved and I would not revert. |
| P-002 | neutral | "The present scheme, however, is adaptive" carries the contrast cleanly and the citations are intact, but without the count the second and third departures (floor, $r$) now read as unconnected sentences rather than a list of three. |
| P-003 | improved | Pure punchline plus a caveat already asserted by the paragraph title ("The schedule must move") and in 6.4 ("is the check we apply"); the paragraph now ends on the measurement. |
| P-004 | changed-message | Token rule only: a `\S\ref{sec:experiments-baselines}` was removed; the same-kernel-and-schedule caveat is stated in the Introduction (l. 61), the Results preamble (l. 247) and the Discussion opener (l. 369), so the limitation is still carried and I would not revert. |
| P-005 | improved | Splitting into two sentences surfaces the main relation (consistency up to a ratio) and keeps every item, both `\S\ref`s and the C4 label. |
| P-006 | improved | The three-link causal chain becomes three sentences with one mechanism each; the problem-versus-machine contrast, the $0.99$ correlation and $80^3$ are unchanged. |
| P-007 | improved | "the ceiling its runtime spread sets ... the ceiling rises" becomes "as its runtime spread predicts, and it rises", which names the relation the paragraph just established; all numbers kept. |
| P-009 | improved | Drops "floor to clear ... and stop" slogan vocabulary while keeping the conclusion (threshold, not trade-off) and both consequences. |
| P-010 | improved | "We use three comparators, each for a different question" replaces a sentence that narrated its own terminology; the three `\emph` names follow immediately. |
| P-011 | improved | "(last row)" as a parenthetical replaces "The decisive row is the last", keeping the mechanism and the cross-reference. |
| P-012 | improved | The restated consequence is folded into the "so workers never wait" clause; every listed property (no shared state, replay recovery, estimate invariance) is kept. |
| P-013 | improved | Removes "Three findings fixed the setup" and the "And" opener; the three findings, their order and all numbers ($12$, $9$, $|\cos|=0.07$) are unchanged. |
| P-014 | improved | The deleted sentence restated the paragraph heading "Two configurations". |
| P-015 | improved | "Snapshots are used in two places" narrated the two titled paragraphs that follow; the paragraph now ends on the substantive point that any snapshot is rebuildable. |
| P-016 | improved | Drops "reliably", which no result in C1 supports, and keeps "predictably", which the barrier prediction does support. |
| P-017 | improved | "comfortably met" to "met" and "the binding one" to "the harder condition" remove an intensifier and a vague term without softening; the $O(\log n)$ evidence is unchanged. |
| P-018 | improved | "with a margin" and "lies between" are the formal-register equivalents of "room to spare" and "lands between"; the Appendix reference is kept. |
| P-019 | improved | Cleft removed; $m=21$ and the `\S\ref` are kept. |
| P-020 | improved | "it disappears once an evaluation has real cost" names the relation directly in place of "lifts" and "carries"; the figure reference is kept. |
| P-021 | improved | "contains" for "carries" in an appendix preamble; the list of contents is identical. |
| P-022 | improved | "lies" for "sits" aligns the appendix with Section 4 for the same number; the $0.2$--$0.5\%$ and all other numbers are unchanged. |
| P-023 | improved | Drops the "changes memory, not the value" restatement; the memory claim is in the preceding sentence and the precise claim (bit-identical) remains. |
| P-025 | improved | "has the largest effective sample" for "carries"; $303$ vs $66$ of $100$ kept. |
| P-027 | neutral | Splitting the abstract's last sentence is cleaner, but "is the remaining limit" (ESS is the method's limiting factor) became "remains limited to" (ESS is bounded), a slight loss of the original point that the Discussion still carries. |
| P-028 | neutral | Verb-first list reads less like an argument template, but "measure performance on a real simulator" is vaguer than the original and "tested exactly" / "where the method is meant to be used" are dropped (the latter is stated in the abstract and 6.4); all refs kept. |
| P-030 | neutral | Three short declaratives remove the "three weightings ... only the last" template, but the original was already compact and the taxonomy signal is now carried only by the `\emph`s; refs unchanged. |
| P-031 | improved | "Consistency therefore depends on how well $\bar q_n$ represents the density from which the pooled sample was drawn" states the dependency the theorem formalizes instead of posing a question. |
| P-034 | improved | The systems mechanism and the problem-specific statistical ratio are separated into their own sentences, "common mechanism across benchmarks" is more precise than "has a mechanism", and both crossover numbers ($4$ ms; $2$--$4$ ms) and all refs are kept. |
| P-035 | neutral | The split helps and "this reversal" has its antecedent ("the order reverses"), but the pointer to the figure's lower panels is lost and "the ceiling binds" became the vaguer "The limitation appears" whose antecedent is two sentences back; all numbers and the `\S\ref` are kept. |
| P-036 | neutral | "For cell volume, contraction is ..." is plainer, but "The cell volume separates them" compactly stated the point the numbers make; every number kept. |
| P-037 | neutral | "The runs differ in the bandwidth used for reporting" is a clean replacement for the cleft, but "It remains consequential" is as vague as the "binds" it replaced; all numbers ($62$, $321\times$, $2\times$, $\epsilon_0=0.1$) kept. |
| P-038 | improved | Three experiments (post-hoc re-report, a real run, the cross-benchmark frontier) each get their own sentence; all numbers and the Appendix reference are kept. |
| P-042 | improved | Each limitation is now an independently visible sentence; nothing is softened and both refs are kept. |
| P-045 | improved | Names the construction (inserting intermediate mixtures) in place of the "walk down" instruction; "everything specific to the code enters through $r$" is already stated in Section 4 ("it absorbs everything the implementation does"). |
| P-046 | regressed | The new heading is a ten-word noun phrase that is narrower than the paragraph (which also covers the bounded factors (c) and (d), the (d) non-events and the looseness of the bound), and the body's "These two contributions" now relies on the heading for its antecedent; the body edit itself is fine. |
| P-047 | improved | States the evidential basis and the action in one sentence and drops the comparison with the paper's other assumptions; the `\ref{ass:stab}` is kept. |
| P-049 | neutral | Removing "a one-line change" drops a concrete, informative statement about implementation cost rather than mere emphasis, but the option and both complexity statements and refs are kept, so nothing the reader needs is lost. |

## Items to revert
- P-001: no revert. Flagged by the mechanical token rule only (a `\ref{ass:filtration}` disappeared). The deleted sentence was a near-verbatim duplicate of the closing sentence of item (d), which still cites the assumption. Keep as applied.
- P-004: no revert. Flagged by the mechanical token rule only (a `\S\ref{sec:experiments-baselines}` disappeared). The same-kernel-and-schedule caveat is asserted three times elsewhere (Introduction, Results preamble, Discussion). Keep as applied.
- P-046: replace the heading and the first sentence rather than revert. Suggested wording, keeping the sentence split the editor introduced:
  `\paragraph{Two factors measured, two bounded.} The prior floor and the snapshot quadrature can be estimated rather than bounded, because $\widehat\pi_n$ is a post-hoc pass. Recomputing \eqref{eq:snapshot-denominator} at a much larger $m$ and at the observed prior share gives a reference against which both are evaluated at every particle.`
  This heading covers the whole paragraph (including the sentence "The truncation constant and factor (d) are bounded rather than measured") and the body no longer needs the heading as an antecedent.

Optional touch-ups (verdict neutral, not required):
- P-035: restore the figure pointer, e.g. "The effective sample sizes (lower panels) explain this reversal."; consider "The ceiling binds once the target is identified well enough ..." so the term matches the Discussion's "effective-sample-size ceiling".
- P-027: if the abstract should keep the "limiting factor" sense, "That effective sample size, a few multiples of the archive size, is the remaining limit." as its own sentence does so.
- P-049: "A one-line change to the post-hoc pass decouples the two (...)" keeps the information without the emphasis.

## Overall
The round improved the manuscript's voice and readability: 24 of 37 items are clear improvements, almost all of them local (dropped announcements, clefts, punchlines and vague verbs, or long chains split into one-mechanism sentences), and no item changed a number, a result or its qualification; the two token-rule flags are pre-authorized duplicate deletions whose caveats survive elsewhere. The one pattern worth watching is in the codex-sourced rewrites (P-028, P-030, P-035, P-036, P-037, P-046, P-049), which tend to replace a compact, slightly informal sentence with a flatter or more generic one ("measure performance", "remains consequential", "The limitation appears") and occasionally drop a useful pointer or a concrete fact along with the emphasis; the Claude-sourced items were more conservative and none regressed. I would run one more round, scoped to the remaining vague verbs the author should decide as a policy ("binds", "sits at the threshold", "turns on") and to the long Appendix A paragraphs, and then stop, since the dominant templates are now gone and further passes would start trading precision for smoothness.
