# Verification of round 01 (Claude verifier)

Scope: the 39 items with `status: applied` in `proposals.md` (P-001 to P-025, P-030, P-031, P-033, P-036 to P-039, P-041, P-044 to P-049). Rejected items were not judged.

Token check: for every applied item the Before and After blocks were compared mechanically for numbers, `\cite`, `\ref`/`\cref`/`\eqref`, `\label` and `\SI`/`\num` tokens (multiset equality). No item added or removed any such token. Every After block is present verbatim in `latex/tmlr/tmlr-article.tex` and no Before block survives. The three formula-heavy items (P-002, P-046, P-049) were word-diffed: only punctuation and connectives differ, every formula is byte-identical.

Context checks against the manuscript: P-004 ($\lambda_n=\max(\nu_n,\delta)$ is the floored prior weight in \eqref{eq:snapshot-denominator}; `$w_n` occurs nowhere), P-010 ("an $O(W/n)$ perturbation" is two sentences earlier in the same Limitations paragraph), P-013 (the third factor from the left in the decomposition display is labelled "(b) quadrature"), P-015 (five replicates times two dimensions), P-021 (\eqref{eq:tracking} is weighted average minus integral, so $A_n,B_n$ are the averages), P-024 (the full order-statistic mechanism remains in the "Which runs the limits describe" paragraph of \S3, which the row cites), P-001 (the appendix's "in full" statement opens with the same words, "$\epsilon_\infty>0$ and $r$ are deterministic").

## Verdicts
| id | verdict | reason (one sentence) |
|---|---|---|
| P-001 | improved | Each hypothesis now has a verb and the wording matches the appendix's own "in full" statement; the leading "In addition," sits slightly oddly beside "clause (ii) implies clause (i)" but the appendix resolves it. |
| P-002 | improved | Two punctuation changes turn a 93-word chain into three sentences that start at the three logical steps; every formula is byte-identical. |
| P-003 | improved | "Both measured factors, (a) and (b)" names the antecedent that "both factors" lacked; numbers unchanged. |
| P-004 | improved | Replaces an undefined symbol by the one the display defines ($\lambda_n$); verified against \eqref{eq:snapshot-denominator}, and `$w_n` appears nowhere else (author confirmation already flagged in the proposal). |
| P-005 | improved | The repeated "which" pins "may be zero" to the limit rather than to the schedule. |
| P-006 | improved | Pointer sentence now leads with where the material is; same three items, same two references. |
| P-007 | improved | Split at the theorem/corollary boundary; both claims and the "so that" consequence kept. |
| P-008 | improved | "the theorem's two conditions" removes a three-way ambiguous "its". |
| P-009 | improved | The two-regime contrast survives with "by contrast"; one sentence becomes two. |
| P-010 | improved | "$O(W/n)$ term" reuses the name given two sentences earlier instead of an unintroduced "boundary term"; the limitation is stated as before. |
| P-011 | improved | Names $r$ as the subject and spells out the elided second "bounded". |
| P-012 | improved | Replaces a pronoun whose nearest antecedent was the wrong quantity. |
| P-013 | improved | "factor (b)" replaces a positional count; verified against the labelled display. |
| P-014 | improved | $W_j$ and $Z_j$ get a reading cue at first use; formula and constants unchanged. |
| P-015 | improved | Supplies the missing noun "marginals"; the count is five replicates times two parameters. |
| P-016 | improved | One split at the semicolon and a colon for "namely"; all four numbers and both references present. |
| P-017 | improved | The unparseable parenthetical becomes a clause with a verb; claim identical. |
| P-018 | improved | "lose" states what actually happens to the bound; the earlier wording read as the bound going to zero. |
| P-019 | improved | The $w_i$ worker-index cue removes a real clash with the parent-weight symbol; a parenthetical inside a lemma statement is a little heavy but acceptable. |
| P-020 | improved | Same device for the index $r$ versus the fidelity ratio. |
| P-021 | improved | The letters are assigned explicitly; checked against \eqref{eq:tracking}. |
| P-022 | improved | The definition of $R$ becomes its own sentence; justifications and references unchanged. |
| P-023 | improved | Colon replaces the second "because"; argument unchanged. |
| P-024 | improved | Removes a near-verbatim repeat of the \S3 mechanism from the status table while keeping the verdict, the one-clause mechanism and the cross-reference; the dropped "$2k$" detail is still stated in \S3. |
| P-025 | improved | Split where the grammar already switched from noun phrase to finite clause; all references kept. |
| P-030 | improved | Definition and validation are separate sentences; "the throughput a generation allows" is clearer than the elided version. |
| P-031 | regressed | The split is right, but "In calibration experiments, the estimator is calibrated where..." puts an awkward echo into the abstract's most-read sentence. |
| P-033 | regressed | Rewriting (B3) in the active voice is good, but "which earlier results it has received" dropped "by then", the time qualifier that the proof of Lemma adapted actually uses ("reached $w_i$ by $T_i$"). |
| P-036 | improved | Theorem scope and diagnostic evidence are separated; the only "Section~\ref" in a manuscript that otherwise uses \S\ref is fine sentence-initially. |
| P-037 | improved | One split; the Jensen inequality, the comparison and the interpretation are untouched. |
| P-038 | neutral | The split and "In particular" are fine, but dropping "The identity says that" loses the identity/bound parallel with the next sentence ("The bound replaces..."). |
| P-039 | improved | The first mixture is defined before the two transformations; the sequential dependence is kept. |
| P-041 | neutral | The split is good, but the added "at that step" has no clear antecedent and reads as a vague hedge. |
| P-044 | improved | Measurability consequence stated after the functional dependence; "therefore" preserves the inference. |
| P-045 | improved | "This boundary" replaces a buried relative-clause antecedent. |
| P-046 | improved | Four sentences follow the proof order; formulas byte-identical. |
| P-047 | improved | The prohibited reading comes first, then the reason; "so Lemma B.x does not control that conditional expectation" is a valid inference from the two facts. |
| P-048 | improved | The two routes of the length bias are stated before their treatments; "This second effect" has an explicit antecedent. |
| P-049 | improved | Algebra first, then the boundedness-plus-convergence step as a "Because" clause. |

## Items to revert
- P-031 (regressed): keep the split, drop the echo. Suggested wording for the second sentence: "Empirically, it is calibrated where a pyABC baseline with the same kernel and schedule under-covers." (Alternatively revert to the Before.)
- P-033 (regressed): keep the rewrite, restore the qualifier: "... which worker proposes next, when it proposes, and which earlier results it has received by then." One-word-pair fix, no revert needed.

Optional (neutral, not revert candidates):
- P-038: consider "The identity says that the tilt moves the target ..." as the first sentence to keep the identity/bound pairing with the following sentence.
- P-041: consider "so the emitted density itself adds no gap" in place of "so no further gap arises at that step".

## Overall
The round improved the manuscript's readability: 35 of 39 edits are genuine gains, almost all of them single splits, pronoun-to-noun substitutions or reading cues at first use, and none touched a number, citation, label or formula. The editor's choices show no worrying pattern: caveats were not deleted (the one duplication removal, P-024, keeps the caveat and the cross-reference), no sentence was flattened into a slogan, and the two regressions are a stylistic echo in the abstract and a dropped two-word time qualifier in an assumption, both fixable without reverting the split. I would not run another full readability round; after the two fixes above, one short targeted pass over the compiled Section 3 and Appendices A-B for transitions around the new sentence boundaries is enough.
