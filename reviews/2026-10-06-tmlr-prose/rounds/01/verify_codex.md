## Verdicts

| id | verdict | reason (one sentence) |
|---|---|---|
| P-001 | changed-message | Although the caveat is repeated later, the edit removes `\ref{ass:filtration}`, which the required token audit treats as a message change. |
| P-002 | improved | The revision removes a formulaic announcement and awkward “And” opener while preserving all three theoretical distinctions. |
| P-003 | improved | The deleted sentence merely restates the bandwidth limitation and the purpose of the preceding order-statistic check. |
| P-004 | changed-message | Despite the duplicated caveat, the edit removes `\ref{sec:experiments-baselines}`, which must be reported as a message change. |
| P-005 | improved | Splitting the overloaded list item exposes the role of the fidelity ratio without losing any result or qualification. |
| P-006 | improved | Three shorter sentences make the causal chain from parameters to cell count, summaries, and runtime substantially clearer. |
| P-007 | improved | The revision replaces an unclear ceiling metaphor with the direct statement that runtime spread predicts the barrier cost. |
| P-009 | improved | The threshold conclusion is stated more directly and in a more natural academic register. |
| P-010 | improved | The concise active construction retains the comparators’ distinct purposes while removing unnecessary commentary about naming. |
| P-011 | improved | Folding the table pointer into a parenthesis foregrounds the substantive synchronization argument. |
| P-012 | improved | The revision removes repetition and uses the colon effectively to introduce the consequences of having no shared mutable state. |
| P-013 | improved | Removing the numbered announcement and sentence-initial “And” improves flow without changing any finding. |
| P-014 | improved | The sentence duplicates the paragraph heading and can be deleted without loss. |
| P-015 | improved | The two emphasized subparagraphs already make the two uses of snapshots evident, so the announcement is unnecessary. |
| P-016 | changed-message | Removing “reliably” drops a distinct claim about dependable delivery rather than merely tightening the prose. |
| P-017 | changed-message | Replacing “the binding one” with “the harder condition” weakens an active-limitation claim into a comparative statement about difficulty. |
| P-018 | improved | “With a margin” and “lies between” provide a more formal rendering of the same three-way comparison. |
| P-019 | improved | The direct subject and semicolon remove an unnecessary cleft while preserving the cost pointer. |
| P-020 | changed-message | Saying the boundary “disappears” is stronger than saying it “lifts,” which only claims that real simulation cost moves the scaling boundary outward. |
| P-021 | improved | “Contains” is more precise and natural than “carries” for describing appendix contents. |
| P-022 | improved | “Lies … in total variation from” is more conventional mathematical prose than “sits.” |
| P-023 | improved | The preceding sentence already establishes the memory effect, while bit identity fully preserves the important value-invariance claim. |
| P-025 | improved | “Has the largest effective sample” is plainer and more idiomatic than “carries.” |
| P-027 | improved | Giving the effective-sample-size limitation its own sentence makes the abstract’s negative result easier to identify. |
| P-028 | improved | The revision directly maps each benchmark class to its experimental purpose and retains all three roles. |
| P-030 | improved | Separate declarative sentences make the three kinds of weights and their roles easier to retrieve. |
| P-031 | improved | The revision states the consistency dependency directly instead of posing it as a narrated question. |
| P-034 | improved | The rewrite clearly separates fixed coordination cost, benchmark-dependent statistical efficiency, and the throughput-based boundary. |
| P-035 | improved | The split sentences clarify the evidence-to-interpretation chain while retaining the unfavorable four-dimensional result and its limitation. |
| P-036 | improved | Naming contraction explicitly is clearer, and the numerical comparison still conveys how cell volume distinguishes the methods. |
| P-037 | changed-message | “Remains consequential” is weaker than saying the transient “binds,” which identifies it as an active limitation in the intended regime. |
| P-038 | improved | Giving each archive-size experiment its own sentence cleanly distinguishes re-reporting, an actual run, and the cross-benchmark result. |
| P-042 | improved | Separating the three theoretical limitations makes each one visible without altering its scope or qualification. |
| P-045 | improved | The revision states the decomposition construction directly and removes an unnecessary instructional metaphor. |
| P-046 | improved | The informative heading and shorter explanation identify exactly which two contributions are estimated and how. |
| P-047 | improved | The revision directly connects reconstructibility to empirical assessment while removing comparative meta-commentary. |
| P-049 | improved | The split removes unsupported emphasis on implementation ease while retaining the option and both complexity results. |

## Items to revert

- P-001: Use “Under asynchronous execution, this factor also includes the filtration gap of Assumption~`\ref{ass:filtration}`.” to retain the protected reference without repeating the full explanation.
- P-004: Use “The comparison uses the same kernel and bandwidth schedule (`\S\ref{sec:experiments-baselines}`).” to preserve the protected cross-reference concisely.
- P-016: Revert the deletion of “reliably,” or establish separately that only predictability is intended to be claimed.
- P-017: Use “The `\emph{distance to the limit}` is the binding condition.” to remove the intensifier while preserving the original claim.
- P-020: Use “This boundary shifts to larger worker counts once an evaluation has real cost (Fig.~`\ref{fig:scaling}`, right).”
- P-037: Use “It remains a binding limitation in the regime the method is designed for.”

## Overall

The round substantially improves the manuscript’s voice and readability, especially by splitting overloaded sentences and replacing narrated structure with direct claims. The main concern is that a few vocabulary substitutions weaken or strengthen technical conclusions, while two deletions violate the explicit protection of reference tokens. I would run one more restrained round focused on remaining clause-heavy prose, with stricter protection for technical qualifiers and LaTeX references.