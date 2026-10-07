## Verdicts

No applied item added or removed a number, `\cite`, `\ref`/`\cref`, `\label`, `\SI`, or `\num` token.

| id | verdict | reason (one sentence) |
|---|---|---|
| P-001 | improved | Adding verbs makes the assumption parseable while preserving its conditions and appendix pointer. |
| P-002 | improved | The sentence breaks expose the three proof steps without altering any formula or qualification. |
| P-003 | improved | Naming factors (a) and (b) removes the ambiguous antecedent and separates the observation about factor (c). |
| P-004 | improved | Replacing the incorrect \(w_n\) with the defined prior-weight symbol \(\lambda_n\) corrects the reference without changing the intended claim. |
| P-005 | improved | Repeating “which” makes clear that the limit, rather than the schedule, may be zero. |
| P-006 | improved | Front-loading the appendix location makes the three listed results easier to identify. |
| P-007 | improved | Separating the finite-run and limiting results clarifies their distinct conditions while preserving both claims. |
| P-008 | improved | “The theorem’s” gives the two conditions an unambiguous antecedent. |
| P-009 | improved | Splitting the two bandwidth regimes makes the contrast easier to follow without weakening it. |
| P-010 | improved | Referring to the previously introduced \(O(W/n)\) term makes the limitation locally identifiable. |
| P-011 | improved | Naming \(r\) as the subject and repeating “bounded” makes both parts of the argument explicit. |
| P-012 | improved | Replacing “it” with \(r_{\mathrm{floor}}\) removes a potentially incorrect antecedent. |
| P-013 | improved | Referring to factor (b) by label is clearer than asking the reader to count factors in the display. |
| P-014 | improved | Defining \(W_j\) and \(Z_j\) at first use makes the probability expression immediately interpretable. |
| P-015 | improved | Supplying “marginals” clarifies that the ten cases are the five-replicate, two-parameter marginal comparisons. |
| P-016 | improved | The split cleanly separates the empirical looseness of the bound from the interpretation of the identity. |
| P-017 | improved | The revised parenthetical states directly that removing one contribution does not establish \(r\equiv1\). |
| P-018 | improved | “Lose” correctly describes the disappearance of an \(n\)-uniform weight bound as \(\delta\) tends to zero. |
| P-019 | improved | The parenthetical resolves a genuine symbol collision between the worker index and the history’s parent weight. |
| P-020 | improved | The added cue prevents the integer index \(r\) from being mistaken for the fidelity ratio, although renaming the index would be cleaner. |
| P-021 | improved | Explicitly assigning the uppercase and lowercase quantities removes a consequential notational ambiguity. |
| P-022 | improved | Giving the definition and role of \(R\) its own sentence makes the displayed bound easier to verify. |
| P-023 | improved | The colon removes repetitive causal wording while preserving the stable-convergence justification. |
| P-024 | improved | The shorter version retains the failed condition and its mechanism while appropriately referring readers to the full discussion. |
| P-025 | improved | The sentence boundary makes the change from a contribution noun phrase to the authors’ derived results easier to follow. |
| P-030 | improved | Repeating “throughput” clarifies the ratio, and the split distinguishes the prediction from its validation. |
| P-031 | regressed | “In calibration experiments, the estimator is calibrated” is conspicuously tautological despite the useful sentence split. |
| P-033 | improved | The revision separates the scheduling rule, independence assumption, filtration, and conditional laws into a clear sequence. |
| P-036 | improved | The formal scope of the theorem is now cleanly separated from the empirical assessment of the production schedule. |
| P-037 | neutral | The original semicolon already connected the variance comparison and its interpretation clearly, so the split makes little practical difference. |
| P-038 | regressed | The revision redundantly paraphrases the equation and makes scale invariance sound like a consequence rather than the concise explanation of the preceding contrast. |
| P-039 | improved | Defining the first mixture before describing the two successive transformations makes their relationship easier to retain. |
| P-041 | improved | “At that step” properly limits the no-gap claim, and the reconstruction boundary is then presented as a separate consequence. |
| P-044 | improved | Separating functional dependence from its measurability consequence makes the proof step easier to scan. |
| P-045 | improved | “This boundary” gives the interpretation an explicit antecedent and clarifies the progression from reconstruction to bound. |
| P-046 | improved | The new sentence boundaries expose the normalizer argument’s definition, per-axis bound, and product-level conclusion. |
| P-047 | improved | The prohibited interpretation is stated before the measurability reason, making the lemma’s scope clearer. |
| P-048 | improved | The direct and indirect deadline effects and their different treatments are now plainly separated. |
| P-049 | improved | Completing the algebraic expansion before invoking boundedness and convergence clarifies why the cross terms vanish. |

## Items to revert

- P-031 — Use: “We prove that the estimator targets the smooth-ABC posterior tilted by a ratio that measures how faithfully its denominator represents the proposals used, with exact consistency when that ratio tends to one. Empirically, it is calibrated where a pyABC baseline with the same kernel and schedule under-covers.”
- P-038 — Revert to the original concise sentence, whose colon directly connects the variation claim to the scale-invariance identity.

## Overall

The round substantially improved the manuscript’s voice and readability, and the protected-token audit found no additions or removals. The only worrying pattern was occasional over-splitting or explanatory expansion of sentences that were already compact and clear, most notably P-031 and P-038. I would not run another broad readability round, but I would make the two targeted revisions above.