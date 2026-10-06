## Verdicts

| id | verdict | reason (one sentence) |
|---|---|---|
| P-001 | improved | Removing the duplicated trend claim keeps this subsection focused on equal-wall-clock conversion while preserving the claim in the following subsection. |
| P-002 | improved | Deleting the repeated practitioner framing makes the sentence more direct without losing information. |
| P-003 | improved | The revision states more precisely that the sampler’s draw mixture must be reflected in the denominator. |
| P-004 | improved | The split reduces clause stacking, while the omitted explanation remains explicit in the main theoretical section. |
| P-005 | improved | Separating the experimental setting from the measurements makes the dense result easier to parse while preserving every qualification. |
| P-006 | improved | The split makes the diagnostics more scannable and retains the interpretation of the narrower marginals. |
| P-007 | improved | Giving the measurement caveat its own sentence improves its visibility without changing its force. |
| P-008 | improved | Reporting the full-range and restricted fits separately clarifies the comparison while retaining the upward-bias explanation. |
| P-009 | improved | The experiment and its result now have clear finite verbs and a more natural sequence. |
| P-010 | improved | The revision separates posterior recovery from the no-simulation-cost check and turns the aside into a direct statement. |
| P-011 | improved | The principal result now stands apart from the list of robustness conditions, making both easier to follow. |
| P-012 | improved | Removing the repeated cost-versus-fidelity framing leaves the technical statement intact and makes the paragraph less formulaic. |
| P-013 | improved | The deletion removes a nearby restatement while the preceding sentences still establish the absence of measurable mean bias for both estimators. |
| P-014 | changed-message | The edit removes the numeric token `6`, which the required token-preservation rule classifies as a message change even though the caveat is retained elsewhere. |
| P-015 | improved | Naming “the two orderings” resolves the former pronoun ambiguity while preserving the ordering discrepancy and its contribution to \(r\). |
| P-016 | improved | Splitting the drift and archive-turnover measurements gives each result a clearer sentence. |
| P-017 | improved | “Recovered it” identifies the regained contraction more naturally than the vague “transferred.” |
| P-018 | improved | “Settings” is more precise and appropriately neutral because \(k\) and \(S\) affect calibration as well as efficiency. |
| P-019 | improved | “Enters the statements as \(r\)” expresses the mathematical relationship more directly than “is carried.” |
| P-020 | regressed | Combining the crisp indexing statement with the explanation creates a long clause chain containing two successive causal links and is harder to read. |
| P-026 | improved | The three-sentence sequence clearly separates the condition, its mathematical consequence, and the empirical assessment. |
| P-027 | improved | Splitting the statement gives the residual-factor caveat and fixed-floor condition appropriate prominence without changing the logic. |
| P-028 | improved | The dependency from growing \(m\) to loss of the uniform bound and the resulting proof consequences is substantially easier to follow. |
| P-029 | improved | Separating the computable-reference setup from the two measurements makes the comparison clearer. |
| P-031 | improved | Giving each benchmark its own sentence exposes the numerical comparisons and leaves the interpretation intact. |
| P-033 | improved | Removing the announced “practical knee” lets the reported accuracy and cost tradeoff establish that interpretation directly. |
| P-034 | improved | The calculation now follows its actual order from generation time through the ceiling to the measured ratio. |
| P-035 | improved | The operational properties are easier to identify individually, and the causal connection to barrier-free execution remains clear. |
| P-036 | improved | “Given the same simulation budget” states explicitly what “fairly budgeted” meant and improves the paragraph’s rhythm. |
| P-037 | improved | Presenting the observations before their interpretation produces a clearer evidence-to-conclusion progression. |
| P-038 | improved | Reporting one archive size per sentence makes the non-monotone calibration values easier to inspect and clarifies what \(0.195\) denotes. |

## Items to revert

- P-014 — Revert the deletion because it removes the numeric token `6`.
- P-020 — Use: “Particles are indexed by \emph{proposal} time, not arrival time. This preserves the martingale structure under asynchronous execution: a worker forms its proposal from the history it has received at that moment, which is part of $\mathcal{F}_{i-1}$, so the conditional density $\tilde q_i$ of the emitted candidate is $\mathcal{F}_{i-1}$-measurable and”

## Overall

Overall, the round improves the manuscript’s voice and readability: most changes remove repetition or split dense result chains without weakening claims. The only worrying pattern is uneven handling of sentence boundaries—P-020 recombines a useful split into a clause stack—and P-014 also violates the explicit numeric-token constraint. I would not run another broad prose round; I would apply the two targeted fixes above and stop.