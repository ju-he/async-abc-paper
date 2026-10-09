# Verification, round 01 (Claude)

Applied items checked: P-001, P-002, P-003, P-006, P-007, P-010. For each one I compared the Before and After in proposals.md with round01.diff and with the surrounding text of latex/tmlr/tmlr-article.tex. I found no added or removed digit, \cite/\citet, \ref/\eqref, \label or \SI/\num token in any applied item. The diff hunks contain only these six changes. The one removed number word is "four" in P-006, a meta count of proof steps, which I discuss below.

## Verdicts
| id | verdict | reason (one sentence) |
|---|---|---|
| P-001 | improved | It removes the "proposes with enter" garden path, keeps the reporting-bandwidth versus proposal-bandwidth distinction, and keeps $\epsilon_n$, \eqref{eq:posterior-estimator} and Theorem~\ref{thm:consistency}. |
| P-002 | improved | Moving the scheduler-type aside to the end gives "The search" its antecedent (the ESS search) again, and every clause survives, including the hard-kernel restriction and "no effect in any reported run" ("every choice" to "each" does not change the scope). |
| P-003 | improved | It removes the misreading in which "but not" negates "guarantees" and keeps the floor value, the split by benchmark and the continuation caveat with its \S\ref{sec:theory} pointer; the counterfactual "would have clipped" is more accurate than "binding", because no reported run had a floor set (minor: the consequence that the two cheap benchmarks are not covered as executed is now implied rather than stated here, but \S\ref{sec:theory} states it and the sentence points there). |
| P-006 | neutral | Dropping "The argument has four steps." removes a counted scaffold but also the only cue that this paragraph outlines the proof rather than starting it; the removed "four" is a structural count, not a result, so I do not treat it as a number change. |
| P-007 | improved | The deleted closer repeats the appendix's opening sentence ("decomposes $r$ and measures part of it") and the main-text paragraph on $r$, so no information is lost and the paragraph ends on its actual point. |
| P-010 | improved | Splitting at the semicolon keeps every number and reference and lets the throughput attribution finish before the per-simulation figure begins; "It" still resolves to the asynchronous sampler because that is the main-clause subject of the previous sentence and "read at pyABC's count" follows. |

## Items to revert
None. No item is regressed or changed-message.

Optional touch-ups, not reverts:
- P-003: to keep the coverage consequence explicit in the Limitations section, append to the clause: "it would have clipped the two near-instantaneous benchmarks, whose runs the limits therefore do not cover as executed, but none of the g-and-k and Cellular Potts runs." Only do this if the author wants the limitation stated on the spot rather than by reference.
- P-010: if "It" reads as ambiguous after "pyABC's per-generation overhead", write "Per simulation the asynchronous sampler is $1.6\times$ more efficient, read at ...".
- A note that applies before and after this round: "none of the g-and-k and Cellular Potts runs" (P-003) has the same scope as the old "the g-and-k and Cellular Potts runs". \S\ref{sec:theory} supports it only for a replayed sample of the calibration trials, and that support depends on the open TODO about sbc_final_bandwidth.json. The edit did not create this issue, but the TODO should be resolved before submission.

## Overall
The round made a modest but real improvement in readability: three edits (P-001, P-002, P-003) fix genuine parsing or antecedent problems, one of which could have been read as reversing the meaning, and two (P-007, P-010) remove a redundant closer or split an overloaded sentence. I saw no worrying pattern. The editor kept every caveat (the floor limitation, the continuation caveat and the hard-kernel restriction all survive), and the only deletions are a restated summary and a counted announcement. Because the open items from this round are all low-severity single-reviewer proposals, another full round would likely give diminishing returns; a short targeted pass over the rejected-but-plausible items (P-004, P-008, P-009) is the most I would run.
