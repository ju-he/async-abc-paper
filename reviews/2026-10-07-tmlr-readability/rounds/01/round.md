# Round 01

- baseline for this round: d1ebcc0 (after the scoped humanizer pass)
- focus set at init: readability (the author's request); the Section 3 / Limitations /
  Conclusion / Appendix A-B text rewritten by the theory restructure (6c1f36d) had never been
  reviewed for prose
- evaluators: Claude subagent (evaluate_claude.md; AI-likeness 3, readability 6, 25 proposals,
  4 H; would not run another readability round) and codex gpt-5.6-sol (evaluate_codex.md;
  AI-likeness 3, readability 6, 24 proposals; would run one more focused round on Section 3 and
  Appendices A-B). The first codex attempt died with "Selected model is at capacity"; the retry
  completed.
- merged: 49 items, 10 overlaps (Assumption 5(ii), the main-result sentence, the ESS-ceiling
  sentence, the Limitations boundary-term sentence, contribution C4, the tracking-theorem gloss,
  the epsilon_infty definition, the (d) capped-rejection sentence, the Cellular Potts factor
  sentence, the growing-m weight bound)
- triage: item by item by the session on the author's behalf (the author was not present;
  the previous run's author decisions were fed to both reviewers and honored). 39 accepted,
  10 rejected, all ten the codex side of an overlap where the Claude item was the smaller edit.
  Two accepted items are notation/noun fixes the author should confirm: P-004 (w_n -> lambda_n
  in Appendix A(a); the floored prior weight is lambda_n in the snapshot-denominator display and
  w_n occurs nowhere else) and P-015 ("nine of ten" -> "nine of the ten marginals").
- apply: 39 applied, 0 failed; build clean (0 LaTeX errors, 0 undefined), 43 pages (43 before)
