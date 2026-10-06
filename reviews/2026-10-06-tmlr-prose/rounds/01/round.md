# Round 01

- baseline for this round: b4d3ab5 (after the stage-1 humanizer commit)
- evaluators: Claude subagent (evaluate_claude.md; AI-likeness 4, readability 6, 25 proposals) and
  codex gpt-5.6-sol (evaluate_codex.md; AI-likeness 8, readability 6, 25 proposals; first run lost
  to the read-only sandbox, rerun with the report returned as the final message)
- merged: 50 items, 8 overlaps; both reviewers independently targeted contribution 4, the Cellular
  Potts runtime sentence, "Three comparators appear", the bandwidth-transient "binds" sentence,
  "The lower panels show the reason", the Discussion's closing paragraph and "Three things are not
  routine"
- triage: batch, all items (user choice). 37 accepted, 13 rejected: five contradict earlier author
  decisions (the two kept short lines "Identifiability does." and "The posterior does not follow.",
  the two Discussion rule titles, "the method a practitioner would run"), six are the duplicate side
  of an overlap, two Claude items superseded by the codex rewrite of the same passage
- apply: 37 applied, 0 failed; build clean (0 LaTeX errors, 0 undefined), 41 pages
- diagnostics (not targets): "is what" 12 -> 1 after stage 1; "carries" 12 -> 7; counted
  signposts ("Three things/findings/comparators/weightings") 4 -> 0
- commit: 8ace65e (round applied at f0adb4e; verification fixes in this commit)

## Verification

- Claude (verify_claude.md): 24 improved, 10 neutral, 1 regressed (P-046 heading), 2 changed-message by
  the token rule only (P-001, P-004; caveats confirmed to survive elsewhere, keep).
- codex (verify_codex.md): 31 improved, 6 changed-message: P-001/P-004 (token rule only), P-016
  ("reliably" dropped), P-017 ("binding one" -> "harder condition"), P-020 ("lifts" -> "disappears"),
  P-037 ("binds" -> "remains consequential").
- Actions: P-046 reverted (regressed). The four split changed-message verdicts went to the user:
  P-016 kept; P-017 and P-037 amended to the codex verifier's wording ("the binding condition",
  "a binding limitation"); P-020 reverted. P-001/P-004 kept as pre-authorized duplicate deletions.
- Verifier note to carry forward: codex-sourced rewrites tended to flatten compact sentences
  ("measure performance", "remains consequential"); Claude-sourced items were conservative.
- Round outcome: 35 applied (two amended), 2 reverted, 13 rejected. Both verifiers would run one
  more round, scoped to clause-heavy Appendix A prose and the remaining vague verbs.
