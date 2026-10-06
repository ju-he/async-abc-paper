# Round 02

- baseline for this round: 8ace65e
- evaluators: Claude subagent (evaluate_claude.md; AI-likeness 3, readability 7, 20 proposals;
  would stop after this round) and codex gpt-5.6-sol (evaluate_codex.md; AI-likeness 5,
  readability 7, 18 proposals; would run one more restrained round)
- scores vs round 1: Claude 4 -> 3 (readability 6 -> 7), codex 8 -> 5 (6 -> 7)
- merged: 38 items, 6 overlaps, all six on Appendix A sentences (the production-replicate
  measurement sentence, the TV-distance sentence, the per-step drift, the exponent fit, the
  (d) non-events, the pinned-bandwidth serial run)
- triage: batch, all items (user choice). 31 accepted, 7 rejected: P-025 (heading rename, same
  pattern as the reverted P-046) and the six codex sides of the overlaps (Claude version taken
  as the more conservative, per the round-1 verification)
- apply: 31 applied, 0 failed; build clean, 40 pages (41 before)
- stop rule: the Claude reviewer writes it would stop; this is the last evaluation round
- commit: 0b6d5df (round applied at 60cfcff; verification fixes in this commit)

## Verification

- Claude (verify_claude.md): 25 improved, 5 neutral, 0 regressed; P-014 changed-message by the token
  rule only (caveat confirmed at 3.5 and in the Table C.1 caption).
- codex (verify_codex.md): 29 improved, 1 regressed (P-020: the merged Appendix B sentence chains two
  causal links), P-014 changed-message by the token rule and an explicit revert request.
- Actions: P-020 reverted; P-014 reverted by hand (a deletion; the strict token rule applied this
  time because one verifier asked for the revert). Round outcome: 29 applied, 2 reverted, 7 rejected.
- Both verifiers would not run another round; codex notes the Appendix A splits leave runs of short
  declaratives next to the author's dense register, worth one read-through for rhythm.

## Stop

Run stopped after round 2: the Claude evaluator wrote it would stop, and both verifiers advise against
another round. Final state: 64 applied edits over two rounds plus the 38-edit humanizer pass; no claim,
number, citation, reference, label or hedge changed.
