# Round 00: humanizer pass (scoped)

- baseline: 6c1f36d (tree clean at start)
- file: latex/tmlr/tmlr-article.tex (main text + appendices, one file)
- scope: the manuscript went through the full humanizer + two evaluation rounds in
  reviews/2026-10-06-tmlr-prose (closed by the stop rule at 8967514). The theory restructure
  that followed (6c1f36d) rewrote Section 3, the Limitations and Conclusion paragraphs, and
  Appendices A and B without any prose review. This pass covers only that new text
  (git diff 8967514..6c1f36d), under the voice-only brief (templates/humanizer_brief.md).
- pass: 8 sentence-level exact-string edits, file 174,866 -> 174,770 characters; report in
  report_tmlr-article.md
- patterns cut: "rather than assumed" defensive tail in the contribution list and its
  restating closer in Table C.1 ("so adaptedness is derived, not assumed"); a "What X is Y"
  cleft (Section 3 closing paragraph); "itself" as emphasis (Appendix A (d)); vague "carries"
  (Appendix A (d)); emphatic "do steer" with three stacked "which" clauses (Appendix B,
  after Lemma B.1); intensifiers "exact" (Section 3) and "actual" (Appendix B, censoring)
- kept on purpose: every content-bearing contrast in the new text (proposal vs arrival time;
  "a diagnostic rather than an asymptotic statement"; "bounded, not measured"; "at the threshold
  rather than clearly inside it"; "the scale below which ... not a threshold above which");
  all hedges and limitations; all headings; the proof prose of Appendix B
- build: latexmk clean, 0 LaTeX errors, 0 undefined, 43 pages
- A/B: blinded copies in ab/claude (version_1 = edited, version_2 = baseline) and ab/codex
  (version_1 = baseline, version_2 = edited); orders set by hand, keys kept outside the repo

## A/B verdicts (blinded; edited version = version_1 for Claude, version_2 for codex)

| reviewer | prefers | margin | AI-likeness baseline / edited | content check |
|---|---|---|---|---|
| Claude subagent (ab_claude.md) | edited | small (6 of 7 sites, one tie) | 4 / 3 | no number, citation or result change |
| codex gpt-5.6-sol (ab_codex.md) | edited | small (6 of 7 sites) | 2.5 / 2.0 | flags "exact recipe" -> "recipe" as a slightly weaker qualification |

Decision: the edited file stays; H-02 ("exact") is reverted on codex's content-check note, since the
brief leaves qualifications alone. 7 edits remain. Codex's sandbox could not write its file;
codex_review.sh captured the report via -o.
