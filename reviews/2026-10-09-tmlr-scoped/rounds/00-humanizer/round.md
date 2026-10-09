# Round 00: humanizer pass (scoped)

- baseline: a447bb9 (tree clean at start)
- file: latex/tmlr/tmlr-article.tex
- scope: text added or rewritten on 2026-10-09 only (complexity pass ebae01a + bandwidth-floor
  draft a447bb9; word diff 93ac399..a447bb9 in ../../scope_diff.txt). The rest of the manuscript
  went through reviews/2026-10-06-tmlr-prose and reviews/2026-10-07-tmlr-readability.
- pass: 0 edits (157,517 characters before and after); report in report_tmlr-article.md.
  Considered and kept as content-bearing: "exactly the path", "as they were executed",
  "at all", "Being covered does not make a budget asymptotic."
- A/B: skipped, nothing to compare.
- claim flag (not a style edit): the Discussion rule said the 1e-4 eps0 floor "on the benchmarks
  here, never binds", contradicting Section 4 and Table C.1 (it binds on Gaussian mean and
  Lotka-Volterra). Corrected to "never binds on the g-and-k and Cellular Potts runs" in a separate
  correctness commit before round 01.
