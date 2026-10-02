# Review brief, round 2: the 2 x 6 strip (A10)

You are an independent reviewer. Repository root: the current working
directory. Do not edit any file. Write a self-contained markdown report as
your final message.

## The figure

`latex/figure-drafts/out/a10_algorithm_strip.png` (a 220-dpi copy is attached
as the image). Source: `latex/figure-drafts/tikz/a10_algorithm_strip.tex`.
Panels: `latex/figure-drafts/py/a8_algorithm_panels.py` (row b, ours, files
`out/a8s_panel_*.pdf`) and `latex/figure-drafts/py/a9_pmc_panels.py` (row a,
ABC-PMC, `out/a9s_panel_*.pdf`), both drawn from miniature runs in
`latex/figure-drafts/py/_toy2d.py`. Intent and caveats:
`latex/figure-drafts/README.md`, section "Algorithm illustrations".

The strip is the candidate Method-section figure comparing the paper's
generation-free, single-arrival-driven ABC update (row b) with textbook
ABC-PMC (row a), column by column. It is designed at the Springer text width
(372 pt; the PDF is 376 pt including a 4 pt standalone border) with 7 pt
labels. The text under the panels in the paper would move to the caption.

## Context

This is a revision. A first round reviewed two ring-shaped drafts of the same
content; the findings and what was implemented are in
`.plans/reviews/digest-figures-a8a9-2026-09-23.md` (read it; the two full
reports are `.plans/reviews/codex-gpt56sol-figures-a8a9-2026-09-23.md` and
`.plans/reviews/claude-fable-figures-a8a9-2026-09-23.md`). The strip is the
form both reviewers recommended for the comparison.

## Ground truth for the method

* Paper: `latex/sn-article-template/sn-article.tex`. Method §3 (lines
  ~73-127, incl. Algorithm 1 and the "three weights" paragraph); the
  reported estimator §4 (lines ~128-180); implementation details App. C
  (lines ~672-707); baselines §5.2 (line ~207).
* Code: `ABCPMC.__call__` in `../propulate/propulate/propagators/abcpmc.py`
  (from line ~981). Say so if unreadable.
* Row (a) is meant to be ABC-PMC with a quantile schedule (Del Moral et al.
  2012; Lenormand et al. 2013; pyABC's default) and a hard threshold
  (Beaumont et al. 2009), not the paper's matched-kernel pyABC baseline.

## Questions

1. **Accuracy.** Check every element of the strip against the paper and the
   code: the six column headers, both row titles, each panel and its formula
   line, the arrows, both return rails (the barrier bar and its label on row
   a; the workers box, the triple rail and its label on row b), the dashed
   bootstrap path and its label, the panel tags, the legend block. List what
   is wrong, misleading, or notation-inconsistent with the paper, with
   severity (blocking / should fix / minor).
2. **Round-1 findings.** For each numbered finding in the digest's agreement
   table (1-11), say whether the strip resolves it, and if not, what remains.
3. **Readability.** For a first-time reader of §3: does the column-aligned
   comparison work? What is the first thing they will misread? Are the
   staggered formula lines (even columns lower) acceptable, or do they hurt?
   Is anything too dense or too small at 372 pt? Which elements could be
   dropped without loss?
4. **Improvements.** Concrete, ranked; say which are essential before the
   figure can go into §3 and which are optional. If the caption should carry
   something the figure currently carries, say what.
5. **Verdict.** Ready for §3 after which changes? One short paragraph.

Format: markdown, numbered findings, each with the element, what the figure
shows, what the source says, and the severity. Be specific (line numbers in
the .tex, column and row). Short and precise beats long.
