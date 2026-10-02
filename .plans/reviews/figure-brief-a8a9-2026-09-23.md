# Review brief: two algorithm-illustration drafts for the async-ABC paper

You are an independent reviewer. Two draft figures have been made for the paper
"generation-free, single-arrival-driven ABC" (repository root: the current
working directory). Review them for accuracy against the method, for how much
they help a reader understand the approach, and for concrete improvements.
Do not edit any file. Write a self-contained markdown report.

## The two figures

1. **A8, "ours": the single-arrival update as a ring of six parameter-space
   panels.** Image: `latex/figure-drafts/out/a8_algorithm_ring.png` (a 220-dpi
   copy may be given to you alongside). Sources: `latex/figure-drafts/tikz/a8_algorithm_ring.tex`
   (layout, labels, formulas) and `latex/figure-drafts/py/a8_algorithm_panels.py`
   plus `latex/figure-drafts/py/_toy2d.py` (the panels are drawn from a
   miniature run of the algorithm on a 2-D toy; read `_toy2d.run_async`).
2. **A9, textbook ABC-PMC drawn in the same six positions**, for comparison.
   Image: `latex/figure-drafts/out/a9_pmc_ring.png`. Sources:
   `latex/figure-drafts/tikz/a9_pmc_ring.tex`, `latex/figure-drafts/py/a9_pmc_panels.py`,
   `_toy2d.run_pmc`. A side-by-side wrapper exists in
   `latex/figure-drafts/out/a9_rings_side_by_side.png`.

`latex/figure-drafts/README.md` describes the intent of both (the section
"Algorithm illustrations"). The figures are meant to be as visual as possible
with one formula per step; they are candidates for the Method section.

## Ground truth for the method

* Paper source: `latex/sn-article-template/sn-article.tex`. The Method is
  §3 (lines ~73-127: history-based state, tolerance reconstruction, archive
  and smooth-kernel proposal, streaming AMIS weight, the "three weights"
  paragraph, and Algorithm 1). The reported estimator and its denominator
  are in §4 (lines ~128-180, eqs. draw-mixture / snapshot-denominator /
  posterior-estimator). Implementation details (schedulers, kernel-aware
  bisection, proposal covariance and jitter, truncation to the box, the
  bounded-memory estimator, the comparison table with classical ABC-SMC)
  are Appendix C (lines ~672-707). The baselines used in the experiments
  are described in §5.2 (line ~207).
* Implementation: the propagator is `ABCPMC.__call__` in
  `../propulate/propulate/propagators/abcpmc.py` (from line ~981; the
  reported estimator is `extract_posterior` in the same class). If that
  path is not readable from your sandbox, say so and rely on the paper.
* The conventional method in A9 is meant to be textbook ABC-PMC
  (Beaumont et al. 2009): quantile threshold, Sigma = 2 Cov, draw until N
  accepted, weight pi/q_{t-1} against the single previous proposal. Note
  that the paper's experimental baseline is pyABC with a matched
  smooth-kernel acceptor, which is not what A9 draws.

## Questions to answer

1. **Accuracy.** Go through every element of each figure (each panel, each
   label and formula, each arrow, the chord, the barrier, the center text)
   and check it against the paper and the code. List everything that is
   wrong, misleading, notation-inconsistent with the paper, or missing in a
   way that would misrepresent the method. Also say which intricacies of
   the real algorithm the figure legitimately omits and whether any of
   those omissions matter for a reader of §3. Also check whether the toy
   run in `_toy2d.py` actually implements what the panels claim to show.
2. **Understanding.** For a reader meeting the method for the first time in
   §3: does A8 help? Does A9 next to it help, and does the position-by-
   position correspondence work? What would such a reader most likely
   misread? Is anything redundant with the existing paper figures
   (`latex/figures/`, and the other drafts in `latex/figure-drafts/`)?
3. **Improvements.** Concrete, ranked changes: to content (what to add,
   drop, relabel), to the formulas (which ones belong in the figure, which
   belong in the caption or text, notation), and to layout (the ring
   form, label placement, print size: the Springer text width is 372 pt and
   each ring PDF is currently about 487 pt wide). Say which changes are
   essential before the figure could go into the paper and which are
   optional. If you think a different figure form would serve the
   comparison better, describe it.
4. **Verdict.** Should each figure go into the paper, and where (main text
   Method section, appendix, or not at all)? One short paragraph each.

Format: markdown with numbered findings; for each accuracy finding give the
element, what the figure shows, what the paper/code says, and a severity
(blocking / should fix / minor). Be specific about lines and labels. Do not
pad the report; a short report with precise findings is better than a long
one. State your token usage at the end if you know it.
