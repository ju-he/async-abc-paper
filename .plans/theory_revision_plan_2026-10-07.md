# Theory revision without a new computational campaign

Date: 2026-10-07
Source: an external plan ("no new computational campaign revision route"),
reviewed against `.plans/proof_review_2026-10-07.md` and the manuscript at
`b75ed74`. This file records the version actually executed.

## What was kept from the plan, with the corrections made

1. **Finite-run theorem as the headline.** Theorem 1 now states the plug-in
   identity: the weighted average equals the smooth-ABC integral at the
   realized bandwidth tilted by the finite-run ratio q̄*_n/q̄_n, plus o(1).
   Correction: the statement is made *almost surely on {ε_∞ > 0}*, because
   the proof localizes on {ε_∞ ≥ 1/j}; the plan's version omitted the
   bandwidth hypothesis. The tilted limit and the exact-fidelity case are
   corollaries. Remark 1 ("about the algorithm as implemented") attaches to
   the finite-run theorem.
2. **Assumption 4 replaced by an execution model (B1–B3) and a lemma.**
   B3 admits exogenous scheduling randomness (message delivery) independent
   of the sampler and simulator streams; the plan required a deterministic
   scheduler. Code basis: propagator RNG per rank (abcpmc.py:678,
   propulate_abc.py:634); per-call hash-derived simulator seed
   (propulate_abc.py:673). Factor (d), the Setting paragraph, Table 4 and the
   Limitations sentence on length bias change together.
3. **Deadline censoring proposition** in Appendix B, stated at a common
   (ε, q̄) for the completed and launched sets, as an O(W/n) statement.
4. **5(ii) subsumes the fidelity clause; redundant √n bandwidth clause
   dropped** after unifying the definition of the predictable bandwidth.
5. **Proposition 3 in ζ₀ form first**, corollary for the set form, exact TV
   identity, prior-floor derivation moved into a lemma.
6. **Bookkeeping:** λ coordinate (D+1), box wording, L^(2) at first use,
   Lemma 3 with predictable ε, Remark 1 typo, Step 2 parenthetical, Cesàro
   line in Step 3(b), Corollary 2 hypothesis bookkeeping.
7. **Floored versus production regime paragraph**, linked to the ESS
   ceiling; Table 4 row for ε_∞ > 0 updated.
8. **Abstract, contribution 4, Limitations:** one plain sentence each, no
   new vocabulary.

## What was dropped from the plan

- The "retrospective adaptive-importance-sampling law" vocabulary and the
  proposed abstract sentence (jargon in the most-read paragraph).
- Rewrites of the plan's sections 6, 8, 12, 14: the manuscript already made
  those statements (Table 4, Limitations, §5.4); only wording touched.

## Follow-ups not done here

- Thesis chapters 6–8 mirror the paper's theory and need the same
  restructuring (see memory `project_thesis_ch6_sync`).
- Overleaf sync of the TMLR project.

## Status

Executed 2026-10-07 on `latex/tmlr/tmlr-article.tex` (uncommitted at the
time of writing). Build clean (`latexmk -pdf`), 43 pages, main text ends on
p. 17 (was p. 16/17 boundary before). Main-text words 8.5k -> 9.1k; whole
file 20.7k -> 22.4k. New environments: Theorem 1 (finite-run tracking),
Corollaries 2-3, Lemma 7 (adapted filtration), Proposition 11 (deadline
censoring), Lemma 12 (prior-floor factor). Table 4 rows for the execution
model and for {ε_∞ > 0} rewritten; `min_tol` row added to the hyperparameter
table.

## Tightening pass (same day)

Cross-section repetition removed: Conclusion cut to four sentences;
Discussion rules 2-4 reduced to pointers (their numbers live in §5.3/§6.4);
Intro paragraph 3 reduced to the straggler fact; Method overview paragraph 2
deduplicated against the Introduction; theory commentary trimmed; Table 4
fidelity and stabilization rows made pointers to §A.1. Net -263 main-text
words (9.1k -> 8.9k) and -370 overall; 43 pages, main text still ends p. 17.
Not done: Metrics vs C1, which on reading define and report respectively
rather than duplicate; the two barrierized-twin appendix subsections.
