# Digest: two independent reviews of the A8 / A9 algorithm rings (2026-09-23)

Reviews: `codex-gpt56sol-figures-a8a9-2026-09-23.md` (GPT-5.6-Sol, xhigh) and
`claude-fable-figures-a8a9-2026-09-23.md` (Claude Fable 5.1, fresh subagent).
Same brief (`figure-brief-a8a9-2026-09-23.md`), same 220-dpi renders, both with
read access to the paper source and the propulate propagator. Neither reviewer
saw the other's report or the author-side audit that produced the current
drafts.

## Where they agree (act on all of these)

| # | Finding | Codex | Fable |
|---|---|---|---|
| 1 | A8 panel 6: `w*` labeled only "weight"; readers will take it for the reported posterior weight. Retitle "proposal-time weight, stored, never reported" and point to eq. posterior-estimator. | blocking | should fix |
| 2 | A8 panel 2: the dashed `ε_n` ellipse looks like a hard acceptance boundary, the same glyph A9 uses for its threshold, so the one distinction §3 insists on (smooth kernel vs hard threshold) reads as "same thing, different colour". Show the kernel as graded shading; keep a boundary only in A9. | should fix | blocking for the pair |
| 3 | A8: asynchrony is invisible. The closing arrow reads as one serial worker. Show arrivals from other workers into panel 1 and "θ* to a free worker; ρ* arrives later"; write the append index as τ*, not n. | should fix | should fix |
| 4 | A8: the bootstrap chord ends on panel 5, so the visible path still runs through the weight step; the code returns a weight-1 prior draw with no denominator. Route it to the dispatch arrow. | should fix | should fix |
| 5 | A9 panel 5: "all M draws, then simulate all M" is not textbook PMC (draw, simulate, accept interleaved until N accepted; M known only afterwards). The toy's own `run_pmc` does it right, the labels do not. | blocking | should fix |
| 6 | A9 barrier: the generation dependency is that P_{t+1} must be complete before q_{t+1}; put the marker on the closing arrow. "Wait for the slowest" overclaims for textbook PMC (and the paper's pyABC baseline discards latecomers, sn-article.tex:260, 787). | blocking | minor |
| 7 | A9 must be captioned as textbook hard-threshold PMC, not the matched-kernel pyABC baseline of §5.2. Both note Table method-comparison carries the same tension. | blocking if included | should fix |
| 8 | A9 notation: P_t should carry ρ_i; ω needs generation indices; ω_i ∝ π/q_t normalized. | should fix | minor |
| 9 | A9 panel 6: accepted points outside the dashed ellipse because the toy adds noise to ρ. Set `RHO_NOISE = 0` for A9 or label the ellipse as a noiseless level set. | minor | minor |
| 10 | Width: each ring is ~487 pt against 372 pt text width, so 8 pt labels print at ~6 pt; side by side, ~3.4 pt. Move most formulas out of the labels (numbered key beneath, or caption) and shrink the ring. | essential | essential |
| 11 | Legend: point size, gray shading, gray contours, dashed ellipse are unexplained. | strongly recommended | minor |
| 12 | Verdict A8: yes, main-text §3, after the fixes above. | yes | yes |
| 13 | Verdict A9: not as a standalone ring in the main text. | omit or appendix | fold into a strip or appendix |

## Where they differ

* **Toy fidelity.** Codex checked the toy numerically and lists four departures from the code (scheduler jumps to the 2k-th order statistic instead of the ESS-retention bisection with a factor-2 cap; weighted covariance without the bias correction, factor 1.13 in the drawn state; draws truncated to the box by rejection but the density untruncated; no τ field, bootstrap ε stamped +∞ instead of None). It wants the toy fixed or the README's "faithful miniature" softened to "illustrative". Fable lists the same departures and judges none of them visible in the panels. Both agree the panels are unaffected. Action: soften the wording; fixing the toy is optional.
* **A9's replacement.** Fable proposes a 2 × 6 strip (top row PMC, bottom row ours, columns = the six steps, one formula line under each panel, barrier as a vertical bar in the top row), reusing all twelve panel PDFs, fitting 372 pt at ≥ 7 pt. Codex proposes a five-panel flow for ours plus a two-row cadence timeline (every completion triggers a proposal vs fill N before the next proposal), which is close to GA1/GA3. Either replaces the second ring.
* **Fable only:** the quantile threshold is Del Moral 2012 / Lenormand 2013 / pyABC's default, not Beaumont 2009 (which prescribes a fixed ε sequence); relabel the attribution. Step badges 1-6 do not map onto Algorithm 1's line numbers; give the map in the caption or drop the badges. The panel-6 snapshots drawn are the whole proposal history, whereas the online buffer holds the S most recent.
* **Codex only:** "S ← q_n" reads as replacement; write "push q_n into S". Contrast of the black!60 notes will not survive reduction.

## Verified by the author side after reading

* propulate `weighted_covariance` (abcpmc.py:825) applies the factor w/(w² − Σw²): Codex's bias-correction finding is correct.
* The paper says the pyABC baseline "closes a generation on the first 100 finishers and discards the latecomers" (sn-article.tex:260) and "discards latecomers instead of waiting for them" (787): Codex's barrier finding is correct for the baseline, and the A9 label "wait for the slowest" describes the barrierized twin, not pyABC or textbook PMC.
* `Individual.tolerance` defaults to None (propulate population.py:42): bootstrap draws carry no stamp, as both reviewers say.

## Action list for the next revision

Essential: 1, 2, 3, 4, 5, 6, 7, 10 above. Should: 8, 9, 11, the attribution
fix, the "push" wording, the README wording on fidelity. Then decide between the
2 × 6 strip and a single A8 ring plus a cadence inset for the comparison.

## Implemented 2026-09-24

All essential and should-fix items above are in the drafts (`latex/figure-drafts`, README section "Algorithm illustrations"):
1, 2 (kernel-profile inset in position 2 of both figures: Gaussian K_ε over ρ for ours, the indicator for PMC; the dashed ellipse remains only in A9 as the hard acceptance region), 3 (workers node, arrivals into the history, "θ* to a free worker", append with τ* = n), 4 (bootstrap routed to the workers), 5 (A9 position 5 shows the propose-simulate-accept loop in progress, position 6 the completed generation), 6 (barrier on the generation transition, "q_{t+1} waits for the N-th acceptance"), 7 (caption stand-ins name textbook PMC and the matched-kernel pyABC baseline), 8 (P_t carries ρ_i; ω^(t), ω^(t+1); normalized), 9 (noiseless discrepancy for PMC), 10 (diagonal-panel labels above/below, one formula per step, the rest in a caption stand-in; rings 367 pt, strip 376 pt with border), 11 (glyph legend), attribution (Del Moral / Lenormand for the quantile schedule), "push q_n into S", README wording "illustrative miniature" with the departures listed. Fable's 2 × 6 strip built as `tikz/a10_algorithm_strip.tex` (labels staggered by column parity to avoid collisions). The side-by-side ring wrapper was dropped. Not done: fixing the toy's scheduler/covariance/truncation (optional per the digest; wording softened instead); Codex's five-panel flow plus cadence timeline (GA1/GA3 already carry the cadence).

## Round 2 (strip only), 2026-09-24

Codex review of `a10_algorithm_strip.tex`: `codex-gpt56sol-strip-a10-2026-09-24.md`. Round-1 findings 1-10 resolved, 11 partly (legend not scoped by panel). No blocking finding. Four should-fix items before §3: (i) the glyph legend states point size / gray shade / gray contours globally although each applies to specific panels only, and the dotted Σ ellipses are unkeyed; (ii) bootstrap draws carry no bandwidth stamp in the code (`tolerance is None`, abcpmc.py:1031-1035, 1303-1309) while §3.1 says every proposed individual stores its bandwidth and the strip's append tuple implies one; define τ* = n or append ε_{τ*}, and reconcile in §3; (iii) the two mixture formulas are written without the centered argument (q = Σ W K_Σ), replace by exact formulas or verbal labels with the exact form in the caption; (iv) the dashed bootstrap path reads as an enclosure around row (b) and dashing is already used for the hard region. Optional: "one candidate" for "one draw", "worker pool" for "W workers" (clash with W̃), drop toy-specific numbers and `amis_interval` from the figure, raise gray contrast. Not yet applied.

Round-2 items applied 2026-09-24: all four should-fix items and the optional ones ("one candidate", "worker pool", shorter row (a) title, gray contrast, snapshot rule as "periodically stored proposals", toy-specific numbers and rug ticks dropped at strip size). Paper: §3.2 now says bootstrap draws store no bandwidth and ε_hist is the minimum over the stamped particles (sn-article.tex:86); TMLR twin regenerated; Springer source compiles with no undefined references.

Layout pass 2026-09-24 (user-reported overlaps): the return rails' vertical legs ran through the column-1 and column-6 labels in both rows; those four labels are now no wider than their panel (evaluated forms of ω^(t+1) and w* moved to the caption stand-in), the legs sit 1.5 mm further out, the lower label band is 1.2 cm below the panels, and the column-2 inset label sits inside the inset on a white backing above the curve. Strip 370 pt.

Integrated 2026-10-01: the strip is `latex/sn-article-template/figures/fig_algorithm_strip.pdf` (shared with the TMLR twin through the `tmlr/figures` symlink), placed in §3.5 before Algorithm 1 as `fig:algorithm-strip`, referenced from the Update-loop paragraph; the in-figure caption stand-in became the LaTeX caption and the b6 lines "before simulating; stored, never reported" were dropped from the figure (the caption carries them).

Variants 2026-10-02: the strip's return for ours is one arrow (worker-pool box and triple rail dropped, no counterpart on the PMC side); eight layout variants from `tikz/a10_body.tikz` (horizontal/vertical spine x formulas/bullets x bands/none), wrappers `a10_{H,V}{F,B}{N,G}.tex`; the integrated figure is HFN and was re-copied into the manuscripts.

2026-10-02, later: dotted bootstrap arrow removed in all variants (text note instead; caption no longer says "dotted path"); vertical barrier moved to the bottom segment; four more variants with the step spine between the rows (`a10_M{F,B}{N,G}.tex`).

2026-10-02: middle-spine layout reordered (title, rail, text, panels, spine, title, panels, text, rail), title (a) lifted clear of the barrier bar; the integrated figure switched to the MBG variant (bullets, gray bands, spine between the rows) in both manuscripts.
