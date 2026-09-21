# Shortening the main text — proposal (2026-09-21)

State at f75a8d9+1: main text 10.5k words (excluding declarations), 8 figures, 8 tables, one
algorithm box; captions alone 1,650 words. Sections: §1 820 · §2 550 · §3 875 · §4 1,086 ·
§5 428 · §6 1,400 · §7 4,030 (7.1 1,596 / 7.2 629 / 7.3 557 / 7.4 1,248) · §8 390 · §9 585 ·
§10 279. Both referees ask for a substantially shorter main paper with a separate supplement.
Target: ~7k words, 6 figures, 5 tables (~28 pp main); ~6k with Tier C.

## Tier A — remove duplication (≈ −2,400 words, no content lost)

| # | cut | words |
|---|---|---|
| A1 | Delete §5 Implementation. Its content exists elsewhere: retroactive O(nk) pass → two sentences in §3.4; matched acceptor → §6.3; twin and rejection ABC → §6.3 (already there). | −380 |
| A2 | Delete §6.1 "What is tested" (restates §1's bullets and §7's subsection titles; §7's opening sentence names the claims). | −120 |
| A3 | §1's "two limits" paragraph duplicates §6.2's calibration-instrument paragraph; keep §6.2's. | −110 |
| A4 | Delete §6.5 "Studies"; each §7.x states its own setup. | −180 |
| A5 | §1 contribution bullets to one line each (numbers stay in §7); drop the roadmap paragraph. | −270 |
| A6 | §2.1–2.3 (textbook) into one 90-word paragraph. | −115 |
| A7 | §7.1: "Two details of the prediction generalise" is Discussion rule 2 → keep once; the 50³ I/O-attribution discussion to one sentence; the estimator paragraph moves to §7.4 (it is the bandwidth-schedule evidence) and is halved. | −390 |
| A8 | §7.3: the LV scaling paragraph duplicates Appendix F.3 → three sentences. | −150 |
| A9 | §7.4: archive-size paragraph → two sentences + Appendix F; 80³ reminder → one clause; k/S sweep sentence → pointer. | −310 |
| A10 | §9: nine bullets → five (merge the two CPM-configuration bullets; merge placement + baseline configuration + baseline weights; drop "the two settings", now in §7.4). | −250 |
| A11 | §10 Conclusion 279 → ~120. | −160 |
| A12 | Captions: fig:scaling 171 → 80, tab:twin 144 → 70, tab:twin-hetero 146 → 60, fig:predictor 134 → 80 (mechanics stay in the text). | −400 |

## Tier B — floats to the supplement (≈ −350 words, 8+8 → 6+5 floats)

- Table 1 (method comparison) with §3.6 → supplement or delete.
- Table 4 (heterogeneity twin) → supplement; the predictor figure carries its points, two sentences keep the misses and the re-report result.
- Table 5 (CPM twin factorisation) → supplement; one sentence keeps 1.2× / 1.5–2.3×.
- Fig. 3 (heterogeneity quality, now archive means only) with the "synthetic instance" paragraph → supplement.
- Fig. 6 (ε curves, four panels) → supplement; Table 6 carries the ratios, the slopes stay as numbers in the text. (Alternative: keep Fig. 6 and fold Fig. 4 into it as a fifth panel.)

Result after A+B: ≈ 7.7k words, 6 figures (predictor, straggler, crossover, scaling, recovery,
CPM corner), 5 tables (benchmarks, straggler twin, matched ε, SBC, CPM production).

## Tier C — restructure (≈ −1,050 words further, → ≈ 6.5k)

- §4 to ≈ 450 words: assumptions stated in words in one paragraph, the five formal assumption
  environments move to Appendix A, Theorem 1 + Corollary stay, one paragraph on r. (−600)
- Merge §8 and §9 into one "Discussion" with the four rules and a closing limitations paragraph. (−150)
- §7.1 to the predictor figure + Table 3: heterogeneity and CPM-twin paragraphs at ≈ 120 words each. (−300)

## Separate supplement document

Both referees ask for it and S&C takes electronic supplementary material. Build the appendices
as a second PDF from a `supplement.tex` that shares the preamble, with `xr` for cross-references
(`\externaldocument{sn-article}`); the main PDF then ends at the bibliography. Mechanical; do
it after A+B so that every "Appendix X" pointer is final.

## Recommendation

Do Tier A and Tier B now (mechanical, nothing is lost, ≈ 7.7k words / 6 + 5 floats), then read
the result before deciding on Tier C, which trades precision in §4 for length.

## Applied (2026-09-21): Tier A + Tier B

Main text 10.5k → 8.7k words (declarations excluded), captions 1,650 → 910, floats 8+8 → 6
figures + 5 tables, main part 25 pages (bibliography starts p. 25), 59 pages with the SI. §5
deleted (retro pass → §3.4, twin/rejection → §6.3), §6.1 and §6.5 deleted, §1 570 words,
§2 flat at 506, limitations five bullets, conclusion 212. Moved to the appendices: Table 1 +
§3.6 (→ C), Fig. 3 + "synthetic instance" (→ F.1), Tables 4 and 5 (→ new F.2), Fig. 6 (→ new
F.3). The estimator paragraph moved from §7.1 to §7.4 ("The schedule must move"). Not yet at
the 7.7k estimate because §2's look-ahead discussion (≈300) and §7.4 (1,240) stayed; Tier C
(§4 assumptions to Appendix A, merge §8–9, trim §7.1) would take it to ≈6.5k.
