# Academic-humanizer pass, applied 2026-09-22 (on 7705529, uncommitted)

Skill: `academic-humanizer`. Section by section over the whole paper (abstract through
Appendix F), then a mechanical audit, a compile, and an independent semantic audit of
the diff (subagent), whose findings were applied.

## Result

- Prose em-dashes 60 -> 0 (the five `---` left are empty table cells).
- Main-text sentences: mean 34 -> 24 words; sentences over 40 words 60 -> 13.
- Word count +20 (16,240 -> 16,260, texcount sum); still 50 pp; compile clean.
- Invariants checked by script (`audit_tokens.py` in the session scratchpad): every
  citation, \ref, \label, display equation, environment, bare number and numeric table
  row unchanged. Inline-math changes, all deliberate: `$r=0.99$` -> "correlation $0.99$"
  (clash with the fidelity ratio r); `$10^{-3}$` -> `$Z_j\approx10^{-3}$`; one duplicated
  "$r\equiv1$" caveat removed; `$k=1000$` split by a rephrase; `$\delta$` added as subject.

## Content corrections (claim vs. the paper's own evidence)

- §6.2: "on the two benchmarks where both exceed one they compound" was false (LV
  throughput is 0.96x); now "the two combine into the net effect ..., which exceeds one on
  two benchmarks".
- §6.3: "not because it samples worse (within 0.78x) but because ..." -> "0.78x as
  efficient; most of the gap comes from the baseline completing 3.5x more simulations".
- App A: CLT variance "always exceeds" the noise-free variance -> "is never smaller than"
  (Jensen gives >=).
- App A (d): the 10^-3 scale is now stated as Z_j, "below which" the residual
  (1-Z_j)^992 is appreciable.
- Status table, stabilization row: "not on the sampler" -> "as much as the sampler"
  (App A's own wording); "the Cellular Potts case by construction" -> "as they do in the
  earlier Cellular Potts configuration"; "now measured rather than left open" -> "measured".
- Hedges matched to evidence: "draws are the same" -> "comparably concentrated"; "the same
  posterior (93%/85%)" -> "a comparable posterior"; 4-D deficit "is" -> "is consistent with"
  weight degeneracy; "That is a reporting-support effect" -> "The evidence points to";
  Discussion "the whole run is spent" -> "can be spent", spread "tends to grow" -> "can
  grow"; Conclusion "the right model" -> "our results support", "On our benchmarks";
  abstract "verify its calibration" -> "check".
- App A: "and costs nothing" dropped (contradicted the O(n m_n k) cost that follows).

## Flagged, not changed (need author judgment or data)

1. C1-C4 are used in §5.2/5.3 but defined only by the §6 subsection headings.
2. §6.1: "From 5x up the three granularities agree to three significant figures" - Table 2
   at 5x has 31.6 vs 31.8, and only two barrier granularities are tabulated.
3. §6.4 "The schedule must move": "no rank made k calls ... both arms report near the prior"
   holds for the asynchronous arm; the twin made 440 calls/rank at sigma=0 and reports 0.19
   there (Table 11 caption says its schedule ran at sigma<=0.5).
4. §3.4: "S=20 captures most of the variance reduction of the full cumulative mixture" -
   the cited sweep (App F) measures calibration, not variance reduction.
5. Limitations "two factors measured and two bounded" vs App A (d): the asynchronous
   filtration part of (d) is "neither measured nor bounded structurally".
6. Abstract "(2.0x attributable to the barrier)" sits on "2.1-2.4x", while §6.1 says only
   1.2x of the 50^3 2.4x is the barrier; the 2.0x seems to be the 80^3 figure. Worth a look.
7. §6.2 "39x looser": 2.9e-3 / 7.6e-5 = 38.2 from the rounded values; check unrounded.
8. Spelling mixes British and American (utilisation x14 / utilization x3, normalized x8 /
   normalised x3, stabilisation / stabilization).

## Deliberately kept

Limitations paragraph (two sentences by design, no tells); "realistic cost and spread"
(echoes Table 1's role column); author's short punch lines ("That factor is not the
barrier's.", "The posterior does not follow.", "The target is."); "prove" for the theorems.

## Follow-up on the flagged list (user decisions, same day)

1. C1-C4 before definition: left as is for now.
2. Straggler "three significant figures": checked against `tab_twin/twin_straggler_raw.csv`
   and `fig_predictor/predictor_rows.csv` (medians over five replicates). The "three
   granularities" were the two barrier granularities plus the prediction: at 5x 31.594
   (every W), 31.759 (every 112), 31.754 (predicted) -> 0.52% spread; 10x 0.20%; 20x 0.06%.
   Text now: "agree with each other and with the prediction within 0.5% (0.2% at 10x,
   0.06% at 20x)".
3. §6.4 heterogeneity-campaign sentence dropped; the App F "synthetic instance" paragraph
   that deferred to §6.4 now points to Table tab:twin-hetero.
4. §3.4 S=20 now justified by calibration: "S=0 makes the reported estimator over-confident
   in one and two dimensions (mean coverage deviation -0.088), and every S>=5 removes that
   deviation" (Table ks-sweep, right block).
5. App A (d): the clause "unlike the other three contributions it is neither measured nor
   bounded structurally, and enters only through the assumption that this factor has a
   finite, positive limit" dropped. (The remaining sentence of (d) still repeats the
   filtration-gap remark made just above the factor list.)
6. Abstract 2.0x: left for now.
7. "39x looser": correct from unrounded medians (rejection eps_(100) 2.943e-3, async
   7.629e-5 -> 38.6x); "4.1x tighter" also checks (3.118e-4 / 7.629e-5 = 4.09).
8. American English throughout the .tex (29 changes: utilisation x14, normalised x3,
   stabilisation x2, normalising, summarised, serialises, realisation, factorised,
   factorises, synchronise, characterises, amortises, centred, centres, centre (verb),
   towards, Acknowledgements). "Julich Supercomputing Centre" kept (proper name). The
   figure PDFs were checked with pdftotext and already use American spelling.

Integrity after these: cites, labels, display math, bare numbers unchanged; refs -2 (the
dropped sentence) with App F's sec:results-posterior -> tab:twin-hetero; compile clean, 50 pp.
