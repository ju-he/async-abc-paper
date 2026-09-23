# Theory diagnostics moved to the production CPM configuration (2026-09-23)

**Why.** Every Cellular Potts number in §4, Appendix A and Table B (ζ = 0.022, 1.1 % TV, ESS 0.213
vs 0.228, drift τ^-0.92, 832 archive changes in 33,748 draws, b ≈ 0.14–0.33) came from the retired
(division rate, motility) run `rerun_20260707`, disclosed as "earlier configuration". After the
inference target changed (`.plans/cpm_two_param_validation_2026-09-19.md`) those numbers no longer
described the runs behind the paper's posterior results, and two statements built on them were
unsupported for the production configuration: Limitations ("the configuration we run, whose target
is too weakly identified for its rate condition") and Table B ("violated as configured").

**Measurement.** `experiments/scripts/diag_theory_cpm_two_param.py` re-runs the two Appendix A
diagnostics (`diag_denominator_mismatch_cpm.py`, `diag_proposal_drift.py`) plus the factor-(d)
event counts on the five asynchronous replicates of `cpm_two_param_fixed` (job 14262214, tol_init
0.1, ~12.9k evaluations each, arrival order by `sim_end_time`, stamped tolerance from the
`proposal_tolerance` column). Output: `experiments/data/diagnostics/theory_cpm_two_param.json`.

| quantity | retired run (was in the paper) | production run, 5 replicates (now in the paper) |
|---|---|---|
| ζ total | 0.022 (quadrature only; floor unmeasurable, split erased) | 0.035–0.039 = floor 0.011 + quadrature 0.024–0.028 |
| r range | [0.92, 1.80] | [0.25, 1.41] |
| exact TV shipped vs reference | 1.1 % | 0.2–0.5 % |
| mean shift | ≤ 1.1 % sd | ≤ 0.012 sd |
| marginal width ratio | not reported | within 1 %, narrower in 9 of 10 (direction (a) predicts) |
| tilt bound vs measured | 2.3 % vs 1.1 % | 3.6–4.0 % vs 0.2–0.5 % |
| factor (d) events | 0 in 12,000 (1-D toy only) | + 0 in 63,699 CPM archive-phase draws |
| archive turnover / k ln n | 832 / 0.64–0.80 | 825–874 / 0.54–0.92 |
| per-step drift exponent | 0.92 | 0.69–0.95 |
| b, full range | 0.33 | 0.50–0.71 per replicate, 0.59 pooled |
| b, restricted (first half) | 0.14–0.23 | 0.25–0.72 per replicate, 0.48 pooled |

The prior floor now binds (bootstrap share 147/12.8k = 1.1 % < δ = 2.4 %), so factor (a) is live
and measured. The stabilization exponent sits at the b > 1/2 threshold: between the confounded
serial case (0.00–0.29) and the point-identified serial case (0.61–0.76).

**Paper edits (Springer source; TMLR twin regenerated, both compile clean, 51 / 39 pp, 0 overfull).**
§4 L180 (TV figure); App A L459 (CLT scope sentence, now also says what "damped/frozen" means);
L495 (Measured, in part: full production numbers, (d) count extended); L497–501 (stabilization
paragraphs rewritten: production exponents, "identifiability does" ordering); §8 L394 (Limitations
sentence); Table B rows 5(i) and 5(ii) (status "measured; at the threshold"); §5.1 L185 and App E
L756 (theory diagnostics attributed to the production configuration). No mention of the retired
run remains in the theory text; the four remaining "earlier configuration" mentions are the 50³
twin/scaling disclosures.

**Not committed** (user to review). The retired-run JSON
`experiments/data/diagnostics/r_denominator_mismatch_cpm_rep0.json` and the two older CPM scripts
are left in place; they are no longer cited by the paper.
