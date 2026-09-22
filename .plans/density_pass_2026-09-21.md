# Editorial density pass, applied 2026-09-21 (on 9837a81)

Source: `.plans/reviews/codex-gpt56sol-density-2026-09-21.md` and
`.plans/reviews/claude-fable-density-2026-09-21.md` (same brief: redundancy /
over-density / over-defensiveness; not a referee report).

## Result (texcount body words; captions separately)

| part | before | after |
|---|---|---|
| main text | 8,275 + 775 captions | 6,098 + 549 captions (after the second pass; 6,556 after the first) |
| appendices | 8,787 + 1,393 captions | 6,706 + 1,252 captions |
| main-text floats | 6 figures, 5 tables | 5 figures, 5 tables |
| pages | 58 | 50 (main text ends p. 20) |

Compile clean (no errors, no undefined refs, no overfull boxes).

## Applied

- Each caveat stated once, at the place that earns it: straggler-factor vs. throughput
  ratio only in §5.3; four-node placement only in §6.3 + Appendix D; schedule mechanism
  only in §3.2; pure-function/buffer pair only in §3.1; earlier-CPM-configuration
  disclosure introduced once in §5.1 and tagged "(earlier configuration)" afterwards.
- §8 Limitations rewritten as one paragraph; dropped the two non-limitations (a
  prediction being a prediction; "S=20 fixed" while Appendix F sweeps S) and the
  deadlock/retry note.
- Contributions and Conclusion without numbers (the abstract keeps them); Conclusion
  "measured fidelity ratio" -> "partly measured", "we think / we have tried" removed.
- Assumption 5(ii), the rate display and the CLT commentary moved from §4 to Appendix A
  (before Theorem 2); proof-mechanics sentence and the normalisation remark in
  Assumption 2 removed from §4.
- Table-reciting prose replaced by pattern statements: straggler paragraph, CPM twin
  paragraph, SBC paragraph, §6.2 CPM paragraph (the 7.1x and 17x "why we do not quote"
  passages cut).
- Practitioner rules live only in §7; the restatements at the ends of §6.3/§6.4 cut.
- Fig. 2 (straggler throughput) moved to Appendix F; captions of Figs. 1, 4, 5, 6, 7 and
  Tables 3, 5 trimmed to setup/encoding; Table 5 replicate ranges moved into the table.
- Referee-facing residue removed: "we no longer report", "earlier drafts", "we state it
  here rather than leave it to be discovered", "we avoid the word accepted", "honest
  but loose", "less contestable fix", the "rather than X" tails in §4, campaign
  reconciliation and script name in captions.
- Appendices: Appendix A "what is and is not new" -> "Relation to existing AMIS
  theory" (positive statement only); rate diagnostic condensed to the finding + the
  serial control + the ridge explanation; parameter-coupled runtime to two paragraphs;
  strong-scaling prose stops narrating Table 12; ablation + sensitivity subsections and
  their two figures replaced by a three-sentence note; LV extinction halved; crash
  recovery halved; effective-ranks paragraph to three sentences; srun flags, bug-log
  paragraph and the CPM setup duplicate in Appendix D removed; Declarations inventory
  cut.

## Deliberately not applied

- Codex's proposal to collapse the abstract's four-item results list and to replace
  the Discussion's four rules with one synthesis paragraph (Fable's counter-proposal
  taken: abstract keeps the numbers, Discussion owns the rules).
- Codex's proposal to move the scheduler cadence out of §3.2 (it is now the single
  statement of the mechanism).
- The Appendix F calibration-sweep prose (886-888) was only lightly trimmed.

## Must-stay caveats retained (each once)

r-tilted consistency with partial measurement (§4, Table 9); CLT not covering the run
configuration (§4, Appendix A, Table 9); filtration assumption (Assumption 4,
Appendix B); two CPM configurations (§5.1, Appendix E); 1.2x/2.0x attribution (§6.1);
unattributed twin duration inflation (§6.1, §8); placement-specific crossover (§6.3,
§8); CPM recovery is containment on one dataset (§6.4); ESS ceiling and bandwidth
initialisation as findings (§6.4, §7); Gaussian negative result 0.78x / 4x looser (§6.2,
§6.3); baseline handicaps 100-on-47 and uniform weighting (§5.2, §8); LV 98% extinct
(§5.1, Appendix F); the predictor's two cautions (§7).

## Second pass (same evening): "can we drop more caveats?"

Applied, main text: §8 to two sentences (theory scope; completion-time selection),
the four restated caveats cut (baseline population/weighting, twin inflation +
earlier configuration, CPM single dataset, placement + reported bandwidth); the CLT
sentence, the bandwidth-floor sentence and the factor-(d) parenthetical cut from §4;
"rather than beside it as a list of caveats" and "genuine assumption" cut; §5.1
calibration-instrument paragraph to one sentence; §3.1 running-sampler caveat to one
clause; §3.3 archive-membership clause cut; the heterogeneity miss parenthetical, the
80^3 equal-wall-clock explanation, the slope extrapolation, the hetero-conversion
pointer, the rejection "story of dimension and cost" remark, the -0.80 correlation and
the 80^3 "no useful posterior" sentence cut; abstract "we decompose and partly measure"
dropped (contributions keep it). Appendices: "partial accounting, not a bound" sentence
in Appendix A shortened; Appendix B ordering-mismatch paragraph halved; Appendix F
"two qualifications" paragraph to two sentences.

Kept (each once): predictor out-of-domain points and misses; 1.2x/2.0x attribution and
unattributed twin inflation (§6.1); Gaussian 0.78x / 4x looser and the g-and-k W1
reversal; placement dependence (§6.3); conservative g-and-k coverage and rank
histograms; containment-not-coverage; ESS ceiling; twin stall; baseline handicaps
(§5.2/§5.3); earlier CPM configuration (§5.1); r-tilted consistency with two factors
measured and two bounded (§4); CLT scope (§8); completion-time selection (§8); the two
predictor cautions (§7).

Status: edits uncommitted at the time of writing.
