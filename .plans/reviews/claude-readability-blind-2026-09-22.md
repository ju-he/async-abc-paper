# Blind readability judgment: Claude (fresh general-purpose subagent), 2026-09-22

Setup: fresh subagent, told to read only the judging folder (two versions + brief), no git history.
Labels assigned by coin flip: **A = 7705529 (before the humanizer pass), B = 8e419e8 (after)**.
The judge was not told which was which. Verdict below verbatim; the shared brief follows it.

---

## 1. Overall verdict
**B**, confidence 85.

B breaks up A's clause-stacked sentences, which often use em-dash insertions (A has about 52 in the main text, B 5, mostly table dashes), and it fixes several referents that are hard to parse in A. An expert can read B paragraph by paragraph without rereading. B is also more accurate: it corrects a claim that A's own Table 3 contradicts and one inequality whose direction A states backwards, and it hedges several overclaims. B's costs are small: a few passages split into slightly choppy runs, one caveat dropped from the appendix, and one new percentage that I could not check against the table.

## 2. Per section

| Section | Verdict | Strength | Reason |
|---|---|---|---|
| Abstract | B | clear | "(2.0× attributable to the barrier)" replaces "(2.0× of it the barrier's)". The 1.2–400× range moves up front instead of dangling at the end. |
| 1 Introduction | B | slight | Sentence splits read more easily. Contribution 4 names "the estimator" instead of an unclear "it". |
| 2 Background | B | slight | The kernel sentence is clearer. But splitting the AMIS sentence cuts "instead" off from what it contrasts with. |
| 3 Method | B | clear | "non-increasing" replaces "monotone-decrease". The tolerance paragraph is reordered so the target comes first. S=20 is now justified by the table. |
| 4 Theory | B | clear | "the sampler need not take any particular form" replaces the subjunctive "not that the sampler take…". A long em-dash insertion becomes a "because" clause. |
| 5 Experimental Design | B | clear | The Cellular Potts design sentence is untangled. "(r=0.99)" is renamed so it no longer clashes with the fidelity ratio r. "D_i" gets a proper "Let … be" definition. |
| 6.1 | B | slight | Fixes "on the ceiling its runtime spread sets" and the vague "where it matters". The out-of-domain list gets choppier. |
| 6.2 | B | clear | Removes a false "both exceed one". |
| 6.3 | B | clear | "0.78× as efficient; most of the gap comes from…" replaces the self-contradicting "not because it samples worse… within 0.78×". |
| 6.4 | B | clear | The ESS sentence parses. "Same" becomes "comparable". An overstated sentence about the heterogeneity campaign is gone. The archive-size paragraph is slightly choppier. |
| 7 Discussion | B | slight | The reporting-rule sentence is untangled. "can grow" / "can be spent" are appropriately hedged. |
| 8 Limitations | tie | — | Unchanged. |
| 9 Conclusion | B | slight | Hedged ("On our benchmarks", "our results support"), but a little staccato. |
| Appendices | B | clear | Fixes the direction of the 10⁻³ inequality. The status table no longer contradicts Appendix A. Spelling is consistent (A mixes "utilisation" and "utilization"). |

## 3. Best passages

**Where A reads better**
1. **Background.** A: "AMIS…, in the adaptive-importance-sampling line…, *instead* re-weights all past particles". B: "AMIS belongs to the … line… It instead re-weights". B's interposed classification sentence weakens the contrast.
2. **§6.1.** A: "drawn open: the cost-free simulator…, and 384 Cellular Potts ranks, where…". B: "drawn open. Two are the cost-free simulator… The third is…". A's colon list is more economical.
3. **§6.4, archive size.** A: "raising k is not free: re-reporting…, running with k=300…, and on the cheap benchmarks…". B gives four separate sentences, so the evidence is no longer visibly attached to the claim.
4. **§3.** A: "we use S=20, which captures most of the variance reduction… at a modest per-call cost" flows. B's "we use S=20, at a modest per-call cost." dangles until the next sentence (although B's content is better).
5. **Conclusion.** A's single enumerating sentence ("removes the barrier, reports an estimator that…, and converts…") reads better than B's four short sentences.

**Where B reads better**
1. **Abstract.** A: "Removing it converts into a tighter tolerance… ($2.0\times$ of it the barrier's)". B: "Removing the barrier yields a tighter tolerance… (2.0× attributable to the barrier)".
2. **§4.** A: "not that the sampler take any particular form, but that whatever it does be asymptotically reflected". B: "the sampler need not take any particular form, but whatever it does must be asymptotically reflected".
3. **§6.4, ESS.** A: "reporting at the tightest bandwidth reached, while that bandwidth is set by…, leaves an effective support of…". B: "The estimator reports at…, and that bandwidth is set by…, so the effective support stays at…".
4. **§6.1.** A: "What the barrier costs on this workload is 1.2×, on the ceiling its runtime spread sets". B: "On this workload the barrier costs 1.2×, the ceiling its runtime spread sets".
5. **§3, tolerance.** A: "tightens at a bounded rate --- each rank's… halve the bandwidth --- towards the bandwidth at which 2k…; on an expensive…". B: "tightens at a bounded rate toward the bandwidth at which 2k… lie within it: each rank's… On an expensive simulator…".

## 4. Accuracy check

**Where A is less precise or wrong**
- **§6.2:** "on the two benchmarks where both exceed one" is false for Lotka–Volterra, whose throughput ratio is 0.96× in Table 3.
- **§6.3:** "not because it samples worse — per simulation it is within 0.78×" contradicts §6.2's "worse per draw".
- **§6.4, overclaims:** "the sampler's draws are the same in both rows" and "reaches the same posterior (93%/85%)", against 91–92%/81%.
- **§6.4, heterogeneity campaign:** "no rank made k calls, so both arms report near the prior". The heterogeneity table shows the twin at up to 440 calls per rank, reporting W₁ 0.19 at σ=0. B deletes this sentence and repoints Appendix F to the table.
- **§6.1:** "the three granularities agree to three significant figures". The table has two barrier granularities, and at 5× shows 31.6 against 31.8.
- **§5:** "scored exactly" overstates the g-and-k reference, which comes from an asymptotic law. The Pearson "r=0.99" collides with the fidelity ratio r.
- **Appendix A(d):** "a threshold below which it is absent" gets the direction wrong, since (1−Z)^992 is large below 10⁻³. "Always exceeds" should be ≥, from Jensen.
- **Status table:** "the condition is one on the inference problem, not on the sampler" contradicts A's own Appendix A ("as much as on the sampler").
- **Unhedged:** "is intermittent weight degeneracy, not bias" and "is the right model".

**Where B is weaker**
- **§6.1, new figures:** "within 0.5% (0.2% at 10×, 0.06% at 20×)" cannot be checked from Table 2's rounded values. At 5× the table suggests about 0.6–1% (31.6 against 31.8; 103× against 102×).
- **Appendix A(d):** B drops "neither measured nor bounded structurally…" for the asynchronous filtration gap. As a result, "factor (d) [is] bounded rather than measured" now implicitly covers a component that is only assumed. A's clause also clashed with that sentence, so each version has a problem here.
- **§6.4:** the heterogeneity evidence is no longer in the main text (defensible, since A's version overstated it).
- **Claim strength:** the status table now says "as much as the sampler", which is weaker than A. It is consistent with Appendix A.
- **Grammar:** no errors found. The "its" in abstract item (iv) is slightly ambiguous in both versions.

## 5. Recommendation
Send **B**. Carry over or fix four things:
1. Restore a qualifier in "Measured, in part" or in (d): the asynchronous filtration-gap component enters only through Assumption 4.
2. Check the 0.5% figure at 5× against the source data, or write "within 1%".
3. Re-merge a few passages that B over-split: the AMIS "instead" contrast in the Background, the out-of-domain list in §6.1, the archive-size evidence in §6.4 (as a colon list), and the S=20 sentence.
4. Optionally, add back a corrected heterogeneity sentence, e.g. "the asynchronous arm made fewer than k calls per rank; both arms recover on re-reporting".

---

## Brief given to both judges

# Blind readability comparison

This folder holds two versions of the same LaTeX manuscript, `version_A.tex` and
`version_B.tex` (a statistics/HPC paper on asynchronous, generation-free approximate Bayesian
computation). One is a copy-edit of the other. The scientific content, numbers, equations and
citations are meant to be the same; the prose differs. You are not told which version came first,
and the labels A/B were assigned at random. Read ONLY the files in this folder; do not look for
other copies, version history, or notes elsewhere on the machine.

## Question
Which version is more readable for the intended readership: statisticians and computational
scientists refereeing for a journal such as *Statistics and Computing*?

"Readable" means:
- ease of a first read: can a careful expert follow each paragraph without rereading?
- sentence-level parsing: sentences parse on the first pass, referents are clear
- flow and cohesion: sentences connect; the prose is neither clause-stacked nor choppy
- precision: claims stay exact and appropriately hedged (readability must not cost accuracy)
- register: fitting a statistics journal

Do not reward a version merely for being shorter or longer, or for any single stylistic feature
(punctuation, sentence length) in itself; judge the reading experience. Short, choppy sentences can
read worse than one well-built longer sentence, and vice versa.

Read the main text (Introduction through Conclusion, including the abstract) in full and compare it
paragraph by paragraph. Then sample the appendices: at least Appendix A, the status table in
Appendix B, and Appendix F. The files are long; `diff version_A.tex version_B.tex` (or a word diff)
is a fine way to find what differs, but judge each difference in the context of its paragraph.

## Report (at most ~1200 words)
1. **Overall verdict**: A, B, or tie; confidence 0-100; three-sentence rationale.
2. **Per section**: a table with one row per main section (Abstract, 1 Introduction, 2 Background,
   3 Method, 4 Theory, 5 Experimental Design, 6.1-6.4 Results, 7 Discussion, 8 Limitations,
   9 Conclusion) plus one row for the appendices together: verdict (A / B / tie), strength
   (slight / clear / strong), one-line reason.
3. **Best passages for each side**: up to five passages where A reads better than B and up to five
   where B reads better than A. Quote the relevant words from both versions briefly.
4. **Accuracy check**: any place where one version is less precise, changes a claim's strength, or
   introduces an ambiguity or a grammatical error.
5. **Recommendation**: which version you would send to a referee, and what (if anything) you would
   carry over from the other one.
