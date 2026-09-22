# Editorial assessment: redundancy, density, defensiveness

Assessed [sn-article.tex](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/sn-article.tex) at `campaign-tooling`, commit `9837a81`.

## A. Overall verdict

The main body could reasonably lose another **1,300–1,600 words—about 15–18%—without sacrificing a result, qualification, or reproducibility detail**, bringing the body from approximately 8,650 to 7,100–7,350 words. The largest problem is **over-density**: results paragraphs transcribe tables, combine several instruments, and carry protocol qualifications at the point of interpretation. Redundancy is a close second, especially across the abstract, contribution list, Results, Discussion, and Conclusion. Over-defensiveness is less pervasive by word count but more conspicuous in tone: the theory discussion, Limitations, and several appendices retain the argumentative shape of referee responses. The appendices could separately lose roughly 2,000 words, chiefly from legacy diagnostics and long narrations of tables. Savings below are approximate and non-additive where findings overlap.

## B. Findings

### Redundancy

1. **Location:** L397–408, Discussion  
   **Quote:** “The results give a practitioner four rules…”  
   **Diagnosis:** All four rules have just been stated, with the same examples and often the same wording, in C1–C4. The section summarizes rather than discusses relationships among the results.  
   **Recommendation:** Replace the four-rule sequence with one synthesis paragraph and one future-work sentence. Suggested core: “Generation-free execution is useful once simulation cost amortizes per-arrival coordination; timing predicts the systems gain, while bandwidth initialization and archive size determine how much of that gain becomes posterior accuracy.”  
   **Estimated saving:** ~140 words.

2. **Location:** L291–314, Results C2  
   **Quote:** “Throughput is instrumental; what a practitioner buys is…”  
   **Diagnosis:** L291 restates the metric defined at L219 and narrates nearly every entry in Table 6; L311 then narrates the Cellular Potts row again.  
   **Recommendation:** Retain one interpretive paragraph. Suggested replacement: “On Cellular Potts, the asynchronous arm completed \(2.4\times\) as many simulations and was \(1.6\times\) more efficient per simulation, ending \(4.1\times\) tighter; approximately \(2.0\times\) of the systems gain is attributable to the barrier.” Move slope, control-bandwidth, rejection, and \(80^3\) details to Appendix F.  
   **Estimated saving:** ~150 words.

3. **Location:** L277–279, Results C1  
   **Quote:** “On Cellular Potts at \(50^3\) the twin ran…”  
   **Diagnosis:** These paragraphs repeat the same utilization, throughput, configuration, and attribution facts later used in C2, Figure 5, Limitations, and Appendix F. The key distinction is obscured by the repetition.  
   **Recommendation:** Merge into one paragraph. Suggested replacement: “On the earlier \(50^3\) configuration, the utilization ratio attributes about \(1.2\times\) to the barrier; on the \(80^3\) production configuration, the asynchronous-to-pyABC throughput ratio was \(2.1\times\), against a \(1.90\times\) timing prediction. Unattributed simulation-duration differences are excluded.”  
   **Estimated saving:** ~140 words.

4. **Location:** L54–60, Introduction  
   **Quote:** “Our contributions are…”  
   **Diagnosis:** The four items reproduce the abstract’s four-part result catalogue, including nearly all its numerical claims, and then provide a section-by-section roadmap.  
   **Recommendation:** Replace the enumeration with: “We introduce a single-arrival ABC sampler with history-reconstructed state and streaming AMIS weights, establish consistency up to a fidelity ratio, and evaluate its systems and statistical behavior on synthetic and Cellular Potts benchmarks. The experiments quantify barrier cost and its crossover, and identify bandwidth initialization and archive size as the main limits on the reported posterior.”  
   **Estimated saving:** ~105 words.

5. **Location:** L391–393, Results C4  
   **Quote:** “The first row of Table…” / “The barrierized twin shows the same transient…”  
   **Diagnosis:** Two consecutive paragraphs explain the same scheduler transient, once on Cellular Potts and once on the twin, with separate arithmetic reconstructions.  
   **Recommendation:** Keep the Cellular Potts example and reduce L393 to: “The twin and short heterogeneity runs show the same effect: when no rank makes enough calls for the schedule to move, order-statistic re-reporting recovers the posterior supported by the history.” Move all bandwidth and \(W_1\) values to Appendix F.  
   **Estimated saving:** ~110 words.

6. **Location:** L230, L238, L274, Results C1  
   **Quote:** “Figure 1 is the paper’s central measurement…”  
   **Diagnosis:** The overview, persistent-straggler paragraph, and heterogeneity paragraph repeatedly transcribe values already visible in Figure 1 and Tables 3–4.  
   **Recommendation:** Keep the domain, agreement range, and causal interpretation; omit the complete sequences of throughputs and ratios. Suggested summary: “From \(5\times\) straggler slowdown onward, prediction and measurement agree within \(1\%\); under lognormal heterogeneity, the largest discrepancies are \(16\%\) and \(19\%\), where short runs and censoring affect the tail.”  
   **Estimated saving:** ~100 words.

7. **Location:** L330–354, Results C4  
   **Quote:** “Table 7 gives simulation-based calibration…”  
   **Diagnosis:** The paragraph reproduces almost every coverage entry in the immediately following table. Its actual editorial work is to distinguish near-nominal, conservative, and under-covering cases.  
   **Recommendation:** Replace the numerical inventory with: “The estimator is near nominal on the one-dimensional targets and conservative for two g-and-k parameters; Table 7 gives the full coverage results. The small-\(k\), higher-dimensional failure in Appendix F motivates problem-specific calibration.”  
   **Estimated saving:** ~85 words.

8. **Location:** L42, Abstract  
   **Quote:** “We prove the reported estimator consistent…”  
   **Diagnosis:** The four numbered results carry more instrument-level detail than an abstract needs and are repeated almost verbatim at L57–59.  
   **Recommendation:** Collapse them to one results sentence: “Across synthetic and Cellular Potts workloads, timing-based predictions agreed with measured barrier costs to within \(20\%\); on the tissue benchmark, the method completed \(2.1\)–\(2.4\times\) as many simulations—about \(2.0\times\) attributable to the barrier—and ended \(4.1\times\) tighter, while effective sample size remained the main statistical limit.”  
   **Estimated saving:** ~70 words.

9. **Location:** L419–420, Conclusion  
   **Quote:** “A generation barrier costs the ratio…”  
   **Diagnosis:** The conclusion recycles the abstract, contribution list, and Discussion, including the three prediction accuracies and all three posterior limitations.  
   **Recommendation:** Replace with: “Generation-free ABC removes synchronization and makes the expected systems gain predictable from asynchronous timing. It improves wall-clock tolerance once simulations amortize per-arrival coordination; fidelity, bandwidth scheduling, and effective sample size delimit the posterior claim.”  
   **Estimated saving:** ~70 words.

10. **Location:** L285, L325, L361, Results figure captions  
    **Quote:** “Strong scaling at a fixed wall-clock budget…”  
    **Diagnosis:** These captions repeat paragraph-level interpretations—where the crossover lies, why scaling fails, and why ESS matters—rather than identifying panels and encodings.  
    **Recommendation:** Limit captions to setup, axes, summaries, and marker conventions. Leave causal interpretation in the text.  
    **Estimated saving:** ~60 words.

11. **Location:** L223 and L947–948, Metrics and Appendix F  
    **Quote:** “We no longer report the distance…”  
    **Diagnosis:** A discarded diagnostic is explained twice, despite contributing no result to the current paper. The second account is largely editorial history.  
    **Recommendation:** Cut the main-text sentence and the appendix subsection. If provenance is desired, retain one sentence in the repository documentation.  
    **Estimated saving:** ~60 words.

12. **Location:** L226, L230, L453, L773–774, L842–843  
    **Quote:** “The four subsections take the four claims in turn…”  
    **Diagnosis:** These sentences announce material that the headings or next sentence already supply.  
    **Recommendation:** Cut the quoted sentence, “The rest of this subsection walks through…,” the Appendix A inventory, the Appendix F inventory, and “Figure 9 shows the curves…”.  
    **Estimated saving:** ~45 words.

### Over-density / detail not fully necessary

1. **Location:** L957–973, Appendix F, Ablation and Sensitivity  
   **Quote:** “This study and the sensitivity grid that follows…”  
   **Diagnosis:** Both studies use a point-mass concentration measure the manuscript itself says cannot support posterior-quality claims. Several paragraphs are then spent delimiting what the studies do not establish.  
   **Recommendation:** Cut both subsections and figures from the manuscript; leave the artifacts in the repository. If retention is required, replace both with: “Legacy one-dimensional concentration checks found no gross implementation failure but do not assess calibration or posterior quality.”  
   **Estimated saving:** ~500–550 words.

2. **Location:** L850–856, Appendix F, Parameter-coupled runtime  
   **Quote:** “The heterogeneity study above slows simulations independently…”  
   **Diagnosis:** One paragraph contains the intervention, two regimes, throughput, weighted and unweighted means, ESS, maximum weights, interval width, draining, interpretation, and future work. Most numerical sub-results do not alter the conclusion.  
   **Recommendation:** Reduce to three short paragraphs: design; posterior-mean result; drain control. Retain only “no trend in mean error,” “drain shift at most 0.013,” and the conclusion that the experiment does not demonstrate correction.  
   **Estimated saving:** ~300 words.

3. **Location:** L859–880, Appendix F, Strong scaling  
   **Quote:** “Lotka–Volterra is deliberately an adversarial simulator…”  
   **Diagnosis:** The prose narrates almost every throughput in Table 12, repeats the main-text crossover, and repeats the utilization figure’s message.  
   **Recommendation:** Keep one mechanism paragraph and one archive-size sentence; let the table carry the throughput sequence. Suggested opening: “On the 4-ms Lotka–Volterra workload, all-to-all arrival exchange dominates beyond one node, causing asynchronous throughput to peak at 48 workers and then decline.”  
   **Estimated saving:** ~250 words.

4. **Location:** L883–938, Appendix F, calibration sweep  
   **Quote:** “The calibration tests of Section 7.4 fix…”  
   **Diagnosis:** The text explains the grid, interprets every dominated \(k\), anticipates two objections, and then provides two tables containing the same comparison.  
   **Recommendation:** Retain the grid definition and one conclusion: “\(k=100\) is best calibrated and within \(7\%\) of peak throughput; \(k=50\) fails at \(d=8\), and larger archives are slower without improving worst-case calibration.”  
   **Estimated saving:** ~200 words.

5. **Location:** L132–172, Theoretical Analysis  
   **Quote:** “The analysis is organized around a single question…”  
   **Diagnosis:** The main text mixes the conceptual draw-mixture/denominator distinction with capped rejection, underflow redraws, asynchronous filtrations, exact snapshot counts, prior-floor mechanics, and a long interpretation of every assumption.  
   **Recommendation:** Keep the estimator, fidelity condition, assumptions needed by the theorem, and exact-fidelity corollary. Move the implementation departures and detailed interpretation of random versus deterministic limits to Appendices A–B.  
   **Estimated saving:** ~140 words.

6. **Location:** L218–223, Experimental Design, Metrics  
   **Quote:** “The quantity that compares samplers on one footing…”  
   **Diagnosis:** The three metric definitions are useful, but they carry record-order rules, checkpoint replay, fallback weighting, rejection-history handling, natural-scale conventions, and a discarded diagnostic.  
   **Recommendation:** Keep one paragraph each defining \(\epsilon_{(k)}\), predicted barrier cost, and posterior scoring. Move all record and checkpoint mechanics to Appendix D.  
   **Estimated saving:** ~140 words.

7. **Location:** L727, Appendix D, Crash recovery  
   **Quote:** “We test the crash-recoverability claim…”  
   **Diagnosis:** The result is buried under seed behavior, MPI ordering, buffer rebuilding, checkpoint corruption, and backup selection.  
   **Recommendation:** Reduce to: “A kill-and-resume test recovered the full simulation budget without duplicate records and produced a valid posterior from the recovered history. The trajectory was not bit-identical because random state, arrival order, and the online snapshot buffer are not fully checkpointed.”  
   **Estimated saving:** ~130 words.

8. **Location:** L192–195, Experimental Design, Benchmarks  
   **Quote:** “Table 2 lists the five configurations…”  
   **Diagnosis:** The prose repeats the table’s roles and costs, then supplies Cellular Potts parameter ranges, summaries, replicate structure, runtime correlation, and production-context justification already covered in Appendix E.  
   **Recommendation:** Retain one sentence per benchmark role and move all Cellular Potts construction details to Appendix E.  
   **Estimated saving:** ~125 words.

9. **Location:** L215–216, Experimental Design, Baselines  
   **Quote:** “Three comparators appear, each for a different question…”  
   **Diagnosis:** A single paragraph defines three comparators while also explaining population arithmetic, seeds, stopping rules, rank allocation, rejection thresholds, and interpretive fairness.  
   **Recommendation:** Use three short comparator definitions; move call-count, seed, and stopping mechanics to Appendix D.  
   **Estimated saving:** ~110 words.

10. **Location:** L98–108, Method, Streaming AMIS weight  
    **Quote:** “A particle drawn from \(q_n\) receives…”  
    **Diagnosis:** The equation is central, but default \(S\), online-buffer cost, snapshot reconstruction, chunked reporting cost, defensive mixture, and three weight systems all arrive together.  
    **Recommendation:** Keep the equation and a three-sentence distinction among adaptation, proposal-time, and reported weights. Move defaults, snapshot selection, cost, and memory behavior to Appendix C.  
    **Estimated saving:** ~110 words.

11. **Location:** L438–445, Declarations  
    **Quote:** “The repository accompanying this paper already contains…”  
    **Diagnosis:** The data and code statements inventory file counts, script glob patterns, refresh behavior, vendoring, scratch paths, and individual tests. That belongs in repository documentation.  
    **Recommendation:** State what is public, what is not yet public, and when raw records will be archived; omit counts and implementation-specific test descriptions.  
    **Estimated saving:** ~110 words.

12. **Location:** L719, Appendix C, bounded-memory estimator  
    **Quote:** “The retroactive AMIS estimator evaluates the snapshot denominator…”  
    **Diagnosis:** Complexity, exact snapshot indices, chunk size, bit identity, benchmark slope, memory growth, extrapolated runtimes, and timing exclusion are crowded into one paragraph.  
    **Recommendation:** Retain complexity, fixed-memory chunking, and the representative \(10^6\)-particle runtime. Move exact indices and residual/memory diagnostics to repository documentation.  
    **Estimated saving:** ~100 words.

13. **Location:** L975–976, Appendix F, Lotka–Volterra extinction  
    **Quote:** “One property of the Lotka–Volterra benchmark bounds…”  
    **Diagnosis:** The survival-conditioned target is explained correctly but at greater length than needed, including several sample sizes and percentages that do not affect its role as a systems benchmark.  
    **Recommendation:** Replace with: “Because about \(98\%\) of simulations become extinct, Lotka–Volterra is used only for systems measurements; its posterior curves are diagnostic rather than evidence of attainable inference accuracy.”  
    **Estimated saving:** ~100 words.

14. **Location:** L356, Results C4  
    **Quote:** “Where a reference exists, Figure 7 scores…”  
    **Diagnosis:** The paragraph combines both benchmarks’ complete \(W_1\) ranges, rejection results, ESS trajectories, sweep results, correlations, dimensional interpretation, and a degeneracy diagnosis.  
    **Recommendation:** Keep the ordering of methods and the ESS conclusion; move ranges and correlation to the caption or appendix. Suggested conclusion: “Accuracy follows effective sample size, which remains a few multiples of \(k\) and becomes unstable in four dimensions.”  
    **Estimated saving:** ~95 words.

15. **Location:** L75–83, Method, history-based state  
    **Quote:** “We design the algorithm so that the reported estimator…”  
    **Diagnosis:** The Propulate call signature and distinction between persistent archive state and assimilation callbacks are implementation mechanics, not prerequisites for understanding the estimator.  
    **Recommendation:** Move the `__call__(inds)` interface discussion to Appendix C. Retain: “The reported estimator is reconstructed from the evaluated log; a restart therefore reproduces the estimator for the recovered history, not the counterfactual trajectory of an uninterrupted run.”  
    **Estimated saving:** ~80 words.

16. **Location:** L764, Appendix D, computing environment  
    **Quote:** “All experiments ran on the JUWELS Cluster…”  
    **Diagnosis:** Hardware and package versions are appropriate; exact `srun` flags and repeated placement history are not needed in the paper.  
    **Recommendation:** Keep machine, MPI/software versions, core placement, and the exceptional four-node layout; omit command-line syntax and reservation mechanics.  
    **Estimated saving:** ~70 words.

17. **Location:** L395, Results C4  
    **Quote:** “The effective sample size is a few multiples…”  
    **Diagnosis:** Archive-size interpretation is mixed with a \(k=300\) schedule failure, a cheap-benchmark Pareto claim, retrospective readability, and weak \(80^3\) posteriors.  
    **Recommendation:** Keep the ESS–bandwidth trade-off in the main text and move the sweep and \(80^3\) parameter contractions to Appendix F.  
    **Estimated saving:** ~60 words.

18. **Location:** L50, Introduction  
    **Quote:** “Approximate Bayesian computation is the standard route…”  
    **Diagnosis:** The opening paragraph defines ABC-SMC, derives both special and general barrier ratios, motivates heterogeneity, and gives Cellular Potts coefficients of variation.  
    **Recommendation:** End after the general barrier-cost definition; move the Cellular Potts example to Experimental Design.  
    **Estimated saving:** ~50 words.

19. **Location:** L85–86, Method, tolerance reconstruction  
    **Quote:** “Each proposed individual stores the bandwidth active…”  
    **Diagnosis:** The monotonicity rule is central; per-rank search cadence, the halving cap, target count \(2k\), and transient diagnosis are implementation details repeated later.  
    **Recommendation:** Keep the running-minimum definition and state that the schedule tightens at a bounded rate. Move the cadence and target count to Appendix C.  
    **Estimated saving:** ~45 words.

### Over-defensiveness

1. **Location:** L510–516, Appendix A, stabilization conditions  
   **Quote:** “We therefore do not claim Theorem 2…”  
   **Diagnosis:** The necessary negative finding is followed by three paragraphs arguing that bandwidth, asynchrony, and the adaptation rule are not to blame. This reads as a rebuttal rather than a report.  
   **Recommendation:** Replace with a neutral evidence paragraph: “The configured Cellular Potts run does not satisfy the measured CLT rate. Identified serial targets meet the rate, whereas ridge-like targets do not, suggesting that weak identification is one mechanism for failure; the production configuration was not tested.”  
   **Estimated saving:** ~220 words.

2. **Location:** L414, Limitations, systems measurements  
   **Quote:** “The \(50^3\) twin and scaling campaigns…”  
   **Diagnosis:** One bullet lists two configurations, duration inflation, an unmeasured extrapolation, intermittent deadlocks, production costs, synthetic data, dimensionality, and Lotka–Volterra extinction. The accumulation makes the evidence appear less secure than the results warrant.  
   **Recommendation:** Retain only configuration identity, exclusion of unexplained duration inflation, and synthetic-calibration scope. Move deadlock/retry and Lotka–Volterra details to the relevant appendices.  
   **Estimated saving:** ~90 words.

3. **Location:** L318–320, Results C3  
   **Quote:** “The asynchronous method pays a roughly fixed cost…”  
   **Diagnosis:** The paragraph first presents the crossover, then pre-empts interpretations involving statistical efficiency, placement, run phase, and causal mechanism. These are legitimate scope conditions but overtake the finding.  
   **Recommendation:** State the result and one scope sentence: “Throughput crossed parity near 4 ms per simulation in the four-node placement used here; tighter placement moved the crossover downward, so this value is a configuration-specific guide rather than a universal threshold.”  
   **Estimated saving:** ~90 words.

4. **Location:** L490–508, Appendix A, decomposition of \(r\)  
   **Quote:** “Everything specific to the code enters through \(r\)…”  
   **Diagnosis:** The text repeatedly says the four factors are multiplicative, depend on insertion order, are only partly measured, are not a full attribution, and do not bound the total. One explicit warning is enough.  
   **Recommendation:** State once, immediately after the decomposition: “This factorization is diagnostic, not unique; only the floor and quadrature factors are measured, so the resulting comparison is not a bound on total fidelity error.” Delete later restatements.  
   **Estimated saving:** ~75 words.

5. **Location:** L412, Limitations, theory  
   **Quote:** “Consistency holds up to a fidelity ratio…”  
   **Diagnosis:** The bullet repeats the main theorem qualification, Appendix A’s measurements, the CLT scope, the causal interpretation of its failure, the filtration assumption, and the unset bandwidth floor.  
   **Recommendation:** Replace with: “Theory is conditional on denominator fidelity, an adapted asynchronous filtration, and a positive limiting bandwidth; only two fidelity components are measured. The CLT’s rate condition is not met by the reported configuration.”  
   **Estimated saving:** ~60 words.

6. **Location:** L530, Appendix B, proof setting  
   **Quote:** “One further mismatch deserves naming…”  
   **Diagnosis:** The arrival/proposal ordering issue is load-bearing, but the paragraph repeats it in several formulations and includes phrases such as “worth stating precisely” and “we do not prove it away.”  
   **Recommendation:** State the mismatch, its formal consequence, and the assumption once. Remove commentary about why it deserves attention.  
   **Estimated saving:** ~70 words.

7. **Location:** L415, Limitations, baselines and placement  
   **Quote:** “The production comparisons give pyABC a population…”  
   **Diagnosis:** The bullet accumulates population mismatch, dispatcher rank, absent weights, their small effect, four-node placement, and a packed-node counterexample.  
   **Recommendation:** Retain the two limitations that alter interpretation: the \(100/47\) population layout and placement-specific crossover. Move historical weight availability to Appendix D.  
   **Estimated saving:** ~45 words.

8. **Location:** L416, Limitations, reported bandwidth  
   **Quote:** “The reported posterior is only as good…”  
   **Diagnosis:** The scheduler limitation is important, but the bullet appends an unseparated archive-sweep confound and fixed \(S=20\), neither of which changes the headline limitation.  
   **Recommendation:** End after the re-reporting check. Move the sweep confound and fixed-buffer fact to Appendix F/C.  
   **Estimated saving:** ~30 words.

9. **Location:** L455, Appendix A  
   **Quote:** “What is and is not new here…”  
   **Diagnosis:** “We do not claim to weaken…” and the comparison with what other arguments “assume away” sound like responses to a priority objection.  
   **Recommendation:** Rename the paragraph “Relation to existing AMIS theory” and state the positive distinction directly: “The result treats denominator mismatch through \(r\) in the limiting target and uses a defensive floor to bound weights.”  
   **Estimated saving:** ~35 words.

10. **Location:** L472, Appendix A  
    **Quote:** “A caveat belongs with this statement rather than…”  
    **Diagnosis:** The caveat must remain, but its rhetorical introduction and the ensuing extended interpretation make the theorem appear to be withdrawn immediately after presentation.  
    **Recommendation:** Begin directly: “The reported experimental configuration does not satisfy the measured rate condition, so this CLT applies to damped- or frozen-adaptation variants.” Retain only the variance interpretation needed for the corollary.  
    **Estimated saving:** ~35 words.

11. **Location:** L589–590, Appendix B  
    **Quote:** “It is also the content that earlier drafts’ condition…”  
    **Diagnosis:** This is explicit revision history and reads exactly like a response to a referee. It contributes nothing to the proof.  
    **Recommendation:** Cut the sentence beginning “It is also the content…”.  
    **Estimated saving:** ~30 words.

12. **Location:** L723, Appendix D  
    **Quote:** “We report the allocated core count \(W\)…”  
    **Diagnosis:** “Flattens the asynchronous arm” and the argument that the difference cannot explain any effect are advocacy. Readers need the resource asymmetry, not its defense.  
    **Recommendation:** Replace with: “At \(W=48\), pyABC has 47 simulating ranks while the asynchronous arm has 48, a \(2.1\%\) difference in effective simulation resources.”  
    **Estimated saving:** ~30 words.

13. **Location:** L413, Limitations, completion-time selection  
    **Quote:** “Keeping every evaluated particle over-represents parameter regions…”  
    **Diagnosis:** The limitation and negative control are appropriate; “we do not claim the reweighting cancels it” is defensive phrasing.  
    **Recommendation:** Replace the ending with: “The tested couplings showed no detectable mean shift, but the method includes no formal completion-time correction.”  
    **Estimated saving:** ~15 words.

## C. Section-by-section target

Counts are approximate TeXcount-style counts, including headings and captions but excluding displayed equations. The manuscript’s reported 8.7k body count corresponds closely to the 8,645 words from Introduction through Conclusion below.

| Section | Current approx. | Recommended target | Main lever |
|---|---:|---:|---|
| Abstract | 240 | 170 | Replace four-item numerical catalogue with one results sentence |
| Introduction | 520 | 430 | Compress barrier derivation and contribution list |
| Background and Related Work | 225 | 200 | Remove repeated streaming-AMIS genealogy |
| Method | 880 | 760 | Move Propulate interface, scheduler cadence, and reporting-cost detail |
| Theoretical Analysis | 980 | 800 | Keep theorem conditions; move implementation interpretation to appendices |
| Experimental Design | 1,155 | 950 | Let tables/appendices carry benchmark and scoring mechanics |
| Results | 3,770 | 3,300 | Stop narrating tables; merge repeated scheduler and systems explanations |
| Discussion | 410 | 280 | Synthesize rather than restate the four result sections |
| Limitations | 485 | 300 | Consolidate into three load-bearing categories |
| Conclusion | 220 | 150 | State contribution, domain, and principal limit once |
| **Body total, excluding abstract** | **8,645** | **7,170** | **Reduction of about 1,475 words** |
| **Body plus abstract** | **8,885** | **7,340** | |

## D. Caveats that must stay

These are load-bearing for honest interpretation; the recommendation is to state each once clearly, not remove it.

- **The consistency target is \(r\)-tilted.** Exact smooth-ABC consistency requires \(r\equiv1\); only two of four fidelity components have been measured. Without this, “consistent” would overstate the theorem.

- **A positive limiting bandwidth is assumed, not established for the reported runs.** The runs leave `min_tol` unset. This is an actual theorem-to-experiment gap.

- **The CLT does not cover the reported configuration.** Its measured stabilization rate fails; the result applies to damped or frozen variants under the stated conditions. This belongs adjacent to the theorem and once in Limitations.

- **Asynchronous filtration and ordering remain assumptions.** Proposal order, arrival-order reconstruction, and completion-time selection are not proved equivalent, and the method has no formal completion-time correction.

- **The Cellular Potts configurations differ.** The \(50^3\) systems/theory diagnostics use the earlier division-rate/motility setup; posterior results and the \(80^3\) comparison use the production setup. Readers must know which evidence supports which claim.

- **Unexplained twin-duration inflation is excluded from the barrier claim.** Otherwise the reported barrier cost could be confused with the larger raw throughput gap.

- **The production pyABC comparison includes design-specific overhead.** Population 100 on 47 simulating ranks and per-generation overhead mean the full \(2.1\)–\(2.4\times\) throughput gain is not all attributable to the barrier; the approximately \(2.0\times\) attribution must remain explicit.

- **The crossover is placement- and budget-specific.** “A few milliseconds” is a measured guide, not a universal constant.

- **Cellular Potts recovery is from one synthetic dataset.** Truth containment is not coverage, and the \(80^3\) run supports systems conclusions rather than posterior-quality conclusions.

- **Bandwidth initialization and the ESS ceiling materially delimit the posterior result.** These are findings, not generic limitations: short per-rank histories can leave the schedule above what the draws support, and ESS remains a few multiples of \(k\).

## E. Top 10 edits by payoff

- [ ] **Cut or collapse the legacy ablation and sensitivity subsections** at L957–973; they cannot support the claims readers will infer from them.

- [ ] **Compress the parameter-coupled-runtime appendix** at L850–856 to design, principal result, and drain control.

- [ ] **Rewrite the strong-scaling appendix around its mechanism** at L859–880; stop narrating Table 12.

- [ ] **Condense the CLT-rate diagnostic** at L510–516 to the negative result, one identified-target control, and its scoped interpretation.

- [ ] **Replace the four-rule Discussion** at L397–408 with one synthesis paragraph and one future-work sentence.

- [ ] **Collapse C2** at L291–314 to the main Cellular Potts decomposition and move controls and slope details to Appendix F.

- [ ] **Merge the Cellular Potts C1 discussion** at L277–279 into one paragraph separating barrier cost, pyABC overhead, and excluded duration inflation.

- [ ] **Shorten Experimental Design** at L192–223 by moving benchmark construction, comparator mechanics, and scoring implementation to Appendices D–E.

- [ ] **Trim the main theory presentation** at L132–188 to the estimator, assumptions, theorem, and one scoped fidelity paragraph.

- [ ] **Remove repeated front/back matter summaries**: shorten the contribution list at L54–60, the abstract catalogue at L42, and the conclusion at L420.