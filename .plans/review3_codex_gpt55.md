**Summary**

The paper proposes a stateless, generation-free ABC method for Propulate in which proposals are updated after each completed simulation and posterior weights use an AMIS-style cumulative-mixture correction. The systems motivation is strong: removing generation barriers should help under stragglers and parameter-dependent simulator runtimes. The empirical evidence supports that qualitative claim, especially in the straggler and Cellular Potts throughput plots. However, the manuscript still overstates the statistical support for the implemented algorithm. The theory applies only to an idealized full-mixture estimator under assumptions the shipped top-`k`, sliding-buffer implementation does not satisfy; the calibration evidence is thin; and several systems explanations, especially the Lotka-Volterra inter-node “coordination wall,” are plausible but not directly measured.

**Concrete Concerns**

[MAJOR] Section 4, Eq. 6, Conditions 1/4, and “Theory versus implementation”: the theory does not cover the shipped algorithm. The theorem assumes a full cumulative AMIS denominator and a bounded proposal-density ratio. The implementation uses a top-`k` local Gaussian archive and an `S=20` sliding snapshot buffer at proposal time. The paper acknowledges this, but still repeatedly says the method “inherits AMIS’s consistency and CLT guarantees” in the abstract, Introduction contribution 4, and Conclusion. That wording is too strong. At most, the idealized estimator has inherited guarantees; the implemented method is empirically motivated.

[MAJOR] Sections 3.4, 4, 5, Appendix A: the role of the sliding buffer versus the full cumulative denominator is internally confusing. Eq. 5 defines the AMIS weight using a sliding buffer `S`; Eq. 6 defines the posterior estimator using the full cumulative mixture; Section 4 says the shipped algorithm only approximates the full mixture; Appendix A says the post-run estimator evaluates the full denominator over the full history. It is unclear which weights drive adaptation, which weights define the reported posterior, and which object the experiments actually evaluate. This ambiguity directly affects the claimed AMIS correction.

[MAJOR] Table 4 / Fig. 5 / Section 7.2: the SBC evidence is overstated. With 100 trials, synchronous baseline coverage at nominal 0.80 is 0.68 and at nominal 0.90 is 0.80, both noticeably low. The text says “both close to nominal” and differences are within MC error; that is not convincing. At 0.90, binomial SE is about 0.03, so 0.80 is not “close” in the usual sense. Also SBC is only shown for a one-parameter Gaussian model, not for the settings where the algorithm’s archive/proposal pathologies are most likely.

[MAJOR] Section 7.1 / Fig. 4 / Table 3: the Lotka-Volterra strong-scaling explanation is plausible but not proven. The claim that the decline after 48 workers is an inter-node coordination wall rests on throughput peaking at one 48-core node, plus a stated 2 ms simulator cost. There is no MPI profiling, communication timing, proposal-reconstruction timing, per-rank idle breakdown, or placement/control experiment. A hostile reader can say this is post hoc attribution.

[MAJOR] Section 7.3 / Fig. 8: Cellular Potts scaling is promising, but attribution to barrier removal is under-instrumented. The async method scales better than the synchronous baseline, but the paper does not show CPM worker idle fractions, simulator runtime distributions, generation-drain losses, or matched overhead decomposition. Without those, the conclusion “generation barrier is the binding cost” is plausible but not nailed down.

[MAJOR] Section 7.3 / Fig. 6 / Fig. 9: the Cellular Potts posterior-quality claim is a little too forgiving. Fig. 6 appears to show the async CPM Wasserstein distance persistently worse than the synchronous baseline, though the caption calls them “comparable.” Fig. 9 supports weak identification, but the statement that this is a structural property of the problem “not of asynchrony” needs more evidence: e.g. posterior/reference diagnostics, summary sensitivity, or an oracle/synthetic identifiability analysis.

[MODERATE] Limitations, item iv: the every-particle-bias experiment is not sufficiently documented. The paper says AMIS weights empirically compensate for parameter-coupled runtime bias, but gives no figure/table, no coupling function, no sweep values, no posterior diagnostics beyond a summarized Wasserstein statement, and no coverage check. This is too important to appear only as prose in Limitations.

[MODERATE] Section 5 / Appendix B: posterior-estimation cost is excluded from wall-clock budgets. That is defensible for measuring simulator throughput, but the method’s reported posterior may require expensive full-history retroactive denominator evaluation. The paper should report this cost separately, especially because the method advertises streaming AMIS and near-instantaneous simulators reach millions of particles.

[MODERATE] Section 3.3 / Appendix A: “accepted,” “evaluated,” and “archive” semantics are not consistently defined. The archive is described as top-`k` history by discrepancy, but Appendix A refers to “individuals already accepted at the current bandwidth.” With smooth probabilistic kernels, “accepted” is ambiguous. This matters because the archive drives proposal adaptation.

[MODERATE] Section 7.5 / Fig. 11: the sensitivity study is under-described. The caption says results are averaged over replicates and smoothing kernel, but the paper does not show variance, failure cases, or whether the same conclusions hold for non-Gaussian/non-sanity benchmarks. The “no delicate tuning” claim should be softened.

[MODERATE] Fig. 10: the ablation evidence is hard to interpret. Axis labels and method labels are tiny, the caption does not specify benchmark/settings, and the right panel does not clearly demonstrate the claimed degradation from removing AMIS. This figure currently does not carry the argumentative load assigned to it.

[MINOR] Abstract and Conclusion: “generation-free ABC is a promising approach for large-scale simulator-based inference” is fine; “inherits AMIS consistency and CLT guarantees” should be limited to the idealized estimator.

[MINOR] Condition 2: “normalized” is defined as `K_epsilon(0)=1`, which is peak normalization, not integral normalization. That is okay for a probabilistic acceptor, but the wording is nonstandard and should be clarified.

[MINOR] Data/code availability: “will be made available” is weak for an empirical HPC paper. Reproducibility would be much stronger with archived logs, raw timings, and scripts.

**Easy Attacks**

The easiest attack is: “The theorem is not for the algorithm you ran.” The second easiest is: “The systems explanations are inferred from throughput curves, not measured.” The third is: “Calibration is claimed from 100 one-dimensional SBC trials, while the actual challenging cases rely on Wasserstein curves and qualitative posterior plots.”

**Recommendation**

Major revision.

Top 3 fixes:

1. Rewrite the theory claims so the idealized full-mixture estimator and shipped implementation are cleanly separated; either prove something for the actual algorithm or remove guarantee language for it.

2. Add instrumentation: proposal/reconstruction time, simulator time, MPI/communication time, idle/drain time, generation counts, and posterior postprocessing cost for Lotka-Volterra and Cellular Potts.

3. Strengthen statistical validation: more SBC trials, clearer calibration intervals/tests, documented runtime-bias experiment, and more cautious Cellular Potts posterior-quality claims.
