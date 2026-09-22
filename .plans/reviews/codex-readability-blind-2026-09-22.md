# Blind readability judgment: Codex (gpt-5.6-sol, xhigh), 2026-09-22

Setup: `codex exec -s read-only --ephemeral -c model_reasoning_effort="xhigh"`, confined to a folder
holding only the two versions and the brief (brief reproduced in
`claude-readability-blind-2026-09-22.md`). Labels assigned by coin flip:
**A = 7705529 (before the humanizer pass), B = 8e419e8 (after)**. The judge was not told which
was which. Verdict below verbatim.

---

## 1. Overall verdict

**B; confidence 97/100.** Across nearly every section, B replaces clause-stacked sentences with units whose logical relations and referents are easier to recover on the first read, while retaining an appropriate technical register. It also improves precision through better hedging and clearer contrasts, although A contains a few stronger connective sentences and useful qualifications that B omits.

## 2. Per-section assessment

| Section | Verdict | Strength | Reason |
|---|---:|---:|---|
| Abstract | B | clear | Separates motivation, mechanism, and findings without losing quantitative detail. |
| 1 Introduction | B | strong | The problem setup and method overview no longer require parsing several claims inside one sentence. |
| 2 Background | B | strong | Cleaner progression from ABC kernels to staged methods, AMIS, and parallel SMC. |
| 3 Method | B | clear | Better sentence boundaries clarify state reconstruction, scheduling, and the three distinct weights. |
| 4 Theory | B | strong | Dense assumptions and proof motivation are unpacked, with several precision improvements. |
| 5 Experimental Design | B | strong | Benchmarks, comparator roles, and metrics are much easier to distinguish on a first read. |
| 6.1 Results | B | strong | Numerical comparisons and exceptions are presented in a more traceable sequence. |
| 6.2 Results | B | clear | Generally easier to parse, though A has one better contrast sentence. |
| 6.3 Results | B | strong | Separates systems mechanism, empirical boundary, placement dependence, and cheap-simulator consequences. |
| 6.4 Results | B | clear | More appropriately hedged and less clause-heavy, despite omitting one useful heterogeneity detail. |
| 7 Discussion | B | clear | The four practical rules are more direct and less rhetorically compressed. |
| 8 Limitations | tie | strong | The text is effectively identical. |
| 9 Conclusion | B | strong | Better pacing and a more defensible final claim. |
| Appendices | B | clear | Technical prose and the assumption-status table read more cleanly; one shared fragment and one lost qualification remain. |

## 3. Best passages for each side

### Where A reads better

1. **Results 6.2:** A preserves the contrast: “The curves add what a single budget cannot: on the two cheap benchmarks … whereas on Cellular Potts …”. B uses “The curves add what a single budget cannot. On … On …”, which is slightly choppier and weakens the explicit comparison.

2. **Results 6.4:** A includes “On the \(60\) s heterogeneity campaign … both arms report near the prior … and both recover on re-reporting.” B omits this useful link between the schedule argument and the appendix table.

3. **Appendix A:** A explicitly says the asynchronous part of factor (d) is “neither measured nor bounded structurally, and enters only through the assumption.” B ends with “which is a property of the schedule,” leaving its epistemic status less explicit.

4. **Discussion:** A’s “four rules, each backed by a measurement rather than an argument” is sharper than B’s “four rules, each backed by a measurement.”

### Where B reads better

1. **Abstract:** A embeds the definition of barrier cost inside “at a cost that … is the straggler factor”; B gives it three clean steps: the barrier idles workers, “For identically distributed simulation times its cost is the straggler factor,” then defines that factor.

2. **Background:** A compresses hard and smooth kernels plus staged proposals into one long sentence. B separates “The classical kernel is the hard indicator” from the smooth-kernel alternative and then introduces generation-staged methods.

3. **Theory opening:** A ends “the proofs in Appendix…,” missing a verb. B correctly writes “the proofs are in Appendix…,” while also separating the explanation of why existing AMIS results do not apply.

4. **Results 6.3:** A says the Gaussian deficit is “not because it samples worse—per simulation it is within \(0.78\times\),” an awkward contradiction. B says directly: “Per simulation it is \(0.78\times\) as efficient; most of the gap comes from … \(3.5\times\) more simulations.”

5. **Conclusion:** A declares generation-free execution “is the right model.” B appropriately narrows this to “our results support generation-free execution.”

## 4. Accuracy check

- A’s “the proofs in Appendix” is a grammatical error; B fixes it.
- Both versions contain the fragment **“The target is.”** in Appendix A. It should be deleted or completed.
- A says the CLT variance “always exceeds” the noise-free variance. Jensen gives a non-strict inequality; B’s “is never smaller than” is exact.
- A says calibration and recovery on both cheap benchmarks can be scored “exactly,” although g-and-k uses a constructed reference posterior. B’s “scored against a reference” is more precise.
- A states that the g-and-k result “is … not a weighting failure” and that the four-dimensional deficit “is … not bias.” B correctly weakens these to “evidence points to” and “is consistent with.”
- A says the two starting-bandwidth runs have “the same” draws despite different reported end tolerances. B’s “comparably concentrated” is safer.
- B should restore A’s qualification that the asynchronous/filtration part of factor (d) is not structurally bounded and enters through an assumption.
- B’s omission of the \(60\) s heterogeneity re-reporting result does not create a false claim, but it removes useful supporting evidence.

## 5. Recommendation

Send **version B** to the referee. Carry over A’s explicit cheap-versus-Cellular-Potts contrast in §6.2, the omitted \(60\) s heterogeneity sentence in §6.4, and the full epistemic qualification for factor (d); also repair the shared “The target is.” fragment.