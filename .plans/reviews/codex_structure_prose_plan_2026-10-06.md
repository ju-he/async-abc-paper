# Prioritized revision plan: structure and prose

## Overall diagnosis

The manuscript has a strong empirical story, but the reader encounters implementation state, estimator corrections, and proof qualifications before receiving a stable mental model of the method. Section 3 is therefore the main structural bottleneck. The prose compounds this by compressing motivation, mechanism, caveats, settings, and results into the same paragraphs.

The revision should establish this narrative order:

1. Why generation barriers matter.
2. What the generation-free algorithm does at a conceptual level.
3. How it constructs proposals and estimates the posterior.
4. What the theorem guarantees.
5. Why each benchmark tests a necessary part of the claim.
6. What the experiments establish.

Do not begin with implementation bookkeeping and then reconstruct the conceptual method from it.

---

## Priority 0: establish a terminology and abstraction contract

Before rewriting paragraphs, fix the vocabulary used throughout the manuscript.

- Use **bandwidth** for the smooth-kernel parameter \(\epsilon_n\).
- Reserve **empirical tolerance** for the order statistic \(\epsilon_{(k)}(n)\).
- Use **candidate** before simulation, **evaluation** for one parameter-to-discrepancy operation, and **particle** after an evaluation is recorded.
- On Cellular Potts, state explicitly that one evaluation contains four simulator replicates. Do not interchange “simulation” and “evaluation” there.
- Define **history** as the append-only record of completed evaluations.
- Define **archive** as the \(k\) lowest-discrepancy history entries.
- Distinguish the two proposal mixtures:
  - the **online snapshot mixture**, used to assign the proposal-time weight;
  - the **retrospective denominator**, reconstructed after the run to estimate the posterior.
- Keep \(S\) for the online buffer and \(m\) for the retrospective mixture. Even if the implementation links them, the mathematical exposition should not make them conceptually identical.
- Avoid the clash between the particle weight \(w_i\) at line 92 and the prior-mixture mass \(w_n\) at line 156. Rename the latter, for example \(\lambda_n\).
- Introduce **worker** in the main text; reserve MPI **rank**, propagator **call**, cache rebuilds, and option names such as `tol_init` for the implementation appendix.
- Define ESS once, including its formula or a precise reference, before using “effective sample,” “ESS retention,” or “effective support.”

This terminology sheet should govern the abstract, figures, captions, pseudocode, theorem, and appendices.

---

## Priority 1: rebuild Section 3 around a high-level explanation

### Placement of Figure 1

Put Figure 1 immediately after the Section 3 heading, before technical definitions. Ensure the float actually appears there. Follow it with a short “Method at a glance” subsection.

The first two paragraphs after the figure should explain, without implementation parameters:

1. A completed simulation is appended to the history.
2. The \(k\) best history entries form an archive.
3. A smooth, weighted mixture over that archive proposes the next candidate.
4. The candidate is immediately assigned to a free worker; no population boundary is required.
5. Sampling weights guide future proposals, while a separate retrospective reweighting of all evaluations produces the reported posterior.

This resolves the feedback at lines 11–14 of the review: the graphical abstract comes first, and the method is explained before state reconstruction, scheduler cadence, snapshots, or Cholesky factors.

### Target outline for Section 3

| Subsection | One-line purpose | Abstraction level | Move out of the main text |
|---|---|---|---|
| **3.1 Method at a glance** | Explain the arrival-driven loop and the separation between online adaptation and retrospective posterior estimation. | Conceptual; almost no notation. | All scheduler settings, buffer sizes, code names, MPI details, and calibration numbers. |
| **3.2 Evaluated history and archive** | Define \(\mathcal H_n\), what one record contains, and \(A_n\) as the \(k\) best completed evaluations; explain why reconstructing state from history permits asynchronous updates and replay. | Definitions and one state equation. | Propulate interface, crash/restart behavior, “two scheduler throttles,” checkpoint semantics, cache behavior. |
| **3.3 Bandwidth adaptation** | State why the bandwidth must decrease monotonically and what statistical target drives tightening. | Algorithmic principle; one compact equation for monotonicity. | `tol_init`, per-rank search cadence, “once per \(k\) calls,” halving caps, alternative schedulers, option names, and cache-triggered searches. |
| **3.4 Archive-based proposal** | Define the weighted mixture \(q_n\), the role of the smooth ABC kernel, and the perturbation covariance. | Core mathematical method. | Cholesky factoring, covariance jitter, truncation-normalizer approximation, underflow fallback, and `perturbation_scale` implementation details. |
| **3.5 Online and retrospective weighting** | Explain why changing proposals require mixture weighting; distinguish adaptation weights, proposal-time weights, and final posterior weights; define the estimator here rather than first defining it in Section 4. | Concept first, then the essential equations. | `amis_interval`, ring-buffer mechanics, \(S=20\), calibration sweep results, chunking, timing, and numerical safeguards. |
| **3.6 Asynchronous update loop** | Give pseudocode that assembles Sections 3.2–3.5 and makes the absence of a generation barrier explicit. | Operational summary, not a second implementation specification. | MPI communication pattern, serialization, restart behavior, and detailed complexity benchmarks. |

### Specific Section 3 changes

- Lines 84–92 currently begin with recovery semantics: “A crashed run is therefore recovered…” and “The running sampler does carry the online snapshot buffer… and two scheduler throttles.” These are not the method’s conceptual entry point. Retain the pure-history design in Section 3.2, but move crash recovery, the Propulate interface, and scheduler throttles to Appendix “Implementation Details” or “Additional Experimental Protocol.”
- Define the archive before saying “Each candidate proposed from the archive” at line 95.
- Move the posterior estimator now at lines 162–165 into Section 3.5. The theory should analyze an estimator already defined by the method section.
- Remove experimental conclusions from the method exposition. In particular, the \(S=0\) coverage result at line 119 belongs in Results or the sensitivity appendix.
- Reduce each technical subsection to the pattern: **purpose → definition/equation → one consequence**. Do not mix settings, empirical validation, runtime complexity, and implementation exceptions into the same paragraph.
- Resolve the apparent inconsistency at line 119: “\(S=0\)… the first and the last reconstructed proposal” appears incompatible with “\(m\le S+1\).” Verify the actual implementation and make the text, equations, and appendix use the same convention.

---

## Concepts introduced before they are motivated or defined

The following audit is ordered by first appearance. It covers nonstandard reader-facing concepts, implementation terms, and symbols; ordinary ABC notation defined in the same sentence is excluded.

| First appearance | Concept and current wording | Where it is eventually explained | Required revision |
|---|---|---|---|
| **52** | “a streaming form of adaptive multiple importance sampling” | Background line 77; operational details lines 107–121 | In the abstract say what it accomplishes: reweights evaluations against a mixture representing the proposals that generated them. Reserve “streaming AMIS” as the name after the plain-language explanation. |
| **52** | “the full evaluated history” | Formalized as \(\mathcal H_n\) at lines 87–92 | In the abstract call it “the append-only record of completed parameter–discrepancy pairs.” Define the richer metadata only in Section 3.2. |
| **52** | “consistent up to a fidelity ratio” | Motivated at lines 149–160 and defined by equation (9), lines 173–177 | Either remove this unexplained phrase from the abstract or add a plain gloss: the limit is exact when the reconstructed proposal mixture matches the true mixture of sampling densities. |
| **52** | “kernel-matched pyABC baseline” | Full configuration at line 224 and Appendix line 702 | Replace with “a generation-based pyABC baseline using the same ABC kernel and bandwidth schedule,” at first use. |
| **52** | “barrierized twin” | Defined only at line 224 | At first use say “the same sampler with a collective barrier inserted before each proposal.” |
| **52** | “Cellular Potts tissue simulator” | Technical setup at lines 201–203 and Appendix lines 751–754 | Introduce it first as a stochastic cell-based tissue model used to infer mechanistic parameters from morphological observations, with parameter-dependent runtime. |
| **52** | “starting bandwidth… prior-predictive discrepancy scale” | Scheduler at line 95; practical explanation at lines 388–390 | Define the starting bandwidth when bandwidth adaptation is first introduced. Move the practical selection rule from late Results into Experimental Design, leaving its empirical effect in Results. |
| **52** | “effective sample size, a few multiples of the archive size” | ESS is mentioned at lines 117 and 231, but its limiting mechanism is not explained until line 353 | Define weighted ESS in Metrics; explain in Section 3.5 why reporting at a tight bandwidth can concentrate weight on only a small part of the history. |
| **52**, then **59** | “archive size”; “a sliding archive of the \(k\) best particles” | Formal definition at line 98 | Replace “archive size” in the abstract with “the number \(k\) of lowest-discrepancy evaluations retained to construct proposals,” or avoid the term there. |
| **52/57** | “straggler factor” | It is actually glossed in line 52 and formally defined at line 57 as \(\mathbb E[\max T_i]/\mu\) | This is not a genuine missing definition. Keep one clear definition, preferably line 57, and shorten the abstract to avoid defining it twice. State its i.i.d.-runtime scope at the definition. |
| **59** | “barrier-free island model” | Never explained in the main body; Propulate details appear later | Either define it as workers proposing independently from their locally available histories or remove “island model” from the main narrative and leave it to Implementation Details. |
| **59** | “pure function of the evaluated history” | Expanded at lines 84–92 | Explain the benefit immediately: posterior reporting can be reconstructed from the log and does not depend on a synchronized population state. |
| **66** | “damped-adaptation variants” | Discussed only in Appendix lines 458 and 496–502 | Remove from the contribution list or add a short definition. It is not needed to state the main consistency contribution. |
| **81** | “filtration and fidelity assumptions” | Assumptions at lines 172–179 | Replace this forward theoretical jargon with plain language: arrival order affects the result only through the conditional sampling law and how accurately the reporting denominator represents it. |
| **87** | “online snapshot buffer” | A snapshot is defined at line 108 | Do not mention the buffer before defining a proposal snapshot. Move the buffer to Section 3.5. |
| **87** | “two scheduler throttles” | Not identified in the main body; implementation behavior appears around Appendix lines 692–698 | Remove from the main body. If retained in the appendix, name both throttles and their purpose. |
| **87** | “Propulate propagator interface” and “call \(n\)” | Operational context is scattered across lines 124, 224, and the appendices | Define one “update call” as one request for a new candidate, or replace it with “update.” Move the framework-specific interface to the appendix. |
| **92** | “stored core importance weight \(\pi/q_{\tau_i}\)” | \(q_n\) is defined at line 100 and proposal-time weighting at lines 110–117 | Do not put \(q_{\tau_i}\) into the history tuple before defining proposals. Section 3.2 can name the stored weight; Section 3.5 should define its formula and function. |
| **95** | “uniform bootstrap draws,” “stamped particles,” \(\epsilon_{\rm hist}\), \(\epsilon_{\rm sched}\), `tol_init`, “each rank’s scheduler,” and “calls” | Bootstrap appears in lines 124 and 136; scheduler mechanics are mainly Appendix lines 692–698 | Rewrite at one level: first explain why prior draws initialize an empty archive, then state the monotone bandwidth rule. Move stamp terminology, option names, per-rank cadence, and search caps to the appendix. |
| **105** | “prior-vs-archive discontinuity” | Not explained explicitly | State the actual contrast: hard-threshold weighting can abruptly remove an archive member when the threshold changes, whereas a smooth kernel changes its contribution continuously. |
| **110** | \(S\), \(\mathcal S\), and `amis_interval` | Defined operationally in the same sentence but not motivated | Motivate snapshots first as an approximation to the changing sequence of proposal densities. Keep the interval and default size in the appendix or Experimental Design. |
| **117** | “online effective-sample-size diagnostic” | No main-text formula; ESS is used extensively from line 231 onward | Define ESS in Metrics. If the online version differs from posterior ESS, give it a distinct label or omit it from Method. |
| **119** | “retroactive AMIS step,” \(m\), and “posterior denominator” | The estimator and denominator are not formally defined until lines 156–165 | Define the retrospective estimator in Section 3.5 before discussing snapshot counts or calibration. |
| **121** | “defensive prior component” | Formula at lines 156–160; assumption at line 171 | At first use explain that a positive prior component prevents the denominator from becoming too small and therefore bounds importance weights. |
| **121** | “retroactive estimator \(\widehat\pi_n\)” | Equation at lines 162–165 | Move that equation to Section 3.5, immediately after the term is introduced. |
| **128** | \(N\), \(\omega_i^{(t)}\), \(L_n\), panel codes “a1–a3,” and multiple line/point encodings | Some can be inferred; \(L_n\) is only implied by “factored once” at line 105 | Simplify Figure 1. Anything needed only to decode individual glyphs belongs in a short figure key, not a long caption. Define \(N\) and \(L_n\) if retained. |
| **149** | “capped in-box rejection” and “underflow redraws” | Explained only in Appendix lines 490–492 | Replace with “implementation approximations” in the theory overview. List the individual mechanisms only in the fidelity-ratio appendix. |
| **156–160** | \(w_n,\alpha_{s,n},\nu_n,\delta\) appear in the equation before a readable conceptual definition | Explained in the sentence after the equation and in later assumptions | Introduce the mixture in words before displaying it. Rename \(w_n\) to avoid collision with particle weights. Define the bootstrap share and defensive floor before using their symbols. |
| **160** | “\(m\le S+1=21\)” | \(S=20\) is an experimental/default setting from line 119 | Remove `21` and preferably \(S\) from the theorem section entirely; see the theory plan below. |
| **196** | “product-of-marginals truncation constant,” “capped in-box rejection,” and “underflow redraws” | Appendix lines 475–494 | Keep only a one-sentence summary in the main theory section and point to the appendix decomposition. These implementation exceptions interrupt the theorem’s main message. |
| **201** | “screening experiment,” “the two directions the summary statistics resolve,” and two benchmark configurations | Detailed only in Appendix lines 751–754 | Give the application-level question and problem class first. Retain parameter ranges and configuration history in the appendix or a compact table. |
| **214** | “calibration (SBC)” | Expanded to simulation-based calibration at line 231 | Spell out simulation-based calibration before the benchmark table or avoid the acronym in the table. |
| **224** | “probabilistic-rejection acceptor” | Detailed in Appendix line 702 | Give a one-clause definition: accept a draw with probability \(K_\epsilon(\rho)\). |
| **238** | “the model’s stated domain” | Limitations are distributed across lines 229, 246, 276, and 280 | State the predictor’s domain explicitly in Metrics: simulation time dominates barrier latency and the timing sample represents the relevant runtime tail. Then “outside the domain” has a fixed referent. |
| **292/295** | “per-simulation efficiency” | Operationally defined only by the Table 3 caption | Define it in Metrics as the ratio of matched-count order-statistic tolerances, including which direction is better. |
| **327** | “reporting-support effect” | Inferred from the subsequent comparison but never defined | Replace with a direct statement that the full-history estimator spreads posterior mass over more particles than the top-\(k\) archive. |
| **353** | “ESS-retention bisection against the archive” | Explained only in Appendix line 698 | This is part of the core bandwidth rule and must be introduced in Section 3.3. Reconcile it with line 95’s different description involving \(2k\) particles. |
| **370/388** | “the shipped starting bandwidth” and “the benchmark shipped with” | No clear source or baseline configuration is named | Replace “shipped” with the precise provenance: default configuration value, earlier benchmark setting, or software default. |

---

## Priority 2: redesign and synchronize Figure 1

The current Figure 1 caption at line 128 tries to serve as method definition, visual key, baseline disclaimer, and implementation note. It is too dense to function as a graphical abstract.

### Recommended content

Reduce the figure to two matched flows:

- **Generation-based baseline:** construct proposal → dispatch a population → wait for the population boundary → update once.
- **Generation-free method:** receive one completed evaluation → update history/archive/proposal → dispatch one new candidate immediately → after the budget, retrospectively reweight the history.

The figure should expose the barrier difference, not reproduce every equation in the algorithm.

### Synchronization corrections

- The top row currently depicts “textbook PMC with a quantile schedule and a hard threshold,” while the experimental baseline is kernel-matched pyABC. Either:
  - redraw the top row to match the actual smooth-kernel baseline, which is preferable; or
  - label it unambiguously as a generic conceptual contrast and do not imply that it is the experimental baseline.
- Correct the wording “weighted against the mixture of all proposals used so far.” The implementation uses:
  - current plus the last \(S\) snapshots online;
  - \(m\) reconstructed snapshots plus a defensive prior component for reporting.
  “All proposals” is false unless the full mixture is actually evaluated.
- Eliminate the “4 proposals” ambiguity. The numeral is a step label, not a count. Write “Step 4: build one proposal mixture \(q_n\)” and visually separate the step number from the noun.
- Use consistent names across the figure, Algorithm 1, and prose: history, archive, bandwidth, proposal, candidate, proposal-time weight, retrospective posterior weight.
- If the toy scatter panels are retained, use identical coordinate limits in corresponding panels. If a zoom is necessary, draw a visible zoom rectangle and connector and state the scale change in the panel itself. The current unexplained zoom difference makes changes in concentration look like changes in scale.
- Remove most formulas and encoding details from the caption. Target roughly 80–120 words:
  1. one sentence stating the comparison;
  2. one sentence explaining the barrier;
  3. one sentence explaining the arrival-driven loop;
  4. one short sentence defining only essential encodings.
- Put the mathematical details in Sections 3.2–3.5 and Algorithm 1, not in the caption.

---

## Priority 3: expand the Background enough to carry the conceptual load

The current Background, lines 69–81, is only three dense paragraphs. It names most relevant areas but does not prepare the reader for the method’s distinctions.

Add two or three focused paragraphs:

1. **How sequential ABC creates a barrier.** Walk through population proposal, simulation, weighting/acceptance, and population-level update. Define population, proposal, bandwidth/tolerance, and the dependence of generation \(t+1\) on completion of generation \(t\).
2. **Why AMIS permits reuse of past evaluations.** Explain the balance heuristic intuitively: because proposals change, a pooled sample must be weighted against the mixture of densities that generated it, rather than only its latest proposal. This should prepare the reader for snapshots without introducing buffers.
3. **What asynchrony changes and what it risks.** Contrast barrier removal with look-ahead scheduling, island methods, and anytime/length-biased sampling. State that parameter-dependent completion times can alter the observed history and foreshadow the limitation at line 408.

Also add a short vocabulary bridge:

- A hard ABC threshold produces an accepted population.
- A smooth ABC kernel gives every evaluation a graded weight.
- In this paper, \(\epsilon\) is the smooth-kernel bandwidth; \(\epsilon_{(k)}\) is a separate empirical tolerance metric.

Do not expand Background into a literature catalogue. Its purpose is to make the choices in Section 3 feel necessary rather than novel terminology appearing without preparation.

---

## Priority 4: remove the experimental constant from the theory

Line 160 currently states:

> “\(m\le S{+}1=21\) history-reconstructed proposals…”

This makes the theoretical object look tied to the experimental choice \(S=20\).

Revise the theory as follows:

- State the denominator using an arbitrary fixed \(m\):
  \[
  \bar q_n=\lambda_n\pi+(1-\lambda_n)\sum_{s=1}^{m}\alpha_{s,n}q_{\tau_s}.
  \]
- The assumptions should require a fixed finite \(m\), valid mixture weights, and a defensive lower bound \(\lambda_n\ge\delta>0\). They should not mention \(S\), `amis_interval`, or `21`.
- Explain the fidelity ratio before the assumptions:
  - \(\bar q_n^\star\) is the average density that actually generated the sample;
  - \(\bar q_n\) is the tractable denominator used for reporting;
  - \(r=\lim\bar q_n^\star/\bar q_n\);
  - \(r\equiv1\) gives the intended smooth-ABC posterior.
- After the theorem, add a clearly labeled implementation remark: the reported experiments reconstruct at most 21 proposals because the implementation uses \(S=20\), and the appendix evaluates the resulting approximation.
- Keep the general growing-\(m_n\) discussion in the appendix.
- Move \(L_\epsilon^{(2)}\), which is needed for the appendix CLT rather than the main consistency theorem, out of the main theorem setup.
- Reduce the main-body list of approximation mechanisms at line 196 to their categories: finite snapshot approximation, defensive prior mass, proposal normalization approximation, and implementation-level redraw/rejection effects. The detailed factorization belongs in Appendix Section A.1.

This directly addresses the “\(S+1\le21\)” criticism and strengthens the impression that the theorem concerns a method class rather than one run configuration.

---

## Priority 5: motivate cellsInSilico as an application class

Lines 201–203 introduce the simulator by name, dimensions, parameters, summary statistics, replicates, runtime correlations, and two historical configurations before explaining why this is an important inference problem.

Open the benchmark subsection with an application-level paragraph along these lines:

> Cell-based tissue models are stochastic, spatial, likelihood-intractable simulators used to infer biophysical parameters from morphological observations. They are a relevant target for asynchronous ABC because the parameters affect the number and arrangement of simulated cells, so they affect both the simulated summaries and the runtime. The resulting evaluations are expensive, noisy, and heterogeneous—the regime in which generation barriers are most costly.

Then introduce cellsInSilico as the concrete end-to-end implementation:

- Biological/inference question: infer division rate and target cell volume from final cell count and cluster radius.
- Why those two parameters and summaries are identifiable, in one sentence.
- Why four replicate simulations form one evaluation.
- Why \(50^3\) and \(80^3\) test different runtime regimes.
- State explicitly that the observations are synthetic and the truth is known. It is an **application-motivated mechanistic inference benchmark**, not a real-data biological result.

Move the following out of the opening main-text paragraph:

- full prior ranges;
- screening-study ratios and confounding angles;
- older division-rate/motility configuration;
- detailed summary-feature list;
- configuration provenance.

Keep those in Appendix lines 751–754 and summarize their role in a compact benchmark table.

Replace line 203—

> “The Cellular Potts benchmark is a calibration instrument…”

—with a less reductive conclusion: it combines a production-grade mechanistic simulator with synthetic known-truth data, allowing both systems measurements and posterior validation.

---

## Priority 6: split or redesign Figure 2

Figure 2 combines three different questions:

- panel (a): method throughput under a persistent worker-level straggler;
- panel (b): idle fraction under evaluation-level heterogeneity;
- panel (c): predicted versus measured barrier cost across workloads.

Panel (c) is described at line 238 as “the paper’s central measurement,” but it shares space and legends with two illustrative panels using different quantities and encodings. The caption at line 243 is correspondingly overloaded.

### Preferred redesign

Split it into two figures:

1. **Central figure:** predicted versus measured barrier cost, currently panel (c). Give it direct workload labels or one workload legend. Mark outside-domain cases directly.
2. **Mechanism figure:** throughput and idle fraction under the two synthetic heterogeneity regimes, currently panels (a) and (b). These share the asynchronous/pyABC legend.

This yields one legend per figure and lets the central result occupy a readable square panel.

### If the combined figure must remain

- Label the top legend “Methods, panels (a–b)” and place it within the left panel group.
- Label the second legend “Workloads, panel (c)” and place it next to panel (c).
- Directly label the open markers “outside predictor domain” rather than putting them into a legend that appears global.
- Add panel subtitles stating both the response and heterogeneity type.
- Visually group panels (a–b) and separate panel (c), rather than using a global top and bottom legend.

### Caption rule

The revised caption should define the data and encodings, not repeat the Results paragraph. Move numerical conclusions such as “pyABC falls from approximately 8800 to 60” and the full experiment inventory into the prose. Target roughly 100–130 words.

---

## Priority 7: replace the AI-sounding prose patterns

### Most conspicuous examples

1. Line 59:

   > “This paper removes the generation.”

   This is a slogan before a precise mechanism.

2. Line 59:

   > “It keeps the generation and recovers the \(10\)--\(50\%\) that the boundary wastes. We remove the boundary and recover the factor.”

   The repeated antithesis is polished but imprecise and sounds promotional.

3. Line 108:

   > “Snapshots are consumed in two places, and the distinction decides what is carried state and what is not.”

   Abstract nouns are made to act, while the actual two uses remain unnamed until later.

4. Line 149:

   > “The analysis is organized around a single question, because that is all the proofs turn out to need…”

   This is theatrical metacommentary. State the mathematical reduction directly.

5. Line 181:

   > “Assumption~\ref{ass:stab} carries everything else.”

   “Carries” hides which properties are being assumed.

6. Line 203:

   > “The Cellular Potts benchmark is a calibration instrument…”

   The metaphor is abrupt and undercuts the application motivation.

7. Line 246:

   > “The prediction is a one-line calculation.”

   This comments on elegance rather than telling the reader what is calculated.

8. Line 278:

   > “That factor is not the barrier’s.”

   The dramatic short sentence delays the actual decomposition.

9. Line 292:

   > “Throughput is instrumental; what a practitioner buys is the tolerance the run reaches.”

   This marketing-like construction recurs elsewhere as “what the choice needs” and “before paying for it.”

10. Lines 353 and 405:

   > “The lower panels say why.”

   > “The effective-sample-size ceiling is where the method has headroom.”

   Both personify evidence or use business-style metaphor instead of giving the causal statement.

### Concrete rewriting rules

- Replace slogans with mechanism:
  - “This paper removes the generation” → “We update the proposal after each completed evaluation rather than after a completed population.”
- Replace personification with an explicit evidential subject:
  - “The lower panels say why” → “The lower panels show that ESS remains nearly constant as the history grows.”
- Replace rhetorical contrasts with quantified attribution:
  - “That factor is not the barrier’s” → “Only \(1.2\times\) is attributable to idle time at the barrier; the remaining slowdown comes from longer simulation durations in the twin.”
- Avoid “what a practitioner buys,” “what the paper reports,” “what bounds it,” and similar repeated constructions. Name the metric or limitation directly.
- Use one claim per sentence. Settings, mechanisms, results, and interpretations should not occupy a single sentence.
- Put the main clause before parenthetical qualifications. Move secondary settings to tables or appendices.
- Replace vague verbs such as “carries,” “hosts,” “books,” “sits,” “binds,” and “turns on” with “represents,” “is evaluated on,” “attributes,” “is,” “limits,” and “becomes advantageous.”
- Do not announce simplicity (“one-line calculation,” “all the proofs need”). Show the calculation or reduction.
- Avoid symmetrical three- and four-part rhetorical sequences unless the parts are genuinely parallel and necessary.
- Qualify broad claims. For example, line 57’s “Simulators worth running on such machines are rarely homogeneous” should become a scoped claim about the targeted class of population-varying stochastic simulators.
- End result paragraphs with the interpretation relevant to the subsection claim, not another numerical detail.

---

## Paragraphs that are too dense

### Highest-priority cuts

| Lines | Problem | Revision |
|---|---|---|
| **52** | The abstract combines motivation, method, theorem, four result families, numerous ratios, a benchmark, and two limitations in one paragraph. | Use five moves: problem; method; theoretical result; central systems result; posterior result and principal limitation. Remove most secondary percentages. |
| **57** | Defines ABC, generations, barriers, runtime notation, straggler factor, scaling behavior, and tissue runtime in one paragraph. | Split after the barrier mechanism. Put the formal straggler factor and its assumptions in a second paragraph. |
| **59** | Method, AMIS, replayability, Propulate, look-ahead work, pyABC, and headline results are mixed. | Separate “our method” from “relation to alternatives/results.” Remove slogans. |
| **77** | Smooth kernels, SMC/PMC, AMIS, balance heuristic, PMC lineage, consistency, and the streaming limit appear without pause. | Use one paragraph for staged ABC and one for AMIS. End with the gap the proposed method fills. |
| **87–92** | Restart semantics, mutable state exceptions, framework interface, and formal history notation appear together. | Keep only the history definition in Section 3.2; move the rest to the appendix. |
| **95** | Bootstrap, bandwidth metadata, two reconstructed bandwidths, monotonicity, scheduler cadence, halving, initial settings, expensive simulators, and prior-predictive tuning appear in one paragraph. | Split conceptual monotonicity from implementation scheduling; move cadence and option names out. |
| **108–121** | Snapshot definition, online buffer, proposal-time weighting, posterior reconstruction, buffer calibration, three weight systems, defensive mixture, and complexity are compressed into three paragraphs. | Give separate subparagraphs for motivation, online weighting, and posterior reporting. Move empirical tuning and timing out. |
| **128** | The Figure 1 caption is effectively a second method section. | Redesign the figure and reduce the caption as described above. |
| **147–160** | Proof motivation, AMIS literature qualification, implementation departures, conditional densities, reconstructed mixtures, prior floor, fixed snapshot count, bandwidth limits, second moments, and the estimator are introduced before the assumptions. | Start with the ideal-versus-used denominator distinction; define \(r\); state the estimator already introduced in Section 3; then give assumptions. |
| **201** | Five benchmarks, their inferential roles, runtime mechanisms, parameter ranges, summary statistics, replicate averaging, sizes, and historical configurations are placed in one paragraph. | Give one short role paragraph per benchmark class; move Cellular Potts specifics into a dedicated paragraph and appendix. |
| **224** | Three comparators, dispatch behavior, population occupancy, barriers, budgets, seeding, MPI, and baseline exclusions are combined. | Give one paragraph per comparator, each following “question tested → construction → fairness condition.” |
| **229–231** | Metric definitions are mixed with derivations, resampling procedures, censoring qualifications, posterior references, scoring rules, and benchmark-specific reporting. | Keep definitions in the main text; put estimator details and benchmark-specific scoring in a protocol table or appendix. |
| **238–243** | The central result paragraph and caption both inventory every configuration, numerical range, domain exception, and interpretation. | Let the paragraph make the claim and discuss exceptions; let the caption identify panels and encodings only. |
| **276–282** | Four consecutive result paragraphs repeatedly alternate among twin, pyABC, prediction, utilization, duration ratios, and two Cellular Potts configurations. | Reorganize by question: predictor validation; barrier-only attribution; practical pyABC comparison. Use a small decomposition table. |
| **315** | Mechanism, four benchmark ratios, crossover inference, placement effects, an alternate run, and Gaussian failure are all combined. | State the measured crossover first, then one paragraph on why it depends on placement, then the below-crossover counterexample. |
| **327** | Calibration results for three targets, rank histograms, archive behavior, support size, budget doubling, and a hyperparameter sweep are combined. | Separate by target and finish with a distinct synthesis paragraph. |
| **353** | Two benchmarks, several \(W_1\) values, rejection ABC, ESS trajectories, a sweep, bandwidth choice, dimension effects, and a causal diagnosis are compressed together. | Split “recovery result,” “ESS observation,” and “interpretation/limitation.” |
| **388–392** | Starting bandwidth, schedule movement, twin behavior, order-statistic re-reporting, and archive-size tuning form three highly technical paragraphs late in Results. | Create a clearly titled “Sensitivity and operational requirements” subsection with one limitation per paragraph; move detailed call counts and schedule mechanics to the appendix. |
| **411** | The conclusion repeats almost every numerical result plus theorem caveats and tuning rules. | State one systems conclusion, one statistical conclusion, and two conditions of use. |

---

## Priority 8: align Results and Discussion with the revised structure

- Keep C1–C4 if they help navigation, but state each claim in plain language rather than relying on the labels.
- In C1, separate:
  1. validation of the timing predictor against the twin;
  2. practical comparison against pyABC;
  3. cases outside the predictor’s domain.
- In C2, define “per-simulation efficiency” before reporting it and avoid switching repeatedly between tolerance ratios, throughput ratios, and combined wall-clock ratios.
- In C3, frame the \(4\) ms crossover as an observed boundary for the tested placement and implementation, not a universal constant.
- In C4, separate calibration, reference-posterior recovery, Cellular Potts recovery, and tuning limitations. They currently compete within one long subsection.
- Move the starting-bandwidth and archive-size guidance into a short “Operational sensitivity” subsection before Discussion. Discussion can then interpret those findings rather than introducing them again.
- Shorten the four italicized practitioner rules at lines 397–403. Each should have:
  - condition;
  - recommendation;
  - evidence reference.
- Avoid presenting \(k=100\) or “one fifth of the prior-predictive median” as universal rules. Identify them as values that transferred across the reported experiments.

---

## Step-by-step work order and effort

| Order | Work item | Priority | Estimated effort |
|---:|---|---|---:|
| 1 | Create the terminology sheet and resolve the \(S/m\), \(w_i/w_n\), bandwidth/tolerance, simulation/evaluation, and snapshot conventions. Verify the \(S=0\) versus \(m\le S+1\) behavior. | P0 | 1.5–2.5 h |
| 2 | Redesign Figure 1’s conceptual content and write its short caption. Decide whether the comparison is the actual matched pyABC baseline or a generic ABC-PMC schematic. | P0 | 3–5 h |
| 3 | Write the new Section 3.1 overview, then reorder Sections 3.2–3.6 around the target outline. Move the posterior estimator into Method. | P0 | 5–7 h |
| 4 | Extend Background with staged-ABC mechanics, AMIS intuition, and asynchronous/anytime context. | P1 | 2–3 h |
| 5 | Generalize the theory presentation: define the fidelity ratio plainly, remove \(S+1=21\), use arbitrary \(m\), and separate the theorem from implementation diagnostics. | P0 | 2–4 h |
| 6 | Rewrite the benchmark introduction so Cellular Potts is motivated as stochastic mechanistic tissue inference with endogenous, parameter-dependent runtime. | P0 | 1.5–2.5 h |
| 7 | Split or regroup Figure 2, fix panel-specific legends, and rewrite the caption. | P1 | 3–5 h |
| 8 | Refactor Experimental Design and Results paragraphs identified above, without changing claims or numbers. | P1 | 4–6 h |
| 9 | Run a dedicated prose pass using the rewriting rules: remove slogans, personification, vague metaphors, promotional phrasing, and overloaded sentences. | P1 | 5–8 h |
| 10 | Rewrite the abstract, contribution list, Discussion, and Conclusion last, once terminology and section order are stable. | P1 | 2–3 h |
| 11 | Perform a consistency audit across text, algorithms, figures, captions, and appendices; compile and inspect float order, legends, cross-references, and symbol definitions. | P0 | 2–3 h |

**Estimated total:** approximately **28–44 hours**, dominated by Figure 1, the Section 3 rewrite, and the full prose pass. No new experiments should be necessary for these revisions.