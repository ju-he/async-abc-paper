# Paper restructure plan — 2026-09-21 (DRAFT, under discussion)

Source: `latex/sn-article-template/sn-article.tex` at `bff3c67` (895 lines, ~26k words, 57 pp).
Last edited 2026-09-18 — before every CPM finding in
`.plans/cpm_dimension_and_bandwidth_study_2026-09-19.md`.

## Diagnosis

The paper is a research log. Symptoms: results narrated in discovery order with corrections
embedded ("we no longer report it", "has since been repeated", "the second was not what we went
looking for"); a 600-word abstract that is a list of caveats; contribution bullets that are
paragraphs; a 1,500-word single-paragraph Limitations that re-tells results; three twin tables;
a point-mass metric the paper itself says has no resolution, still reported with two figures;
theory hedged in the main text at length (CLT that does not apply to the reported runs, four-factor
r decomposition, stabilisation-rate fits). Every CPM number is from the retired
(division_rate, motility) setup; every rejection-ABC number is a prior-sampler number; the
cross-method budgets were counted on mixed record kinds.

## The story (one paragraph)

Generation barriers cost a factor equal to the straggler factor `E[max of P]/E[mean]` of the
simulator's in-run runtime distribution; the asynchronous method removes it at ~99% utilisation
and that factor is predictable from timing data alone. The per-simulation efficiency of the
sampler is at least that of the matched synchronous baseline (1.4-2.2x better on CPM, 1.5x on
g-and-k/LV) and 50-4,600x that of rejection ABC, so the throughput multiplies through into
posterior quality (4.1x tighter tolerance at equal wall clock on CPM). Below ~1-3 s per simulation
the per-arrival coordination dominates and the method loses — a measured boundary, not an excuse.
The reported posterior is calibrated (SBC) and recovers a two-parameter CPM posterior at 92%/81%
contraction with coverage, provided the bandwidth is set from the prior-predictive discrepancy
scale; its effective sample size is bounded by ~3k, which is the method's real limit.

## Four claims = four result subsections

| # | claim | evidence (existing unless marked NEW) |
|---|---|---|
| C1 | The barrier costs the straggler factor; async removes it | twin under straggler/hetero (Tables twin, twin-hetero); CPM 50³ twin 1.18-1.21x vs 1.2 predicted; CPM 80³ pyABC 2.01x vs 1.90 predicted; NEW summary figure: measured ratio vs predicted factor across all workloads |
| C2 | Throughput converts: per-simulation efficiency >= baseline | NEW table: eps at k-th order statistic at matched simulations, 4 benchmarks x 3 methods; eps(n) exponents; CPM 4.09x at equal wall clock |
| C3 | Boundary: cost per simulation ~1-3 s | NEW small figure: throughput ratio vs cost/sim (monotone); LV strong-scaling decline (existing fig, compressed) |
| C4 | The reported posterior is valid and what limits it | SBC (1-D/multimodal/4-D, one table); reference-posterior recovery (1-D 2x better, 4-D 2x worse); CPM 2-param corner + contraction/coverage (5 reps); bandwidth rule (13% -> 81%); ESS ~ 3k ceiling; k guidance |

## Section-by-section

Legend: KEEP / CUT (main) / SI / REWRITE / NEW. Word targets are for the main text (~9-10k total).

### Abstract — REWRITE, <= 250 words
Four claims, one number each. No caveat list; the ESS ceiling gets one clause.

### 1 Introduction — REWRITE, ~600 words
Problem (barrier idle on heterogeneous simulators, with the straggler-factor framing up front);
the idea; contributions as four one-line bullets: algorithm; implementation (history-reconstructed
state in Propulate); a predictive account of the systems gain validated on synthetic and real
workloads; consistency for the reported estimator + calibration. Roadmap sentence.

### 2 Background and related work — KEEP, cut to ~700 words
2.1 ABC, 2.2 the generation barrier, 2.3 AMIS: keep, tighten. 2.4 async SMC / parallel ABC /
look-ahead scheduling: keep, halve; the look-ahead paragraph is good and should stay, shorter.

### 3 Method — KEEP structure, ~1,500 words
3.1 History-based state: keep first paragraph; the stateless-vs-not discussion -> two sentences
(detail to SI). 3.2 Bandwidth: keep, ADD the schedule and the `tol_init` rule (~1/5 of the
prior-predictive median discrepancy) — it is now a load-bearing configuration choice. 3.3 Archive +
proposal: keep. 3.4 Streaming AMIS weight: keep; move the posterior-estimator display here;
"three weight objects" -> 5-line definition; "stored fields" -> SI. 3.5 Algorithm box: keep.
3.6 Relation to ABC-SMC: keep table, cut prose.

### 4 Theory — REWRITE to ~500 words in main; the rest -> SI
Main: assumptions in words; consistency theorem (one display); r in one paragraph (what it is,
r = 1 recovers smooth ABC, two factors measured at ~1% TV); one sentence that the CLT holds for
damped-adaptation variants and that the reported configuration measurably misses its rate
condition, pointer to SI. SI: CLT + corollary + proposition; the four-factor decomposition; the
stabilisation-rate measurements and the ridge explanation; fixed-vs-growing m.

### 5 Implementation — KEEP, ~400 words
Propulate propagator; retroactive O(nk) pass off the timed path; matched-kernel pyABC baseline;
INTRODUCE the barrierized twin here as an instrument (currently introduced mid-results); rejection
ABC as best-k at fair resourcing.

### 6 Experimental design — REWRITE, ~800 words
Benchmark table gains two columns: cost per simulation and in-run runtime CV (the organising
variables). Baselines: matched pyABC, twin, best-k rejection. Metrics: throughput/utilisation;
eps at the k-th order statistic at matched simulations (define; the cross-method quantity; count
only `simulation_attempt` records); SBC; contraction + coverage on CPM; W1 to a reference where
one exists. RETIRE the point-mass metric from the main text (its critique -> SI). LV extinction:
two sentences + SI. CPM: state its role as a calibration instrument (3.7 s / 170 s per
simulation vs hours in production) and the 50³ / 80³ pair.

### 7 Results — REWRITE by claim
7.1 (C1) The barrier's cost is predictable. Open with the NEW predictor figure. Straggler figure
(existing) + one merged twin table (straggler + hetero, with a predicted-factor column; drop the
non-budget-matched reported-posterior rows -> SI). CPM: 50³ twin 1.18-1.21x on the 1.2 ceiling,
80³ pyABC 2.01x vs 1.90; utilisation 97.5% vs 48.6%. Scaling to eight nodes: async 97% parallel
efficiency at 50³ (solid); the "5x vs pyABC at 384" is mostly pyABC per-generation overhead at
13 s/sim — either caveat it or replace with the predicted 2.1x at P=384 (decision below).
7.2 (C2) Throughput converts. NEW matched-simulation eps table; the exponent per benchmark;
CPM production: 2.37x throughput x 1.44x per-simulation = 4.09x at equal wall clock (fixed
arms); the hetero-quality figure (existing) as the synthetic instance.
7.3 (C3) The boundary. NEW crossover figure; LV strong scaling (existing combined figure,
one paragraph; LV timing decomposition, scaling table, k=1000 rows -> SI).
7.4 (C4) The reported posterior. SBC condensed to one table (1-D, multimodal, 4-D); reference
recovery (existing figure, text halved); CPM two-parameter posterior (NEW corner from 14262214,
5-replicate contraction/coverage table, vs fair rejection and vs sync); the two knobs — bandwidth
(tol_init: 13% +/- 14% -> 81% +/- 1%; order-statistic re-report as the check) and archive size
(ESS ~ 3k; k=100 Pareto on LV, k=30 >= k=100 on CPM, so the advice is benchmark-dependent).
Ablation, sensitivity, k/S calibration sweep, k-frontier, parameter-coupled runtime, archive
concentration -> SI with one-sentence pointers.

### 8 Discussion — REWRITE, ~400 words
Practitioner guidance: use it above ~1-3 s/sim; predict the gain from the runtime CV; set
tol_init from the prior-predictive scale; k ~ 100 and check calibration; the ESS ceiling is the
headroom. The calibration-instrument framing of the benchmark.

### 9 Limitations — REWRITE as a bulleted list, <= 300 words
Theory scope (r, CLT rate); completion-time selection (iv) with the one-sentence result;
twin not budget-matched (unless C1 is run); benchmark cost/heterogeneity vs production; LV
extinction; the twin's slower simulations unattributed.

### 10 Conclusion — REWRITE, <= 150 words

### SI (appendices)
A Proofs (keep). B Theory detail (moved). C Implementation details (keep). D Protocol (keep) +
CPM setup: screening summary (why two parameters, what the summaries resolve), bandwidth-transient
study, 80³ assets, best-k rejection, record-kind rule. E Moved results: full twin tables incl.
reported-posterior rows; parameter-coupled runtime; k/S sweep + k-frontier; LV timing; ablation;
sensitivity; point-mass metric critique; CPM twin duration decomposition.

## Figures / tables in main (target 8 / 5)
F1 NEW predictor (measured vs predicted ratio, all workloads). F2 straggler throughput. F3 hetero
quality. F4 NEW crossover vs cost/sim. F5 scaling combined. F6 reported recovery. F7 NEW CPM
corner. F8 NEW CPM eps-vs-n (async / sync / rejection) or contraction vs tol_init.
T1 benchmarks (+cost, +CV). T2 merged twin (+predicted). T3 NEW matched-simulation eps. T4 SBC
condensed. T5 CPM production summary.

## Regeneration prerequisites (no new science)
1. Every CPM number/figure from 14262214 (posterior), 14262841 (systems), 14262032 (fair
   rejection); config table rows.
2. Rejection numbers re-derived with best_k everywhere (check LV).
3. All cross-method budgets on `simulation_attempt` rows.
4. Twin table from utilisation, not throughput (script `twin_allW.py` -> experiments/scripts).
5. Straggler factors for the injected-straggler/hetero twins from stored records (for F1).
6. fair_convert.py decomposition line.
7. Soften the k advice; D1/D2 theory scoping text.

## Open decisions
- Venue / length (memory says AISTATS or TMLR; the class file is sn-jnl).
- Theory in main: 500 words + consistency only, or keep the CLT statement in main?
- CPM 8-node scaling at 80³ (~30 node-h) vs caveated 50³ scaling + predicted P=384.
- Run C1 (matched-budget twin) in the background, or drop the twin's posterior rows.

## Regeneration status (2026-09-21, end of day)

Done, vendored, scripted (`experiments/scripts/`):
- `make_cpm_production_table.py` -> `tab_cpm_production/` (50³ control, 50³ fixed, 80³; fair rejection at both sizes; weighted synchronous posteriors from pyABC's histories).
- `make_cpm_eps_fig.py` -> `fig_cpm_eps` (ε(k=100) vs simulations, three methods, exponents).
- `make_cpm_corner_fig.py` rewritten for (division_rate, cell_volume) -> `fig_cpm_corner`.
- `make_predictor_fig.py` -> `fig_predictor` (measured vs predicted barrier cost, all workloads).
- `make_twin_cpm_decomposition.py` -> `tab_twin_cpm/twin_cpm_decomposition.csv` (utilisation × duration).
- `make_reported_recovery_fig.py --refresh-rejection` -> best-k rejection rows spliced (gaussian 0.011, gandk 0.47).
- `repair_two_param_cpm_records.py` + `pyabc_populations.csv.gz` (two record-layer bugs found and fixed today; see previous-fixes.md).
- C1 matched-budget twin: done, decisive (study log).

Running: `make_matched_eps_table.py --refresh` (streams the three big campaigns; feeds tab:matched-eps and fig_crossover).

Still to do before prose: `make_crossover_fig.py` (from the matched-ε summary); config-table rows for the CPM setups; the 50³ `fig_cpm_util`/scaling caption caveat (no regeneration needed); reported-recovery sync rows stay uniformly weighted (no histories kept) and the text must say so.

## Regeneration complete (2026-09-21, later)

All "before prose" items are done and committed (33778d6 … 342f77e). Added since the status
above: `tab_cpm_knobs` (tol_init / k / re-report), the C1 arm in `tab_twin`, `tab_matched_eps` +
`fig_eps_curves` + `fig_crossover`. **C3 changes:** the boundary is ~4 ms per simulation
(throughput parity at Lotka–Volterra's 4 ms, net-positive between 2 and 4 ms), not 1–3 s — the
study log's LV cost was wrong. State C3 as "milliseconds", and note that per-simulation efficiency
also rises monotonically with cost (0.78 → 1.60).

Figure/table inventory for the main text (final): F1 fig_predictor · F2 fig_straggler_throughput ·
F3 fig_hetero_quality · F4 fig_crossover · F5 fig_scaling_combined (caveated caption) ·
F6 fig_reported_recovery · F7 fig_cpm_corner · F8 fig_eps_curves (fig_cpm_eps with rejection → SI).
T1 benchmarks (+cost, +CV) · T2 tab_twin merged (+predicted, +matched-budget posterior row) ·
T3 tab_matched_eps summary · T4 SBC condensed · T5 tab_cpm_production. SI: tab_cpm_knobs,
twin_cpm_decomposition, fig_cpm_eps, plus everything moved.

Writing order: abstract + §1 + §6 (vocabulary), then §7 by claim, then §3–5, then §8–10, then SI.

## Rewrite status (2026-09-21, end)

Full first pass of the new structure is in the tex and committed (5214deb … f3bee0d): abstract,
§1, §3 trims (+ε₀ rule), §4 short (CLT/r → Appendix A), §5 (+twin, +best-k), §6, §7.1–7.4, §8
(four rules), §9 (bullets), §10, Appendix A (theory), E (CPM setup), F (additional results).
Compiles clean (no undefined refs). 59 pages with SI.

Next: a cold read for consistency — §2 (background/related work still long), the method
comparison table's prose, Appendix D protocol (mentions of old figures), captions of moved
figures; then figure polish (fig_predictor size/labels, fig_crossover label overlap); then the
abstract ≤250-word check for S&C and the keywords.

## Cold read and figure polish (2026-09-21, later still)

Done (uncommitted at the end of the session that did it):
- Consistency: running head `\methodname` is now "generation-free ABC" (was "asynchronous
  steady-state ABC"); §2.4 look-ahead discussion halved (§2 ≈ 560 words); §3.6 comparison-table
  prose cut to three sentences; §3.4 stored-record fields now point at eq:history; §6.5 no longer
  says the scaling results are supplement-only; the runtime-CV bases are separated (in-run vs
  prior-wide: intro/discussion say "0.05 to above 0.2", §7.1 flags the 0.10 as the scaling runs');
  the "2.1× at 384 workers" is attributed to the 80³ runtime spread everywhere it appears (§7.1,
  fig:scaling caption, limitations); the twin-table pointer for the wall-limited W1 fixed; stale
  phrases removed from the proofs ("item (iv)"), the assumption-status table ("Limitation (iv)")
  and the protocol ("reproducible up to MPI arrival order", "resumes from the next generation");
  Appendix D benchmark notes rewritten for the two-parameter CPM setup (they still described
  division rate + motility with the whitened morphology discrepancy); config table gains the 80³
  row and a barrierized-twin row (T); Appendix F: LV cost 2 → 4 ms (two places), four
  cross-references re-pointed (§7.1 → §7.3, coupling study, ks-sweep), three redundant
  `\paragraph` headers dropped.
- The CPM per-simulation ratio is standardised on **1.6×** (tab:matched-eps rule: each
  replicate matched at its own count). The production-table rule (match at the baseline's median
  count) gives 1.54, which the abstract/intro/§7.2/conclusion had rounded to 1.5.
- Abstract 292 → 249 words; six keywords (Springer asks for 4–6; no MSC codes needed).
- fig_predictor: legend below the axes, ratio tick labels, 0.6\linewidth. fig_crossover: legend
  upper left, per-benchmark label offsets, parity label, headroom. Tables benchmarks /
  matched-eps / cpm-production reflowed (were 228 / 30 / 127 pt overfull; the all-5/5 coverage
  column of tab:cpm-production is now a caption sentence). Log clean: no overfull boxes, no
  undefined references; 58 pages.

Still open:
- §4 is ≈1000 words in the main text against the plan's ≈500: Assumption 5(ii) (the CLT rate
  condition) and the r-accounting paragraph could move to Appendix A, which needs re-labelling
  because thm:clt cites ass:stab(ii). §6 ≈1250 vs 800, §1 ≈760 vs 600; total main text ≈9.4k
  words, inside the 9–10k budget, so this is optional.
- Submission checklist for S&C: author block, funding, acknowledgements, data deposit.

## External reviews (2026-09-21, after commit 07364d5)

Codex (GPT-5.6-Sol, xhigh) and a fresh Claude Fable subagent reviewed the manuscript
independently with the same brief. Both: **major revision**. Reports and a verified digest with
a suggested order of work: `.plans/reviews/{codex-gpt56sol,claude-fable,digest}-2026-09-21.md`.
The three largest findings: the 50³ twin/scaling runs and the theory diagnostics are on the
retired (division rate, motility) setup, undisclosed; the reported estimator is collapsed on the
heterogeneity campaign for every arm (async full-history W1 up to 2.1 vs archive 0.02); the
measured C1 ratios are not the straggler factor E[max]/μ the abstract defines.

## Review response pass (2026-09-21, later)

All seven digest items worked through (see `.plans/reviews/digest-2026-09-21.md`, section
"Resolution"). Two findings from the raw records changed the paper's story: the twin's
reported-posterior collapse is the per-rank bandwidth-search throttle (re-reporting at ε₍₁₀₀₎
restores it; new `make_twin_rereport.py` + two vendored CSVs; Tables 3/4 lower blocks), and the
C3 boundary was measured with 48 ranks spread 12 per node over four nodes (Appendix D corrected).
Compiles clean, 62 pp. Open items (a)–(g) listed in the digest.
