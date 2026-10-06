# TMLR draft feedback (2026-10-06): combined revision plan, v2

v2 (same day): second pass focused on readability and understandability under a flat page
budget. Section 6 (budget) and the readability rules in Section 2 are new; the work order was
re-cut accordingly.

Inputs
- Feedback: `.plans/reviews/tmlr_draft_feedback_2026-10-06.md`
- Plan A (Fable): `.plans/reviews/fable_feedback_plan_2026-10-06.md`
- Plan B (codex-cli, read-only run): `.plans/reviews/codex_structure_prose_plan_2026-10-06.md`
- Manuscript: `latex/tmlr/tmlr-article.tex` (main body lines 55-427; 17 pages before the
  references; about 5,400 words outside floats; 6 figures, 5 tables, 1 algorithm in the main body)

## 1. How the two plans compare

Both plans give the same diagnosis: the text was written and then compressed from the inside
out, so every section opens with its most specific true statement, and terms coined for internal
use (twin, snapshot, stamp, fidelity ratio, retroactive estimator, production configuration)
leaked into the abstract and introduction. Both propose the same six-subsection outline for
Section 3 with a new conceptual 3.1 in front of Fig. 1, moving the posterior estimator from
Section 4 into Section 3, splitting Fig. 2 into the central predictor panel and a mechanism
figure, motivating Cellular Potts as a problem class before naming the software, and stating
`m <= S+1` without the `= 21`.

Where they differ, and the decision taken:

| Topic | Plan A | Plan B | Decision |
|---|---|---|---|
| Terminology contract before editing | not explicit | P0: fixed vocabulary (bandwidth vs empirical tolerance; candidate / evaluation / particle; worker vs rank; rename the denominator mass `w_n`; define ESS once) | **Adopt B.** Cheap, and every later step depends on it. |
| Fig. 1 content | keep the strip, fix frames, proposal accumulation, vocabulary, caption | redraw as two matched flows; optionally redraw row (a) as the smooth-kernel pyABC baseline | **Keep the strip, fix it.** The reviewer calls it the graphical abstract and asked for synchronisation, not a new concept. Row (a) stays schematic PMC, labelled so in the row title. Fallback if the 3.1 draft shows the strip is too detailed to open the section: the lighter drafts in `latex/figure-drafts` (A1 method overview, GA1 barrier Gantt). |
| "4 proposal" | b6 shows several contours while column 4 shows one | the step numeral "4" next to "proposal" reads as "4 proposals" | **Both.** Separate step numbers from step names; fix the b6 label "all proposals used so far" (false: current proposal plus the last S snapshots). |
| "Zoom difference" | verified: all twelve panels share the unit-box frame; the rows are at non-comparable stages, so PMC's wide proposal next to our tight one reads as a scale change | draw a zoom box if any zoom is used | **Draw the frame** in every panel and bring the rows to a comparable stage, or say in each row title which stage it shows. |
| Results and Discussion | no structural change | P8: reorganise C1-C4, add an "operational sensitivity" subsection, shorten the practitioner rules, rewrite the conclusion | **v1 deferred this; v2 adopts the cuts** (Section 6 below), because the number-dense results prose is the main readability cost and the only place the added words can be paid for. Claims and numbers do not change; prose that repeats a table is removed. |
| Theory | fix `=21`, re-open with the question, pointer for implementation departures | same, plus arbitrary fixed `m`, move `L^{(2)}_epsilon` to the appendix, implementation remark after the theorem | **Adopt B's fuller version.** |
| Prose rules | sequenced only | ten quoted AI-sounding patterns and rewriting rules | **Input to the prose pass**, which runs last. |

Two factual points raised by Plan B were checked against the implementation
(`propulate/propagators/abcpmc.py` at e148f4f in the sibling propulate checkout):

- `S=0` and the two-proposal denominator (line 119 vs `m <= S+1`): the post-hoc pass picks
  `max(1, S)` evenly spaced history indices and then appends the last index if it is missing, so
  `S=0` gives `m=2` and `S>=1` gives `m <= S+1`. Write "at most S+1 (two when S=0)".
- The bandwidth rule (line 95 "toward the bandwidth at which 2k particles lie within it" vs
  line 353 "ESS-retention bisection"): one scheduler. Per call it proposes the largest bandwidth
  retaining 95% of the kernel-weighted ESS, capped at halving; an acceptance gate holds the
  bandwidth until 2k evaluated particles lie inside it, so it settles near that order statistic.
  New 3.3 states both clauses once; 6.4 refers back.

## 2. Readability rules (apply in every step; none of them adds words)

1. **Claim first, evidence second.** Every subsection and every paragraph opens with the
   sentence a hurried reader should take away. Results subsections already have claim titles;
   their paragraphs do not yet have topic sentences (e.g. 6.1 "Runtime heterogeneity" opens with
   the injection mechanism, not with "the prediction holds to within 20%").
2. **Numbers live in one place.** A number that is in a table or figure is not repeated in the
   prose unless it is the one number the sentence is about. A comparison with more than two
   numbers is a table row, not a sentence. (Today: the 6.1 heterogeneity paragraph carries 14
   numbers, the two Cellular Potts paragraphs about 20, the C3 paragraph 12.)
3. **One comparison per sentence.** No sentence contains two ratios and a caveat.
4. **Fig. 1 is the anchor.** Sections 3.2 to 3.6 and Algorithm 1 refer to its six step numbers
   ("step 2 of Fig. 1") instead of re-describing the step. This is the cheapest understandability
   aid available: the reader always has a picture to attach a definition to.
5. **Symbol diet in the main text.** Drop from the main body: `epsilon_hist`, `epsilon_sched`
   (describe in words), `tau_i` and the weight formula inside the history tuple, `L^{(2)}`,
   `nu_n` and `alpha_{s,n}` (say "mixture weights proportional to the draws each snapshot stands
   in for"), `N` in the Fig. 1 caption. Keep: `H_n, A_n, k, n, W, S, m, epsilon_n, epsilon_0,
   epsilon_(k), q_n, q_s, \bar q_n, w^*, W_{i,n}, Sigma_n, K_epsilon, r, delta`.
6. **Names that say what they do.** The three weightings become *archive weights* (define the
   proposal), *parent weight* `w^*` (chooses the parent and nothing else), *posterior weights*
   `W_{i,n}` (the estimate). "Adaptation weight" and "proposal-time weight" disappear.
7. **One configuration story.** The Cellular Potts "earlier configuration" is mentioned once in
   5.1 (one sentence: the 50^3 systems runs used an earlier parameterisation of the same
   simulator; only its runtime distribution enters those results) and afterwards only as
   "(earlier param.)" in captions. Today it recurs six times in the main text.
8. **Cross-references are not sentences.** At most one parenthetical reference per sentence;
   chains like "(Section 5.3; the curves are in Appendix F, Fig. 9)" are cut to the one target
   the reader needs now.
9. **Paragraph headings only where there are three or more parallel items** (6.1, 6.4,
   Discussion). Elsewhere topic sentences do the job.

## 3. Terminology sheet (step 1, governs everything after)

- **bandwidth** `epsilon_n`: the smooth-kernel parameter. **empirical tolerance**
  `epsilon_(k)(n)`: the order-statistic metric. Never "tolerance" for the former in the main text.
- **candidate** (before simulation), **evaluation** (one parameter to discrepancy; on Cellular
  Potts one evaluation is four simulations), **particle** (a recorded evaluation).
- **history** `H_n`: the append-only record of completed evaluations. **archive** `A_n`: its k
  lowest-discrepancy entries.
- **proposal snapshot** `q_s`; **online mixture** (current proposal plus the last S snapshots;
  gives the parent weight); **posterior denominator** `\bar q_n` (m reconstructed snapshots plus a
  prior component; gives the posterior weights). S and m stay distinct symbols.
- Rename the prior-component mass in \eqref{eq:snapshot-denominator} from `w_n` to `\lambda_n`.
- **worker** in the main text; MPI rank, propagator call, cache, and option names
  (`tol_init`, `perturbation_scale`, `amis_interval`, `amis_snapshots`) only in Appendix B, in a
  hyperparameter table (symbol, option name, value, where set).
- **twin**: "the same sampler with a collective barrier before each proposal", in words at first
  use. **pyABC**: "pyABC with the same kernel and bandwidth schedule". **fidelity ratio** does not
  appear before Section 4.
- Define ESS (weighted, formula) once in 5.3.

## 4. Changes by section

### 4.1 Abstract, introduction, contributions (net -170 words)
- Abstract to about 200 words in five moves: problem; method in plain words (update after every
  arrival; weight each particle against the mixture of proposals it could have come from;
  estimate from everything simulated); theorem in one clause; the central systems result; the
  posterior result and its limit. Four numbers (cost range, 2.1-2.4x, 4.1x, few ms). Drop
  "fidelity ratio", "barrierized twin", "kernel-matched", "archive size", "prior-predictive
  discrepancy scale".
- Intro paragraph 2: split "our method" from "relation to alternatives and headline numbers";
  add the reason for the history-only design (nothing to synchronise, restart is a replay) and
  one clause for the island model; replace the slogan sentences (Plan B examples 1-2).
- Contributions: four items of about 30 words each, in words, with the forward references.

### 4.2 Background (+250 words; the reviewer allowed it)
1. How sequential ABC makes a barrier: population, proposal, bandwidth, and why generation t+1
   waits for generation t; "generation" and "population" made operational in pyABC terms.
2. Why AMIS lets past evaluations be reused: one displayed equation for the balance-heuristic
   weight beside the single-proposal PMC weight. Section 3.5 then only says "the streaming case".
3. The simulator class: cell-based and agent-based spatial stochastic models (Cellular Potts is
   the canonical cell-based model), likelihood-intractable, seconds to hours per run, cost that
   tracks the number of agents and so varies by orders of magnitude across the prior. Two or
   three application citations. This plants the Cellular Potts benchmark.
4. One bridging sentence: hard threshold gives an accepted population, smooth kernel gives graded
   weights; `epsilon` is the bandwidth, `epsilon_(k)` a separate metric.

### 4.3 Section 3 Method (net about -100 words: new 3.1 replaces old 3.1; details leave)

| New | Says | Level | To Appendix B |
|---|---|---|---|
| 3.1 Overview | Fig. 1 directly under the heading. Para 1: what one update does, in the order of Fig. 1's six steps (an evaluation arrives and is appended; the k best form the archive; a smooth weighted mixture over the archive proposes the next candidate; it goes to the free worker at once; it is weighted against the proposals it could have come from). Para 2: the one design rule (every quantity is recomputed from the history) and why; its one exception, the online snapshot buffer, named only. Para 3: the three weightings in words (archive weights, parent weight, posterior weights), moved up from 3.4. | plain words; symbols k, n, epsilon only | Propulate interface, crash recovery, scheduler throttles |
| 3.2 History and archive (steps 1-2) | `H_n`, `A_n`, why history-only state allows asynchronous updates and replay. | definitions | restart semantics |
| 3.3 Bandwidth (step 2) | Why the bandwidth must shrink and never grow; reconstruction as the running minimum of stored bandwidths; the rule in two clauses (per-call ESS-retention target, acceptance gate at 2k inside); the starting value matters on expensive simulators (forward reference to 6.4). | principle, one equation | per-rank cadence, halving cap, `tol_init` |
| 3.4 Archive proposal (steps 3-5) | \eqref{eq:steady-state-proposal}, `Sigma_n`, what the smooth kernel buys (an archive member's contribution changes continuously with the bandwidth instead of dropping out). | core math | Cholesky, jitter, truncation normaliser, `perturbation_scale` |
| 3.5 Weighting and the posterior estimate (step 6) | Why a moving proposal needs mixture weighting; the parent weight \eqref{eq:amis-weight}; then the posterior estimator \eqref{eq:posterior-estimator} with the denominator \eqref{eq:snapshot-denominator} moved here from Section 4, the prior component in one sentence (keeps the denominator bounded away from zero); "at most S+1 snapshots (two when S=0)"; O(nk) post-hoc pass excluded from throughput. | concept, then equations | ring buffer, `amis_interval`, the S=0 coverage number (to the sensitivity appendix) |
| 3.6 Update loop | Algorithm 1 with its lines labelled by the same step numbers, and one paragraph on W workers. | procedural | MPI pattern, serialisation |

Pattern for every technical subsection: purpose, definition or equation, one consequence.
No settings, calibration numbers or complexity figures inside definitions.

### 4.4 Section 4 Theory (net -20 words)
- `m\le S{+}1=21` -> fixed finite `m`; the assumptions mention neither S nor 21. One-line
  implementation remark after Theorem 1: the experiments use S=20, so at most 21 snapshots.
- Open with the question in plain words (ideal denominator vs the one used; `r` is their limiting
  ratio; `r = 1` gives the smooth-ABC posterior), then the estimator by reference to 3.5, then the
  assumptions.
- Replace the inline lists of implementation departures (lines 149 and 196) by their four
  categories and a pointer to Appendix A.1. Move `L^{(2)}_epsilon` to the appendix.

### 4.5 Section 5.1 Benchmarks (net 0)
- Open with what the suite must cover: a reference posterior (calibration), a known runtime law
  (barrier prediction), a real simulator of the heterogeneous class.
- Cellular Potts paragraph in this order: the class and why it is the target regime (parameters
  set the cell count, cell count sets both the summaries and the runtime); the model and the
  implementation we run; the inference question (division rate and target volume from cell count
  and cluster radius) and one sentence on identifiability; why one evaluation averages four
  simulations; why 50^3 and 80^3 are two runtime regimes; synthetic observations with known
  truth. The earlier-parameterisation sentence (rule 7). Prior ranges, screening ratios and the
  feature list stay in Appendix E with one pointer.

### 4.6 Section 5.3 Metrics (+120 words)
- State the predictor's domain explicitly (simulation time dominates the collective's latency;
  the timing sample is longer than the runtime tail), so "outside the domain" in 6.1 and the
  open markers in Fig. 2 have a referent.
- Define per-simulation efficiency (direction stated), weighted ESS, and spell out
  simulation-based calibration before the benchmark table uses "SBC".
- The starting-bandwidth rule ("about a fifth of the prior-predictive median discrepancy") moves
  here from 6.4; 6.4 keeps its measured effect.

### 4.7 Section 6 Results (net about -550 words, one table fewer)
- **6.1 (C1)** becomes three paragraphs with topic sentences: (i) the prediction holds: Fig. 2c,
  the summary statistic (median ratio 0.996, range 0.84-1.19) and the three out-of-domain cases
  in one sentence each; (ii) what the twin does not isolate on Cellular Potts (its simulations ran
  longer; pyABC's per-generation overhead) and why 2.0x is the number quoted; (iii) against
  pyABC, pointing at the mechanism figure. The per-spread and per-worker-count numbers stay in
  Tables twin-hetero and twin-cpm (appendix) and are not repeated. The 50^3 / 80^3 bookkeeping
  collapses under rule 7.
- **Table 2 (twin, straggler)** moves to Appendix F. Its content is already the straggler points
  of Fig. 2c; the "barrier every W / every 112" rows and the re-reported `W_1` block are
  appendix-level detail. 6.1 keeps one sentence on the straggler result.
- **6.3 (C3)**: rule + mechanism + placement caveat in about 180 words; the four throughput
  ratios are in Fig. 4 and Table 3 and are not listed again.
- **6.4 (C4)**: the paragraph "The schedule must move" (the twin's transient on the cheap
  simulator) moves to Appendix F with Table 2. "The starting bandwidth" and "The archive size"
  stay, each cut to claim + the one decisive number + pointer.

### 4.8 Discussion and Conclusion (net about -400 words)
- Discussion: the opening paragraph loses its number list (it repeats Table 4 and the abstract).
  Each of the four practitioner rules becomes condition, recommendation, pointer (about 40 words
  each). The headroom paragraph stays.
- Conclusion: one systems conclusion, one statistical conclusion, two conditions of use; about
  120 words; no numbers beyond the two headline ratios.

### 4.9 Figures
**Fig. 1** (`latex/figure-drafts/tikz/a10_body.tikz`, panels from `py/a8_algorithm_panels.py`,
`py/a9_pmc_panels.py`)
- Shared unit-box frame drawn in every panel; both rows at a comparable stage or stage named in
  the row titles.
- Step numbers separated from step names ("step 4: proposal"); b6 label "current proposal and the
  S stored snapshots"; snapshot accumulation visible (faded earlier contours in b4) if it fits.
- Row (a) title "generation-based ABC-PMC, schematic"; caption disclaimer dropped.
- Vocabulary equals the terminology sheet: history, archive, bandwidth, proposal, candidate,
  stored weight; no tau, no "stamp".
- Caption 80-120 words: the comparison, the barrier, the arrival-driven loop, the essential
  encodings only. The glyph key becomes a figure note in Appendix B.

**Fig. 2** (`experiments/scripts/make_barrier_fig.py`)
- Panel (c) alone becomes Fig. 2, square, at about 0.55 linewidth, one workload legend, the
  out-of-domain markers labelled in the panel. Panels (a) and (b) go to Appendix F next to Tables
  twin-hetero and twin-cpm, not to a new main-text figure: their numbers are quoted in 6.1
  anyway, and this keeps the main-body float count flat. This reverses the combination of commit
  7393064 on purpose.
- Captions identify panels and encodings only; numbers stay in the prose. 80-130 words each.

**All other captions**: the Fig. 3 (scaling), Table 3, Fig. 5 (recovery) and Table 4 captions
each carry a sentence of interpretation that belongs in the prose; cut them to data and
encodings (about -150 words together).

## 5. Work order

| Step | Work | Depends on | Effort |
|---:|---|---|---:|
| 1 | Terminology sheet as a grep-and-replace list; rename `w_n`; weighting names (rule 6); hyperparameter table in Appendix B | - | 2 h |
| 2 | New 3.1 overview; Fig. 1 moved under the Section 3 heading; short caption; step numbers wired into 3.2-3.6 and Algorithm 1 | 1 | 3 h |
| 3 | Reorder and rewrite 3.2-3.6; move the estimator and denominator into 3.5; push details to Appendix B; symbol diet (rule 5) | 2 | 5 h |
| 4 | Section 4: arbitrary m, plain-words opening, departures by category, implementation remark | 3 | 2 h |
| 5 | Background: three paragraphs and the bridging sentence | 1 | 2 h |
| 6 | 5.1 Cellular Potts rewrite and the one-configuration sentence; 5.3 predictor domain, efficiency, ESS, SBC, bandwidth rule | 5 | 2 h |
| 7 | Results cuts: 6.1 three paragraphs, Table 2 and "The schedule must move" to Appendix F, 6.3 and 6.4 trimmed; rules 1-3 and 7-8 applied | 6 | 3 h |
| 8 | Discussion and Conclusion cuts; abstract, intro paragraph 2, contributions | 2, 5, 7 | 2 h |
| 9 | Fig. 1 TikZ and panel scripts; both schemes | 1 | 4 h |
| 10 | Fig. 2 split; (a),(b) to Appendix F; caption trims on Figs. 3, 5 and Tables 3, 4; both schemes | - | 2.5 h |
| 11 | Compile; undefined references; first-use-after-definition grep for every term in the sheet; page and word count recorded | 1-10 | 1.5 h |
| 12 | Prose pass with the `prose-pipeline` skill, seeded with Plan B's pattern list | 11 | 1 day |

Steps 5, 9 and 10 are independent of 2-4 and can run in parallel. Step 12 runs last. Total
before the prose pass: about 29 hours.

## 6. Page budget

| Change | Words |
|---|---:|
| Background, three paragraphs | +250 |
| 5.3 definitions (predictor domain, efficiency, ESS, SBC) | +120 |
| Section 4 plain-words opening | +40 |
| Section 3 (new 3.1 replaces old 3.1; details to Appendix B) | -100 |
| Section 4 departure lists and `L^{(2)}` | -60 |
| Abstract and contributions | -170 |
| 6.1 three paragraphs, Table 2 out, 6.3 and 6.4 trims | -550 |
| Discussion and Conclusion | -400 |
| Captions (Figs. 1, 2, 3, 5; Tables 3, 4) | -350 |
| **Net** | **about -1,200** |

Floats in the main body: 12 today; 11 after (Table 2 to the appendix; Fig. 2's (a),(b) to the
appendix; no new main-text figure). Expected main body: 15-16 pages instead of 17. The appendix
grows by Table 2, two panels and one paragraph, which does not count toward the TMLR
recommendation.

## 7. Verification and bookkeeping
- `latexmk` builds clean; zero `\todo`; every `\ref` resolves.
- Each term in the terminology sheet is defined at or before its first use (scripted grep).
- Main-body page and word count before and after in each commit message; the budget above is
  the target, not a hope.
- Number density check after step 7: no paragraph in Section 6 with more than six numbers, none
  repeating a table cell.
- Figures regenerated under `ABC_FIG_SCHEME=okabe` and `kit`; the KIT PDFs feed the thesis.
- One commit per step; Overleaf push after step 11 and after 12.

## 8. Execution record (2026-10-06, same day)

Steps 1-11 executed on branch `campaign-tooling`, one commit per step group
(baseline 94f8a29; Section 3/4 fac09f8; Background/Section 5 363085f; figures 78384b6;
Results 5058b02; abstract/intro/Discussion/Conclusion/captions/terminology: last commit).
Step 12, the prose pass, has not been run.

Outcome against the budget:

| | baseline | now |
|---|---:|---:|
| main-body pages (before References) | 17 | 16 |
| main-body words outside floats | 5,356 | 5,766 |
| floats in the main body | 12 | 11 |

The page count fell by one, so the "no inflation" constraint holds, but the word budget
(-1,200) was missed: Background (+280), Section 3 (+330) and Section 5 (+380, the
Cellular Potts motivation and the metric definitions) grew more than planned, and the
Results cut (-390) and Discussion/Conclusion cut (-120) were smaller than planned. The
remaining fat is in 6.4 (calibration paragraph) and the four Discussion rules; the prose
pass can take it.

Deviations from the plan worth knowing:
- The parent weight also feeds the scheduler's ESS search, so it is "used only inside the
  running sampler", not "only to choose parents".
- Lotka-Volterra and the four-dimensional calibration runs use the geometric-decay schedule
  family; the hyperparameter table in Appendix B says so, 3.3 describes the kernel-aware
  rule they all share.
- Fig. 1 row (a) stage is "generation t -> t+1 after 135 simulations" (48 of them a toy
  pilot for epsilon_0); row (b) "after 84 evaluations". The counts were annotated, not
  matched, because the only PMC stage near 84 is a near-prior population.
- Fig. 2's legend sits below the panel (the longest label is wider than the empty corner).
- Three references were added from memory and must be checked against the originals:
  Kursawe, Baker & Fletcher 2018 (J. Theor. Biol. 443:66-81); Lambert et al. 2018
  (J. Math. Biol. 76:1673-1697); Hirashima, Rens & Merks 2017 (Dev. Growth Differ.
  59(5):329-339).
- Overleaf not pushed (needs the user's token interactively).

## 9. Step 12, the prose pass (run 2026-10-06, same day)

Run with the `prose-pipeline` skill; full record in `reviews/2026-10-06-tmlr-prose/` (config,
ledger, per-round proposals with status and reason, both reviewers' reports per stage). Five
commits on `campaign-tooling`, b4d3ab5..8967514:

| stage | edits applied | reviewers' verdicts |
|---|---:|---|
| humanizer voice pass (academic-humanizer, voice-only brief) | 38 | blinded A/B: both reviewers prefer the edited text by a moderate margin (Claude AI-likeness 3 -> 2, codex 6 -> 2) |
| round 1 (50 proposals, batch triage, all items) | 35 (2 reverted, 2 reworded after split verdicts) | AI-likeness Claude 4, codex 8 before the round |
| round 2 (38 proposals, batch triage, all items) | 29 (2 reverted) | Claude 3, codex 5 before the round; both verifiers advise stopping |

Kept by author decision and fed to the reviewers as rejected: the two short lines
"Identifiability does." and "The posterior does not follow.", the four italic Discussion rule
titles, "the method a practitioner would run" (5.2), the Appendix F "lifts" sentence, the
"Measured, in part." heading, "binds"/"sits at the threshold"/"turns on" as policy verbs.
Word count 20,970 -> 20,694 (texcount-free `wc -w` on the .tex); 41 pages. Open: one
read-through of Appendix A and 3.1 for rhythm where the splits left runs of short sentences.
