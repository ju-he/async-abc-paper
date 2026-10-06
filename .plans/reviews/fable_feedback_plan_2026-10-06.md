# Plan A (Fable): addressing the TMLR draft feedback of 2026-10-06

Source: `.plans/reviews/tmlr_draft_feedback_2026-10-06.md`. Manuscript: `latex/tmlr/tmlr-article.tex`
(main body lines 55-427). This plan covers structure, motivation and figures. The prose pass
(AI-sounding language) is a separate, later step and is only sequenced here.

## 0. Diagnosis

The reviewer's two complaints have one root cause: the manuscript was written and then compressed
(density passes of 2026-09-21) from the inside out. Every section opens with the most specific
true statement about its topic (the history tuple with five fields, `tol_init`, the ring buffer,
`m <= S+1 = 21`) and never states the plain idea first. Terms coined for internal use (twin,
snapshot, stamp, fidelity ratio, retroactive estimator, production configuration) leaked into the
abstract and the introduction. Fig. 1 was built as the method's own graphical abstract but sits at
the end of Section 3, after every detail it was meant to preview, with a caption that re-explains
the encoding instead of the idea.

Concretely, in order of first appearance, these terms are used before they are motivated or
defined:

| Term / symbol | First use | Should be introduced |
|---|---|---|
| straggler factor | abstract | fine (defined in the intro); keep |
| "streaming form of AMIS", "fidelity ratio", "kernel-matched baseline", "barrierized twin", "archive size", "starting bandwidth", "prior-predictive discrepancy scale" | abstract | abstract must use plain words; at most two coined terms (barrier, straggler factor) |
| "pure/replayable function of the evaluated history" | intro para 2, contribution 1 | say *why* in the same sentence (no shared mutable state, so workers need not synchronise; a crash is recovered by replay) |
| island model of Propulate | intro para 2 | one clause: "a framework in which every worker proposes from the history it has received, with no collective step" |
| twin, kernel-matched, damped-adaptation variants, fidelity ratio | contributions | describe in words there; name them in Sections 4-5 |
| scheduler, `tol_init`, "stamped" bandwidth, transient | 3.2 | the schedule has never been motivated: say what it is for (tighten the kernel as the archive improves, at a rate the archive can follow) before saying how it is reconstructed |
| `perturbation_scale`, Cholesky | 3.3 | appendix hyperparameter table |
| snapshot, ring buffer, `amis_interval`, parent selection, online ESS diagnostic | 3.4 | the *reason* for snapshots (the proposal moves with every arrival, so no single proposal is "the one the particle came from") must come first; buffer mechanics to the appendix |
| retroactive estimator, defensive prior component, \eqref{eq:posterior-estimator} | 3.4 "Three weights" paragraph | the estimator is referenced in Section 3 but defined in Section 4 (line 162): move the definition into Section 3 |
| capped in-box rejection, underflow redraws, product-of-marginals truncation constant, bootstrap share, draw-proportional deterministic mixture | 4 | first paragraph of Section 4 lists them as if known; introduce as "four implementation departures from the ideal, listed in Appendix B" and name them only there |
| `m <= S+1 = 21` | 4, line 160 | the 21 is the experimental S=20; general statement is `m <= S+1` |
| cellsInSilico, screening experiment, earlier configuration | 5.1 | motivate the model class before the software; screening and the earlier configuration belong in Appendix E with one pointer sentence |
| "the model's stated domain" | 6.1 | the domain of the barrier-cost predictor is never stated as such in 5.3; add two sentences there |

## 1. Section-by-section changes

### 1.1 Abstract and introduction (small)
- Rewrite the abstract so that a reader who knows ABC-SMC understands it without the paper:
  barrier, straggler factor, "update after every arrival", "weight each particle against the
  mixture of proposals it could have come from", "estimate from everything simulated". Drop
  "fidelity ratio", "barrierized twin", "kernel-matched", "archive size" from the abstract; keep
  the four numbers that matter (cost range, 2.1-2.4x, 4.1x, few ms).
- Intro paragraph 2: add the one clause each for *why* the history-only design (asynchrony
  without shared state) and what the island model is. Strengthen the tissue-simulator sentence
  into a named problem class (see 1.5).
- Contributions: rewrite C1-C4 in words; keep the forward references.

### 1.2 Background (+150-250 words)
- Add a short paragraph on the simulator class: cell-based and agent-based spatial stochastic
  models (Cellular Potts is the canonical cell-based model, Graner-Glazier-Hogeweg), where the
  likelihood is intractable, one simulation costs seconds to hours, and cost varies by orders of
  magnitude across the prior because it tracks the number of agents. State that ABC is the
  standard route for these and cite two or three applications. This plants the Cellular Potts
  benchmark so that Section 5 can refer back.
- Give the AMIS balance heuristic as one displayed equation (weight against the mixture of all
  proposals used so far) and the single-proposal PMC weight beside it, so that Section 3.4 only
  has to say "the streaming version of this".
- One sentence making "generation", "population", "barrier" operational in pyABC terms.

### 1.3 Section 3 Method: new 3.1 and reorder (the reviewer's [not ok])

Target outline (reviewer's numbering: new 3.1, old 3.1-3.5 become 3.2-3.6):

| New | Content | Level | Pushed to appendix |
|---|---|---|---|
| 3.1 Overview (new) | Fig. 1 here. One paragraph: what a generation does in PMC, and what replaces it: a sliding archive of the k best particles defines the proposal; the proposal is rebuilt at every arrival; each particle is weighted against the mixture of proposals actually used; the posterior is read off the full history afterwards. Second paragraph: the one design rule (every quantity is recomputed from the evaluated history), why (workers never synchronise; restart = replay), and its one exception (the online snapshot buffer). Third paragraph: the "three weights, one posterior" orientation, moved up from 3.4, in words. | plain words, no symbols except k, n, epsilon | Propulate propagator interface, scheduler throttles |
| 3.2 History and archive | H_n (drop tau_i and w_i from the tuple or explain both in one clause), A_n = top-k, proposal mixture \eqref{eq:steady-state-proposal}, Sigma_n from the weighted archive. | formal | Cholesky, `perturbation_scale` name |
| 3.3 Bandwidth | What the schedule is for; the invariant (non-increasing, reconstructed as the running minimum of stamped bandwidths); the target ("the bandwidth at which 2k particles lie inside"); one sentence that the starting value governs expensive runs (forward ref to 6.4). | formal, short | "once per k calls", "at most halve", `tol_init` |
| 3.4 Weighting against the proposals used | The reason (proposal moves every call); online weight \eqref{eq:amis-weight} with the S-snapshot buffer; what it is used for and not used for. | formal | ring-buffer mechanics, `amis_interval`, S=0 sweep numbers (-0.088) |
| 3.5 Posterior estimate (new home) | \eqref{eq:posterior-estimator} and the m-snapshot denominator \eqref{eq:snapshot-denominator} move here from Section 4, with the defensive prior component explained in one sentence; O(nk) post-hoc pass, excluded from throughput. Section 4 then refers back. | formal | why m is fixed (already in App. A) |
| 3.6 Update loop | Algorithm 1 and one paragraph on execution with W workers (what "asynchronous" means operationally; the island model in one sentence). | procedural | Propulate details |

- Fig. 1 caption target: <= 90 words, says what each row is and what the two differences are
  (no population and no generation; weights against all used proposals). The encoding legend
  (point sizes, shading, ellipses, insets) moves into a small key drawn in the figure, or into
  a figure note in Appendix B.
- Drop code names (`tol_init`, `perturbation_scale`, `amis_interval`) from the main text; keep
  the symbols and add a hyperparameter table (symbol, code name, value, where set) to Appendix B.

### 1.4 Section 4 Theory
- Line 160: `m\le S{+}1=21` -> `m\le S{+}1` with "(S=20 in every experiment, Section 5)" in
  the experimental design, not in the theorem setup. Check `\delta=0.5/(m+1)` is presented as the
  implementation's choice of a fixed delta, not as part of the theorem.
- First two paragraphs: after moving the estimator into 3.5, the section can open with the
  question it answers ("is the denominator the density the sample was drawn from?") in plain
  words, then the assumptions. Replace the inline list of implementation departures by one
  sentence and a pointer.

### 1.5 Section 5.1 Benchmarks: motivate Cellular Potts
- Open the subsection with what the suite needs to cover: (i) targets with a reference
  posterior for calibration, (ii) a known runtime law for the barrier prediction, (iii) a real
  simulator of the heterogeneous class the paper is about.
- Cellular Potts paragraph, new order: the problem class (cell-based tissue simulation; cost
  tracks cell count, which the parameters set, hence runtime spread by construction), the model
  (Cellular Potts; cellsInSilico is the implementation we run), the inference problem (division
  rate and target volume against cell count and cluster radius), and why it is a calibration
  instrument (synthetic truth). Screening, the "earlier configuration" and the four-replicate
  averaging stay in Appendix E with one pointer sentence.
- Cross-reference the Background paragraph of 1.2.

### 1.6 Section 5.3 Metrics
- State the domain of the barrier-cost predictor explicitly (durations longer than the
  collective's latency; a timing sample longer than the runtime tail), so that 6.1 can say
  "outside the domain" and the open markers in Fig. 2c are defined before use.

### 1.7 Figures

**Fig. 1 (algorithm strip, `latex/figure-drafts/tikz/a10_algorithm_strip.tex`)**
- Zoom: panels a4-a6 and b4-b6 are drawn at different scales from a1-a3/b1-b3 and from each
  other, with no indication. Use one frame per column across both rows, or draw an explicit
  zoom box in column 3 that both rows share.
- "4 proposal": column 4 shows one mixture in each row, but b6 shows several stored proposals.
  Either show the accumulation (b4: current proposal drawn on top of the faded earlier ones,
  so b6 follows) or label b6 "the S stored proposals and the current one".
- Vocabulary in the figure must equal the vocabulary of new 3.1: stored bandwidth, archive,
  stored proposals, history. No tau, no "epsilon stamp" unless 3.2 uses the word.
- Caption shortened as in 1.3.

**Fig. 2 (barrier, `fig_barrier.pdf`)**
- Split. Fig. 2 becomes panel (c) alone, the C1 measurement, with its own legend, placed with
  6.1's "Panel (c) is the paper's central measurement". Panels (a) and (b), which share one
  legend and one comparator, become a separate figure "Against pyABC under injected
  heterogeneity" placed at the "Against pyABC" paragraph of 6.1 (or in Appendix F if space
  matters). This reverses the combination done in commit 7393064, deliberately: the reviewer
  reads the combined figure as three unrelated panels.
- Caption of the new Fig. 2: <= 80 words, no numbers that the text repeats. Move the
  configuration details (worker counts, utilization ratio, earlier configuration) into the
  legend labels or a table footnote.

### 1.8 Section 6 and later: no structural change
Rename nothing. After the Section 3 rewrite, re-read 6.1-6.4 for terms whose definition moved
(snapshot, m, S, twin) and fix the cross-references.

## 2. Work order

| Step | What | Depends on | Effort |
|---|---|---|---|
| 1 | Write the new 3.1 overview and the Fig. 1 caption; move Fig. 1 there | - | 3 h |
| 2 | Reorder/merge 3.2-3.6 as in the table; move \eqref{eq:posterior-estimator} and \eqref{eq:snapshot-denominator} into 3.5; push details to Appendix B (new hyperparameter table) | 1 | 3 h |
| 3 | Section 4: fix `=21`, re-open with the question, pointer for departures | 2 | 30 min |
| 4 | Background: simulator class paragraph, AMIS equation, pyABC vocabulary sentence | - | 1.5 h |
| 5 | Section 5.1 rewrite of the Cellular Potts paragraph; 5.3 predictor domain | 4 | 1.5 h |
| 6 | Abstract, intro paragraph 2, contributions in plain words | 1, 4 | 1.5 h |
| 7 | Fig. 1 TikZ: common frames, proposal accumulation, vocabulary | 1 | 3-4 h |
| 8 | Fig. 2 split: plotting script, two captions, cross-references in 6.1 | - | 2 h |
| 9 | Compile, check every \ref, check main-body word count (target: unchanged +/- 300) | 1-8 | 1 h |
| 10 | Prose pass (prose-pipeline skill) on the restructured text; not before | 9 | 1 day |

Steps 4, 7 and 8 are independent of 1-3 and can run in parallel. Step 10 must come last;
running the prose pipeline on the current text would be redone after the restructure.

## 3. Verification
- `latexmk` builds without undefined references; `\todo` count unchanged (zero).
- Every term in the table of Section 0 has its first use at or after its definition (grep check).
- Main body word count before/after recorded in the commit message.
- Fig. 1 and Fig. 2 regenerated under both `ABC_FIG_SCHEME` values.
- One commit per step on branch `campaign-tooling` (or a `tmlr-feedback` branch), then Overleaf push.
