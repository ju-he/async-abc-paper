# Figure drafts (2026-09-23)

Drafts for three graphical-abstract concepts (GA1–GA3), five additional
illustrations (A1–A5) and three algorithm illustrations (A8 ours, A9 ABC-PMC, A10 both as a strip; A6 and A7 from the
same discussion are not drafted, see the end) proposed for the paper. TikZ for the diagrams, small python scripts for the
plot components (same palette and rcParams as the paper figures, via
`experiments/async_abc/plotting/paper_style.py`). Only A10 is wired into the manuscript (as `figures/fig_algorithm_strip.pdf`,
Fig. algorithm-strip in §3.5 of both versions, caption in the Springer source);
every other file is a starting point to be judged, not a final figure.

Build everything (python components first, then TikZ, then PNG previews):

    ./build.sh            # or ./build.sh py | ./build.sh tikz

Outputs land in `out/` as PDF + PNG (`out/a8_panel_1..6.pdf`, `a8s_`, `a9_`,
`a9s_` and `out/ga2_predictor_panel.pdf` are intermediate panels included by
the TikZ wrappers). `py/fetch_records.sh` pulls the per-particle subsets that `a2`
needs from the cluster scratch mount into `data/` (already done once, marker
`data/.fetch_done`; rerun only if `data/` is missing).

## Graphical abstract concepts

| file | concept | notes |
|---|---|---|
| `tikz/ga1_barrier_gantt.tex` | **GA1 The barrier as idle time.** Two worker timelines with the same multiset of simulation durations: generation-based on top (hatched idle, dashed barriers, proposal updated only at barriers), generation-free below (packed lanes, proposal updated at every arrival). | The 32 / 76 completed counts are computed inside the picture from the drawn durations. The drawn workload's straggler factor is 2.3; the count ratio (2.4) differs slightly because of window-edge effects. Pure cartoon, no data. |
| `tikz/ga2_predictor_ga.tex` + `py/ga2_predictor_panel.py` | **GA2 Prediction meets measurement.** Left: asynchronous run yields the timing record; barrierized twin is the same sampler with a barrier drawn in. Right: the predictor scatter stripped to three marker types, medians, equality line. | Scatter data = vendored `fig_predictor/predictor_rows.csv`; the three out-of-domain points are dropped; the 50^3 and 80^3 tissue points are merged into one series. |
| `tikz/ga3_stream_vs_generation.tex` | **GA3 The stream replaces the generation.** Left: three populations separated by barriers. Right: arrivals on a timeline, filled dots = in the top-k archive, one tick per arrival = proposal rebuilt, five of the proposals drawn narrowing, and the AMIS reweighting formula. | Cartoon. Dot positions and curve widths are hand-set. |

Recommendation from the concept discussion: GA1 as the main panel with the GA2
scatter as an inset.

## Additional illustrations

| file | illustration | data / caveats |
|---|---|---|
| `tikz/a1_method_overview.tex` | **A1 Method overview** (candidate Fig. 1, §3.1): workers → append-only history log → propagator call (four steps of Alg. 1) → one candidate back; second branch: replay the log → reported estimator. | Cartoon. |
| `py/a2_cumulative_completions.py` | **A2 Cumulative completions vs wall clock**, asynchronous arm against the barrierized twin. (a) Cellular Potts 50^3, W = 48, k = 100, replicate 0, first 200 s. (b) Persistent straggler 20x, W = 16, replicate 0, first 40 s, log axis. | Records from scratch: twin = `cpmtwin_20260729/scaling_cpm_twin` and `twin2_20260729/straggler_twin_f20`; async = `rerun_20260707/scaling_cpm` (w48 k100) and `twin2_20260729/straggler_async_wall` (20x). **Caveats:** (i) in (a) the twin's simulations themselves ran longer than the asynchronous arm's (median 16 s vs 5.2 s in these records, the contention the paper notes in §6.1), so the raw rate gap in the window (1.8x) is *not* the barrier's cost alone; Table twin-cpm attributes about 1.2x to the barrier. The flat tread annotated at 142–169 s is one generation waiting for its slowest simulation. (ii) In (b) the asynchronous arm's early rate (4.7k/s in the first 40 s) is above its whole-run rate because the per-call cost grows with the history; the paper's 402x is a whole-run active-wall-time ratio (Table twin). The figure shows the mechanism, the tables carry the numbers. The `generation` column in the twin records is a per-particle index, not the barrier generation, so it is not used. |
| `py/a3_straggler_factor.py` | **A3 The straggler factor** E[max_{i≤W} T_i]/μ against W: lognormal σ ∈ {0.1, 0.2, 0.5, 1, 2} (exact via the quantile integral) and the measured asynchronous per-simulation durations on Cellular Potts 50^3 (staged `cellular_potts` campaign, earlier configuration, n = 170k, CV 0.09; exact order-statistic sum). W = 16, 48, 384 marked. | Values: σ=1 gives 4.2x / 6.4x / 12.6x at W = 16 / 48 / 384; σ=2 gives 9.5x / 20x / 71x; measured CPM 1.20x at 48 and 1.25x at 384 (lighter tail than a lognormal of the same CV). Lotka–Volterra and g-and-k durations are not in the local staging mirror, so they are not drawn. |
| `tikz/a4_three_weights.tex` | **A4 Three weightings, one reported** (§3.4): archive → adaptation weights → next proposal → candidate → proposal-time weight → appended to the log; whole history → reported weights → reported posterior. Tags mark which are never reported. | Cartoon. |
| `py/a5_fidelity_ratio.py` | **A5 The fidelity ratio made visible** (Appendix A). Rebuilds the 1-D Gaussian-mean history of `diag_denominator_mismatch.py` (seed 20260730, n = 12000, k = 100; ~20 s). (a) prior, a few snapshot proposals, the reported denominator (m = 21, prior floor) and the draw-mixture reference (m = 401, true ν_n) on a θ grid. (b) r(θ) = q̄*/q̄ with the reported posterior's shape behind it. | Reproduces the paper's number: TV(reported, reference) = 0.26 %. r is within 5 % of one where the posterior has mass and drops to about 0.22 in the tails, where the prior floor (0.0227 against the true bootstrap share 0.0083) and the early wide snapshots inflate the reported denominator. E_π̂|r−1| = 0.042. |

## Algorithm illustrations (as visual as possible, one formula per step)

All three are drawn from `py/_toy2d.py`: a unit-box prior, a tilted
elliptical discrepancy around a hidden truth, and two *illustrative
miniature runs* on it. `run_async` follows Algorithm 1 (bootstrap prior draws
until k particles exist, monotone bandwidth, top-k archive, adaptation
weights w_j·K_ε(ρ_j) with the *stored* proposal-time weight, Σ = s·Cov, one
parent-and-perturb draw with in-box retry, balance-heuristic weight over a
snapshot ring buffer, a snapshot pushed every `amis_interval` calls; every
particle stores (θ, ρ, ε, τ, w)). `run_pmc` is ABC-PMC with a quantile
schedule (Del Moral et al. 2012; Lenormand et al. 2013; pyABC's default) and
a hard threshold (Beaumont et al. 2009): propose, simulate and test one
candidate at a time until N are accepted, Σ = 2·Cov, weights π/q_t against
the one previous proposal, normalized; run without discrepancy noise so the
drawn acceptance region is exact. Every quantity drawn is a state the
respective algorithm produces at that call; only the illustrated draw's
perturbation z is fixed. The miniature simplifies the implementation in ways
that do not show in the panels (listed in the module docstring: the
scheduler jumps to the 2k-particle bandwidth instead of the ESS-retention
bisection, no ε_0, plain weighted covariance without bias correction, no box
truncation of the mixture, no underflow redraws). Toy sizes: k = N = 12, 84
calls for ours (18 snapshots), three PMC generations (M = 48 draws for the
last one, acceptance 0.25). Panels are generated at the size they are
included at (2.4 cm for the rings, 1.6 cm for the strip), so fonts print at
nominal size; both figure widths are within the 372 pt text width.

Reviewed 2026-09-23 by codex (GPT-5.6-Sol) and a fresh Claude Fable subagent,
and the strip again by codex on 2026-09-24
(`.plans/reviews/digest-figures-a8a9-2026-09-23.md`); the current state
implements both rounds' essential and should-fix items. The round-2 finding on
the bootstrap bandwidth stamp was also applied to the paper (§3.2, one
sentence: bootstrap draws store no bandwidth; the running minimum is over the
stamped particles), and the TMLR twin regenerated.

| file | illustration | notes |
|---|---|---|
| `tikz/a8_algorithm_ring.tex` + `py/a8_algorithm_panels.py` | **A8 Ours: the single-arrival update as a ring of six panels.** (1) history shaded by discrepancy, (2) the k best; inset: the smooth kernel K_ε(ρ) over ρ with the archive's ρ_j as ticks (no acceptance boundary in θ-space), (3) archive sized by W̃_j ∝ w_j K_ε(ρ_j) with Σ_n, (4) q_n as contours, (5) parent J ~ W̃ perturbed by L_n z to θ*, (6) θ* against q_n and the stored snapshots. A workers node carries the asynchrony: the closing arrow hands θ* to a free worker, two gray arrows bring arrivals from all workers into the history in any order, and the dashed arrow is the bootstrap (a prior draw with w* = 1 straight to a worker). | One title and one discriminating formula per step next to its panel; definitions, the append tuple (θ*, ρ*, ε_n, τ* = n, w*), the snapshot push rule, the Alg. 1 line map and the glyph legend in a caption stand-in beneath (goes into the caption in the paper). Step 6 is titled "proposal-time weight, stored, never reported" and the caption points to the reported estimator. Labels of the diagonal panels sit above/below them, which is what brings the ring to 367 pt. |
| `tikz/a9_pmc_ring.tex` + `py/a9_pmc_panels.py` | **A9 ABC-PMC in the same six positions**, vermillion. (1) population t with ρ and ω^(t), (2) hard threshold ε_{t+1} = quantile of the population's ρ as the acceptance region; inset: the indicator over ρ, (3) same particles, same sizes, Σ_t = 2 Cov_ω, (4) q_t, (5) the propose-simulate-accept loop in progress (first 12 candidates: kept filled, rejected crossed, the black star being simulated), (6) population t+1: the N kept sized by ω^(t+1) ∝ π/q_t, the rest crossed out. The generation dependency sits on the closing arrow (bar: q_{t+1} waits for the N-th acceptance). | Read against A8 position by position: history vs population, smooth kernel vs hard threshold (the two insets), weights re-formed each call vs carried over, one draw vs propose-simulate-accept until N kept, weight before simulation against the mixture vs after acceptance against one proposal, barrier on the generation transition vs none. The caption stand-in states that this is textbook PMC and not the paper's matched-kernel pyABC baseline (which discards latecomers rather than waiting). |
| `tikz/a10_body.tikz` + wrappers `a10_algorithm_strip.tex`, `a10_HFG`, `a10_HBN`, `a10_HBG`, `a10_VFN`, `a10_VFG`, `a10_VBN`, `a10_VBG`, `a10_MFN`, `a10_MFG`, `a10_MBN`, `a10_MBG` | **A10 The comparison, eight layout variants from one body.** Four switches: H/V/M (M = horizontal with the step spine 1-6 between the rows, PMC above with labels and rail above, ours below) (two rows of six columns, or steps 1-6 as a centre spine with PMC left and ours right: rail, text, panel, spine, panel, text, rail), F/B (formulas or plain-language bullets beside each panel), N/G (no backgrounds or alternating gray bands per step). Each method has one return arrow (the worker-pool box and the triple rail are gone); the bootstrap rule is a text note (no arrow); row (a) carries the barrier bar on its return, at the bottom in the vertical layout. `a10_algorithm_strip.tex` is the wrapper integrated in the manuscripts (fig:algorithm-strip, caption in the Springer source); since 2026-10-02 it carries the MBG switches (step spine between the rows, bullets, gray bands). | All eight fit the 372 pt text width at 7 pt labels (H: 368 pt wide, 284-314 pt tall; V: 362-368 pt wide, 428 pt tall). The 2026-10-02 pass answered the complaints about the detached 1-6 headers (V attaches them as the spine) and the awkwardly placed, not very explanatory formulas (B variants). Previews `out/a10_*.png`. |

## Not drafted (from the same discussion)

* A6 bandwidth transient on Cellular Potts (ε_sched vs calls per rank for ε_0 = 10 vs 0.1) — needs the per-record `proposal_tolerance` column from the production run.
* A7 tissue renders + runtime histogram — needs a NAStJA rendering step; no renders exist in the repo.
