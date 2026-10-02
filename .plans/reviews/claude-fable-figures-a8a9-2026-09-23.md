# Figure review — Claude Fable 5.1 (fresh subagent), 2026-09-23

Target: the two algorithm-illustration drafts `latex/figure-drafts/tikz/a8_algorithm_ring.tex` (ours) and `a9_pmc_ring.tex` (textbook ABC-PMC), with their panel scripts and `_toy2d.py`, at the state of 2026-09-23 (uncommitted, branch `campaign-tooling`). Reviewer had read access to the repository and the propulate checkout and was given the brief in `.plans/reviews/figure-brief-a8a9-2026-09-23.md` with 220-dpi renders attached.

---

# Review of the two algorithm-illustration drafts (A8 "ours", A9 ABC-PMC)

Reviewed: `latex/figure-drafts/tikz/a8_algorithm_ring.tex`, `a9_pmc_ring.tex`, `a9_rings_side_by_side.tex`, `py/a8_algorithm_panels.py`, `py/a9_pmc_panels.py`, `py/_toy2d.py`, the README section, the 220-dpi renders, the paper (`sn-article.tex` §2-§5.2, App. C) and the propagator (`propulate/propagators/abcpmc.py`, `__call__` lines 981-1186, `_build_proposal` 856-932, cache 265-367, scheduler 1431-1720).

## 1. Accuracy

### A8 (ours)

1. **Panel 2, the dashed "ε_n" ellipse** (`a8_algorithm_panels.py` lines 44-48: `T.ellipse(T.MU, T.C_TARGET, nsd=eps_n/RHO_SCALE, ls=(0,(3,2)))`). *Shows:* the bandwidth as a closed dashed boundary around the archive, the same glyph A9 uses for its hard threshold. *Paper/code:* §3.3 "every archive member contributes continuously through its kernel weight"; "smooth kernels remove the prior-vs-archive discontinuity present in hard-threshold ABC-PMC"; Table method-comparison row "hard threshold vs smooth kernel". In the code the archive is top-k by loss with no threshold (`get_archive(float("inf"), k)`) and ε enters only through `log_kernel`. The figure's own visual language turns the one distinction the text insists on into "same thing, different colour". The ellipse is also drawn from the hidden truth (`MU`, `C_TARGET`), a quantity the algorithm never has. **Severity: blocking for the side-by-side use; should fix for A8 alone.**

2. **Bootstrap chord 1→5** (`\draw[->,dashed] (p1) -- (p5)`). *Shows:* the bootstrap draw entering panel 5 ("one draw"), from which the blue arrow continues to 6 ("weight") and only then to the simulate arrow. *Paper/code:* Alg. 1 lines 2-3; `__call__` lines 1030-1035 return a uniform draw with `weight = 1.0` immediately: no perturbation, no AMIS denominator, no snapshot-counter increment. The label says w* = 1, the routing says it is weighted at step 6. **Should fix:** end the chord on the 6→1 arrow (or a small "→ simulate" node), not on panel 5.

3. **Step-6 title "weight, before simulating" and tag w*.** *Shows:* an unqualified weight, and the ring has no other weight. *Paper:* §3.4 "Three weights, one of them reported": w* "never enters the reported posterior"; the reported weights are eq. (posterior-estimator) with the prior-floored denominator q̄_n. A first-time reader will take w* as the posterior weight. **Should fix:** retitle "proposal-time weight (stored, never reported)" and one caption sentence pointing to §4.

4. **Closing arrow "simulate ρ*∼p(ρ|θ*); append".** *Shows:* one sequential actor that proposes, simulates, appends. *Paper:* §3.1 the propagator "hands the whole evaluated history to each call and takes one candidate back"; the simulation is done by a worker, and between this call and the next, other workers' results land in H_n out of order. No second worker, no external arrival is visible anywhere in A8, so the ring reads as "ABC-PMC with N = 1" (a fair description of the per-call update, not of what makes it generation-free). **Should fix:** two or three short gray arrows into panel 1 "arrivals from other workers, any order"; closing arrow "θ* to a free worker; ρ* arrives later".

5. **Step badges 1-6 vs Algorithm 1 line numbers.** Figure step 3 = Alg. lines 5-6, figure step 6 = Alg. line 7, etc. (the README admits it). Readers will try to match them. **Minor:** give the map in the caption or drop the badges.

6. **Chord bypasses step 2.** Alg. 1 line 1 computes ε_n before the `|H_n| < k` branch; the code computes ε_hist before the branch but ε_sched only after. The figure follows the code. **Minor.**

7. **Append tuple (θ*, ρ*, ε_n, n, w*).** Bootstrap draws carry no tolerance stamp (`child.tolerance = effective_tol` is line 1156, after the bootstrap return; the toy stores +∞). τ = n is right (τ_i is the proposal-time index). **Minor.**

8. **Notation.** Figure W̃_j, paper W̃_j^{(n)}; Σ_n = s Cov_W̃(A_n) without the App. C jitter (fine); K_{Σ_n}(·−θ_j) vs paper K_Σ(θ−θ_j) (fine); the buffer is written 𝒮 while the paper uses S for its size and 𝒮 for the buffer (fine); the monospace `amis_interval` is a code identifier inside a figure (the paper does use it in §3.4, but in a figure it is a style break). **Minor.**

9. **Panel 1 shading (light = far) has no key.** **Minor.**

10. **Panel 6 shows every third of 18 snapshots**, i.e. the whole proposal history; the online buffer holds the S = 20 most recent, which at production scale are all late, narrow proposals. The picture is closer to the reported denominator's coverage than to the online buffer. **Minor:** caption "recent snapshots q_s".

### A9 (ABC-PMC)

11. **Attribution.** The tex comment and README call it "textbook ABC-PMC (Beaumont et al. 2009)" with ε_{t+1} = quantile_α{ρ_i}. Beaumont 2009 uses a prescribed decreasing sequence ε_1 > … > ε_T; the α-quantile schedule is Del Moral et al. 2012 / Lenormand et al. 2013 (both cited in §2) and pyABC's default. **Should fix (caption/comment):** "ABC-PMC with the quantile schedule".

12. **Position 5 "M draws … until N are accepted", then the barrier, then position 6 "accept".** *Shows:* all M candidates drawn up front (tag M = 36), then simulated, then filtered. Real PMC, and the toy's own `run_pmc` loop (`while sum(flags) < N: draw, simulate, flag`), interleave draw-simulate-accept until N acceptances; M is known only afterwards. The superimposed picture is legitimate; the label order is not. **Should fix:** "draw and simulate until N are accepted (M draws in all)".

13. **Barrier on 5→6.** Where the barrier bites is before position 1/2 of the *next* generation: ε_{t+2}, Σ_{t+1}, q_{t+1} need the complete P_{t+1} (§2 "the full population is needed before the next proposal can be formed"); accepting and weighting one particle needs only q_t, which is known. **Minor:** move the bar to the closing arrow ("P_{t+1} complete only when the slowest returns") or keep it with "nothing below can start".

14. **Symbol reuse.** ω_i is both the carried weight (position 3, generation t) and the new weight (position 6, generation t+1); the threshold formula uses ρ_i although P_t = {(θ_i, ω_i)} carries no ρ. **Minor:** ω^{(t)}, ω^{(t+1)}, P_t = {(θ_i, ρ_i, ω_i)}.

15. **"P_{t+1}" label overlaps the top edge of panel 6** in the render. **Minor layout.**

16. **Panel 6: a few accepted dots sit outside the dashed ε_{t+1} ellipse** because the toy adds N(0, 0.012) noise to ρ (`RHO_NOISE`). A careful reader sees accepted points outside the acceptance region. **Minor:** set `RHO_NOISE = 0` for the PMC panels.

17. **Relation to the paper's baseline.** A9 is hard-threshold, quantile-scheduled PMC. The experimental baseline (§5.2; App. C "Matched pyABC acceptor") is pyABC with a probabilistic acceptor using the *same* smooth kernel and the same ESS-retention bandwidth rule once per generation. Placed in §3, A9 will be read as the baseline. **Should fix:** caption sentence. (Independent of the figure: the paper's Table method-comparison has the same tension, caption "the pyABC baseline used here" against the row "ABC likelihood: hard threshold"; worth reconciling.)

### Does the toy do what the panels claim?

18. **`run_async`: yes**, with these departures, none of which changes what the six panels show. (i) ε_sched = the 2k-th order statistic of ρ (`sched_eps`): the toy jumps straight to the schedule's target; the code runs the ESS-retention bisection with a 0.5 per-search floor, once per k calls, only after k + `additional_needed_inds` = 2k accepted (`_kernel_aware_from_accepted`); §3.2's "tightens at a bounded rate toward the bandwidth at which 2k … lie within" has the rate, the toy does not. (ii) No ε_0: ε_hist falls back to +∞ instead of `tol_init`, so the first archive-phase bandwidth is the largest bootstrap ρ. (iii) Plain weighted covariance vs the code's bias-corrected `weighted_covariance`; jitter 1e-4 vs 1e-9·tr/d + 1e-12·b̄². (iv) No truncation to the box, no in-box mass. (v) No underflow retries, no weight-0 eligibility rule. (vi) `amis_interval = 4` vs default k. Everything the panels depend on matches the code: Gaussian kernel exp(−ρ²/2ε²), top-k archive by ρ, W̃_j ∝ w_j K_ε(ρ_j) with the stored weight, parent from W̃ then θ_J + L z with in-box retry, denominator = mean of current proposal and snapshots (code lines 1146-1148, `logsumexp − log(len(terms))`), bootstrap weight 1, snapshot pushed *after* the weight and the counter incremented only in the archive phase (lines 1181-1184). The panels are computed from the state after call 84 for the update at call 85, which is the state the code would use.

19. **`run_pmc`: yes.** Quantile threshold from the previous population, Σ = 2 Cov_ω, draw until N accepted, ω ∝ 1/q_t (π = 1). Textbook adaptive PMC.

### Legitimately omitted, and whether it matters for a §3 reader

Box truncation and box mass, log-space weights, underflow redraws, weight-0 particles, the jitter, the scheduler's throttle and bisection: no. Two omissions matter: the reported estimator with its prior floor (finding 3: the ring must at least point to it) and the multi-worker, out-of-order arrival (finding 4). One is worth a caption word: the snapshot buffer and the scheduler are per rank ("each rank's scheduler searches once per k of *its own* calls"), so "every amis_interval calls" is per rank.

## 2. Understanding

For a reader meeting the method in §3, A8 helps with the *shape* of one call: the archive, the kernel weights, the mixture and the parent-and-perturb draw become concrete, and "weight before simulating" is a good hook. It does not help with the three things §3 most needs a picture for: why this is generation-free (no other worker, no out-of-order arrival), that everything is recomputed from H_n on every call (the ring implies flow, not recomputation), and that w* is not the reported weight. Most likely misreads, in order: ε_n is a threshold; w* is the posterior weight; bootstrap draws also pass through steps 5-6; the loop is one sequential worker.

A9 beside A8: the position-by-position reading works only after the reader mentally rotates one ring onto the other, and the two rings put the simulation on different edges (5→6 vs 6→1), which is the point but is easy to miss at the size where both fit a page. One draw vs M, weight before vs after, and the barrier do come through. Monotone bandwidth vs quantile threshold does not (same dashed ellipse), and history vs population reads as "more dots vs fewer dots".

Redundancy: the paper currently has no method figure at all (the first `\includegraphics` is `fig_predictor` at line 226), so A8 fills a real slot. Among the drafts, A1's "Propagator call" box lists the same four formulas; if A8 goes in, that box should become a pointer to A8. A4 (three weights) is complementary and carries the "never reported" message A8 lacks. GA3 and A9 make the same generation-vs-stream point; keep one. Table method-comparison (App. C) is A9's textual twin.

## 3. Improvements, ranked

**Essential before either figure goes in**

- E1. Make the smooth kernel visible (finding 1): in A8 panels 2-3 replace the dashed ε_n ellipse by a soft halo (fill K_{ε_n}(ρ(θ)) as an alpha gradient, or shade the archive dots by K_ε(ρ_j) with no boundary at all); keep the hard dashed boundary only in A9. Cheapest single change with the largest gain.
- E2. Fit the text width (372 pt; the ring PDF is ~487 pt, so `\small`/`\footnotesize`/`\scriptsize` print at about 6.9/6.1/5.3 pt). Either one ring per figure with the long formulas moved to the caption (the q̄^on definition, Σ_n = L Lᵀ, the append tuple, the amis_interval note), radius ~3.2 cm and panels ~2.2 cm so labels stay ≥ 7 pt; or the strip form below.
- E3. Re-route the bootstrap chord to the closing arrow (finding 2); label "|H_n| < k: θ* ∼ π, w* = 1; no weight step".
- E4. Retitle step 6 "proposal-time weight (stored, never reported)" and add the caption pointer to eq. (posterior-estimator) (finding 3).
- E5. Show the asynchrony (finding 4): incoming arrows into panel 1, closing arrow "θ* to a free worker; ρ* arrives later", centre text "one arrival → one update; everything recomputed from H_n".
- E6. A9 caption: "ABC-PMC with a quantile schedule and a hard threshold; not the paper's matched-kernel pyABC baseline (App. C)" (findings 11, 17).

**Should**

- S1. A9 position 5 wording and the barrier label (findings 12-13).
- S2. ω^{(t)}/ω^{(t+1)}, ρ_i in P_t, fix the P_{t+1} overlap (findings 14-15).
- S3. Caption map from step badges to Alg. 1 lines, or drop the badges (finding 5).
- S4. Replace `amis_interval` by "every I calls" or "periodically" (finding 8).
- S5. Key for the panel-1 shading, or drop the shading (finding 9).
- S6. `RHO_NOISE = 0` for the A9 panels (finding 16).

**Optional**

- O1. Panel 6: draw only the most recent snapshots plus one wide early one, label "recent proposals q_s" (finding 10).
- O2. Step 2: "ε_n non-increasing" in the figure, min(ε_hist, ε_sched) in the caption.

**Formulas: in the figure vs the caption.** Keep in the figure one line per step: H_n tuple; ε_n and A_n = Top_k; W̃_j ∝ w_j K_{ε_n}(ρ_j); q_n; θ* = θ_J + L_n z; w* = π(θ*)/q̄^on_n(θ*). Move to the caption: the q̄^on_n definition, Σ_n = L_n L_nᵀ, the snapshot rule, the append tuple. That alone removes about a third of the label width.

**A better form for the comparison.** A 2 × 6 strip: columns = the six steps (state, bandwidth, weights & Σ, proposal, draw, weight), top row A9 in vermilion, bottom row A8 in blue, panels 2.0-2.2 cm, one formula line under each panel, a return arrow at the right end of each row, the orange barrier as a vertical bar in the top row between the draw and weight columns, and a small "simulate" glyph in column 5 for PMC and after column 6 for ours. Column alignment delivers the position-by-position reading the two rings only promise, it fits 372 pt at ≥ 7 pt, and the different position of the simulate glyph becomes the visual headline. All twelve panel PDFs can be reused unchanged.

## 4. Verdict

**A8:** yes, main text, in §3.5 beside Algorithm 1 (or as Fig. 1 in §3.1 if A1 is not used), after E1-E5. As it stands it is a sound visual skeleton of one call, checked against the code, with three misreads built in (threshold look, unqualified weight, bootstrap routing), the asynchrony invisible, and a print-size problem.

**A9:** not as a standalone ring in the main text. It draws textbook hard-threshold PMC rather than the baseline the experiments use, and it duplicates Table method-comparison and GA3. Either fold it into the 2 × 6 strip with A8 (then the pair belongs in §3), or place it in Appendix C next to Table method-comparison with the E6 caption.
