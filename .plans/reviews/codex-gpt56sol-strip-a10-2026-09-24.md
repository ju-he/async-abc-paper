# Figure review, round 2 — codex CLI, GPT-5.6-Sol (xhigh reasoning), 2026-09-24

Target: the 2 × 6 strip `latex/figure-drafts/tikz/a10_algorithm_strip.tex` (state of 2026-09-24, uncommitted, branch `campaign-tooling`), with its panel scripts and `_toy2d.py`. Reviewer had read-only access to the repository and the propulate checkout, the 220-dpi render attached as an image, the round-1 digest and both round-1 reports, and the brief in `.plans/reviews/figure-brief-strip-2026-09-24.md`. Token usage not reported by the CLI.

---

# Independent review: A10 algorithm strip

The strip is methodologically sound overall. I found no blocking error in the six-step comparison, either return loop, or the distinction between textbook hard-threshold ABC-PMC and the paper’s method. Four issues should be corrected before insertion into §3, principally the over-general legend and the bootstrap record semantics.

Sources checked: [strip source](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/figure-drafts/tikz/a10_algorithm_strip.tex:25), [manuscript §3](/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/sn-article.tex:73), [implementation](/home/juhe/bwSyncShare/Code/propulate/propulate/propagators/abcpmc.py:981), both panel generators, `_toy2d.py`, the README, and the round-1 digest and reports. The implementation was readable.

## 1. Accuracy

1. **Legend: point size, gray shading, and gray contours are stated too generally.**

   - **Element:** `a10_algorithm_strip.tex:71–72`.
   - **Figure shows:** “point size = weight; gray shade = \(\rho_i\); gray contours = stored proposals \(q_s\).”
   - **Source says:** These encodings are panel-specific. In row (b), history and archive points have fixed sizes; only panel b3 uses size for \(\tilde W_j\) (`a8_algorithm_panels.py:44–61`). In row (a), panel a5’s accepted stars have fixed size, while a1–a3 and a6 use \(\omega\) (`a9_pmc_panels.py:40–90`). Gray shading encodes \(\rho_i\) only in b1. Gray contours in a5 and b5 are the current proposal, not stored proposals; only b6 shows snapshots. The dotted covariance ellipses are not keyed.
   - **Severity:** **should fix.** This is particularly risky because §3 distinguishes three kinds of weights. Suggested wording: “Where sizes vary: a1–a3 use \(\omega^{(t)}\), a6 uses \(\omega^{(t+1)}\), and b3 uses \(\tilde W^{(n)}\); shading in b1 encodes \(\rho\); gray contours in b6 are stored proposals; dotted ellipses show \(\Sigma\).”

2. **Bootstrap return: the appended tolerance disagrees with the implementation and exposes a paper/code inconsistency.**

   - **Element:** b1 history tuple, bootstrap label, and common return annotation (`a10_algorithm_strip.tex:48,51,67,69`).
   - **Figure shows:** A bootstrap draw goes directly to a worker with \(w^\star=1\), then shares the return annotation that appends \((\theta^\star,\rho^\star,\epsilon_n,\tau^\star,w^\star)\).
   - **Source says:** The manuscript defines every history item with an \(\epsilon_i\) and says every proposal stores its active bandwidth (`sn-article.tex:80–86`). The implementation returns bootstrap draws before assigning `child.tolerance` (`abcpmc.py:1031–1035`); only archive-phase draws are stamped (`1155–1156`), and `extract_posterior` explicitly identifies bootstrap draws by `tolerance is None` (`1303–1309`).
   - **Severity:** **should fix.** Distinguish the bootstrap record explicitly, e.g. \(\epsilon^\star=\varnothing\) for bootstrap, and reconcile the same point in §3. Also define \(\tau^\star=n\), or append \(\epsilon_{\tau^\star}\), so the stored proposal-time bandwidth cannot be mistaken for the bandwidth current when the result arrives.

3. **Proposal and covariance formulas are compact to the point of being formally incomplete.**

   - **Element:** a2–a6 and b2–b4 formula lines (`a10_algorithm_strip.tex:33–37,52–54`).
   - **Figure shows:** \(q_t=\sum_i\omega_i^{(t)}K_{\Sigma_t}\), \(q_n=\sum_j\tilde W_jK_{\Sigma_n}\), \(\widehat{\mathrm{Cov}}_\omega(P_t)\), and \(\omega^{(t+1)}\propto\pi/q_t\).
   - **Source says:** The manuscript writes the centered kernel explicitly:
     \[
     q_n(\theta)=\sum_{j\in A_n}\tilde W_j^{(n)}
     K_{\Sigma_n}(\theta-\theta_j)
     \]
     (`sn-article.tex:89–96`). The analogous PMC expression must likewise be evaluated at \(\theta-\theta_i\); covariance is over the particles’ \(\theta_i\), and the new weight is evaluated at \(\theta_i^{(t+1)}\). The quantile is over the \(\rho_i\) in \(P_t\), not an unspecified collection of discrepancies.
   - **Severity:** **should fix.** If the exact formulas will not fit, replace the pseudo-formulas with “mixture centered on \(P_t\)” and “mixture centered on \(A_n\),” and put the exact definitions in the caption.

4. **The online snapshot rule omits two implementation-order qualifiers.**

   - **Element:** legend definition of \(\bar q_n^{\mathrm{on}}\) and snapshot pushing (`a10_algorithm_strip.tex:72`).
   - **Figure shows:** \(q_n\) is pushed every `amis_interval` calls.
   - **Source says:** The current weight first uses \(q_n\) separately from the existing snapshots (`abcpmc.py:1137–1148`); only afterward is \(q_n\) pushed into the local ring buffer (`1177–1184`). In parallel execution this buffer belongs to the propagator instance/rank.
   - **Severity:** **minor.** If this implementation detail remains, say “pushed after weighting, periodically on each rank.” Preferably drop the code identifier from the figure and point to Eq. `amis-weight`.

5. **“One draw” means one emitted candidate, not necessarily one random attempt.**

   - **Element:** b5 formula line (`a10_algorithm_strip.tex:55`).
   - **Figure shows:** “one draw.”
   - **Source says:** One candidate is returned, but the implementation may reject proposals outside the box, retry after denominator underflow, or fall back to the prior (`abcpmc.py:1085–1175`). The README properly declares the toy illustrative.
   - **Severity:** **minor.** “One candidate” is exact and preserves the intended contrast with “until \(N\) kept.”

6. **The dashed bootstrap arrow is visually ambiguous.**

   - **Element:** long dashed path around row (b), `a10_algorithm_strip.tex:69`.
   - **Figure shows:** A dashed branch from the history directly to the worker pool.
   - **Source says:** This is correctly the bootstrap bypass: the prior draw does not pass through b2–b6 (`abcpmc.py:1010–1035`).
   - **Severity:** **should fix for readability.** It resembles a dashed enclosure around row (b), and dashing is already used for the hard-threshold contour. Use a shorter directly labelled bypass or a distinct line style.

7. **All remaining algorithmic elements are accurate.**

   - **Shared headers and titles:** The six headers correctly align state, tolerance, adaptation weights/covariance, proposal, draw, and importance weight. Both titles correctly state their update cadence.
   - **Row (a), panels 1–4:** \(P_t\) contains \((\theta_i,\rho_i,\omega_i^{(t)})\); the threshold comes from \(P_t\); \(\Sigma_t=2\widehat{\mathrm{Cov}}_\omega\); and \(q_t\) is formed from \(P_t\). These match the intended quantile-scheduled hard-threshold ABC-PMC miniature.
   - **Row (a), panels 5–6:** The candidates are proposed, simulated, and tested one at a time until \(N\) are accepted; \(M=48\) is known only on completion; weights are assigned after acceptance and normalized in the toy. With discrepancy noise disabled, every retained point lies inside the displayed region.
   - **Top return rail:** \(P_{t+1}\) consists of the \(N\) accepted weighted particles; rejects are discarded. The barrier is correctly on the generation transition, with the precise claim that \(q_{t+1}\) needs complete \(P_{t+1}\), not that textbook PMC necessarily waits for every launched latecomer.
   - **Row (b), panels 1–4:** The history, monotone smooth bandwidth, top-\(k\) archive, adaptation weights, covariance, and proposal agree with `sn-article.tex:78–104` and `abcpmc.py:1003–1083`.
   - **Row (b), panels 5–6:** Parent-and-perturb sampling and the proposal-time balance-heuristic weight before simulation agree with `abcpmc.py:1085–1161`. “Never reported” correctly separates \(w^\star\) from Eq. `posterior-estimator`.
   - **Bottom return rails:** The worker pool, later discrepancy arrival, and additional gray arrival rails now make the asynchronous cadence visible. The solid path correctly places external simulation after weighting.
   - **Panel tags:** \(\epsilon\), \(\Sigma_t/\Sigma_n\), \(q_t/q_n\), \(\theta^\star\), \(w^\star\), “4 of \(N\),” and \(M=48\) correspond to the generated toy state.
   - **Method identity:** The row title and legend correctly identify row (a) as textbook hard-threshold, quantile-scheduled PMC rather than the paper’s matched smooth-kernel pyABC baseline.
   - **Severity:** **no issue**, apart from the notation and legend qualifications above.

## 2. Round-1 findings 1–11

| # | Status in the strip | Residual |
|---:|---|---|
| 1 | **Resolved.** | b6 says “before simulating; never reported,” and the legend points to the reported estimator. |
| 2 | **Resolved.** | Row (b) uses a Gaussian kernel profile with no parameter-space boundary; only row (a) has the dashed hard region. |
| 3 | **Resolved in substance.** | Worker pool, multiple arrival rails, and “arrives later” make asynchrony visible. Define \(\tau^\star=n\); make the bootstrap path less enclosure-like. |
| 4 | **Resolved.** | Bootstrap goes from history straight to a worker and bypasses the AMIS weight step. |
| 5 | **Resolved.** | a5 explicitly shows the propose–simulate–test loop in progress, not \(M\) candidates drawn in advance. |
| 6 | **Resolved.** | The barrier is on the return rail and says that the next proposal needs complete \(P_{t+1}\); “wait for the slowest” is gone. |
| 7 | **Resolved.** | Textbook PMC and the experimental matched-kernel pyABC baseline are explicitly distinguished. |
| 8 | **Resolved.** | \(P_t\) carries \(\rho_i\), weights have generation indices, and proportional new weights are shown. Exact formula arguments should still be restored. |
| 9 | **Resolved.** | The PMC toy runs without discrepancy noise; accepted points match the hard region. |
| 10 | **Resolved.** | The PDF is \(375.65\) pt including the 4 pt standalone border, so fitting it to 372 pt scales nominal 7 pt text only to about 6.9 pt. |
| 11 | **Partly resolved.** | A legend exists, but its point-size, shade, and contour statements are not correctly scoped by panel; see finding 1. |

## 3. Readability

The column-aligned comparison works. Columns 1, 2, 5, and 6 carry the essential contrasts immediately: population versus history, hard threshold versus smooth kernel, repeated acceptance versus one candidate, and post-acceptance versus pre-simulation weighting. Columns 3 and 4 are necessarily similar and show where the methods share PMC machinery.

The first visual misread will be the dashed bootstrap path: it looks like a boundary enclosing row (b), not a branch to the workers. The first semantic misread will be the global “point size = weight” statement, which encourages readers to treat \(\omega\), \(\tilde W\), and fixed decorative marker sizes as one quantity.

The staggered formula lines are acceptable. Both rows use the same parity, so vertical column comparison is preserved, and the staggering prevents adjacent formulas from colliding. It creates a mild zigzag and extra height, but it is not the main readability problem. Uniform baselines would only be preferable after shortening the formulas.

At 372 pt, the nominal text size is adequate. The marginal elements are:

- the tiny kernel rug ticks and inset labels;
- pale non-archive points and 0.5 pt gray contours;
- the long bottom return sentence;
- the four-line legend block.

The following can be dropped without loss:

- toy-specific `4 of N`, `M=48`, and “\(k=N=12\)” numbers;
- “2-D box prior”;
- the rug ticks in the kernel insets;
- the explicit `amis_interval` implementation detail;
- \(\Sigma_n=L_nL_n^\top\), if \(L_n\) is defined once in the caption;
- the full append tuple, provided the caption defines the stored fields and the bootstrap exception.

## 4. Ranked improvements

### Essential before §3

1. Correct and panel-scope the glyph legend; add the dotted covariance ellipse.

2. Resolve the bootstrap tolerance discrepancy between manuscript and code, and state \(\tau^\star=n\) or use \(\epsilon_{\tau^\star}\) on return.

3. Make the two proposal-mixture formulas exact, or replace them with verbal labels and put the exact centered formulas in the caption.

4. Redraw the bootstrap bypass so it reads unmistakably as an arrow rather than a dashed enclosure.

When the lower text moves to the caption, the caption must retain:

- that row (a) is textbook hard-threshold quantile ABC-PMC, not the matched-kernel pyABC baseline;
- that \(w^\star\) is a stored proposal-time weight and never the reported posterior weight;
- the pointer to Eq. `posterior-estimator`;
- the worker/arrival semantics and bootstrap special case;
- an accurate, panel-scoped glyph key.

### Optional refinements

1. Replace “one draw” with “one candidate.”

2. Shorten the top title to “Textbook hard-threshold ABC-PMC: one update per generation”; put “quantile schedule” in the caption.

3. Replace “\(W\) workers” with “worker pool” to avoid visual competition with \(\tilde W\).

4. Raise the contrast or line weight of gray points and contours for print.

5. State the snapshot rule only as “periodically stored proposals”; leave push order, interval, and rank-locality to §3 or Appendix C.

## 5. Verdict

The strip is ready for §3 after four targeted changes: correct the panel-specific legend, reconcile bootstrap tolerance/timestamp semantics, make the proposal formulas exact, and make the bootstrap bypass visually unambiguous. No redesign of the 2×6 structure is needed; the column comparison, PMC loop, asynchronous return rails, and barrier placement are all successful.