# Plan — incorporate Review 3 ("Asynchronous, Generation-Free ABC")

**Created:** 2026-06-29 · **Branch:** `refactor/general` · **Paper:** `latex/sn-article-template/sn-article.tex`

## ⏳ IN-FLIGHT cluster jobs (2026-06-29)
- **WS3 SBC 1000 trials** — sweep `sweep-c9fbd0`, jobs `14069282`–`14069291` (10 shards × 100 trials,
  `--num-shards 10 --shard-run-id sbc1k0629`). Output: `/p/scratch/tissuetwin/herold2/async-abc/sbc_1k_20260629`
  (runner writes `sbc/`). ~6 h wall. Validated running (48 ranks). Last shard finalizes the merged coverage
  table. When done: re-plot Table 4 (l.252) + `fig:sbc`, soften §7 calibration prose (l.251).
- **WS4 every-particle bias** — RETUNED re-run, job `14069409` (single, 1 node, ~50 min). Output:
  `/p/scratch/tissuetwin/herold2/async-abc/parambias2_20260629/runtime_heterogeneity/`.
  - First attempt (job 14069313, `parambias_20260629`) was a NULL result — diagnosed as mis-tuned: the
    delay gradient was normalized by the PRIOR half-width (5), but the gaussian posterior is narrow
    (std ~0.1), so runtime was near-uniform over the explored region → throughput flat across coupling,
    no bias. LESSON: couple runtime to the POSTERIOR scale, not the prior.
  - Retune: added `coupling_scale` knob (config 0.15) + `delay_cap_s` (0.5, prevents far-tail prior draws
    sleeping unboundedly), levels [0, 0.5, 1.0, 2.0]. Runner: `_make_param_coupled_simulate` now caps the
    delay; `_make_delay_simulate` passes coupling_scale/cap. Opt-in; lognormal default preserved.
  - Analysis: `experiments/scripts/analyze_param_bias.py`. The non-shard runner does NOT emit
    `gaussian_analytic_summary.csv` (only the sharded finalize does); use
    `runtime_performance_summary.csv` -> `final_quality_wasserstein` by (base_method, sigma=coupling).
    Bias signature = async Wasserstein degrades MORE than sync (discard-latecomers) as coupling grows.
    Validate the gradient bites first: throughput must DROP with coupling (flat = no bite, re-tune steeper).
    Write the quantified result into Limitation iv (l.~329).
- Budget now bumped: hard/job 2000, hard/session 2000, soft 500 node-h (was 48/96/24).

Review 3 verdict: *good idea, cleanly executed, unusually honest; the real contribution is **systems**
with adequate inference quality. The theory overpromises; some empirical numbers undercut the prose.*
Spine of the response = the reviewer's own **framing suggestion**: reposition systems-first, demote the
CLT to "inherited guarantees validated by SBC", be uniformly honest about the plateau and the
every-particle bias.

## Decisions locked (from user, 2026-06-29)
- **Major 1 (theory gap):** REFRAME ONLY. No defensive-mixture code change. Demote CLT everywhere.
- **Major 2 (scaling decline + sync non-monotone):** Options **A + C + D**, NOT B.
  Confirmed root cause = inter-node coordination wall (peak at 48 = 1 node; declines once job spans
  multiple nodes: 128 = 3 nodes, 256 = 6 nodes). Within a node throughput is ~linear (16→115→295→595).
  Islands are *not* the fix (single-island headline still hits the per-node wall). No mitigation re-run.
- **Major 3 (SBC N=100):** Bump to 1000 trials; re-plot; soften "if anything better" → "comparable".
- **Moderate iv (every-particle bias):** BUILD the parameter-dependent-delay experiment to quantify it.
- **Anonymization:** Venue is single-blind → leave Propulate / JUWELS named as-is. No edits.

---

## Workstreams

### WS1 — Reframe spine (writing-only) · Major 1 + framing suggestion
- Abstract (l.35): demote "consistency and a CLT" to "inherited asymptotic guarantees (AMIS), with the
  *implemented* estimator validated by SBC"; lead benefit clause with barrier-removal/utilization.
- Contributions (l.45–51): reorder to put Propulate/barrier-free systems contribution first; reword
  contribution #4 (l.50) from "Consistency and a CLT" to inherited-guarantees framing.
- Conclusion (l.322): same demotion ("carries consistency and CLT guarantees" → inherited, SBC-validated).
- Keep the honest §4 "Theory versus implementation" paragraph (l.169) but make it consistent with the
  now-softened headline (no longer a lone hedge contradicting a strong abstract).

### WS2 — Scaling narrative (A + C + D) · Major 2 + single-island moderate
- **A (diagnose/attribute):** mine existing LV strong-scaling logs for a wall-clock breakdown
  (simulator vs per-arrival proposal/AMIS vs MPI-coordination). If logs lack the split, one light
  instrumented re-run at 48/128/256 (1 rep) to get the breakdown. Rewrite §7.2 (l.226): throughput
  scales ~linearly *within a node*; the knee is the single-node→multi-node boundary (inter-node
  coordination latency in the single-island propagator), which only bites because LV is
  near-instantaneous. Replace "coordination-bound plateau" with the explicit node-boundary explanation.
- **C (lead with CPM):** make CPM the headline strong-scaling result (cost-bearing simulator amortizes
  coordination → async ~linear, sync plateaus). Present LV explicitly as the *adversarial
  near-instantaneous stress case* → the decline becomes evidence FOR the thesis. Uses the 5-point CPM
  scaling figure (data already in hand; see WS5).
- **D (sync + variance):** add IQR / error bars to Table 3 (l.228) and Fig 3 (`fig:scaling`); explain the
  non-monotone sync baseline (148→86→111→146→121) as fixed-900s-budget-window quantization (few
  generations complete) + barrier stalls + node-crossing. 5 reps already exist — recompute spread.
- State the decline directly (a sentence), not soft "plateau" language (reviewer's explicit ask).

### WS3 — SBC N=1000 (experiment) · Major 3
- Edit `experiments/configs/sbc.json` `n_trials` 100 → 1000. Submit on cluster (Gaussian is cheap;
  dry-run cost estimate first). Runner: `experiments/scripts/sbc_runner.py`.
- Re-plot Table 4 (l.252) + `fig:sbc` (`fig_sbc_coverage.pdf`, `fig_sbc_rank.pdf`). Tightens MC error
  ~±0.03 → ~±0.01. Update caption to report N=1000.
- Soften §7 calibration prose (l.251): "if anything better" → "comparable; both close to nominal".
  (Do this softening even if 1000 trials still favors async, per reviewer.)

### WS4 — Every-particle bias experiment (code + experiment + new result) · Moderate iv
- **Hook:** add a *parameter-dependent* delay path so simulator runtime depends on θ (slow simulator in
  part of parameter space). Likely a new small runner derived from
  `experiments/scripts/runtime_heterogeneity_runner.py:79` (currently delay is θ-independent lognormal).
  e.g. `delay = base · g(θ)` so one region is systematically slow.
- **Measure:** posterior bias toward fast (cheap-θ) regions — async (keeps every particle) vs a
  discard-latecomers variant and/or ground truth. Quantify magnitude (mean shift / Wasserstein toward
  the fast region) on a model with known truth (Gaussian or g-and-k).
- **Write:** convert Limitation (iv) (l.319) from "measured but not corrected" to a *quantified* result
  with a small figure/table; keep it honest (still uncorrected, now bounded).

### WS5 — CPM figure + abstract softening + Fig 6 difference plot · Moderate CPM (+ pending HANDOFF item)
- Build the **5-point CPM scaling figure** (1,4,16 from `run_cpm_fillin_20260628`, 48,96 from
  `run_cpm_20260626_1906`) — data confirmed present. Regenerate `figures/fig_cpm_scaling.pdf` as the
  5-point version; wire into §7.3 (currently shows `fig_cpm_corner.pdf`, `fig:cpm-posterior`).
  Throughput is **metric-invariant** → unaffected by the improved nano configs; proceed as-is.
- **Reframe CPM quality via structural degeneracy (NEW, decided 2026-06-29).** The paper's CPM infers
  division_rate + motility jointly (`parameter_space_division_motility.json`, true div=0.0499, mot=0.2).
  The user's nano validation campaign (nastjapy `inference-campaign`, D1f/D1g) proves these are
  *structurally degenerate at nano scale* — division_rate is unrecoverable with motility free (D1f failed;
  D1g succeeded only with motility fixed). So "weakly identified by both, async ~0.47 vs sync ~0.40 near
  the floor" is a property of the 2-parameter nano problem, shared equally by both methods — NOT an async
  weakness. Reword §7.3 to say so: both methods are correctly limited by a known identifiability
  degeneracy, which *isolates* the systems advantage as orthogonal to identifiability. This is free and
  defuses the reviewer's "figure doesn't favor you" better than a rerun.
- **Decision: do NOT rerun CPM on the improved configs.** Systems claim is metric-invariant; a quality
  showcase is unnecessary (reviewer: CPM's job is feasibility + throughput; quality carried by
  g-and-k/LV); a rerun would force rerunning BOTH arms on identical improved configs (expensive) with
  risk the every-particle bias bites once identifiable (studied cleanly in WS4 instead).
- Soften abstract CPM clause (l.35): "we demonstrate … realistic Cellular Potts workload" →
  practical, *comparable* quality + throughput advantage (don't oversell quality).
- **Fig 6 difference plot:** add a difference panel (async − sync Wasserstein vs budget) so "comparable"
  is legible instead of the async curve sitting above sync.
- **OPTIONAL macro-spheroid feasibility capstone (NON-gating).** If an async-only macro-spheroid run
  (~8–17 h/sim, ~50× slower, heterogeneous runtime; nastjapy `spheroid_inf_small`) is available before
  submission, fold it in as a *non-comparative* feasibility + worker-utilization demonstration: method
  scales to a frontier workload and sustains high utilization / low idle fraction. NO matched sync
  baseline (running it would be prohibitively wasteful — which IS the argument). Frame strictly as
  feasibility, never head-to-head. Sharpens RQ3 (see WS6). To be reusable, macro runs must log:
  per-worker utilization/idle time series, throughput, wall-clock-to-posterior, posterior vs reference.
  If unavailable, paper stands on nano CPM + LV + straggler; RQ3 still sharpened in prose.

### WS6 — Minors (writing-only)
- Condition 3 typo (l. near 165 / theory block): `n ↓ ∞ > 0` → `ε_n ↓ ε_∞ > 0`; fix garbled ε
  subscripts throughout (check source, not just PDF extraction).
- RQ3 phrasing: "Does it remain practical" → sharpen to a *frontier-scale* practicality claim (the
  method scales to workloads where a synchronous matched baseline would itself be prohibitively
  wasteful); concretize with the macro capstone (WS5) if available, else in prose.
- Table 1 (l.141 area): trim rows that restate the single event-driven-vs-generation distinction.
- Appendix A bit-identicality: add one clause — bit-identical *because* log-sum-exp is associative over
  chunks (l.319 reproducibility note / Appendix A).

---

## Compute plan (lighter after decisions — no islands, no defensive mixture, no mitigation sweep)
New cluster jobs only:
1. **SBC 1000 trials** (Gaussian, 48 workers) — cheap; `estimate_cost`/dry-run first.
2. **Bias experiment** (parameter-dependent delay, ~heterogeneity scale, 48 workers) — modest.
3. **Scaling diagnostic (A)** — prefer pure log-mining of existing LV scaling run; only if the breakdown
   isn't logged, one 1-rep instrumented run at 48/128/256.
Budget: soft 24 / hard 96 node-h rolling 24 h (~80 nh nominal, ~12 nh phantom-reserved). All three above
are small; keep reservations modest to avoid the soft-limit elicitation. Deploy via rsync (see HANDOFF
infra cheat-sheet). venv: cluster `/p/project1/tissuetwin/herold2/nastjapy/.venv`; local tests
`nastjapy_copy/.venv/bin/python`.

## Sequencing
1. **Code + local test (venv):** parameter-dependent-delay hook (WS4); SBC config bump (WS3); check
   whether scaling logs already carry the timing breakdown (WS2-A).
2. **Submit cluster jobs early** (SBC 1000, bias experiment, optional diagnostic run) — they have latency.
3. **While jobs run (writing-only, no blockers):** WS1 spine reframe; WS5 CPM 5-point figure (data ready)
   + abstract softening + Fig 6 difference plot; WS2-C/D prose + Table 3 variance; all of WS6.
4. **When jobs land:** update Table 4 + SBC figs (WS3); build bias result figure/table (WS4); finalize
   §7.2 attribution (WS2-A).
5. **Recompile** (`latexmk … sn-article.tex`; verify 0 undefined refs); regenerate compressed PDF.
   Optional: 3rd codex review to confirm revisions hold.

## Coverage checklist (every review item)
- [ ] Major 1 theory overpromise → WS1 (reframe)
- [ ] Major 2 async decline + sync non-monotone → WS2 (A+C+D)
- [ ] Major 3 SBC N → WS3 (1000 trials)
- [ ] Moderate iv every-particle bias → WS4 (experiment)
- [ ] Moderate CPM oversold → WS5 (abstract + 5-pt fig + diff plot)
- [ ] Moderate single-island bottleneck → WS2 (node-boundary attribution; islands explicitly not the fix)
- [ ] Minor de-anonymization → N/A (single-blind, leave as-is)
- [ ] Minor Condition 3 typo / ε subscripts → WS6
- [ ] Minor RQ3 phrasing → WS6
- [ ] Minor Fig 6 difference plot → WS5
- [ ] Minor Table 1 redundancy → WS6
- [ ] Minor log-sum-exp clause → WS6
