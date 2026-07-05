# Full Review: Paper Concept, Experiment Implementation, and Figures

**Date:** 2026-07-05
**Scope:** `latex/sn-article-template/sn-article.tex` (paper), `experiments/async_abc/` (implementation), `experiments/scripts/` + `experiments/configs/` (experiment pipeline), `propulate/propulate/propagators/abcpmc.py` (core algorithm), `latex/sn-article-template/figures/` (shipped figures).
**Method:** paper read in full; four independent review passes (core method, baseline fairness, experiment pipeline/metrics, plotting/publication readiness), each checking code against the paper's specific claims with file:line evidence.

---

## 0. Executive summary

The concept is strong and the **systems evidence (RQ2/RQ3) holds up**: throughput accounting is symmetric between methods, partial final generations are counted for the baseline, delay injection is applied identically to both arms, and the fair-baseline (population = worker count) scaling data is real and used in the figures.

The problems concentrate in **claim-vs-code mismatches** — places where the paper asserts something the implementation does not do — and in **figure production quality**. The five most consequential:

1. **"Matched bandwidth schedule" is not implemented** — the pyABC baseline hardcodes a median quantile-epsilon while the async side uses ESS-retention bisection (critical; fairness framing).
2. **The retroactive posterior estimator is *not* the exact cumulative mixture** the theory section claims it is — it is a ≤20-snapshot approximation with a defensive prior floor (critical; theory firewall).
3. **"Wasserstein to the analytic posterior" is actually Wasserstein to the true-parameter point mass** in the param-bias and Gaussian-recovery figures (critical; referee-checkable, values impossible as labeled).
4. **The baseline's smooth-kernel acceptor RNG is correlated across MPI workers** and non-reproducible from the config seed (high; plausibly contributes to the baseline's SBC under-coverage the paper reports as a finding).
5. **Two shipped figure PDFs contain "Cellular Potts" panel titles** (`fig_scaling_combined.pdf`, `fig_posterior_recovery.pdf`) while the surrounding paper-clean captions say "realistic workload" — a figure/caption mismatch. *(Re-scoped 2026-07-05: the scrub was a test only and the final paper names CPM/NAStJA, so this dissolves into the naming-consistency item, II.5; the missing `fig_posterior_recovery` generator remains a real gap.)*

Additionally, all 15 referenced figures embed **Type 3 fonts** (commonly rejected by Springer production), effective print font sizes are 4–6.5 pt (target ≥7–8 pt), 5 of 15 referenced figures have **no in-repo generator script**, and figure styling (colors, labels, error conventions) is inconsistent across figures — including one figure where the async/sync **colors are inverted** relative to the rest of the paper.

---

## 1. Paper concept review

### 1.1 Strengths

- The core idea — streaming limit of AMIS (stage size one, per-arrival proposal adaptation), smooth-kernel ABC, and a history-reconstructed "stateless" propagator riding Propulate's barrier-free island model — is coherent, well-motivated, and well matched to the HPC heterogeneity problem it targets.
- Claim hygiene is unusually good in most places: the Lotka–Volterra "honest boundary" (adversarial near-instant simulator, erosion of the advantage at scale, single-worker slowdown admitted), the scoped sensitivity claim (explicitly limited to the Gaussian-mean benchmark), the SBC non-over-interpretation (MC standard error acknowledged), and the explicit Limitations section.
- The three-weight-object disambiguation (§2.5, "Three weight objects") is a genuinely useful expository device — though see §3 below: one of its three statements is contradicted by the code.
- The experimental program is well designed: matched-kernel baseline, straggler + heterogeneity + parameter-coupled-runtime studies, SBC calibration, ablation isolating the two ingredients, sensitivity grid.

### 1.2 Concept-level concerns

**(a) The theory citation is shakier than the text admits.**
Theorems 1–2 (§4) are presented as "a corollary of [Thm. 1–2, cornuet2012amis]". To the best of available knowledge, the original AMIS paper (Cornuet, Marin, Mira & Robert 2012) *conjectured* convergence and left consistency open; consistency was later proven by Marin, Pudlo & Sedki only for a **modified** AMIS (adaptation restricted so the weighting stage is decoupled). If the proof sketch leans on theorems that do not exist in the cited paper in the form invoked, a knowledgeable referee will catch it.
→ **Action:** verify what `cornuet2012amis` actually proves; either cite the Marin–Pudlo–Sedki consistency result with its modification caveat, or soften Theorems 1–2 to "inherits the AMIS asymptotic rationale" with the caveat that the original AMIS convergence is heuristic.

**(b) The novelty claim is a strong universal.**
"To our knowledge it is the first ABC algorithm whose proposal mixture updates after every evaluated particle" (contribution 1). The hedge is present, but a literature sweep on anytime/streaming/asynchronous ABC (beyond the cited Paige et al. / Murray et al. anytime SMC line) is cheap insurance before submission.

**(c) The theory-vs-implementation "firewall" paragraph is the load-bearing hedge — and it is factually wrong as written** (see finding 2.1 below). The paper's whole strategy is: prove asymptotics for the idealized estimator, claim the *reported* estimator evaluates that object exactly, and confine approximations to the online path validated by SBC. The middle claim is false in code. The SBC-validation hedge already covers the honest version, so this is fixable by text alone — but it must be fixed.

---

## 2. Core method implementation vs. paper (propagator, estimator, schedulers)

Files: `propulate/propulate/propagators/abcpmc.py` (algorithm), `experiments/async_abc/inference/propulate_abc.py` (wrapper), `method_registry.py`, `_attempt_trace.py`. Interface confirmed: `Propulator._breed` passes the evaluated population to `propagator(active_pop)` (`propulate/propulate/propulator.py:255-256`, individuals appended only after evaluation at `propulator.py:329`) — matching the paper's `__call__(inds) -> Individual` framing.

### 2.1 CRITICAL (paper): retroactive estimator is not the exact cumulative mixture

- **Paper claims** (tex:115): reported posterior reweights by π·K/q̄ₙ "with q̄ₙ the cumulative proposal mixture over the *full* evaluated history"; (tex:171): "The reported posterior evaluates *exactly* this object retroactively … the approximations we introduce for efficiency both sit on the *online* path."
- **Code:** `extract_posterior` builds q̄ from at most `n_proposals` reconstructed proposals, defaulting to `amis_snapshots` (= 20) (`abcpmc.py:1318-1324`), as a draw-proportional mixture **plus a defensive prior component floored at 0.5/(m+1)** (`abcpmc.py:1362-1371`). The docstring itself calls this "the (C4) *approximation* to the full cumulative mixture" (`abcpmc.py:1248-1250`).
- **Consequence:** a third approximation sits on the *reported-posterior* path, contradicting the theory-versus-implementation paragraph; the prior-floor component appears nowhere in Eq. (7).
- **Fix options:** (i) raise `n_proposals` to cover the full history for reported runs (cost is the O(nk) pass the paper already budgets), or (ii) rewrite tex:115/171 to describe the m-snapshot + prior-floor estimator honestly — the abstract's "sliding-buffer, top-k approximation … validated by SBC" hedge already accommodates this.

### 2.2 MAJOR (paper): statelessness and buffer determinism overstated

- **Paper claims** (tex:85): "every algorithmic quantity is a pure function of the evaluated history, with no mutable state carried between calls"; (tex:113): the snapshot buffer "is itself a deterministic function of Hₙ, so the propagator stores nothing between calls"; (tex:85): "exactly reproducible."
- **Code:** four pieces of mutable state on `self` change algorithmic outputs:
  - AMIS snapshot ring buffer `self._snapshots` / `self._calls_since_snapshot` (`abcpmc.py:703-704`), appended with the *live* proposal each interval (`abcpmc.py:1181-1184`); never reconstructed from history on the live path (truncated-history reconstruction exists only inside `extract_posterior`, `abcpmc.py:1329-1337`). After crash/restart the buffer is empty → proposal-time weights differ.
  - Cross-rebuild tolerance memory: `rebuild` takes `min(self.tol_from_history, new_min)` (`abcpmc.py:303`) — deliberately not a pure function of the current history (docstring `abcpmc.py:281-295` admits post-crash transient loosening).
  - Kernel-aware bandwidth throttle `_calls_since_bisect` / `_held_eps` (`abcpmc.py:1486-1487, 1640-1646`): bandwidth sequence depends on per-instance call counting (monotone, but not history-pure).
  - Geometric-decay watermark `_cached_consumed` / `_cached_tol` (`abcpmc.py:1816-1817`) — see 2.5.
- The class docstring (`abcpmc.py:374-405`) honestly scopes statelessness to the *estimator*; the paper text carries no such qualification. Run-level bit-reproducibility is additionally limited by MPI arrival order — documented in code (`abcpmc.py:1269-1281`), not qualified in the paper.
- **Fix:** qualify §2.1/§2.4 (statelessness of the *estimator and archive reconstruction*; the online AMIS buffer is performance state that the retroactive estimator supersedes), and qualify "exactly reproducible."

### 2.3 MAJOR (paper): "proposal-time weight used only for the ESS diagnostic" is false

- **Paper** (tex:115, weight object (ii)): the proposal-time weight is "used only for the effective-sample-size diagnostic."
- **Code:** the stored weight also feeds (a) the adaptation mixture weights W̃ (`abcpmc.py:880-891`) — as the paper's own Eq. (2) says (W̃ⱼ ∝ wⱼ·K(ρⱼ)), so tex:115 contradicts Eq. (2); (b) parent selection (`abcpmc.py:957, 1108-1110`); (c) the kernel-aware ESS bandwidth search (`abcpmc.py:1616-1620`); (d) archive eligibility — `weight == 0` individuals are excluded (`abcpmc.py:361-364, 822`), an undocumented rule; (e) replayed proposals inside `extract_posterior` (`abcpmc.py:1222-1227`).
- What *does* hold: it is never used as a particle's posterior importance weight. Downstream separation is respected (ESS diagnostic uses streaming `weight`, `analysis/ess.py:32, 93`; SBC uses `posterior_weights`, `analysis/sbc.py:60-64`).
- **Fix:** reword weight object (ii) — "assigned online, enters proposal adaptation via Eq. (2) and the ESS diagnostic, but never the reported posterior."

### 2.4 Verified claims (core method)

| Paper claim | Verdict | Evidence |
|---|---|---|
| Tolerance: stamp at proposal, running-min reconstruction, min with scheduler, monotone | **Matches** | stamp `abcpmc.py:1156`; running min `abcpmc.py:301-303, 324-325`; empty-history fallback `abcpmc.py:265-271`; `min(current, proposed)` `abcpmc.py:1044` + hard-kernel archive gate `1052-1056` |
| Archive = top-k by lowest ρ | **Matches** (+ undocumented weight==0 exclusion) | `abcpmc.py:349-366, 1067` |
| W̃ⱼ ∝ wⱼ·K(ρⱼ) in log space, uniform fallback on underflow | **Matches exactly** | `abcpmc.py:885-907` |
| Cholesky once per call | **Matches** (memoised across calls) | `abcpmc.py:923-927, 934-959` |
| Bootstrap: uniform prior draws until history ≥ k | **Matches** (smooth path) | `abcpmc.py:1028-1035`, defensive re-emission `1069-1075`; hard-kernel mode uses "k accepted below tol" instead (ablation only) |
| AMIS weight formula, buffer sampled every `amis_interval` | **Matches** | logsumexp denominator `abcpmc.py:1145-1148`; w* = π/q̄ `1161`; interval default k `abcpmc.py:702, 1181-1184` |
| Retroactive chunking 2^16, bit-identical, O(nk) | **Matches** | `abcpmc.py:140, 1375-1382`; each point's logsumexp is chunk-local so bit-identity is trivially true (appendix's "associative across chunk boundaries" wording imprecise but claim holds); cost O(n·m·k), m ≤ 20 |
| Quantile scheduler (lower-rank p=50, gated on k+m accepted) | **Matches** | `abcpmc.py:1728, 1751-1771` |
| Kernel-aware mode: ESS-retention 0.95 via bisection | **Matches** (implementation richer: geometric grid scan + bracketed bisection + `max_tighten_factor=0.5` cap + `bisect_interval` throttle, undocumented) | `abcpmc.py:1524-1604, 1640-1646` |
| Log-space numerics; crash-loudly on NaN/negative loss | **Matches** | kernels return log weights; `_check_loss` raises `abcpmc.py:710-726` |

### 2.5 Code bugs (independent of paper text)

- **MAJOR — geometric-decay cached-path divergence under MPI.** `GeometricDecayScheduler.compute_cached` (`abcpmc.py:1861-1879`) violates its own equivalence contract ("must remain equivalent to a full replay from history", `abcpmc.py:1786-1788`): a cross-rank individual whose `(generation, island, rank)` key sorts *before* the `_cached_consumed` watermark is inserted into the already-consumed prefix, shifting all epoch boundaries; the cached replay then partitions batches differently from the pure `compute` replay. The `_gen_order` tie-break was introduced precisely to make epochs arrival-order-invariant and the cached path defeats it. Affects hard-kernel geometric-decay runs only (paper's main runs are smooth-kernel → kernel-aware path), but it is a genuine correctness bug.
- **MINOR — latent TypeError.** `run_propulate_abc` forwards `low_rate`/`expand_factor` from config (`propulate_abc.py:555-556`) but `AcceptanceRateScheduler.__init__` no longer accepts them → construction crash if any config sets them (none currently do). Relatedly, the paper appendix (tex:391) documents the removed `r_lo = 0.1` guard — a nonexistent parameter (behaviourally identical: hold either way; docstring `abcpmc.py:1896-1904`).
- **MINOR — wall-time semantics leak.** `extract_posterior` runs on the *unfiltered* population (`propulate_abc.py:700-705`) while post-deadline records are dropped afterwards (`propulate_abc.py:763-767`): surviving records carry posterior weights normalised over (and with q̄ including) particles that "shouldn't exist" under the hard-abort semantics; surviving weights no longer sum to 1.
- **MINOR — silent fallbacks (violate the repo's crash-loudly rule).** `except Exception` around `extract_posterior` falls back to streaming weights with only a warning (`propulate_abc.py:706-710`) — the theory's estimator can be silently swapped in downstream SBC. `extract_posterior` returns uniform weights when all log-weights are non-finite or no snapshot reconstructs (`abcpmc.py:1339-1340, 1383-1385`) — a flat prior masquerading as a posterior.
- **MINOR — live/replay archive divergence for failed simulations.** Live smooth path excludes `loss == inf` (`abcpmc.py:361`, `irange_key(None, inf, inclusive=(True, False))`) but `_reconstruct_archive`'s smooth branch admits them (`abcpmc.py:1203`, `accepted = prefix`) — a replayed archive can contain inf-loss members the live call never used (edge case: < k finite losses in a prefix).
- **MINOR — conflicting `amis_snapshots` defaults.** Wrapper default 0 (`propulate_abc.py:541`) vs ABCPMC default 20 (`abcpmc.py:491`) vs paper "S=20 by default". All experiment configs pin 20 so runs match the paper, but any config omitting the key silently gets S=0 legacy weighting *and* an n_proposals=1 retroactive denominator.
- **Paper drift (appendix):** covariance jitter is scale-aware in code, `1e-9·trace(cov)/d + 1e-12·mean_box_sq` with 1e3× escalation on Cholesky failure (`abcpmc.py:919, 926`), not the stated fixed λ=10⁻⁶ (tex:395). Truncation normalising mass is exact only for diagonal covariance — a diagonal-marginal approximation under the full weighted covariance actually used (docstring `abcpmc.py:122-127`), so "the proposal integrates to one on the support" (tex:395) is approximate for any d ≥ 2 benchmark. Bootstrap draws stamp no tolerance (`abcpmc.py:1031-1035`) vs. "each proposed individual stores the bandwidth" (tex:93). Edge-case semantics not in the paper: underflow retry loop (5 retries then weight=0, `abcpmc.py:1104-1175`); prior fallback with weight=1 after 1000 rejected box samples (`abcpmc.py:1122-1135`).

---

## 3. Baseline fairness (pyABC synchronous baseline, rejection ABC)

Files: `abc_smc_baseline.py` (the paper's `main_sync_baseline` per `method_registry.py:23-37`), `pyabc_wrapper.py` (near-identical secondary; targets `minimum_epsilon` instead of fixed generations — both collapse to "run until the clock" in wall-time mode), `_pyabc_common.py`, `_pyabc_history.py`, `pyabc_sampler.py`, `rejection_abc.py`. The baseline uses genuine pyABC 0.12.17 (`ABCSMC` + custom acceptor + `MappingSampler`); **no monkey-patching**.

### 3.1 CRITICAL (paper): "matched bandwidth schedule" is false

- **Paper claims** it in five places: abstract (tex:35), contribution 5 (tex:51), §5 (tex:174), Appendix A (tex:397), conclusion (tex:353) — "matched on the ABC kernel and bandwidth schedule (so the systems comparison turns on synchronization rather than the acceptance rule)."
- **Code:** the kernel *is* genuinely shared (`_pyabc_common.py:134` imports propulate's `_make_kernel`; acceptor accepts with p = clamp(K_ε(ρ),0,1), `_pyabc_common.py:163-170`; hard kernel falls back to `UniformAcceptor`, `_pyabc_common.py:129-130`; kernels peak-normalized `abcpmc.py:49-86`). **The bandwidth schedule is not:** both wrappers hardcode `pyabc.QuantileEpsilon(initial_epsilon=tol_init, alpha=0.5)` (`abc_smc_baseline.py:106`, `pyabc_wrapper.py:101`); there is no config hook — `scheduler_type`/`percentile` are never passed to pyABC. Meanwhile all paper experiments use `kernel:"gaussian"`, and under a smooth kernel the async ABCPMC **ignores `scheduler_type`** and selects ε by bisection to a kernel-weighted ESS-retention target of 0.95 (`abcpmc.py:671-677`; tex:393). Median-distance-quantile vs 95%-ESS-retention are structurally different schedules.
- **Impact:** the RQ2 throughput headline is largely insensitive (both arms simulate every proposal; ε mainly moves acceptance rates, not sim rate). The damage is to the *methodological control* and to the RQ1 posterior-quality/SBC comparison, which the ε schedule directly drives.
- **Fix options:** implement a config hook giving pyABC the same ε rule (best), or reword all five claim sites to "matched on the ABC kernel and acceptor" and disclose the differing ε adaptation.

### 3.2 MAJOR: baseline acceptor RNG correlated across MPI workers, run non-reproducible

- `_SmoothKernelAcceptor` closes over a single `rng = np.random.default_rng(seed)` (`_pyabc_common.py:136`; seeded with the raw replicate seed — not the BLAKE2b `stable_seed` the appendix (tex:403) implies for derived seeds). pyABC's `simulate_one` embeds the acceptor; `MappingSampler` cloudpickles it once per generation to every worker (`pyabc/.../mapping.py:118-123`), and `map_function` reseeds only the *global* `np.random`/`random` (`mapping.py:87-88`) — never this private Generator.
- **Consequences:** (i) within a replicate every worker draws the identical acceptance-uniform stream, and each generation re-pickles the same initial state (root's copy never advances) → acceptance decisions cross-worker-correlated, a statistical defect the async side does not have, and a plausible contributor to the baseline's reported SBC under-coverage (0.68/0.80 vs async 0.84/0.86) — i.e. it can bias the RQ1 comparison *toward* the async method. (ii) Because proposal sampling relies on the argument-less global reseed, the baseline run is **not reproducible from the config seed**.
- **Fix:** derive a per-worker, per-generation seed (e.g. `stable_seed(seed, rank, generation)`) inside the acceptor, or reseed the acceptor's Generator in the worker loop. Then consider re-running SBC for the baseline — its coverage may improve, which changes a reported number.

### 3.3 MAJOR (small N) / MINOR (large N): baseline loses one rank to coordination

- The MappingSampler root only dispatches, never simulates (`pyabc_sampler.py:106-190` dispatch vs `:210-247` worker loop). So "N workers" = **N−1 simulating ranks** for the sync baseline vs N for async (both arms launched with the same `srun -n $n_workers`, `jobs/scaling_single.sh:82,133`).
- ~25% handicap at the 4-worker LV point, ~6% at 16 — a real contributor to the "≈4× on one node" async lead; negligible at 48–384. Also caps sync utilization at (n−1)/n structurally (≤2% effect vs the reported 39–61% idle).
- **Fix:** give the baseline N+1 ranks, or disclose the master–worker overhead in §5/§7.

### 3.4 Verified claims (baseline fairness)

| Paper claim | Verdict | Evidence |
|---|---|---|
| Matched acceptor: probabilistic rejection with same peak-normalized kernel; hard kernel → UniformAcceptor | **Matches** (kernel genuinely shared) | `_pyabc_common.py:129-170`, `abcpmc.py:49-86` |
| Synchronous semantics: real generation barrier, workers busy within a generation, no artificial throttle | **Matches** | `MappingSampler.sample_until_n_accepted` maps n=k tasks and blocks (`mapping.py:123`); `CommWorldMap.map` dynamic one-at-a-time dispatch (`pyabc_sampler.py:149-190`); look-ahead cap `None` for mapping sampler (`pyabc_sampler.py:345-346`). Honest sync overhead: root does serial KDE fit / ε update / DB write between generations |
| Wall-clock budget with partial final generation counted | **Matches for throughput** | per-attempt JSONL trace on every worker (`_attempt_trace.py:43-60`); post-hoc trim to `wall_time ≤ budget` (`abc_smc_baseline.py:220-224`, `pyabc_wrapper.py:233-237`) mirrors async hard-abort (`propulate_abc.py:763-767`). Nuance: pyABC checks walltime only between generations and discards the incomplete population for the *posterior* (classical SMC behavior); the trace preserves the sims — if anything **generous** to the baseline (pre-deadline sims of a discarded generation still count). The paper's "serialises its current state" (tex:405) is loose wording for the sync arm |
| Population = worker count at scale, nothing caps parallelism below k | **Matches (via separate runs)** | `population_size=k` → `ABCSMC` (`abc_smc_baseline.py:105`); concurrency == k (`mapping.py:123`). The k=W runs live in cluster-side run dirs selected by `make_scaling_combined_fig.py:50,67` (`LV_FAIR`/`RW_FAIR`, k=w); the checked-in `scaling_fair_baseline.json` / `scaling_realistic_fair_baseline.json` have `k:[100]` — k is overridden at submit time → reproducibility caveat, not a bias |
| Throughput accounting parity | **Matches** | both arms count `record_kind == "simulation_attempt"` (all attempts incl. rejections) / elapsed span: `runtime_summary.py:201-203`, `scaling_runner.py:135-154`; async records every evaluated individual (`propulate_abc.py:740-758`); neither counts pre-simulation prior-support rejections. Symmetric |
| Rejection ABC reference correct | **Matches** | `rejection_abc.py:74-118`: uniform prior draws, accept ρ < tol, stop on max_sims/k/deadline; every draw counted |

### 3.5 Minor findings (baseline)

- **NaN-population-weight `AssertionError` swallowed:** both wrappers catch pyABC's assertion and continue with partial history as if it were an early wall-time stop (`abc_smc_baseline.py:136-145`, `pyabc_wrapper.py:134-143`) — violates crash-loudly; interacts with 3.2 (correlated RNG makes degenerate populations likelier) to silently understate baseline quality.
- **Progress miscount:** `abc_smc_baseline.py:227` reports `simulations=eval_count` which stays 0 under MPI (model runs on workers); `pyabc_wrapper.py:251` does it correctly. Cosmetic (metrics use the trace).
- **Missing pyabc → baseline silently skipped** with a warning (`utils/runner.py:660-674`, `straggler_runner.py:360-371`) — crash-loudly violation.

---

## 4. Experiment pipeline, configs, and metrics

### 4.1 HIGH: metric mislabeling — "Wasserstein to the analytic posterior" is W-to-truth

- `final_quality_wasserstein` computes W₁ between posterior samples and the **true-parameter point mass** (`runtime_summary.py:220-241` → `analysis/convergence.py:90-148`; 1-D case = mean |µᵢ − µ_true|), not distance to the analytic Gaussian posterior.
- Numerically verified: a posterior exactly equal to the analytic posterior scores ≈0.001 against the analytic posterior but ≈0.09 against the point mass. The paper's reported 0.07–0.09 band is exactly W-to-truth.
- Mislabeled in: param-bias figure + caption + text (tex:228, 233; `analyze_param_bias.py:40-46`), Gaussian-recovery figure axis (`make_gaussian_recovery_fig.py:53`) + caption (tex:306). The paper elsewhere correctly says "to the truth" (tex:182, 329, 338) — internally inconsistent.
- **Fix:** relabel (cheapest; numbers are internally consistent as W-to-truth) or actually compute W to the analytic posterior for the Gaussian-mean figures.

### 4.2 HIGH: param-bias mechanism attribution unsupported by the measurement

- **Paper claims** (tex:228, 350): the flat posterior error under runtime–parameter coupling occurs "because the AMIS importance weights … empirically cancel the over-representation."
- **Code:** the plotted metric is computed on the **unweighted** top-k archive — no weight column enters `_wasserstein_to_true_params` (`runtime_summary.py:220-241`, `convergence.py:449-476`).
- What the experiment shows: the *raw unweighted archive* stays accurate under coupling — evidence the over-representation doesn't bite at these levels, **not** evidence that the weights cancel it. Related contradiction: tex:115 claims "all reported posteriors and quality metrics are computed from" the retroactive estimator — the convergence/quality Wasserstein trajectories are not (they use the unweighted archive for async and the unweighted final population for sync; sync's SMC importance weights equally ignored).
- **Fix:** compute a weighted-posterior metric for the param-bias study (and ideally the recovery curves), or soften the attribution in §6 ¶param-coupled and Limitation (iv) to "the raw archive remains accurate; the retroactive weights provide an additional correction not needed at these coupling levels."

### 4.3 Config table vs repo (paper Appendix B, tex:409-429)

| Experiment | Paper | Repo | Verdict |
|---|---|---|---|
| Gaussian mean | 48/300 s/5/k=100 | `gaussian_mean.json` | ✓ |
| g-and-k | 48/600/5/100 | `gandk.json` | ✓ |
| Lotka–Volterra | 48/600/5/100 | `lotka_volterra.json` | ✓ |
| Realistic workload | 48/3600/5/100 | `realistic_workload.json` | ✓ |
| SBC | 100 trials | `sbc.json`: **`n_trials: 1000`** | **Mismatch** — in-flight SBC-1000 upgrade; checked-in config no longer reproduces published 100-trial numbers |
| Straggler | 16/300/5/100 | `straggler.json` | ✓ |
| Heterogeneity | 48/60/5/100 | `runtime_heterogeneity.json` | ✓ |
| LV scaling | 1–288, packed ×48 | `scaling.json`: `[1,4,16,48,128,256]` | **Mismatch** — published packed points (144/192/240/288) came from cluster-side runs (`lv_scaling_packed_20260630`, `lv_fair_baseline_20260701` read by `make_scaling_combined_fig.py:39-50`); only the w=144 fair config is checked in |
| Realistic scaling | 1–384 @ 1800 s | `scaling_realistic.json` max 96; no 384 config; `scaling_realistic_fillin.json` uses **900 s** | **Partial** — Fig. 5 caption says 1800 s for all points; the 1/4/16 fill-in points ran at 900 s (throughput ≈ budget-invariant, but the caption is strictly wrong) |
| Sensitivity | 48 workers | `sensitivity.json` has **no `n_workers`** | **Partial** — worker count = MPI world size at launch (48 only via `jobs/run_experiments.sh` `--ntasks=48`; a sharded submit computes ntasks=1 from the missing key, `jobs/submit_replicate_shards.py:321`) |
| Ablation | 48/300/5/100 | `ablation.json` | ✓ |

### 4.4 Verified claims (pipeline)

| Paper claim | Verdict | Evidence |
|---|---|---|
| Straggler: one worker slowed 1×/5×/10×/20×, identically for both methods | **Partial** | Injection is a post-evaluation additive sleep `slowdown_factor × 0.1 s` (`straggler_runner.py:71-86`), not a scaled simulator. Worker-slot mapping is asymmetry-aware and fail-fast (async rank 0 vs sync rank 1, since pyABC rank 0 is master; `straggler_runner.py:89-136`); sleep lands inside both methods' timed simulate intervals → symmetric. **Caveat:** factor 1 is *not* a no-straggler control — the 1× worker already sleeps 0.1 s/eval vs near-instant peers; "N×" is relative to the base sleep, not peer runtime. Sync has 15 simulating ranks vs async 16 (§3.3) |
| Heterogeneity: log-normal post-eval delay, σ sweep, 60 s | **Matches** | deterministic per-(seed, σ, worker, attempt) delay (`runtime_heterogeneity_runner.py:53-85`); σ ∈ {0, 0.5, 1, 1.5, 2}; idle fraction = 1 − Σbusy/(n_workers·span) (`runtime_summary.py:19-41`). Caveat: the generator script for `fig_hetero_quality.pdf` is not in the repo, so the plotted metric's provenance is unverifiable (posterior-mean-error path exists: `benchmark_reports.py:33-46`, `shard_finalizers.py:456-457`) |
| Param-bias delay d(µ)=min(d₀·e^{c(µ−µc)/ℓ}, d_max), exact constants, c ∈ {0,0.5,1,2} | **Matches** (design); metric label wrong (§4.1) | `runtime_heterogeneity_runner.py:88-112`; `configs/parameter_bias.json` |
| SBC: weighted ranks, equal-tailed coverage at 0.5/0.8/0.9/0.95, same observed data per trial for both methods | **Matches** | weighted resample then rank of truth (`analysis/sbc.py:14-38, 55-78`); async uses retroactive `posterior_weight` (fallback streaming; `sbc_runner.py:76-108`); sync uses pyABC population weights (`abc_smc_baseline.py:197`); per-trial observed data from trial seed, shared across methods (`sbc_runner.py:129-152`), inference seed differs per method (`:347`); sample counts matched (100 vs 100). **Caveats:** (a) figure bins 101 discrete rank values into 10 equal bins over [0,100] (`make_sbc_fig.py:64`) — top bin covers 11 values (~10% expected-count excess) and the 99% binomial band assumes p=1/10 exactly; (b) ranks pooled without checking `n_samples==100` per trial; (c) per-method trial dropout excluded (logged; potential selection effect if dropouts correlate with extreme θ; `sbc_runner.py:155-191`) |
| Throughput = sims/s within timed budget; retroactive estimator off the timed path; utilization = fraction of wall-clock in simulator; symmetric | **Matches** | attempt records both arms (`propulate_abc.py:740-758`; `_attempt_trace.py:36-62, 104-144`); post-deadline trim both arms; retroactive weights after the timed loop or disabled in scaling configs (`compute_posterior_weights: false`; `propulate_abc.py:543-552`); utilization identical formula both arms (`scaling_runner.py:451-463`) |
| LOCF shared time grid | **Matches** | `checkpoint_strategy="time_uniform"`, searchsorted-right minus one (`convergence.py:700-763`; `plotting/reporters.py:1116-1119, 1308-1311`). Nits: grid spans [global-min, max] not [0, max] as docstring says; when all grid points map to distinct rows, native wall-times are kept instead of grid times (`convergence.py:737-757`) |
| Sliced Wasserstein | **Partial** | 1-D exact W₁ (scipy); multi-D `ot.sliced_wasserstein_distance`, 50 projections, coordinate-wise fallback (`convergence.py:90-148`). **Weights never used** (see §4.2); **no fixed projection seed** → multi-D values non-deterministic across reruns |
| Seeding | **Matches** | `make_seeds` = `random.Random(base)`, unique non-negative 31-bit ints (`utils/seeding.py:38-64`); `stable_seed` = BLAKE2b(JSON) mod 2³¹ (`seeding.py:10-14`); shared benchmark instance → identical observed data across methods (`utils/runner.py:608-623`; benchmarks). Note: all five replicates share the same observed dataset (seed 42) — replicates vary inference only; the paper does not state this |
| Scaling design (packed ×48; sync k=W above 100; realistic 48→96→192→384) | **Matches via figure scripts + cluster data** | `make_scaling_combined_fig.py:42-67`, `make_realistic_util_fig.py:28-36`; medians+IQR as captioned. Reproducibility rests on scratch-path data + configs not fully checked in |

### 4.5 Other pipeline findings

- **MEDIUM — silent fallbacks** (see also §2.5, §3.5): realistic-workload simulate returns NaN on any failure, warns only above a 15% NaN rate (`benchmarks/realistic_workload.py:585-646, 666-678`); NaN attempts still count as throughput for both arms (symmetric, but inflates "simulations/s" if failures are frequent — worth a disclosed failure-rate number).
- **MEDIUM — reproducibility gaps:** `fig_hetero_quality.pdf`, `fig_hetero_idle.pdf`, `fig_straggler_throughput.pdf` have no in-repo generator (only committed PDFs; `.plans/HANDOFF.md:111-116`); straggler raw data lost (job cancelled); scaling figure scripts hardcode scratch mounts; `run_all_paper_experiments.py:87-99` omits `parameter_bias` and `scaling_realistic`. The Declarations promise scripts that "regenerate every figure and table from the deposited outputs" (tex:372, 378) — currently not true.
- **LOW — config name collision:** `configs/parameter_bias.json` reuses `experiment_name: "runtime_heterogeneity"` → output directory collision risk with the real heterogeneity experiment.
- **LOW — mixed summary statistics:** medians+IQR (scaling), mean±sd (param-bias, realistic-util), mean±95% CI (in-repo straggler plot) — each disclosed per caption, no cherry-picking evidence, but the straggler "median ≈965→311" quote cannot be re-derived from in-repo data (see also §5: the shipped straggler figure plots mean±CI while the text quotes medians).
- **OK — sample-count symmetry where it matters:** final-state comparisons use async top-k=100 vs sync population 100 (`analysis/final_state.py:89-132`); SBC likewise; the fair-baseline k=W asymmetry is deliberate and disclosed.
- **Tests:** `test_sbc.py`, `test_seeding.py`, `test_walltime_parity.py` pass (57 tests) under `sim_backend_venv/.venv`.

---

## 5. Figures and plotting: publication readiness

Context: sn-jnl `\textwidth` = 372 pt ≈ 13.1 cm (from `sn-article.log`). "Effective print font" = script font size × (fraction × 372 pt) / PDF page width. Springer guidance ≈ 8 pt lettering at final size (hard floor ~6 pt). All 15 referenced PDFs are pure vector (13–60 KB) — rasterization/DPI is a non-issue.

### 5.1 Blockers

1. **"Cellular Potts" panel titles inside two shipped PDFs** *(re-scoped 2026-07-05 — no scrub, CPM/NAStJA named in the final paper; see II.5)*: `fig_scaling_combined.pdf` has a panel titled "Cellular Potts (costly simulator)" while the paper-clean script says "Realistic workload" (stale PDF vs script), and `fig_posterior_recovery.pdf` has a "Cellular Potts" panel title **and no generator script exists anywhere** (must be written first — the real remaining gap). With CPM naming restored, this reduces to using one canonical name in captions and figure labels; the `run_cpm_*`/`scaling_cpm` scratch-path constants in `make_scaling_combined_fig.py:53-57` and `make_realistic_scaling_fig.py` are unobjectionable.
2. **Type 3 fonts in all 15 figures** (pdffonts: DejaVu Sans, `Type 3, Custom`, every file). Springer production commonly rejects Type 3. Nothing in the codebase sets `pdf.fonttype`. Fix: `matplotlib.rcParams["pdf.fonttype"] = 42` in a shared style module; regenerate everything.
3. **Five referenced figures have no in-repo generator:** `fig_straggler_throughput` (run-output `throughput_vs_slowdown.pdf` from `shard_finalizers.py:223`, manually renamed), `fig_hetero_idle` and `fig_hetero_quality` (nearest code `reporters.plot_idle_fraction_comparison` / `plot_quality_by_sigma`, `reporters.py:955, 983`, produce *different titles* than the committed PDFs → not reproducible as-is), `fig_posterior_recovery` (no matching code at all), `fig_sensitivity_heatmap` (`reporters.plot_sensitivity_summary` writes into run dirs; manually renamed). `replot.py` regenerates run-dir diagnostics, not paper PDFs.

### 5.2 Per-figure summary

| Figure | Generator | Page (pt) | \linewidth | Eff. font | Issues |
|---|---|---|---|---|---|
| fig_straggler_throughput | **none** | 425×279 | 0.60 | ~5.2 pt | Raw code keys as legend labels (`abc_smc_baseline`, `async_propulate_abc`); **colors inverted** (blue=sync, orange=async — `sorted()` order artifact, `shard_finalizers.py:228`); in-figure title; plots mean±95% CI while tex:206 quotes medians |
| fig_hetero_idle | **none** | 424×291 | 0.60 | ~5.3 pt | sync=orange (vs red elsewhere); in-figure title; caption (tex:219) says IQR bands, nearest code computes mean±95% CI (`reporters.py:983`) |
| fig_hetero_quality | **none** | 785×292 | 1.00 | ~4.7 pt | In-figure title; sync=orange; same IQR-vs-CI caption mismatch (tex:225) |
| fig_param_bias | `analyze_param_bias.py` | 651×265 | 0.92 | ~6.3 pt | Panel titles in figure; hardcoded scratch input (line 28) |
| fig_scaling_combined | `make_scaling_combined_fig.py` | 817×324 | 1.00 | ~5.0 pt (legend 4.3, annot. 3.6) | **STALE PDF: "Cellular Potts" panel title**; in-figure panel titles; cpm path tokens in script |
| fig_lv_timing | `make_lv_timing_fig.py` | 520×302 | 0.66 | ~6.1 pt (annot. 4.7) | In-figure title; literal `--` renders as two hyphens |
| fig_sbc_coverage | `make_sbc_fig.py` | 363×303 | 0.48 | ~5.9 pt | OK otherwise |
| fig_sbc_rank | `make_sbc_fig.py` | **495**×303 | 0.48 | **~4.3 pt** | Above-axes legend widens bbox → prints ~27% smaller than its side-by-side sibling |
| fig_posterior_recovery | **none** | 927×293 | 1.00 | **~4.0 pt** (worst) | **"Cellular Potts" panel title survives scrub**; sync=orange; in-figure titles; irreproducible |
| fig_realistic_recovery_diff | `make_realistic_diff_fig.py` | 374×274 | 0.55 | ~6.5 pt | In-figure title; "async − sync" label style diverges |
| fig_gaussian_recovery | `make_gaussian_recovery_fig.py` | 463×289 | 0.60 | ~5.8 pt (inset 3.8) | Inset far below floor; caption "replicate confidence bands" vague (data = 95% CI); y-label mislabeled (§4.1) |
| fig_realistic_corner | `make_realistic_corner_fig.py` | 432×403 | 0.62 | ~5.9 pt | **red-vs-green** (sync vs rejection) CVD pair, distinguished by color alone; reads old non-regenerable scratch runs (docstring admits) |
| fig_realistic_util | `make_realistic_util_fig.py` | 495×275 | 0.52 | ~4.7 pt (annot. 3.9) | Small print size |
| fig_ablation | `make_ablation_fig.py` | 778×317 | 1.00 | ~6.2 pt (inset 3.8) | **red/green/grey bars — CVD pair, color-only**; panel titles in figure |
| fig_sensitivity_heatmap | **none** | 644×380 | 1.00 | ~5.8 pt | In-figure suptitle; viridis ✓ CVD-safe |

### 5.3 Cross-cutting style findings

1. **No shared style module.** `async_abc/plotting/` has no rcParams/style file; each `make_*_fig.py` inlines its own partial `plt.rcParams.update` with base sizes drifting 11–13 pt (`make_scaling_combined_fig.py:138` = 11; `make_lv_timing_fig.py:57` = 13; legacy reporter figures = default 10). `STYLE = {...}` palette/label constants copy-pasted into 5+ scripts.
2. **Three conflicting color schemes** for the same two methods: blue #1f77b4 + red #d62728 (new scripts), blue + orange tab10 (hetero, posterior_recovery), and *inverted* blue=sync/orange=async (straggler).
3. **Five method-label variants:** "Asynchronous (ours)", "asynchronous (ours)", "asynchronous"/"synchronous baseline", "Synchronous baseline (population = cores)", raw keys `async_propulate_abc`/`abc_smc_baseline` (shipped straggler figure).
4. **Error-convention zoo:** IQR bands (scaling), 95% CI bands (straggler, gaussian, hetero-by-code), ±1 sd (param_bias, ablation-b, util), 95% CI bars (ablation-a), 99% binomial band (sbc_rank). Captions mostly declare which — but both hetero captions say "inter-quartile ranges" while the code computes mean±95% CI, and the straggler text quotes medians over a mean±CI plot.
5. **Figures never designed for print width:** figsize 5.0–11.5 in against a 5.14 in text block → LaTeX downscales 0.36–0.58× → effective fonts 4.0–6.8 pt everywhere (target ≥7–8 pt). Fix by designing at final width (figwidth = 5.14 in × \linewidth-fraction, 8–9 pt fonts).
6. **TeX-font mismatch:** DejaVu Sans + DejaVu mathtext for θ₁/W vs the paper's serif/CM math — caption math won't match figure math. Consider `mathtext.fontset="cm"` + serif family.
7. **Hardcoded scratch inputs:** every make script reads `/home/juhe/remotes/scratch/herold2/async-abc/...` (sshfs mount; JUWELS scratch is purge-prone) and writes an absolute `OUT` — no CLI args, no committed per-figure CSVs (the `save_figure` `_data.csv` mechanism exists in `plotting/export.py` but the make scripts don't use it). All figures become irreproducible if scratch purges.
8. **Orphans/bloat:** 13 unreferenced PDFs in `figures/` (~33 MB, incl. `fig_gandk_progress.pdf` 15.8 MB, `fig_lotka_progress.pdf` 14.8 MB); empty `benchmarks/`, `overview/`, `runtime/` placeholder dirs.
9. **In-figure titles on ≥8 figures** — journals want captions only (keep (a)/(b) panel tags).

### 5.4 Recommended figure work order

1. Write the five missing `make_*_fig.py` generators (fixes straggler labels/colors at the same time); regenerate `fig_scaling_combined.pdf` (scrub) and write + run a `make_posterior_recovery_fig.py` (scrub).
2. Create one shared style module (rcParams incl. `pdf.fonttype=42`, method colors/labels/markers, `save_paper_figure()` that also writes the `_data.csv`); port all make scripts to it; design at final print width with 8–9 pt fonts.
3. Fix CVD encodings (ablation red/green bars; corner-fig red/green fills — add hatching/markers or switch to an Okabe–Ito pair).
4. Reconcile caption↔code error-band statements (hetero IQR vs CI; straggler median vs mean) and standardize on one convention.
5. Parameterize data roots (CLI/env var); commit the small per-figure CSVs; purge orphan PDFs.

---

## 6. What checks out (verified positives)

- **Throughput accounting is genuinely symmetric:** both arms count every simulation attempt (including rejections) over the same elapsed-span definition; partial final generations are recorded for the baseline via the attempt trace; post-deadline work trimmed identically.
- **Delay injection (straggler/heterogeneity/param-bias) is applied identically** to both methods, with correct rank mapping (pyABC master vs worker) and fail-fast presence checks.
- **SBC ranks correctly use weighted resampling** for each method's own weight object; observed data per trial is identical across methods; sample counts matched.
- **Seeding matches the appendix** (make_seeds / stable_seed / shared observed data per replicate); per-eval sim seeds are stable hashes on both arms.
- **LOCF time-grid resampling** works as described.
- **Fair-baseline k=W scaling data is real** and drives the figures as captioned.
- **The retroactive estimator's chunked evaluation is correctly bit-identical** to unchunked (chunk-local logsumexp), and the estimator runs off the timed path (or is disabled in scaling configs) as claimed.
- **No pyABC monkey-patching**; the baseline is genuine pyABC 0.12.17 with within-generation dynamic dispatch keeping workers busy; no artificial throttling below the population size.
- **Relevant unit tests pass** (57: `test_sbc.py`, `test_seeding.py`, `test_walltime_parity.py`) under `sim_backend_venv/.venv`.

---

## 7. Prioritized action list

### Must fix before submission (referee-checkable contradictions / scrub blockers)

| # | Item | Fix type | Where |
|---|---|---|---|
| 1 | "Matched bandwidth schedule" false (§3.1) | **code — decided 2026-07-05** (matched-ε mode for pyABC; see II.1) | `abc_smc_baseline.py:106`, `pyabc_wrapper.py:101`; tex:35, 51, 174, 353, 397 |
| 2 | Retroactive estimator not exact cumulative mixture (§2.1) | code (`n_proposals` = full history for reported runs) **or** text (rewrite tex:115/171; add prior floor to Eq. 7 description) | `abcpmc.py:1318-1371`; tex:115, 171 |
| 3 | "Wasserstein to analytic posterior" mislabeled (§4.1) | text/labels (numbers are consistent as W-to-truth) | tex:228, 233, 306; `make_gaussian_recovery_fig.py:53`; `analyze_param_bias.py` |
| 4 | Param-bias "AMIS weights cancel it" unsupported (§4.2) | code (weighted metric) **or** text (soften §6 + Limitation iv; fix tex:115 "all quality metrics") | `runtime_summary.py:220-241`, `convergence.py:449-476`; tex:228, 350, 115 |
| 5 | Figure/caption naming mismatch + missing generator (§5.1; **re-scoped — no scrub, CPM/NAStJA named in final paper**) | pick canonical naming (II.5.1), regenerate, write missing generator | figures dir; `make_scaling_combined_fig.py`; new `make_posterior_recovery_fig.py` |
| 6 | Baseline acceptor RNG correlated + non-reproducible (§3.2) | code (per-worker/per-generation seed); consider SBC rerun for baseline | `_pyabc_common.py:136`; `mapping.py:87-88` |
| 7 | Type 3 fonts in all figures (§5.1) | code (`pdf.fonttype=42` in shared style) + regenerate | new shared style module |

### Should fix (soundness / honesty of framing)

- Statelessness/"exactly reproducible"/buffer-determinism qualifications (§2.2); "used only for ESS" wording (§2.3). — tex:85, 113, 115.
- Baseline N−1 simulating ranks: disclose or run baseline with N+1 ranks (§3.3).
- Geometric-decay cached-path MPI bug (§2.5) — fix regardless of paper.
- Crash-loudly violations: `extract_posterior` fallback, pyABC NaN-assert swallow, missing-pyabc skip (§2.5, §3.5, §4.5).
- Straggler 1× is not a no-straggler control — state the base-sleep semantics (§4.4).
- SBC config drift (1000 vs 100 trials), missing scaling configs (packed LV, realistic 192/384), fill-in budget 900 s vs caption 1800 s, sensitivity `n_workers` unpinned (§4.3).
- Missing figure generators (5) + hardcoded scratch paths + per-figure CSVs (§5.1, §5.3.7) — required to honor the Declarations' reproducibility promise (tex:372, 378).
- Verify `cornuet2012amis` theorem citation; consider Marin–Pudlo–Sedki (§1.2a).

### Nice to have

- Appendix drift: jitter formula, truncation-mass approximation, removed r_lo, bootstrap tolerance stamp (§2.5).
- `low_rate`/`expand_factor` latent TypeError (§2.5); `amis_snapshots` default unification (§2.5).
- Wall-time filter vs posterior-weight normalization leak (§2.5).
- Fixed projection seed for sliced Wasserstein (§4.4); SBC binning 101→10 bins edge (§4.4); LOCF grid nits (§4.4).
- Figure style unification: shared module, consistent colors/labels, CVD-safe ablation/corner palettes, print-width design at 8–9 pt, no in-figure titles, error-convention standardization, orphan-PDF purge, TeX-matching math fonts (§5.3–5.4).
- State in the appendix that replicates share one observed dataset (seed 42) and vary inference only (§4.4).
- Disclose realistic-workload NaN/failure rate; `parameter_bias` experiment_name collision; add `parameter_bias`/`scaling_realistic` to `run_all_paper_experiments.py` (§4.5).

---
---

# Part II — Precise fix instructions

## II.0 Conventions and order of operations

> **Decision log (2026-07-05):** the Propulate propagator has been further improved since the experiments were last run, so **all experiments rerun regardless**. Consequently: the matched-ε **code fix (II.1) is chosen** over the text fix; every behavior- or analysis-affecting code change must land **before** the rerun campaign (reruns are the expensive step — anything not in the binary that produces the final data forces a repeat); and every number quoted in the paper is stale by construction until re-derived from the new runs.

> **Decision log (2026-07-05, addendum — no scrub):** the `paper-clean` scrub was a **test only**; the final paper **will name NAStJA and the Cellular Potts model** and cite them. All scrub items in this document are re-scoped to *naming consistency* (one canonical benchmark name everywhere: text, captions, figure labels, configs) — see II.5.
> **Branch consequence:** do **not** continue from old `main` — it lacks the entire review-3/4 revision line (`245dc6c`…`4fd79d7`) and the newer configs (`parameter_bias.json`, `scaling_realistic*.json`, `scaling_fair_baseline.json` exist only on `paper-clean`). Branch the campaign from the pre-scrub baseline **`4fd79d7`** ("baseline: carry review-4 working state onto paper-clean"), which is the latest working state with original CPM/NAStJA naming; `paper-clean`'s five scrub commits are then simply not carried forward. Carry over the uncommitted `experiments/jobs/*` edits from the current working tree.
> **Path mapping for this document:** the review was conducted on `paper-clean`, so it uses the scrubbed names throughout. On the pre-scrub baseline map: `benchmarks/realistic_workload.py` ↔ `benchmarks/cellular_potts.py`, `configs/realistic_workload.json` ↔ `configs/cellular_potts.json`, `make_realistic_*` ↔ `make_cpm_*`, `scaling_realistic*` ↔ `scaling_cpm*`, "realistic workload" ↔ "Cellular Potts (NAStJA)". All file:line findings refer to code that is identical up to these renames (the scrub renamed, it did not change logic).

- **Python:** `sim_backend_venv/.venv/bin/python` for all local runs and tests.
- **Test loop after every code change:** `sim_backend_venv/.venv/bin/python -m pytest experiments/tests -x -q` (the SBC/seeding/walltime subset is fast; run the full suite before committing).
- **Order of operations (revised for the full-rerun plan):**
  1. **Code fixes, all before any job is submitted:** II.1 matched-ε mode; II.6 acceptor RNG + worker reseed; II.8.3 geometric-decay cache; II.8.4 crash-loudly; II.9.3 dead scheduler kwargs; II.9.4 `amis_snapshots` default; II.9.5 wall-time filter ordering; II.2.5 `posterior_n_proposals` plumb; II.4.2 weighted quality metric; II.3.b analytic-posterior metric; II.9.6–II.9.8 analysis fixes. Run the full test suite green, then freeze the commit the campaign runs from.
  2. **Config updates (II.8.5, II.8.6):** straggler factor-0 control, sensitivity `n_workers`, `parameter_bias` experiment name, packed/fair scaling configs checked in, SBC at 1000 trials.
  3. **Rerun campaign** (matrix in II.0.a) via the jsc-mpc MCP flow (`sync_code` → `submit_job`/`submit_sweep`; production command + scratch path per `reference_asyncabc_cluster_deploy` memory). `estimate_cost` the scaling sweeps first — the realistic-workload scaling dominates the budget.
  4. **Analysis + figure infrastructure (II.7):** style module, the five missing generators, vendored per-figure CSVs — all 15 figures regenerated once from the new data.
  5. **Paper text edits last** (II.1.b, II.2–II.4, II.8), including re-deriving every quoted number (II.10.6).
- **Text-edit conventions:** line numbers refer to `latex/sn-article-template/sn-article.tex` as reviewed (438 lines). Recompile and re-check line refs after each batch of edits.

### II.0.a Rerun matrix

Everything in App. B Table 5 reruns with the new propagator + matched-ε baseline + fixed acceptor RNG. Per-experiment notes:

| Experiment | Config | Scale / budget | Rerun notes |
|---|---|---|---|
| Gaussian mean | `gaussian_mean.json` | 48 w, 300 s, 5 reps | also feeds the II.2.5 m-convergence check and the II.3.b analytic-W metric |
| g-and-k | `gandk.json` | 48 w, 600 s, 5 reps | — |
| Lotka–Volterra | `lotka_volterra.json` | 48 w, 600 s, 5 reps | — |
| Realistic workload | `realistic_workload.json` | 48 w, 3600 s, 5 reps | record the NaN/failure rate for II.9.10 |
| SBC | `sbc.json` | 48 w, 300 s, **1000 trials** | go straight to SBC-1000 (II.8.6); Table 3 + the MC-error sentence are rewritten from it |
| Straggler | `straggler.json` | 16 w, 300 s, 5 reps | add the factor-0 control first (II.8.5); replaces the lost raw data; median+IQR figure (II.7.b) |
| Runtime heterogeneity | `runtime_heterogeneity.json` | 48 w, 60 s, 5 reps | new generators plot median+IQR so the captions become true (II.7.b) |
| Parameter bias | `parameter_bias.json` | 48 w, 60 s, 5 reps | confirm `compute_posterior_weights` stays enabled (II.4.1); weighted metric now emitted by the analysis (II.4.2) |
| LV scaling + packed + fair | `scaling.json`, new `scaling_packed.json`, `scaling_fair_baseline.json` | 1–288 w, 900 s | check in the packed/fair configs first (II.8.6); include the instrumented LV timing run (`scaling_timing.json`) for Fig. 4 |
| Realistic scaling + fill-in + fair | `scaling_realistic*.json` | 1–384 w, 1800 s | rerun the sub-node fill-in points at 1800 s so the Fig. 5 caption caveat disappears (II.8.6) |
| Sensitivity | `sensitivity.json` | 48 w, 300 s, 5 reps | pin `n_workers: 48` first (II.8.6) |
| Ablation | `ablation.json` | 48 w, 300 s, 5 reps | exercises the hard-kernel path — the geometric-decay cache fix (II.8.3) must be in |

---

## II.1 Matched bandwidth schedule (must-fix #1)

**Decision (2026-07-05): code fix.** All experiments rerun anyway (II.0), so the baseline gets a genuinely matched ε rule, and the paper's "matched bandwidth schedule" claim becomes true in the only sense the two execution models allow: **same selection rule, applied at each model's natural granularity** (per generation for the baseline, per arrival for the asynchronous method). The former text-fix option is superseded; the residual text edits are the granularity disclosure in II.1.b.

### II.1.a Implementation

1. **Refactor the async ε-selection into a pure function.** In `propulate/propulate/propagators/abcpmc.py`, extract the kernel-aware selection core (geometric grid scan + bracketed bisection, ~:1524–1604) into a module-level function:
   `select_eps_by_ess_retention(losses: np.ndarray, weights: np.ndarray, eps_current: float, kernel_fn, retention: float = 0.95, max_tighten_factor: float = 0.5) -> float`
   The existing kernel-aware scheduler calls it; the per-call throttle (`_calls_since_bisect`/`_held_eps`, :1640–1646) stays in the scheduler, **outside** the pure function. This is a behavior-preserving refactor — existing abcpmc tests must stay green before anything else changes.
2. **pyABC Epsilon subclass.** In `experiments/async_abc/inference/_pyabc_common.py`, add
   `make_matched_epsilon(kernel: str, tol_init: float, retention: float = 0.95) -> pyabc.epsilon.Epsilon`:
   subclass `pyabc.epsilon.Epsilon` mirroring `QuantileEpsilon`'s structure (pyABC 0.12.17: `initialize(t, get_weighted_distances, ...)`, `update(t, ...)`, `__call__(t)`). In `update`, pull the generation's weighted distances (DataFrame with `distance` and `w` columns), call `select_eps_by_ess_retention(distances, weights, eps_t, kernel_fn, retention)`, and clamp `eps_{t+1} = min(eps_t, proposal)` for monotonicity. `initialize` returns `tol_init`. Import the kernel via the same `_make_kernel` already used by `make_acceptor` (`_pyabc_common.py:133`), so the kernel and the ε rule share one definition on both arms.
3. **Wire with a config gate.** In `abc_smc_baseline.py:106` and `pyabc_wrapper.py:101`, replace the hardcoded `pyabc.QuantileEpsilon(initial_epsilon=tol_init, alpha=0.5)` with:
   ```python
   epsilon_mode = inference_cfg.get("epsilon_mode", "matched" if kernel != "hard" else "quantile")
   eps = (
       make_matched_epsilon(kernel, tol_init, retention=inference_cfg.get("ess_retention", 0.95))
       if epsilon_mode == "matched"
       else pyabc.QuantileEpsilon(initial_epsilon=tol_init, alpha=0.5)
   )
   ```
   The async side reads the same `ess_retention` key (default 0.95; add it to the ABCPMC kwargs plumbing in `propulate_abc.py` next to `scheduler_kwargs` if not already configurable), so one config value governs both arms. The hard-kernel fallback keeps `QuantileEpsilon` — only the async-only ablation uses hard kernels, so the baseline never runs mismatched.
4. **Tests** (new `experiments/tests/test_matched_epsilon.py`):
   - *Rule identity (the "matched" property):* for a fixed batch of (distance, weight) pairs and a current ε, the subclass's `update` result equals `select_eps_by_ess_retention(...)` called directly — asserted cross-arm against the propulate function, not a reimplementation.
   - *Monotonicity:* over 5 synthetic generations the ε sequence is non-increasing and starts at `tol_init`.
   - *Integration:* 3 generations of the baseline on gaussian-mean with `epsilon_mode="matched"` run end-to-end and record a monotone per-generation ε.
   - *Gate:* `kernel="hard"` yields `QuantileEpsilon`; `epsilon_mode="quantile"` forces the legacy rule.
5. **What remains structurally different (by design — disclose, don't hide):** the baseline applies the rule once per generation to that generation's accepted distances; the asynchronous method applies it per arrival over its sliding history window. The *rule* is matched; the *granularity* cannot be — that granularity difference is precisely the synchronization variable under test.

### II.1.b Residual text edits (after the code fix, wording may keep "matched bandwidth schedule")

1. **tex:174 (§5):** after "The acceptor and kernel are thus matched across methods;" insert:
   > "the tolerance schedule is matched at the level of the selection rule — both sides choose $\epsilon$ by the same kernel-weighted ESS-retention criterion (target $0.95$, Appendix~\ref{app:implementation}) from the same $\epsilon_0$, applied at each execution model's natural granularity: once per generation for the baseline, per arrival for the asynchronous method."
2. **tex:397 (App. A, matched acceptor paragraph):** add a closing sentence:
   > "The baseline's tolerance schedule is the same ESS-retention rule as the asynchronous side's kernel-aware scheduler (see \emph{Tolerance schedulers} above), evaluated once per generation on the accepted population's weighted distances and clamped monotone."
3. **tex:35 (abstract), tex:51, tex:353:** the "matched on the ABC kernel and bandwidth schedule" wording can now stand as written; re-read each after the rerun to confirm no number-bearing clause changed.

---

## II.2 Retroactive estimator exactness (must-fix #2)

The full cumulative mixture is O(n²k) for fast simulators (n ≈ 10⁷) — computationally out of reach, so this is a **text fix plus an empirical m-convergence check**; do not chase exactness.

1. **tex:115, weight object (iii):** replace "in which every evaluated particle is reweighted by $\pi(\theta_i)K_{\epsilon_n}(\rho_i)/\bar q_n(\theta_i)$ with $\bar q_n$ the cumulative proposal mixture over the \emph{full} evaluated history" with:
   > "in which every evaluated particle is reweighted by $\pi(\theta_i)K_{\epsilon_n}(\rho_i)/\bar q_n(\theta_i)$, where $\bar q_n$ is a draw-proportional deterministic mixture of $m \le S$ past proposals reconstructed from history, plus a defensive prior component with mass floored at $0.5/(m{+}1)$ so the weights stay bounded on the whole box — the Condition~\ref{cond:c4} approximation to the cumulative proposal mixture over the full evaluated history"
2. **tex:171 (theory-vs-implementation paragraph):** replace "The reported posterior evaluates exactly this object retroactively (\S\ref{sec:method-amis}); the approximations we introduce for efficiency both sit on the \emph{online} path" with:
   > "The reported posterior evaluates a bounded-$m$ retroactive approximation of this object (\S\ref{sec:method-amis}): the cumulative-mixture denominator is reconstructed from $m \le S$ history-truncated proposals with draw-proportional weights and a defensive prior floor. Together with the online sliding buffer and the archive-locality of the proposal, this makes three approximations relative to the idealized estimator"
   and adjust the following sentences ("First… Second…") to "First … Second … Third, the retroactive denominator uses $m \le S$ reconstructed proposals rather than all $n$; Condition~\ref{cond:c4} is exactly the assumption that this bounded mixture tracks the full one."
3. **tex:399 (App. A, bounded-memory paragraph):** after the chunking sentence, add: "The denominator mixture itself uses $m \le S$ reconstructed proposals with draw-proportional weights and a prior component floored at $0.5/(m{+}1)$ (the Condition~4 approximation); the chunking changes memory, not the estimate."
4. **Eq. (7) context (tex:155-158):** no equation change needed — it defines the *idealized* estimator; the text edits above scope it.
5. **Empirical m-convergence check (cheap, laptop-side):** add a config key `posterior_n_proposals` in `run_propulate_abc` (`propulate_abc.py`, next to `compute_posterior_weights` at :552) and pass it through to `extract_posterior(population, n_proposals=...)` (`abcpmc.py:1318` already accepts it). This plumb lands **before** the campaign (II.0 step 1). After the campaign, load one fresh gaussian-mean history and compute weights at m ∈ {5, 20, 50, 100, 200}; report `0.5·Σ|w^{(m)} − w^{(200)}|` (total-variation distance) and the SBC-relevant quantiles. If TV < ~1e-2 by m=20, add one sentence to §6.2: "Increasing the retroactive mixture size $m$ beyond $S{=}20$ changes the reported weights by less than X in total variation on the Gaussian-mean benchmark." This turns the approximation into a measured non-issue.

---

## II.3 "Wasserstein to the analytic posterior" mislabeling (must-fix #3)

Relabel everywhere; the numbers are internally consistent as W-to-truth. Do **not** recompute unless you want the (better) analytic-posterior metric — instructions for both.

### II.3.a Relabel (required regardless)

1. **tex:228:** "the asynchronous posterior error (Wasserstein to the analytic Gaussian posterior)" → "the asynchronous posterior error (Wasserstein distance to the true parameter, the same metric as \S\ref{sec:experiments-suite})".
2. **tex:233 (fig_param_bias caption):** "(b) Posterior error (Wasserstein to the analytic posterior)" → "(b) Posterior error (Wasserstein distance to the true parameter)".
3. **tex:306 (fig_gaussian_recovery caption):** "Wasserstein distance to the analytic posterior versus wall-clock time" → "Wasserstein distance to the true parameter versus wall-clock time". Check the inset text mentions "rejection-ABC reference ($W\approx2.4$)" — that value is also W-to-truth; keep consistent.
4. **`experiments/scripts/make_gaussian_recovery_fig.py:53`:** change the y-label string to `"Wasserstein distance to true $\\mu$"`.
5. **`experiments/scripts/analyze_param_bias.py`:** change the panel-(b) y-label/title strings to "Wasserstein to true parameter" (grep the file for "analytic").
6. Grep the tex for remaining instances: `grep -n "analytic" latex/sn-article-template/sn-article.tex` — keep "analytic posterior" only where the metric genuinely is the analytic posterior (SBC references, posterior-mean error at tex:214/289, App. B tex:431) and reword the rest.
7. The figures regenerate from the rerun data in the II.7 pass anyway; only the label strings need changing here.

### II.3.b Recommended now (rerun makes it free): real W-to-analytic-posterior for Gaussian-mean figures

Add to `experiments/async_abc/analysis/convergence.py` a sibling of `_wasserstein_to_true_params` (convergence.py:90):
`_wasserstein_to_analytic_posterior(frame, analytic_samples: np.ndarray) -> float` — 1-D exact `scipy.stats.wasserstein_distance(samples, analytic_samples)` where `analytic_samples` are ~10⁴ draws from the benchmark's closed-form posterior (the gaussian_mean benchmark knows µ_post/σ_post; expose a `analytic_posterior_samples(n, seed)` helper on `benchmarks/gaussian_mean.py`). Wire it as a new column (`final_quality_wasserstein_analytic`) in `runtime_summary.py` next to :220-241 — do NOT replace the existing column (other experiments rely on it). Land this **before** the campaign (II.0 step 1) so the rerun summaries emit the column directly; the Gaussian-mean figures can then report the honest analytic-posterior metric instead of merely relabeling (II.3.a still applies to the param-bias figure and any remaining W-to-truth labels).

---

## II.4 Param-bias "AMIS weights cancel it" attribution (must-fix #4)

The param-bias experiment reruns in the campaign, so build the weighted metric into the analysis **first** and let the rerun emit it directly.

1. **Before the campaign:** confirm `configs/parameter_bias.json` does not set `compute_posterior_weights: false` (it currently doesn't — the default true stands), so the rerun records carry `posterior_weight` for the async arm.
2. **Add the weighted metric (analysis code, lands pre-campaign — II.0 step 1):**
   - In `convergence.py`, add `_weighted_resample(frame, weight_col, n=500, seed)` → resample rows with replacement, p ∝ weight (guard: renormalize, drop NaN weights).
   - In `runtime_summary.py:220-241`, compute an additional `final_quality_wasserstein_weighted`: async → resample by `posterior_weight`; sync → resample by pyABC population `weight` (already in records via `abc_smc_baseline.py:197`); then call `_wasserstein_to_true_params` on the resample.
   - In `analyze_param_bias.py`, plot the weighted metric in panel (b) (or both, unweighted as hollow markers).
   - If weighted ≈ unweighted across c: the paper's conclusion survives with an honest mechanism sentence (see 3, variant A). If they differ: the paper's numbers change — re-write the paragraph from the new data.
3. **Text edits:**
   - **Variant A (expected path — weighted metric from the rerun agrees with unweighted):** tex:228 "because the AMIS importance weights, which divide by the cumulative proposal density, empirically cancel the over-representation" → "and the weighted posterior (AMIS weights, which divide by the cumulative proposal density) agrees with the unweighted archive, confirming the over-representation does not distort the reported posterior at these coupling levels". Mirror at tex:350 (Limitation iv).
   - **Variant B (fallback if the weighted metric diverges or is unusable):** tex:228 → "indicating that at these coupling levels the over-representation does not measurably distort the archive from which the posterior is built; the AMIS importance weights, which divide by the cumulative proposal density, provide an additional correction on top of this raw robustness". Mirror at tex:350. If weighted and unweighted genuinely diverge on the new data, the paragraph's conclusion changes — rewrite it from the data, don't paper over it.
   - **Either way, fix tex:115:** "all reported posteriors and quality metrics are computed from it" is false (the convergence Wasserstein uses the unweighted archive). Replace with: "the reported posterior densities and calibration metrics (SBC) are computed from it; the Wasserstein-vs-time quality curves are computed on the top-$k$ archive (asynchronous) and final population (synchronous), matched in size."

---

## II.5 Benchmark naming consistency + missing generator (must-fix #5 — re-scoped, no scrub)

**Re-scoped 2026-07-05:** the final paper names NAStJA/CPM, so there is nothing to scrub. What remains is (a) picking one canonical name and using it everywhere, (b) the genuinely missing figure generator, (c) the data vendoring for reproducibility.

1. **Pick the canonical naming and citations.** Recommended: introduce once as "a production Cellular Potts model (CPM) tissue simulation in the NAStJA framework" with citations (NAStJA: Berghoff et al.; CPM: Graner & Glazier 1992), then use "Cellular Potts" consistently in section headers, captions, figure labels, and Table 1/5 rows. Rename θ₁/θ₂ to the actual CPM parameter names — this materially improves the weak-identifiability discussion (tex:320), which currently has to say "both chiefly modulate the same summary statistic" in the abstract. Un-scrubbing the paper text means reverting to (or continuing from) the pre-scrub tex at `4fd79d7` and porting any post-scrub text improvements manually.
2. **`fig_scaling_combined.pdf`:** regenerate in the II.7 pass from the campaign data; with CPM naming restored, the panel label and caption agree again — just make sure both use the canonical name from step 1 (and in-figure titles go away anyway, II.7.b).
3. **`fig_posterior_recovery.pdf`:** write `experiments/scripts/make_posterior_recovery_fig.py` (it does not exist under any name — checked both naming schemes):
   - Three panels (g-and-k, Lotka–Volterra, Cellular Potts), each: per-method median Wasserstein-vs-time with IQR band over replicates, on the LOCF grid.
   - Data source: the per-run quality-over-time CSVs produced by `posterior_quality_curve` (`convergence.py:151`) in each benchmark run dir. Model the loading logic on `make_realistic_diff_fig.py` (pre-scrub name: the CPM diff-figure script), which already reads the CPM pair of curves — reuse its loader for panel 3 and generalize for panels 1–2.
   - Panel labels "(a) g-and-k", "(b) Lotka–Volterra", "(c) Cellular Potts" — as axis text tags, not titles.
4. **Vendor the figure data (reproducibility, §5.3.7):** copy each figure's input CSVs into `experiments/data/paper_figures/<fig_name>/`, commit them, and point the make scripts at the in-repo copies (scratch paths only behind an optional `--refresh-from-scratch` flag). The `run_cpm_*` / `scaling_cpm` scratch directory constants in `make_scaling_combined_fig.py:53-57` are real remote dir names and now unobjectionable — leave them under the refresh flag.
5. **Verify consistency:** `grep -rin "realistic.workload\|realistic workload" latex/ experiments/scripts/` after the un-scrub — the scrubbed framing should be gone from paper-facing text and figure labels; `pdftotext` sweep over the regenerated `figures/*.pdf` confirms every figure uses the canonical name.

---

## II.6 Baseline acceptor RNG (must-fix #6)

**Goal:** per-worker, per-generation decorrelated acceptance draws, reproducible from the config seed.

1. In `experiments/async_abc/inference/_pyabc_common.py`, rework `make_acceptor` (currently `_pyabc_common.py:101-172`):
   - Delete the closure `rng = np.random.default_rng(rng_seed)` (line 136).
   - Give `_SmoothKernelAcceptor` a per-process, per-generation generator derived inside `__call__`:
     ```python
     from ..utils.seeding import stable_seed   # BLAKE2b-derived

     def _rng_for(self, t: int):
         # Re-derived after every unpickle (worker receives a fresh copy per
         # generation): key on (base seed, MPI rank, generation) so every
         # worker/generation gets an independent, reproducible stream.
         if self._rng is None or self._rng_t != t:
             rank = _get_mpi_rank()          # reuse the helper propulate_abc.py uses; 0 if no MPI
             self._rng = np.random.default_rng(stable_seed(rng_seed, "acceptor", rank, t))
             self._rng_t = t
         return self._rng
     ```
     and in `__call__`: `accept = bool(self._rng_for(t).random() < p_accept)`. Initialize `self._rng = None; self._rng_t = None` in `__init__` (both survive cloudpickle as plain attributes; after unpickle on a worker the first call re-derives with that worker's rank — exactly the desired keying).
   - This also satisfies the appendix's stable-seed wording (tex:403): the acceptor stream is now BLAKE2b-derived.
2. **Reproducibility of the proposal side (same file family):** `abc_smc_baseline.py:113` does `np.random.seed(seed % 2**31)` on the root, but workers reseed the global RNG from OS entropy in the map wrapper. For full config-seed reproducibility, seed the worker loop: in `pyabc_sampler.py`'s worker loop (`:210-247`), on receiving work for generation `t`, call `np.random.seed(stable_seed(run_seed, "worker-global", rank, t) % 2**32)` once per (rank, t). Plumb `run_seed` into `CommWorldMap` at construction. (If you skip this, note in App. B that baseline proposal sampling is not seed-reproducible; but it's a 10-line fix — do it.)
3. **Tests:** add `experiments/tests/test_acceptor_rng.py`:
   - two acceptor copies (simulating two workers) at the same `t` with different ranks produce different acceptance streams;
   - the same (seed, rank, t) reproduces the same stream after a pickle/unpickle round-trip;
   - hard kernel still returns `UniformAcceptor`.
4. **Reruns:** subsumed by the full campaign (II.0.a) — just ensure this fix is merged into the frozen campaign commit before the first baseline job is submitted. Expect the baseline SBC coverage numbers (Table 3, tex:276) to move; Table 3, tex:267, and the abstract's SBC sentence are rewritten from the SBC-1000 rerun regardless (II.8.6).

---

## II.7 Figures: Type 3 fonts, shared style, missing generators (must-fix #7 + §5 items)

### II.7.a Shared style module

Create `experiments/async_abc/plotting/paper_style.py`:

```python
"""Single source of truth for paper-figure styling. Import in every make_*_fig.py."""
import matplotlib

TEXTWIDTH_IN = 5.147          # sn-jnl \textwidth = 372 pt / 72.27
COLORS = {                    # Okabe–Ito, CVD- and grayscale-safe
    "async":     "#0072B2",   # blue
    "sync":      "#D55E00",   # vermillion
    "rejection": "#009E73",   # bluish green  (add hatch/marker redundancy where filled)
}
LABELS = {
    "async":     "Asynchronous (ours)",
    "sync":      "Synchronous baseline",
    "rejection": "Rejection ABC",
}
MARKERS = {"async": "o", "sync": "s", "rejection": "^"}

RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, no Type 3
    "font.family": "serif", "mathtext.fontset": "stix",  # matches sn-mathphys Times-like text
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}

def apply():
    matplotlib.rcParams.update(RC)

def fig_size(width_frac: float, aspect: float = 0.68):
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)

def save_paper_figure(fig, name: str, data: dict | None = None):
    """Save PDF into latex figures dir + input data CSVs into experiments/data/paper_figures/<name>/."""
    ...  # reuse plotting/export.py's _data.csv mechanism
```

Rules encoded by this module: **figures are drawn at final print width** (`fig_size(0.6)` for a `0.6\linewidth` figure) so an 8 pt label prints at 8 pt; no `figsize=(10,4)`-then-downscale.

### II.7.b Port and write generators

1. Port the 10 existing `make_*_fig.py` scripts: delete their local `STYLE`/`rcParams` blocks, `import paper_style as ps; ps.apply()`, use `ps.COLORS/LABELS/MARKERS/fig_size/save_paper_figure`. Remove all in-figure `suptitle`/axes titles (keep "(a)"/"(b)" corner tags via `ax.text`). Fix `make_lv_timing_fig.py`'s literal `--` (use an en-dash or math minus).
2. Write the five missing generators (all read the vendored CSVs from II.5.4; add `--refresh-from-scratch` to re-pull):
   - `make_straggler_fig.py` — throughput vs slowdown factor per method. **Data:** the campaign's straggler rerun (the old raw data is lost anyway; the rerun includes the factor-0 no-straggler control, II.8.5). Plot **median + IQR** to match tex:206 (which quotes medians — update the quoted numbers from the rerun), with proper labels (fixes the raw `abc_smc_baseline` legend keys and the inverted colors).
   - `make_hetero_fig.py` — two outputs: `fig_hetero_idle.pdf` (idle fraction vs σ) and `fig_hetero_quality.pdf` (two panels: posterior-mean error vs σ; sims completed vs σ). Base the data loading on `reporters.plot_idle_fraction_comparison` / `plot_quality_by_sigma` (`plotting/reporters.py:955-983`) but compute **median + IQR** so the captions (tex:219, 225, "inter-quartile ranges") become true — or, if you prefer keeping mean±CI, edit both captions instead. Pick one convention and apply it to captions and scripts together.
   - `make_posterior_recovery_fig.py` — see II.5.3.
   - `make_sensitivity_fig.py` — extract the heatmap-grid logic from `reporters.plot_sensitivity_summary` into a paper-facing script writing `fig_sensitivity_heatmap.pdf`, without the suptitle.
3. `make_sbc_fig.py`: export both panels at identical page width — either put the rank-histogram legend inside the axes (so the tight bbox matches `fig_sbc_coverage`) or compose both panels into one figure with two subplots and reference it once in the tex. Also fix the binning (see II.9.6).
4. CVD fixes: `make_ablation_fig.py` — recolor the three bar classes to Okabe–Ito + add hatching (`hatch="//"` for removals); `make_realistic_corner_fig.py` — sync/rejection fills get distinct hatch or contour linestyles in addition to color.
5. **Regenerate all 15 referenced figures**, then verify:
   - `for f in latex/sn-article-template/figures/fig_*.pdf; do pdffonts "$f" | grep -q "Type 3" && echo "TYPE3: $f"; done` → no output.
   - `pdftotext` naming sweep: every figure label uses the canonical benchmark name from II.5.1 (no leftover "realistic workload" strings, no mixed naming).
   - Recompile the paper; visually check font sizes now ≈ constant across figures.
6. **Housekeeping:** `git rm` the 13 unreferenced PDFs (`fig_ablation_amis, fig_ablation_comparison, fig_gandk_corner, fig_gandk_progress, fig_hetero_throughput, fig_lotka_corner, fig_lotka_progress, fig_realistic_progress, fig_realistic_scaling, fig_realistic_scaling_quality, fig_realistic_scaling_throughput, fig_scaling_attempts, fig_scaling_throughput`) and the empty `benchmarks/ overview/ runtime/` dirs; move to an `archive/` outside `latex/` if you want to keep them.

---

## II.8 Should-fix items

### II.8.1 Statelessness / reproducibility / weight-usage wording

- **tex:85:** after "with no mutable state carried between calls" add "(the online AMIS snapshot buffer and two scheduler throttles are per-instance performance state; the reported estimator and the archive are pure functions of history, which is what crash-recovery requires)". Change "exactly reproducible" → "reproducible up to MPI arrival order (which the estimator is empirically insensitive to; \S\ref{sec:theory})".
- **tex:113:** replace "Because each snapshot $q_s$ is the proposal reconstructed from the history truncated at call $s$, the buffer $\mathcal{S}$ is itself a deterministic function of $\mathcal{H}_n$, so the propagator stores nothing between calls." with:
  > "In the retroactive estimator each snapshot $q_s$ is reconstructed from the history truncated at call $s$, so the reported posterior is a pure function of $\mathcal{H}_n$; the online buffer caches the live proposals for efficiency and is rebuilt empty after a restart, which affects only the proposal-time diagnostic weight, not the reported estimator."
- **tex:115, weight (ii):** "assigned online and used only for the effective-sample-size diagnostic" → "assigned online; it enters proposal adaptation through Eq.~\eqref{eq:steady-state-proposal} and the effective-sample-size diagnostic, but never the reported posterior."

### II.8.2 Baseline N−1 simulating ranks

Decide before the campaign; recommendation: **keep same-allocation (N ranks for both arms) and disclose.** Rationale: the packed multi-node points (48 ranks/node) cannot take an N+1st rank without spilling to another node, so launching the sync arm with N+1 ranks only at sub-node scales would make the "workers" axis mean different things at different points. Same-allocation is the defensible fairness definition ("both methods get the same cores"). Actions: in §5 (tex:174) add "The baseline's sampler dedicates one rank to coordination, so at $N$ ranks it runs $N{-}1$ simulators; both methods receive the same allocation, and at the smallest Lotka–Volterra points this favours the asynchronous method by up to $1/N$, negligible from 48 workers up." Add the same note to the Table 2 caption (tex:239). If you instead want simulating-rank parity, do it consistently at every scale (sync arm gets `srun -n $((n_workers + 1))`, spilling nodes at packed counts) and relabel the axis as "simulating workers" — more disruption than the disclosure is worth.

### II.8.3 Geometric-decay cached-path MPI bug

In `propulate/propulate/propagators/abcpmc.py` (`compute_cached`, :1861-1879): store alongside the watermark the **sort key of the last consumed individual** (`self._cached_last_key`). On entry, compute the minimum sort key among individuals not yet consumed; if it is `< self._cached_last_key`, an out-of-order arrival landed inside the consumed prefix — invalidate: `return self.compute(inds, ...)` (full replay) and rebuild the cache from scratch. Add a regression test: construct a history, consume it via `compute_cached`, then insert an individual whose `(generation, island, rank)` key sorts before the watermark, and assert `compute_cached(...) == compute(...)`.

### II.8.4 Crash-loudly violations

- `propulate_abc.py:706-710`: replace the blanket `except Exception` with `raise` by default; keep the fallback only behind an explicit config flag `allow_streaming_weight_fallback: true` (default false), and if used, mark affected records (e.g. `posterior_weight_source="streaming"`) so SBC can refuse them.
- `abcpmc.py:1339-1340` and `:1383-1385` (uniform-weight fallbacks): raise `RuntimeError("extract_posterior degenerate: <reason>")` instead of returning a flat prior; the empty-history/bootstrap-only case at :1306-1309 is legitimate and stays.
- `abc_smc_baseline.py:136-145` + `pyabc_wrapper.py:134-143`: re-raise the NaN-weight `AssertionError` unless `inference_cfg.get("allow_degenerate_stop", False)`.
- `utils/runner.py:660-674` + `straggler_runner.py:360-371`: missing pyabc → `raise ImportError("pyabc required for the synchronous baseline")` instead of warn-and-skip.
- Rerun the full test suite; some tests may rely on the lenient behavior — update them to assert the raise.

### II.8.5 Straggler 1× is not a control

**Decision: config fix** (the straggler experiment reruns in the campaign anyway). Add factor `0` to `configs/straggler.json`'s slowdown-factor list before the campaign, giving the figure a true no-straggler anchor. Text edits after the rerun: tex:200 "one worker permanently slowed by $1\times,5\times,10\times,20\times$" → "one worker given a permanent post-evaluation delay of $0.1$\,s scaled by $0\times$ (control)$,1\times,5\times,10\times,20\times$"; tex:206 "As a single worker is slowed from $1\times$ to $5\times$" → "As the injected delay grows from the control to $5\times$ the base delay" (numbers from the rerun); mention the control anchor in the Fig. 1 caption.

### II.8.6 Config drift and reproducibility of configs

- `configs/sbc.json`: the campaign runs SBC at `n_trials: 1000` directly — no 100-trial companion config needed. After the rerun, update Table 3 (tex:270-280), the App. B row (tex:420), and **rewrite the MC-error hedge at tex:267**: at 1000 trials the coverage standard error is $\sqrt{p(1-p)/1000} \approx 0.007$–$0.016$, so "comparable to the gap" no longer holds — the paragraph's argument must be re-made from the new numbers (either the async–sync coverage gap is now resolvable and real, or it shrank; say which).
- Check in the packed/fair scaling configs actually used: create `configs/scaling_packed.json` (worker_counts `[48,144,192,240,288]`, k=100) and extend `scaling_fair_baseline.json` to `worker_counts: [144,192,240,288]` with a comment that k is set to the worker count at submit time by `jobs/scaling_packed.sh` — or better, add an explicit `"k_equals_workers": true` flag read by the submit script so the rule is in-config, not tribal knowledge. Same for `scaling_realistic_fair_baseline.json` (192/384).
- `configs/scaling_realistic_fillin.json`: leave the 900 s budget (data exists) but fix the caption: tex:255 "Right: Realistic workload (costly external simulator; $1800$\,s budget..." → "($1800$\,s budget; the sub-node fill-in points at $1$–$16$ workers used $900$\,s, which leaves throughput unchanged)".
- `configs/sensitivity.json`: add `"n_workers": 48` to the inference block (verify `submit_replicate_shards.py:321` picks it up).
- `configs/parameter_bias.json`: set `"experiment_name": "parameter_bias"`; grep for consumers of the old name (`analyze_param_bias.py` reads a run dir path passed on the CLI, so only directory naming changes).
- `run_all_paper_experiments.py:87-99`: add `parameter_bias` and `scaling_realistic` entries so the orchestrator covers the whole paper.

### II.8.7 AMIS citation (theory)

1. Check `latex/sn-article-template/sn-bibliography.bib` for `cornuet2012amis`; confirm which results that paper actually states.
2. Add: J.-M. Marin, P. Pudlo, M. Sedki, "Consistency of adaptive importance sampling and recycling schemes", Bernoulli 25(3):1977–1998, 2019 (arXiv:1211.2548) — the consistency proof for modified AMIS.
3. tex:153: "as a corollary of the AMIS theory \cite{cornuet2012amis}" → "as a corollary of the consistency theory for AMIS-type recycling schemes \cite{marin2019consistency}, in the smooth-likelihood ABC framework \cite{wilkinson2013}; AMIS itself was proposed in \cite{cornuet2012amis}, whose original convergence argument is heuristic."
4. tex:169 (proof sketch): replace "The result follows from \cite[Thm.~1--2]{cornuet2012amis}" with a citation to the actual theorem used in marin2019consistency, and verify its assumptions map onto Conditions 1–4 (their scheme restricts adaptation — if the mapping is not clean, downgrade Theorems 1–2 to a Proposition with a proof-sketch caveat, and say so in Limitations (iii)).

---

## II.9 Nice-to-have items

1. **Appendix jitter formula (tex:395):** replace "a fixed jitter $\lambda=10^{-6}$" with "a scale-aware jitter $\lambda = 10^{-9}\,\mathrm{tr}(\widehat{\mathrm{Cov}})/d + 10^{-12}\,\overline{\ell^2}$ (with $\overline{\ell^2}$ the mean squared box length), escalated $10^{3}\times$ on Cholesky failure" (matches `abcpmc.py:919,926`).
2. **Truncation mass (tex:395):** "with its log normalising mass computed analytically" → "with its log normalising mass computed analytically from the diagonal marginals (exact for diagonal $\Sigma$, an approximation under correlation)".
3. **Remove dead scheduler kwargs:** delete `"low_rate"` and `"expand_factor"` from the forwarding tuple at `propulate_abc.py:555-556`; delete the $r_{\text{lo}}$ sentence at tex:391. Add a test constructing the scheduler from a config containing `high_rate`/`shrink_factor` only.
4. **`amis_snapshots` default unification:** change the wrapper default at `propulate_abc.py:541` from `0` to `20` (matching `abcpmc.py:491` and the paper). Run the full test suite; any test relying on S=0 legacy weighting should set it explicitly. Alternatively add a validation error when `kernel != "hard"` and `amis_snapshots == 0` unless explicitly configured.
5. **Wall-time filter vs posterior weights (`propulate_abc.py:700-767`):** move the deadline filter before weight extraction — build `timed_population = [ind for ind in population if ind.evaltime is None or (float(ind.evaltime) - run_start) <= max_wall_time_s]` right after the run loop, pass it to `extract_posterior` and the record loop, and delete the post-hoc record filter at :763-767. Add a test: an individual past the deadline influences neither records nor the weight normalization.
6. **SBC binning (`make_sbc_fig.py:64`):** with 100 posterior draws, ranks live on {0,…,100} (101 values). Either (a) resample to 99 draws in `analysis/sbc.py` so ranks live on {0,…,99} and 10 bins are exact, or (b) keep 101 values but set per-bin expected proportions $p_b = |\text{bin}_b|/101$ (10 or 11 values per bin) in both the expected line and the binomial band. (b) is a figure-only change.
7. **Sliced-Wasserstein determinism (`convergence.py:142-148`):** pass `seed=0` to `ot.sliced_wasserstein_distance` (POT supports a `seed` argument). Only affects multi-D benchmarks.
8. **LOCF nits (`convergence.py:700-763`):** fix the docstring ([global-min, max], not [0, max]) and, in the all-distinct branch, emit grid times rather than native wall-times for consistency with the fill-in branch.
9. **Observed-data note:** App. B (tex:403, Seeding paragraph) add: "Within each benchmark experiment the five replicates share one observed dataset (generated from the base seed); replicates vary the inference randomness only."
10. **Realistic-workload failure rate:** extract the NaN fraction from the run logs (`benchmarks/realistic_workload.py` logs each NaN return); if ≳1%, state it in §7.3 and note NaN attempts count as simulations for both methods symmetrically.
11. **Error-convention standardization:** with the II.7 regeneration, converge on median+IQR for all 5-replicate figures (scaling already is; straggler/hetero become so; param-bias and realistic-util move from mean±sd — keep mean±CI only where a caption argues for it) and align every caption. Update tex:206 straggler numbers to the regenerated medians.
12. **Memory hygiene note:** `_pyabc_history.py`, `pyabc_sampler.py:345-346` look-ahead resolution, and `Deadline` semantics are fine as-is — no action.

---

## II.10 Final verification checklist

Run after all fixes, before the next submission-ready commit:

1. `sim_backend_venv/.venv/bin/python -m pytest experiments/tests -q` — all green (including the new tests from II.1.b/II.6/II.8.3/II.8.4/II.9.5).
2. Naming-consistency sweep (no scrub — CPM/NAStJA are named in the final paper): `grep -rin "realistic.workload" latex experiments/scripts --include="*.py" --include="*.tex"` → no paper-facing hits of the scrubbed framing; `pdftotext` sweep over `latex/.../figures/*.pdf` → every figure uses the canonical "Cellular Potts" naming from II.5.1.
3. Font sweep: `pdffonts` over all referenced figures → no `Type 3`.
4. Figure regeneration from a clean checkout: every `\includegraphics` target regenerable via an in-repo `make_*_fig.py` reading in-repo `experiments/data/paper_figures/` CSVs (satisfies the Declarations, tex:372/378).
5. Paper claim re-check against final code — the five rewritten claim sites (matched schedule, exact estimator, statelessness, weight usage, W-metric labels) must match what the code now does; grep the tex for "matched", "exactly", "stateless", "analytic" and re-read each hit.
6. Numbers re-check — **every quoted number in the paper is stale** (new propagator + matched-ε baseline + fixed acceptor RNG): re-derive the abstract's throughput/collapse/coverage figures; §6's straggler (tex:206), heterogeneity idle fractions (tex:219), param-bias band (tex:228), LV scaling series (tex:236 + Table 2); SBC (Table 3, tex:267 — now at 1000 trials); recovery numbers (tex:289); realistic scaling/utilization series (tex:312, 317); ablation values (tex:329); sensitivity ranges (tex:338); and App. B if any budget changed (fill-in now 1800 s). Walk table-by-table and figure-by-figure against the new summaries; do not grep-and-patch digits.
7. Recompile: `cd latex/sn-article-template && pdflatex sn-article && bibtex sn-article && pdflatex sn-article && pdflatex sn-article` — no undefined references, no overfull figure boxes.
