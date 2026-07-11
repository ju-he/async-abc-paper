# Continue the async-abc rerun-campaign (handoff)

Full state is in auto-memory — read `project_full_review_rerun_campaign.md` (bottom
entries), `reference_asyncabc_cluster_deploy.md`, and `.plans/bug-fixes/previous-fixes.md`
(2026-07-08 OOM entry) first.

## Setup
- Repo `/home/juhe/bwSyncShare/Code/async-abc-paper`, `main@0aa1237` frozen (the campaign
  runs from the *deployed* cluster code, not git). propulate frozen `e148f4f`.
- Cluster: `ssh juwels` — if SSH is dead, ask the user to re-auth (2FA). Scratch mount
  `/home/juhe/remotes/scratch/herold2/async-abc` may be down; use `ssh` for cluster reads.
- Campaign output dir: `/p/scratch/tissuetwin/herold2/async-abc/rerun_20260707`.

## Core issue (settled)
The faster frozen propagator makes wall-time-limited fast-sim runs generate ~1.3M evals/rank
→ OOM at packed 48/node on 94 GB batch nodes. No engine fix (it breaks per-rank checkpointing).
Non-scaling workaround: **mem192 24/node × 2 nodes `--exclusive`** = 7.5 GB/rank
(`--exclusive` MANDATORY, else half-node memory). SBC/sensitivity kept wall-limited (paper's
"fixed-walltime" framing; sim-limited is the cheaper alternative if the user reconsiders).

## Done (7/9 non-scaling, assembled in rerun_20260707)
gaussian_mean, gandk, lotka_volterra, ablation, runtime_heterogeneity, cellular_potts,
SBC-1000 (async well-calibrated, baseline over-confident — see `sbc/data/coverage.csv`).

## In flight — CHECK FIRST
- **sensitivity**: jobs 14100263–267 (de-duped to 64 combos, mem192 24/node `--exclusive`,
  24 h, ~17 h/shard). Verify completion + `sensitivity/data/`. Watch for timeout (17h est).

## Open blockers / next steps
1. **SCALING OOM — DIAGNOSIS CORRECTED + FIX IN FLIGHT (2026-07-11).** NOT cross-combo
   accumulation: scaling_single.sh already runs one srun per combo; sacct shows EVERY combo of
   14100320/21 OOM'd individually ~100–165 s into its 900 s run (full history on every rank,
   ~4.5k sims/s island-wide ~flat in N, ~3.7 KB/individual for LV). Full write-up in
   .plans/bug-fixes/previous-fixes.md (2026-07-11 entry). **User chose the wall-time rescale:**
   ÷5 axis, budgets [12,24,60,120,180], wall 180 s (≈ eval count of the old-propagator 300 s
   regime) in scaling.json / scaling_fair_baseline.json / scaling_timing.json / small tier;
   run mem192 packed 48/node --exclusive (submit_scaling.py grew --exclusive);
   PROPULATE_DISABLE_CHECKPOINT=1 in both scaling wrappers. Deployed. Memtest 14100336
   (worst case N=48 k=100) gating the full submit. Never reuse scaling_lv_20260711 (old
   checkpoints load even with dumping disabled). Then submit: LV `scaling.json` +
   `scaling_fair_baseline.json` + `scaling_timing.json` (mem192 --exclusive); CPM
   `scaling_cpm.json` + `scaling_cpm_fillin.json` + `scaling_cpm_fair_baseline.json`
   (batch, unchanged — slow sims fit packed). Paper text must adopt the new budget axis
   (quote 180 s, not 900 s) in the §II.0 step-4/5 pass.
2. **straggler MERGE**: runs done (data at `rerun_20260707/_shards/straggler/runs/run_20260708_074437`),
   but the finalizer is O(n²)-slow (timed out 6 h even on mem192 via `experiments/jobs/finalize_shards.py`).
   Investigate `experiments/async_abc/utils/shard_finalizers.py` straggler path — fix the
   slowness, don't just add nodes.
3. **Commit** the uncommitted tooling to the `campaign-tooling` branch: the `--exclusive` flag
   in `submit_replicate_shards.py` and the `sensitivity.json` scheduler_type de-dup.
4. After all experiments land: figures + paper-text edits + re-derive every number
   (plan `.plans/paper_review_2026-07-05.md` §II.0 steps 4–5).

## Recipes
- Non-scaling fast-sim → `submit_replicate_shards.py <out> --experiments X --partition mem192
  --ntasks-per-node 24 --exclusive --jobs-per-experiment N --time HH:MM:SS` (1 unit/shard).
- Merge-OOM → `experiments/jobs/finalize_shards.py` on a 1-rank `--exclusive` node.
- Rerun matrix: `.plans/paper_review_2026-07-05.md` §II.0.a.
