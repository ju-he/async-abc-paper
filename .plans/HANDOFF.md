# HANDOFF — async-ABC paper revision (resume after /clear)

**Last updated:** 2026-06-28 · **Branch:** `refactor/general` · **Paper:** `latex/sn-article-template/sn-article.tex` (20 pp, compiles clean)

Read this first, then `.plans/sensitivity_backfill_ready.md` (older context) and
`memory/reference_asyncabc_cluster_deploy.md` + `memory/project_scaling_mpi_fixes.md` (cluster how-to).

---

## TL;DR — what's the state
The paper is a results-complete draft that has been through **two independent codex reviews**
(`/code-review` via the `codex` CLI) and a large revision pass. All revision work is **committed** on
`refactor/general`. One cluster job is **in flight** (CPM scaling fill-in); when it lands there's a figure
to build + wire. A new multi-node submission script was just added.

## ⏳ IN-FLIGHT — do this when it finishes
**Job `14067753`** (jsc-mpc MCP, cluster `juwels-cluster`, project `async-abc-paper`): single-node CPM
scaling fill-in, workers **1/4/16**, k=100, 900 s budget, 3 reps. Output:
`/p/scratch/tissuetwin/herold2/async-abc/run_cpm_fillin_20260628/scaling_cpm/`.
- Check: `mcp__jsc-mpc__job_status(cluster=juwels-cluster, job_id=14067753, project=async-abc-paper)`.
- When **COMPLETED**: build a **5-point CPM scaling figure** (1,4,16 from the fill-in + 48,96 from the
  existing run `run_cpm_20260626_1906/scaling_cpm/`) and **wire it into §7.3**, upgrading CPM from the
  current feasibility framing to a real throughput-scaling result (async throughput scales ~linearly; sync
  plateaus at k=100) **with the honest quality caveat** (async Wasserstein stays slightly *behind* sync on
  CPM — ~0.47 vs ~0.40, near the rejection floor). Don't overclaim CPM quality.
  - Throughput data: `<run>/scaling_cpm/data/throughput_summary_w{N}_k100.csv` (`throughput_sims_per_s`).
  - Quality data: `<run>/scaling_cpm/data/budget_summary.csv` (`quality_wasserstein_by_budget`, filter
    `budget_s==900`).
  - A 2-point draft figure already exists at `figures/fig_cpm_scaling.pdf` (untracked) — regenerate it as a
    5-point version, then `\includegraphics` it in the §7.3 figure block (currently shows
    `fig_cpm_corner.pdf`, label `fig:cpm-posterior`). Recompile + commit.

## ▶ NEW TOOL (committed `12f48ad`)
`experiments/jobs/submit_cpm_node_scaling.py` — multi-node, power-of-2 CPM scaling submitter, thin wrapper
over `submit_scaling_cpm.py` (reuses its sbatch rendering, `ceil(N/48)` node math, bin-packing, per-combo
MPI-isolated wrappers, site detection). Adds power-of-2 generation + a total **CPU-hour estimate**.
- `--max-workers N` (powers of 2: 1,2,4,…,N) or `--max-nodes M` (full nodes: 48·[1,2,4,…,M]).
- `--dry-run` prints the node-h / CPU-h estimate (works locally). Real submit / sbatch preview must run on
  the **JUWELS login node** (needs `$SYSTEMNAME` site detection) or with `NASTJAPY_PATH` set.
- Example: a 256-worker sweep ≈ **32 node-h / 1550 CPU-h**; a 128-worker sweep @900s/3reps ≈ 17 node-h / 834 CPU-h.
- NOTE: not yet deployed to cluster. If used, `rsync` the repo (or just the file) to
  `/p/project1/tissuetwin/herold2/async-abc-paper` first (see deploy below).

---

## Session commits on `refactor/general` (newest first)
- `12f48ad` jobs: power-of-2 CPM node-scaling submitter (+CPU-hour estimate)
- `7974584` paper: theory caveat + Fig 2/5 legibility (review 2)
- `3193df4` paper: inferential-efficiency figure under heterogeneity (review 2 MAJOR)
- `5833c2b` paper: review 2 — estimator ref, filled appendices, softened claims
- `870b728` paper: CPM → feasibility framing
- `88aad33` paper: sensitivity Fig 9 → posterior quality (not internal tolerance)
- `1bcb406` paper: review 1 — reorder results, reframe claims, Fig 3 replot, Table 4
- `dfba903` paper: fill gaussian recovery + sensitivity panels (results-complete draft)
- `245dc6c` docs: validated gaussian OOM fix + cluster MCP/deploy notes

## What both reviews changed (all done)
Reorder results to lead with HPC advantage (straggler = Fig 1) · reframe "sole methodological difference"
overclaim · fix τ_i notation bug · soften "statistically non-inferior" → "comparable" · fix the
**estimator cross-ref** (Eq 5 proposal weight π/q̄ vs §4 posterior estimator π·K_ε/q̄) · statelessness
clarification · **fill the two empty appendices** from code (schedulers, covariance, matched acceptor,
chunking; seeding, JUWELS setup, config table) · **theory-vs-implementation caveat** (Conditions 1/4 are
approximations of the idealized cumulative-mixture variant) · replot Fig 3 gaussian (async/sync visible +
inset) · Fig 9 sensitivity → posterior Wasserstein · **Fig 2 posterior-recovery** (was plotting *tolerance*
while caption said *Wasserstein* — now actually plots Wasserstein; async beats sync on g-and-k/Lotka) ·
Fig 5 hetero → clean single idle-fraction panel · **new inferential-efficiency Fig** (`fig:hetero-quality`:
under heterogeneity, sync posterior error blows up ~10–30× while async stays flat — the strongest result
after the straggler).

## Remaining review items (NOT yet done)
- **CPM multi-worker evidence** — in progress (job 14067753 above); bigger sweeps via the new node script.
- **Define uncertainty bands** in the *pipeline* figure captions (straggler/scaling/SBC) — replotted figs
  already define them (IQR / CI). Confirm band type in the pipeline before asserting.
- Minor editorial: Table 3 vs Fig 3 redundancy; abstract tightening; remaining composite-figure legibility.
- Author-only: funding / acknowledgements / author-contributions (currently neutral placeholders, not red TODOs).

## Caption/content bugs found & fixed (watch for more)
1. §5 cited the wrong equation for the reported posterior. 2. §7.1 claimed Fig 1 showed "comparable posterior
quality" (it's throughput-only) → cross-ref added. 3. Fig 2 panels plotted tolerance, caption said Wasserstein.

---

## Infra cheat-sheet
- **jsc-mpc MCP** — always pass `cluster="juwels-cluster"`, `project="async-abc-paper"`. Budget: soft 24 /
  hard 96 node-h, rolling 24 h; ~80 nh remaining but ~12 nh is **stale/phantom reserved** (no running job),
  which eats soft-limit headroom — keep new reservations small or they'll prompt the user (you can't
  self-answer the soft-limit elicitation).
- **Mount** — `/home/juhe/remotes/scratch/herold2/async-abc` ⇄ cluster
  `/p/scratch/tissuetwin/herold2/async-abc` (read run outputs locally here).
- **Cluster code** — `/p/project1/tissuetwin/herold2/async-abc-paper`. **Deploy = plain rsync** (no alias):
  `rsync -avP --exclude=".*" --exclude="*.png" --exclude="*.pdf" --exclude="*__pycache__" ./ herold2@juwels.fz-juelich.de:/p/project1/tissuetwin/herold2/async-abc-paper`
- **venv** — local tests: `nastjapy_copy/.venv/bin/python`; cluster: `/p/project1/tissuetwin/herold2/nastjapy/.venv`.
- **Job recipe** (submit_job `command`): `module purge; module load Stages/2025 GCC Python; module restore
  nastjapy && module load ParaStationMPI && source <cluster-venv>/bin/activate && export
  PROPULATE_SKIP_DISCONNECT=1 && srun --ntasks=N ... python <runner> ...` (set SKIP_DISCONNECT for any
  multi-rank job to dodge the pscom teardown hang).
- **Compile paper**: `cd latex/sn-article-template && latexmk -pdf -interaction=nonstopmode -halt-on-error
  sn-article.tex` then one more `pdflatex` pass; verify `0 undefined refs`. Build artifacts (.aux/.bbl/...)
  are gitignored; `sn-article.pdf` is tracked (32 MB — heavy). `sn-article-compressed.pdf` is a stale,
  untracked shareable variant.
- **codex review**: `codex exec -s read-only "<prompt>" -i <page-images...>` from `latex/sn-article-template`;
  render pages with `pdftoppm -png -r 110 sn-article.pdf /tmp/pp/pg`. Two reviews so far (gpt-5.4); a 3rd
  would confirm the revisions hold.

## Data locations (on the mount)
- Benchmarks: `run_full_20260626_1816/{gaussian_mean,gandk,lotka_volterra,cellular_potts,sbc,
  runtime_heterogeneity,sensitivity,ablation}/` (+ `_shards/` for sharded).
- Gaussian re-run (validated OOM fix): `gaussian_rerun_20260628/gaussian_mean/`.
- CPM scaling 48/96: `run_cpm_20260626_1906/scaling_cpm/`.
- CPM scaling fill-in (in progress): `run_cpm_fillin_20260628/scaling_cpm/`.
- Straggler data is NOT in run_full (job was cancelled); Fig 1 uses the committed `fig_straggler_throughput.pdf`.

## Figures regenerated this session (in `latex/.../figures/`, tracked)
`fig_gaussian_recovery` · `fig_sensitivity_heatmap` · `fig_cpm_corner` (now §7.3, label `fig:cpm-posterior`) ·
`fig_posterior_recovery` (Wasserstein, replaced the 3 tolerance panels) · `fig_hetero_idle` (single panel) ·
`fig_hetero_quality` (new). Orphaned-but-tracked (unused now, left in place): `fig_cpm_scaling_throughput/quality`,
`fig_hetero_throughput`, `fig_{gandk,lotka,cpm}_progress`. Untracked draft: `fig_cpm_scaling.pdf` (2-point CPM).
