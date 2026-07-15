# async-abc rerun-campaign — status

**Campaign COMPLETE (2026-07-15).** Compute matrix done + all figures regenerated +
every number re-derived. See `.plans/campaign_final_report_2026-07-15.md` for the full
report and the **author-review checklist of claim-level changes**. Auto-memory
`project_full_review_rerun_campaign.md` (bottom entry) has the condensed state.

## Done
- Compute: 11 non-scaling groups + LV scaling + CPM scaling in
  `/p/scratch/tissuetwin/herold2/async-abc/rerun_20260707`, plus `parameter_bias`
  (was missing; ran 2026-07-15, jobs 14107933-936).
- Figures: all 15 regenerated from campaign data via `paper_style` (0/15 Type-3,
  vendored CSVs under `experiments/data/paper_figures/`). Commit `623456c`.
- Paper text: every quoted number + caption re-derived. Commit `e5bdf52`. Compiles clean.

## Figure regeneration mechanics (if a figure needs re-doing)
- Each `experiments/scripts/make_*_fig.py` uses `_figdata.py`: default reads the committed
  vendored CSV; `--refresh [<campaign_root>]` re-derives from the campaign data
  (default root = local staging mirror `/home/juhe/async-abc-rerun-staging`).
- Venv: `nastjapy_copy/.venv/bin/python`. Cluster reads: `ssh juwels` (scratch mount stale).
- Sensitivity quality summary was computed on-cluster (`_jobs/sensitivity_agg/agg.py`);
  lv_timing is a 2-way split (frozen commit lacks the phase-timing instrumentation).

## Remaining (optional polish — author's call)
1. Review the 8 claim-level changes in the final report (esp. g-and-k comparability, LV
   scaling crossover, ablation AMIS-neutrality) — the figures/text already reflect them.
2. II.1.b: optional granularity-disclosure sentence in §5 (matched-ε claim is true in code).
3. II.7.b.6 housekeeping: `git rm` the 13 unreferenced `latex/.../figures/*.pdf` orphans.
4. Nothing is compute-blocked; branch `campaign-tooling` not pushed (per instruction).
