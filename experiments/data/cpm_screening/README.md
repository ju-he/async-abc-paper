# CPM screening corpora

Produced by `experiments/scripts/diag_cpm_screening.py` on 2026-09-18; findings written up in
`.plans/cpm_screening_results_2026-09-18.md`.

Committed here: the `protocol.json` describing each run (protocol settings and the exact candidate
priors the corpus was generated under), the `screening_report*.json` holding every number the write-up
cites, and the rendered tables under `logs/`.

**Not committed: the raw corpora** (48 JSONL shards per run, ~110 MB each). They live on JUWELS at
`/p/scratch/tissuetwin/herold2/async-abc/cpm_screen_{50,80,r2,shipped}.tar.gz` (24 MB each). To
re-analyse, unpack one next to its `protocol.json` so that `<dir>/corpus/rank_*.jsonl` exists, then

    python experiments/scripts/diag_cpm_screening.py --mode analyze --out <dir>

| directory | corpus | design |
|---|---|---|
| `blocksize50` | `cpm_screen_50.tar.gz` | six parameters, reparameterised priors, 50³ (`--design-seed 20260918`) |
| `blocksize80` | `cpm_screen_80.tar.gz` | the same design at 80³ |
| `round2_50` | `cpm_screen_r2.tar.gz` | seven parameters, refined priors, 50³ (`--design-seed 20260919`) |
| `shipped_prior` | `cpm_screen_shipped.tar.gz` | the paper's own two parameters and priors (`--design-seed 20260920`) |

`shipped_prior` carries a second report, `screening_report_shipped_model.json`, scored with the
paper's own feature-space model and reference simulation (`--shipped-model`) rather than a metric
refitted on the design. That is the run directly comparable to the stored campaign.
