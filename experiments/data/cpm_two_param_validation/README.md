# Two-parameter CPM validation artefacts (2026-09-19)

Kept because scratch is inode-constrained and these are the measurement record behind
`.plans/cpm_two_param_validation_2026-09-19.md`.

* `prior_corpus_posterior.json` — the rejection-ABC posterior read off 2000 prior draws scored
  through the shipped benchmark (`diag_cpm_prior_corpus.py`, job 14261882). This is the
  forecast check that involves no sampler. The full corpus is archived at
  `/p/scratch/tissuetwin/herold2/async-abc/cpm_prior_corpus.tar.gz`.
* `async_validation_rep0_raw_results.csv` — the one async replicate of job 14261878,
  13,115 evaluations under `cellular_potts_two_param_validate.json`. Re-read with::

      python experiments/scripts/diag_cpm_posterior_contraction.py \
          --results experiments/data/cpm_two_param_validation/async_validation_rep0_raw_results.csv \
          --config experiments/configs/cellular_potts_two_param_validate.json \
          --prefix 500 2000 5000 10000

The production run (3 methods x 5 replicates, job 14261956) writes to
`/p/scratch/tissuetwin/herold2/async-abc/cpm_two_param_production`.

## Added 2026-09-21

* `cpm_two_param_fixed/`, `cpm_two_param_production/`, `cpm_80_comparison/` — the gzipped
  `raw_results.csv` of jobs 14262214 (`tol_init` 0.1), 14261956 (`tol_init` 10.0) and 14262841
  (80³), **repaired** by `experiments/scripts/repair_two_param_cpm_records.py`: the synchronous
  rows had their two parameter columns swapped by a writer bug (see
  `.plans/bug-fixes/previous-fixes.md`, 2026-09-21). Verified against pyABC's histories.
* `pyabc_populations.csv.gz` — every synchronous population of those three runs with pyABC's
  importance weights, distances and epsilons, extracted from the SQLite histories (the CSV records
  had lost the weights). The histories themselves stay in the tarballs on scratch.
* `cpm_fair_rejection.tar.gz` (13,000 prior draws on 48 ranks, job 14262032), `cpm_80_prior.tar.gz`
  (1,000 draws at 80³, job 14262842), `cpm_80_comparison.tar.gz` (job 14262841, with histories and
  traces) — the corpora behind the fair rejection baseline and the 80³ systems comparison.
