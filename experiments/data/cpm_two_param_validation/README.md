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
