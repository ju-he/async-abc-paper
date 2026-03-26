# async-abc-paper

Paper experiments for asynchronous ABC with Propulate.

## Requirements

- Python 3.10+
- Core Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `POT`
- `propulate` for `async_propulate_abc`
- `pyabc` for `pyabc_smc` and `abc_smc_baseline`
- `mpi4py` for MPI launches and pyABC runs with `n_workers > 1`
- `nastjapy` plus a working `nastja` build for the `cellular_potts` benchmark

The `cellular_potts` benchmark first tries the active Python environment and
then falls back to a repo-local `nastjapy_copy/.venv` if it exists.

Cluster helper scripts under `experiments/jobs/` assume SLURM plus an MPI
environment.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[pyabc,propulate,mpi,test]'
```

For a minimal install without optional methods:

```bash
pip install -e .
```

If you want to run `cellular_potts`, also install `nastjapy`/`nastja` in the
active environment or provide the repo-local fallback environment.

## Experiment Configs

Main configs live in `experiments/configs/`:

- `gaussian_mean.json`, `gandk.json`, `lotka_volterra.json`, `cellular_potts.json`:
  benchmark comparisons across inference methods
- `sbc.json`: simulation-based calibration
- `straggler.json`: persistent straggler sweep
- `runtime_heterogeneity.json`: log-normal runtime heterogeneity sweep
- `scaling.json`: scaling over different worker counts
- `sensitivity.json`: hyperparameter grid sweep
- `ablation.json`: ablation variants

Reduced-cost configs live in `experiments/configs/small/` and are selected with
`--small`.

## Common Options

All runner scripts share the same basic flags:

- `--config`: path to the JSON config
- `--output-dir`: root directory for results
- `--test`: cheap smoke-test mode; clamps budgets, reduces replicates/trials,
  and caps workers to 8 locally or 48 on SLURM
- `--small`: load the matching config from `experiments/configs/small/`; can be
  combined with `--test`
- `--extend`: skip combinations already present in the existing output CSVs

Advanced sharding flags are also available on runner scripts:

- `--shard-index`
- `--num-shards`
- `--shard-run-id`
- `--finalize-only`

In practice those are usually driven through `submit_replicate_shards.py`
instead of being passed by hand.

You can override the test-mode worker cap with:

```bash
ASYNC_ABC_TEST_MAX_WORKERS=4 python ...
```

## Running Experiments

Run one benchmark locally in test mode:

```bash
python experiments/scripts/gaussian_mean_runner.py \
  --config experiments/configs/gaussian_mean.json \
  --output-dir results/local \
  --test
```

Run all paper experiments:

```bash
python experiments/run_all_paper_experiments.py \
  --output-dir results/test_all \
  --test
```

Run only a subset:

```bash
python experiments/run_all_paper_experiments.py \
  --output-dir results/subset \
  --test \
  --experiments gaussian_mean gandk sbc
```

Run the reduced small tier:

```bash
python experiments/run_all_paper_experiments.py \
  --output-dir results/small \
  --small
```

Resume an interrupted run:

```bash
python experiments/run_all_paper_experiments.py \
  --output-dir results/test_all \
  --test \
  --extend
```

## MPI And SLURM

Most full configs are set up for multi-worker execution. For a direct MPI run,
launch the runner under `mpirun` or `srun` and match the worker count to the
config:

```bash
srun -n 48 python experiments/scripts/gaussian_mean_runner.py \
  --config experiments/configs/gaussian_mean.json \
  --output-dir /path/to/results
```

Production batch job for all non-scaling experiments:

```bash
sbatch experiments/jobs/run_experiments.sh /path/to/results
```

Cluster smoke test for the full paper pipeline:

```bash
sbatch experiments/jobs/test_all.sh /path/to/results
```

## Scaling And Sharding

The scaling experiment is handled separately from the main production batch job.

Single scaling run for a specific worker count:

```bash
srun -n 48 python experiments/scripts/scaling_runner.py \
  --config experiments/configs/scaling.json \
  --output-dir /path/to/results \
  --n-workers 48
```

Preview SLURM scaling submissions:

```bash
python experiments/jobs/submit_scaling.py /path/to/results \
  --timing-csv /path/to/test_run/timing_summary_test.csv \
  --dry-run
```

Submit the scaling sweep:

```bash
python experiments/jobs/submit_scaling.py /path/to/results \
  --timing-csv /path/to/test_run/timing_summary_test.csv
```

Preview the small-tier scaling sweep:

```bash
python experiments/jobs/submit_scaling.py /path/to/results \
  --small \
  --timing-csv /path/to/test_run/timing_summary_small.csv \
  --dry-run
```

Preview the small test-mode scaling sweep:

```bash
python experiments/jobs/submit_scaling.py /path/to/results \
  --small --test \
  --timing-csv /path/to/test_run/timing_summary_small_test.csv \
  --dry-run
```

Preview replicate-based sharded submissions:

```bash
python experiments/jobs/submit_replicate_shards.py /path/to/results \
  --experiments gaussian_mean gandk sbc \
  --jobs-per-experiment 4 \
  --dry-run
```

Add more full replicates to an existing sharded output:

```bash
python experiments/jobs/submit_replicate_shards.py /path/to/results \
  --experiments gaussian_mean gandk \
  --add-replicates
```

## Useful Commands

Re-generate plots from existing outputs without re-running inference:

```bash
python experiments/scripts/replot.py results/test_all all
```

Run the unit and integration-style tests:

```bash
pytest
```

Run the local sharded smoke test:

```bash
bash experiments/jobs/test_sharded.sh
```

Run the SLURM sharded smoke test:

```bash
bash experiments/jobs/test_sharded_slurm.sh /path/to/output
```

Generate a new Cellular Potts reference dataset:

```bash
python experiments/scripts/generate_cpm_reference.py \
  --config-template experiments/assets/cellular_potts/sim_config.json \
  --config-builder-params experiments/assets/cellular_potts/config_builder_params.json \
  --parameter-space experiments/assets/cellular_potts/parameter_space_division_motility.json \
  --true-params '{"division_rate": 0.049905, "motility": 0.2}' \
  --output-dir experiments/data/cpm_reference_generated \
  --seed 0
```

## Output Layout

Each run writes results under the output root you pass via `--output-dir`:

```text
<output-root>/
  timing_summary_<run_mode>.csv
  timing_comparison.csv
  <experiment-name>/
    data/
    plots/
    logs/
  _jobs/      # created by submit_replicate_shards.py
  _shards/    # created by sharded execution
```

Common output files include:

- `data/raw_results.csv` for standard benchmark runs
- `data/timing.csv` for per-experiment wall-clock summaries
- `data/metadata.json` for config, package, platform, and git provenance
- `data/throughput_summary.csv` for scaling runs
- `data/sbc_ranks.csv` and `data/coverage.csv` for SBC runs
