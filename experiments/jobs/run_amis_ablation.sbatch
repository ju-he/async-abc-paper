#!/bin/bash -x
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_amis_ablation
#SBATCH --output=/tmp/abc_amis_ablation-%j.out
#
# AMIS ablation on the *runtime-heterogeneity* benchmark.
#
# Why this run exists: the ablation reported in the paper (Fig. 13) is on the
# uniform-runtime Gaussian target, where removing AMIS changes nothing because
# there is no arrival-order skew for the reweighting to correct. That leaves no
# experiment showing the AMIS estimator -- the one the theory of §4 is stated
# for -- buying anything. This run puts it in the regime where it should: every
# worker draws a fresh lognormal runtime multiplier per evaluation, so the
# order in which particles arrive is genuinely skewed.
#
# Two arms, matched on everything but `amis_snapshots` (20 vs 0), same seeds,
# same sigma grid {0, 1.0, 1.5, 2.0}, five replicates, 60 s wall per replicate.
# sigma=0 is the control that reproduces the paper's null result.
#
# Cost: 2 arms x 4 sigmas x 5 replicates x 60 s = 40 min of run time on one
# node; 2 h is requested for setup, finalize and plot generation.
#
# Usage (from the login node, repo root):
#   experiments/jobs/submit.sh experiments/jobs/run_amis_ablation.sbatch \
#       /p/scratch/tissuetwin/herold2/async-abc/amis_ablation_$(date +%Y%m%d)
set -euo pipefail

# Paths are injected by experiments/jobs/submit.sh via `sbatch --export`.
nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"
output_dir="${1:?Usage: $(basename "$0") <output_dir>}"

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Multi-rank non-scaling job: avoid the intermittent pscom MPI-teardown hang.
export PROPULATE_SKIP_DISCONNECT=1

# The arms run sequentially so both see an otherwise idle node -- throughput is
# not the measurement here, but a co-scheduled arm would perturb the arrival
# process that IS.
for arm in hetero_ablation_amis hetero_ablation_no_amis; do
    srun python "$experiments_dir/scripts/runtime_heterogeneity_runner.py" \
        --config "$experiments_dir/configs/${arm}.json" \
        --output-dir "$output_dir"
done

echo "AMIS ablation complete: $output_dir"
