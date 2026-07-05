#!/usr/bin/env bash
# Thin sbatch wrapper that auto-selects --account/--partition based on $SYSTEMNAME
# and injects the per-site filesystem paths into the job environment.
# Forwards all positional arguments to sbatch (target script + its args).
#
# A submitted #SBATCH script runs from SLURM's spool copy and cannot locate the
# repo, so this login-node wrapper resolves EXPERIMENTS_DIR / SIM_BACKEND_PATH /
# SITE_SCRATCH_ROOT here and exports them via `sbatch --export`; the batch
# scripts read them from the environment (failing loudly if unset).
#
# Usage:
#   experiments/jobs/submit.sh experiments/jobs/run_experiments.sh /p/scratch/.../out
#   experiments/jobs/submit.sh experiments/jobs/test_all.sh
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiments_dir="$(cd "$script_dir/.." && pwd)"
# shellcheck source=/dev/null
source "$script_dir/site_env.sh"
exec sbatch \
    --account="$SITE_ACCOUNT" \
    --partition="$SITE_PARTITION" \
    --export="ALL,EXPERIMENTS_DIR=$experiments_dir,SIM_BACKEND_PATH=$SITE_SIM_BACKEND_PATH,SITE_SCRATCH_ROOT=$SITE_SCRATCH_ROOT" \
    "$@"
