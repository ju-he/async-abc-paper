#!/bin/bash -x
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=WALLTIME
#SBATCH --partition=batch
#SBATCH --job-name=abc_production
#SBATCH --output=/tmp/abc_production-%j.out
# Override SLURM log path at submission time: sbatch --output=<dir>/abc_production-%j.out ...

# Paths are injected by experiments/jobs/submit.sh via `sbatch --export`.
sim_backend_path="${SIM_BACKEND_PATH:?SIM_BACKEND_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"
output_dir="${1:?Usage: $(basename "$0") <output_dir> [--extend]}"
extend_flag="${2:-}"


module restore sim_backend
module load ParaStationMPI
source "$sim_backend_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Run all experiments except scaling (scaling is submitted separately via submit_scaling.py)
srun python "$experiments_dir/run_all_paper_experiments.py" \
    --experiments gaussian_mean gandk lotka_volterra realistic_workload sbc \
                  straggler runtime_heterogeneity sensitivity ablation \
    --output-dir "$output_dir" \
    ${extend_flag:+"$extend_flag"}
