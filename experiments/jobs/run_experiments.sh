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

nastjapy_path=/p/project1/tissuetwin/herold2/nastjapy
output_dir="${1:?Usage: $(basename "$0") <output_dir> [--extend]}"
extend_flag="${2:-}"
experiments_dir=/p/project1/tissuetwin/herold2/async-abc-paper/experiments


module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Run all experiments except scaling (scaling is submitted separately via submit_scaling.py)
srun python "$experiments_dir/run_all_paper_experiments.py" \
    --experiments gaussian_mean gandk lotka_volterra cellular_potts sbc \
                  straggler runtime_heterogeneity sensitivity ablation \
    --output-dir "$output_dir" \
    ${extend_flag:+"$extend_flag"}
