#!/usr/bin/env bash
# Thin sbatch wrapper that auto-selects --account/--partition based on $SYSTEMNAME.
# Forwards all positional arguments to sbatch (target script + its args).
#
# Usage:
#   experiments/jobs/submit.sh experiments/jobs/run_experiments.sh /p/scratch/.../out
#   experiments/jobs/submit.sh experiments/jobs/test_all.sh
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$script_dir/site_env.sh"
exec sbatch --account="$SITE_ACCOUNT" --partition="$SITE_PARTITION" "$@"
