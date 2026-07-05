#!/bin/bash -x
# Diagnostic harness: run ONE scaling combo (one MPI world, one teardown) and
# capture WHERE the k>=192 teardown wedges.
#
# Background
# ----------
# After the extract_posterior O(n*S*k) fix (compute_posterior_weights=false in
# scaling.json), the k<=48 scaling combos finalize at full record volume (~1.5M),
# but k>=192 combos still die: each runs its full wall budget, then at ~wall+5s
# the srun step is Force Terminated with 2 arbitrary ranks SIGKILLed and the rest
# SIGTERMed. No traceback, no OOM, no configured per-step timeout. The most likely
# mechanism: at the wall-time break the fast ranks finish teardown and exit, and
# srun's default --wait grace (~5s after the FIRST task exits) then kills the
# still-wedged stragglers — so we never see where they are stuck.
#
# This harness removes the two things that hide the wedge:
#   1. `srun --wait=0 --kill-on-bad-exit=0` — never kill stragglers, so the wedge
#      PERSISTS (until the job --time) instead of being torn down at wall+5s.
#   2. A stack watchdog that, once some ranks have exited (teardown started) but
#      others remain, dumps the SURVIVORS' native stacks (py-spy --native, else
#      gdb) to shared scratch.
#
# Two outcomes, each pointing at the fix:
#   * The stragglers EVENTUALLY exit (live -> 0) and the combo writes its shard
#     => teardown is merely slow on a couple ranks; the real bug is srun killing
#     stragglers. Fix = add `--wait=0` to scaling_single.sh. (The harness log
#     shows "live=N/NT" dropping to 0 and a raw_results shard appearing.)
#   * The stragglers NEVER exit (live stays >0 until job --time) => a true
#     teardown deadlock; the captured stack shows the exact wedged frame
#     (MPI collective vs numpy vs propulate drain) to fix.
#
# Submit (account/partition/paths auto-injected):
#   experiments/jobs/submit.sh experiments/jobs/scaling_single_combo.sh [output_dir]
#
# Knobs (export before submitting; ALL is forwarded by submit.sh):
#   COMBO_K            archive size k for the combo            (default 1000)
#   COMBO_REP         replicate index                          (default 0)
#   STACK_WATCH_START seconds before first liveness check      (default 290)
#   STACK_WATCH_INTERVAL  seconds between checks               (default 6)
#   STACK_WATCH_COUNT     number of checks                     (default 60)
#   SCALING_ENV_SETUP    alt MPI/venv setup script (optional)
#
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=00:25:00
#SBATCH --partition=batch
#SBATCH --job-name=scaling_combo_dbg
#SBATCH --output=/tmp/scaling_combo_dbg-%j.out

set -u

# Paths injected by experiments/jobs/submit.sh via `sbatch --export`.
nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"
config_path="$experiments_dir/configs/scaling.json"
runner="$experiments_dir/scripts/scaling_runner.py"

combo_k="${COMBO_K:-1000}"
combo_rep="${COMBO_REP:-0}"
n_workers="${SLURM_NTASKS}"

output_dir="${1:-${SITE_SCRATCH_ROOT:-/tmp}/scaling_combo_dbg_${SLURM_JOB_ID:-local}}"
mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Mirror ALL output to shared scratch — the #SBATCH --output above goes to the
# compute node's /tmp, which is not on any shared mount.
exec > "$output_dir/job.log" 2>&1
echo "[combo-dbg] full log -> $output_dir/job.log"
echo "[combo-dbg] k=$combo_k rep=$combo_rep n_workers=$n_workers output_dir=$output_dir"

# Match the real scaling wrapper's teardown environment exactly.
export PROPULATE_SKIP_DISCONNECT="${PROPULATE_SKIP_DISCONNECT:-1}"

# Per-rank in-process traceback dumper (no ptrace needed — py-spy/gdb are blocked
# by ptrace_scope on the compute node). Each rank writes its OWN Python traceback
# (all threads) every SCALING_FAULTHANDLER_S seconds to pytraces/rank_NNN.txt, so
# the wedged ranks' LAST snapshot shows the stuck frame. Handled in
# scaling_runner.py:_install_faulthandler_dumper.
export SCALING_FAULTHANDLER_S="${SCALING_FAULTHANDLER_S:-5}"
export SCALING_FAULTHANDLER_DIR="$output_dir/pytraces"

# Environment setup (same hook as scaling_single.sh / repro_pscom_teardown.sh).
if [ -n "${SCALING_ENV_SETUP:-}" ]; then
    # shellcheck source=/dev/null
    source "$SCALING_ENV_SETUP"
else
    module restore nastjapy
    module load ParaStationMPI
    source "$nastjapy_path/.venv/bin/activate"
fi

# --- Liveness logger: the per-rank faulthandler dumps (above) do the actual
#     stack capture, ptrace-free. This watchdog only LOGS how many rank processes
#     remain alive over the teardown window, so the timeline shows whether ranks
#     exit (slow teardown) or all stay blocked until the kill (deadlock). ---
watch_start="${STACK_WATCH_START:-295}"
watch_interval="${STACK_WATCH_INTERVAL:-5}"
watch_count="${STACK_WATCH_COUNT:-20}"
host="$(hostname -s)"
# Match only the python rank processes (exclude the `srun` launcher).
rank_pat="python[0-9.]* ${runner}"
(
    sleep "$watch_start"
    echo "[watchdog] start: ntasks=$n_workers on ${host}; logging rank liveness through teardown"
    for i in $(seq 1 "$watch_count"); do
        live=$(pgrep -u "$USER" -f "$rank_pat" 2>/dev/null | grep -c .)
        echo "[watchdog] check $i: live_ranks=${live}/${n_workers}"
        [ "$live" -eq 0 ] && { echo "[watchdog] all ranks exited — teardown completed."; break; }
        sleep "$watch_interval"
    done
    echo "[watchdog] done."
) &
watchdog_pid=$!

# One combo, one MPI world, one teardown. NOTE: an earlier run showed the step is
# still SIGTERMed at ~wall+40s even with --wait=0 --kill-on-bad-exit=0, so the
# ~305s kill is NOT srun's straggler grace (likely an MPI_Abort when a rank exits
# mid-collective). We keep these flags (harmless); the per-rank faulthandler files
# capture the wedged frame regardless of the kill.
srun --wait=0 --kill-on-bad-exit=0 -n "$n_workers" \
    python "$runner" \
    --config "$config_path" \
    --output-dir "$output_dir" \
    --n-workers "$n_workers" \
    --k "$combo_k" \
    --replicate "$combo_rep" \
    --skip-finalize \
    --small < /dev/null
rc=$?

kill "$watchdog_pid" 2>/dev/null || true
echo "[combo-dbg] srun rc=$rc"

# Did the combo actually finish and write its shard? (present => the combo
# completed; absent => the wedge is terminal.) The runner nests outputs under an
# experiment-name subdir, so the shard lives at <out>/scaling/data/.
shard="$output_dir/scaling/data/raw_results_w${n_workers}_k${combo_k}.csv"
if [ -f "$shard" ]; then
    echo "[combo-dbg] SHARD WRITTEN: $shard ($(($(wc -l < "$shard") - 1)) records) — teardown completed."
else
    echo "[combo-dbg] NO SHARD at $shard — combo did not finish; see $output_dir/pytraces/ for the wedged ranks' last traceback."
fi
exit "$rc"
