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

# Environment setup (same hook as scaling_single.sh / repro_pscom_teardown.sh).
if [ -n "${SCALING_ENV_SETUP:-}" ]; then
    # shellcheck source=/dev/null
    source "$SCALING_ENV_SETUP"
else
    module restore nastjapy
    module load ParaStationMPI
    source "$nastjapy_path/.venv/bin/activate"
fi

# --- Stack watchdog: once teardown has started (some ranks gone) but stragglers
#     remain, dump the survivors' native stacks. Repeats so we both (a) capture
#     the wedged frame and (b) observe whether the stragglers ever clear. ---
# Wall-time break is ~300s; start dumping just after it and continue through the
# teardown window (the step is killed ~wall+40s, so ~15 rounds at 5s covers it).
watch_start="${STACK_WATCH_START:-300}"
watch_interval="${STACK_WATCH_INTERVAL:-5}"
watch_count="${STACK_WATCH_COUNT:-15}"
host="$(hostname -s)"
dump_dir="$output_dir/stacks"
mkdir -p "$dump_dir"

# Match only the python rank processes (exclude the `srun` launcher, whose
# cmdline also contains scaling_runner.py).
rank_pat="python[0-9.]* ${runner}"
(
    sleep "$watch_start"
    echo "[watchdog] start: ntasks=$n_workers on ${host}; dumping every ${watch_interval}s through the teardown window"
    prev_live=-1
    for i in $(seq 1 "$watch_count"); do
        pids=$(pgrep -u "$USER" -f "$rank_pat" 2>/dev/null)
        live=$(printf '%s\n' "$pids" | grep -c .)
        echo "[watchdog] check $i: live_ranks=${live}/${n_workers}"
        if [ "$live" -eq 0 ]; then
            echo "[watchdog] all ranks exited — teardown completed (slow, NOT a deadlock)."
            break
        fi
        # Dump UNCONDITIONALLY through the window — the wall-time break is at
        # ~300s so by watch_start we are in drain/post-run/teardown. Capturing
        # every round shows the progression (numpy sort/build -> the wedged
        # collective) and survives the step kill: dumps taken before the kill
        # are durable on scratch. A rank stuck across consecutive rounds at the
        # same frame is the wedge.
        seq_tag="$(printf '%02d' "$i")"
        echo "[watchdog] dumping ${live} rank stacks IN PARALLEL (round $seq_tag)"
        # Parallel snapshot: all ranks stay blocked at the collective until the
        # step is killed (~wall+40s), so we must catch all 48 within a few seconds
        # — sequential gdb (~1s each) would not finish before the kill. Each pid
        # is dumped in its own background job; we wait for the batch.
        for pid in $pids; do
            out="$dump_dir/stack_${host}_${pid}_${seq_tag}.txt"
            if command -v py-spy >/dev/null 2>&1; then
                ( py-spy dump --native --pid "$pid" > "$out" 2>&1 || true ) &
            else
                ( gdb -p "$pid" -batch -ex "thread apply all bt" > "$out" 2>&1 || true ) &
            fi
        done
        wait
        echo "[watchdog] round $seq_tag stacks -> $dump_dir/stack_${host}_*_${seq_tag}.txt"
        prev_live="$live"
        sleep "$watch_interval"
    done
    echo "[watchdog] done."
) &
watchdog_pid=$!

# One combo, one MPI world, one teardown. NOTE: an earlier run showed the step is
# still SIGTERMed at ~wall+40s even with --wait=0 --kill-on-bad-exit=0, so the
# ~305s kill is NOT srun's straggler grace (likely an MPI_Abort when a rank exits
# mid-collective). We keep these flags (harmless) but rely on the watchdog dumping
# the wedged stacks BEFORE that kill rather than on keeping the step alive.
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

# Did the combo actually finish and write its shard? (present => --wait=0 alone
# unblocks it; absent => the wedge is terminal.)
shard="$output_dir/data/raw_results_w${n_workers}_k${combo_k}.csv"
if [ -f "$shard" ]; then
    echo "[combo-dbg] SHARD WRITTEN: $shard ($(($(wc -l < "$shard") - 1)) records) — teardown completed."
else
    echo "[combo-dbg] NO SHARD at $shard — combo did not finish; see $dump_dir for the wedged stack."
fi
exit "$rc"
