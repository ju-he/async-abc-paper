#!/bin/bash -x
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=00:20:00
#SBATCH --partition=batch
#SBATCH --job-name=repro_pscom
#SBATCH --output=/tmp/repro_pscom-%j.out
#
# Reproduce + localize the ParaStation pscom MPI_Comm_free teardown hang that
# kills the >=48-rank standalone scaling jobs. See repro_pscom_teardown.py.
#
# Submit (account/partition/paths auto-injected):
#   experiments/jobs/submit.sh experiments/jobs/repro_pscom_teardown.sh [output_dir]
#
# Knobs (export before submitting, or via submit.sh --export):
#   REPRO_COMBOS (default 6)  REPRO_WALL_S (default 20, integer s)  REPRO_K (default 1000)
#   REPRO_PYSPY=1  -> auto py-spy dump of local ranks if the run hangs (needs `pip install py-spy`)
#
# A/B a candidate fix by re-running with one of:
#   --export=ALL,...,PROPULATE_SKIP_DISCONNECT=1     (skip the hanging Free)
#   an OpenMPI-linked mpi4py instead of ParaStation   (different teardown path)
#   --export=ALL,...,PSP_<tunable>=<val>              (pscom teardown tuning)
#
# For w=96 (2 nodes): set --nodes=2 --ntasks=96 at submit time. The auto py-spy
# watchdog only dumps ranks on THIS (batch) node; for the other node attach
# manually (see MANUAL ATTACH below).
#

set -u

# Paths injected by experiments/jobs/submit.sh via `sbatch --export`.
nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"

output_dir="${1:-${SITE_SCRATCH_ROOT:-/tmp}/repro_pscom_${SLURM_JOB_ID:-local}}"
mkdir -p "$output_dir"
export REPRO_OUT="$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Environment setup. Override SCALING_ENV_SETUP to point at a script that loads a
# different MPI stack + activates a matching venv — e.g. an OpenMPI-linked mpi4py
# venv to sidestep the ParaStation pscom teardown hang at high message volume.
# Default keeps ParaStation+nastja unchanged. The setup script owns both the
# module loads AND `source <venv>/bin/activate`.
if [ -n "${SCALING_ENV_SETUP:-}" ]; then
    # shellcheck source=/dev/null
    source "$SCALING_ENV_SETUP"
else
    module restore nastjapy
    module load ParaStationMPI
    source "$nastjapy_path/.venv/bin/activate"
fi

# --- Optional py-spy watchdog: if the step hangs past the expected runtime, dump
#     native C stacks of the local ranks. A rank wedged in pscom_close shows the
#     ParaStation teardown frames; that confirms the Free() is the stuck call. ---
watchdog_pid=""
if [ "${REPRO_PYSPY:-0}" = "1" ] && command -v py-spy >/dev/null 2>&1; then
    watchdog_after=$(( ${REPRO_COMBOS:-6} * ${REPRO_WALL_S:-20} + 120 ))
    (
        sleep "$watchdog_after"
        host="$(hostname -s)"
        echo "[watchdog] run exceeded ${watchdog_after}s — dumping local python stacks on ${host}" >&2
        for pid in $(pgrep -u "$USER" -f 'repro_pscom_teardown'); do
            py-spy dump --native --pid "$pid" > "$output_dir/pyspy_${host}_${pid}.txt" 2>&1 || true
        done
        echo "[watchdog] dumps written to $output_dir/pyspy_${host}_*.txt" >&2
    ) &
    watchdog_pid=$!
fi

srun -n "${SLURM_NTASKS}" python "$experiments_dir/jobs/repro_pscom_teardown.py"
rc=$?

[ -n "$watchdog_pid" ] && kill "$watchdog_pid" 2>/dev/null
exit "$rc"

# === MANUAL ATTACH (no py-spy watchdog, or multi-node) =======================
# While the job is hung (log shows "FREE start" on some rank with no "FREE end"):
#   1. Find the hung node(s):   squeue -j <jobid> -o "%N"
#   2. Open a shell on a node:  srun --jobid=<jobid> --overlap -w <node> --pty bash
#   3. Find a stuck rank PID:    pgrep -u "$USER" -f repro_pscom_teardown
#   4. Native stack (C frames):  py-spy dump --native --pid <PID>
#      (or:  gdb -p <PID> -batch -ex 'thread apply all bt' )
# A stack ending in pscom_close / psport / MPI_Comm_free confirms the teardown
# is the stuck call -> PROPULATE_SKIP_DISCONNECT / pscom tuning / MPI swap apply.
# A stack elsewhere (e.g. a collective in the next combo's setup) means the fix
# must target that call instead.
