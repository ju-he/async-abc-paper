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
backend_path="${SIM_BACKEND_PATH:?SIM_BACKEND_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"

output_dir="${1:-${SITE_SCRATCH_ROOT:-/tmp}/repro_pscom_${SLURM_JOB_ID:-local}}"
mkdir -p "$output_dir"
export REPRO_OUT="$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Mirror ALL job output (incl. the bash -x module-load trace and srun/python
# output) to the shared scratch dir. The #SBATCH --output above goes to the
# compute node's /tmp, which is NOT on any shared mount, so without this the log
# is unreadable from a login node / the sshfs mount. job.log lands next to the
# markers and is readable off the mount.
exec > "$output_dir/job.log" 2>&1
echo "[repro] full log -> $output_dir/job.log"

# Environment setup. Override SCALING_ENV_SETUP to point at a script that loads a
# different MPI stack + activates a matching venv — e.g. an OpenMPI-linked mpi4py
# venv to sidestep the ParaStation pscom teardown hang at high message volume.
# Default keeps ParaStation and the simulation backend unchanged. The setup script owns both the
# module loads AND `source <venv>/bin/activate`.
if [ -n "${SCALING_ENV_SETUP:-}" ]; then
    # shellcheck source=/dev/null
    source "$SCALING_ENV_SETUP"
else
    module restore sim_backend
    module load ParaStationMPI
    source "$backend_path/.venv/bin/activate"
fi

# --- Stack-dump watchdog: if the run hasn't finished by the expected time, dump
#     native stacks of the local ranks so a hang shows its wedged frame (the MPI/
#     pscom call, or wherever it is stuck after teardown). Defaults ON for this
#     diagnostic script; set REPRO_PYSPY=0 to skip. Uses py-spy if available, else
#     falls back to gdb (a system tool). Stacks land on the scratch mount. ---
watchdog_pid=""
if [ "${REPRO_PYSPY:-1}" != "0" ]; then
    watchdog_after=$(( ${REPRO_COMBOS:-6} * ${REPRO_WALL_S:-20} + 120 ))
    (
        sleep "$watchdog_after"
        host="$(hostname -s)"
        echo "[watchdog] run exceeded ${watchdog_after}s — dumping local rank stacks on ${host}" >&2
        for pid in $(pgrep -u "$USER" -f 'repro_pscom_teardown.py'); do
            out="$output_dir/stack_${host}_${pid}.txt"
            if command -v py-spy >/dev/null 2>&1; then
                py-spy dump --native --pid "$pid" > "$out" 2>&1 || true
            else
                gdb -p "$pid" -batch -ex "thread apply all bt" > "$out" 2>&1 || true
            fi
        done
        echo "[watchdog] stacks -> $output_dir/stack_${host}_*.txt" >&2
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
