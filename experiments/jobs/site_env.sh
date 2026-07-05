#!/usr/bin/env bash
# Source this file (do not execute). Exports SITE_ACCOUNT, SITE_PARTITION,
# SITE_SIM_BACKEND_PATH and SITE_SCRATCH_ROOT based on $SYSTEMNAME (set by JSC on
# each login node). This is the single source of truth for per-cluster SLURM
# and filesystem defaults; the Python mirror lives in _site.py.
#
# Source it ON THE LOGIN NODE (e.g. from submit.sh) — a submitted #SBATCH
# script runs from SLURM's spool copy and cannot locate this file, so the
# launchers inject the resolved paths into the job environment via
# `sbatch --export` instead.
#
# Honors SLURM_ACCOUNT_OVERRIDE / SLURM_PARTITION_OVERRIDE / SIM_BACKEND_PATH /
# SCRATCH_ROOT if set. Fails loudly on unknown $SYSTEMNAME — extend the case
# statement to add a site.

case "${SYSTEMNAME:-}" in
    jureca)
        SITE_ACCOUNT="eats-rna"   ; SITE_PARTITION="dc-cpu"
        SITE_SIM_BACKEND_PATH="/p/project1/eats-rna/$USER/sim_backend"
        SITE_SCRATCH_ROOT="/p/scratch/eats-rna/$USER/async-abc"
        ;;
    juwels)
        SITE_ACCOUNT="tissuetwin" ; SITE_PARTITION="batch"
        SITE_SIM_BACKEND_PATH="/p/project1/tissuetwin/herold2/sim_backend"
        SITE_SCRATCH_ROOT="/p/scratch/tissuetwin/herold2/async-abc"
        ;;
    jupiter)
        SITE_ACCOUNT="tissuetwin" ; SITE_PARTITION="batch"
        SITE_SIM_BACKEND_PATH="/p/project1/tissuetwin/herold2/sim_backend"
        SITE_SCRATCH_ROOT="/p/scratch/tissuetwin/herold2/async-abc"
        ;;
    *)
        echo "site_env.sh: unknown SYSTEMNAME='${SYSTEMNAME:-<unset>}'." >&2
        echo "  Set SLURM_ACCOUNT_OVERRIDE / SLURM_PARTITION_OVERRIDE / SIM_BACKEND_PATH / SCRATCH_ROOT, or extend site_env.sh." >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

SITE_ACCOUNT="${SLURM_ACCOUNT_OVERRIDE:-$SITE_ACCOUNT}"
SITE_PARTITION="${SLURM_PARTITION_OVERRIDE:-$SITE_PARTITION}"
SITE_SIM_BACKEND_PATH="${SIM_BACKEND_PATH:-$SITE_SIM_BACKEND_PATH}"
SITE_SCRATCH_ROOT="${SCRATCH_ROOT:-$SITE_SCRATCH_ROOT}"
export SITE_ACCOUNT SITE_PARTITION SITE_SIM_BACKEND_PATH SITE_SCRATCH_ROOT
