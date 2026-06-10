#!/usr/bin/env bash
# Source this file (do not execute). Exports SITE_ACCOUNT and SITE_PARTITION
# based on $SYSTEMNAME (set by JSC on each login node).
#
# Honors SLURM_ACCOUNT_OVERRIDE / SLURM_PARTITION_OVERRIDE if set.
# Fails loudly on unknown $SYSTEMNAME — extend the case statement to add a site.

case "${SYSTEMNAME:-}" in
    jureca)  SITE_ACCOUNT="eats-rna"   ; SITE_PARTITION="dc-cpu" ;;
    juwels)  SITE_ACCOUNT="tissuetwin" ; SITE_PARTITION="batch"  ;;
    jupiter) SITE_ACCOUNT="tissuetwin" ; SITE_PARTITION="batch"  ;;
    *)
        echo "site_env.sh: unknown SYSTEMNAME='${SYSTEMNAME:-<unset>}'." >&2
        echo "  Set SLURM_ACCOUNT_OVERRIDE and SLURM_PARTITION_OVERRIDE, or extend site_env.sh." >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

SITE_ACCOUNT="${SLURM_ACCOUNT_OVERRIDE:-$SITE_ACCOUNT}"
SITE_PARTITION="${SLURM_PARTITION_OVERRIDE:-$SITE_PARTITION}"
export SITE_ACCOUNT SITE_PARTITION
