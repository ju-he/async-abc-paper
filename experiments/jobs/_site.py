"""Site-specific SLURM defaults based on $SYSTEMNAME.

Mirror of experiments/jobs/site_env.sh in Python. Used by the submit_*.py
launchers to auto-pick --account/--partition defaults. CLI flags continue
to override.
"""
from __future__ import annotations

import os

SITE_TABLE: dict[str, tuple[str, str]] = {
    "jureca":  ("eats-rna",   "dc-cpu"),
    "juwels":  ("tissuetwin", "batch"),
    "jupiter": ("tissuetwin", "batch"),
}


def detect_defaults() -> tuple[str, str]:
    """Return (account, partition) for the current $SYSTEMNAME.

    Honors SLURM_ACCOUNT_OVERRIDE / SLURM_PARTITION_OVERRIDE env vars.
    Raises RuntimeError on unknown $SYSTEMNAME unless both override env vars
    are set (in which case the overrides alone fully define the defaults —
    useful for test harnesses and one-off runs on unlisted sites).
    """
    override_account = os.environ.get("SLURM_ACCOUNT_OVERRIDE")
    override_partition = os.environ.get("SLURM_PARTITION_OVERRIDE")
    if override_account and override_partition:
        return override_account, override_partition

    sysname = os.environ.get("SYSTEMNAME", "")
    if sysname not in SITE_TABLE:
        raise RuntimeError(
            f"Unknown SYSTEMNAME={sysname!r}. "
            f"Pass --account/--partition explicitly, set SLURM_ACCOUNT_OVERRIDE / "
            f"SLURM_PARTITION_OVERRIDE, or extend experiments/jobs/_site.py."
        )
    account, partition = SITE_TABLE[sysname]
    account = override_account or account
    partition = override_partition or partition
    return account, partition
