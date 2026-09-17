#!/bin/bash -l
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=2
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_purge_attempts
#SBATCH --output=/tmp/abc_purge_attempts-%j.out
#
# Delete the `*_attempts/` directories that archive_attempts.sbatch verified.
# Reads `_archive/verify.log` and touches ONLY the directories logged `OK` — a
# directory that failed to archive or failed to verify is never deleted.
#
# Before each removal the archive is re-checked: it must exist, be non-empty,
# and pass `pigz -t` (full decompression + CRC32). That is a second, independent
# read of the archive after the byte-for-byte `tar --diff` the archive job
# already did.
#
# Usage (from the login node):
#   sbatch experiments/jobs/purge_verified_attempts.sbatch /p/scratch/tissuetwin/herold2/async-abc
#
# Dry run — list what would go, delete nothing:
#   sbatch experiments/jobs/purge_verified_attempts.sbatch <base> --dry-run
set -euo pipefail

BASE="${1:?usage: $(basename "$0") <base-dir> [--dry-run]}"
DRY="${2:-}"
[ -d "$BASE" ] || { echo "FATAL: no such directory: $BASE" >&2; exit 1; }

ARC="$BASE/_archive"
VERIFYLOG="$ARC/verify.log"
[ -s "$VERIFYLOG" ] || { echo "FATAL: no verify log at $VERIFYLOG — run archive_attempts.sbatch first" >&2; exit 1; }

exec > >(tee -a "$ARC/purge-${SLURM_JOB_ID:-manual}.log") 2>&1

NPAR=${SLURM_CPUS_PER_TASK:-48}
echo "=== purge_verified_attempts start $(date -Is) ==="
echo "base=$BASE  parallel=$NPAR  dry_run=${DRY:-no}"

# Refuse to run if the archive pass left anything unresolved: a partial verify
# log means the archive job did not complete a full pass, and the OK lines in it
# may not cover every directory.
n_bad=$(grep -c -e $'^ARCHIVE_FAIL\t' -e $'^VERIFY_FAIL\t' "$VERIFYLOG" || true)
if [ "$n_bad" -ne 0 ]; then
    echo "FATAL: verify log holds $n_bad failed directories — resolve them before purging" >&2
    grep -e $'^ARCHIVE_FAIL\t' -e $'^VERIFY_FAIL\t' "$VERIFYLOG" | head -20 >&2
    exit 1
fi

grep $'^OK\t' "$VERIFYLOG" | cut -f2 > "$ARC/verified.txt"
echo "verified directories: $(wc -l < "$ARC/verified.txt")"

purge_one() {
    local d="$1"
    local t="${d}.tar.gz"

    [ -d "$d" ] || { printf 'ALREADY_GONE\t%s\n' "$d"; return 0; }

    if [ ! -s "$t" ]; then
        printf 'SKIP_NO_ARCHIVE\t%s\n' "$d"
        return 0
    fi
    if ! pigz -t "$t" 2>/dev/null; then
        printf 'SKIP_BAD_ARCHIVE\t%s\n' "$d"
        return 0
    fi

    local n
    n=$(find "$d" -mindepth 1 | wc -l)
    if [ -n "$DRY_RUN" ]; then
        printf 'WOULD_PURGE\t%s\t%s\n' "$d" "$n"
        return 0
    fi
    rm -rf "$d"
    printf 'PURGED\t%s\t%s\n' "$d" "$n"
}
export -f purge_one
export DRY_RUN=""
[ "$DRY" = "--dry-run" ] && export DRY_RUN=1

xargs -a "$ARC/verified.txt" -d '\n' -P "$NPAR" -I{} \
    bash -c 'purge_one "$@"' _ {} > "$ARC/purge.log"

echo "=== report ==="
for tag in PURGED WOULD_PURGE ALREADY_GONE SKIP_NO_ARCHIVE SKIP_BAD_ARCHIVE; do
    printf '%-18s %s\n' "$tag" "$(grep -c "^$tag"$'\t' "$ARC/purge.log" || true)"
done
echo "inodes freed     : $(grep -e '^PURGED'$'\t' -e '^WOULD_PURGE'$'\t' "$ARC/purge.log" \
    | awk -F'\t' '{s+=$3+1} END {print s+0}')"
echo "=== purge_verified_attempts done $(date -Is) ==="
