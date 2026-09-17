#!/bin/bash -l
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=2
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_archive_attempts
#SBATCH --output=/tmp/abc_archive_attempts-%j.out
#
# Archive every `*_attempts/` trace directory under <base> into a sibling
# `<name>_attempts.tar.gz`, then VERIFY each archive byte-for-byte against the
# still-present original. Deletes NOTHING — purge_verified_attempts.sbatch does
# that, gated on this job's verify log.
#
# Usage (from the login node):
#   sbatch experiments/jobs/archive_attempts.sbatch /p/scratch/tissuetwin/herold2/async-abc
#
# Idempotent: an existing, complete `.tar.zst` is not rebuilt, so a job that hits
# the wall clock can simply be resubmitted. Archives are published via a `.part`
# + `mv`, so an interrupted job never leaves a truncated archive that a later
# verify could mistake for a good one.
set -euo pipefail

BASE="${1:?usage: $(basename "$0") <base-dir>}"
[ -d "$BASE" ] || { echo "FATAL: no such directory: $BASE" >&2; exit 1; }

ARC="$BASE/_archive"
mkdir -p "$ARC"

DIRLIST="$ARC/attempts_dirs.txt"
MANIFEST="$ARC/manifest_pre.tsv"
VERIFYLOG="$ARC/verify.log"

# Mirror everything to shared scratch; the #SBATCH --output above lands on the
# compute node's /tmp and is lost when the allocation ends.
exec > >(tee -a "$ARC/archive-${SLURM_JOB_ID:-manual}.log") 2>&1

# 24 directories in flight x 2 pigz threads each = 48 hardware threads.
# pigz, not zstd: the compute nodes carry no zstd binary and JUWELS ships no zstd
# module (checked 2026-08-14), while /usr/bin/pigz is present unconditionally.
# gzip is also the friendlier format for pulling one run back out years later.
NPAR=$(( ${SLURM_CPUS_PER_TASK:-48} / 2 ))

echo "=== archive_attempts start $(date -Is) ==="
echo "base=$BASE  parallel=$NPAR"
command -v pigz >/dev/null || { echo "FATAL: pigz not found" >&2; exit 1; }

# ---------------------------------------------------------------- phase 0
# Inventory BEFORE anything is written. This is the permanent record of what
# existed, and it survives even if every later phase fails.
if [ ! -s "$DIRLIST" ]; then
    echo "[phase 0] enumerating *_attempts directories ..."
    # `! -path '*_attempts/*'` drops nested matches: a nested dir is already
    # inside its parent's tarball, and archiving it twice would double-count.
    find "$BASE" -type d -name '*_attempts' ! -path '*_attempts/*' -print \
        | sort > "$DIRLIST.part"
    mv "$DIRLIST.part" "$DIRLIST"
fi
echo "[phase 0] $(wc -l < "$DIRLIST") attempt directories"

if [ ! -s "$MANIFEST" ]; then
    echo "[phase 0] writing pre-archive manifest ..."
    find "$BASE" -type f -path '*_attempts/*' -printf '%p\t%s\n' > "$MANIFEST.part"
    mv "$MANIFEST.part" "$MANIFEST"
fi
echo "[phase 0] manifest: $(wc -l < "$MANIFEST") files, $(awk -F'\t' '{s+=$2} END {printf "%.2f GB", s/1073741824}' "$MANIFEST")"

# ------------------------------------------------------- phases 1 and 2
# Per directory: build the archive if absent, then verify it against the live
# original. `tar --diff` compares CONTENTS, size, mode and mtime of every member
# and is silent only on an exact match; piping through `pigz -dc` additionally
# validates the gzip CRC32 end to end. A directory is reported OK only when tar
# exits 0 AND prints nothing.
archive_and_verify() {
    local d="$1"
    local t="${d}.tar.gz"
    local parent base out rc
    parent="$(dirname "$d")"
    base="$(basename "$d")"

    set -o pipefail

    if [ ! -f "$t" ]; then
        rm -f "${t}.part"
        if ! tar -C "$parent" -cf - "$base" | pigz -6 -p 2 > "${t}.part"; then
            rm -f "${t}.part"
            printf 'ARCHIVE_FAIL\t%s\n' "$d"
            return 0
        fi
        mv "${t}.part" "$t"
    fi

    set +e
    out=$( { pigz -dc "$t" | tar --diff -C "$parent" -f - ; } 2>&1 )
    rc=$?
    set -e

    if [ "$rc" -eq 0 ] && [ -z "$out" ]; then
        printf 'OK\t%s\n' "$d"
    else
        printf 'VERIFY_FAIL\t%s\trc=%s\n' "$d" "$rc"
        printf '%s\n' "$out" | sed "s|^|VERIFY_DIFF\t$d\t|"
    fi
}
export -f archive_and_verify

echo "[phase 1+2] archiving and verifying ..."
# Rebuild the verify log from scratch every run: a resubmitted job re-verifies
# the archives it already made, so the log always reflects one complete pass.
xargs -a "$DIRLIST" -d '\n' -P "$NPAR" -I{} \
    bash -c 'archive_and_verify "$@"' _ {} > "$VERIFYLOG.part"
mv "$VERIFYLOG.part" "$VERIFYLOG"

# ---------------------------------------------------------------- report
n_total=$(wc -l < "$DIRLIST")
n_ok=$(grep -c $'^OK\t' "$VERIFYLOG" || true)
n_afail=$(grep -c $'^ARCHIVE_FAIL\t' "$VERIFYLOG" || true)
n_vfail=$(grep -c $'^VERIFY_FAIL\t' "$VERIFYLOG" || true)

echo "=== report ==="
echo "directories      : $n_total"
echo "verified OK      : $n_ok"
echo "archive failures : $n_afail"
echo "verify failures  : $n_vfail"
# Upper bound on the recoverable inodes: every file the manifest recorded, plus
# the directories themselves. The purge job reports what was actually freed.
echo "inodes recoverable: <= $(( $(wc -l < "$MANIFEST") + n_total ))"
echo "archive bytes    : $(find "$BASE" -name '*_attempts.tar.gz' -printf '%s\n' \
    | awk '{s+=$1} END {printf "%.2f GB", s/1073741824}')"
echo "=== archive_attempts done $(date -Is) ==="

# Fail the job if anything did not verify, so a bad pass is visible in sacct
# rather than silently producing a short verified list.
[ "$n_afail" -eq 0 ] && [ "$n_vfail" -eq 0 ]
