# Scratch inode recovery — async-abc on JUWELS

Measured 2026-08-14 against `/p/scratch/tissuetwin/herold2/async-abc`.
Every number below is reproducible with the commands in "How this was measured".

## 1. The constraint is inodes, not bytes

`jutil project dataquota -p tissuetwin`, 2026-08-14 17:21:

| filesystem | data usage | data soft | inode usage | inode soft | inode hard |
|---|---|---|---|---|---|
| **scratch** | 32.4 TB | 96.6 TB (33%) | **3,925,347** | 4,000,000 | 4,400,000 |
| project1 | 11.5 TB | 16.0 TB (72%) | 1,901,611 | 3,000,000 | 3,300,000 |

Scratch is at **98.1% of the inode soft limit** — 74,653 inodes of headroom — while
using only a third of its byte allowance. Any fix has to remove *files*, not bytes.

## 2. Where async-abc's inodes are

`async-abc` holds **269,319 inodes / 5.2 TB**, i.e. 6.9% of the project's inodes.

File-extension histogram over the whole folder:

| ext | count | note |
|---|---|---|
| **jsonl** | **196,318** | **73% of the folder's inodes** |
| csv | 12,951 | `raw_results_*`, `throughput_*`, `budget_*` — the analysed data |
| pickle / bkp | 7,621 / 7,621 | propulate checkpoints |
| json | 7,426 | configs, `status.json`, metrics |
| pdf / png | 4,830 / 4,828 | figures |
| db | 4,218 | pyABC SQLite stores |
| everything else | ~3,900 | sbatch, logs, out/err |

Of the 196,318 jsonl:

- **196,171 live in 4,208 `*_attempts/` directories** — 2,758 GB.
- **147 are `sbc_trials.jsonl`** — the real SBC payload. Keep.

The attempt traces are therefore **73% of the folder's inodes and 53% of its bytes**.

### What they are

`experiments/async_abc/inference/_attempt_trace.py:33` opens one file per MPI rank
per run, `worker_<rank>_pid_<pid>.jsonl`, and appends one JSON line per simulator
call. A 384-rank run leaves 383 files in a single directory. One line:

```json
{"end_abs":1784047685.37,"loss":0.5458,"param_key":"[[\"division_rate\",0.0757],[\"motility\",0.2814]]",
 "params":{"division_rate":0.0757,"motility":0.2814},"pid":1417779,"seed":1824376106,
 "start_abs":1784047676.28,"worker_id":"358"}
```

Top-level distribution of the jsonl (all in `*_attempts/`):

```
37739 rerun_20260707      26716 old2       4300 lv_fair_baseline_20260701
36472 sbc_1k_v2_20260629  21492 full1      4106 run_full_20260626_1816
30932 old                 11267 small4     2350 sbc_1k_20260629
                           9863 small1     ... (tail)
```

## 3. Root cause

`experiments/scripts/scaling_runner.py:630` defines `_cleanup_combo_artifacts()`,
which removes exactly these directories once a combo's records are merged:

```python
_remove_path(output_dir.logs / f"{method}_rep{replicate}_seed{seed}__{tag}_attempts")
```

**It has no production caller.** The only reference in the repo is
`experiments/tests/test_runners.py:706`, which monkeypatches it and asserts
`cleanup_calls == []`. So every scaling run since the function was written has
leaked its full attempt trace, and the leak scales with rank count — the 384-rank
CPM sweeps are the worst offenders.

## 4. Why deletion is not safe — archive instead

The traces *are* folded into the run's CSV at finalize
(`load_attempt_events` → `attempt_records_from_events`, `_attempt_trace.py:65,104`).
But the CSV is **not a strict superset of the traces**. Measured on
`rerun_20260707/scaling_cpm`, combo `w384_k1000`:

| source | rows |
|---|---|
| 5 replicate `*_attempts/` dirs, all jsonl lines | **272,044** |
| `raw_results_w384_k1000.csv`, `abc_smc_baseline…\|simulation_attempt` | **250,976** |
| deficit | **21,068 (7.7%)** |

Cause not determined — candidates are `_dedupe_records()` in
`_merge_and_write_combo_records` (`scaling_runner.py:649`) and truncation at the
wall-time budget. Separately, the conversion drops `pid` and `param_key`.

**Conclusion: the traces contain rows the CSVs do not.** Deleting them loses data.
Archiving them does not. The plan below never deletes an unarchived byte.

### Compressibility

Measured on a real trace directory
(`rerun_20260707/scaling_cpm/logs/abc_smc_baseline_rep0_seed1654615998__w384_k1000_attempts`,
383 files): **16.5 MB → 3.5 MB, 4.70×** with `tar | pigz -6`.

Projected: **2,758 GB → ~590 GB**, i.e. ~2.17 TB returned.

## 5. Plan

### Phase 0 — inventory, no mutation

Write a manifest per run directory before touching anything:

```
find $RUN -type d -name '*_attempts' -print0 |
  xargs -0 -I{} sh -c 'find "{}" -type f -printf "%p\t%s\n"' > $RUN/attempts_manifest.tsv
```

One extra inode per run directory (~40 total). Keeps the file list greppable
without decompressing anything, forever.

### Phase 1 — archive, still no deletion

For each of the 4,208 `X_attempts/` directories, write a sibling
`X_attempts.tar.zst`. Run it as a **Slurm job, not on a login node** — this reads
2.76 TB.

```bash
#!/bin/bash
# archive_attempts.sh — one arg: the run directory to process
set -euo pipefail
RUN="$1"
find "$RUN" -type d -name '*_attempts' -print0 | xargs -0 -P 24 -I{} bash -c '
  d="$1"; t="${d}.tar.zst"
  [ -e "$t" ] && exit 0                      # idempotent: skip finished dirs
  tar -C "$(dirname "$d")" -cf - "$(basename "$d")" \
    | zstd -3 -T2 -q -o "${t}.part"
  mv "${t}.part" "$t"                        # atomic publish
' _ {}
```

`.part` + `mv` means an interrupted job never leaves a truncated archive that a
later verify could mistake for a good one. `-P 24 × -T2` saturates a 48-core node.

Estimated cost: 2.76 TB read + ~0.5 TB write, ~2–4 h on one node ≈ **4 node-hours**
against the `async-abc-paper` budget (2000 nh/session). Negligible.

### Phase 2 — verify, then delete

Delete a directory only after its archive proves complete:

```bash
for d in $(find "$RUN" -type d -name '*_attempts'); do
  t="${d}.tar.zst"
  want=$(find "$d" -type f | wc -l)
  have=$(zstd -dc "$t" | tar -tf - | grep -c '\.jsonl$')
  if [ "$want" -eq "$have" ]; then rm -rf "$d"; else echo "MISMATCH $d $want != $have"; fi
done
```

Count equality is the minimum bar. For the runs the paper actually cites
(`rerun_20260707`, `kfrontier_20260729`, `sbc1000_20260729`, `twin*_20260729`)
compare per-file sha256 instead — the extra read is cheap relative to being wrong.

### Phase 3 — stop the recurrence

`_cleanup_combo_artifacts` must not be wired up as-is: §4 shows deletion loses
rows. Change it to *archive* on finalize — tar+zstd the attempts dir, then remove
the directory — and call it from the scaling runner's finalize path. Then every
future run costs 1 inode per combo instead of `n_workers`.

### Phase 4 — the rest of scratch (out of this folder's scope)

async-abc is 6.9% of the project's inodes. A cached scan on scratch
(`/p/scratch/tissuetwin/herold2/du_inodes_200326d2.out`) put `spheroids` at
**2,410,108 inodes** — 63% of the project — with `spheroids/inf` alone at 2,179,078.
A fresh whole-scratch scan is running; the async-abc work should not be mistaken
for a fix to the quota.

## 6. Result — DONE 2026-08-14

| | predicted | **measured** |
|---|---|---|
| async-abc inodes | 269,319 → ~73,200 | 269,319 → **73,158** |
| async-abc bytes | 5.2 TB → ~2.9 TB | 5.2 TB → **2.8 TB** |
| trace bytes | 2,758 GB → ~590 GB | 2,758.18 GB → **530.10 GB (5.20×)** |
| inodes freed | ~196,171 | **200,379** |
| project inodes | — | 3,925,347 → ~3,724,968 |
| % of inode soft limit | 98.1% → 93.2% | **98.1% → 93.1%** |
| headroom to soft limit | 74,653 → ~270,800 | 74,653 → **~275,000 (3.7×)** |

Post-run state verified directly (`du --inodes`, not the quota table, which
refreshes only a few times a day and still read 17:21 an hour after the purge):

- 4,208 archives present, **0** residual `*_attempts/` directories, **0** stray
  `.part` files.
- **147 `.jsonl` remain** — exactly the `sbc_trials.jsonl` set. The SBC payload was
  never in scope and was not touched.
- Spot-restore of a real archive (`…w384_k1000_attempts.tar.gz`): 383 files, exact
  name+size match against `manifest_pre.tsv`, first record byte-identical to the
  copy read out of the live directory before any of this ran.

Jobs: **14198871** archive+verify, COMPLETED 00:46:31, 4208/4208 verified, 0
failures. **14198894** purge, COMPLETED 00:04:54, 4208 PURGED / 0 skipped.
Combined cost ~0.85 node-hours.

Not a permanent fix on its own — Phase 3 stops the bleeding, Phase 4 is where the
quota actually gets solved.

### Getting data back

```bash
cd /p/scratch/tissuetwin/herold2/async-abc
pigz -dc <run>/logs/<name>_attempts.tar.gz | tar -x -C <run>/logs/    # or: tar -xzf
grep -F '<name>_attempts/' _archive/manifest_pre.tsv                  # what was in it
```

## 7. Decisions taken (2026-08-14)

1. **Archives live in place on scratch**, as a sibling `X_attempts.tar.zst` next to
   each directory they replace. Author's call.
2. **Scope is all 4,208 directories** — no run tree exempted. Author's call:
   "archive everything, we will unzip what is needed later."
3. **Verification is byte-for-byte everywhere**, not count-equality. `tar --diff`
   through `zstd -dc` compares contents, size, mode and mtime of every member
   against the live original, and the zstd frame checksum validates the stream. It
   costs one extra full read and removes the need to decide which runs "matter".

## 8. Implementation

Two job scripts, deliberately split so nothing is deleted until an archive pass has
been inspected:

- `experiments/jobs/archive_attempts.sbatch <base>` — inventory, archive, verify.
  **Deletes nothing.** Writes `_archive/manifest_pre.tsv` (the permanent pre-archive
  record), `_archive/attempts_dirs.txt`, and `_archive/verify.log`. Idempotent: a
  complete archive is never rebuilt, so a job that hits the wall clock is just
  resubmitted. Archives are published `.part` → `mv`, so an interrupted job cannot
  leave a truncated file that a later verify would accept.
- `experiments/jobs/purge_verified_attempts.sbatch <base> [--dry-run]` — deletes
  only directories logged `OK`, and refuses to start at all if the verify log holds
  any failure. Re-tests each archive with `zstd -t` immediately before removing its
  directory: a second independent read, after the archive job's `tar --diff`.

### Compressor: pigz, not zstd

The first submission (job 14198859) failed in 11 s on `FATAL: zstd not found`.
`/usr/bin/zstd` exists on the **login** nodes but not on the **compute** nodes, and
JUWELS ships no zstd module (`module spider zstd` → not found). `/usr/bin/pigz` is
present unconditionally, so both scripts use `tar | pigz -6 -p 2` and `.tar.gz`.
gzip's CRC32 plays the same verification role zstd's frame checksum would have, and
`.tar.gz` is the friendlier format for pulling one run back out later.

**Lesson: validate on a compute node, not the login node** — the two environments
differ in exactly the way that matters here.

### Validated on a compute node before resubmission

`experiments/jobs` scripts run against a 3-directory / 9-file fixture under
`srun --partition=devel`, with the real `pigz` / `tar 1.34`:

| test | result |
|---|---|
| archive + verify pass | 3/3 verified, 0 failures |
| `--dry-run` deletes nothing | PASS — 9/9 files intact |
| deliberately corrupted archive (7 bytes overwritten at offset 20) | PASS — `SKIP_BAD_ARCHIVE`, its directory survived; the two sound ones purged |
| extract every archive, `diff -r` against a gold copy | PASS — byte-identical |

Fixtures removed afterwards.

### Submitted

Job **14198871**, 1 node, 8 h limit, `--account=tissuetwin --partition=batch`,
submitted via `ssh juwels sbatch` (the established path — the MCP budget gate is
avoided per `.plans/HANDOFF_2026-07-30.md`). Worst case 8 node-hours against the
2000 nh `async-abc-paper` allocation.

The purge job is **not** submitted until job 14198871's report shows 0 archive
failures and 0 verify failures.

## 9. Follow-ups not covered here

- **Phase 3 above is not done**: `_cleanup_combo_artifacts` is still uncalled, so
  the next scaling run leaks traces again. It must be rewritten to archive rather
  than delete, per §4.
- Untouched inode pools inside async-abc: 7,621 `.pickle` + 7,621 `.bkp` propulate
  checkpoints (15,242 inodes) and 4,218 `.db` pyABC stores. Same tar-in-place
  treatment would apply if more headroom is needed.
- `spheroids` (§5, Phase 4) is where the project's quota problem actually lives.

## How this was measured

```bash
ssh juwels 'jutil project dataquota -p tissuetwin'
B=/p/scratch/tissuetwin/herold2/async-abc
ssh juwels "du --inodes -d 2 $B | sort -rn | head -80"
ssh juwels "find $B -type f -printf '%f\n' | sed -E 's/.*\.//' | sort | uniq -c | sort -rn"
ssh juwels "find $B -type d -name '*_attempts' | wc -l"
ssh juwels "find $B -type d -name '*_attempts' -exec find {} -type f \; | wc -l"
ssh juwels "find $B -type f -name '*.jsonl' -not -path '*_attempts/*' | wc -l"
```
