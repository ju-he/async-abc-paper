# CommWorldMap 48-Rank Verification — MPI-01

**Phase:** 02-mpi-hardening
**Requirement:** MPI-01
**Driver:** experiments/scripts/verify_commworldmap_48.py
**SLURM script:** experiments/jobs/verify_commworldmap_48.sh
**Benchmark:** gaussian_mean, population=100, 3 generations, 48 ranks
**Cluster:** JUWELS, ParaStationMPI

## Submission

- **sbatch command:** `sbatch experiments/jobs/verify_commworldmap_48.sh <output_dir>`
- **Job ID:** _to be filled in_
- **Submitted at:** _to be filled in_
- **Output dir:** _to be filled in_

## Local Smoke Test (prerequisite)

- Command: `mpirun -n 2 ./nastjapy_copy/.venv/bin/python experiments/scripts/verify_commworldmap_48.py /tmp/verify_commworldmap_48_smoke.json`
- Outcome: _PASS / FAIL_
- Records returned: _n_
- Elapsed: _s_

## Cluster Run Outcome

- **SLURM exit code:** _0 = PASS, non-zero = FAIL_
- **Wall-clock elapsed (from JSON `elapsed_s`):** _s_
- **Records returned (`n_records`):** _n_
- **Hang observed:** _yes / no_
- **If hang:** py-spy dump of rank 0, identification of the bcast/send/recv call at hang point (see Risk 2 in mpi-evaluation.md)
- **Final status:** _PASS / FAIL_

## Follow-up

- If PASS: mark MPI-01 as complete in REQUIREMENTS.md traceability; proceed
  to Plan 02-03 (scaling runner migration to CommWorldMap per D-03).
- If FAIL: MPI-01 remains open; scaling runner stays on Candidate 2 (D-03);
  diagnose the specific hang point before attempting further hardening.
