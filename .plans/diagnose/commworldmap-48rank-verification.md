# CommWorldMap 48-Rank Verification — MPI-01

**Phase:** 02-mpi-hardening
**Requirement:** MPI-01
**Driver:** experiments/scripts/verify_commworldmap_48.py
**SLURM script:** experiments/jobs/verify_commworldmap_48.sh
**Benchmark:** gaussian_mean, population=100, 3 generations, 48 ranks
**Cluster:** JUWELS, ParaStationMPI

## Submission

- **sbatch command:** `sbatch experiments/jobs/verify_commworldmap_48.sh <output_dir>`
- **Job ID:** _not recorded by user_
- **Submitted at:** 2026-04-13 (approx)
- **Output dir:** runs/commworldmap_48_verify

## Local Smoke Test (prerequisite)

- Command: `mpirun -n 2 ./nastjapy_copy/.venv/bin/python experiments/scripts/verify_commworldmap_48.py /tmp/verify_commworldmap_48_smoke.json`
- Outcome: PASS
- Records returned: 193
- Elapsed: 2.6s

## Cluster Run Outcome

- **SLURM exit code:** 0 (PASS)
- **Wall-clock elapsed (from JSON `elapsed_s`):** _not recorded_
- **Records returned (`n_records`):** >0 (PASS)
- **Hang observed:** no
- **world_size:** 48
- **Final status:** PASS

## Follow-up

- If PASS: mark MPI-01 as complete in REQUIREMENTS.md traceability; proceed
  to Plan 02-03 (scaling runner migration to CommWorldMap per D-03).
- If FAIL: MPI-01 remains open; scaling runner stays on Candidate 2 (D-03);
  diagnose the specific hang point before attempting further hardening.
