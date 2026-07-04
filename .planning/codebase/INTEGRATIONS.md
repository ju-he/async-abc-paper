# External Integrations

**Analysis Date:** 2026-04-08

## APIs & External Services

**No external web APIs or cloud services.** This is a self-contained scientific computing project. All computation runs locally or on HPC clusters.

## Inference Libraries (Key Integration Points)

**pyABC:**
- Purpose: Sequential Monte Carlo ABC inference (synchronous baseline)
- Package: `pyabc` (optional dependency)
- Entry point: `experiments/async_abc/inference/pyabc_wrapper.py`
- Sampler factory: `experiments/async_abc/inference/pyabc_sampler.py`
- Parallel backends: `MulticoreEvalParallelSampler` (multicore), `MappingSampler` (MPI via `CommWorldMap`), `DaskDistributedSampler` (Dask)
- History storage: SQLite via pyABC's built-in `History` class
- Helper: `experiments/async_abc/inference/_pyabc_history.py`, `experiments/async_abc/inference/_pyabc_common.py`

**Propulate:**
- Purpose: Asynchronous evolutionary ABC-PMC (the novel method)
- Package: `propulate` (vendored in `propulate/` and as optional dependency)
- Entry point: `experiments/async_abc/inference/propulate_abc.py`
- Uses `ABCPMC` propagator and `Propulator` from the Propulate library
- Communication: MPI via `mpi4py`, `MPI.COMM_WORLD.Dup()` for per-run isolation

**sim_backend / sim_backend:**
- Purpose: realistic workload Model simulation backend for the realistic workload benchmark
- Package: Vendored in `sim_backend_venv/src/`
- Entry point: `experiments/async_abc/benchmarks/realistic_workload.py`
- Modules: simulation, data, inference, ML, CLI, visualization
- Venv: `sim_backend_venv/.venv` (fallback path)

## Data Storage

**Databases:**
- SQLite - pyABC history databases (generated at runtime, path constructed in `_pyabc_common.py`)
  - Transient, per-run databases for pyABC's internal state tracking

**File Storage:**
- Local filesystem / HPC parallel filesystem only
- CSV files: Primary output format for particle records (`experiments/async_abc/io/records.py`)
- JSON files: Experiment configuration (`experiments/configs/*.json`)
- HDF5 files: sim_backend simulation data (via `h5py` in sim_backend)
- PNG/PDF: Generated figures (`experiments/async_abc/plotting/export.py`)

**Caching:**
- None

## MPI Communication

**Framework:** mpi4py
- Rank detection: `experiments/async_abc/utils/mpi.py` (checks `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `SLURM_PROCID`, then `MPI.COMM_WORLD`)
- Communicator management: `MPI.COMM_WORLD.Dup()` for run isolation in Propulate
- Execution modes per method:
  - `async_propulate_abc`: `all_ranks` - all MPI ranks participate
  - `pyabc_smc`: `rank_zero` - rank 0 orchestrates, workers via sampler
  - `rejection_abc`: `rank_parallel` - independent parallel evaluations
  - `abc_smc_baseline`: `rank_zero` - rank 0 orchestrates

## Authentication & Identity

Not applicable - no authentication required. HPC access managed externally via SLURM account (`tissuetwin`).

## Monitoring & Observability

**Error Tracking:**
- None (crash loudly per CLAUDE.md policy)

**Logs:**
- Python `logging` module throughout
- Configured via `experiments/async_abc/utils/logging_utils.py`
- SLURM job output to `/tmp/abc_*-%j.out`

**Progress:**
- Custom `MethodProgressReporter` (`experiments/async_abc/utils/progress.py`)

**Metadata:**
- Git commit hash captured via `experiments/async_abc/utils/git.py`
- Run metadata written via `experiments/async_abc/utils/metadata.py`

## CI/CD & Deployment

**Hosting:**
- JUWELS HPC (Forschungszentrum Juelich) for production runs
- Local machines for development and small-scale testing

**CI Pipeline:**
- None detected (no `.github/workflows/`, no CI config files)

**Job Submission:**
- SLURM batch scripts in `experiments/jobs/`
  - `run_experiments.sh` - Main production experiments (48 MPI tasks)
  - `scaling_packed.sh` - Packed scaling experiments
  - `scaling_single.sh` - Single scaling experiments
  - `submit_scaling.py` - Python-driven SLURM submission for scaling grid
  - `submit_replicate_shards.py` - Sharded replicate submission

## Environment Configuration

**Required env vars:**
- None explicitly required (MPI rank vars detected opportunistically)
- SLURM environment provides: `SLURM_PROCID`, `OMPI_COMM_WORLD_RANK`, `PMI_RANK`

**Secrets:**
- None - no external services requiring credentials

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Optimal Transport (POT)

**Purpose:** Wasserstein distance computation for posterior convergence analysis
- Package: `POT>=0.8`
- Used in: `experiments/async_abc/analysis/convergence.py`
- Provides sliced-Wasserstein approximation for multi-dimensional posterior comparison
- Falls back to coordinate-wise average when POT unavailable

## Sharding System

**Purpose:** Parallel-safe output aggregation across SLURM jobs
- Shard I/O: `experiments/async_abc/utils/sharding.py`
- Shard finalization: `experiments/async_abc/utils/shard_finalizers.py`
- Pattern: Each worker writes `*_w<N>_k<K>.csv` shards; finalization merges them
- Enables cluster jobs to fill experiment grids in parallel without file conflicts

---

*Integration audit: 2026-04-08*
