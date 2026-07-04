# Architecture

**Analysis Date:** 2026-04-08

## Pattern Overview

**Overall:** Registry-driven experiment framework with MPI-aware distributed execution

**Key Characteristics:**
- JSON config files drive all experiment parameters; validated against a schema at load time
- Inference methods and benchmark models are accessed via registries (dict lookups)
- MPI rank coordination is abstracted behind a thin helper layer (`async_abc.utils.mpi`)
- Runner scripts are thin entry points that delegate to a shared `run_benchmark_runner()` harness
- Results are captured as `ParticleRecord` dataclasses and serialized to CSV

## Layers

**Configuration Layer:**
- Purpose: Load, validate, and transform JSON experiment configs
- Location: `experiments/async_abc/io/`
- Contains: Config loading (`config.py`), validation schema (`schema.py`), output directory management (`paths.py`), record I/O (`records.py`)
- Depends on: Nothing (leaf layer)
- Used by: All runner scripts and the runner harness

**Benchmark Layer:**
- Purpose: Define simulation models with a uniform `simulate(params, seed) -> float` interface
- Location: `experiments/async_abc/benchmarks/`
- Contains: `GaussianMean` (`gaussian_mean.py`), `GandK` (`gandk.py`), `LotkaVolterra` (`lotka_volterra.py`), `RealisticWorkload` (`realistic_workload.py`)
- Depends on: numpy, scipy; RealisticWorkload depends on external sim_backend
- Used by: Runner harness via `make_benchmark(cfg["benchmark"])`
- Registry: `experiments/async_abc/benchmarks/__init__.py` maps name strings to classes

**Inference Layer:**
- Purpose: Implement ABC inference algorithms with a common runner signature
- Location: `experiments/async_abc/inference/`
- Contains: `propulate_abc.py` (async Propulate-ABC), `pyabc_wrapper.py` (pyABC SMC), `rejection_abc.py` (rejection ABC), `abc_smc_baseline.py` (ABC-SMC baseline)
- Depends on: Benchmark layer (simulate_fn), IO layer (OutputDir, ParticleRecord), optional external libs (propulate, pyabc, mpi4py)
- Used by: Runner harness via `run_method()` dispatch
- Registry: `experiments/async_abc/inference/method_registry.py` maps method names to runner callables
- Execution modes: `METHOD_EXECUTION_MODE` dict classifies each method as `all_ranks`, `rank_zero`, or `rank_parallel`

**Execution / Utils Layer:**
- Purpose: Shared runner harness, MPI coordination, sharding, seeding, progress reporting
- Location: `experiments/async_abc/utils/`
- Contains:
  - `runner.py` — Core `run_experiment()` and `run_method_distributed()` functions, timing estimation
  - `benchmark_runner.py` — Generic `run_benchmark_runner()` entry point with sharding support
  - `sharding.py` — `ShardLayout`, shard plan management, finalization for cluster parallel execution
  - `mpi.py` — Thin MPI/SLURM rank helpers (`get_rank`, `allgather`, `is_root_rank`)
  - `seeding.py` — Deterministic seed generation
  - `progress.py` — `MethodProgressReporter` for logging progress
  - `shard_finalizers.py` — Post-shard CSV merge and plot regeneration
- Depends on: Config layer, inference layer, benchmark layer
- Used by: All runner scripts

**Analysis Layer:**
- Purpose: Post-processing of ParticleRecord outputs (convergence, ESS, SBC, trajectories)
- Location: `experiments/async_abc/analysis/`
- Contains: `convergence.py` (Wasserstein curves, time-to-threshold), `sbc.py` (simulation-based calibration ranks/coverage), `ess.py` (effective sample size), `barrier.py` (barrier overhead), `trajectory.py` (loss/tolerance trajectories), `final_state.py` (final posterior extraction), `sensitivity.py`, `audit.py`
- Depends on: IO layer (ParticleRecord), numpy, scipy, POT (optimal transport)
- Used by: Plotting layer, runner scripts for post-run diagnostics
- Uses lazy imports via `__getattr__` in `__init__.py`

**Plotting Layer:**
- Purpose: Generate matplotlib figures for paper and diagnostics
- Location: `experiments/async_abc/plotting/`
- Contains: `common.py` (reusable plot primitives), `reporters.py` (high-level plot orchestrators), `export.py` (figure saving with metadata), `sbc.py` (SBC-specific plots)
- Depends on: Analysis layer, matplotlib
- Used by: Runner scripts (post-experiment), `replot.py` script
- Uses lazy imports via `__getattr__` in `__init__.py`

**Reporting Layer:**
- Purpose: Generate summary tables and runtime performance metrics
- Location: `experiments/async_abc/reporting/`
- Contains: `benchmark_reports.py` (analytic summary tables), `runtime_summary.py` (idle fraction, throughput), `plot_metadata.py` (metadata for paper figures)
- Depends on: IO layer, analysis layer
- Used by: Runner scripts, plotting layer

**Runner Scripts (Entry Points):**
- Purpose: CLI entry points for each experiment type
- Location: `experiments/scripts/`
- Contains: One `*_runner.py` per experiment type (see Entry Points below)
- Depends on: All layers above
- Most delegate to `run_benchmark_runner()` with injected dependencies

**Job Scripts:**
- Purpose: SLURM submission scripts and orchestration
- Location: `experiments/jobs/`
- Contains: Shell scripts for cluster submission, Python helpers for sharded job submission

## Data Flow

**Standard Benchmark Experiment:**

1. Runner script parses CLI args (`--config`, `--output-dir`, `--test`, `--small`, `--extend`)
2. `load_config()` reads JSON, validates schema, applies test-mode overrides, annotates run tier
3. `run_benchmark_runner()` sets up `OutputDir`, handles shard logic if applicable
4. `run_experiment()` iterates over `methods x replicates`
5. For each combination: `run_method_distributed()` handles MPI rank coordination based on method's execution mode
6. `run_method()` dispatches to the registered inference runner via `METHOD_REGISTRY`
7. Inference runner calls `benchmark.simulate(params, seed)` repeatedly, returns `List[ParticleRecord]`
8. `RecordWriter` appends records to `raw_results.csv`
9. Post-run: timing CSV written, plots generated, metadata saved

**MPI Execution Modes:**

- `all_ranks`: All MPI ranks cooperate (Propulate-ABC). Errors gathered via `allgather`, root writes CSV.
- `rank_zero`: Only root rank runs the method (pyABC). Non-root ranks poll a status file. Root writes status JSON for coordination.
- `rank_parallel`: Each rank independently handles a subset of replicates (rejection ABC). Results gathered via `allgather` after all complete.

**Sharded Execution (Cluster):**

1. `submit_replicate_shards.py` or `submit_scaling.py` creates SLURM array jobs
2. Each shard reads a shared plan JSON (`ShardLayout.plan_path`)
3. Shards execute their assigned replicate/workload indices independently
4. Last shard to complete triggers finalization: merge CSVs, regenerate plots

**State Management:**
- No in-memory state persistence between experiment runs
- All state persisted to filesystem: CSV files (records, timing), JSON (shard status, metadata), PNG (plots)
- `--extend` mode reads existing CSVs to skip completed `(method, replicate)` combinations

## Key Abstractions

**ParticleRecord:**
- Purpose: Represents one simulation evaluation result
- Defined in: `experiments/async_abc/io/records.py`
- Pattern: Dataclass with CSV serialization (`to_csv_row`, `from_csv_row`)
- Fields: method, replicate, seed, step, params (dict), loss, weight, tolerance, wall_time, worker_id, generation, record_kind, time_semantics, attempt_count

**OutputDir:**
- Purpose: Manages the `<base>/<name>/{plots,data,logs}/` directory tree
- Defined in: `experiments/async_abc/io/paths.py`
- Pattern: Simple path holder with `ensure()` for directory creation

**Benchmark (protocol):**
- Purpose: Uniform interface for simulation models
- Pattern: Duck-typed protocol — any object with `simulate(params, seed) -> float` and `limits: Dict[str, Tuple[float, float]]`
- Examples: `experiments/async_abc/benchmarks/gaussian_mean.py`, `experiments/async_abc/benchmarks/gandk.py`

**METHOD_REGISTRY / METHOD_EXECUTION_MODE:**
- Purpose: Map method name strings to runner callables and their MPI execution mode
- Defined in: `experiments/async_abc/inference/method_registry.py`
- Pattern: Module-level dicts, central dispatch via `run_method()`

**ShardLayout:**
- Purpose: Manages filesystem layout for one shard of a distributed experiment
- Defined in: `experiments/async_abc/utils/sharding.py`
- Pattern: Dataclass holding paths to shard workspace, plan JSON, status files

## Entry Points

**`experiments/run_all_paper_experiments.py`:**
- Location: `experiments/run_all_paper_experiments.py`
- Triggers: `srun python run_all_paper_experiments.py --experiments ... --output-dir ...`
- Responsibilities: Orchestrates all registered experiments sequentially, writes timing summaries

**Standard benchmark runners (delegate to `run_benchmark_runner`):**
- `experiments/scripts/gandk_runner.py`
- `experiments/scripts/gaussian_mean_runner.py`
- `experiments/scripts/lotka_volterra_runner.py`
- `experiments/scripts/cellular_potts_runner.py`
- `experiments/scripts/ablation_runner.py`
- `experiments/scripts/straggler_runner.py`
- `experiments/scripts/runtime_heterogeneity_runner.py`

**Specialized runners (custom main loops):**
- `experiments/scripts/scaling_runner.py` — Grid over `(n_workers, k)` combinations
- `experiments/scripts/sensitivity_runner.py` — Grid sweep over hyperparameter combinations
- `experiments/scripts/sbc_runner.py` — Simulation-based calibration with per-trial posterior extraction

**Utility scripts:**
- `experiments/scripts/replot.py` — Regenerate plots from existing data
- `experiments/scripts/generate_cpm_reference.py` — Generate cellular Potts reference data
- `experiments/scripts/repair_scaling_from_cluster_logs.py` — Fix scaling CSVs from logs

## Error Handling

**Strategy:** Crash loudly (per CLAUDE.md); errors propagated up MPI rank tree

**Patterns:**
- `run_method_distributed()` catches `ImportError` (missing optional deps like pyabc/propulate) and `Exception` separately
- `ImportError` skips the method with a warning; other exceptions re-raise
- MPI error coordination: errors from any rank collected via `allgather()`, first error re-raised on all ranks
- Rank-zero methods write error status to a JSON file so non-root ranks detect failures instead of hanging
- `TimeoutError` raised if non-root ranks wait too long for rank-zero status (configurable via `ABC_RANK_ZERO_TIMEOUT_S`)
- Shard failures write failure status JSON so other shards can detect and report

## Cross-Cutting Concerns

**Logging:** Python `logging` module; configured via `experiments/async_abc/utils/logging_utils.py`
**Validation:** JSON config validated at load time against schema in `experiments/async_abc/io/schema.py`; `ValidationError` raised for missing keys
**Authentication:** Not applicable (scientific computing, no user auth)
**Seeding:** Deterministic seed generation via `experiments/async_abc/utils/seeding.py` using `make_seeds(count, base_seed)`
**MPI Safety:** Communicator duplication for Propulate runs; barrier synchronization between execution phases; status-file coordination for rank-zero methods

---

*Architecture analysis: 2026-04-08*
