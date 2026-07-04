# Technology Stack

**Analysis Date:** 2026-04-08

## Languages

**Primary:**
- Python >=3.10 - All experiment code, analysis, plotting, and inference
- LaTeX - Paper manuscript (`latex/sn-article-template/sn-article.tex`)

**Secondary:**
- Bash - SLURM job scripts and bootstrap (`experiments/jobs/*.sh`)

## Runtime

**Environment:**
- CPython >=3.10 (root project), >=3.11 (sim_backend_venv), >=3.9 (propulate)
- MPI runtime (ParaStationMPI on JUWELS HPC; OpenMPI also supported)
- SLURM workload manager for HPC job submission

**Package Manager:**
- uv - Primary package manager (lockfile: `uv.lock` present)
- pip/setuptools - Build backend for all three subprojects
- hatchling - Build backend for propulate subproject only

## Frameworks

**Core:**
- pyABC - Sequential Monte Carlo ABC inference (`experiments/async_abc/inference/pyabc_wrapper.py`)
- Propulate 1.2.2 - Asynchronous evolutionary optimization / ABC-PMC (`propulate/`)
- sim_backend (nastja) 0.1.0 - Simulation management for sim_backend / realistic workload model (`sim_backend_venv/`)

**Scientific Computing:**
- NumPy >=1.24,<3 - Array operations throughout
- SciPy >=1.10 - Statistical distributions and optimization
- pandas >=1.5 - Tabular data, CSV I/O, analysis
- POT >=0.8 - Optimal transport / Wasserstein distance (`experiments/async_abc/analysis/convergence.py`)
- matplotlib >=3.6 - All plotting (Agg backend, `experiments/async_abc/plotting/`)

**Testing:**
- pytest - Test runner (`experiments/tests/`)
- pytest-cov - Coverage for sim_backend and propulate subprojects

**Build/Dev:**
- setuptools >=68 - Build backend for root project
- setuptools >=61,<81 - Build backend for sim_backend
- hatchling - Build backend for propulate

## Key Dependencies

**Critical (root `pyproject.toml`):**
- `matplotlib>=3.6` - Figure generation for paper
- `numpy>=1.24,<3` - Core numerical operations
- `pandas>=1.5` - Record I/O and analysis DataFrames
- `POT>=0.8` - Wasserstein distance for convergence metrics
- `scipy>=1.10` - Statistical computations

**Optional (root `pyproject.toml`):**
- `pyabc` - pyABC SMC inference (optional dependency group `pyabc`)
- `propulate` - Asynchronous ABC via Propulate (optional dependency group `propulate`)
- `mpi4py` - MPI parallelism (optional dependency group `mpi`)
- `pytest` - Testing (optional dependency group `test`)

**sim_backend critical (`sim_backend_venv/pyproject.toml`):**
- `h5py` - HDF5 data I/O
- `mpi4py` - MPI communication
- `pydantic>=2.11.0` - Config validation
- `scikit-learn` / `scikit-image` - ML and image processing
- `torch>=2.0.0` / `lightning>=2.0.0` - ML training (optional `ml` group)
- `sbi>=0.22.0` / `nflows>=0.14` / `zuko>=1.0.0` - Probabilistic ML (optional `ml-probabilistic`)
- `typer` / `rich` - CLI tooling

**Propulate critical (`propulate/pyproject.toml`):**
- `mpi4py` - Core MPI communication
- `GPy~=1.13.2` - Bayesian optimization
- `deepdiff` - Object comparison
- `sortedcontainers` - Sorted data structures
- `colorlog` - Colored logging

## Configuration

**Environment:**
- No `.env` files detected - configuration is via JSON experiment configs in `experiments/configs/`
- HPC module system used (`module restore sim_backend`, `module load ParaStationMPI`)
- Virtual environment at `sim_backend_venv/.venv` (or HPC path `/p/project1/tissuetwin/herold2/sim_backend/.venv`)

**Build:**
- `pyproject.toml` - Root project build config
- `sim_backend_venv/pyproject.toml` - sim_backend build config
- `propulate/pyproject.toml` - Propulate build config
- `uv.lock` - Dependency lockfile

**Experiment configs (JSON):**
- `experiments/configs/gaussian_mean.json`
- `experiments/configs/gandk.json`
- `experiments/configs/lotka_volterra.json`
- `experiments/configs/realistic_workload.json`
- `experiments/configs/scaling.json`
- `experiments/configs/sbc.json`
- `experiments/configs/sensitivity.json`
- `experiments/configs/straggler.json`
- `experiments/configs/runtime_heterogeneity.json`
- `experiments/configs/ablation.json`
- `experiments/configs/small/` - Reduced-size configs for testing

## Platform Requirements

**Development:**
- Python >=3.10
- uv package manager
- MPI runtime (optional, for parallel execution)

**Production (HPC):**
- JUWELS supercomputer (Forschungszentrum Juelich)
- SLURM job scheduler
- ParaStationMPI
- 48 tasks per node typical configuration
- Account: `tissuetwin`
- Project path: `/p/project1/tissuetwin/herold2/`

---

*Stack analysis: 2026-04-08*
