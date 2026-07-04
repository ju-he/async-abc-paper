# Coding Conventions

**Analysis Date:** 2026-04-08

## Naming Patterns

**Files:**
- Use `snake_case.py` for all Python modules: `rejection_abc.py`, `benchmark_runner.py`, `logging_utils.py`
- Test files: `test_<module>.py` (e.g., `test_sbc.py`, `test_records.py`, `test_inference.py`)
- Private/internal modules: prefix with underscore (`_helpers.py`, `_pyabc_history.py`, `_attempt_trace.py`, `_pyabc_common.py`)

**Functions:**
- Use `snake_case` for all functions: `compute_rank()`, `run_rejection_abc()`, `make_benchmark()`
- Private functions: prefix with underscore: `_summary_stats()`, `_gandk_quantile()`, `_validate()`
- Factory functions: `make_*` pattern (e.g., `make_benchmark()`, `make_seeds()`, `make_arg_parser()`)

**Variables:**
- Use `snake_case`: `sim_count`, `param_names`, `run_start`
- Constants: `UPPER_SNAKE_CASE` (e.g., `REQUIRED_TOP_LEVEL`, `VALID_SCHEDULER_TYPES`, `_QUANTILE_LEVELS`)
- Module-level loggers: `logger = logging.getLogger(__name__)`

**Types/Classes:**
- Use `PascalCase`: `ParticleRecord`, `OutputDir`, `GaussianMean`, `RecordWriter`, `ValidationError`
- Benchmarks are classes: `GaussianMean`, `GandK`, `LotkaVolterra`, `RealisticWorkload`
- Data structures use `@dataclass`: `ParticleRecord` in `experiments/async_abc/io/records.py`

## Code Style

**Formatting:**
- No explicit formatter config (no `.prettierrc`, `pyproject.toml [tool.black]`, `ruff.toml`). Follow existing style.
- 4-space indentation (standard Python)
- Line length appears to be ~100-110 characters max (no enforced limit detected)

**Linting:**
- No explicit linter config detected (no `.flake8`, `ruff.toml`, `[tool.ruff]` in pyproject.toml)
- Code is clean: zero TODO/FIXME/HACK comments anywhere in the codebase

**Type Hints:**
- Use type hints on function signatures consistently: `def run_rejection_abc(simulate_fn: Callable, limits: Dict, ...) -> List[ParticleRecord]:`
- Use `typing` module types: `Dict`, `List`, `Optional`, `Callable`, `Union`, `Any`, `Tuple`
- Modern syntax for newer code: `list[str] | None` alongside `Optional[str]` (mixed)
- Prefer `from __future__ import annotations` in modules using modern syntax

## Import Organization

**Order:**
1. Standard library (`import json`, `import logging`, `import time`, etc.)
2. Third-party packages (`import numpy as np`, `import pandas as pd`, `from scipy import stats`)
3. Local/project imports (`from ..io.records import ParticleRecord`, `from async_abc.benchmarks import make_benchmark`)

**Path Setup:**
- Test files and runner scripts manually insert the `experiments/` directory into `sys.path`:
  ```python
  sys.path.insert(0, str(Path(__file__).parent.parent))
  ```
- The package is structured under `experiments/async_abc/` with relative imports within the package

**Path Aliases:**
- No path aliases configured. Use relative imports (`from ..io.records import ParticleRecord`) within `async_abc` package
- Use absolute imports (`from async_abc.benchmarks import make_benchmark`) from tests and scripts

**Lazy Imports:**
- Heavy optional dependencies (e.g., `ot` for Wasserstein, `propulate`, `pyabc`, `mpi4py`) are imported lazily at function call time or via `__getattr__` in `__init__.py`
- See `experiments/async_abc/analysis/__init__.py` and `experiments/async_abc/plotting/__init__.py` for the `__getattr__` lazy-load pattern

## Error Handling

**Philosophy (from CLAUDE.md):**
- Never hide failures, prefer to crash loudly
- Only handle errors with a clear recovery path, otherwise pass upwards
- No silent handling

**Patterns:**
- Raise specific exceptions with descriptive messages:
  ```python
  raise ValidationError(f"Config missing required top-level key: '{key}'")
  raise ValueError(f"unknown benchmark '{name}'. Available: {sorted(_REGISTRY.keys())}")
  raise KeyError(f"Unknown method: {name!r}. Available: {sorted(...)}")
  raise ImportError(message)  # When optional dependency missing
  raise RuntimeError(message)  # When MPI/runtime setup fails
  ```
- Custom exception class: `ValidationError(ValueError)` in `experiments/async_abc/io/schema.py`
- Guard clauses for optional dependencies return gracefully only when the feature is truly optional:
  ```python
  if weights is None:
      return compute_rank(samples, true_value)  # fallback, not error suppression
  ```
- MPI errors use `TimeoutError` with clear context: `experiments/async_abc/utils/runner.py:568`

## Logging

**Framework:** Python `logging` module

**Setup:** Centralized in `experiments/async_abc/utils/logging_utils.py`
- Format: `"%(name)s %(levelname)s: %(message)s"`
- MPI-aware: `_RootRankFilter` suppresses logs on non-root MPI ranks
- Call `configure_logging()` once at script entry point

**Patterns:**
- Module-level logger: `logger = logging.getLogger(__name__)`
- Use `logger.info()` for progress updates, `logger.warning()` for recoverable issues
- Warnings module integrated: `logging.captureWarnings(True)`

## Comments

**When to Comment:**
- Module-level docstrings on every file explaining purpose (mandatory pattern)
- Class/function docstrings use NumPy-style format with `Parameters`, `Returns`, `Raises` sections
- No inline comments for obvious code; comments only for domain-specific logic

**Docstring Style:**
```python
"""Short one-line summary.

Longer description if needed.

Parameters
----------
param_name:
    Description of the parameter.

Returns
-------
Type
    Description of return value.

Raises
------
ExceptionType
    When this happens.
"""
```

## Function Design

**Size:** Functions are generally focused and under 50-80 lines. Longer functions exist in runner/orchestration code (`experiments/async_abc/utils/runner.py`).

**Parameters:**
- Required params first, then keyword-only params with `*` separator
- Use `**kwargs` sparingly; prefer explicit parameters
- Config dicts are passed as `dict` (not typed dicts)

**Return Values:**
- Return concrete types, not `Any`
- Functions returning records return `List[ParticleRecord]`
- Analysis functions return `pd.DataFrame`

## Module Design

**Exports:**
- `__init__.py` files serve as public API surfaces with explicit `__all__` lists
- Lazy loading via `__getattr__` for heavy-dependency modules (see `experiments/async_abc/analysis/__init__.py`)

**Registry Pattern:**
- Benchmarks: `_REGISTRY` dict in `experiments/async_abc/benchmarks/__init__.py` maps string names to classes
- Methods: `METHOD_REGISTRY` dict in `experiments/async_abc/inference/method_registry.py` maps string names to callables
- Both use factory functions: `make_benchmark()`, `run_method()`

**Config-Driven Design:**
- All experiments are driven by JSON config files in `experiments/configs/`
- Config is loaded, validated, and annotated via `experiments/async_abc/io/config.py`
- Schema validation uses explicit required-key lists in `experiments/async_abc/io/schema.py`

---

*Convention analysis: 2026-04-08*
