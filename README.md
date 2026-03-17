# async-abc-paper
Paper exploring the implementation of an asynchronous ABC within propulate

## Installation

The Python package lives under `experiments/async_abc` and can now be installed
from the repository root with:

```bash
pip install .
```

Core Python dependencies installed through `pip`:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `POT` (`import ot`)

Optional/runtime-specific dependencies:

- `propulate`: required for the `async_propulate_abc` method. This is expected
  to come from a Git checkout or an already-prepared virtual environment.
- `pyabc`: required for `pyabc_smc` and `abc_smc_baseline`.
- `mpi4py`: required only when `parallel_backend="mpi"`.
- `nastjapy` plus a compiled `nastja`: required for the `cellular_potts`
  benchmark. The package will use the active environment first and fall back to
  the repo-local `nastjapy_copy/` checkout if present.
