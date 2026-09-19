# Exception Handling
Never hide failures, prefer to crash loudly, only handle errors with clear recovery path, otherwise pass upwards. No silent handling. 

# Testing
Use the venv at .venv (repo root). Full suite: 744 passed, 10 skipped.

To rebuild it (propulate and nastjapy come from the sibling checkouts, editable):

    uv venv .venv --python 3.11
    uv pip install --python .venv/bin/python -e ../propulate
    uv pip install --python .venv/bin/python -e ".[test,mpi]"
    uv pip install --python .venv/bin/python -e ../nastjapy

propulate must be installed WITH its dependencies: its `__init__` chain reaches
`surrogate.py`, which imports GPy, so every propulate import fails without it.
`../nastjapy` (package `nastja`) is needed by the Cellular Potts benchmark tests.
`pyproject.toml`/`uv.lock` have no `[tool.uv.sources]` entry for propulate, so
`uv sync` would pull it from PyPI instead of the local checkout — use the
commands above.

The 10 skips are not an environment gap: 8 are the mpirun-dependent tests in
test_mpi_hardening.py, which skip because `from mpi4py import MPI` initializes
MPI in the pytest process and mpirun then refuses to launch. They pass when that
file is run on its own.

# Bug fixing
Keep track of previous bug fixes by updating .plans/bug-fixes/previous-fixes.md. Consult it when needed.

# Ressources
Plans are meant to be stored in .plans. Further ressources on the intentions of the experiments can be found in .plans/ressources if needed.