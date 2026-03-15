"""Cellular Potts model benchmark (stub).

This benchmark requires the ``nastjapy`` / ``cellsInSilico`` package which is
not part of the standard dependencies.  The class is present so that the
method registry can reference it; attempting to call ``simulate`` raises an
``ImportError`` with installation instructions.
"""
from typing import Dict, Tuple


class CellularPotts:
    """Stub benchmark for the Cellular Potts model.

    Raises ``ImportError`` on ``simulate`` — install ``nastjapy`` to enable.

    Parameters
    ----------
    config:
        Benchmark sub-config dict (unused by the stub).
    """

    def __init__(self, config: dict) -> None:
        self.limits: Dict[str, Tuple[float, float]] = {}

    def simulate(self, params: dict, seed: int) -> float:
        """Not implemented — requires nastjapy.

        Raises
        ------
        ImportError
        """
        raise ImportError(
            "The Cellular Potts benchmark requires the 'nastjapy' package. "
            "Please install it and make it importable before using this benchmark."
        )
