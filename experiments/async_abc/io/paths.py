"""Output directory management."""
from pathlib import Path
from typing import Union


class OutputDir:
    """Manages the output directory tree for a single experiment run.

    Structure::

        <base>/<name>/
            plots/
            data/
            logs/

    Parameters
    ----------
    base:
        Parent directory under which the experiment directory is created.
    name:
        Experiment name (used as the directory name).
    """

    def __init__(self, base: Union[str, Path], name: str) -> None:
        self.root = Path(base) / name
        self.plots = self.root / "plots"
        self.data = self.root / "data"
        self.logs = self.root / "logs"

    def ensure(self) -> "OutputDir":
        """Create all subdirectories (idempotent).

        Returns
        -------
        OutputDir
            self, for chaining.
        """
        for d in (self.root, self.plots, self.data, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self
