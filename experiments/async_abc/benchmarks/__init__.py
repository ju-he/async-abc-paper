"""Benchmark model registry."""
from .gaussian_mean import GaussianMean
from .gandk import GandK
from .lotka_volterra import LotkaVolterra
from .cellular_potts import CellularPotts

_REGISTRY = {
    "gaussian_mean": GaussianMean,
    "gandk": GandK,
    "lotka_volterra": LotkaVolterra,
    "cellular_potts": CellularPotts,
}


def make_benchmark(benchmark_config: dict):
    """Instantiate a benchmark model from a benchmark sub-config dict.

    Parameters
    ----------
    benchmark_config:
        Dict with at least a ``"name"`` key.  Passed directly to the
        benchmark constructor.

    Returns
    -------
    Benchmark instance with ``simulate(params, seed) -> float`` and
    ``limits`` attributes.

    Raises
    ------
    ValueError
        If ``name`` is not a registered benchmark.
    """
    name = benchmark_config.get("name", "")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown benchmark '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return cls(benchmark_config)
