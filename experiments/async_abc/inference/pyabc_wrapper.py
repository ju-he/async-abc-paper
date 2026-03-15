"""pyABC inference wrapper (optional dependency).

If ``pyabc`` is not installed, :func:`run_pyabc_smc` raises ``ImportError``
with installation instructions.  The function is still importable so that the
method registry can reference it unconditionally.
"""
from typing import Callable, Dict, List

from ..io.paths import OutputDir
from ..io.records import ParticleRecord


def run_pyabc_smc(
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
) -> List[ParticleRecord]:
    """Run synchronous ABC-SMC via pyABC.

    Parameters
    ----------
    simulate_fn:
        Callable ``(params: dict, seed: int) -> float``.
    limits:
        Propulate-style limits dict ``{name: (lo, hi)}``.
    inference_cfg:
        ``config["inference"]`` sub-dict.
    output_dir:
        Output directory (used for pyABC database path).
    replicate:
        Replicate index stored in each record.
    seed:
        Base RNG seed.

    Returns
    -------
    List[ParticleRecord]

    Raises
    ------
    ImportError
        If pyabc is not installed.
    """
    try:
        import pyabc
    except ImportError as exc:
        raise ImportError(
            "The pyabc method requires the 'pyabc' package. "
            "Install it with: pip install pyabc"
        ) from exc

    import numpy as np
    import time

    max_sims = inference_cfg["max_simulations"]
    k = inference_cfg.get("k", 100)
    tol_init = inference_cfg.get("tol_init", 10.0)
    n_procs = inference_cfg.get("n_workers", 1)

    # Build pyABC prior from limits
    prior = pyabc.Distribution(
        **{
            name: pyabc.RV("uniform", lo, hi - lo)
            for name, (lo, hi) in limits.items()
        }
    )

    # pyABC simulator: returns a dict with a single "distance" key
    rng_counter = [0]

    def pyabc_model(params):
        sim_seed = abs(hash((seed, rng_counter[0]))) % (2**31)
        rng_counter[0] += 1
        loss = simulate_fn(dict(params), seed=sim_seed)
        return {"distance": loss}

    def pyabc_distance(x, x0):
        return x["distance"]

    db_path = f"sqlite:///{output_dir.data / f'pyabc_rep{replicate}.db'}"

    abc = pyabc.ABCSMC(
        models=pyabc_model,
        parameter_priors=prior,
        distance_function=pyabc_distance,
        population_size=k,
        transitions=pyabc.MultivariateNormalTransition(),
        eps=pyabc.QuantileEpsilon(alpha=0.5),
    )
    abc.new(db_path, {"distance": 0.0})

    run_start = time.time()
    history = abc.run(
        minimum_epsilon=tol_init * 0.01,
        max_total_nr_simulations=max_sims,
    )

    # Collect records from the pyABC history
    records: List[ParticleRecord] = []
    step = 0
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        eps_t = history.get_all_populations()["epsilon"].iloc[t]
        for i, row in df.iterrows():
            step += 1
            params = {col: float(row[col]) for col in limits}
            records.append(ParticleRecord(
                method="pyabc_smc",
                replicate=replicate,
                seed=seed,
                step=step,
                params=params,
                loss=float(eps_t),   # distance at this generation
                weight=float(w.iloc[i]) if hasattr(w, "iloc") else None,
                tolerance=float(eps_t),
                wall_time=time.time() - run_start,
            ))

    return records
