"""Lotka-Volterra stochastic population dynamics benchmark.

Reactions (Gillespie SSA):
    1. X → X+1   propensity = theta1 * X     (prey birth)
    2. Y → Y+1   propensity = theta2 * X * Y (predator birth via predation)
    3. X → X-1   propensity = theta3 * X * Y (prey death via predation)
    4. Y → Y-1   propensity = theta4 * Y     (predator death)

Parameters: theta = (theta1, theta2, theta3, theta4)
True parameters: (0.5, 0.025, 0.025, 0.5) produce oscillations.

Summary statistics (6 values):
    mean(X), std(X), mean(Y), std(Y),
    log(1 + var(X) / (mean(X)+1e-8)),
    log(1 + var(Y) / (mean(Y)+1e-8))

Distance: Euclidean between simulated and observed summary statistics.
Extinction: if either population reaches 0 before T_max, return a large
fallback loss (EXTINCTION_LOSS).
"""
from typing import Dict, List, Tuple

import numpy as np

EXTINCTION_LOSS = 1e6


def _gillespie(
    x0: int,
    y0: int,
    theta: Tuple[float, float, float, float],
    T_max: float,
    rng: np.random.Generator,
    max_steps: int = 100_000,
) -> Tuple[List[float], List[int], List[int]]:
    """Run Gillespie SSA for the 4-reaction LV system.

    Returns
    -------
    times : list of float
    xs    : list of int  (prey counts)
    ys    : list of int  (predator counts)
    """
    theta1, theta2, theta3, theta4 = theta
    t = 0.0
    x, y = x0, y0
    times = [t]
    xs = [x]
    ys = [y]

    for _ in range(max_steps):
        a1 = theta1 * x
        a2 = theta2 * x * y
        a3 = theta3 * x * y
        a4 = theta4 * y
        a_total = a1 + a2 + a3 + a4

        if a_total == 0.0:
            break  # absorbing state

        # Time to next event
        dt = rng.exponential(1.0 / a_total)
        t += dt
        if t > T_max:
            break

        # Which reaction fires?
        u = rng.uniform(0.0, a_total)
        if u < a1:
            x += 1
        elif u < a1 + a2:
            y += 1
        elif u < a1 + a2 + a3:
            x -= 1
        else:
            y -= 1

        # Safety: populations can't go negative (shouldn't happen with correct propensities)
        x = max(x, 0)
        y = max(y, 0)

        times.append(t)
        xs.append(x)
        ys.append(y)

        if x == 0 or y == 0:
            break

    return times, xs, ys


def _summary_stats(xs: List[int], ys: List[int]) -> np.ndarray:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.array([
        mean_x,
        np.std(x),
        mean_y,
        np.std(y),
        np.log1p(np.var(x) / (mean_x + 1e-8)),
        np.log1p(np.var(y) / (mean_y + 1e-8)),
    ])


class LotkaVolterra:
    """ABC benchmark: infer Lotka-Volterra rate parameters.

    Parameters
    ----------
    config:
        Benchmark sub-config dict.  Recognised keys:

        - ``observed_data_seed`` (int, default 42)
        - ``T_max`` (float, default 30.0)
        - ``x0`` (int, default 50) — initial prey count
        - ``y0`` (int, default 100) — initial predator count
        - ``true_theta1/2/3/4`` (float defaults: 0.5, 0.025, 0.025, 0.5)
    """

    def __init__(self, config: dict) -> None:
        self.T_max = float(config.get("T_max", 30.0))
        self.x0 = int(config.get("x0", 50))
        self.y0 = int(config.get("y0", 100))
        self.normalize_stats = bool(config.get("normalize_stats", True))

        true_theta = (
            float(config.get("true_theta1", 0.5)),
            float(config.get("true_theta2", 0.025)),
            float(config.get("true_theta3", 0.025)),
            float(config.get("true_theta4", 0.5)),
        )

        obs_rng = np.random.default_rng(config.get("observed_data_seed", 42))
        base_seed = config.get("observed_data_seed", 42)
        max_retries = int(config.get("max_extinction_retries", 100))
        times, xs, ys = _gillespie(self.x0, self.y0, true_theta, self.T_max, obs_rng)

        retry = 0
        while (xs[-1] == 0 or ys[-1] == 0) and retry < max_retries:
            retry += 1
            retry_rng = np.random.default_rng(base_seed + retry)
            times, xs, ys = _gillespie(self.x0, self.y0, true_theta, self.T_max, retry_rng)

        if xs[-1] == 0 or ys[-1] == 0:
            raise RuntimeError(
                f"Observed trajectory went extinct after {max_retries} retries. "
                f"Consider adjusting true parameters or increasing max_extinction_retries."
            )

        self.observed_stats = _summary_stats(xs, ys)

        self.limits: Dict[str, Tuple[float, float]] = {
            "theta1": (0.05, 2.0),
            "theta2": (0.001, 0.2),
            "theta3": (0.001, 0.2),
            "theta4": (0.05, 2.0),
        }

    def simulate(self, params: dict, seed: int) -> float:
        """Simulate and return the ABC distance.

        Parameters
        ----------
        params:
            Dict with keys ``theta1``, ``theta2``, ``theta3``, ``theta4``.
        seed:
            Integer RNG seed.

        Returns
        -------
        float
            Euclidean distance between simulated and observed summary stats,
            or ``EXTINCTION_LOSS`` if either population goes extinct before
            ``T_max``.
        """
        theta = (
            float(params["theta1"]),
            float(params["theta2"]),
            float(params["theta3"]),
            float(params["theta4"]),
        )
        rng = np.random.default_rng(seed)
        times, xs, ys = _gillespie(self.x0, self.y0, theta, self.T_max, rng)

        if xs[-1] == 0 or ys[-1] == 0:
            return float(EXTINCTION_LOSS)

        sim_stats = _summary_stats(xs, ys)
        if self.normalize_stats:
            eps = 1e-8
            diff = (sim_stats - self.observed_stats) / (np.abs(self.observed_stats) + eps)
            return float(np.linalg.norm(diff))
        return float(np.linalg.norm(sim_stats - self.observed_stats))
