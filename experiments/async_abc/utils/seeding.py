"""Reproducible seeding utilities."""
import random
from typing import List

import numpy as np


def make_seeds(n: int, base: int) -> List[int]:
    """Generate *n* deterministic, unique integer seeds derived from *base*.

    Uses a seeded ``random.Random`` instance so results are portable across
    platforms and independent of any global RNG state.

    Parameters
    ----------
    n:
        Number of seeds to generate.
    base:
        Base seed that fully determines the output list.

    Returns
    -------
    List[int]
        Length-*n* list of non-negative integers, unique within the list.
    """
    rng = random.Random(base)
    seeds: List[int] = []
    seen: set = set()
    while len(seeds) < n:
        s = rng.randint(0, 2**31 - 1)
        if s not in seen:
            seeds.append(s)
            seen.add(s)
    return seeds


def seed_everything(seed: int) -> None:
    """Set the global RNG seeds for ``random`` and ``numpy``.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
