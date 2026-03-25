"""Reproducible seeding utilities."""
import hashlib
import json
import random
from typing import Dict, List, Tuple

import numpy as np


def stable_seed(*parts: object) -> int:
    """Return a deterministic 31-bit seed derived from structured inputs."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def canonical_param_key(
    params: Dict[str, float],
    *,
    decimals: int = 10,
) -> Tuple[Tuple[str, float], ...]:
    """Return a stable rounded parameter key suitable for hashing and joins."""
    return tuple(
        sorted((str(key), round(float(value), decimals)) for key, value in params.items())
    )


def canonical_param_key_json(
    params: Dict[str, float],
    *,
    decimals: int = 10,
) -> str:
    """Return a JSON-serialized canonical parameter key."""
    key = canonical_param_key(params, decimals=decimals)
    return json.dumps(key, separators=(",", ":"), ensure_ascii=True)


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
