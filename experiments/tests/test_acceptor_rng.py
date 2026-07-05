"""II.6 — baseline acceptor RNG: cross-rank decorrelation + reproducibility.

The smooth-kernel acceptor is cloudpickled to every MPI worker and unpickled
anew for EACH work item (pyABC ``MappingSampler.map_function``), so its
rejection stream must live in a per-process module-level cache keyed by
(replicate seed, MPI rank, generation) rather than in instance state. An
instance-held Generator restarts identically on every item and shares the
root copy's initial state across all workers — the original defect: every
worker drew the identical acceptance-threshold stream, and the run was not
reproducible from the config seed.

These tests pin:

* decorrelation — different ranks draw different acceptance streams;
* reproducibility — the same (seed, rank, t) reproduces the same stream;
* stream continuation — re-unpickling the acceptor mid-generation continues
  the stream instead of restarting it;
* ``SeededMappingSampler`` — deterministic once-per-(seed, rank, generation)
  reseed of the worker global RNGs (replacing pyABC's per-item OS-entropy
  reseed) and pickle round-trip of its seeding state.
"""
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pyabc = pytest.importorskip("pyabc")
import cloudpickle

from async_abc.inference import _pyabc_common
from async_abc.inference import pyabc_sampler as ps
from async_abc.inference._pyabc_common import make_acceptor


@pytest.fixture(autouse=True)
def _fresh_streams():
    _pyabc_common._reset_acceptor_streams()
    ps._reset_seeded_global_keys()
    yield
    _pyabc_common._reset_acceptor_streams()
    ps._reset_seeded_global_keys()


def _decisions(acceptor, t: int, n: int = 256) -> list:
    """Drive the acceptor n times at borderline acceptance probability.

    Gaussian kernel at rho=1.0, eps=1.0 gives p_accept = exp(-0.5) ~ 0.607,
    so the boolean decision sequence fingerprints the underlying stream.
    """
    dist = lambda x, x0, tt, par: 1.0
    eps = lambda tt: 1.0
    return [
        bool(acceptor(dist, eps, None, None, t, None).accept) for _ in range(n)
    ]


class TestAcceptorStreams:
    def test_cross_rank_decorrelation(self, monkeypatch):
        acc = make_acceptor("gaussian", rng_seed=123)
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        d0 = _decisions(acc, t=0)
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
        d1 = _decisions(acc, t=0)
        assert d0 != d1

    def test_reproducible_from_seed(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
        first = _decisions(make_acceptor("gaussian", rng_seed=7), t=2)
        _pyabc_common._reset_acceptor_streams()
        again = _decisions(make_acceptor("gaussian", rng_seed=7), t=2)
        assert first == again

    def test_distinct_seeds_distinct_streams(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        assert _decisions(make_acceptor("gaussian", rng_seed=1), t=0) != _decisions(
            make_acceptor("gaussian", rng_seed=2), t=0
        )

    def test_distinct_generations_distinct_streams(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        acc = make_acceptor("gaussian", rng_seed=7)
        assert _decisions(acc, t=0) != _decisions(acc, t=1)

    def test_unpickled_copy_continues_stream(self, monkeypatch):
        """Per-item re-unpickling must CONTINUE the stream, not restart it.

        pyABC unpickles the acceptor once per work item; a restarted stream
        would replay the same decisions for every item (the original bug).
        """
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        acc = make_acceptor("gaussian", rng_seed=99)
        first = _decisions(acc, t=0, n=64)
        clone = cloudpickle.loads(cloudpickle.dumps(acc))
        cont = _decisions(clone, t=0, n=64)
        assert cont != first  # not a restart
        # The clone's decisions are exactly the continuation of the stream:
        _pyabc_common._reset_acceptor_streams()
        replay = _decisions(make_acceptor("gaussian", rng_seed=99), t=0, n=128)
        assert first + cont == replay

    def test_hard_kernel_falls_back_to_uniform_acceptor(self):
        assert isinstance(make_acceptor("hard", rng_seed=1), pyabc.UniformAcceptor)


class _FakeSample:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)


def _accept_after(n_rejects: int):
    """simulate_one stub: draws from global np.random, accepts after n rejects."""
    state = {"calls": 0}

    def simulate_one():
        np.random.random()  # consume the worker-global stream like a proposal
        state["calls"] += 1
        return SimpleNamespace(accepted=state["calls"] > n_rejects)

    return simulate_one


class TestSeededMappingSampler:
    def test_worker_globals_reproducible(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
        ps._seed_worker_globals(11, 0)
        a = np.random.random(8).tolist()
        ps._reset_seeded_global_keys()
        ps._seed_worker_globals(11, 0)
        b = np.random.random(8).tolist()
        assert a == b

    def test_worker_globals_seed_once_per_generation(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
        ps._seed_worker_globals(11, 0)
        a = np.random.random(8).tolist()
        ps._seed_worker_globals(11, 0)  # same key: must NOT reseed
        b = np.random.random(8).tolist()
        assert a != b  # stream continued rather than restarting

    def test_worker_globals_rank_decorrelation(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        ps._seed_worker_globals(11, 3)
        a = np.random.random(8).tolist()
        ps._reset_seeded_global_keys()
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
        ps._seed_worker_globals(11, 3)
        b = np.random.random(8).tolist()
        assert a != b

    def test_pickle_roundtrip_preserves_seeding_state(self):
        s = ps.SeededMappingSampler(map_=map, run_seed=5)
        s._t = 4
        clone = pickle.loads(pickle.dumps(s))
        assert clone._run_seed == 5
        assert clone._t == 4

    def test_map_function_is_reproducible(self, monkeypatch):
        """Two identical runs give identical worker streams and eval counts.

        Upstream MappingSampler reseeds from OS entropy per item, so this
        assertion fails against the base class by construction.
        """
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")

        def run_once():
            ps._reset_seeded_global_keys()
            s = ps.SeededMappingSampler(map_=map, run_seed=5)
            s._t = 0
            s.sample_factory = _FakeSample  # _create_empty_sample() calls it
            payload = s.pickle(_accept_after(3))
            sample, nr_sims = s.map_function(payload, None)
            return nr_sims, len(sample.items), np.random.random(4).tolist()

        assert run_once() == run_once()
