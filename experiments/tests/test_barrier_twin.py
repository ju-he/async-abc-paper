"""Barrierized twin (external review, concern 4)."""
import pytest

from async_abc.inference.abcpmc_barrier import (
    assert_barrier_safe,
    make_barrier_propagator_class,
)


class _FakeBase:
    """Stand-in for the frozen ABCPMC propagator."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = 0

    def __call__(self, inds):
        self.calls += 1
        return f"child-{self.calls}"


class _FakeComm:
    def __init__(self):
        self.barriers = 0

    def Barrier(self):
        self.barriers += 1


def test_wall_clock_budget_is_refused():
    """A collective barrier deadlocks under first-rank-hit termination."""
    with pytest.raises(ValueError, match="simulation-limited"):
        assert_barrier_safe({"max_wall_time_s": 300.0})


def test_simulation_limited_budget_is_allowed():
    assert_barrier_safe({"max_simulations": 20000,
                         "propulate_budget_mode": "total_simulations"}) is None


def test_barrier_runs_once_per_breed():
    cls = make_barrier_propagator_class(_FakeBase)
    prop = cls(barrier=True)
    comm = _FakeComm()
    prop.attach_comm(comm)
    for _ in range(5):
        prop([])
    assert comm.barriers == 5
    assert prop.barrier_calls == 5


def test_disabled_barrier_never_synchronises():
    """barrier=False must leave the inherited path untouched."""
    cls = make_barrier_propagator_class(_FakeBase)
    prop = cls(barrier=False)
    comm = _FakeComm()
    prop.attach_comm(comm)
    for _ in range(5):
        prop([])
    assert comm.barriers == 0
    assert prop.barrier_calls == 0


def test_missing_comm_fails_loudly():
    """Silently skipping the barrier would make the twin a copy of async."""
    cls = make_barrier_propagator_class(_FakeBase)
    prop = cls(barrier=True)
    with pytest.raises(RuntimeError, match="never given a communicator"):
        prop([])


def test_twin_delegates_breeding_to_the_base_propagator():
    """Proposals/weights/archive must be inherited unchanged."""
    cls = make_barrier_propagator_class(_FakeBase)
    prop = cls(barrier=True)
    prop.attach_comm(_FakeComm())
    assert prop([]) == "child-1"
    assert prop([]) == "child-2"
