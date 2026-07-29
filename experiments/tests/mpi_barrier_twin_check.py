#!/usr/bin/env python3
"""MPI check that the barrierized twin's barrier actually bites.

Run with::

    mpirun -n 4 python experiments/tests/mpi_barrier_twin_check.py

Not a pytest test: it needs a real multi-rank communicator. It isolates the
*mechanism* (does a collective barrier before each breed convert one slow rank
into idle time for everyone?) from the full inference stack, so a failure here
points at the barrier rather than at the benchmark or Propulate.

Setup: rank 0 is a straggler, sleeping ``SLOW`` per iteration while the others
sleep ``FAST``.

  - Without a barrier, each rank proceeds at its own pace, so total wall-clock
    is set by the straggler alone and the fast ranks finish early:
    ``T_async ~= N_ITERS * SLOW`` for rank 0 and ``N_ITERS * FAST`` for others.
  - With a barrier before every iteration, every rank is held to the slowest,
    so *all* ranks take ``N_ITERS * SLOW`` and the fast ranks accumulate
    ``N_ITERS * (SLOW - FAST)`` of idle time.

The assertion is on the fast ranks: their wall-clock must inflate to the
straggler's pace under the barrier, and must not without it.
"""
from __future__ import annotations

import sys
import time

from mpi4py import MPI

N_ITERS = 20
FAST = 0.005
SLOW = 0.050


def run(comm, barrier: bool) -> float:
    """Return this rank's wall-clock for N_ITERS of breed+simulate."""
    delay = SLOW if comm.rank == 0 else FAST
    comm.Barrier()  # align starts so the measurement is comparable
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        if barrier:
            comm.Barrier()  # exactly what ABCPMCBarrier.__call__ does
        time.sleep(delay)  # stands in for the simulation
    return time.perf_counter() - t0


def main() -> int:
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        print("needs >= 2 ranks; run under mpirun -n 4")
        return 2

    t_async = run(comm, barrier=False)
    t_twin = run(comm, barrier=True)

    async_all = comm.gather(t_async, root=0)
    twin_all = comm.gather(t_twin, root=0)

    if comm.rank != 0:
        return 0

    straggler_pace = N_ITERS * SLOW
    fast_pace = N_ITERS * FAST
    fast_async = async_all[1:]
    fast_twin = twin_all[1:]

    print(f"ranks={comm.size} iters={N_ITERS} fast={FAST}s slow={SLOW}s")
    print(f"  straggler (rank 0): async={async_all[0]:.3f}s twin={twin_all[0]:.3f}s")
    print(f"  fast ranks async   : {[f'{t:.3f}' for t in fast_async]}"
          f"  (expect ~{fast_pace:.3f}s)")
    print(f"  fast ranks twin    : {[f'{t:.3f}' for t in fast_twin]}"
          f"  (expect ~{straggler_pace:.3f}s)")

    ok = True
    # Without a barrier the fast ranks must NOT be dragged to the straggler's pace.
    if not all(t < 0.5 * straggler_pace for t in fast_async):
        print("FAIL: fast ranks were already synchronised without a barrier")
        ok = False
    # With a barrier they must be.
    if not all(t > 0.8 * straggler_pace for t in fast_twin):
        print("FAIL: barrier did not hold fast ranks to the straggler's pace")
        ok = False
    # And the twin must cost strictly more wall-clock on the fast ranks.
    if not all(tw > 2.0 * ta for tw, ta in zip(fast_twin, fast_async)):
        print("FAIL: twin did not inflate fast-rank wall-clock")
        ok = False

    inflation = sum(fast_twin) / sum(fast_async)
    print(f"  fast-rank wall-clock inflation: {inflation:.1f}x")
    print("PASS: the barrier converts one straggler into fleet-wide idle time"
          if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
