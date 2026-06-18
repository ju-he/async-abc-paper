#!/usr/bin/env python3
"""Minimal reproducer for the ParaStation pscom MPI_Comm_free teardown hang.

Background
----------
The >=48-rank standalone scaling jobs (e.g. scaling_96) are SIGKILLed at a
combo boundary — not OOM (sacct: ~1.7 GB used of 188 GB), not the SLURM wall
clock (died ~5 min into a 2h29m allocation), and with no Python traceback. The
job survives a few Propulate combos then hangs as the next one starts. The
``_free_propulate_comm`` docstring fingers the cause: ``MPI_Comm_free`` can hang
in ParaStation ``pscom_close`` at >=48 ranks. A clean A/B in the small run
showed it is *message-volume* dependent: CPU scaling (~266k records/combo) hangs
at w=48/96, while CPM scaling (~24k records/combo) survives at the *same* ranks.

What this script does
---------------------
Runs ``async_propulate_abc`` ``REPRO_COMBOS`` times **in one process** — i.e.
one ``MPI.COMM_WORLD.Dup()`` + ``MPI_Comm_free`` per combo, exactly the pattern
the standalone scaling job uses — with a trivial, maximally fast simulator so
message/isend volume piles up quickly. It wraps ``_free_propulate_comm`` with
per-rank timing, so a hang is unmistakable in the log:

    [rank 29/96] FREE start          <-- this rank entered MPI_Comm_free ...
    (no matching "FREE end")         <-- ... and never returned == pscom_close wedged

Use it to (a) confirm the hang reproduces and how fast, (b) confirm *where* it
hangs by attaching ``py-spy``/``gdb`` to a rank stuck after "FREE start", and
(c) A/B candidate fixes (``PROPULATE_SKIP_DISCONNECT=1``, an OpenMPI-linked
mpi4py, ``PSP_*`` pscom tuning). See repro_pscom_teardown.sh.

Tunables (environment variables)
--------------------------------
  REPRO_COMBOS   number of sequential Dup/Free combos   (default 6)
  REPRO_WALL_S   wall-time budget per combo, seconds     (default 20)
  REPRO_K        archive size k per combo                (default 1000)
  REPRO_OUT      scratch output dir                      (default /tmp/repro_pscom)

Run with one rank per worker, e.g. ``srun -n 48 python repro_pscom_teardown.py``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_DIR))

from async_abc.io.paths import OutputDir  # noqa: E402
from async_abc.utils.logging_utils import configure_logging  # noqa: E402
from async_abc.utils.mpi import is_root_rank  # noqa: E402
from async_abc.utils.runner import run_method_distributed  # noqa: E402
import async_abc.inference.propulate_abc as pabc  # noqa: E402


def _rank_size() -> tuple[int, int]:
    try:
        from mpi4py import MPI
    except Exception:
        return 0, 1
    return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()


def main() -> None:
    configure_logging()
    rank, size = _rank_size()

    combos = int(os.environ.get("REPRO_COMBOS", "6"))
    wall_s = float(os.environ.get("REPRO_WALL_S", "20"))
    k = int(os.environ.get("REPRO_K", "1000"))
    out = OutputDir(os.environ.get("REPRO_OUT", "/tmp/repro_pscom"), "repro").ensure()

    # Wrap the teardown with per-rank timing. A rank that prints "FREE start"
    # but never "FREE end" is wedged in MPI_Comm_free -> pscom_close: that is the
    # rank/PID to attach py-spy to. Skip-disconnect short-circuits before this.
    _orig_free = pabc._free_propulate_comm

    def _timed_free(comm):  # noqa: ANN001
        t0 = time.time()
        print(f"[rank {rank}/{size}] FREE start", flush=True)
        _orig_free(comm)
        print(f"[rank {rank}/{size}] FREE end {time.time() - t0:.2f}s", flush=True)

    pabc._free_propulate_comm = _timed_free

    # Trivial, maximally fast simulator -> highest message volume per second,
    # which is the driver of the pscom teardown hang.
    limits = {"mu": (-5.0, 5.0)}

    def simulate(params, seed):  # noqa: ANN001, ARG001
        return abs(float(params["mu"]))

    inference_cfg = {
        "max_simulations": 10_000_000,  # effectively unbounded; wall-time bounds the combo
        "k": k,
        "tol_init": 10.0,
        "scheduler_type": "acceptance_rate",
        "perturbation_scale": 0.8,
        "kernel": "gaussian",
        "amis_snapshots": 20,
        "max_wall_time_s": wall_s,
        "n_workers": size,
        "progress_log_interval_s": 10.0,
    }

    if is_root_rank():
        print(
            f"[repro] world_size={size} combos={combos} wall_s={wall_s} k={k} "
            f"skip_disconnect={os.environ.get('PROPULATE_SKIP_DISCONNECT', '<unset>')}",
            flush=True,
        )

    for i in range(combos):
        if is_root_rank():
            print(f"[repro] === COMBO {i + 1}/{combos} start ===", flush=True)
        t0 = time.time()
        records = run_method_distributed(
            "async_propulate_abc", simulate, limits, inference_cfg, out, i, 1234 + i
        )
        if is_root_rank():
            n = len(records) if records else 0
            print(
                f"[repro] === COMBO {i + 1}/{combos} done in {time.time() - t0:.1f}s, "
                f"records={n} ===",
                flush=True,
            )

    if is_root_rank():
        print(f"[repro] ALL {combos} COMBOS COMPLETED — no teardown hang.", flush=True)


if __name__ == "__main__":
    main()
