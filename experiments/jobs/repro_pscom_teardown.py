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

    # Per-rank marker file on SHARED scratch (REPRO_OUT), so the decisive
    # drain-vs-Free signal survives the job and is readable off the mount — no
    # py-spy and no node access needed. Line-buffered + flushed so the last marker
    # before a hang is durable. Read after a hang: if a combo has no "FREE start"
    # line, the hang is BEFORE _free_propulate_comm (the post-loop intra-island
    # drain); "FREE start" with no "FREE end" means the Free itself hangs.
    markers_dir = Path(out.root) / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)
    _marker_f = open(markers_dir / f"rank_{rank:03d}.log", "a", buffering=1)

    def mark(msg: str) -> None:
        line = f"{time.time():.2f} [rank {rank}/{size}] {msg}"
        print(line, flush=True)
        _marker_f.write(line + "\n")
        _marker_f.flush()

    # Wrap the teardown with per-rank timing. A rank that records "FREE start"
    # but never "FREE end" is wedged in MPI_Comm_free -> pscom_close. No FREE
    # start at all => wedged earlier, in the post-loop drain. Skip-disconnect
    # short-circuits the Free.
    _orig_free = pabc._free_propulate_comm

    def _timed_free(comm):  # noqa: ANN001
        t0 = time.time()
        mark("FREE start")
        _orig_free(comm)
        mark(f"FREE end {time.time() - t0:.2f}s")

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
        mark(f"COMBO {i + 1}/{combos} start (k={k})")
        t0 = time.time()
        records = run_method_distributed(
            "async_propulate_abc", simulate, limits, inference_cfg, out, i, 1234 + i
        )
        n = len(records) if records else 0
        mark(f"COMBO {i + 1}/{combos} done in {time.time() - t0:.1f}s records={n}")

    mark(f"ALL {combos} COMBOS COMPLETED — no teardown hang.")


if __name__ == "__main__":
    main()
