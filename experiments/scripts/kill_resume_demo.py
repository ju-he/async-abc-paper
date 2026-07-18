#!/usr/bin/env python3
"""Kill-and-resume experiment (external review concern 7).

Demonstrates empirically what the crash-recovery claim does and does not mean:

  * A clean uninterrupted run and an interrupted+resumed run (same config, same
    seed) both recover the analytic posterior — resume produces a **valid**
    posterior trajectory over the actual evaluated history.
  * They are **not** bit-identical: the candidate RNG is re-seeded from
    (replicate seed, rank) on restart rather than checkpointed, and MPI arrival
    order is nondeterministic, so post-restart draws follow a fresh-but-valid
    stream (this is exactly the honest scoping in the paper's reproducibility
    paragraph — "reproducible up to MPI arrival order", not deterministic replay).

Mechanics: Propulate checkpoints the evaluated population to
``island_<i>_ckpt.pickle`` each generation; on restart Propulator reloads it and
resumes from ``max(generation)+1``. This driver launches the gaussian runner
under ``mpirun``, SIGKILLs the whole process group partway through inference
(before any raw_results is written, so no duplicate records), verifies a
checkpoint exists, relaunches the identical command (which resumes), and then
compares the reported posterior of the clean vs resumed run — both reconstructed
directly from ``raw_results`` so the driver is independent of the runner's
summary artifacts.

Usage:
    python kill_resume_demo.py --workdir /tmp/killresume [--ranks 4] [--kill-after 18]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from async_abc.analysis import final_state_results  # noqa: E402
from async_abc.io.records import load_records  # noqa: E402
from async_abc.benchmarks.gaussian_mean import GaussianMean  # noqa: E402

_CONFIG = _SCRIPT_DIR.parent / "configs" / "kill_resume_gaussian.json"
_RUNNER = _SCRIPT_DIR / "gaussian_mean_runner.py"


def _launch(output_dir: Path, ranks: int) -> subprocess.Popen:
    cmd = [
        "mpirun", "-n", str(ranks),
        sys.executable, str(_RUNNER),
        "--config", str(_CONFIG),
        "--output-dir", str(output_dir),
    ]
    # start_new_session=True -> the child leads its own process group, so we can
    # SIGKILL the whole mpirun+ranks tree with killpg.
    return subprocess.Popen(cmd, start_new_session=True)


def _checkpoint_files(output_dir: Path) -> list[Path]:
    logs = output_dir / "kill_resume_gaussian" / "logs"
    if not logs.exists():
        return []
    return sorted(logs.glob("propulate_rep*/island_*_ckpt.pickle")) + \
        sorted(logs.glob("propulate_rep*/island_*_ckpt.bkp"))


def _run_clean(output_dir: Path, ranks: int) -> None:
    print(f"[clean] launching {ranks}-rank run -> {output_dir}", flush=True)
    p = _launch(output_dir, ranks)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"clean run exited with code {rc}")
    print("[clean] done", flush=True)


def _run_interrupted(output_dir: Path, ranks: int, kill_after: float) -> dict:
    """Launch, SIGKILL the process group after kill_after s, then relaunch to resume."""
    info: dict = {}
    print(f"[kill] launching {ranks}-rank run -> {output_dir}", flush=True)
    p = _launch(output_dir, ranks)

    # Wait until a checkpoint has been dumped AND kill_after has elapsed, so the
    # resume has something to load. Poll for the checkpoint file.
    deadline = time.time() + max(kill_after, 1.0)
    ckpt_seen = False
    while time.time() < deadline or not ckpt_seen:
        if p.poll() is not None:
            raise RuntimeError(
                f"interrupted run finished (rc={p.returncode}) before we could kill it — "
                "increase max_simulations or lower --kill-after."
            )
        if not ckpt_seen and _checkpoint_files(output_dir):
            ckpt_seen = True
        if time.time() >= deadline and ckpt_seen:
            break
        time.sleep(0.5)

    ckpts = _checkpoint_files(output_dir)
    info["checkpoints_before_kill"] = [str(c) for c in ckpts]
    info["n_checkpoints_before_kill"] = len(ckpts)
    raw_before = output_dir / "kill_resume_gaussian" / "data" / "raw_results.csv"
    info["raw_results_existed_before_kill"] = raw_before.exists()

    print(f"[kill] SIGKILL after {time.time() - (deadline - max(kill_after,1.0)):.1f}s "
          f"({len(ckpts)} checkpoint(s) present, raw_results={'yes' if raw_before.exists() else 'no'})",
          flush=True)
    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    p.wait()
    time.sleep(1.0)

    # Relaunch identical command -> Propulator reloads the checkpoint and resumes.
    print("[resume] relaunching identical command (resumes from checkpoint)", flush=True)
    p2 = _launch(output_dir, ranks)
    rc = p2.wait()
    if rc != 0:
        raise RuntimeError(f"resumed run exited with code {rc}")
    print("[resume] done", flush=True)
    return info


def _reported_posterior(output_dir: Path, archive_size: int) -> dict:
    raw = output_dir / "kill_resume_gaussian" / "data" / "raw_results.csv"
    records = load_records(raw)
    n_records = len(records)
    samples, weights = [], []
    for result in final_state_results(records, archive_size=archive_size):
        for r in result.records:
            if "mu" in r.params:
                samples.append(float(r.params["mu"]))
                pw = getattr(r, "posterior_weight", None)
                weights.append(float(pw) if pw is not None else (float(r.weight) if r.weight is not None else 1.0))
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(samples)
    w = weights / weights.sum()
    return {
        "n_records": int(n_records),
        "archive_size": int(samples.size),
        "unweighted_mean": float(samples.mean()) if samples.size else float("nan"),
        "weighted_mean": float(np.sum(w * samples)) if samples.size else float("nan"),
        "weighted_var": float(np.sum(w * (samples - np.sum(w * samples)) ** 2)) if samples.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default="/tmp/killresume", help="Scratch dir (wiped).")
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--kill-after", type=float, default=18.0)
    args = ap.parse_args()

    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    clean_dir = work / "clean"
    kill_dir = work / "interrupted"
    clean_dir.mkdir(parents=True)
    kill_dir.mkdir(parents=True)

    cfg = json.loads(_CONFIG.read_text())
    archive_size = int(cfg["inference"]["k"])
    bm = GaussianMean(cfg["benchmark"])
    analytic_mean = float(bm.analytic_posterior_mean())
    analytic_std = float(bm.sigma_obs) / float(np.sqrt(bm.n_obs))

    _run_clean(clean_dir, args.ranks)
    interrupt_info = _run_interrupted(kill_dir, args.ranks, args.kill_after)

    clean = _reported_posterior(clean_dir, archive_size)
    resumed = _reported_posterior(kill_dir, archive_size)

    report = {
        "config": str(_CONFIG),
        "ranks": args.ranks,
        "analytic_posterior_mean": analytic_mean,
        "analytic_posterior_std": analytic_std,
        "clean": clean,
        "resumed": resumed,
        "interrupt": interrupt_info,
        "clean_mean_abs_error": abs(clean["weighted_mean"] - analytic_mean),
        "resumed_mean_abs_error": abs(resumed["weighted_mean"] - analytic_mean),
        "clean_vs_resumed_weighted_mean_diff": abs(clean["weighted_mean"] - resumed["weighted_mean"]),
    }
    out = work / "kill_resume_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n===== KILL / RESUME REPORT =====")
    print(json.dumps(report, indent=2))
    print(f"\nreport written to {out}")

    # Sanity assertions (the experiment's claims).
    assert interrupt_info["n_checkpoints_before_kill"] >= 1, "no checkpoint was dumped before the kill"
    assert not interrupt_info["raw_results_existed_before_kill"], \
        "raw_results existed before kill -> the run had already finished (kill too late)"
    assert resumed["archive_size"] > 0 and resumed["n_records"] > 0, "resumed run produced no history"
    assert report["resumed_mean_abs_error"] < 0.2, \
        f"resumed posterior did not recover the analytic mean (err={report['resumed_mean_abs_error']:.3f})"
    print("\nAll kill/resume assertions passed: resume produced a valid posterior "
          "that recovers the analytic target.")


if __name__ == "__main__":
    main()
