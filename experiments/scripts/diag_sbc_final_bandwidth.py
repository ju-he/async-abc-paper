"""Replay a sample of SBC trials and record the final schedule bandwidth.

The SBC campaign stores only posterior samples per trial, not the evaluated
history, so the bandwidth each trial's schedule reached is not on disk. This
script rebuilds trials exactly as ``sbc_runner.py`` does (same seeds, same true
parameters, same observed data, same inference settings) and runs the
asynchronous sampler on them, recording the smallest stamped bandwidth and the
k-th and 2k-th order statistics of the discrepancies.

The question it answers: does a bandwidth floor of ``10^-4 * tol_init`` ever
bind on the g-and-k SBC trials? If the final bandwidth stays above it, those
runs are path-identical to runs with that floor set.

Run under MPI (the sampler is an all-ranks method), e.g.

    mpirun -n 12 .venv/bin/python experiments/scripts/diag_sbc_final_bandwidth.py \
        --config experiments/configs/sbc_gandk_fullhist.json --n-sample 30

The production campaign used 48 workers per trial; the worker count is
recorded in the output so the comparison is explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from async_abc.benchmarks import make_benchmark
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.mpi import get_world_size, is_root_rank
from async_abc.utils.runner import run_method_distributed
from async_abc.utils.seeding import make_seeds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbc_runner import _build_trial_context, _resolve_benchmark_configs  # noqa: E402

METHOD = "async_propulate_abc"
FLOOR_FRACTION = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-sample", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=20261009)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "diagnostics" / "sbc_final_bandwidth.json"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config, test_mode=False, small_mode=False)
    n_trials = int(cfg["sbc"]["n_trials"])
    trial_offset = int(cfg["execution"].get("trial_offset", 0))
    seeds = make_seeds(trial_offset + n_trials, int(cfg["execution"]["base_seed"]))
    method_idx = cfg["methods"].index(METHOD)

    (bench_cfg_entry,) = _resolve_benchmark_configs(cfg)
    inference_cfg = {**cfg["inference"], **bench_cfg_entry.get("inference_overrides", {})}
    k = int(inference_cfg["k"])
    tol_init = float(inference_cfg["tol_init"])
    base_benchmark = make_benchmark(bench_cfg_entry)
    param_names = list(base_benchmark.limits.keys())

    rng = np.random.default_rng(args.sample_seed)
    trials = sorted(int(t) for t in rng.choice(np.arange(trial_offset, trial_offset + n_trials),
                                               size=args.n_sample, replace=False))

    tmp = Path(tempfile.mkdtemp(prefix="diag_sbc_bw_")) if is_root_rank() else Path(tempfile.gettempdir())
    output_dir = OutputDir(str(tmp), "diag_sbc_final_bandwidth").ensure()

    rows = []
    for trial_idx in trials:
        seed = seeds[trial_idx]
        true_params, trial_benchmark = _build_trial_context(
            base_benchmark=base_benchmark,
            benchmark_cfg=bench_cfg_entry,
            param_names=param_names,
            seed=seed,
        )
        records = run_method_distributed(
            METHOD,
            trial_benchmark.simulate,
            trial_benchmark.limits,
            {**inference_cfg, "_checkpoint_tag": f"diag_{trial_idx}"},
            output_dir,
            replicate=trial_idx,
            seed=int(seed + 1000 * (method_idx + 1)),
        )
        if not is_root_rank():
            continue
        stamped = [r.proposal_tolerance for r in records if r.proposal_tolerance is not None]
        if not stamped:
            raise RuntimeError(f"trial {trial_idx}: no archive-phase draws carry a proposal tolerance")
        losses = np.sort(np.asarray([r.loss for r in records], dtype=float))
        eps_final = float(min(stamped))
        rows.append({
            "trial": trial_idx,
            "n": len(records),
            "eps_final": eps_final,
            "eps_final_over_tol_init": eps_final / tol_init,
            "eps_k": float(losses[k - 1]),
            "eps_2k": float(losses[2 * k - 1]),
        })
        print(f"trial {trial_idx}: n={len(records)} eps_final={eps_final:.4g} "
              f"ratio={eps_final / tol_init:.3g} eps_(k)={losses[k - 1]:.4g}", flush=True)

    if is_root_rank():
        ratios = np.asarray([r["eps_final_over_tol_init"] for r in rows])
        summary = {
            "config": args.config,
            "method": METHOD,
            "n_workers": get_world_size(),
            "production_n_workers": int(inference_cfg.get("n_workers", 0)),
            "tol_init": tol_init,
            "k": k,
            "floor_fraction": FLOOR_FRACTION,
            "n_sampled": len(rows),
            "min_ratio": float(ratios.min()),
            "median_ratio": float(np.median(ratios)),
            "max_ratio": float(ratios.max()),
            "n_binding": int((ratios < FLOOR_FRACTION).sum()),
            "trials": rows,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(json.dumps({k_: v for k_, v in summary.items() if k_ != "trials"}, indent=2))


if __name__ == "__main__":
    main()
