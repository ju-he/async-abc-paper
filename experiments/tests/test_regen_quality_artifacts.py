"""Tests for the post-hoc quality-artifact repair pass.

The pass exists because the kernel fix (f62f143) corrected how the archive is
reconstructed, but every artifact written before it is still on disk with the
hard-kernel values. ``regen_experiment`` rebuilds the quality *curves*;
``regen_summaries`` rebuilds the copy of the curve's final value that the
summary CSVs carry. These tests pin the summary path, which has to key on two
different column namings and must not touch the weighted metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import conftest as test_helpers


@pytest.fixture
def regen_module():
    return test_helpers.import_runner_module("regen_quality_artifacts.py")


def _write_experiment(root, name, *, summary_name, method_col, kernel="gaussian"):
    """Lay out a minimal experiment directory the pass can operate on."""
    data = root / name / "data"
    data.mkdir(parents=True)
    (data / "metadata.json").write_text(json.dumps({
        "config": {
            "inference": {"kernel": kernel, "k": 4},
            "benchmark": {"name": "gaussian_mean", "true_mu": 0.0},
        }
    }))
    # Two replicates of one method, losses far apart so the archive top-k is
    # well defined and the reconstructed final value is deterministic.
    raw = []
    for replicate in (0, 1):
        for step, (loss, mu, wall) in enumerate(
            [(3.0, 3.0, 0.1), (2.0, 2.0, 0.2), (1.0, 1.0, 0.3), (0.5, 0.5, 0.4)], start=1
        ):
            raw.append({
                "method": "async_propulate_abc__tagged",
                "replicate": replicate,
                "step": step,
                "param_mu": mu + replicate,
                "loss": loss,
                "tolerance": 5.0 - 0.5 * step,
                "wall_time": wall,
                "sim_start_time": wall - 0.05,
                "sim_end_time": wall,
                "generation": 0,
                "record_kind": "simulation_attempt",
                "time_semantics": "event_end",
                "attempt_count": 1,
            })
    pd.DataFrame(raw).to_csv(data / "raw_results.csv", index=False)

    summary = pd.DataFrame([
        {
            method_col: "async_propulate_abc__tagged",
            "replicate": replicate,
            "k": 4,
            # Deliberately wrong sentinel: the pass must overwrite it.
            "final_quality_wasserstein": -1.0,
            "final_quality_wasserstein_weighted": 0.123,
            "final_quality_wasserstein_analytic": 0.456,
        }
        for replicate in (0, 1)
    ])
    summary.to_csv(data / summary_name, index=False)
    return data / summary_name


@pytest.mark.parametrize(
    "summary_name,method_col",
    [
        # straggler / heterogeneity shape
        ("throughput_vs_slowdown_summary.csv", "method"),
        # scaling shape: the tagged run is called method_variant there
        ("throughput_summary_w48_k100.csv", "method_variant"),
    ],
)
def test_regen_summaries_rewrites_unweighted_metric(
    regen_module, tmp_path, summary_name, method_col, capsys
):
    path = _write_experiment(tmp_path, "exp", summary_name=summary_name, method_col=method_col)

    regen_module.regen_summaries(tmp_path, "exp")

    table = pd.read_csv(path)
    assert len(table) == 2
    # The sentinel is gone and every row got a real, finite distance.
    assert (table["final_quality_wasserstein"] != -1.0).all()
    assert table["final_quality_wasserstein"].notna().all()
    assert (table["final_quality_wasserstein"] >= 0).all()
    # The weighted metrics do not go through the kernel-dependent archive
    # reconstruction, so the pass must leave them exactly as they were.
    assert table["final_quality_wasserstein_weighted"].tolist() == [0.123, 0.123]
    assert table["final_quality_wasserstein_analytic"].tolist() == [0.456, 0.456]
    # Replicates differ (mu is offset by the replicate index), so the pass keyed
    # on the replicate rather than reusing one value everywhere.
    assert table["final_quality_wasserstein"].nunique() == 2


def test_regen_summaries_preserves_the_original_once(regen_module, tmp_path):
    path = _write_experiment(
        tmp_path, "exp", summary_name="throughput_vs_slowdown_summary.csv", method_col="method"
    )

    regen_module.regen_summaries(tmp_path, "exp")
    backup = path.with_suffix(".prekernelfix.csv")
    assert backup.exists()
    assert pd.read_csv(backup)["final_quality_wasserstein"].tolist() == [-1.0, -1.0]

    first = pd.read_csv(path)["final_quality_wasserstein"].tolist()
    # A second pass must not overwrite the backup with the already-corrected
    # values -- that would destroy the only copy of the pre-fix artifact.
    regen_module.regen_summaries(tmp_path, "exp")
    assert pd.read_csv(backup)["final_quality_wasserstein"].tolist() == [-1.0, -1.0]
    assert pd.read_csv(path)["final_quality_wasserstein"].tolist() == first


def test_regen_summaries_skips_when_no_such_column(regen_module, tmp_path, capsys):
    data = tmp_path / "exp" / "data"
    data.mkdir(parents=True)
    (data / "metadata.json").write_text(json.dumps({
        "config": {"inference": {"kernel": "gaussian", "k": 4},
                   "benchmark": {"name": "gaussian_mean", "true_mu": 0.0}}
    }))
    pd.DataFrame([{"method": "m", "replicate": 0}]).to_csv(data / "raw_results.csv", index=False)
    pd.DataFrame([{"method": "m", "replicate": 0, "throughput_sims_per_s": 1.0}]).to_csv(
        data / "some_summary.csv", index=False
    )

    regen_module.regen_summaries(tmp_path, "exp")

    assert "no final_quality_wasserstein column" in capsys.readouterr().out
    assert not (data / "some_summary.prekernelfix.csv").exists()
