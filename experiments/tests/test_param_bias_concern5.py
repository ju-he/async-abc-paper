"""Tests for the concern-5 (T2.1) completion: posterior_weight persistence,
the weighted-posterior quality metric, and the drain-after-deadline flag.

External peer review concern 5: AMIS reweighting does not by itself correct
runtime-dependent completion bias. The Tier-1 ablation reported the *unweighted*
top-k archive mean (AMIS-insensitive); these tests cover the machinery that makes
the *reweighted* posterior and the censoring measurement reportable.
"""
import dataclasses

import numpy as np

from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord
from async_abc.reporting.benchmark_reports import (
    _weighted_quantiles,
    write_gaussian_weighted_posterior_summary,
)


def _record(mu: float, loss: float, posterior_weight, *, method="async_propulate_abc",
            replicate=0, tol=1.0) -> ParticleRecord:
    return ParticleRecord(
        method=method,
        replicate=replicate,
        seed=0,
        step=1,
        params={"mu": float(mu)},
        loss=float(loss),
        weight=1.0,
        posterior_weight=posterior_weight,
        tolerance=tol,
        record_kind="simulation_attempt",
    )


# ---------------------------------------------------------------------------
# T2.1a — the record_transform must preserve posterior_weight
# ---------------------------------------------------------------------------

def test_record_transform_preserves_posterior_weight():
    """The heterogeneity runner tags method via dataclasses.replace, which must
    carry *every* field — the hand-listed transform previously dropped
    posterior_weight, leaving it all-empty on disk (the concern-5 blocker)."""
    rec = _record(mu=0.3, loss=0.1, posterior_weight=0.7)
    tagged = dataclasses.replace(rec, method=f"{rec.method}__sigma1.0")
    assert tagged.method == "async_propulate_abc__sigma1.0"
    assert tagged.posterior_weight == 0.7
    # Field-completeness: nothing but method changed.
    for f in dataclasses.fields(ParticleRecord):
        if f.name == "method":
            continue
        assert getattr(tagged, f.name) == getattr(rec, f.name), f.name


# ---------------------------------------------------------------------------
# _weighted_quantiles
# ---------------------------------------------------------------------------

def test_weighted_quantiles_reduce_to_unweighted_for_uniform_weights():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    weights = np.ones_like(values)
    q = _weighted_quantiles(values, weights, [0.5])
    assert abs(q[0] - 2.0) < 1e-9


def test_weighted_quantiles_shift_toward_heavy_atoms():
    values = np.array([0.0, 10.0])
    # Almost all mass on the high atom -> median near the high atom.
    q = _weighted_quantiles(values, np.array([0.01, 0.99]), [0.5])
    assert q[0] > 5.0


# ---------------------------------------------------------------------------
# T2.1b — weighted-posterior summary
# ---------------------------------------------------------------------------

def _cfg():
    return {
        "benchmark": {
            "name": "gaussian_mean",
            "observed_data_seed": 42,
            "n_obs": 100,
            "true_mu": 0.0,
            "sigma_obs": 1.0,
            "prior_low": -5.0,
            "prior_high": 5.0,
        },
        "inference": {"k": 100},
    }


def test_weighted_summary_reweights_away_from_unweighted_mean(tmp_path):
    """The weighted posterior mean must differ from the plain archive mean when
    posterior_weight is skewed — this is exactly the AMIS sensitivity the
    unweighted metric misses."""
    out = OutputDir(str(tmp_path), "wtest").ensure()
    # Archive at mu in {-0.2, +0.2}; weight concentrates on the -0.2 side.
    records = (
        [_record(mu=-0.2, loss=0.01, posterior_weight=9.0) for _ in range(5)]
        + [_record(mu=+0.2, loss=0.02, posterior_weight=1.0) for _ in range(5)]
    )
    write_gaussian_weighted_posterior_summary(
        records, cfg=_cfg(), output_dir=out, archive_size=100
    )
    import pandas as pd

    df = pd.read_csv(out.data / "gaussian_weighted_posterior_summary.csv")
    assert len(df) == 1
    row = df.iloc[0]
    unweighted_mean = 0.0  # symmetric archive
    # Weighted mean pulled toward the heavier -0.2 atoms.
    assert row["weighted_posterior_mean"] < unweighted_mean - 0.05
    # Diagnostics present and sane.
    assert 0.0 < row["ess"] <= len(records)
    assert 0.0 < row["ess_fraction"] <= 1.0
    assert 0.0 < row["max_norm_weight"] <= 1.0
    assert row["weighted_posterior_q05"] <= row["weighted_posterior_q50"] <= row["weighted_posterior_q95"]


def test_weighted_summary_falls_back_to_uniform_without_posterior_weight(tmp_path):
    """pyABC-style records (no posterior_weight) must not crash — they fall back
    to the streaming weight / uniform, giving the unweighted mean."""
    out = OutputDir(str(tmp_path), "wtest2").ensure()
    records = [
        _record(mu=m, loss=0.01, posterior_weight=None, method="abc_smc_baseline")
        for m in (-0.2, 0.2)
    ]
    write_gaussian_weighted_posterior_summary(
        records, cfg=_cfg(), output_dir=out, archive_size=100
    )
    import pandas as pd

    df = pd.read_csv(out.data / "gaussian_weighted_posterior_summary.csv")
    assert len(df) == 1
    assert abs(df.iloc[0]["weighted_posterior_mean"]) < 1e-9


def test_weighted_summary_skips_non_gaussian(tmp_path):
    out = OutputDir(str(tmp_path), "wtest3").ensure()
    cfg = _cfg()
    cfg["benchmark"]["name"] = "gandk"
    write_gaussian_weighted_posterior_summary([], cfg=cfg, output_dir=out, archive_size=100)
    assert not (out.data / "gaussian_weighted_posterior_summary.csv").exists()
