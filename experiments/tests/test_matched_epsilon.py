"""II.1 — matched bandwidth schedule for the synchronous pyABC baseline.

The paper claims the baseline is matched to the asynchronous method on the
ABC kernel AND the bandwidth schedule. The asynchronous side selects ε per
arrival with the kernel-weighted ESS-retention rule
(``select_eps_by_ess_retention`` in propulate's abcpmc); the baseline's
``make_matched_epsilon`` must apply the *identical implementation* once per
generation, clamped monotone. These tests pin:

* rule identity — the pyABC adapter's update equals a direct call to the
  propulate module function (asserted cross-arm, not a reimplementation);
* monotonicity — the per-generation ε sequence never increases and starts
  at ``tol_init``;
* the config gate — hard kernels reject the matched mode, ``epsilon_mode``
  selects between matched and legacy quantile;
* integration — the baseline runs end-to-end with the matched epsilon.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pyabc = pytest.importorskip("pyabc")

from propulate.propagators.abcpmc import _make_kernel, select_eps_by_ess_retention

from async_abc.benchmarks import make_benchmark
from async_abc.inference._pyabc_common import make_matched_epsilon
from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
from async_abc.io.paths import OutputDir


def _update_eps(eps, t: int, distances, weights) -> float:
    """Drive the pyABC update convention: update(t) uses generation t-1's
    weighted distances to set ε for generation t; return ε(t)."""
    wd = pd.DataFrame({"distance": distances, "w": weights})
    eps.update(
        t=t,
        get_weighted_distances=lambda: wd,
        get_all_records=lambda: [],
        acceptance_rate=0.5,
        acceptor_config={},
    )
    return float(eps(t))


class TestMatchedEpsilonRuleIdentity:
    def test_update_equals_propulate_rule(self):
        """The adapter must call the SAME implementation the async side uses."""
        rng = np.random.default_rng(0)
        distances = np.abs(rng.normal(0.0, 0.4, size=60))
        weights = rng.uniform(0.5, 1.5, size=60)
        tol_init = 2.0
        kfn = _make_kernel("gaussian")

        eps = make_matched_epsilon("gaussian", tol_init, ess_retention=0.9)
        assert float(eps(0)) == pytest.approx(tol_init)
        got = _update_eps(eps, 1, distances, weights)

        expected = min(
            tol_init,
            select_eps_by_ess_retention(
                weights, distances, tol_init, kfn,
                ess_target=0.9, max_tighten_factor=0.5,
            ),
        )
        assert got == pytest.approx(expected, rel=1e-12)

    def test_ess_retention_knob_changes_result(self):
        # Distance spread on the order of the tolerance, so the ESS curve
        # actually crosses the retention goal above the tightening floor
        # (a too-narrow spread hits the floor for every retention value).
        rng = np.random.default_rng(1)
        distances = np.abs(rng.normal(0.0, 2.0, size=60))
        weights = np.ones(60)
        results = []
        for retention in (0.7, 0.95):
            eps = make_matched_epsilon("gaussian", 2.0, ess_retention=retention)
            eps(0)
            results.append(_update_eps(eps, 1, distances, weights))
        assert results[0] < results[1]  # lower retention tightens harder

    def test_sequence_monotone_from_tol_init(self):
        rng = np.random.default_rng(2)
        eps = make_matched_epsilon("gaussian", 5.0)
        values = [float(eps(0))]
        assert values[0] == pytest.approx(5.0)
        for t in range(1, 6):
            distances = np.abs(rng.normal(0.0, 0.5, size=40))
            values.append(_update_eps(eps, t, distances, np.ones(40)))
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


class TestMatchedEpsilonGate:
    def test_hard_kernel_rejected(self):
        with pytest.raises(ValueError, match="smooth kernel"):
            make_matched_epsilon("hard", 1.0)

    def test_epanechnikov_supported(self):
        eps = make_matched_epsilon("epanechnikov", 1.0)
        assert float(eps(0)) == pytest.approx(1.0)

    def test_get_config_records_rule(self):
        eps = make_matched_epsilon("gaussian", 1.0, ess_retention=0.9)
        config = eps.get_config()
        assert config["rule"] == "ess_retention"
        assert config["ess_retention"] == pytest.approx(0.9)


class TestBaselineEpsilonModeIntegration:
    def _cfg(self, **overrides):
        cfg = {
            "max_simulations": 600,
            "n_workers": 1,
            "k": 15,
            "tol_init": 5.0,
            "n_generations": 3,
            "kernel": "gaussian",
        }
        cfg.update(overrides)
        return cfg

    def _bm(self):
        return make_benchmark(
            {"name": "gaussian_mean", "observed_data_seed": 42, "n_obs": 20}
        )

    def test_baseline_runs_with_matched_epsilon_default(self, tmp_path):
        """Smooth kernel defaults to epsilon_mode='matched' and completes."""
        bm = self._bm()
        od = OutputDir(tmp_path, "matched").ensure()
        records = run_abc_smc_baseline(
            bm.simulate, bm.limits, self._cfg(), od, replicate=0, seed=1
        )
        assert records
        # Per-generation tolerance must be monotone non-increasing.
        tols = {}
        for r in records:
            if r.generation is not None and r.tolerance is not None:
                tols.setdefault(int(r.generation), float(r.tolerance))
        gens = sorted(tols)
        assert len(gens) >= 2
        assert all(
            tols[b] <= tols[a] + 1e-12 for a, b in zip(gens, gens[1:])
        )

    def test_quantile_mode_still_available(self, tmp_path):
        bm = self._bm()
        od = OutputDir(tmp_path, "quantile").ensure()
        records = run_abc_smc_baseline(
            bm.simulate, bm.limits, self._cfg(epsilon_mode="quantile"),
            od, replicate=0, seed=1,
        )
        assert records

    def test_unknown_epsilon_mode_raises(self, tmp_path):
        bm = self._bm()
        od = OutputDir(tmp_path, "bad").ensure()
        with pytest.raises(ValueError, match="epsilon_mode"):
            run_abc_smc_baseline(
                bm.simulate, bm.limits, self._cfg(epsilon_mode="bogus"),
                od, replicate=0, seed=1,
            )

    def test_matched_mode_with_hard_kernel_raises(self, tmp_path):
        bm = self._bm()
        od = OutputDir(tmp_path, "hard").ensure()
        with pytest.raises(ValueError, match="smooth kernel"):
            run_abc_smc_baseline(
                bm.simulate, bm.limits,
                self._cfg(kernel="hard", epsilon_mode="matched"),
                od, replicate=0, seed=1,
            )
