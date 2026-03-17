"""Tests for Phase 5: plotting module (export + common)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make async_abc importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from matplotlib.figure import Figure

from async_abc.analysis import tolerance_over_wall_time, wasserstein_at_checkpoints
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord
from async_abc.plotting.export import save_figure, get_git_hash
from async_abc.plotting.common import (
    corner_plot,
    gantt_plot,
    posterior_plot,
    quality_vs_time_plot,
    sensitivity_heatmap,
    scaling_plot,
    tolerance_trajectory_plot,
    archive_evolution_plot,
    compute_wasserstein,
)
from async_abc.plotting.reporters import (
    plot_corner,
    plot_quality_vs_time,
    plot_tolerance_trajectory,
    plot_worker_gantt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fig():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _multi_param_records():
    return [
        ParticleRecord(
            method="async_propulate_abc",
            replicate=0,
            seed=1,
            step=i + 1,
            params={"x": 0.1 + 0.1 * i, "y": 1.0 - 0.1 * i},
            loss=0.2 * (i + 1),
            weight=1.0 / 4.0,
            tolerance=1.0,
            wall_time=0.1 * (i + 1),
        )
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# export.py tests
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_export_creates_pdf_and_png(self, tmp_path):
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        save_figure(fig, stem)
        assert (tmp_path / "test_fig.pdf").exists()
        assert (tmp_path / "test_fig.png").exists()

    def test_export_creates_data_csv(self, tmp_path):
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        data = {"x": [1, 2], "y": [3, 4]}
        save_figure(fig, stem, data=data)
        csv_path = tmp_path / "test_fig_data.csv"
        assert csv_path.exists()
        rows = _read_csv(csv_path)
        assert len(rows) == 2
        assert set(rows[0].keys()) == {"x", "y"}

    def test_export_creates_meta_json(self, tmp_path):
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        save_figure(fig, stem)
        meta_path = tmp_path / "test_fig_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "git_hash" in meta
        assert "timestamp" in meta

    def test_export_no_data_csv_when_none(self, tmp_path):
        """No _data.csv should be created when data=None."""
        fig = _make_fig()
        stem = tmp_path / "test_fig"
        save_figure(fig, stem, data=None)
        assert not (tmp_path / "test_fig_data.csv").exists()


class TestGetGitHash:
    def test_returns_string(self):
        h = get_git_hash()
        assert isinstance(h, str)
        assert len(h) > 0


# ---------------------------------------------------------------------------
# common.py tests
# ---------------------------------------------------------------------------

class TestPosteriorPlot:
    def test_posterior_plot_saves_files(self, tmp_path):
        samples = np.random.default_rng(0).normal(0, 1, size=(200,))
        posterior_plot(samples, param_name="mu", path_stem=tmp_path / "posterior")
        assert (tmp_path / "posterior.pdf").exists()
        assert (tmp_path / "posterior.png").exists()

    def test_posterior_plot_returns_paths(self, tmp_path):
        samples = np.random.default_rng(1).normal(size=(100,))
        paths = posterior_plot(samples, param_name="theta", path_stem=tmp_path / "post")
        assert "pdf" in paths and "png" in paths


class TestScalingPlot:
    def test_scaling_plot_saves_files(self, tmp_path):
        throughput = {1: 10.0, 2: 18.0, 4: 32.0, 8: 55.0}
        scaling_plot(throughput, path_stem=tmp_path / "scaling")
        assert (tmp_path / "scaling.pdf").exists()
        assert (tmp_path / "scaling.png").exists()

    def test_scaling_plot_efficiency_curve(self, tmp_path):
        """efficiency[n] == throughput[n] / (n * throughput[1])"""
        throughput = {1: 10.0, 2: 18.0, 4: 32.0}
        scaling_plot(throughput, path_stem=tmp_path / "scaling")
        rows = _read_csv(tmp_path / "scaling_data.csv")
        assert set(rows[0].keys()) >= {"n_workers", "throughput", "efficiency"}
        t1 = throughput[1]
        for row in rows:
            n = float(row["n_workers"])
            t = float(row["throughput"])
            eff = float(row["efficiency"])
            expected = t / (n * t1)
            assert abs(eff - expected) < 1e-9


class TestArchiveEvolutionPlot:
    def test_archive_evolution_plot_saves_files(self, tmp_path):
        sim_counts = np.array([0, 100, 200, 300, 400, 500])
        tolerances = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.8])
        archive_evolution_plot(sim_counts, tolerances, path_stem=tmp_path / "evolution")
        assert (tmp_path / "evolution.pdf").exists()
        assert (tmp_path / "evolution.png").exists()

    def test_archive_evolution_plot_data_csv(self, tmp_path):
        sim_counts = np.array([0, 100, 200])
        tolerances = np.array([5.0, 3.0, 1.5])
        archive_evolution_plot(sim_counts, tolerances, path_stem=tmp_path / "evo")
        rows = _read_csv(tmp_path / "evo_data.csv")
        assert set(rows[0].keys()) >= {"sim_count", "tolerance"}


class TestSensitivityHeatmap:
    def test_sensitivity_heatmap_supports_facets(self, tmp_path):
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ])
        paths = sensitivity_heatmap(
            data,
            row_labels=["50", "100"],
            col_labels=["0.4", "0.8"],
            path_stem=tmp_path / "sensitivity_facet",
            facet_labels=["0.5x", "2.0x"],
        )
        assert "pdf" in paths and "png" in paths and "csv" in paths


class TestPhase3PlotFunctions:
    def test_gantt_plot_returns_figure(self, abc_smc_records):
        fig = gantt_plot(abc_smc_records)
        assert isinstance(fig, Figure)

    def test_quality_vs_time_plot_returns_figure(self, sample_records):
        df = wasserstein_at_checkpoints(sample_records, {"mu": 0.0}, [10, 50])
        fig = quality_vs_time_plot(df)
        assert isinstance(fig, Figure)

    def test_corner_plot_returns_figure(self):
        fig = corner_plot(_multi_param_records(), param_names=["x", "y"], true_params={"x": 0.0, "y": 1.0})
        assert isinstance(fig, Figure)

    def test_tolerance_trajectory_plot_returns_figure(self, sample_records):
        df = tolerance_over_wall_time(sample_records)
        fig = tolerance_trajectory_plot(df)
        assert isinstance(fig, Figure)


class TestPhase3Reporters:
    def test_plot_worker_gantt_exports_files(self, tmp_path, abc_smc_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_worker_gantt(abc_smc_records, output_dir)
        assert (output_dir.plots / "worker_gantt.pdf").exists()
        assert (output_dir.plots / "worker_gantt.png").exists()
        assert (output_dir.plots / "worker_gantt_data.csv").exists()
        assert (output_dir.plots / "worker_gantt_meta.json").exists()

    def test_plot_quality_vs_time_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_time(sample_records, {"mu": 0.0}, [10, 50], output_dir)
        assert (output_dir.plots / "quality_vs_time.pdf").exists()
        assert (output_dir.plots / "quality_vs_time.png").exists()
        assert (output_dir.plots / "quality_vs_time_data.csv").exists()
        assert (output_dir.plots / "quality_vs_time_meta.json").exists()

    def test_plot_corner_exports_files(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_corner(_multi_param_records(), ["x", "y"], output_dir, true_params={"x": 0.0, "y": 1.0})
        assert (output_dir.plots / "corner.pdf").exists()
        assert (output_dir.plots / "corner.png").exists()
        assert (output_dir.plots / "corner_data.csv").exists()
        assert (output_dir.plots / "corner_meta.json").exists()

    def test_plot_tolerance_trajectory_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_tolerance_trajectory(sample_records, output_dir)
        assert (output_dir.plots / "tolerance_trajectory.pdf").exists()
        assert (output_dir.plots / "tolerance_trajectory.png").exists()
        assert (output_dir.plots / "tolerance_trajectory_data.csv").exists()
        assert (output_dir.plots / "tolerance_trajectory_meta.json").exists()


class TestWassersteinMetric:
    def test_wasserstein_metric_returns_float(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(1, 1, 200)
        d = compute_wasserstein(a, b)
        assert isinstance(d, float)

    def test_wasserstein_same_distribution_near_zero(self):
        a = np.random.default_rng(7).normal(0, 1, 1000)
        d = compute_wasserstein(a, a)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_wasserstein_positive_for_different(self):
        rng = np.random.default_rng(99)
        a = rng.normal(0, 1, 500)
        b = rng.normal(5, 1, 500)
        d = compute_wasserstein(a, b)
        assert d > 0
