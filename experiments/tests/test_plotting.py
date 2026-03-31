"""Tests for Phase 5: plotting module (export + common)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make async_abc importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
from matplotlib.figure import Figure

from async_abc.analysis import final_state_results, posterior_quality_curve, tolerance_over_wall_time
import async_abc.analysis as analysis_mod
from async_abc.inference.abc_smc_baseline import _prepare_db_path as prepare_abc_db_path
from async_abc.inference.pyabc_wrapper import _prepare_db_path as prepare_pyabc_db_path
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord
import async_abc.plotting.export as export_mod
from async_abc.plotting.export import save_figure, get_git_hash
from async_abc.plotting.common import (
    corner_plot,
    gantt_plot,
    posterior_quality_plot,
    posterior_plot,
    quality_vs_time_plot,
    sensitivity_heatmap,
    scaling_plot,
    threshold_summary_plot,
    tolerance_trajectory_plot,
    archive_evolution_plot,
    compute_wasserstein,
)
from async_abc.plotting.reporters import (
    _compute_idle_fraction,
    _final_population,
    _parse_variant_stem,
    plot_ablation_summary,
    plot_attempts_to_target_summary,
    plot_benchmark_diagnostics,
    plot_corner,
    plot_idle_fraction,
    plot_idle_fraction_comparison,
    plot_generation_timeline,
    plot_quality_vs_attempt_budget,
    plot_quality_vs_posterior_samples,
    plot_quality_vs_time,
    plot_quality_vs_wall_time,
    plot_quality_vs_wall_time_diagnostic,
    plot_sensitivity_summary,
    plot_throughput_over_time,
    plot_time_to_target_summary,
    plot_tolerance_trajectory,
    plot_worker_gantt,
)
from async_abc.plotting.common import (
    idle_fraction_plot,
    idle_fraction_comparison_plot,
    posterior_comparison_plot,
    throughput_over_time_plot,
)
from async_abc.utils.git import find_repo_root


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

    def test_export_uses_agg_fallback_when_external_tools_unavailable(self, tmp_path, monkeypatch):
        """When pdftoppm/convert are unavailable, the Agg fallback should create the PNG."""
        fig = _make_fig()
        stem = tmp_path / "fallback_fig"
        monkeypatch.setattr(export_mod, "_save_png_from_pdf", lambda *args, **kwargs: False)

        save_figure(fig, stem)

        assert (tmp_path / "fallback_fig.pdf").exists()
        assert (tmp_path / "fallback_fig.png").exists()
        meta = json.loads((tmp_path / "fallback_fig_meta.json").read_text())
        assert meta["png"] is not None


class TestGetGitHash:
    def test_returns_string(self):
        h = get_git_hash()
        assert isinstance(h, str)
        assert len(h) > 0


class TestGitHelpers:
    def test_find_repo_root_from_test_file(self):
        repo_root = find_repo_root(Path(__file__).resolve())
        assert repo_root is not None
        assert (repo_root / ".git").exists()


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
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
            ],
            [
                [[1.5, 2.5], [3.5, 4.5]],
                [[2.5, 3.5], [4.5, 5.5]],
            ],
        ])
        paths = sensitivity_heatmap(
            data,
            row_labels=["50", "100"],
            col_labels=["0.4", "0.8"],
            path_stem=tmp_path / "sensitivity_facet",
            facet_row_labels=["0.5x", "2.0x"],
            facet_col_labels=["acceptance_rate", "median"],
        )
        assert "pdf" in paths and "png" in paths and "csv" in paths
        rows = _read_csv(paths["csv"])
        assert set(rows[0].keys()) == {
            "tol_init_multiplier",
            "scheduler_type",
            "k",
            "perturbation_scale",
            "value",
        }
        assert len(rows) == 16

    def test_sensitivity_heatmap_handles_all_nan_data(self, tmp_path):
        data = np.full((2, 2, 2), np.nan)
        paths = sensitivity_heatmap(
            data,
            row_labels=["50", "100"],
            col_labels=["0.4", "0.8"],
            path_stem=tmp_path / "sensitivity_nan",
            facet_labels=["0.5x", "2.0x"],
        )
        assert "pdf" in paths and "png" in paths and "csv" in paths


class TestPhase3PlotFunctions:
    def test_gantt_plot_returns_figure(self, abc_smc_records):
        fig = gantt_plot(abc_smc_records)
        assert isinstance(fig, Figure)

    def test_quality_vs_time_plot_returns_figure(self, sample_records):
        df = posterior_quality_curve(sample_records, {"mu": 0.0}, axis_kind="wall_time", archive_size=20)
        fig = quality_vs_time_plot(df)
        assert isinstance(fig, Figure)

    def test_posterior_quality_plot_supports_attempt_budget(self, sample_records):
        df = posterior_quality_curve(sample_records, {"mu": 0.0}, axis_kind="attempt_budget", archive_size=20)
        fig = posterior_quality_plot(df, axis_kind="attempt_budget")
        assert isinstance(fig, Figure)

    def test_posterior_quality_plot_handles_generation_snapshot_duplicates(self, abc_smc_records):
        df = posterior_quality_curve(abc_smc_records, {"mu": 0.0}, axis_kind="wall_time")
        fig = posterior_quality_plot(df, axis_kind="wall_time")
        assert isinstance(fig, Figure)

    def test_threshold_summary_plot_returns_figure(self, sample_records):
        summary_df = (
            posterior_quality_curve(sample_records, {"mu": 0.0}, axis_kind="wall_time", archive_size=20)
            .head(2)
            .rename(columns={"axis_value": "axis_value_to_threshold"})
        )
        fig = threshold_summary_plot(summary_df, axis_kind="wall_time")
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

    def test_plot_worker_gantt_metadata_is_complete_for_nonbenchmark_plot(self, tmp_path, abc_smc_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_worker_gantt(abc_smc_records, output_dir)
        meta = json.loads((output_dir.plots / "worker_gantt_meta.json").read_text())
        assert meta["plot_name"] == "worker_gantt"
        assert meta["diagnostic_plot"] is True
        assert meta["summary_plot"] is False
        assert meta["experiment_name"] == "plots"
        assert meta["benchmark"] is False
        assert "methods" in meta

    def test_plot_worker_gantt_separates_base_methods_and_replicate_worker_lanes(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        records = [
            ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=1,
                step=1,
                params={"mu": 0.1},
                loss=0.1,
                wall_time=0.1,
                worker_id="0",
                sim_start_time=0.0,
                sim_end_time=0.1,
                record_kind="simulation_attempt",
                time_semantics="event_end",
            ),
            ParticleRecord(
                method="async_propulate_abc",
                replicate=1,
                seed=2,
                step=1,
                params={"mu": 0.2},
                loss=0.2,
                wall_time=0.2,
                worker_id="0",
                sim_start_time=0.1,
                sim_end_time=0.2,
                record_kind="simulation_attempt",
                time_semantics="event_end",
            ),
            ParticleRecord(
                method="abc_smc_baseline",
                replicate=0,
                seed=3,
                step=1,
                params={"mu": -0.1},
                loss=0.3,
                wall_time=0.3,
                worker_id="0",
                sim_start_time=0.0,
                sim_end_time=0.3,
                record_kind="simulation_attempt",
                time_semantics="event_end",
            ),
            ParticleRecord(
                method="abc_smc_baseline",
                replicate=1,
                seed=4,
                step=1,
                params={"mu": -0.2},
                loss=0.4,
                wall_time=0.4,
                worker_id="0",
                sim_start_time=0.1,
                sim_end_time=0.4,
                record_kind="simulation_attempt",
                time_semantics="event_end",
            ),
        ]
        plot_worker_gantt(records, output_dir)
        meta = json.loads((output_dir.plots / "worker_gantt_meta.json").read_text())
        rows = _read_csv(output_dir.plots / "worker_gantt_data.csv")

        assert meta["base_methods"] == ["abc_smc_baseline", "async_propulate_abc"]
        async_lanes = {row["lane_label"] for row in rows if row["base_method"] == "async_propulate_abc"}
        sync_lanes = {row["lane_label"] for row in rows if row["base_method"] == "abc_smc_baseline"}
        assert async_lanes == {"rep 0 | worker 0", "rep 1 | worker 0"}
        assert sync_lanes == {"rep 0 | worker 0", "rep 1 | worker 0"}

    def test_plot_quality_vs_time_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_time(sample_records, {"mu": 0.0}, [10, 50], output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_time.pdf").exists()
        assert (output_dir.plots / "quality_vs_time.png").exists()
        assert (output_dir.plots / "quality_vs_time_data.csv").exists()
        assert (output_dir.plots / "quality_vs_time_meta.json").exists()
        assert (output_dir.plots / "quality_vs_wall_time_diagnostic.pdf").exists()
        meta = json.loads((output_dir.plots / "quality_vs_time_meta.json").read_text())
        assert meta["deprecated_alias_for"] == "quality_vs_wall_time_diagnostic"

    def test_plot_quality_vs_wall_time_diagnostic_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_wall_time_diagnostic(sample_records, {"mu": 0.0}, output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_wall_time_diagnostic.pdf").exists()
        meta = json.loads((output_dir.plots / "quality_vs_wall_time_diagnostic_meta.json").read_text())
        assert meta["plot_name"] == "quality_vs_wall_time_diagnostic"
        assert meta["axis_kind"] == "wall_time"
        assert not (output_dir.plots / "quality_vs_time.pdf").exists()

    def test_plot_benchmark_diagnostics_omits_legacy_quality_alias(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        cfg = {
            "benchmark": {"true_mu": 0.0},
            "inference": {"k": 20},
            "analysis": {"target_wasserstein": 1.0},
            "plots": {"quality_vs_time": True},
        }
        plot_benchmark_diagnostics(sample_records, cfg, output_dir)
        assert (output_dir.plots / "progress_summary.pdf").exists()
        assert (output_dir.plots / "progress_diagnostic.pdf").exists()
        assert (output_dir.plots / "quality_vs_wall_time_diagnostic.pdf").exists()
        assert (output_dir.plots / "quality_vs_posterior_samples.pdf").exists()
        assert (output_dir.plots / "quality_vs_attempt_budget.pdf").exists()
        assert not (output_dir.plots / "quality_vs_time.pdf").exists()

    def test_plot_benchmark_diagnostics_writes_audit_and_diagnostic_companions(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        cfg = {
            "benchmark": {"true_mu": 0.0},
            "inference": {"k": 20},
            "analysis": {"target_wasserstein": 1.0, "min_particles_for_threshold": 5},
            "plots": {
                "archive_evolution": True,
                "tolerance_trajectory": True,
                "quality_vs_time": True,
                "emit_paper_summaries": True,
                "emit_diagnostics": True,
            },
        }
        plot_benchmark_diagnostics(sample_records, cfg, output_dir)
        assert (output_dir.data / "plot_audit.csv").exists()
        assert (output_dir.data / "plot_audit_summary.json").exists()
        assert (output_dir.plots / "archive_evolution.pdf").exists()
        assert (output_dir.plots / "archive_evolution_diagnostic.pdf").exists()
        assert (output_dir.plots / "progress_summary.pdf").exists()
        assert (output_dir.plots / "progress_diagnostic.pdf").exists()
        assert (output_dir.plots / "tolerance_trajectory.pdf").exists()
        assert (output_dir.plots / "tolerance_trajectory_diagnostic.pdf").exists()
        assert (output_dir.plots / "time_to_target_diagnostic_meta.json").exists()

    def test_plot_benchmark_diagnostics_writes_skip_metadata_when_true_params_missing(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        cfg = {
            "benchmark": {},
            "inference": {"k": 20},
            "analysis": {"target_wasserstein": 1.0, "min_particles_for_threshold": 5},
            "plots": {"quality_vs_time": True, "emit_paper_summaries": True, "emit_diagnostics": True},
        }
        plot_benchmark_diagnostics(sample_records, cfg, output_dir)
        meta = json.loads((output_dir.plots / "quality_vs_wall_time_meta.json").read_text())
        assert meta["skip_reason"] == "missing_true_params_or_quality_rows"
        assert meta["skipped"] is True
        diag_meta = json.loads((output_dir.plots / "quality_vs_wall_time_diagnostic_meta.json").read_text())
        assert diag_meta["skip_reason"] == "missing_true_params_or_quality_rows"
        progress_meta = json.loads((output_dir.plots / "progress_summary_meta.json").read_text())
        assert progress_meta["has_tolerance_panel"] is True
        assert progress_meta["has_wasserstein_panel"] is False
        progress_diag_meta = json.loads((output_dir.plots / "progress_diagnostic_meta.json").read_text())
        assert progress_diag_meta["has_tolerance_panel"] is True
        assert progress_diag_meta["has_wasserstein_panel"] is False

    def test_plot_benchmark_diagnostics_writes_lotka_tol_init_diagnostic(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        records = [
            ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=1,
                step=1,
                params={"theta1": 0.5, "theta2": 0.02, "theta3": 0.02, "theta4": 0.5},
                loss=1_000_000.0,
                tolerance=500_000.0,
                wall_time=0.1,
                attempt_count=1,
            ),
            ParticleRecord(
                method="async_propulate_abc",
                replicate=0,
                seed=1,
                step=2,
                params={"theta1": 0.6, "theta2": 0.03, "theta3": 0.03, "theta4": 0.6},
                loss=120.0,
                tolerance=400_000.0,
                wall_time=0.2,
                attempt_count=2,
            ),
            ParticleRecord(
                method="abc_smc_baseline",
                replicate=0,
                seed=1,
                step=1,
                params={"theta1": 0.7, "theta2": 0.04, "theta3": 0.04, "theta4": 0.7},
                loss=1_000_000.0,
                tolerance=500_000.0,
                wall_time=0.3,
                generation=0,
                record_kind="population_particle",
                time_semantics="generation_end",
                attempt_count=3,
            ),
        ]
        cfg = {
            "benchmark": {
                "name": "lotka_volterra",
                "true_theta1": 0.5,
                "true_theta2": 0.025,
                "true_theta3": 0.025,
                "true_theta4": 0.5,
            },
            "inference": {"k": 20, "tol_init": 500_000.0},
            "plots": {},
        }
        plot_benchmark_diagnostics(records, cfg, output_dir)
        diagnostic_json = json.loads((output_dir.data / "lotka_tol_init_diagnostic.json").read_text())
        rows = _read_csv(output_dir.data / "lotka_tol_init_diagnostic.csv")
        audit_rows = _read_csv(output_dir.data / "plot_audit.csv")
        assert diagnostic_json["current_tol_init"] == 500000.0
        assert diagnostic_json["recommended_tol_init"] == pytest.approx(120.0)
        assert diagnostic_json["pathological_fallback"] is True
        assert rows
        assert any("pathological_fallback_or_extinction" in row["invalid_reason"] for row in audit_rows)

    def test_plot_quality_vs_posterior_samples_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_posterior_samples(sample_records, {"mu": 0.0}, output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_posterior_samples.pdf").exists()

    def test_plot_quality_vs_attempt_budget_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_quality_vs_attempt_budget(sample_records, {"mu": 0.0}, output_dir, archive_size=20)
        assert (output_dir.plots / "quality_vs_attempt_budget.pdf").exists()

    def test_quality_summary_exports_clamp_wasserstein_ci_low_to_zero(self, tmp_path, monkeypatch):
        quality_df = pd.DataFrame(
            [
                {"method": "async_propulate_abc", "replicate": 0, "axis_value": 1.0, "wasserstein": 0.1},
                {"method": "async_propulate_abc", "replicate": 1, "axis_value": 1.0, "wasserstein": 4.0},
                {"method": "async_propulate_abc", "replicate": 0, "axis_value": 2.0, "wasserstein": 0.2},
                {"method": "async_propulate_abc", "replicate": 1, "axis_value": 2.0, "wasserstein": 5.0},
            ]
        )
        monkeypatch.setattr(analysis_mod, "posterior_quality_curve", lambda *args, **kwargs: quality_df.copy())
        output_dir = OutputDir(tmp_path, "plots").ensure()

        plot_quality_vs_wall_time([], {"mu": 0.0}, output_dir, archive_size=20)
        plot_quality_vs_posterior_samples([], {"mu": 0.0}, output_dir, archive_size=20)
        plot_quality_vs_attempt_budget([], {"mu": 0.0}, output_dir, archive_size=20)

        for stem in (
            "quality_vs_wall_time_data.csv",
            "quality_vs_posterior_samples_data.csv",
            "quality_vs_attempt_budget_data.csv",
        ):
            rows = _read_csv(output_dir.plots / stem)
            lows = [float(row["wasserstein_ci_low"]) for row in rows if row["wasserstein_ci_low"] and row["wasserstein_ci_low"].lower() != "nan"]
            assert lows
            assert all(low >= 0.0 for low in lows)

    def test_plot_time_to_target_summary_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_time_to_target_summary(sample_records, {"mu": 0.0}, 10.0, output_dir, archive_size=20)
        assert (output_dir.plots / "time_to_target_summary.pdf").exists()

    def test_plot_time_to_target_summary_respects_min_particles_threshold(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_time_to_target_summary(
            sample_records,
            {"mu": 0.0},
            10.0,
            output_dir,
            archive_size=20,
            min_particles_for_threshold=1000,
        )
        meta = json.loads((output_dir.plots / "time_to_target_summary_meta.json").read_text())
        assert meta["skip_reason"] == "threshold_not_reached"

    def test_plot_attempts_to_target_summary_exports_files(self, tmp_path, sample_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_attempts_to_target_summary(sample_records, {"mu": 0.0}, 10.0, output_dir, archive_size=20)
        assert (output_dir.plots / "attempts_to_target_summary.pdf").exists()

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

    def test_plot_generation_timeline_exports_files(self, tmp_path, abc_smc_records):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        plot_generation_timeline(abc_smc_records, output_dir)
        assert (output_dir.plots / "generation_timeline.pdf").exists()
        rows = _read_csv(output_dir.plots / "generation_timeline_data.csv")
        assert set(rows[0].keys()) >= {
            "method",
            "replicate",
            "generation",
            "gen_start",
            "gen_end",
            "gen_duration",
            "n_particles",
        }

    def test_plot_sensitivity_summary_exports_files(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        data_dir = output_dir.data
        rows = [
            {"tolerance": "2.0"},
            {"tolerance": "1.5"},
            {"tolerance": "1.0"},
        ]
        for stem in (
            "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=0.5.csv",
            "sensitivity_k=100__perturbation_scale=0.4__tol_init_multiplier=0.5.csv",
            "sensitivity_k=50__perturbation_scale=0.8__tol_init_multiplier=2.0.csv",
            "sensitivity_k=100__perturbation_scale=0.8__tol_init_multiplier=2.0.csv",
        ):
            with open(data_dir / stem, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["tolerance"])
                writer.writeheader()
                writer.writerows(rows)
        grid = {
            "k": [50, 100],
            "perturbation_scale": [0.4, 0.8],
            "tol_init_multiplier": [0.5, 2.0],
        }
        plot_sensitivity_summary(data_dir, grid, output_dir)
        assert (output_dir.plots / "sensitivity_heatmap.pdf").exists()
        assert (output_dir.plots / "sensitivity_heatmap.png").exists()
        assert (output_dir.plots / "sensitivity_heatmap_data.csv").exists()
        meta = json.loads((output_dir.plots / "sensitivity_heatmap_meta.json").read_text())
        assert meta["plot_name"] == "sensitivity_heatmap"
        assert meta["summary_plot"] is True
        assert meta["experiment_name"] == "plots"
        assert meta["benchmark"] is False

    def test_plot_ablation_summary_exports_complete_metadata(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        data_dir = output_dir.data
        for name in ("baseline", "variant"):
            with open(data_dir / f"ablation_{name}.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["tolerance"])
                writer.writeheader()
                writer.writerows([{"tolerance": "2.0"}, {"tolerance": "1.0"}])
        plot_ablation_summary(
            data_dir,
            variants=[{"name": "baseline"}, {"name": "variant"}],
            output_dir=output_dir,
        )
        meta = json.loads((output_dir.plots / "ablation_comparison_meta.json").read_text())
        assert meta["plot_name"] == "ablation_comparison"
        assert meta["summary_plot"] is True
        assert meta["experiment_name"] == "plots"
        assert meta["benchmark"] is False


class TestSensitivityVariantParsing:
    def test_parses_new_variant_stem(self):
        variant = _parse_variant_stem(
            "k=50__perturbation_scale=0.4__scheduler_type=acceptance_rate__tol_init_multiplier=0.5",
            ["k", "perturbation_scale", "scheduler_type", "tol_init_multiplier"],
        )
        assert variant == {
            "k": 50,
            "perturbation_scale": 0.4,
            "scheduler_type": "acceptance_rate",
            "tol_init_multiplier": 0.5,
        }

    def test_parses_legacy_variant_stem(self):
        variant = _parse_variant_stem(
            "k50_perturbation_scale0.4_scheduler_typeacceptance_rate_tol_init_multiplier0.5",
            ["k", "perturbation_scale", "scheduler_type", "tol_init_multiplier"],
        )
        assert variant == {
            "k": 50,
            "perturbation_scale": 0.4,
            "scheduler_type": "acceptance_rate",
            "tol_init_multiplier": 0.5,
        }


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


# ---------------------------------------------------------------------------
# _final_population per-method fix
# ---------------------------------------------------------------------------

class TestFinalPopulationPerMethod:
    def test_async_archive_reconstruction_uses_final_epsilon_and_archive_cap(self):
        records = [
            ParticleRecord(method="async_propulate_abc", replicate=0, seed=1, step=1,
                           params={"x": 1.0}, loss=0.45, tolerance=4.0, wall_time=1.0),
            ParticleRecord(method="async_propulate_abc", replicate=0, seed=1, step=2,
                           params={"x": 1.1}, loss=0.30, tolerance=2.0, wall_time=2.0),
            ParticleRecord(method="async_propulate_abc", replicate=0, seed=1, step=3,
                           params={"x": 1.2}, loss=0.20, tolerance=1.0, wall_time=3.0),
            ParticleRecord(method="async_propulate_abc", replicate=0, seed=1, step=4,
                           params={"x": 1.3}, loss=0.10, tolerance=0.5, wall_time=4.0),
            ParticleRecord(method="async_propulate_abc", replicate=0, seed=1, step=5,
                           params={"x": 9.9}, loss=0.80, tolerance=0.5, wall_time=5.0),
        ]
        results = final_state_results(records, archive_size=3)
        assert len(results) == 1
        result = results[0]
        assert result.state_kind == "archive_reconstruction"
        assert result.n_particles_used == 3
        assert [r.params["x"] for r in result.records] == [1.3, 1.2, 1.1]

    def test_sync_final_population_pools_final_generation_per_replicate(self):
        records = [
            ParticleRecord(method="abc_smc_baseline", replicate=0, seed=1, step=1,
                           params={"x": 0.9}, loss=0.9, tolerance=2.0, wall_time=1.0, generation=0),
            ParticleRecord(method="abc_smc_baseline", replicate=0, seed=1, step=2,
                           params={"x": 0.1}, loss=0.1, tolerance=1.0, wall_time=2.0, generation=1),
            ParticleRecord(method="abc_smc_baseline", replicate=1, seed=2, step=1,
                           params={"x": 1.9}, loss=0.8, tolerance=3.0, wall_time=1.5, generation=0),
            ParticleRecord(method="abc_smc_baseline", replicate=1, seed=2, step=2,
                           params={"x": 1.1}, loss=0.2, tolerance=1.5, wall_time=2.5, generation=1),
        ]
        final = _final_population(records)
        assert len(final) == 2
        assert {r.replicate for r in final} == {0, 1}
        assert {r.generation for r in final} == {1}
        assert {r.params["x"] for r in final} == {0.1, 1.1}


# ---------------------------------------------------------------------------
# Posterior comparison plot
# ---------------------------------------------------------------------------

class TestPosteriorComparisonPlot:
    def test_saves_files(self, tmp_path):
        rng = np.random.default_rng(42)
        method_samples = {
            "method_a": rng.normal(0, 1, 100),
            "method_b": rng.normal(1, 1, 100),
        }
        posterior_comparison_plot(method_samples, param_name="mu", path_stem=tmp_path / "post_cmp")
        assert (tmp_path / "post_cmp.pdf").exists()
        assert (tmp_path / "post_cmp.png").exists()


# ---------------------------------------------------------------------------
# Corner plot with method labels
# ---------------------------------------------------------------------------

class TestCornerPlotMethodLabels:
    def test_corner_plot_with_methods(self):
        records = [
            ParticleRecord(method="A", replicate=0, seed=1, step=i,
                           params={"x": float(i), "y": float(i) * 2}, loss=0.1, wall_time=0.1)
            for i in range(10)
        ] + [
            ParticleRecord(method="B", replicate=0, seed=1, step=i,
                           params={"x": float(i) + 5, "y": float(i) * 3}, loss=0.1, wall_time=0.1)
            for i in range(10)
        ]
        fig = corner_plot(records, param_names=["x", "y"], method_labels=["A", "B"])
        assert fig is not None


# ---------------------------------------------------------------------------
# Idle fraction / throughput / comparison plots
# ---------------------------------------------------------------------------

def _heterogeneity_records():
    """Synthetic records mimicking runtime_heterogeneity output."""
    records = []
    for sigma in [0.0, 0.5, 1.0]:
        for rep in range(2):
            for step in range(5):
                records.append(ParticleRecord(
                    method=f"async_propulate_abc__sigma{sigma}",
                    replicate=rep, seed=42, step=step,
                    params={"x": float(step)}, loss=float(step) * 0.1,
                    tolerance=1.0,
                    wall_time=0.1 * step,
                    worker_id=str(step % 2),
                    sim_start_time=0.1 * step,
                    sim_end_time=0.1 * step + 0.08,
                ))
    return records


def _mixed_runtime_records():
    return [
        ParticleRecord(
            method="async_propulate_abc__sigma1.0",
            replicate=0,
            seed=42,
            step=1,
            params={"x": 0.0},
            loss=0.1,
            tolerance=1.0,
            wall_time=0.2,
            worker_id="0",
            sim_start_time=0.0,
            sim_end_time=0.2,
            attempt_count=1,
        ),
        ParticleRecord(
            method="async_propulate_abc__sigma1.0",
            replicate=0,
            seed=42,
            step=2,
            params={"x": 1.0},
            loss=0.2,
            tolerance=0.8,
            wall_time=0.6,
            worker_id="1",
            sim_start_time=0.3,
            sim_end_time=0.6,
            attempt_count=2,
        ),
        ParticleRecord(
            method="abc_smc_baseline__sigma1.0",
            replicate=0,
            seed=42,
            step=1,
            params={"x": 0.5},
            loss=0.5,
            tolerance=1.2,
            wall_time=1.0,
            sim_start_time=0.0,
            sim_end_time=0.8,
            generation=0,
            attempt_count=2,
        ),
        ParticleRecord(
            method="abc_smc_baseline__sigma1.0",
            replicate=0,
            seed=42,
            step=2,
            params={"x": 0.6},
            loss=0.4,
            tolerance=1.2,
            wall_time=1.0,
            sim_start_time=0.0,
            sim_end_time=1.0,
            generation=0,
            attempt_count=2,
        ),
    ]


def _mixed_runtime_records_multi_sigma():
    records = _mixed_runtime_records()
    records.extend(
        [
            ParticleRecord(
                method="async_propulate_abc__sigma2.0",
                replicate=1,
                seed=43,
                step=1,
                params={"x": 0.0},
                loss=0.1,
                tolerance=1.0,
                wall_time=0.4,
                worker_id="0",
                sim_start_time=0.0,
                sim_end_time=0.4,
                attempt_count=1,
            ),
            ParticleRecord(
                method="abc_smc_baseline__sigma2.0",
                replicate=1,
                seed=43,
                step=1,
                params={"x": 0.5},
                loss=0.5,
                tolerance=1.2,
                wall_time=1.5,
                sim_start_time=0.0,
                sim_end_time=1.5,
                generation=0,
                attempt_count=3,
            ),
        ]
    )
    return records


class TestIdleFractionPlot:
    def test_idle_fraction_plot_saves_files(self, tmp_path):
        idle_fraction_plot([0.0, 0.5, 1.0], [0.1, 0.3, 0.5], tmp_path / "idle")
        assert (tmp_path / "idle.pdf").exists()

    def test_plot_idle_fraction_reporter(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction(_heterogeneity_records(), output_dir)
        assert (output_dir.plots / "idle_fraction.pdf").exists()

    def test_plot_idle_fraction_includes_async_and_sync_measurement_methods(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction(_mixed_runtime_records(), output_dir)
        rows = _read_csv(output_dir.plots / "idle_fraction_data.csv")
        assert {row["measurement_method"] for row in rows} == {"worker_idle", "barrier_overhead"}
        assert {row["base_method"] for row in rows} == {"async_propulate_abc", "abc_smc_baseline"}
        barrier_rows = [row for row in rows if row["measurement_method"] == "barrier_overhead"]
        assert {row["base_method"] for row in barrier_rows} == {"abc_smc_baseline"}

    def test_plot_idle_fraction_handles_multiple_sigmas_without_shape_mismatch(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction(_mixed_runtime_records_multi_sigma(), output_dir)
        rows = _read_csv(output_dir.plots / "idle_fraction_data.csv")
        assert {row["sigma"] for row in rows} == {"1.0", "2.0"}
        assert (output_dir.plots / "idle_fraction.pdf").exists()

    def test_plot_idle_fraction_clips_exported_confidence_intervals_to_unit_interval(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        records = _mixed_runtime_records_multi_sigma() + [
            ParticleRecord(
                method="async_propulate_abc__sigma1.0",
                replicate=2,
                seed=44,
                step=1,
                params={"x": 0.0},
                loss=0.1,
                tolerance=1.0,
                wall_time=2.0,
                worker_id="0",
                sim_start_time=0.0,
                sim_end_time=2.0,
                attempt_count=1,
                record_kind="simulation_attempt",
            ),
            ParticleRecord(
                method="async_propulate_abc__sigma1.0",
                replicate=2,
                seed=44,
                step=2,
                params={"x": 1.0},
                loss=0.2,
                tolerance=0.8,
                wall_time=2.1,
                worker_id="1",
                sim_start_time=2.0,
                sim_end_time=2.1,
                attempt_count=2,
                record_kind="simulation_attempt",
            ),
        ]
        plot_idle_fraction(records, output_dir)
        rows = _read_csv(output_dir.plots / "idle_fraction_data.csv")
        for row in rows:
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            if np.isfinite(ci_low):
                assert 0.0 <= ci_low <= 1.0
            if np.isfinite(ci_high):
                assert 0.0 <= ci_high <= 1.0

    def test_plot_idle_fraction_comparison_metadata_is_complete(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction_comparison(_mixed_runtime_records(), output_dir)
        meta = json.loads((output_dir.plots / "idle_fraction_comparison_meta.json").read_text())
        assert meta["plot_name"] == "idle_fraction_comparison"
        assert meta["summary_plot"] is True
        assert meta["benchmark"] is False
        assert meta["experiment_name"] == "test"


class TestThroughputOverTimePlot:
    def test_throughput_over_time_plot_saves_files(self, tmp_path):
        time_bins = {"σ=0.0": np.array([0.5, 1.5, 2.5])}
        throughput_bins = {"σ=0.0": np.array([5.0, 3.0, 2.0])}
        throughput_over_time_plot(time_bins, throughput_bins, tmp_path / "throughput")
        assert (tmp_path / "throughput.pdf").exists()

    def test_plot_throughput_over_time_reporter(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_throughput_over_time(_heterogeneity_records(), output_dir)
        assert (output_dir.plots / "throughput_over_time.pdf").exists()
        rows = _read_csv(output_dir.plots / "throughput_over_time_data.csv")
        assert set(rows[0].keys()) == {"sigma", "base_method", "bin_mid", "throughput_sims_per_s", "n_replicates"}
        meta = json.loads((output_dir.plots / "throughput_over_time_meta.json").read_text())
        assert meta["plot_name"] == "throughput_over_time"
        assert meta["facet_by"] == "sigma"


class TestIdleFractionComparisonPlot:
    def test_idle_fraction_comparison_plot_saves_files(self, tmp_path):
        idle_by_sigma = {0.0: [0.1, 0.15], 0.5: [0.3, 0.35], 1.0: [0.5, 0.55]}
        idle_fraction_comparison_plot([0.0, 0.5, 1.0], idle_by_sigma, tmp_path / "idle_cmp")
        assert (tmp_path / "idle_cmp.pdf").exists()

    def test_plot_idle_fraction_comparison_reporter(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction_comparison(_heterogeneity_records(), output_dir)
        assert (output_dir.plots / "idle_fraction_comparison.pdf").exists()

    def test_plot_idle_fraction_comparison_exports_measurement_method_column(self, tmp_path):
        output_dir = OutputDir(tmp_path, "test").ensure()
        plot_idle_fraction_comparison(_mixed_runtime_records(), output_dir)
        rows = _read_csv(output_dir.plots / "idle_fraction_comparison_data.csv")
        assert {row["measurement_method"] for row in rows} == {"worker_idle", "barrier_overhead"}


class TestComputeIdleFraction:
    def test_returns_per_method_per_replicate(self):
        records = _heterogeneity_records()
        idle = _compute_idle_fraction(records)
        assert "async_propulate_abc__sigma0.0" in idle
        assert 0 in idle["async_propulate_abc__sigma0.0"]
        assert 0 <= idle["async_propulate_abc__sigma0.0"][0] <= 1


class TestToleranceTrajectory:
    def test_tolerance_over_wall_time_deduplicates_sync_generation_duplicates(self, abc_smc_records):
        df = tolerance_over_wall_time(abc_smc_records)
        assert len(df) == 2
        assert list(df["tolerance"]) == [1.0, 0.5]


class TestTaggedDbPaths:
    def test_prepare_abc_db_path_tags_db_by_replicate_seed_and_checkpoint(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        db_file = output_dir.data / "abc_smc_baseline_rep3_seed11__sigma0_5.db"
        wal_file = Path(f"{db_file}-wal")
        db_file.write_text("stale")
        wal_file.write_text("stale")
        url = prepare_abc_db_path(
            output_dir,
            method_name="abc_smc_baseline",
            replicate=3,
            seed=11,
            checkpoint_tag="sigma0.5",
        )
        assert "abc_smc_baseline_rep3_seed11__sigma0_5.db" in url
        assert not db_file.exists()
        assert not wal_file.exists()

    def test_prepare_pyabc_db_path_tags_db_by_replicate_seed_and_checkpoint(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        url = prepare_pyabc_db_path(
            output_dir,
            method_name="pyabc_smc",
            replicate=2,
            seed=7,
            checkpoint_tag="slowdown=5",
        )
        assert "pyabc_smc_rep2_seed7__slowdown_5.db" in url


# ---------------------------------------------------------------------------
# PNG Agg fallback
# ---------------------------------------------------------------------------

class TestQualityBySigmaPlot:
    """Phase 3: plot_quality_by_sigma integration tests."""

    def test_creates_pdf(self, runtime_heterogeneity_runner_artifact):
        plots_dir = (
            runtime_heterogeneity_runner_artifact["root"]
            / "runtime_heterogeneity"
            / "plots"
        )
        assert (plots_dir / "quality_by_sigma.pdf").exists(), (
            "quality_by_sigma.pdf must be created by the runner"
        )

    def test_creates_data_csv(self, runtime_heterogeneity_runner_artifact):
        plots_dir = (
            runtime_heterogeneity_runner_artifact["root"]
            / "runtime_heterogeneity"
            / "plots"
        )
        csv_path = plots_dir / "quality_by_sigma_data.csv"
        assert csv_path.exists(), "quality_by_sigma_data.csv must exist"

    def test_meta_records_n_sigma_levels(self, runtime_heterogeneity_runner_artifact):
        plots_dir = (
            runtime_heterogeneity_runner_artifact["root"]
            / "runtime_heterogeneity"
            / "plots"
        )
        meta = json.loads((plots_dir / "quality_by_sigma_meta.json").read_text())
        assert "n_sigma_levels" in meta
        assert int(meta["n_sigma_levels"]) >= 1

    def test_unit_returns_pdf_without_quality_data(self, tmp_path):
        """plot_quality_by_sigma creates output even when quality curves are empty."""
        from async_abc.plotting.reporters import plot_quality_by_sigma
        from async_abc.io.paths import OutputDir

        # Records with __sigma tags but no true_params derivable → graceful skip
        rec = ParticleRecord(
            method="fake__sigma1.0",
            replicate=0,
            seed=0,
            step=1,
            params={"mu": 0.1},
            loss=0.4,
            weight=0.5,
            tolerance=1.0,
            wall_time=1.0,
        )
        output_dir = OutputDir(tmp_path, "test_quality").ensure()
        cfg = {
            "benchmark": {"name": "gaussian_mean"},  # no true_mu → skip
            "inference": {"k": 5},
        }
        plot_quality_by_sigma([rec], cfg, output_dir)
        # Should create a meta file (either skip metadata or actual plot)
        assert (
            (output_dir.plots / "quality_by_sigma.pdf").exists()
            or (output_dir.plots / "quality_by_sigma_meta.json").exists()
        )


class TestPngAggFallback:
    def test_png_created_when_external_tools_unavailable(self, tmp_path, monkeypatch):
        """Even without pdftoppm/convert, PNG should be created via Agg fallback."""
        fig = _make_fig()
        monkeypatch.setattr(export_mod, "_save_png_from_pdf", lambda *a, **kw: False)
        stem = tmp_path / "fallback_fig"
        result = save_figure(fig, stem)
        assert (tmp_path / "fallback_fig.png").exists()
        meta = json.loads((tmp_path / "fallback_fig_meta.json").read_text())
        assert meta["png"] is not None


# ---------------------------------------------------------------------------
# Phase 1 wiring: quality_df parameter
# ---------------------------------------------------------------------------

class TestSensitivitySummaryWithQualityDf:
    """plot_sensitivity_summary accepts quality_df and uses wasserstein_mean."""

    def _make_quality_df(self, grid: dict) -> pd.DataFrame:
        import itertools
        keys = sorted(grid.keys())
        rows = []
        for combo in itertools.product(*[grid[k] for k in keys]):
            row = dict(zip(keys, combo))
            row["wasserstein_mean"] = 0.5
            row["wasserstein_std"] = 0.05
            row["n_replicates"] = 3
            rows.append(row)
        return pd.DataFrame(rows)

    def _make_tol_csvs(self, data_dir: Path, grid: dict) -> None:
        import itertools
        keys = sorted(grid.keys())
        for combo in itertools.product(*[grid[k] for k in keys]):
            variant = dict(zip(keys, combo))
            name = "__".join(f"{k}={v}" for k, v in sorted(variant.items()))
            with open(data_dir / f"sensitivity_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["tolerance"])
                w.writeheader()
                w.writerows([{"tolerance": "1.0"}])

    def test_quality_df_parameter_accepted(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {"k": [50, 100], "perturbation_scale": [0.4, 0.8], "tol_init_multiplier": [1.0]}
        self._make_tol_csvs(output_dir.data, grid)
        quality_df = self._make_quality_df(grid)
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)

    def test_quality_df_produces_heatmap_file(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {"k": [50, 100], "perturbation_scale": [0.4, 0.8], "tol_init_multiplier": [1.0]}
        self._make_tol_csvs(output_dir.data, grid)
        quality_df = self._make_quality_df(grid)
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)
        pdfs = list(output_dir.plots.glob("sensitivity_heatmap*.pdf"))
        assert len(pdfs) >= 1, f"Expected at least one PDF, got: {[p.name for p in pdfs]}"

    def test_quality_summary_csv_written(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {"k": [50, 100], "perturbation_scale": [0.4, 0.8], "tol_init_multiplier": [1.0]}
        self._make_tol_csvs(output_dir.data, grid)
        quality_df = self._make_quality_df(grid)
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)
        assert (output_dir.data / "sensitivity_quality_summary.csv").exists()


# ---------------------------------------------------------------------------
# Phase 2: per-scheduler heatmap figures
# ---------------------------------------------------------------------------

class TestSensitivityHeatmapPerScheduler:
    """When scheduler_type is in the grid, one figure per scheduler is produced."""

    def _quality_df(self, grid: dict) -> pd.DataFrame:
        import itertools
        keys = sorted(grid.keys())
        rows = []
        for combo in itertools.product(*[grid[k] for k in keys]):
            row = dict(zip(keys, combo))
            row["wasserstein_mean"] = 0.5
            row["wasserstein_std"] = 0.05
            row["n_replicates"] = 3
            rows.append(row)
        return pd.DataFrame(rows)

    def _tol_csvs(self, data_dir: Path, grid: dict) -> None:
        import itertools
        keys = sorted(grid.keys())
        for combo in itertools.product(*[grid[k] for k in keys]):
            variant = dict(zip(keys, combo))
            name = "__".join(f"{k}={v}" for k, v in sorted(variant.items()))
            with open(data_dir / f"sensitivity_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["tolerance"])
                w.writeheader()
                w.writerows([{"tolerance": "1.0"}])

    def test_generates_one_figure_per_scheduler(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {
            "k": [50, 100],
            "perturbation_scale": [0.4, 0.8],
            "tol_init_multiplier": [0.5, 1.0],
            "scheduler_type": ["acceptance_rate", "quantile"],
        }
        self._tol_csvs(output_dir.data, grid)
        quality_df = self._quality_df(grid)
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)
        pdfs = {p.name for p in output_dir.plots.glob("sensitivity_heatmap*.pdf")}
        assert "sensitivity_heatmap__scheduler_type=acceptance_rate.pdf" in pdfs, pdfs
        assert "sensitivity_heatmap__scheduler_type=quantile.pdf" in pdfs, pdfs

    def test_per_scheduler_csv_exported(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {
            "k": [50, 100],
            "perturbation_scale": [0.4, 0.8],
            "tol_init_multiplier": [1.0],
            "scheduler_type": ["acceptance_rate", "geometric_decay"],
        }
        self._tol_csvs(output_dir.data, grid)
        quality_df = self._quality_df(grid)
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)
        for sched in ("acceptance_rate", "geometric_decay"):
            csv_path = output_dir.plots / f"sensitivity_heatmap__scheduler_type={sched}_data.csv"
            assert csv_path.exists(), f"Missing CSV for scheduler={sched}"

    def test_missing_scheduler_data_skips_gracefully(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {
            "k": [50],
            "perturbation_scale": [0.4],
            "tol_init_multiplier": [1.0],
            "scheduler_type": ["acceptance_rate", "quantile"],
        }
        self._tol_csvs(output_dir.data, grid)
        import itertools
        keys = sorted(grid.keys())
        rows = []
        for combo in itertools.product(*[grid[k] for k in keys]):
            row = dict(zip(keys, combo))
            if row["scheduler_type"] == "quantile":
                row["wasserstein_mean"] = float("nan")
                row["wasserstein_std"] = float("nan")
                row["n_replicates"] = 0
            else:
                row["wasserstein_mean"] = 0.5
                row["wasserstein_std"] = 0.05
                row["n_replicates"] = 3
            rows.append(row)
        quality_df = pd.DataFrame(rows)
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)

    def test_no_scheduler_type_in_grid_gives_single_heatmap(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {
            "k": [50, 100],
            "perturbation_scale": [0.4, 0.8],
            "tol_init_multiplier": [0.5, 1.0],
        }
        self._tol_csvs(output_dir.data, grid)
        quality_df = self._quality_df(grid)
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)
        assert (output_dir.plots / "sensitivity_heatmap.pdf").exists()


# ---------------------------------------------------------------------------
# Phase 3: per-replicate std in heatmap
# ---------------------------------------------------------------------------

class TestSensitivityHeatmapUncertainty:

    def test_csv_output_contains_wasserstein_std(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        import itertools
        grid = {"k": [50, 100], "perturbation_scale": [0.4, 0.8], "tol_init_multiplier": [1.0]}
        keys = sorted(grid.keys())
        rows = []
        for combo in itertools.product(*[grid[k] for k in keys]):
            row = dict(zip(keys, combo))
            row["wasserstein_mean"] = 0.5
            row["wasserstein_std"] = 0.08
            row["n_replicates"] = 5
            rows.append(row)
        quality_df = pd.DataFrame(rows)
        for row in rows:
            name = "__".join(f"{k}={row[k]}" for k in keys)
            with open(output_dir.data / f"sensitivity_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["tolerance"])
                w.writeheader()
                w.writerow({"tolerance": "1.0"})
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)
        csv_path = output_dir.plots / "sensitivity_heatmap_data.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        assert "wasserstein_std" in cols, f"wasserstein_std missing: {cols}"

    def test_n_replicates_in_heatmap_metadata(self, tmp_path):
        output_dir = OutputDir(tmp_path, "plots").ensure()
        grid = {"k": [50], "perturbation_scale": [0.4], "tol_init_multiplier": [1.0]}
        quality_df = pd.DataFrame([{
            "k": 50, "perturbation_scale": 0.4, "tol_init_multiplier": 1.0,
            "wasserstein_mean": 0.3, "wasserstein_std": 0.02, "n_replicates": 5,
        }])
        with open(output_dir.data / "sensitivity_k=50__perturbation_scale=0.4__tol_init_multiplier=1.0.csv",
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tolerance"])
            w.writeheader()
            w.writerow({"tolerance": "1.0"})
        plot_sensitivity_summary(output_dir.data, grid, output_dir, quality_df=quality_df)
        metas = list(output_dir.plots.glob("sensitivity_heatmap*meta.json"))
        assert metas, "No metadata JSON found"
        meta = json.loads(metas[0].read_text())
        assert "n_replicates_min" in meta.get("extra", {}), f"n_replicates_min missing: {meta}"
