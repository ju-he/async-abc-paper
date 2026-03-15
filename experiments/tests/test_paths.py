"""Tests for async_abc.io.paths."""
import pytest
from async_abc.io.paths import OutputDir


class TestOutputDir:
    def test_creates_subdirs(self, tmp_output_dir):
        od = OutputDir(tmp_output_dir, "my_exp")
        od.ensure()
        assert od.root.exists()
        assert od.plots.exists()
        assert od.data.exists()
        assert od.logs.exists()

    def test_root_name(self, tmp_output_dir):
        od = OutputDir(tmp_output_dir, "my_exp")
        od.ensure()
        assert od.root.name == "my_exp"

    def test_idempotent(self, tmp_output_dir):
        od = OutputDir(tmp_output_dir, "exp")
        od.ensure()
        od.ensure()  # must not raise
        assert od.root.exists()

    def test_plots_subdir_inside_root(self, tmp_output_dir):
        od = OutputDir(tmp_output_dir, "exp")
        od.ensure()
        assert od.plots.parent == od.root

    def test_data_subdir_inside_root(self, tmp_output_dir):
        od = OutputDir(tmp_output_dir, "exp")
        od.ensure()
        assert od.data.parent == od.root

    def test_str_base_path(self, tmp_output_dir):
        od = OutputDir(str(tmp_output_dir), "exp")
        od.ensure()
        assert od.root.exists()
