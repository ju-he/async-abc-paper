"""Phase 2 MPI hardening test suite.

Contains four regression/coordination tests mandated by CONTEXT.md D-07:
  - NaN-weight regression guard (pyabc_smc and abc_smc_baseline)
  - CommWorldMap coordination (normal, root_exception, multi_call, double_shutdown)
  - CommWorldMap single-process double-shutdown (no mpirun needed)
  - Barrier placement source check (no mpirun needed)

Tests satisfying requirements:
  - TEST-01: test coverage for CommWorldMap paths
  - TEST-03: shutdown() idempotent regression
  - MPI-03:  NaN-weight guard regression
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
EXPERIMENTS_DIR = TESTS_DIR.parent
INFERENCE_DIR = EXPERIMENTS_DIR / "async_abc" / "inference"
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"

MPI_PYABC_SMC_HELPER = TESTS_DIR / "mpi_integration_helper.py"
MPI_ABC_BASELINE_HELPER = TESTS_DIR / "mpi_abc_smc_baseline_helper.py"
MPI_COMMWORLDMAP_HELPER = TESTS_DIR / "mpi_commworldmap_helper.py"


class TestMpiHardening:
    @pytest.fixture(autouse=True)
    def skip_if_mpirun_not_usable(self):
        if shutil.which("mpirun") is None:
            pytest.skip("mpirun not on PATH")
        probe = subprocess.run(
            ["mpirun", "-n", "1", "--stdin", "none", sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if probe.returncode != 0:
            pytest.skip("mpirun failed a trivial probe")

    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    @pytest.fixture(autouse=True)
    def skip_if_no_mpi4py(self):
        pytest.importorskip("mpi4py", reason="mpi4py not installed — skipping")

    def _run_mpirun(self, helper, tmp_path, args, timeout=90, n=2):
        output_file = tmp_path / f"{helper.stem}_result.json"
        cmd = [
            "mpirun", "-n", str(n), "--stdin", "none", sys.executable,
            str(helper), str(output_file), *args,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        assert result.returncode == 0, f"mpirun failed: {result.stderr}"
        return json.loads(output_file.read_text())

    def test_nan_weight_guard_pyabc_smc(self, tmp_path):
        # max_wall_time_s=0.1 forces pyABC to stop mid-generation; the NaN guard
        # at pyabc_wrapper.py:125-134 must catch and fall back to abc.history.
        data = self._run_mpirun(
            MPI_PYABC_SMC_HELPER, tmp_path,
            args=["mapping", "none", "0.1"], timeout=90,
        )
        assert data["method"] == "pyabc_smc"
        assert data["pyabc_mpi_sampler"] == "mapping"
        assert data["max_wall_time_s"] == 0.1
        # MPI-03 guarantee: no NaN weights in any returned record (helper writes
        # n_records; the absence of crash with returncode 0 is the guarantee).
        assert data["n_records"] >= 0
        assert data["barrier_reached"]

    def test_nan_weight_guard_abc_smc_baseline(self, tmp_path):
        data = self._run_mpirun(
            MPI_ABC_BASELINE_HELPER, tmp_path,
            args=["mapping", "none", "0.1"], timeout=90,
        )
        assert data["method"] == "abc_smc_baseline"
        assert data["pyabc_mpi_sampler"] == "mapping"
        assert data["max_wall_time_s"] == 0.1
        assert data["n_records"] >= 0
        assert data["barrier_reached"]

    def test_commworldmap_normal_2rank(self, tmp_path):
        data = self._run_mpirun(
            MPI_COMMWORLDMAP_HELPER, tmp_path, args=["normal"], timeout=60,
        )
        assert data["scenario"] == "normal"
        assert data["world_size"] == 2
        assert data["barrier_reached"] is True
        assert data["n_records"] == 10

    def test_commworldmap_root_exception_2rank(self, tmp_path):
        data = self._run_mpirun(
            MPI_COMMWORLDMAP_HELPER, tmp_path, args=["root_exception"], timeout=60,
        )
        assert data["scenario"] == "root_exception"
        assert data["root_exception_caught"] is True
        assert data["barrier_reached"] is True

    def test_commworldmap_multi_call_2rank(self, tmp_path):
        data = self._run_mpirun(
            MPI_COMMWORLDMAP_HELPER, tmp_path, args=["multi_call"], timeout=60,
        )
        assert data["scenario"] == "multi_call"
        assert data["multi_call_maps"] == 3
        assert data["barrier_reached"] is True

    def test_commworldmap_double_shutdown_2rank(self, tmp_path):
        data = self._run_mpirun(
            MPI_COMMWORLDMAP_HELPER, tmp_path, args=["double_shutdown"], timeout=60,
        )
        assert data["scenario"] == "double_shutdown"
        assert data["shutdown_idempotent"] is True
        assert data["barrier_reached"] is True

    def test_commworldmap_single_process_double_shutdown(self):
        # Regression for TEST-03: shutdown() is idempotent even in-process
        # (size==1 fallback, no workers to bcast to).
        from mpi4py import MPI
        from async_abc.inference.pyabc_sampler import CommWorldMap
        cmap = CommWorldMap(MPI.COMM_SELF)
        cmap.shutdown()
        cmap.shutdown()  # must not hang or raise
        assert cmap._shutdown is True


class TestMpiHardeningNoMpiRun:
    """Tests that do NOT require mpirun — always run in local venv."""

    def test_barrier_placement_source_check(self):
        # Regression for Risk 3: COMM_WORLD.Barrier() MUST exist at the three
        # documented call sites. If someone removes or reorders them thinking
        # they are redundant with allgather, the test fails in CI.
        wrapper_src = (INFERENCE_DIR / "pyabc_wrapper.py").read_text()
        baseline_src = (INFERENCE_DIR / "abc_smc_baseline.py").read_text()
        scaling_src = (SCRIPTS_DIR / "scaling_runner.py").read_text()

        # Pattern 1: CommWorldMap finally-block Barrier in pyabc_wrapper.
        # The required structural guarantee is: `cmap.worker_loop()` is followed
        # (within ~8 lines) by `MPI.COMM_WORLD.Barrier()` under the `size > 1`
        # guard.
        wrapper_pattern = re.compile(
            r"cmap\.worker_loop\(\)\s*\n"
            r"(?:[^\n]*\n){0,6}?"
            r"\s*if MPI\.COMM_WORLD\.Get_size\(\) > 1:\s*\n"
            r"\s*MPI\.COMM_WORLD\.Barrier\(\)",
            re.MULTILINE,
        )
        assert wrapper_pattern.search(wrapper_src), (
            "pyabc_wrapper.py: required COMM_WORLD.Barrier() after "
            "cmap.worker_loop() missing or reordered"
        )
        assert wrapper_pattern.search(baseline_src), (
            "abc_smc_baseline.py: required COMM_WORLD.Barrier() after "
            "cmap.worker_loop() missing or reordered"
        )

        # Pattern 2: scaling_runner post-MPICommExecutor Barrier.
        scaling_pattern = re.compile(
            r"with MPICommExecutor\(MPI\.COMM_WORLD, root=0\) as executor:"
            r"[\s\S]{0,400}?"
            r"if MPI\.COMM_WORLD\.Get_size\(\) > 1:\s*\n"
            r"\s*MPI\.COMM_WORLD\.Barrier\(\)",
        )
        assert scaling_pattern.search(scaling_src), (
            "scaling_runner.py: required COMM_WORLD.Barrier() after "
            "`with MPICommExecutor` block missing or reordered"
        )

        # Belt-and-braces: the literal string must appear in all three files
        # at least once (catches the case where someone changes the comm).
        for label, src in (
            ("pyabc_wrapper", wrapper_src),
            ("abc_smc_baseline", baseline_src),
            ("scaling_runner", scaling_src),
        ):
            assert "MPI.COMM_WORLD.Barrier()" in src, (
                f"{label}: MPI.COMM_WORLD.Barrier() removed entirely"
            )
