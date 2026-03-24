"""Tests for async_abc.io.records."""
import csv
import time
from pathlib import Path

import pytest

from async_abc.io.records import ParticleRecord, RecordWriter


def make_record(**kwargs):
    defaults = dict(
        method="async_propulate_abc",
        replicate=0,
        seed=42,
        step=1,
        params={"mu": 0.5},
        loss=1.23,
        weight=0.8,
        tolerance=5.0,
        wall_time=0.01,
    )
    defaults.update(kwargs)
    return ParticleRecord(**defaults)


class TestParticleRecord:
    def test_creation(self):
        r = make_record()
        assert r.method == "async_propulate_abc"
        assert r.loss == pytest.approx(1.23)

    def test_params_stored(self):
        r = make_record(params={"mu": 1.0, "sigma": 2.0})
        assert r.params["mu"] == 1.0

    def test_optional_weight_none(self):
        r = make_record(weight=None)
        assert r.weight is None

    def test_optional_tolerance_none(self):
        r = make_record(tolerance=None)
        assert r.tolerance is None

    def test_worker_event_fields_default_to_none(self):
        r = ParticleRecord(
            method="m",
            replicate=0,
            seed=1,
            step=0,
            params={},
            loss=0.1,
        )
        assert r.worker_id is None
        assert r.sim_start_time is None
        assert r.sim_end_time is None
        assert r.generation is None
        assert r.record_kind is None
        assert r.time_semantics is None
        assert r.attempt_count is None


class TestRecordWriter:
    def test_creates_file(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        writer = RecordWriter(path)
        writer.write([make_record()])
        assert path.exists()

    def test_csv_has_header(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        writer = RecordWriter(path)
        writer.write([make_record()])
        with open(path) as f:
            header = f.readline()
        assert "method" in header
        assert "loss" in header

    def test_roundtrip_single_record(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        rec = make_record(method="abc", loss=3.14, step=7)
        writer = RecordWriter(path)
        writer.write([rec])
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["method"] == "abc"
        assert float(rows[0]["loss"]) == pytest.approx(3.14)
        assert int(rows[0]["step"]) == 7

    def test_append_mode_preserves_rows(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        writer = RecordWriter(path)
        writer.write([make_record(step=1)])
        writer.write([make_record(step=2)])
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        steps = {int(r["step"]) for r in rows}
        assert steps == {1, 2}

    def test_multiple_records_in_one_call(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        records = [make_record(step=i) for i in range(5)]
        writer = RecordWriter(path)
        writer.write(records)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5

    def test_params_flattened_to_columns(self, tmp_output_dir):
        """Each param key becomes param_<key> column."""
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        rec = make_record(params={"mu": 0.3, "sigma": 1.1})
        writer = RecordWriter(path)
        writer.write([rec])
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "param_mu" in rows[0]
        assert "param_sigma" in rows[0]
        assert float(rows[0]["param_mu"]) == pytest.approx(0.3)

    def test_none_weight_written_as_empty(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        writer = RecordWriter(path)
        writer.write([make_record(weight=None)])
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["weight"] == ""

    def test_csv_roundtrip_with_worker_events(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        rec = make_record(
            worker_id="rank_3",
            sim_start_time=1.2,
            sim_end_time=2.5,
            generation=2,
            record_kind="population_particle",
            time_semantics="generation_end",
            attempt_count=99,
        )
        writer = RecordWriter(path)
        writer.write([rec])
        with open(path) as f:
            row = next(csv.DictReader(f))
        loaded = ParticleRecord.from_csv_row(row)
        assert loaded.worker_id == "rank_3"
        assert loaded.sim_start_time == pytest.approx(1.2)
        assert loaded.sim_end_time == pytest.approx(2.5)
        assert loaded.generation == 2
        assert loaded.wall_time == pytest.approx(rec.wall_time)
        assert loaded.record_kind == "population_particle"
        assert loaded.time_semantics == "generation_end"
        assert loaded.attempt_count == 99

    def test_csv_roundtrip_with_none_worker_events(self, tmp_output_dir):
        tmp_output_dir.mkdir(parents=True)
        path = tmp_output_dir / "results.csv"
        rec = make_record(
            worker_id=None,
            sim_start_time=None,
            sim_end_time=None,
            generation=None,
        )
        writer = RecordWriter(path)
        writer.write([rec])
        with open(path) as f:
            row = next(csv.DictReader(f))
        loaded = ParticleRecord.from_csv_row(row)
        assert loaded.worker_id is None
        assert loaded.sim_start_time is None
        assert loaded.sim_end_time is None
        assert loaded.generation is None
        assert loaded.record_kind is None
        assert loaded.time_semantics is None
        assert loaded.attempt_count is None
