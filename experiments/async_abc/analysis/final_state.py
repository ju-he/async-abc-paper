"""Helpers for reconstructing a method's final observable posterior state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..io.records import ParticleRecord


@dataclass(frozen=True)
class FinalStateResult:
    """A reconstructed final posterior state for one method and replicate."""

    method: str
    replicate: int
    state_kind: str
    records: List[ParticleRecord]

    @property
    def n_particles_used(self) -> int:
        return len(self.records)


def base_method_name(method: str) -> str:
    """Strip sweep/checkpoint suffixes from a tagged method name."""
    return str(method).split("__", 1)[0]


def final_state_results(
    records: Iterable[ParticleRecord],
    *,
    archive_size: int | None = None,
) -> List[FinalStateResult]:
    """Return final posterior states grouped by method and replicate."""
    grouped: dict[tuple[str, int], list[ParticleRecord]] = {}
    for record in records:
        grouped.setdefault((record.method, int(record.replicate)), []).append(record)

    results: list[FinalStateResult] = []
    for (method, replicate), group in sorted(grouped.items(), key=lambda item: item[0]):
        state_kind, state_records = _final_state_for_group(
            group,
            archive_size=archive_size,
        )
        if not state_records:
            continue
        results.append(
            FinalStateResult(
                method=method,
                replicate=replicate,
                state_kind=state_kind,
                records=state_records,
            )
        )
    return results


def final_state_records(
    records: Iterable[ParticleRecord],
    *,
    archive_size: int | None = None,
) -> List[ParticleRecord]:
    """Return the pooled final posterior state across all method/replicate groups."""
    pooled: list[ParticleRecord] = []
    for result in final_state_results(records, archive_size=archive_size):
        pooled.extend(result.records)
    return pooled


def _final_state_for_group(
    records: list[ParticleRecord],
    *,
    archive_size: int | None,
) -> tuple[str, list[ParticleRecord]]:
    if not records:
        return "empty", []

    family = base_method_name(records[0].method)
    if family == "async_propulate_abc":
        return "archive_reconstruction", _async_archive_state(records, archive_size=archive_size)
    if family in {"abc_smc_baseline", "pyabc_smc"}:
        return "generation_population", _sync_generation_state(records)
    if family == "rejection_abc":
        return "accepted_prefix", _accepted_state(records, archive_size=archive_size)
    return "prefix", _generic_state(records, archive_size=archive_size)


def _async_archive_state(
    records: list[ParticleRecord],
    *,
    archive_size: int | None,
) -> list[ParticleRecord]:
    observed = [record for record in records if record.tolerance is not None]
    if not observed:
        return []

    epsilon_final = min(float(record.tolerance) for record in observed)
    archive = [
        record
        for record in observed
        if float(record.loss) < epsilon_final
    ]
    archive.sort(
        key=lambda record: (
            float(record.loss),
            float(record.wall_time),
            int(record.step),
        )
    )
    if archive_size is not None and archive_size > 0:
        archive = archive[: int(archive_size)]
    return archive


def _sync_generation_state(records: list[ParticleRecord]) -> list[ParticleRecord]:
    population_records = [
        record
        for record in records
        if record.record_kind in (None, "", "population_particle")
    ]
    if population_records:
        records = population_records
    generations = [record.generation for record in records if record.generation is not None]
    if generations:
        final_generation = max(int(generation) for generation in generations)
        state = [record for record in records if record.generation == final_generation]
    else:
        final_wall_time = max(float(record.wall_time) for record in records)
        state = [record for record in records if float(record.wall_time) == final_wall_time]
    state.sort(key=lambda record: (int(record.step), float(record.wall_time)))
    return state


def _accepted_state(
    records: list[ParticleRecord],
    *,
    archive_size: int | None,
) -> list[ParticleRecord]:
    state = sorted(records, key=lambda record: (float(record.wall_time), int(record.step)))
    if archive_size is not None and archive_size > 0:
        state = state[: int(archive_size)]
    return state


def _generic_state(
    records: list[ParticleRecord],
    *,
    archive_size: int | None,
) -> list[ParticleRecord]:
    state = sorted(records, key=lambda record: (float(record.wall_time), int(record.step)))
    if archive_size is not None and archive_size > 0:
        state = state[: int(archive_size)]
    return state
