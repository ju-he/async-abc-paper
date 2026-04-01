# Fix: `_collective_stop_requested` allreduce serializes async workers

## Context

The test3 run (`/p/scratch/tissuetwin/herold2/async-abc/test3/`) reveals that
`async_propulate_abc` dispatches workers **in lock-step** rather than
asynchronously.  The Gantt data confirms this — all 8 workers start and finish
each generation at nearly identical timestamps (spread < 0.3 ms across workers
per generation), with uniform ~1.5 ms gaps between generations.  This is
**not** a measurement artefact; it is caused by a real synchronization barrier
in the code.

### Root cause

Both `straggler.json` and `runtime_heterogeneity.json` set
`"max_wall_time_s"` (300 s and 60 s respectively).  When this value is not
`None`, `run_propulate_abc` (line 364 of `propulate_abc.py`) takes the
`_propulate_with_wall_time_limit` path instead of calling
`propulator.propulate()`.

Inside `_propulate_with_wall_time_limit` (line 224-230), **every iteration**
calls:

```python
if _collective_stop_requested(propulator, run_start=..., max_wall_time_s=...):
    break
```

And `_collective_stop_requested` (line 175) does:

```python
propulate_comm.allreduce(bool(local_stop), op=MPI.LOR)
```

`MPI_Allreduce` is a **collective** operation — all ranks must enter it before
any can leave.  This turns every single generation into a global
synchronization point, completely negating Propulate's asynchronous SPMD
design.

### Why the native path is fine

Propulate's own `propulator.propulate()` (propulator.py lines 450-469) has
**no** collective calls in the hot loop — only `isend`/`iprobe` (non-blocking).
Workers race ahead independently.  Barriers only appear at startup (line 447)
and after all generations complete (lines 477-490).

### Evidence from Gantt data

**Straggler / async_propulate_abc / 20x slowdown** — all 8 workers execute
generation 0 at t≈0.00128s, gen 1 at t≈0.00297s, gen 2 at t≈0.00460s, etc.
The inter-generation gap (~1.5 ms) is dominated by the allreduce
round-trip, not simulation time (~0.15 ms).

In a truly async run the workers would desynchronize after the first
generation and the Gantt chart would show staggered, overlapping bars.

### Why test mode masks this partially

Test mode skips the artificial `time.sleep()` in both straggler and
runtime_heterogeneity runners.  With sub-millisecond simulations the allreduce
overhead is small in absolute terms, so all workers still appear "fast".  In a
full production run (with real sleep/simulation times), the allreduce will
force the fastest worker to wait for the slowest **every single iteration**,
destroying the straggler-resilience advantage the paper claims.

---

## Plan: Local time check + post-hoc result filtering

### Design rationale

On HPC, the idiomatic pattern is: every rank runs independently until the
job scheduler kills it.  We emulate this cleanly:

1. Each rank checks **its own clock** to decide when to exit the loop — no
   collective operations in the hot path.
2. Post-loop barriers still fire (needed for Propulate's population
   consistency), but since all ranks share the same `run_start` and
   `max_wall_time_s`, the barrier wait is bounded by at most one straggler
   evaluation — identical to what would happen if SLURM aborted the job.
3. Results are **filtered post-hoc**: any `ParticleRecord` whose
   `sim_end_time > max_wall_time_s` is discarded.  This makes the
   experimental semantics identical to "run until killed, only count work
   completed before the deadline."

### File to modify

`experiments/async_abc/inference/propulate_abc.py`

### Changes

#### 1. `_propulate_with_wall_time_limit` (lines 224-230) — local-only time check

Replace:
```python
if _collective_stop_requested(
    propulator, run_start=run_start, max_wall_time_s=max_wall_time_s,
):
    break
```
with:
```python
if (time.time() - float(run_start)) >= float(max_wall_time_s):
    break
```

#### 2. Same for the non-MPI fallback path (lines 204-210)

Replace the `_collective_stop_requested` call with the same local check.

#### 3. Post-hoc filter in `run_propulate_abc` (after line 427)

After the records list is built (line 427), filter out any record that
completed after the wall-time deadline:

```python
if max_wall_time_s is not None:
    records = [
        r for r in records
        if r.sim_end_time is None or r.sim_end_time <= max_wall_time_s
    ]
```

This is where `sim_end_time` is already computed as
`ind.evaltime - run_start` (relative, in seconds) — so comparing against
`max_wall_time_s` directly is correct.

#### 4. Remove or deprecate `_collective_stop_requested`

No remaining callers.  Either delete it or mark it with a deprecation
comment.  No other files in the codebase reference it.

#### 5. Keep post-loop barriers (lines 243-252) unchanged

These are necessary for Propulate's internal population consistency
(`_receive_intra_island_individuals`, `_dump_final_checkpoint`).  The wait
is bounded because all ranks exit the loop at roughly the same wall-time.

### Verification

1. **Unit tests**: existing tests in `test_inference.py` should still pass
   (they mock propulate and don't use real MPI).

2. **Gantt chart validation**: re-run `--test` for straggler and check that
   `worker_gantt_data.csv` shows **desynchronized** start times across
   workers.  In a correct async run, workers should show staggered,
   overlapping bars rather than the current aligned columns.

3. **Post-hoc filter correctness**: verify that records with
   `sim_end_time > max_wall_time_s` are absent from the returned list.

4. **Full integration**: run `--test` for straggler and runtime_heterogeneity:
   - Throughput should increase (no allreduce idle time)
   - Idle fraction for async should drop significantly
   - Plot audit still passes (valid posteriors, tolerance monotonicity)

---

## Notes on the other plots

Since this was a `--test` run (1 replicate, 100 max simulations, 2
generations, 30s wall time cap), the following are expected limitations:

- **Benchmark posteriors** (gaussian_mean, gandk, lotka_volterra,
  cellular_potts): posteriors are broad/undersampled with only 100 simulations
  — this is fine for pipeline validation, not for paper-quality inference.

- **SBC**: only 2 trials instead of 300 — rank histograms will be sparse and
  coverage estimates noisy.  Not meaningful for calibration assessment.

- **Ablation**: only `small_archive` (k=10) showed tolerance improvement
  (1.78 vs 5.0 initial).  Other variants stayed at initial tolerance — expected
  with a 100-simulation budget where most variants can't converge.

- **Sensitivity**: only 1 grid point evaluated — the heatmap is trivially a
  single cell.

- **Quality-vs-wall-time / quality-vs-budget**: curves are truncated and
  based on single replicates, so no CI bands.  Shape/trends can't be assessed.

All of these will resolve with a full production run.
