# Bug Hunt 2026-04-07: 48-rank MPI teardown hang in abc_smc_baseline

## Timeline

### test10 (pre-fix: concurrent_futures default)

Both scaling jobs used `pyabc_mpi_sampler=concurrent_futures`.

**scaling_48** (job 13667547):
- `async_propulate_abc`: completed 2 workloads (k=10 at 41s/55k records, k=100 at 33s/42k records)
- `abc_smc_baseline` workload 1: rank 0 finished at 2.9s, workers took 40.5s (37s teardown)
- `abc_smc_baseline` workload 2: rank 0 finished at 4.7s, **workers never logged finish**
- Output stopped at 19:17 — hung for 7+ min, never recovered
- Only completed 2 of 3 test k-values

**scaling_bundle_1_4** (job 13667546):
- k=1 serial srun: completed all 3 workloads (including 292s baseline for k=1000)
- k=4 srun: completed 2 workloads; 3rd (k=1000) abc_smc_baseline ran slowly but completed
- Workers using `concurrent_futures` at 4 ranks: 16–21s teardown delay per baseline, but no hang

### Fix applied: switch default to `mapping`

Changed `resolve_pyabc_mpi_sampler()` default from `concurrent_futures` to `mapping` (commit pending).

### test11 (post-fix: mapping default)

Both jobs now show `pyabc_mpi_sampler=mapping n_workers=48 client_max_jobs=ignored`.

**scaling_bundle_1_4** (job 13667586): progressing cleanly. All workloads completing, workers logging finish with short teardown times. The mapping sampler works correctly at 4 ranks.

**scaling_48** (job 13667587): **hung again**.
- `async_propulate_abc` workload 1: completed (39.6s, 55k records)
- `abc_smc_baseline` workload 1: rank 0 finished at 2.3s, workers finished at 40.5s (ok but slow)
- `async_propulate_abc` workload 2: completed (33s, 42k records)
- `abc_smc_baseline` workload 2: rank 0 finished at 2.3s, last log line:
  ```
  [abc_smc_baseline] abc.run returned under mapping MPI sampler; exiting MPICommExecutor context
  ```
- Output stopped at 19:47 — **no output for 5+ min**, file unchanged at 874 lines

**Conclusion:** The hang is NOT in the sampler. Both `mapping` and `concurrent_futures` hang at 48 ranks. The problem is in `MPICommExecutor.__exit__()` itself.

---

## Root Cause Analysis: mpi4py `MPICommExecutor` teardown

### Source files examined

- `/home/juhe/bwSyncShare/Code/mirrors/nastjapy_copy/.venv/lib/python3.11/site-packages/mpi4py/futures/pool.py`
- `/home/juhe/bwSyncShare/Code/mirrors/nastjapy_copy/.venv/lib/python3.11/site-packages/mpi4py/futures/_core.py`

All pure Python (MPI ops use C extensions in `mpi4py/MPI.*.so`).

### `MPICommExecutor.__enter__` flow (pool.py:402–417)

```python
def __enter__(self):
    comm = self._comm       # COMM_WORLD
    root = self._root       # 0
    executor = None
    if comm.Get_rank() == root:
        executor = MPIPoolExecutor(**options)
    _comm_executor_helper(executor, comm, root)
    self._executor = executor
    return executor
```

`_comm_executor_helper` (_core.py:534–555):

**Root (rank 0):**
1. Creates `Pool(executor, _manager, comm, root)` — spawns a manager **thread**
2. Manager thread calls `comm_split(comm, root)` → `Create_intercomm` (collective with workers)
3. Manager thread enters `_manager_comm()` → `client_exec()` — loops processing work from queue

**Workers (ranks 1–47):**
1. Call `comm_split(comm, root)` → `Create_intercomm` (collective with root's thread)
2. Enter `server_main_comm(comm)` → `server_exec(comm, options)` — **block** in recv loop
3. Workers remain blocked in `__enter__` until root sends shutdown signal

### `MPICommExecutor.__exit__` flow (pool.py:419–428)

```python
def __exit__(self, *args):
    executor = self._executor
    self._executor = None
    if executor is not None:          # Root
        executor.shutdown(wait=True)  # Blocks until manager thread finishes
        return False
    else:                             # Workers
        return True                   # Workers already exited __enter__
```

**Root's `executor.shutdown(wait=True)`:**
1. Puts `None` in the work queue → manager thread's `client_exec()` loop breaks
2. Manager thread calls `client_stop(comm)` (_core.py:790–792):
   ```python
   def client_stop(comm):
       serialized(sendtoall)(comm, None)   # 47 individual issend calls
       serialized(disconnect)(comm)         # comm.Disconnect() — COLLECTIVE
   ```
3. `pool.join()` waits for manager thread to finish

**Workers:**
1. Receive `None` from `sendtoall` → `server_exec()` recv loop breaks
2. Call `server_stop(comm)` → `disconnect(comm)` → `comm.Disconnect()` — COLLECTIVE
3. Return from `server_main_comm()` → `__enter__` finally returns `None`
4. `with` body runs (nothing since executor is None)
5. `__exit__` returns `True`

### The hang mechanism

The `disconnect(comm)` call (`comm.Disconnect()`) is a **collective** operation on the inter-communicator. All ranks (root's manager thread + all 47 workers) must call it. Additionally, `sendtoall` uses `issend` (synchronous send) which completes only when the receiver posts a matching `recv`.

**At 48 ranks on ParaStation MPI:**
- `sendtoall` issues 47 individual `issend(None)` calls then `waitall()` (_core.py:643–659)
- Workers are in `server_exec` recv loop and should receive the `None` sentinel
- After receiving, workers call `server_stop()` → `comm.Disconnect()`
- Root's manager thread also calls `comm.Disconnect()`
- **`Disconnect()` deadlocks** — this is a known fragility of `Create_intercomm`/`Disconnect` on ParaStation MPI at scale

**Evidence:**
- `submit_scaling.py` documents `MPI_EXECUTOR_OVERHEAD_S = 60` — measured ~50s for `Create_intercomm + Disconnect` at 48 workers in the **healthy** case
- At 4 ranks, teardown completes in 16–21s
- At 48 ranks, teardown sometimes completes (40.5s on workload 1) but sometimes hangs indefinitely (workload 2)
- The non-deterministic hang suggests a race condition in ParaStation MPI's `Disconnect` implementation at scale

### Key mpi4py source details

**`comm_split`** (_core.py:498–528): Creates inter-communicator via `intracomm.Create_intercomm(local_leader, comm, remote_leader, tag=0)`. Root gets a 1-process intracomm, workers get a 47-process intracomm.

**`sendtoall`** (_core.py:643–664):
```python
def isendtoall(comm, data, tag=0):
    size = comm.Get_remote_size()
    return [comm.issend(data, pid, tag) for pid in range(size)]

def sendtoall(comm, data, tag=0):
    requests = isendtoall(comm, data, tag)
    waitall(comm, requests)
```

**`disconnect`** (_core.py:672–676):
```python
def disconnect(comm):
    try:
        comm.Disconnect()
    except NotImplementedError:
        comm.Free()
```

---

## Fix Plan: Bypass MPICommExecutor for mapping sampler

### Approach

Replace `MPICommExecutor` with a simple `COMM_WORLD`-based parallel map for the `mapping` path. No inter-communicator is created, so no `Create_intercomm` or `Disconnect` — just `bcast`/`send`/`recv` on `COMM_WORLD`.

### New component: `CommWorldMap` in `pyabc_sampler.py`

A class providing a `map(func, iterable)` callable for `pyabc.MappingSampler(map_=...)`.

**Root side (`__call__`):**
1. `bcast` the function to all workers
2. Distribute work items to workers using `send()` (dynamic: send to idle workers for load balance)
3. Send `None` sentinels to mark end of batch
4. Receive results from workers in submission order
5. Return combined results

**Worker side (`worker_loop`, static method):**
1. Loop: receive function via `bcast`
2. If function is `None` → shutdown, break
3. Loop: receive work items via `recv()` from root
4. If item is sentinel → done with batch, send results back
5. Repeat

**Shutdown (`close`):**
- `bcast(None)` → workers break out of loop
- No inter-communicator, no `Disconnect`, no hang

### Code sketch

```python
class CommWorldMap:
    """Blocking map() over COMM_WORLD using point-to-point messaging.

    Avoids MPICommExecutor and its inter-communicator lifecycle
    (Create_intercomm + Disconnect) which hangs on ParaStation MPI
    at high rank counts.
    """

    _WORK_TAG = 42
    _RESULT_TAG = 43

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.n_workers = self.size - 1  # rank 0 is coordinator

    def __call__(self, func, iterable):
        """map(func, iterable) distributing work across MPI ranks."""
        items = list(iterable)
        if not items:
            return []

        # Broadcast function to workers
        self.comm.bcast(func, root=0)

        # Distribute items to workers and collect results
        # (dynamic work distribution for load balance)
        ...

        return results

    def close(self):
        """Signal workers to exit."""
        self.comm.bcast(None, root=0)

    @staticmethod
    def worker_loop(comm):
        """Worker event loop — call on ranks != 0."""
        while True:
            func = comm.bcast(None, root=0)
            if func is None:
                break
            # receive and process work items
            ...
```

### Changes to `abc_smc_baseline.py` and `pyabc_wrapper.py`

Replace the `with MPICommExecutor(...)` block for the `mapping` path:

```python
if mpi_sampler == "mapping":
    if rank == 0:
        comm_map = CommWorldMap(MPI.COMM_WORLD)
        sampler = build_pyabc_sampler(..., mpi_map=comm_map, ...)
        result = _run_abc_smc_baseline_with_sampler(sampler=sampler, ...)
        comm_map.close()
    else:
        CommWorldMap.worker_loop(MPI.COMM_WORLD)
        result = []
    MPI.COMM_WORLD.Barrier()
else:
    # concurrent_futures path: keep MPICommExecutor as-is (opt-in only)
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        ...
```

### Design decisions

- **Root (rank 0) does NOT compute** — matches current `MPICommExecutor` behavior where root is coordinator only. Keeps `n_workers` semantics consistent.
- **Dynamic work distribution** — send items one-at-a-time to idle workers for better load balance (pyABC's acceptance sampling has variable runtime per particle).
- **`bcast` for function** — ensures all workers have the callable before work starts.
- **Point-to-point for items/results** — `send`/`recv` for individual work items and results.
- **No inter-communicator** — uses `COMM_WORLD` directly, avoiding the entire `Create_intercomm`/`Disconnect` lifecycle.

### Files to modify

1. `experiments/async_abc/inference/pyabc_sampler.py` — add `CommWorldMap` class
2. `experiments/async_abc/inference/abc_smc_baseline.py` — split MPI path by sampler type
3. `experiments/async_abc/inference/pyabc_wrapper.py` — same split
4. `experiments/tests/test_inference.py` — update tests for new code path
5. `experiments/tests/mpi_integration_helper.py` — update for new mapping path
6. `experiments/tests/mpi_abc_smc_baseline_helper.py` — update for new mapping path
7. `.plans/bug-fixes/previous-fixes.md` — document the fix

### Verification

1. Run unit tests: `pytest tests/test_inference.py -x -v`
2. MPI integration tests (if mpirun available): mapping-default tests should exercise the new path
3. Cluster validation: rerun scaling_48 to confirm no hang

---

## Actual fix applied (simpler approach)

The `CommWorldMap` plan above was **not implemented**. A simpler restructure was used instead.

### test12 (shared executor + run_method fix)

**Approach:** Share one `MPICommExecutor` across all baseline workloads per `n_workers` value. Non-MPI methods run first, then all pyABC baselines execute under the shared executor. One `Create_intercomm` + one `Disconnect` total.

**scaling_bundle_1_4** (job 13667682): completed successfully in 7m 27s. All k-values and methods completed without issues.

**scaling_48** (job 13667683): **hung after first abc_smc_baseline k=10 completed** (19.7s elapsed, 939 records). Output stopped at 765 lines with no further growth.

**Root cause of test12 hang:** `_run_workloads` called `run_method_distributed` even with the shared executor. For `all_ranks` mode, `run_method_distributed` ends with `allgather(error_payload)` (runner.py:876) expecting all 48 ranks to participate. But workers were trapped in `MPICommExecutor`'s server recv loop — they never reached `allgather`, so root hung waiting for them.

**Fix:** When `mpi_executor` is provided, call `run_method` directly on root instead of `run_method_distributed`. Workers process work internally via the executor's dispatch — no `allgather` coordination needed at the user-code level.

### test13 (small1 run — revealed broader problem)

The scaling runner `allgather` fix worked (scaling_48 progressed through `async_propulate_abc` k-values). But two non-scaling experiments hung:

- **straggler shard_000** (16 ranks): completed 3 `abc_smc_baseline` cycles, hung after 3rd teardown — repeated `MPICommExecutor` cycles
- **lotka_volterra shard_001** (48 ranks): completed 1 `pyabc_smc` call, hung during single `MPICommExecutor.__exit__` teardown

These experiments use `run_experiment()` / `run_method_distributed()` which creates a new `MPICommExecutor` per method call. The shared executor fix only applies to the scaling runner.

### CommWorldMap fix (implemented)

Replaced the self-managed `MPICommExecutor` path with `CommWorldMap` — a custom `COMM_WORLD`-based parallel map using `bcast`/`send`/`recv`. No inter-communicators, no `Create_intercomm`, no `Disconnect`.

All ranks enter and exit `CommWorldMap` within their `run_method` call, so `allgather` in `run_method_distributed` works unchanged. No runner modifications needed.

**Files changed:** `pyabc_sampler.py` (new class), `abc_smc_baseline.py`, `pyabc_wrapper.py`, `test_inference.py`

### test14 (pending)

Awaiting cluster validation with `CommWorldMap`.
