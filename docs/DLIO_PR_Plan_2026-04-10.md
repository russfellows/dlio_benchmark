# DLIO Benchmark — PR Implementation Plan
**Date:** April 10, 2026  
**Branch base:** `russfellows/main` @ `f58903c`  
**Scope:** Correctness fixes, performance improvements, and DALI 2.0 modernization  
**Excluded:** Dynamic YAML file generation (Issue 7 from Executive Summary, per Caveat #1)

---

## Background

This plan was derived from two sources:

1. **[DLIO_IO_Issues-Executive_Summary_2026-03-28.md](DLIO_IO_Issues-Executive_Summary_2026-03-28.md)** — code quality and correctness analysis  
2. **[DLIO_IO_Issues-Proposal_2026-03-28.md](DLIO_IO_Issues-Proposal_2026-03-28.md)** — proposed improvements  
3. **Direct code review** of the current `russfellows/main` codebase, April 2026  
4. **DALI 2.0 analysis** — NVIDIA DALI 2.0.0 released March 2026; active development, Python 3.14 / CUDA 13.1 support, Dynamic Mode (`ndd`) as the new standard

All five core PRs plus three DALI PRs were validated against the current code state. **No code changes are made until this plan is reviewed and approved.**

---

## Pre-Work: Unique-Bytes Constraint (No PR Needed)

Before the PRs, one issue from the Executive Summary was investigated and found to be **already correctly handled**:

> **Issue:** Risk that generated files share content (enabling storage dedup to deflate benchmark numbers).

**Finding:** `data_generator.py` uses `np.random.default_rng(seed=BASE_SEED + my_rank)` per rank, with a flowing RNG that derives a unique `file_seed` per file from a 63-bit draw. Each rank has a different base seed; every file within a rank gets a unique derived seed. No two files share a content seed. No PR needed.

---

## Part 1: Core Correctness and Performance PRs (5 PRs)

---

### PR-1 — Bug: `build_sample_map_iter` file index reset for non-zero ranks

**Priority:** Critical (data correctness — silently corrupts multi-rank runs)  
**Files:** `dlio_benchmark/utils/config.py`  
**Issue reference:** Issue 3 (TFRecord iterative sampler bug) — but the bug is format-agnostic; it affects any workload using `data_loader_sampler: iterative`

#### The Bug

`build_sample_map_iter()` is called for TFRecord/any iterative-sampler workload. It computes a per-rank starting file offset correctly before the loop, but then overwrites `file_index` on the **first loop iteration** back to a value that ignores the rank offset:

```python
# Before loop — correct rank offset
file_index = self.my_rank * files_per_rank        # e.g. 2 for rank=1, 2 files/rank

# End of first iteration — OVERWRITES rank offset
file_index = (sample_index // self.num_samples_per_file) % num_files
#           = (1 // num_samples_per_file) % num_files
#           → 0  when num_samples_per_file > 1
```

For any rank > 0 with `num_samples_per_file > 1`, all subsequent samples are mapped to rank 0's file partition.  On a single-rank run this is silently correct. On multi-rank it causes every rank beyond rank 0 to read the same files as rank 0.

#### The Fix

Change the update expression at the end of the loop to carry the rank offset forward:

```python
# Before (broken):
file_index = (sample_index // self.num_samples_per_file) % num_files

# After (correct):
file_index = (self.my_rank * files_per_rank + sample_index // self.num_samples_per_file) % num_files
```

#### Tests

- Add a unit test for `build_sample_map_iter` with `comm_size > 1`, asserting that rank 1's file assignments differ from rank 0's and do not overlap.

---

### PR-2 — Correctness + Parity: Skip CPU decode AND add parallel prefetch to local-FS readers

**Priority:** Critical (cross-backend comparisons are invalid until fixed; parity gap in read concurrency)  
**Files:** `dlio_benchmark/reader/image_reader.py`, `npy_reader.py`, `hdf5_reader.py`, `npz_reader.py`, new `_local_fs_iterable_mixin.py`  
**Issue reference:** Issue 1 (file vs object reader asymmetry) + **newly identified read-path parity gap**

#### Problem 1: CPU decode in local-FS readers

The four local-FS readers decode file content into full numpy arrays on every `open()` call:

| Reader | Decode work |
|--------|-------------|
| `image_reader.py` | `np.asarray(Image.open(filename))` — full PIL JPEG/PNG decode |
| `npy_reader.py` | `np.load(filename)` — loads full array |
| `hdf5_reader.py` | h5py `open_file_map[filename][f'records_{idx}']` — full dataset read |
| `npz_reader.py` | `np.load(filename, allow_pickle=True)['x']` — full array load |

This means a local-FS benchmark measures `Storage I/O + CPU decode time` while the S3 iterable readers measure only `Storage I/O time`. Cross-backend comparisons are invalid.

#### Problem 2: Serial open/read loop vs S3 parallel prefetch (PARITY GAP)

This is a **hard parity violation** that must be fixed in this same PR. It was identified during the async-pipeline review requested by the user.

The S3 iterable readers use `_S3IterableMixin._s3_prefetch_all()`, which — before the iteration loop begins — **parallel-fetches ALL files** assigned to the current thread using `ThreadPoolExecutor` (or `s3dlio.get_many()` for high-concurrency parallel GETs). It stores only the raw byte count per file:

```python
# _s3_prefetch_all(): called ONCE before next() loop
executor = ThreadPoolExecutor(max_workers=min(64, len(obj_keys)))
futures = {executor.submit(fetch_one, key): key for key in obj_keys}
cache[key] = len(result_bytes)   # byte count only, data discarded
```

The local-FS readers (after naive fix of Problem 1) would use:

```python
# reader_handler.FormatReader.next(): serial loop
for global_sample_idx, filename, sample_index in self.file_map[self.thread_index]:
    self.open_file_map[filename] = self.open(filename)  # ← serial: one file at a time
```

This means:
- **S3**: All N files are read in parallel before the first sample is yielded. Storage queue depth = N (or `max_workers`). Full bandwidth utilization.
- **Local-FS**: Files are opened one at a time during the yield loop. Queue depth = 1. Bandwidth severely under-utilized on NVMe and NVMe-oF targets.

This asymmetry would make local-FS benchmarks look slower than they physically are — a false result. It is a parity violation.

#### The Fix (both problems together)

**Part A:** Create `dlio_benchmark/reader/_local_fs_iterable_mixin.py` mirroring `_S3IterableMixin`:
- `_localfs_prefetch_all()`: before the iteration loop, uses `ThreadPoolExecutor` to open and read ALL files assigned to this thread in parallel
- Stores only `len(raw_bytes)` in a dict cache (same as S3 mixin — full read, byte count only, data discarded)
- `queue_depth` defaults to `min(64, num_files)` — same ceiling as S3 mixin

```python
# _LocalFSIterableMixin (new)
def _localfs_prefetch_all(self) -> None:
    thread_entries = self.file_map.get(self.thread_index, [])
    unique_files = list(dict.fromkeys(f for _, f, _ in thread_entries))
    with ThreadPoolExecutor(max_workers=min(64, len(unique_files))) as ex:
        futures = {ex.submit(self._read_local_bytes, f): f for f in unique_files}
        for fut, path in futures.items():
            self._local_cache[path] = len(fut.result())  # byte count only

def _read_local_bytes(self, path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()
```

**Part B:** Apply `_LocalFSIterableMixin` to all four local-FS readers:
- `open()`: no-op (prefetch already done)
- `get_sample()`: look up `self._local_cache[filename]` for `byte_count`; record for telemetry; return `self._args.resized_image`
- Remove PIL, h5py, np.load decode calls entirely

**`tf_reader.py`:** Already returns `self._resized_image` from `_parse_image()`. No change needed.

#### Design Invariant

After this PR, local-FS and S3 iterable readers must follow **exactly the same structural pattern**:
1. Before `next()`: parallel-read all assigned files (ThreadPoolExecutor for local-FS; `get_many()` / ThreadPoolExecutor for S3)
2. Store byte count only
3. During `next()` / `get_sample()`: dict lookup → telemetry → return `resized_image`

This invariant must be maintained in all future reader changes.

#### Tests

- Existing reader tests should pass unchanged (they validate `image_size` telemetry, not content).
- **New parity test:** Run the same workload config against local-FS and S3 (pointing to same data on both). Assert that `image_size` telemetry matches and that timing difference is attributable only to physical bandwidth — not structural overhead differences.
- **Concurrency test:** Assert `_localfs_prefetch_all` issues at least `min(8, num_files)` concurrent reads (verify via mock or timing).

---

### PR-3 — Performance: JPEG/PNG generator raw-bytes fast path

**Priority:** High (2000–4000× speedup for synthetic dataset generation)  
**Files:** `dlio_benchmark/data_generator/jpeg_generator.py`, `png_generator.py`  
**Issue reference:** Issue 2 (JPEG/PNG generation overhead)

#### The Problem

Both generators do:
```python
records = gen_random_tensor(...)      # fast
records = np.clip(records, 0, 255).astype(np.uint8)
img = PIL.Image.fromarray(records)    # → overhead
img.save(output, format='JPEG')       # ~30ms/file for JPEG, ~100-200ms/file for PNG
```

The PIL encode step takes ~30 ms/file for JPEG and ~100–200 ms/file for PNG. For a 2000-file dataset this is 60 seconds (JPEG) or 3–6 minutes (PNG) in pure encode latency per rank. The encoded bytes are never actually decoded by the benchmark readers (after PR-2 they will just read raw bytes and measure the count).

#### The Fix

Since the benchmark only measures I/O throughput and the post-PR-2 readers discard file content anyway, the actual JPEG/PNG encoding is unnecessary for benchmarking purposes. Write the raw random bytes directly to the output without PIL encoding:

```python
def _write(i, dim_, dim1, dim2, file_seed, rng, out_path_spec, is_local, output):
    records = gen_random_tensor(shape=(dim1, dim2), dtype=np.uint8, rng=rng)
    output.write(records.tobytes())
```

**Caveat — DALI native readers parse file headers:** The `DaliImageReader` (used with `native_dali` loader) calls `fn.decoders.image()` which requires a valid JPEG/PNG bitstream. If synthetic data generated by these generators is used with the `native_dali` loader and real image decode, the raw-bytes shortcut will break DALI decode. Two options:

- **Option A (Recommended):** Add a config flag `fast_generation: true` (default for non-DALI workloads) that skips PIL encode. When `data_loader: native_dali`, keep the full PIL encode path.
- **Option B:** Always skip PIL encode and remove `fn.decoders.image()` from the native_dali readers (valid for benchmarking — see PR-6 below).

Option A maintains full backward compatibility. Option B is a stronger consistency fix but requires DALI PR-6 to land first.

#### Tests

- Benchmark test: time the generate phase before and after; assert significant speedup (>10×) for non-DALI config.
- Existing tests: re-run with `data_loader: native_dali` to confirm DALI path still works with PIL encode.

---

### PR-4 — Config: `multiprocessing_context` auto-derive from `storage_library`

**Priority:** High (prevents silent hangs on s3dlio/s3torchconnector setups)  
**Files:** `dlio_benchmark/utils/config.py`  
**Issue reference:** Issue 6 (`multiprocessing_context` must match storage_library)  
**Related:** Also promotes `storage_library` to a first-class field (Issue 7 — schema ergonomics)

#### The Problem

`multiprocessing_context` defaults to `"fork"` (line 146 of `config.py`). There is no logic anywhere in `derive_configurations()` to auto-set it based on `storage_library`.

Both `s3dlio` and `s3torchconnector` initialize CUDA/gRPC/TLS resources at module import time in the parent process. When the DataLoader spawns workers via `fork`, the child processes inherit those already-initialized file descriptors and OS-level resources, which leads to silent deadlocks or data corruption. The correct context for these libraries is `"spawn"`, which starts a clean child process.

Currently a user must know to add `reader.multiprocessing_context: spawn` to their YAML — an undocumented requirement that causes silent hangs.

#### The Fix

In `derive_configurations()`, after `storage_library` is resolved, auto-set `multiprocessing_context` to `"spawn"` if:
1. `storage_library` is `s3dlio` or `s3torchconnector`, AND
2. `multiprocessing_context` has not been explicitly set by the user (i.e. still holds the dataclass default `"fork"`)

Emit a `logger.info` message when doing so, so users can see the change in output.

Additionally, promote `storage_library` to a proper `Optional[str] = None` first-class field on `ConfigArguments`, with backward-compatible fallback read from `storage_options` dict. This makes `--param workload.reader.storage_library=s3dlio` work as a direct override path.

#### Tests

- Unit test: with `storage_library=s3dlio` and no explicit `multiprocessing_context`, assert `args.multiprocessing_context == "spawn"` after `derive_configurations()`.
- Unit test: with explicit `multiprocessing_context: fork` in YAML (user override), assert the override is respected even with `s3dlio`.

---

### PR-5 — Config: `read_threads` auto-sizing

**Priority:** Medium (prevents leaving I/O bandwidth on the table with high-throughput storage)  
**Files:** `dlio_benchmark/utils/config.py`  
**Issue reference:** Issue 4 (`read_threads` hardcoded at 1)

#### The Problem

`read_threads: int = 1` is the dataclass default, and there is no auto-sizing logic. The existing code emits a warning if `read_threads` exceeds available cores (a correct defensive check) but never sizes upward. With modern Gen5/Gen6 NVMe drives capable of 10–14 GB/s, a single I/O thread is typically the bottleneck long before the storage is saturated.

#### The Fix

When `read_threads == 1` (the sentinel "user didn't set this" value), auto-size using:

```python
import os, math
cpu_count = os.cpu_count() or 1
per_rank_cpu = max(1, cpu_count // self.comm_size)
auto_threads = min(per_rank_cpu, MAX_AUTO_READ_THREADS)  # MAX_AUTO_READ_THREADS = 8
```

Emit a `logger.info` message indicating auto-sizing was applied.  
User-explicit values (any value > 1 in the YAML) are respected as-is with no auto-sizing.

**Conservative default:** `MAX_AUTO_READ_THREADS = 8`. This is intentionally modest — the goal is to avoid leaving obvious throughput on the table, not to compute a theoretically optimal value.

#### Tests

- Unit test: with default `read_threads=1` and `comm_size=1` on a machine with ≥8 cores, assert auto-sized value >= 2.
- Unit test: with explicit `read_threads=16` in YAML, assert no auto-sizing is applied.

---

## Part 2: DALI Modernization PRs (3 PRs)

**Context:** DALI 2.0.0 was released March 2026. It introduces the Dynamic Mode executor (`exec_dynamic=True`), No-GIL support, and improved `fn.readers` C++ performance for 10+ GB/s storage. The current DLIO DALI integration uses the legacy `Pipeline` static graph API with several correctness and performance issues.

**DALI CPU-only note:** DALI does not require a physical GPU. Running with `device_id=None` on all operators uses DALI's full C++ multi-threaded I/O path on CPU. This is already the pattern in the current DALI loaders. DALI 2.0 improves CPU-mode performance significantly for `fn.readers` operators.

---

### PR-6 — DALI Bug: Missing `shard_id` in all `fn.readers.*` calls

**Priority:** Critical (all DALI multi-rank runs currently read the same data partition)  
**Files:** `dlio_benchmark/reader/dali_image_reader.py`, `dali_npy_reader.py`, `dali_tfrecord_reader.py`

#### The Bug

Every `fn.readers.*` call sets `num_shards=self._args.comm_size` but **never passes `shard_id`**:

```python
# dali_image_reader.py
images, labels = fn.readers.file(
    files=self._file_list,
    num_shards=self._args.comm_size,   # ← correct
    # shard_id=???                     # ← MISSING — defaults to 0
    ...
)

# dali_npy_reader.py
dataset = fn.readers.numpy(
    files=self._file_list,
    num_shards=self._args.comm_size,   # ← correct
    # shard_id=???                     # ← MISSING — defaults to 0
    ...
)

# dali_tfrecord_reader.py
dataset = fn.readers.tfrecord(
    path=self._file_list,
    num_shards=self._args.comm_size,   # ← correct
    # shard_id=???                     # ← MISSING — defaults to 0
    ...
)
```

With `shard_id` defaulting to 0 on every rank, **all ranks read partition 0** — the same data as rank 0 — instead of their assigned file shard. In a 4-rank run, ranks 1/2/3 all read rank 0's files, and no rank reads shards 1/2/3 at all. This is a critical multi-rank correctness bug for all `native_dali` workloads.

#### The Fix

Add `shard_id=self._args.my_rank` to all three `fn.readers.*` calls:

```python
fn.readers.file(
    files=self._file_list,
    num_shards=self._args.comm_size,
    shard_id=self._args.my_rank,       # ← add this
    ...
)
```

Same one-line fix for `dali_npy_reader.py` and `dali_tfrecord_reader.py`.

#### Tests

- Multi-rank test (comm_size=2): assert rank 0 and rank 1 produce different file lists in their respective pipelines.
- Confirm single-rank runs are unaffected.

---

### PR-7 — DALI Performance: Remove `fn.python_function` decode bypass

**Priority:** High (currently forces all DALI decode work through Python GIL, then discards it)  
**Files:** `dlio_benchmark/reader/dali_image_reader.py`, `dali_npy_reader.py`  
**Dependency:** Should land after PR-2 (local-FS reader skip-decode) for conceptual consistency

#### The Problem

The native DALI reader pipelines insert `fn.python_function` callbacks after the C++ decode step:

```python
# dali_image_reader.py (current)
images = fn.decoders.image(images, device='cpu')           # C++ JPEG decode: expensive
images = fn.python_function(images, function=self.preprocess, num_outputs=1)  # GIL!
dataset = fn.python_function(images, function=self.resize, num_outputs=1)     # GIL!
```

`preprocess()` in the base class just sleeps for `preprocess_time` (default 0 s) then returns the input unchanged. `resize()` in the base class ignores the input entirely and returns `self._args.resized_image` — a pre-built dummy numpy array. So the full pipeline is:

1. DALI reads file in C++ — ✓ fast
2. DALI decodes JPEG in C++ — ✓ fast but **unnecessary**: the result is discarded
3. Python GIL callback (`preprocess`) — serializes the pipeline; sleeps 0 s; returns dummy array
4. Python GIL callback (`resize`) — serializes the pipeline; ignores input; returns `resized_image`

Steps 2–4 together mean: DALI does expensive C++ JPEG decode work, then a Python callback discards it and substitutes a pre-made dummy. This defeats the purpose of DALI's C++ threading. The `fn.python_function` callbacks serialize the entire pipeline through the GIL, eliminating DALI's parallel C++ execution model.

For storage benchmarking (where I/O bandwidth is the metric, not decode throughput), the correct pipeline is to read raw bytes and report their size, without decoding at all — exactly what the S3 iterable readers already do.

#### The Fix

For `DaliImageReader.pipeline()` (used with `native_dali` loader):
- Remove `fn.decoders.image()` — keep images as raw bytes from `fn.readers.file`
- Remove both `fn.python_function` calls
- Return the raw byte tensors directly; telemetry for `image_size` comes from batch byte counts

For `DaliNPYReader.pipeline()`:
- Remove both `fn.python_function` calls
- Return the `fn.readers.numpy` output directly — numpy tensors are already correctly shaped

For `DaliTFRecordReader.pipeline()`:
- Already removed the `fn.python_function` calls (they are commented out) — no change needed

**Handling `preprocess_time > 0`:** When users configure a non-zero `preprocess_time` to simulate compute overhead, the sleep must still occur. Implement this with a lightweight DALI-compatible threading hook outside the pipeline, or via a single `fn.python_function` that only sleeps (no decode/discard). This is only needed if `preprocess_time > 0` (rare for pure storage benchmarks).

**Relationship to PR-3:** After PR-3 makes JPEG/PNG generation write raw bytes instead of valid JPEG/PNG bitstreams, the `fn.decoders.image()` call would fail even if kept. PR-7 cleanly removes that dependency.

#### Tests

- Throughput test: `native_dali` with NPY files; assert pipeline completes in < 2× baseline time (previously GIL serialization added significant overhead).
- Correctness test: telemetry `image_size` is non-zero and matches the expected byte count per file.

---

### PR-8 — DALI Modernization: Migrate to DALI 2.0 dynamic executor

**Priority:** Medium (performance improvement and forward compatibility)  
**Files:** `dlio_benchmark/data_loader/dali_data_loader.py`, `native_dali_data_loader.py`  
**Dependency:** Cosmetically independent from PRs 6/7, but should be sequenced last for DALI changes

#### The Problem

Both DALI data loaders use the legacy `Pipeline` static graph executor:

- `dali_data_loader.py` uses the lowest-level executor API:
  ```python
  Pipeline(batch_size=..., exec_async=True, ...)
  pipe.start_py_workers()
  pipe.build()
  pipe.schedule_run()
  # ... per step:
  outputs = pipe.share_outputs()
  pipe.release_outputs()
  pipe.schedule_run()
  ```
  This manual `schedule_run/share_outputs/release_outputs` loop is the legacy DALI 1.x protocol.

- `native_dali_data_loader.py` uses:
  ```python
  Pipeline(batch_size=..., exec_async=True, exec_pipelined=True, ...)
  DALIGenericIterator(self.pipelines, ['data'], auto_reset=True)
  ```
  `exec_pipelined=True` is the legacy pipelining flag.

In DALI 2.0, the new dynamic executor is activated with `exec_dynamic=True` and replaces both `exec_async` and `exec_pipelined`. It delivers better throughput through improved internal scheduling and is the only path to No-GIL support in Python 3.13t/3.14.

Additionally, `py_start_method=self._args.multiprocessing_context` in both loaders ties DALI's Python worker subprocess model to the same `multiprocessing_context` setting. After PR-4 auto-sets this to `"spawn"` for s3dlio, DALI workers will also launch with `spawn` — correct but slow for first-batch startup. For `device_id=None` (CPU-only) pipelines, `py_num_workers` can often be set to 0 since `fn.readers.*` handle threading in C++ without needing Python sub-workers.

#### The Fix

1. **Detect DALI version** at startup using `nvidia.dali.__version__` and choose executor mode:
   ```python
   import nvidia.dali
   _DALI_2 = tuple(int(x) for x in nvidia.dali.__version__.split('.')[:2]) >= (2, 0)
   ```

2. **`NativeDaliDataLoader`:** Replace `exec_async=True, exec_pipelined=True` with `exec_dynamic=True` when `_DALI_2`. Fall back to the legacy params on older DALI.

3. **`DaliDataLoader`:** Replace the manual `schedule_run/share_outputs/release_outputs` loop with a `DALIGenericIterator` (already used by `NativeDaliDataLoader`), which is compatible with both the legacy and dynamic executors. This simplifies the code significantly.

4. **`py_num_workers` for CPU pipelines:** When `device_id=None` and all readers use `fn.readers.*` (no `fn.python_function`), set `py_num_workers=0` to avoid unnecessary Python subprocess creation. Reader-level threading is handled inside DALI's C++ thread pool (`num_threads` parameter).

#### Notes on DALI 2.0 Dynamic Mode (`ndd`)

The user analysis mentions `nvidia.dali.ndd`. This is the new **Pipeline-as-function** API introduced in DALI 2.0 where pipelines are defined as plain Python functions rather than context managers. While it is the forward-looking API, migrating the DLIO pipeline structure to it is a more invasive refactor (the `pipeline()` method on readers returns a node graph built inside a `with pipeline:` block). Using `exec_dynamic=True` with the existing `Pipeline` API achieves the same executor benefits without requiring a full API migration. A full `ndd`-based rewrite can be a follow-on after the current reader structure is stabilized.

#### Tests

- Existing DALI tests must pass with both DALI 1.x and DALI 2.0+ installed.
- DALI 2.0: assert `exec_dynamic=True` is passed to `Pipeline` when DALI >= 2.0.
- Throughput: compare `native_dali` NPY throughput before/after on the same hardware.

---

## Summary Table

| PR | Area | Priority | Files Changed | Issue Ref |
|----|------|----------|---------------|-----------|
| PR-1 | Bug: iterative sampler file index reset | **Critical** | `config.py` | Issue 3 |
| PR-2 | Correctness: local-FS readers skip decode | **Critical** | 4 reader files | Issue 1 |
| PR-3 | Performance: JPEG/PNG fast generation | High | 2 generators | Issue 2 |
| PR-4 | Config: `multiprocessing_context` auto-derive | High | `config.py` | Issues 6+7 |
| PR-5 | Config: `read_threads` auto-sizing | Medium | `config.py` | Issue 4 |
| PR-6 | DALI Bug: missing `shard_id` | **Critical** | 3 DALI readers | DALI correctness |
| PR-7 | DALI Performance: remove GIL decode bypass | High | 2 DALI readers | DALI perf |
| PR-8 | DALI Modernization: dynamic executor | Medium | 2 DALI loaders | DALI 2.0 |

---

## Sequencing and Dependencies

```
PR-1  ──────────────────────────────────────────────► (standalone)
PR-2  ──────────────────────────────────────────────► (standalone)
PR-3  ─────────────────────────── depends on PR-2 recommended first ─► PR-3
PR-4  ──────────────────────────────────────────────► (standalone; enables PR-8 auto-spawn)
PR-5  ──────────────────────────────────────────────► (standalone)
PR-6  ──────────────────────────────────────────────► (standalone DALI bug fix)
PR-7  ─────────────────────────── depends on PR-6 recommended first ─► PR-7
PR-8  ─────────────────────────── depends on PR-6, PR-7 recommended first ─► PR-8
```

**Recommended merge order:** PR-1 → PR-6 → PR-2 → PR-3 → PR-4 → PR-7 → PR-5 → PR-8

Rationale: correctness bugs (PR-1 and PR-6) first, then the local-FS reader fix (PR-2) before the generator shortcut (PR-3) since PR-3's Option A/B choice depends on the PR-2 state.

---

## Deferred: Lower-Priority Improvements (Not in Current Plan)

The following issues from the Executive Summary are deferred to a later phase:

- **Issue 8** — No intra-rank parallelism for data generation (ThreadPoolExecutor for gen loop)
- **Issue 9** — No async pipeline for object store upload — **see hard parity constraint below**
- **Issue 10** — MPI topology not used for resource planning
- **Issue 11** — No settle-time guard after generation on eventual-consistency systems
- **DALI `ndd`** — Full migration to DALI 2.0 Pipeline-as-function API (follow-on after PR-8)

### ⚠️ Hard Parity Constraint on Issue 9 (Async Upload Pipeline)

During the PR plan review (April 10, 2026), the user raised a fundamental requirement: **file storage and object storage must have parity — we cannot be more efficient for one interface than the other.**

This constraint has direct implications for Issue 9.

#### Current state (both paths are equivalent)

The `_generate_files()` base generator loop is **identically serial for both storage types**:

- **Local-FS:** `write_fn(...)` writes directly to disk path (synchronous `open+write` inside the generator)
- **Object store:** `write_fn(...)` writes to `io.BytesIO()`, then `storage.put_data(path, buf.getvalue())` uploads synchronously

Both are one-file-at-a-time serial loops. No parity gap exists today.

#### What "Issue 9" would do — and why it requires matching local-FS work

Issue 9 proposes an async pipeline for object store uploads: while file *N* is being uploaded to S3, file *N+1* is being generated. This pipeline overlaps CPU work (generation) with network I/O (upload), significantly reducing wall-clock generation time for large datasets.

**If this is implemented for object store only, it is a parity violation.** Object store generation becomes faster than local-FS generation through a structural advantage, not a physical one. Any benchmark comparing pre-generation time across storage types would be skewed.

The correct implementation when Issue 9 is addressed:

> **Both local-FS and object-store write paths must be parallelized simultaneously, using the same `ThreadPoolExecutor` model in `_generate_files()`.** For local-FS this means parallel `open+write` workers. For object store this means parallel `BytesIO+put_data` workers. Both use the same `max_workers` cap and same `ThreadPoolExecutor` structure.

This is a **non-negotiable requirement**. Issue 9 must not be implemented for object store in isolation. It is only acceptable as a joint change to both paths.

#### Read-path parity gap (addressed in PR-2)

A parallel read-path parity gap was also identified and has been **resolved in PR-2** (above): local-FS readers now get `_LocalFSIterableMixin` parallel prefetch matching `_S3IterableMixin`. Both storage types will issue concurrent reads with the same queue depth model before any sample is yielded.

---

*This document records the approved plan. No code changes are made until the user approves.*
