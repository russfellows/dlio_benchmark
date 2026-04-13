#!/usr/bin/env python3
"""
Real benchmark: JPEG/PNG generation throughput — fast path vs PIL encode path.

Tests exactly the two code paths that exist in the current codebase:
  - Fast path (non-DALI): records.tobytes() written directly
  - Slow path (DALI): full PIL image encode

Measures: files/sec, MB/s written, time per file.
No DLIO framework required — exercises the exact same logic directly.
"""
import os
import sys
import time
import shutil
import tempfile
import io
import numpy as np
import PIL.Image as im

# ── config ──────────────────────────────────────────────────────────────────
NUM_FILES    = 200          # files to generate per run
IMG_H, IMG_W = 224, 224    # standard ImageNet-like size (224x224 RGB)
CHANNELS     = 3
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

def gen_random_rgb(h, w, rng):
    a = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return a

# ── FAST PATH (current code for non-DALI) ───────────────────────────────────
def bench_fast_path_local(tmpdir, n=NUM_FILES):
    """Write raw bytes — no PIL encode. Same as current JPEGGenerator fast path."""
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    total_bytes = 0
    for i in range(n):
        records = gen_random_rgb(IMG_H, IMG_W, rng)
        raw = records.tobytes()
        path = os.path.join(tmpdir, f"f{i:05d}.jpg")
        with open(path, 'wb') as f:
            f.write(raw)
        total_bytes += len(raw)
    elapsed = time.perf_counter() - t0
    return elapsed, total_bytes

# ── SLOW PATH (PIL encode — what the old code did, what DALI path still does) ─
def bench_pil_jpeg(tmpdir, n=NUM_FILES):
    """Full PIL JPEG encode. What the OLD code did for all workloads."""
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    total_bytes = 0
    for i in range(n):
        records = gen_random_rgb(IMG_H, IMG_W, rng)
        buf = io.BytesIO()
        img = im.fromarray(records)
        img.save(buf, format='JPEG', quality=95)
        path = os.path.join(tmpdir, f"f{i:05d}.jpg")
        with open(path, 'wb') as f:
            f.write(buf.getvalue())
        total_bytes += len(buf.getvalue())
    elapsed = time.perf_counter() - t0
    return elapsed, total_bytes

def bench_pil_png(tmpdir, n=NUM_FILES):
    """Full PIL PNG encode."""
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    total_bytes = 0
    for i in range(n):
        records = gen_random_rgb(IMG_H, IMG_W, rng)
        buf = io.BytesIO()
        img = im.fromarray(records)
        img.save(buf, format='PNG')
        path = os.path.join(tmpdir, f"f{i:05d}.png")
        with open(path, 'wb') as f:
            f.write(buf.getvalue())
        total_bytes += len(buf.getvalue())
    elapsed = time.perf_counter() - t0
    return elapsed, total_bytes

# ── NPY generation ─────────────────────────────────────────────────────────
def bench_npy_fast(tmpdir, n=NUM_FILES):
    """NPY: save raw array (current fast path)."""
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    total_bytes = 0
    for i in range(n):
        arr = rng.random((IMG_H, IMG_W, CHANNELS), dtype=np.float32)
        path = os.path.join(tmpdir, f"f{i:05d}.npy")
        np.save(path, arr)
        total_bytes += arr.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, total_bytes

def fmt(elapsed, total_bytes, n):
    mb = total_bytes / 1e6
    return (f"{elapsed:.3f}s  |  {n/elapsed:.1f} files/s  |  "
            f"{mb/elapsed:.1f} MB/s  |  {elapsed/n*1000:.2f} ms/file  |  "
            f"{mb:.1f} MB total")

def extrapolate(elapsed, n, target=100_000):
    """Estimate time for target number of files."""
    secs = elapsed / n * target
    if secs < 60:
        return f"~{secs:.0f}s"
    elif secs < 3600:
        return f"~{secs/60:.1f} min"
    else:
        return f"~{secs/3600:.1f} hr"

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  DLIO Data Generator Benchmark — {NUM_FILES} files @ {IMG_H}x{IMG_W} RGB")
    print(f"{'='*70}\n")

    with tempfile.TemporaryDirectory(prefix="dlbench_gen_") as tmpdir:

        # JPEG fast path (current code, non-DALI)
        d = os.path.join(tmpdir, "jpeg_fast")
        os.makedirs(d)
        e, b = bench_fast_path_local(d, NUM_FILES)
        jpeg_fast_elapsed = e
        print(f"JPEG fast (raw bytes, NO PIL):  {fmt(e, b, NUM_FILES)}")
        print(f"  → ImageNet scale (1.28M):     {extrapolate(e, NUM_FILES, 1_280_000)}")

        # JPEG PIL (old code / DALI path)
        d = os.path.join(tmpdir, "jpeg_pil")
        os.makedirs(d)
        e, b = bench_pil_jpeg(d, NUM_FILES)
        jpeg_pil_elapsed = e
        print(f"\nJPEG PIL encode (OLD / DALI):   {fmt(e, b, NUM_FILES)}")
        print(f"  → ImageNet scale (1.28M):     {extrapolate(e, NUM_FILES, 1_280_000)}")

        speedup_jpeg = jpeg_pil_elapsed / jpeg_fast_elapsed
        print(f"\n  *** JPEG speedup (fast vs PIL): {speedup_jpeg:.0f}x ***")

        # PNG PIL (old code for PNG)
        d = os.path.join(tmpdir, "png_pil")
        os.makedirs(d)
        e, b = bench_pil_png(d, NUM_FILES)
        png_pil_elapsed = e
        print(f"\nPNG PIL encode (OLD code):      {fmt(e, b, NUM_FILES)}")
        print(f"  → 10k PNG files:              {extrapolate(e, NUM_FILES, 10_000)}")

        # PNG fast path
        d = os.path.join(tmpdir, "png_fast")
        os.makedirs(d)
        e, b = bench_fast_path_local(d, NUM_FILES)
        speedup_png = png_pil_elapsed / e
        print(f"\nPNG fast (raw bytes, NO PIL):   {fmt(e, b, NUM_FILES)}")
        print(f"  *** PNG speedup (fast vs PIL):  {speedup_png:.0f}x ***")

        # NPY
        d = os.path.join(tmpdir, "npy")
        os.makedirs(d)
        e, b = bench_npy_fast(d, NUM_FILES)
        print(f"\nNPY (np.save, always fast):     {fmt(e, b, NUM_FILES)}")

    print(f"\n{'='*70}\n")
