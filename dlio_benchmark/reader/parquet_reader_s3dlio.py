"""
Parquet reader backed by s3dlio's ParquetRowGroupDataset.

Supports BOTH object storage (s3://) and local filesystem (file://) paths
through s3dlio's unified URI scheme.  The same code path runs in both cases;
the only difference is the URI prefix constructed from the DLIO config.

Architecture
------------
s3dlio.create_async_loader(uri_prefix, {"format": "parquet", "prefetch": N})
returns a PyBytesAsyncDataLoader whose __iter__ drives up to N concurrent
row-group fetches via Tokio's buffer_unordered.  The bounded channel provides
natural backpressure — no manual thread-count tuning required.

One loader is created per file in open().  Each call to get_sample() fetches
the next row-group extent from the loader (1 I/O per row group, not per
sample), stores the compressed byte count for dlp telemetry, and discards the
data immediately.  DLIO's resized_image tensor is returned — this is a storage
benchmark, not a training framework.

Per-file row-group count drives DLIO's sample accounting: num_samples_per_file
in the YAML must equal the total number of rows across all row groups in each
file.  The reader itself does not validate this (footer parsing happens in Rust).

Configuration (storage_options in the DLIO YAML)
-------------------------------------------------
  storage_library:  s3dlio          # required — selects this reader
  endpoint_url:     http://...      # S3 endpoint; also via AWS_ENDPOINT_URL_S3
  prefetch:         16              # row groups to buffer (default 16)
  footer_cap:       4194304         # footer window bytes (default 4 MiB)
  columns:          null            # list[int] column indices, null = all

Example YAML (S3)
-----------------
  dataset:
    format: parquet
    storage_type: s3
    storage_root: mlp-flux
    data_folder: data/dlrm/train
    storage_options:
      storage_library: s3dlio
      endpoint_url: http://127.0.0.1:9200
      prefetch: 16

Example YAML (file)
-------------------
  dataset:
    format: parquet
    storage_type: local
    data_folder: /mnt/nvme/dlrm/train
    storage_options:
      storage_library: s3dlio   # use this reader for file paths too
      prefetch: 8
"""
import bisect
import os

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import ReadType as _ReadType, StorageType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ParquetReaderS3dlio(FormatReader):
    """
    Row-group-granular Parquet reader using s3dlio.create_async_loader.

    One s3dlio loader is constructed per file in open().  The loader prefetches
    row groups concurrently in a Tokio background task; each get_sample() call
    just grabs the next bytes off the channel (O(1), GIL released while waiting).

    Total I/O operations per file = number of row groups (e.g. 1,968), NOT
    number of samples (e.g. 16,000,000).  This is the 8,000x reduction that
    breaks the Python overhead ceiling seen with per-sample readers.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        args = self._args
        opts = getattr(args, "storage_options", {}) or {}
        self._opts = opts
        self._epoch = epoch

        # Column indices: list[int] or None (all columns)
        col_opt = opts.get("columns")
        self._columns = list(col_opt) if col_opt is not None else None

        # Prefetch depth: row groups to buffer ahead of the Python consumer
        self._prefetch = int(opts.get("prefetch", 16))

        # Footer cap: bytes to fetch from file tail for Parquet footer parsing
        self._footer_cap = int(opts.get("footer_cap", 4 * 1024 * 1024))

        # Set S3 endpoint early so s3dlio picks it up on first use.
        ep = opts.get("endpoint_url")
        if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
            os.environ["AWS_ENDPOINT_URL_S3"] = ep

        # Footer + row-group-offset cache: filename → list[int] (cumulative row offsets)
        # Populated on first open(), held for the epoch lifetime.
        self._rg_offsets: dict = {}  # filename → [0, rg0_rows, rg0+rg1_rows, ...]

        # Loader cache: filename → s3dlio PyBytesDataLoaderSyncIter
        # The iterator is positioned at the next row group to fetch.
        self._loader_iters: dict = {}

        # Byte-count cache: (filename, rg_idx) → int (compressed bytes for dlp)
        self._rg_bytes: dict = {}

        self.logger.info(
            f"{utcnow()} ParquetReaderS3dlio thread={thread_index} epoch={epoch} "
            f"columns={self._columns} prefetch={self._prefetch}"
        )

    # ── URI helpers ───────────────────────────────────────────────────────────

    def _prefix_for_file(self, filename: str) -> str:
        """
        Return the s3dlio URI prefix for a single parquet file.

        DLIO passes filenames in several forms depending on storage_type:
          S3/AIStore : "data/dlrm/train/img_00_of_64.parquet"  (key relative to bucket)
          local      : "/mnt/nvme/dlrm/train/img_00_of_64.parquet"  (absolute path)

        s3dlio.create_async_loader wants the *directory* prefix so it can list
        all files under it.  We pass the full file URI instead — ParquetRowGroupDataset
        in Rust calls list_objects on the prefix, which returns just that one object
        when the prefix is an exact key.
        """
        if "://" in filename:
            return filename  # already a full URI

        storage_type = getattr(self._args, "storage_type", StorageType.LOCAL)
        if storage_type in (StorageType.S3, StorageType.AISTORE):
            bucket = self._args.storage_root.rstrip("/")
            key = filename.lstrip("/")
            return f"s3://{bucket}/{key}"
        else:
            # Local filesystem — s3dlio supports file:// URIs
            path = os.path.abspath(filename)
            return f"file://{path}"

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """
        Create a s3dlio loader for this parquet file (at most once per epoch).

        Returns (loader_iter, rg_offsets) and stores it in open_file_map.
        The loader starts prefetching row groups immediately in a background
        Tokio task; get_sample() just pulls from the channel.
        """
        if filename in self._loader_iters:
            return self.open_file_map.get(filename)

        import s3dlio

        uri = self._prefix_for_file(filename)

        loader_opts = {
            "format": "parquet",
            "prefetch": self._prefetch,
            "footer_cap": self._footer_cap,
        }
        if self._columns is not None:
            loader_opts["columns"] = self._columns

        loader = s3dlio.create_async_loader(uri, loader_opts)
        loader_iter = iter(loader)  # starts background Tokio task immediately

        self._loader_iters[filename] = loader_iter

        # Build row-group cumulative offsets by inspecting the first few items?
        # No — we don't have the footer here.  Use pyarrow to get metadata once.
        # This is a one-time cost per file per epoch (~1 range GET for the footer).
        rg_offsets = self._build_rg_offsets(uri, filename)
        self._rg_offsets[filename] = rg_offsets

        handle = (loader_iter, rg_offsets)
        self._loader_iters[filename] = loader_iter
        self.open_file_map[filename] = handle

        self.logger.debug(
            f"{utcnow()} ParquetReaderS3dlio.open {filename} "
            f"row_groups={len(rg_offsets)-1} total_rows={rg_offsets[-1]}"
        )
        return handle

    def _build_rg_offsets(self, uri: str, filename: str) -> list:
        """
        Build cumulative row-group offsets using pyarrow's footer read.

        This is a one-time cost per file per epoch.  pyarrow reads only the
        footer (a small range GET), not the row data.  Result is cached in
        _rg_offsets for the lifetime of the epoch.
        """
        import pyarrow.parquet as pq

        storage_type = getattr(self._args, "storage_type", StorageType.LOCAL)
        if storage_type in (StorageType.S3, StorageType.AISTORE):
            # Use s3dlio range file adapter for S3 footer reads
            from dlio_benchmark.reader.parquet_reader_s3_iterable import _S3RangeFile
            pf = pq.ParquetFile(_S3RangeFile(uri))
        else:
            # Local file: pyarrow opens directly
            path = uri.replace("file://", "")
            pf = pq.ParquetFile(path)

        meta = pf.metadata
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)
        return offsets

    @dlp.log
    def close(self, filename):
        """
        No-op during the epoch — caches are kept until finalize().

        In ON_DEMAND mode the base class calls close() after every sample.
        Evicting the loader here would discard the prefetch buffer and force
        a fresh loader (and a new background Tokio task) for each sample.
        """
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Pull the next row-group bytes from the s3dlio prefetch channel.

        The loader yields row groups in sequential order.  bisect maps
        sample_index → rg_idx so we know which row group to expect next.
        If the cache already has it (previously fetched for an earlier sample
        in the same row group) we skip the fetch.  Otherwise we pull from
        the channel — this is the actual measured I/O.
        """
        loader_iter, rg_offsets = self.open_file_map[filename]

        rg_idx = max(0, bisect.bisect_right(rg_offsets, sample_index) - 1)
        rg_idx = min(rg_idx, len(rg_offsets) - 2)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_bytes:
            # Pull the next item from the channel — GIL released in Rust (py.detach) while waiting.
            # Returns a PyBytesView (implements the buffer protocol).
            item = next(loader_iter)
            compressed_bytes = len(item)
            self._rg_bytes[cache_key] = compressed_bytes

        dlp.update(image_size=self._rg_bytes[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        """
        Fast read_index — avoids base-class utcnow() overhead per sample.

        With 16M samples per worker, datetime.now().strftime() twice per call
        adds ~48s of pure Python overhead.  We replicate the essential logic
        without logging cost.
        """
        dlp.update(step=step)
        filename, sample_index = self.global_index_map[image_idx]
        if (
            filename not in self.open_file_map
            or self.open_file_map[filename] is None
        ):
            self.open_file_map[filename] = self.open(filename)
        self.get_sample(filename, sample_index)
        if self._args.read_type is _ReadType.ON_DEMAND:
            self.open_file_map[filename] = None
        return self._args.resized_image

    @dlp.log
    def finalize(self):
        """Flush all caches and drop loader iterators at epoch boundary."""
        self._loader_iters.clear()
        self._rg_offsets.clear()
        self._rg_bytes.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
