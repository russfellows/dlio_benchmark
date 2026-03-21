"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
"""
NPZ reader using parallel/streaming fetch from object storage, as opposed to
the sequential per-file pattern in NPZReaderS3.

Supported libraries:
  s3dlio           — uses s3dlio.get_many() (parallel, up to 64 in-flight requests)
  s3torchconnector — uses S3IterableDataset.from_objects() with sequential reader
                     (single streaming GET per file via s3torchconnector's own API)
  minio            — uses concurrent.futures.ThreadPoolExecutor with Minio SDK

All objects assigned to this DLIO thread are fetched before iteration begins.
Note: listing is handled by ObjStoreLibStorage.list_objects(), which dispatches
per library — each library (s3dlio, s3torchconnector, minio) handles its own
listing independently. Delete is not yet implemented for object storage (no-op).

The reader integrates cleanly with DLIO's existing file_map / FormatReader
pipeline: open(filename) simply returns the pre-fetched array from the cache,
and get_sample / next / read_index all work through the standard parent chain.
"""
import io
import os
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.npz_reader import NPZReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class NPZReaderS3Iterable(NPZReader):
    """
    Parallel-prefetch NPZ reader for S3-compatible object stores.

    Replaces the sequential get_data()-per-object pattern of NPZReaderS3 with a
    parallel prefetch of all objects assigned to this DLIO worker thread, using
    whichever storage library is configured via storage_options.storage_library.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        # NPZReader.__init__ → FormatReader.__init__ sets up file_map, thread_index, etc.
        # It does NOT create a storage connection, so it is safe to call here.
        super().__init__(dataset_type, thread_index, epoch)

        args = self._args
        opts = getattr(args, "storage_options", {}) or {}
        self._storage_library = opts.get("storage_library", "s3dlio")
        self._opts = opts
        self._epoch = epoch
        self._object_cache = {}  # obj_key → np.ndarray, populated in next()

        # Configure endpoint for s3dlio / s3torchconnector at construction time
        # so that any lazy import inside get_many picks it up immediately.
        if self._storage_library in ("s3dlio", "s3torchconnector"):
            ep = opts.get("endpoint_url")
            if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
                os.environ["AWS_ENDPOINT_URL_S3"] = ep

        # Minio client is cached per worker process so TCP connections persist
        # across epochs (avoids rebuilding the urllib3 PoolManager every epoch).
        self._minio_client = None

        self.logger.info(
            f"{utcnow()} NPZReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch}"
        )

    # ── URI helpers ──────────────────────────────────────────────────────────

    def _uri_for_obj_key(self, obj_key: str) -> str:
        """Return a full s3:// URI for a DLIO object key (relative or absolute)."""
        if "://" in obj_key:
            return obj_key
        root = self._args.storage_root.rstrip("/")
        return f"s3://{root}/{obj_key.lstrip('/')}"

    # ── Parallel prefetch per library ────────────────────────────────────────

    def _prefetch_s3dlio(self, obj_keys: list) -> dict:
        """Fetch all objects in parallel using s3dlio.get_many()."""
        import s3dlio
        from s3dlio.compat.s3torchconnector import _BytesViewIO

        uris = [self._uri_for_obj_key(k) for k in obj_keys]
        uri_to_key = dict(zip(uris, obj_keys))

        # Cap max_in_flight to actual object count — no benefit provisioning semaphore
        # permits that will never be acquired.
        max_in_flight = min(64, len(uris))
        results = s3dlio.get_many(uris, max_in_flight=max_in_flight)

        cache = {}
        for uri, data in results:
            obj_key = uri_to_key.get(uri, uri)
            # _BytesViewIO wraps the Rust BytesView via the buffer protocol.
            # io.BufferedReader triggers readinto() (in-place copy into numpy's C
            # buffer) instead of bytes() (a separate 147 MB Python allocation).
            # Peak memory: Rust buffer only, no simultaneous Python bytes copy.
            raw = io.BufferedReader(_BytesViewIO(data))
            cache[obj_key] = np.load(raw, allow_pickle=True)["x"]
        return cache

    def _get_minio_client(self):
        """Return a cached Minio client, creating it once per worker process.

        The Minio client holds a urllib3 PoolManager with keep-alive TCP
        connections.  Creating it once per worker (in __init__) rather than
        per epoch avoids rebuilding the connection pool on every prefetch call,
        allowing TCP connections established during epoch 1 to be reused in
        subsequent epochs.
        """
        if self._minio_client is not None:
            return self._minio_client

        from minio import Minio
        import urllib3

        opts = self._opts
        endpoint = opts.get("endpoint_url", "")
        if endpoint.startswith("https://"):
            host = endpoint[8:]
            secure = True
        elif endpoint.startswith("http://"):
            host = endpoint[7:]
            secure = False
        else:
            host = endpoint
            secure = False

        access_key = (
            opts.get("access_key_id")
            or os.environ.get("AWS_ACCESS_KEY_ID")
        )
        secret_key = (
            opts.get("secret_access_key")
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        # maxsize=16 matches max_workers=min(16, n_files) so no thread ever
        # blocks waiting for a free connection slot in the urllib3 pool.
        pool = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=300, read=300),
            maxsize=16,
            cert_reqs="CERT_NONE",  # match secure= flag below
            retries=urllib3.Retry(total=5, backoff_factor=0.2,
                                  status_forcelist=[500, 502, 503, 504]),
        )
        if secure:
            import certifi
            pool = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=300, read=300),
                maxsize=16,
                cert_reqs="CERT_REQUIRED",
                ca_certs=certifi.where(),
                retries=urllib3.Retry(total=5, backoff_factor=0.2,
                                      status_forcelist=[500, 502, 503, 504]),
            )
        self._minio_client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=opts.get("region", "us-east-1"),
            http_client=pool,
        )
        return self._minio_client

    def _prefetch_minio(self, obj_keys: list) -> dict:
        """Fetch all object keys concurrently using Minio SDK + ThreadPoolExecutor.

        Uses a cached Minio client (see _get_minio_client) so that TCP keep-alive
        connections persist across epochs, avoiding per-epoch pool reconstruction.
        """
        from concurrent.futures import ThreadPoolExecutor
        from urllib.parse import urlparse

        client = self._get_minio_client()

        def _fetch_one(obj_key):
            uri = self._uri_for_obj_key(obj_key)
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            resp = client.get_object(bucket, key)
            try:
                raw = resp.read()
            finally:
                resp.close()
                resp.release_conn()
            return obj_key, np.load(io.BytesIO(raw), allow_pickle=True)["x"]

        n_workers = min(16, max(1, len(obj_keys)))
        cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for obj_key, arr in pool.map(_fetch_one, obj_keys):
                cache[obj_key] = arr
        return cache

    def _prefetch_s3torchconnector(self, obj_keys: list) -> dict:
        """Fetch all objects using s3torchconnector's S3IterableDataset.

        Uses S3ReaderConstructor.sequential() for a single streaming GET per
        object — no range splitting, no extra HEAD requests.  S3IterableDataset
        iterates in URI order, yielding one S3Reader (BufferedIOBase) per object.
        np.load reads directly from the S3Reader — no intermediate copy.

        Listing is handled by ObjStoreLibStorage.list_objects() using
        S3Client.list_objects() — s3dlio is NOT required when using
        s3torchconnector. Delete is not yet implemented for object storage (no-op).
        """
        from s3torchconnector import S3IterableDataset
        from s3torchconnector.s3reader import S3ReaderConstructor

        opts = self._opts
        endpoint = opts.get("endpoint_url", "")
        region = opts.get("region", "us-east-1")

        uris = [self._uri_for_obj_key(k) for k in obj_keys]

        # sequential() → one streaming GET per object (no range splitting).
        # Iteration order matches uris order, so zip with obj_keys is safe.
        dataset = S3IterableDataset.from_objects(
            uris,
            region=region,
            endpoint=endpoint,
            reader_constructor=S3ReaderConstructor.sequential(),
        )

        cache = {}
        for obj_key, reader in zip(obj_keys, dataset):
            # S3Reader is a BufferedIOBase — np.load consumes it without copying.
            cache[obj_key] = np.load(reader, allow_pickle=True)["x"]
        return cache

    def _prefetch(self, obj_keys: list) -> dict:
        lib = self._storage_library
        if lib == "s3dlio":
            return self._prefetch_s3dlio(obj_keys)
        elif lib == "s3torchconnector":
            return self._prefetch_s3torchconnector(obj_keys)
        elif lib == "minio":
            return self._prefetch_minio(obj_keys)
        else:
            raise ValueError(
                f"NPZReaderS3Iterable: unknown storage_library {lib!r}; "
                f"supported: s3dlio, s3torchconnector, minio"
            )

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """Return the pre-fetched array from the cache (no I/O at this point)."""
        return self._object_cache.get(filename)

    @dlp.log
    def close(self, filename):
        # Evict from cache to free memory once DLIO is done with this object.
        self._object_cache.pop(filename, None)

    @dlp.log
    def get_sample(self, filename, sample_index):
        # Delegates to NPZReader.get_sample which reads self.open_file_map[filename]
        # (already populated by FormatReader.next via open()) and updates dlp metrics.
        super().get_sample(filename, sample_index)

    def next(self):
        """Pre-fetch all this thread's objects in parallel, then yield batches."""
        thread_entries = self.file_map.get(self.thread_index, [])
        # Preserve order but deduplicate object keys (each object may contain multiple samples)
        seen = set()
        obj_keys = []
        for _, obj_key, _ in thread_entries:
            if obj_key not in seen:
                seen.add(obj_key)
                obj_keys.append(obj_key)

        if obj_keys:
            self.logger.info(
                f"{utcnow()} NPZReaderS3Iterable thread={self.thread_index} "
                f"prefetching {len(obj_keys)} objects via [{self._storage_library}]"
            )
            self._object_cache = self._prefetch(obj_keys)

        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        """For ON_DEMAND reads: fetch a single object on demand if not cached."""
        filename, _ = self.global_index_map[image_idx]
        if filename not in self._object_cache:
            self._object_cache.update(self._prefetch([filename]))
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
