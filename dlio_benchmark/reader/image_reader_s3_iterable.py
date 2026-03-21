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
JPEG/PNG image reader using parallel/streaming fetch from object storage.

Each image file contains exactly one sample (one image). Prefetch downloads the
raw encoded bytes, decodes them with Pillow into a numpy uint8 array, and caches
the result. DLIO's standard FormatReader.next() / read_index() machinery then
drives training without any S3 I/O on the hot path.

Supported libraries:
  s3dlio           — uses s3dlio.get_many() (parallel, up to 64 in-flight requests)
  s3torchconnector — uses S3IterableDataset.from_objects() with sequential reader
                     (single streaming GET per file via s3torchconnector's own API;
                     no s3dlio dependency)
  minio            — uses concurrent.futures.ThreadPoolExecutor with Minio SDK

Each library is STRICTLY isolated — there is NO silent fallback to another
library. Configuring a library that is not installed raises ImportError immediately
at construction time, not later during I/O.
"""
import io
import os
import numpy as np
from PIL import Image

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.image_reader import ImageReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ImageReaderS3Iterable(ImageReader):
    """
    Parallel-prefetch JPEG/PNG reader for S3-compatible object stores.

    Replaces ImageReader.open(local_path) with a parallel prefetch of all
    image objects assigned to this DLIO worker thread. Each image is decoded
    from bytes to a numpy array during prefetch; open() simply returns the
    cached array.

    Images are 1 sample per file, so get_sample() and next() work identically
    to the local ImageReader — no index arithmetic required.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index, epoch)

        args = self._args
        opts = getattr(args, "storage_options", {}) or {}
        self._storage_library = opts.get("storage_library", "s3dlio")
        self._opts = opts
        self._epoch = epoch
        self._object_cache = {}  # obj_key → np.ndarray, populated in next()

        # s3dlio reads AWS_ENDPOINT_URL_S3 at runtime; set it early if provided.
        if self._storage_library == "s3dlio":
            ep = opts.get("endpoint_url")
            if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
                os.environ["AWS_ENDPOINT_URL_S3"] = ep

        # s3torchconnector: validate the library is installed and usable NOW,
        # not later during I/O. There is NO silent fallback to s3dlio or any
        # other library.
        if self._storage_library == "s3torchconnector":
            try:
                from s3torchconnector import S3IterableDataset as _S3ITD  # noqa: F401
                from s3torchconnector.s3reader import S3ReaderConstructor as _S3RC  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "ImageReaderS3Iterable: storage_library='s3torchconnector' requires "
                    "the s3torchconnector package. "
                    "Install with: pip install s3torchconnector"
                ) from exc

        self.logger.info(
            f"{utcnow()} ImageReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch}"
        )

    def _uri_for_obj_key(self, obj_key: str) -> str:
        if "://" in obj_key:
            return obj_key
        root = self._args.storage_root.rstrip("/")
        return f"s3://{root}/{obj_key.lstrip('/')}"

    def _prefetch_s3dlio(self, obj_keys: list) -> dict:
        import s3dlio

        uris = [self._uri_for_obj_key(k) for k in obj_keys]
        uri_to_key = dict(zip(uris, obj_keys))
        results = s3dlio.get_many(uris)

        cache = {}
        for uri, data in results:
            obj_key = uri_to_key.get(uri, uri)
            cache[obj_key] = np.asarray(Image.open(io.BytesIO(bytes(data))))
        return cache

    def _prefetch_s3torchconnector(self, obj_keys: list) -> dict:
        """Fetch all images using s3torchconnector's S3IterableDataset.

        Uses S3ReaderConstructor.sequential() for a single streaming GET per
        object — appropriate for image files which must be decoded in full before
        the pixel data is accessible.  S3IterableDataset iterates in URI order,
        yielding one BufferedIOBase reader per object.  PIL.Image.open reads
        directly from the reader without an intermediate copy.

        s3dlio is NOT required or used in any way when this method is called.
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
            # reader is a BufferedIOBase — PIL.Image.open consumes it directly.
            cache[obj_key] = np.asarray(Image.open(reader))
        return cache

    def _prefetch_minio(self, obj_keys: list) -> dict:
        from concurrent.futures import ThreadPoolExecutor
        from urllib.parse import urlparse
        from minio import Minio

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

        client = Minio(
            host,
            access_key=opts.get("access_key_id"),
            secret_key=opts.get("secret_access_key"),
            secure=secure,
            region=opts.get("region", "us-east-1"),
        )

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
            return obj_key, np.asarray(Image.open(io.BytesIO(raw)))

        n_workers = min(16, max(1, len(obj_keys)))
        cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for obj_key, arr in pool.map(_fetch_one, obj_keys):
                cache[obj_key] = arr
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
                f"ImageReaderS3Iterable: unknown storage_library {lib!r}; "
                f"supported: s3dlio, s3torchconnector, minio"
            )

    @dlp.log
    def open(self, filename):
        # Return the pre-fetched, already-decoded numpy array.
        # If somehow not cached (e.g. read_index before next()), fetch on demand.
        return self._object_cache.get(filename)

    @dlp.log
    def close(self, filename):
        self._object_cache.pop(filename, None)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)

    def next(self):
        thread_entries = self.file_map.get(self.thread_index, [])
        seen = set()
        obj_keys = []
        for _, obj_key, _ in thread_entries:
            if obj_key not in seen:
                seen.add(obj_key)
                obj_keys.append(obj_key)

        if obj_keys:
            self.logger.info(
                f"{utcnow()} ImageReaderS3Iterable thread={self.thread_index} "
                f"prefetching {len(obj_keys)} images via [{self._storage_library}]"
            )
            self._object_cache = self._prefetch(obj_keys)

        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
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
