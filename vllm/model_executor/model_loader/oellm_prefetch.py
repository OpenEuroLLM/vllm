# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenEuroLLM: read a whole safetensors checkpoint into the page cache in
parallel, before the weights are loaded.

On GPFS (MareNostrum 5) one sequential reader gets ~2 GB/s and 64 threads
reading 64 MiB byte ranges ~14 GB/s, while the default safetensors loader (an
mmap, one tensor at a time) read the 5.4 GB FP8 expert shards of an 84B export
at ~80 MB/s: thousands of small tensors, page-sized random reads. vLLM's own
prefetch (``_prefetch_all_checkpoints``) reads one file per thread in the
background, which leaves an 83 GB Engram-table shard to a single reader.

``prefetch_checkpoint_parallel`` reads every file in ``chunk_size`` ranges
with ``num_threads`` threads and returns when all bytes are cached. One process
per node reads: the data-parallel engines of a node take an flock on a file in
/tmp keyed by the checkpoint; the first reads, the others wait and find a fresh
done-marker (the cache is warm for them then). Only the standard library, so
it is testable without a vLLM install.

Environment:
  OELLM_PREFETCH          auto (default: on for GPFS), 1 (any fs), 0 (off)
  OELLM_PREFETCH_THREADS  reader threads (default 64)
  OELLM_PREFETCH_CHUNK_MB byte range per read (default 64)
  OELLM_PREFETCH_DIR      lock / marker directory (default /tmp)
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# a marker older than this is not trusted: the page cache may have dropped the
# pages since
MARKER_MAX_AGE_S = 900.0


def enabled(fs_type: str) -> bool:
    mode = os.environ.get("OELLM_PREFETCH", "auto").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return False
    if mode in ("1", "on", "true", "yes"):
        return True
    return fs_type == "gpfs"


def _signature(files: list[str]) -> list[tuple[str, int, int]]:
    out = []
    for path in files:
        st = os.stat(path)
        out.append((os.path.realpath(path), st.st_size, int(st.st_mtime)))
    return out


def _read_range(job: tuple[str, int, int], block: int) -> int:
    path, offset, length = job
    fd = os.open(path, os.O_RDONLY)
    try:
        done = 0
        while done < length:
            data = os.pread(fd, min(block, length - done), offset + done)
            if not data:
                break
            done += len(data)
        return done
    finally:
        os.close(fd)


def _read_all(files: list[str], num_threads: int, chunk_size: int) -> int:
    jobs: list[tuple[str, int, int]] = []
    for path in files:
        size = os.path.getsize(path)
        jobs.extend(
            (path, off, min(chunk_size, size - off))
            for off in range(0, size, chunk_size)
        )
    block = min(chunk_size, 16 << 20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as ex:
        sizes = ex.map(_read_range, jobs, [block] * len(jobs))
        return sum(sizes)


def prefetch_checkpoint_parallel(
    files: list[str],
    num_threads: int | None = None,
    chunk_size: int | None = None,
    lock_dir: str | None = None,
) -> bool:
    """Every byte of ``files`` into the page cache. True if this process read
    them, False if a fresh marker (another engine of the node, or an earlier
    call) covered them."""
    if not files:
        return False
    threads = num_threads or int(os.environ.get("OELLM_PREFETCH_THREADS", "64"))
    chunk = chunk_size or int(os.environ.get("OELLM_PREFETCH_CHUNK_MB", "64")) << 20
    lock_dir = lock_dir or os.environ.get("OELLM_PREFETCH_DIR", "/tmp")
    files = sorted(files)
    sig = _signature(files)
    key = hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:16]
    os.makedirs(lock_dir, exist_ok=True)
    marker = os.path.join(lock_dir, f"oellm_prefetch_{key}.done")
    with open(os.path.join(lock_dir, f"oellm_prefetch_{key}.lock"), "a+") as lock:
        t_wait = time.perf_counter()
        fcntl.flock(lock, fcntl.LOCK_EX)
        waited = time.perf_counter() - t_wait
        try:
            try:
                age = time.time() - os.path.getmtime(marker)
            except OSError:
                age = float("inf")
            if age < MARKER_MAX_AGE_S:
                logger.info(
                    "checkpoint prefetch: %d file(s) already read on this node "
                    "(waited %.1f s)",
                    len(files),
                    waited,
                )
                return False
            t0 = time.perf_counter()
            total = _read_all(files, threads, chunk)
            dt = max(time.perf_counter() - t0, 1e-9)
            tmp = f"{marker}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"files": sig, "bytes": total, "seconds": dt}, fh)
            os.replace(tmp, marker)
            logger.info(
                "checkpoint prefetch: %.1f GB in %.1f s (%.2f GB/s, %d threads, "
                "%d MiB ranges)",
                total / 1e9,
                dt,
                total / 1e9 / dt,
                threads,
                chunk >> 20,
            )
            return True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
