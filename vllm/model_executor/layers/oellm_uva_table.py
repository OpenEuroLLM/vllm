# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram tables in pinned host memory, gathered by a GPU kernel (zero-copy, UVA).

The host lookup (``rows.cpu()``, a CPU gather, an upload) synchronizes the GPU and
cannot be captured in a CUDA graph. Here the table lives once per node in a shared
memory file (``OELLM_ENGRAM_UVA_DIR``, default ``/dev/shm``), every rank maps it and
registers the mapping with ``cudaHostRegister``; a Triton kernel reads the rows over
PCIe/C2C directly into a GPU tensor. No host synchronization: capturable.
"""

import fcntl
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_CUDA_HOST_REGISTER_PORTABLE = 1
_CUDA_HOST_REGISTER_MAPPED = 2


class _HostPtr:
    """A Triton pointer argument for registered host memory (UVA: the host address
    is the device address)."""

    def __init__(self, ptr: int, dtype: torch.dtype) -> None:
        self._ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self._ptr


@triton.jit
def _gather_rows(table, rows, out, n, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    ok = r < n
    row = tl.load(rows + r, mask=ok, other=0).to(tl.int64)
    d = tl.arange(0, D)
    vals = tl.load(table + row[:, None] * D + d[None, :], mask=ok[:, None])
    tl.store(out + r[:, None].to(tl.int64) * D + d[None, :], vals, mask=ok[:, None])


def _stage(path: str, offset: int, nbytes: int, key: str) -> str:
    """Copy ``nbytes`` at ``offset`` of ``path`` (the export's table) once per node into
    shared memory, with parallel positional reads / writes (no GIL, no page faults)."""
    root = os.environ.get("OELLM_ENGRAM_UVA_DIR", "/dev/shm")
    dst = os.path.join(root, f"oellm_engram_{key}.bin")
    with open(dst + ".lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(dst + ".done") and os.path.getsize(dst) == nbytes:
            return dst
        started = time.perf_counter()
        step = 256 << 20
        fi = os.open(path, os.O_RDONLY)
        fo = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fo, nbytes)

            def piece(a: int) -> None:
                n = min(step, nbytes - a)
                done = 0
                while done < n:
                    buf = os.pread(fi, n - done, offset + a + done)
                    if not buf:
                        raise EOFError(f"{path}: short read at {offset + a + done}")
                    os.pwrite(fo, buf, a + done)
                    done += len(buf)

            with ThreadPoolExecutor(16) as ex:
                list(ex.map(piece, range(0, nbytes, step)))
        finally:
            os.close(fi)
            os.close(fo)
        open(dst + ".done", "w").close()
        logger.info(
            "staged Engram table %s: %.1f GB in %.1fs",
            dst,
            nbytes / 1e9,
            time.perf_counter() - started,
        )
    return dst


class UvaTable:
    """``rows [...] -> [..., dim]`` from a pinned, node-shared host copy of ``src``."""

    def __init__(
        self, src: torch.Tensor, location: tuple[str, int], ident: str
    ) -> None:
        """``src``: the export's mapped table; ``location``: (shard, data offset)."""
        assert src.dim() == 2, src.shape
        self.rows, self.dim = src.shape
        self.dtype = src.dtype
        assert self.dim & (self.dim - 1) == 0, f"dim {self.dim}: not a power of 2"
        key = hashlib.sha1(
            f"{ident}:{tuple(src.shape)}:{src.dtype}".encode()
        ).hexdigest()[:16]
        nbytes = src.numel() * src.element_size()
        path = _stage(location[0], location[1], nbytes, key)
        np_dtype = {2: np.uint16, 4: np.uint32}[src.element_size()]
        # MAP_SHARED: every rank pins the same physical pages
        self._arr = np.memmap(
            path, dtype=np_dtype, mode="r+", shape=(self.rows, self.dim)
        )
        self.ptr = self._arr.ctypes.data
        started = time.perf_counter()
        flags = _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED
        err = torch.cuda.cudart().cudaHostRegister(self.ptr, nbytes, flags)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({path}, {nbytes}): error {err}")
        logger.info(
            "registered Engram table %s (%.1f GB) in %.1fs",
            path,
            nbytes / 1e9,
            time.perf_counter() - started,
        )
        self._host = _HostPtr(self.ptr, self.dtype)
        self._check(src)

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        flat = rows.reshape(-1)
        out = torch.empty(
            (flat.numel(), self.dim), dtype=self.dtype, device=rows.device
        )
        n = flat.numel()
        if n:
            block = 16
            _gather_rows[(triton.cdiv(n, block),)](
                self._host, flat, out, n, D=self.dim, BLOCK=block
            )
        return out.view(*rows.shape, self.dim)

    def _check(self, src: torch.Tensor) -> None:
        """A few rows (first, last, random) against the mapped export."""
        g = torch.Generator().manual_seed(0)
        idx = torch.cat(
            [
                torch.tensor([0, self.rows - 1]),
                torch.randint(0, self.rows, (62,), generator=g),
            ]
        )
        got = self(idx.cuda()).cpu()
        ref = src[idx]
        bits = torch.int16 if src.element_size() == 2 else torch.int32
        if not torch.equal(got.view(bits), ref.view(bits)):
            raise RuntimeError("UVA Engram gather does not match the host table")
