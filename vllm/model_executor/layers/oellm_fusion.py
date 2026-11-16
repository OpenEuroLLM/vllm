# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference fusions preserving the eager model's BF16 rounding boundaries."""

import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms(
    X, W, Y, H: tl.constexpr, EPS: tl.constexpr, ZC: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    w = tl.load(W + h, h < H, 0).to(tl.float32)
    if ZC:
        w += 1.0
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    tl.store(Y + row * H + h, w * (x * inv), h < H)


def rms_norm(x, weight, eps, zero_centered=False):
    x = x.contiguous()
    out = torch.empty_like(x)
    h = x.shape[-1]
    if x.numel():
        _rms[(x.numel() // h,)](
            x,
            weight,
            out,
            h,
            eps,
            zero_centered,
            triton.next_power_of_2(h),
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _mhc_read(
    S,
    C,
    APRE,
    APOST,
    ARES,
    BPRE,
    BPOST,
    BRES,
    X,
    POST,
    RES,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
    PART: tl.constexpr = 0,  # 0: all; 1: x and post; 2: the Sinkhorn residual mix
):
    t = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    s = tl.arange(0, 4)
    dtype = S.dtype.element_ty
    stream = tl.load(S + t * 4 * H + s[:, None] * H + h[None, :], h[None, :] < H, 0).to(
        tl.float32
    )
    # torch.linalg.vector_norm returns the input dtype before .float().
    norm = tl.sqrt(tl.sum(tl.sum(stream * stream, 1), 0)).to(dtype).to(tl.float32)
    rms = tl.maximum(norm / ((4 * H) ** 0.5), EPS)
    cp = tl.load(C + t * 24 + s).to(tl.float32) / rms
    cq = tl.load(C + t * 24 + 4 + s).to(tl.float32) / rms
    pre = tl.sigmoid(
        tl.load(APRE).to(tl.float32) * cp + tl.load(BPRE + s).to(tl.float32)
    )
    post = 2.0 * tl.sigmoid(
        tl.load(APOST).to(tl.float32) * cq + tl.load(BPOST + s).to(tl.float32)
    )
    j = tl.arange(0, 4)
    offsets = s[:, None] * 4 + j[None, :]
    matrix = tl.load(C + t * 24 + 8 + offsets).to(tl.float32) / rms
    matrix = tl.load(ARES).to(tl.float32) * matrix + tl.load(BRES + offsets).to(
        tl.float32
    )
    if PART != 1:
        for _ in range(ITERS):
            maximum = tl.max(matrix, 0)
            lse = maximum + tl.log(tl.sum(tl.exp(matrix - maximum[None, :]), 0))
            matrix -= lse[None, :]
            maximum = tl.max(matrix, 1)
            lse = maximum + tl.log(tl.sum(tl.exp(matrix - maximum[:, None]), 1))
            matrix -= lse[:, None]
        tl.store(RES + t * 16 + offsets, tl.exp(matrix))
    if PART != 2:
        weighted = (
            (pre.to(dtype).to(tl.float32)[:, None] * stream).to(dtype).to(tl.float32)
        )
        value = tl.sum(weighted, 0)
        tl.store(X + t * H + h, value, h < H)
        tl.store(POST + t * 4 + s, post)


@triton.jit
def _mhc_coeffs(S, PT, C, K: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr):
    """C[t, c] = sum_k S[t, k] PT[c, k] (fp32 accumulation, stored in C's dtype)."""
    t, c = tl.program_id(0).to(tl.int64), tl.program_id(1)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        k = k0 + tl.arange(0, BLOCK)
        x = tl.load(S + t * K + k, k < K, 0).to(tl.float32)
        w = tl.load(PT + c * K + k, k < K, 0).to(tl.float32)
        acc += x * w
    tl.store(C + t * NC + c, tl.sum(acc, 0).to(C.dtype.element_ty))


_SIDE: dict = {}


def _side_stream(device) -> torch.cuda.Stream:
    if device not in _SIDE:
        _SIDE[device] = torch.cuda.Stream(device=device)
    return _SIDE[device]


# warps of the _mhc_read program (one token per program): 2 instead of 8 halves the time
# of prefill-sized calls (micro-benchmark, unsplit path bit-identical)
_MHC_READ_WARPS = int(os.environ.get("OELLM_MHC_READ_WARPS", "2"))


def mhc_read(
    streams,
    phi,
    alpha_pre,
    alpha_post,
    alpha_res,
    b_pre,
    b_post,
    b_res,
    eps,
    iters,
    split=False,
    phi_t=None,
):
    """``split``: x / post on the current stream, the Sinkhorn residual mix (only needed
    by :func:`mhc_write`, after the sublayer) on a side stream; ``phi_t``: [24, n h]
    (contiguous) for the coefficient kernel instead of the cuBLAS GEMV."""
    tokens, n, h = streams.shape
    assert n == 4 and streams.is_contiguous()
    if phi_t is not None and tokens:
        coeffs = torch.empty(
            (tokens, phi_t.shape[0]), device=streams.device, dtype=streams.dtype
        )
        _mhc_coeffs[(tokens, phi_t.shape[0])](
            streams, phi_t, coeffs, n * h, phi_t.shape[0], 2048, num_warps=4
        )
    else:
        coeffs = streams.view(tokens, n * h) @ phi.to(streams.dtype)
    x = torch.empty((tokens, h), device=streams.device, dtype=streams.dtype)
    post = torch.empty((tokens, n), device=streams.device, dtype=torch.float32)
    res = torch.empty((tokens, n, n), device=streams.device, dtype=torch.float32)
    if not tokens:
        return x, (post, res)

    def launch(part):
        _mhc_read[(tokens,)](
            streams,
            coeffs,
            alpha_pre,
            alpha_post,
            alpha_res,
            b_pre,
            b_post,
            b_res,
            x,
            post,
            res,
            h,
            eps,
            iters,
            triton.next_power_of_2(h),
            part,
            num_warps=_MHC_READ_WARPS,
            enable_fp_fusion=False,
        )

    if not split:
        launch(0)
        return x, (post, res)
    main = torch.cuda.current_stream()
    side = _side_stream(streams.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        launch(2)
        done = torch.cuda.Event()
        done.record(side)
    launch(1)
    return x, (post, res, done)


@triton.jit
def _mhc_write(S, OUT, POST, RES, Y, H: tl.constexpr, BLOCK: tl.constexpr):
    t, j, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = tile * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, 4)
    dtype = S.dtype.element_ty
    stream = tl.load(S + t * 4 * H + s[:, None] * H + h[None, :], h[None, :] < H, 0).to(
        tl.float32
    )
    res = tl.load(RES + t * 16 + j * 4 + s).to(dtype).to(tl.float32)
    mixed = (
        tl.sum((res[:, None] * stream).to(dtype).to(tl.float32), 0)
        .to(dtype)
        .to(tl.float32)
    )
    post = tl.load(POST + t * 4 + j).to(dtype).to(tl.float32)
    out = tl.load(OUT + t * H + h, h < H, 0).to(tl.float32)
    weighted = (post * out).to(dtype).to(tl.float32)
    tl.store(Y + t * 4 * H + j * H + h, mixed + weighted, h < H)


@triton.jit
def _mhc_write4(S, OUT, POST, RES, Y, H: tl.constexpr, BLOCK: tl.constexpr):
    # _mhc_write with all four output streams per program: the 4 x BLOCK stream tile and
    # the sublayer output are loaded once instead of once per output stream (same
    # arithmetic per stream)
    t, tile = tl.program_id(0), tl.program_id(1)
    h = tile * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, 4)
    dtype = S.dtype.element_ty
    stream = tl.load(S + t * 4 * H + s[:, None] * H + h[None, :], h[None, :] < H, 0).to(
        tl.float32
    )
    out = tl.load(OUT + t * H + h, h < H, 0).to(tl.float32)
    for j in tl.static_range(4):
        res = tl.load(RES + t * 16 + j * 4 + s).to(dtype).to(tl.float32)
        mixed = (
            tl.sum((res[:, None] * stream).to(dtype).to(tl.float32), 0)
            .to(dtype)
            .to(tl.float32)
        )
        post = tl.load(POST + t * 4 + j).to(dtype).to(tl.float32)
        weighted = (post * out).to(dtype).to(tl.float32)
        tl.store(Y + t * 4 * H + j * H + h, mixed + weighted, h < H)


# _mhc_write4 is bit-identical to _mhc_write and ~2x faster on prefill-sized steps
# (_mhc_write's 256-wide per-stream tile reads the stream tile 4x); OELLM_MHC_WRITE4=0
# keeps _mhc_write, OELLM_MHC_WRITE4_CHECK=1 compares both once per worker and logs the
# result
_MHC_WRITE4 = os.environ.get("OELLM_MHC_WRITE4", "1") == "1"
_MHC_WRITE4_BLOCK = int(os.environ.get("OELLM_MHC_WRITE4_BLOCK", "512"))
_MHC_WRITE4_WARPS = int(os.environ.get("OELLM_MHC_WRITE4_WARPS", "4"))
_MHC_WRITE4_CHECK = [os.environ.get("OELLM_MHC_WRITE4_CHECK", "0") == "1"]


def mhc_write(streams, out, pending):
    tokens, n, h = streams.shape
    assert n == 4 and streams.is_contiguous() and out.is_contiguous()
    post, res = pending[:2]
    if len(pending) == 3:  # the side-stream Sinkhorn of a split read
        torch.cuda.current_stream().wait_event(pending[2])
    y = torch.empty_like(streams)
    if tokens:
        if _MHC_WRITE4:
            _mhc_write4[(tokens, triton.cdiv(h, _MHC_WRITE4_BLOCK))](
                streams,
                out,
                post,
                res,
                y,
                h,
                _MHC_WRITE4_BLOCK,
                num_warps=_MHC_WRITE4_WARPS,
                enable_fp_fusion=False,
            )
            if _MHC_WRITE4_CHECK[0]:
                _MHC_WRITE4_CHECK[0] = False
                ref = torch.empty_like(streams)
                _mhc_write[(tokens, n, triton.cdiv(h, 256))](
                    streams, out, post, res, ref, h, 256, enable_fp_fusion=False
                )
                import logging

                logging.getLogger("vllm").warning(
                    "OELLM_MHC_WRITE4 check: %d tokens, same bits as _mhc_write: %s",
                    tokens,
                    bool(torch.equal(y, ref)),
                )
        else:
            _mhc_write[(tokens, n, triton.cdiv(h, 256))](
                streams, out, post, res, y, h, 256, enable_fp_fusion=False
            )
    return y
