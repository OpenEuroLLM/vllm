# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused small-op chains of the ambitious model (``OELLM_DECODE_KERNELS=triton``).

At decode the model's time goes into thousands of tiny elementwise launches
(experiment 012: ~5,200 kernels per step). Each function here replaces one chain:

- :func:`engram_rows`: the Engram n-gram hashes (XOR of id x multiplier products,
  Python modulo, row offset) -- exact int64 arithmetic, one launch per hash set;
- :func:`sigmoid_topk_route`: the sigmoid router (scores, + expert bias or logits -
  quantile beta for the selection, top-k, renormalized unbiased scores x scaling);
- :func:`swiglu_post_act_norm`: SwiGLU with the optional clamp and the bounded RMS
  post-activation norm in fp32, one rounding (Megatron's ``normed_swiglu``).
"""

import os

import torch

from vllm.triton_utils import tl, triton


def enabled(name: str | None = None) -> bool:
    """``OELLM_DECODE_KERNELS=triton``; ``OELLM_DECODE_KERNELS_OFF``: comma-separated
    parts kept on the reference path (engram_hash, route, swiglu, ckda_proj,
    o_norm, ckda_inplace, mhc_split, mhc_coeffs)."""
    if os.environ.get("OELLM_DECODE_KERNELS", "reference") != "triton":
        return False
    return name is None or name not in os.environ.get(
        "OELLM_DECODE_KERNELS_OFF", ""
    ).split(",")


# ---------------------------------------------------------------- Engram hashing
@triton.jit
def _engram_rows_kernel(
    shifted,  # [B, W] int64, W = hist + 1 (oldest first)
    mult,  # [L, MAXN] int64
    orders,  # [L] int32
    sizes,  # [L] int64
    offsets,  # [L] int64
    out,  # [B, L] int64
    W: tl.constexpr,
    L: tl.constexpr,
    MAXN: tl.constexpr,
    BL: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    lk = tl.arange(0, BL)
    ok = lk < L
    n = tl.load(orders + lk, mask=ok, other=0)
    h = tl.zeros([BL], dtype=tl.int64)
    for i in tl.static_range(MAXN):
        # history[i] = shifted[:, hist - i]
        tok = tl.load(shifted + b * W + (W - 1 - i))
        m = tl.load(mult + lk * MAXN + i, mask=ok & (i < n), other=0)
        h = h ^ (tok * m)
    size = tl.load(sizes + lk, mask=ok, other=1)
    r = h % size
    r = tl.where(r < 0, r + size, r)  # Python / torch.remainder: sign of the divisor
    r = r + tl.load(offsets + lk, mask=ok, other=0)
    tl.store(out + b * L + lk, r, mask=ok)


class EngramHash:
    """The hash constants of one Engram layer's row (or sign) set, on the device."""

    def __init__(self, orders_per_lookup, mult, sizes, offsets, device) -> None:
        self.L = len(orders_per_lookup)
        self.maxn = max(orders_per_lookup)
        m = torch.zeros(self.L, self.maxn, dtype=torch.int64)
        for j, row in enumerate(mult):
            m[j, : len(row)] = torch.tensor(row, dtype=torch.int64)
        self.mult = m.to(device)
        self.orders = torch.tensor(orders_per_lookup, dtype=torch.int32, device=device)
        self.sizes = torch.tensor(sizes, dtype=torch.int64, device=device)
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=device)

    def __call__(self, shifted: torch.Tensor) -> torch.Tensor:
        b, w = shifted.shape
        assert w >= self.maxn, (w, self.maxn)
        shifted = shifted.contiguous()
        out = torch.empty((b, self.L), dtype=torch.int64, device=shifted.device)
        if b:
            _engram_rows_kernel[(b,)](
                shifted,
                self.mult,
                self.orders,
                self.sizes,
                self.offsets,
                out,
                W=w,
                L=self.L,
                MAXN=self.maxn,
                BL=triton.next_power_of_2(self.L),
            )
        return out


# ------------------------------------------------------------------- MoE router
@triton.jit
def _route_kernel(
    logits,  # [T, E] fp32
    bias,  # [E] fp32
    w_out,  # [T, K] fp32
    i_out,  # [T, K] int32
    scaling,
    E: tl.constexpr,
    K: tl.constexpr,
    BE: tl.constexpr,
    BK: tl.constexpr,
    QUANTILE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BE)
    ok = e < E
    x = tl.load(logits + t * E + e, mask=ok, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    bb = tl.load(bias + e, mask=ok, other=0.0)
    sel = (x - bb) if QUANTILE else (s + bb)
    sel = tl.where(ok, sel, float("-inf"))
    kk = tl.arange(0, BK)
    wv = tl.zeros([BK], dtype=tl.float32)
    iv = tl.zeros([BK], dtype=tl.int32)
    for k in tl.static_range(K):
        best = tl.max(sel, axis=0)
        # lowest index among the maxima (torch.topk on distinct values)
        idx = tl.min(tl.where(sel == best, e, BE), axis=0)
        wk = tl.sum(tl.where(e == idx, s, 0.0), axis=0)
        wv = tl.where(kk == k, wk, wv)
        iv = tl.where(kk == k, idx.to(tl.int32), iv)
        sel = tl.where(e == idx, float("-inf"), sel)
    if K > 1:
        wv = wv / (tl.sum(wv, axis=0) + 1e-20)
    tl.store(w_out + t * K + kk, wv * scaling, mask=kk < K)
    tl.store(i_out + t * K + kk, iv, mask=kk < K)


def sigmoid_topk_route(
    logits: torch.Tensor, bias: torch.Tensor, k: int, scaling: float, quantile: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    t, e = logits.shape
    logits = logits.float().contiguous()
    w = torch.empty((t, k), dtype=torch.float32, device=logits.device)
    idx = torch.empty((t, k), dtype=torch.int32, device=logits.device)
    if t:
        _route_kernel[(t,)](
            logits,
            bias,
            w,
            idx,
            float(scaling),
            E=e,
            K=k,
            BE=triton.next_power_of_2(e),
            BK=triton.next_power_of_2(k),
            QUANTILE=quantile,
        )
    return w, idx


# ------------------------------------------------- SwiGLU + post-activation norm
@triton.jit
def _swiglu_pab_kernel(
    gate,
    up,
    out,
    g_stride,
    u_stride,
    o_stride,
    limit,
    eps,
    floor,
    D: tl.constexpr,
    BD: tl.constexpr,
    CLAMP: tl.constexpr,
    NORM: tl.constexpr,
    SMOOTH: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BD)
    ok = d < D
    g = tl.load(gate + r * g_stride + d, mask=ok, other=0.0)
    u = tl.load(up + r * u_stride + d, mask=ok, other=0.0)
    if CLAMP:  # torch.clamp of the bf16 tensors: in fp32, rounded to the input dtype
        g = tl.minimum(g.to(tl.float32), limit).to(g.dtype)
        u = tl.minimum(tl.maximum(u.to(tl.float32), -limit), limit).to(u.dtype)
    gf = g.to(tl.float32)
    act = gf / (1.0 + tl.exp(-gf)) * u.to(tl.float32)
    if NORM:
        ms = tl.sum(tl.where(ok, act * act, 0.0), axis=0) / D
        if SMOOTH:
            scale = 1.0 / tl.sqrt(1.0 + ms / (floor * floor))
        else:
            scale = 1.0 / tl.sqrt(tl.maximum(ms + eps, floor * floor))
        act = act * scale
    tl.store(out + r * o_stride + d, act.to(out.dtype.element_ty), mask=ok)


def swiglu_post_act_norm(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float | None,
    norm: tuple[float, float, bool] | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rows of ``gate`` / ``up`` ([M, D], last dim contiguous) -> [M, D]."""
    d = gate.shape[-1]
    gate2, up2 = gate.reshape(-1, d), up.reshape(-1, d)
    assert gate2.stride(-1) == 1 and up2.stride(-1) == 1
    m = gate2.shape[0]
    if out is None:
        out = torch.empty((m, d), dtype=gate.dtype, device=gate.device)
    out2 = out.view(-1, d)
    eps, floor, smooth = norm if norm is not None else (0.0, 0.0, False)
    if m:
        _swiglu_pab_kernel[(m,)](
            gate2,
            up2,
            out2,
            gate2.stride(0),
            up2.stride(0),
            out2.stride(0),
            float(limit) if limit is not None else 0.0,
            float(eps),
            float(floor),
            D=d,
            BD=triton.next_power_of_2(d),
            CLAMP=limit is not None,
            NORM=norm is not None,
            SMOOTH=bool(smooth),
        )
    return out.view(*gate.shape[:-1], d)


# ------------------------------------------------------- gated RMS output norm
@triton.jit
def _rms_gate_kernel(x, gate, w, out, eps, D: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BD)
    ok = d < D
    xf = tl.load(x + r * D + d, mask=ok, other=0.0).to(tl.float32)
    gf = tl.load(gate + r * D + d, mask=ok, other=0.0).to(tl.float32)
    wf = tl.load(w + d, mask=ok, other=0.0).to(tl.float32)
    ms = tl.sum(tl.where(ok, xf * xf, 0.0), axis=0) / D
    y = xf / tl.sqrt(ms + eps) * wf * (1.0 / (1.0 + tl.exp(-gf)))
    tl.store(out + r * D + d, y.to(out.dtype.element_ty), mask=ok)


def rms_gate(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, eps: float):
    """``rms_norm(x) * w * sigmoid(gate)`` over the last dim in fp32, rounded once."""
    d = x.shape[-1]
    x2, g2 = x.reshape(-1, d).contiguous(), gate.reshape(-1, d).contiguous()
    out = torch.empty_like(x2)
    if x2.shape[0]:
        _rms_gate_kernel[(x2.shape[0],)](
            x2, g2, w, out, float(eps), D=d, BD=triton.next_power_of_2(d)
        )
    return out.view(x.shape)
