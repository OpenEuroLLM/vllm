# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenEuroLLM "ambitious" hybrid (Megatron HybridModel) for vLLM.

The HF export ``oellm_ambitious`` (oellm-autoexp ``oellm_autoexp/hf_export``) is the
reference: one layer per Megatron pattern symbol (``block_types``), each wrapped by
the residual read / write of mHC (``residual="mhc"``) or the Linear Depth Memory
(``residual="ldm"``):

* ``attention``: NoPE softmax attention with qk-norm, exclusive self attention (XSA)
  and the sigmoid output gate (vLLM ``Attention``, paged KV cache).
* ``ckda``: the FLA ComplexKimiDeltaAttention layer (signed gate) on the FLA
  kernels; its short-conv and recurrent states live in the mamba cache (GDN
  metadata).
* ``moe``: sigmoid router with expert-bias or quantile-balancing selection,
  clamped SwiGLU experts with the optional bounded post-activation norm, an
  ungated shared expert; experts sharded over the TP ranks.
* ``engram``: hashed n-gram tables (rows sharded over the TP ranks), the gate, the
  dilated causal conv; the last ``max(orders) - 1`` token ids and the conv window
  live in the mamba cache (short-conv metadata).

mHC and LDM are per token (no state across tokens). Eager only (no torch.compile /
CUDA graphs), no speculative decoding, no prefix caching.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .interfaces import HasInnerState, IsHybrid
from .utils import make_empty_intermediate_tensors_factory


def _tp() -> tuple[int, int]:
    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def _register(prefix: str, layer: nn.Module) -> None:
    ctx = get_current_vllm_config().compilation_config.static_forward_context
    if prefix in ctx:
        raise ValueError(f"Duplicate layer name: {prefix}")
    ctx[prefix] = layer


def _tp_shard_loader(param: torch.Tensor, loaded: torch.Tensor) -> None:
    """The rank's contiguous slice of a vector sharded by heads."""
    tp, rank = _tp()
    size = param.shape[0]
    param.data.copy_(loaded[rank * size : (rank + 1) * size])


def _layer_metadata(prefix: str):
    raw = get_forward_context().attn_metadata
    if raw is None:
        return None
    assert isinstance(raw, dict)
    return raw.get(prefix)


class OellmRMSNorm(nn.Module):
    """fp32 reductions, ``(w * x_hat).to(dtype)``; ``zero_centered``: the weight
    holds gamma - 1."""

    def __init__(self, size: int, eps: float, zero_centered: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size) if zero_centered else torch.ones(size))
        self.eps, self.zero_centered = eps, zero_centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight.float() + (1.0 if self.zero_centered else 0.0)
        return (w * xf).to(x.dtype)


# ----------------------------------------------------------------------- residuals
def _sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    x = logits.float()
    for _ in range(iters):
        x = x - torch.logsumexp(x, dim=-2, keepdim=True)
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return torch.exp(x)


class OellmMHC(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        n, h = config.mhc_expansion_rate, config.hidden_size
        self.n, self.iters, self.eps = n, config.mhc_sinkhorn_iters, config.mhc_eps
        self.phi = nn.Parameter(torch.zeros(n * h, 2 * n + n * n))
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))

    def read(self, streams: torch.Tensor):
        t, n, h = streams.shape
        flat = streams.reshape(t, n * h)
        r = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).float() / (n * h) ** 0.5
        coeffs = torch.matmul(flat, self.phi.to(flat.dtype)).float() / r.clamp(min=self.eps)
        pre, post, res = torch.split(coeffs, [n, n, n * n], dim=-1)
        h_pre = torch.sigmoid(self.alpha_pre.float() * pre + self.b_pre.float())
        h_post = 2.0 * torch.sigmoid(self.alpha_post.float() * post + self.b_post.float())
        h_res = _sinkhorn(
            self.alpha_res.float() * res.view(t, n, n) + self.b_res.float(), self.iters
        )
        x = (h_pre.to(streams.dtype).unsqueeze(-1) * streams).sum(dim=-2)
        return x, (h_post, h_res)

    @staticmethod
    def write(streams: torch.Tensor, out: torch.Tensor, pending) -> torch.Tensor:
        h_post, h_res = (p.to(streams.dtype) for p in pending)
        mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-3)).sum(dim=-2)
        return mixed + h_post.unsqueeze(-1) * out.unsqueeze(-2)


class OellmLDMNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        h, hid = config.hidden_size, config.ldm_hidden_dim
        self.groups, self.key_dim, self.eps = (
            config.ldm_groups,
            config.ldm_key_dim,
            config.rms_norm_eps,
        )
        self.norm_weight = nn.Parameter(torch.ones(h))
        self.w1 = nn.Linear(h, hid)
        self.address = nn.Linear(hid, self.groups * self.key_dim)
        self.gate = nn.Linear(hid, self.groups)

    def forward(self, x: torch.Tensor):
        xf = x.float()
        w1 = self.w1.weight
        xn = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(w1.dtype)
        hidden = F.silu(F.linear(xn, w1 * self.norm_weight.to(w1.dtype), self.w1.bias))
        shape = (*x.shape[:-1], self.groups, self.key_dim)
        address = F.normalize(self.address(hidden).float().view(shape), dim=-1, eps=1e-6)
        return address, 2.0 * torch.sigmoid(self.gate(hidden).float())


class OellmLDMWrite(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.net = OellmLDMNet(config)
        self.forget_logit = nn.Parameter(torch.zeros(config.ldm_groups))


class OellmLDMRead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.net = OellmLDMNet(config)
        self.read_gain = nn.Parameter(torch.ones(config.ldm_groups))

    def query(self, x: torch.Tensor) -> torch.Tensor:
        q, gate = self.net(x)
        return q * (gate * self.read_gain.float()).unsqueeze(-1)


def ldm_step(write, read, x, value, memory, covariance):
    key, weight = write.net(x)
    g = key.shape[-2]
    scale = weight.clamp_min(1e-6).sqrt()
    kbar = key * scale.unsqueeze(-1)
    vbar = value.float().reshape(*value.shape[:-1], g, value.shape[-1] // g)
    vbar = vbar * scale.unsqueeze(-1)
    lam = torch.sigmoid(write.forget_logit.float()).clamp_min(1e-6)
    pk = torch.einsum("...gij,...gj->...gi", covariance, kbar)
    kappa = pk / (lam + (kbar * pk).sum(-1)).unsqueeze(-1)
    u = vbar - torch.einsum("...gkd,...gk->...gd", memory, kbar)
    kp = torch.einsum("...gi,...gij->...gj", kbar, covariance)
    memory = memory + kappa.unsqueeze(-1) * u.unsqueeze(-2)
    covariance = (covariance - kappa.unsqueeze(-1) * kp.unsqueeze(-2)) / lam[:, None, None]
    r = torch.einsum("...gkd,...gk->...gd", memory, read.query(x))
    return r.reshape(x.shape).to(x.dtype), memory, covariance


class OellmLDMMixer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.write_site = OellmLDMWrite(config)
        self.next_query = OellmLDMRead(config)


class OellmLDMBoundary(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.groups, self.key_dim, self.alpha = (
            config.ldm_groups,
            config.ldm_key_dim,
            config.ldm_alpha,
        )
        self.embed_write = OellmLDMWrite(config)
        self.first_query = OellmLDMRead(config)

    def enter(self, x: torch.Tensor):
        g, k = self.groups, self.key_dim
        memory = x.new_zeros((x.shape[0], g, k, x.shape[-1] // g), dtype=torch.float32)
        eye = torch.eye(k, device=x.device, dtype=torch.float32) * self.alpha
        covariance = eye.expand(x.shape[0], g, k, k)
        r, memory, covariance = ldm_step(
            self.embed_write, self.first_query, x, x, memory, covariance
        )
        return x, r, memory, covariance


# ----------------------------------------------------------------------- attention
class OellmXSAAttention(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        tp, _ = _tp()
        h, d = config.hidden_size, config.head_dim
        self.total_heads, self.total_kv = config.num_attention_heads, config.num_key_value_heads
        assert self.total_heads % tp == 0
        self.nh = self.total_heads // tp
        self.nkv = max(1, self.total_kv // tp)
        self.d = d
        self.rep = self.nh // self.nkv
        self.xsa, self.xsa_eps = config.exclusive_self_attention, config.xsa_eps
        self.qkv_proj = QKVParallelLinear(
            h, d, self.total_heads, self.total_kv, bias=False, prefix=f"{prefix}.qkv_proj"
        )
        self.gate_proj = (
            ColumnParallelLinear(
                h, self.total_heads * d, bias=False, prefix=f"{prefix}.gate_proj"
            )
            if config.attention_output_gate
            else None
        )
        self.o_proj = RowParallelLinear(
            self.total_heads * d, h, bias=False, prefix=f"{prefix}.o_proj"
        )
        zcg = config.layernorm_zero_centered_gamma
        self.q_norm = OellmRMSNorm(d, config.rms_norm_eps, zcg) if config.qk_layernorm else None
        self.k_norm = OellmRMSNorm(d, config.rms_norm_eps, zcg) if config.qk_layernorm else None
        self.attn = Attention(
            self.nh,
            d,
            d**-0.5,
            num_kv_heads=self.nkv,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.nh * self.d, self.nkv * self.d, self.nkv * self.d], dim=-1)
        if self.q_norm is not None:
            q = self.q_norm(q.view(t, self.nh, self.d)).view(t, -1)
            k = self.k_norm(k.view(t, self.nkv, self.d)).view(t, -1)
        o = self.attn(q, k, v).view(t, self.nh, self.d)
        gate = None
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x)[0].view(t, self.nh, self.d).float())
        if self.xsa:
            vf = v.view(t, self.nkv, self.d).float()
            if self.rep > 1:
                vf = vf.repeat_interleave(self.rep, dim=1)
            of = o.float()
            c = (of * vf).sum(-1, keepdim=True) / (vf.pow(2).sum(-1, keepdim=True) + self.xsa_eps)
            of = of - c * vf
            if gate is not None:
                of = of * gate
            o = of.to(x.dtype)
        elif gate is not None:
            o = (o.float() * gate).to(x.dtype)
        out, _ = self.o_proj(o.reshape(t, self.nh * self.d))
        return out


# ----------------------------------------------------------------------- CKDA
class OellmCKDA(MambaBase, nn.Module):
    """FLA ``ComplexKimiDeltaAttention`` (``fla.layers.complex_kda_layer``) on the
    FLA kernels, heads sharded over the TP ranks. Parameter names follow the FLA
    layer (``mixer.layer.*`` in the export)."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        from fla.layers.complex_kda_layer import is_signed, safe_gate_ok

        kw = dict(config.fla_layer["kwargs"])
        self.tp, self.tp_rank = _tp()
        h = config.hidden_size
        self.head_dim = kw.get("head_dim", 128)
        self.num_heads = kw.get("num_heads", 16)
        self.num_v_heads = kw.get("num_v_heads") or self.num_heads
        expand_v = kw.get("expand_v", 1)
        self.head_v_dim = int(self.head_dim * expand_v)
        assert self.num_heads % self.tp == 0 and self.num_v_heads == self.num_heads, (
            "GVA / uneven head shards are not supported"
        )
        assert self.head_v_dim == self.head_dim, "expand_v != 1 is not supported"
        self.nh = self.num_heads // self.tp
        self.key_dim, self.value_dim = self.num_heads * self.head_dim, self.num_v_heads * self.head_v_dim
        self.lk, self.lv = self.key_dim // self.tp, self.value_dim // self.tp
        self.gate = kw.get("gate", "signed_sigmoid2")
        self.signed = is_signed(self.gate)
        self.safe_gate = safe_gate_ok(self.gate)
        self.lower_bound = kw.get("lower_bound", -5.0)
        self.forget_offset = float(kw.get("forget_offset", 0.0))
        self.allow_neg_eigval = kw.get("allow_neg_eigval", False)
        self.conv_size = kw.get("conv_size", 4)
        assert kw.get("use_short_conv", True) and not kw.get("conv_bias", False)
        drop_silu = kw.get("drop_silu", False)
        conv_silu = "" if drop_silu else kw.get("conv_silu", "qkv")
        if kw.get("drop_key_silu", False):
            conv_silu = conv_silu.replace("k", "")
        self.conv_act = {c: ("silu" if c in conv_silu else None) for c in "qkv"}
        self.xsa = kw.get("xsa", False)
        self.output_gate = kw.get("output_gate", "lowrank")

        self.q_proj = ColumnParallelLinear(h, self.key_dim, bias=False, prefix=f"{prefix}.q_proj")
        self.k_proj = ColumnParallelLinear(h, self.key_dim, bias=False, prefix=f"{prefix}.k_proj")
        self.v_proj = ColumnParallelLinear(h, self.value_dim, bias=False, prefix=f"{prefix}.v_proj")
        for name, dim in (("q", self.key_dim), ("k", self.key_dim), ("v", self.value_dim)):
            conv = ColumnParallelLinear(self.conv_size, dim, bias=False, prefix=f"{prefix}.{name}_conv1d")
            conv.weight.data = conv.weight.data.unsqueeze(1)  # [dim, 1, K] as nn.Conv1d
            setattr(self, f"{name}_conv1d", conv)
        self.f_proj0 = ReplicatedLinear(h, self.head_v_dim, bias=False, prefix=f"{prefix}.f_proj.0")
        self.f_proj1 = ColumnParallelLinear(
            self.head_v_dim, self.num_v_heads * self.head_dim, bias=False, prefix=f"{prefix}.f_proj.1"
        )
        self.b_proj = ColumnParallelLinear(
            h, self.num_v_heads, bias=kw.get("beta_init_style", "standard") == "spread",
            prefix=f"{prefix}.b_proj",
        )
        self.A_log = nn.Parameter(torch.zeros(self.nh, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.nh * self.head_dim, dtype=torch.float32))
        self.A_log.weight_loader = _tp_shard_loader
        self.dt_bias.weight_loader = _tp_shard_loader
        if self.output_gate == "lowrank":
            self.g_proj0 = ReplicatedLinear(h, self.head_v_dim, bias=False, prefix=f"{prefix}.g_proj.0")
            self.g_proj1 = ColumnParallelLinear(
                self.head_v_dim, self.value_dim, bias=True, prefix=f"{prefix}.g_proj.1"
            )
        else:
            self.g_proj = ColumnParallelLinear(h, self.value_dim, bias=True, prefix=f"{prefix}.g_proj")
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.o_norm_eps = kw.get("norm_eps", 1e-5)
        self.o_proj = RowParallelLinear(self.value_dim, h, bias=False, prefix=f"{prefix}.o_proj")

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.prefix = prefix
        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        _register(prefix, self)

    # -- mamba state
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self):
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype, "float32"
        )

    def get_state_shape(self):
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp, self.num_heads, self.head_dim, conv_kernel_size=self.conv_size
        )

    # -- forward
    def _gate_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_gate == "lowrank":
            return self.g_proj1(self.g_proj0(x)[0])[0]
        return self.g_proj(x)[0]

    def _o_norm(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        of = o.float()
        of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.o_norm_eps)
        return (of * self.o_norm_weight.float() * torch.sigmoid(gate.float())).to(o.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from fla.layers.complex_kda_layer import compute_gate
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        t = x.shape[0]
        m = _layer_metadata(self.prefix)
        q, k, v = self.q_proj(x)[0], self.k_proj(x)[0], self.v_proj(x)[0]
        g = self.f_proj1(self.f_proj0(x)[0])[0]
        beta = self.b_proj(x)[0]
        gate_out = self._gate_out(x)
        if m is None:  # profile run
            o = v.view(t, self.nh, self.head_v_dim)
            return self.o_proj(self._o_norm(o, gate_out.view_as(o)).reshape(t, -1))[0]

        n = m.num_actual_tokens
        q, k, v, g, beta, gate_out = (z[:n] for z in (q, k, v, g, beta, gate_out))
        conv_state, rec_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        cs = dict(zip("qkv", conv_state.split([self.lk, self.lk, self.lv], dim=-2)))
        idx = m.non_spec_state_indices_tensor
        assert m.spec_sequence_masks is None, "speculative decoding is not supported"
        prefill = m.num_prefills > 0
        outs = []
        for name, z in (("q", q), ("k", k), ("v", v)):
            w = getattr(self, f"{name}_conv1d").weight
            w = w.view(w.size(0), w.size(2))
            if prefill:
                z = causal_conv1d_fn(
                    z.transpose(0, 1), w, None, conv_states=cs[name],
                    query_start_loc=m.non_spec_query_start_loc, cache_indices=idx,
                    has_initial_state=m.has_initial_state, activation=self.conv_act[name],
                    metadata=m,
                ).transpose(0, 1)
            else:
                z = causal_conv1d_update(
                    z.contiguous(), cs[name], w, None, activation=self.conv_act[name],
                    conv_state_indices=idx[:n],
                )
            outs.append(z)
        q, k, v = (z.reshape(1, n, self.nh, -1) for z in outs)
        g = g.reshape(1, n, self.nh, self.head_dim)
        beta = beta.reshape(1, n, self.nh)
        sign, g = compute_gate(self.gate, g, self.A_log, self.dt_bias, self.lower_bound)
        if self.forget_offset:
            g = g - self.forget_offset
            if self.safe_gate:
                g = g.clamp_min(self.lower_bound)

        if prefill:
            cu = m.non_spec_query_start_loc
            init = rec_state[idx].clone()
            init[~m.has_initial_state] = 0
        else:
            cu = torch.arange(n + 1, device=x.device, dtype=torch.int32)
            init = rec_state[idx[:n]].clone()
        kernel = chunk_kda if prefill else fused_recurrent_kda
        extra = {"safe_gate": self.safe_gate} if prefill else {}
        o, final = kernel(
            q=q, k=k, v=v, g=g, beta=beta, sign=sign, scale=self.head_dim**-0.5,
            initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=self.allow_neg_eigval, lower_bound=self.lower_bound,
            state_v_first=True, cu_seqlens=cu.long(), **extra,
        )
        rec_state[idx[: final.shape[0]]] = final.to(rec_state.dtype)
        if self.xsa:
            from fla.modules.xsa import exclusive_self_attention

            o = exclusive_self_attention(o.reshape(-1, *o.shape[-2:]), v.reshape(-1, *v.shape[-2:])).view_as(o)
        o = self._o_norm(o.reshape(n, self.nh, self.head_v_dim), gate_out.view(n, self.nh, self.head_v_dim))
        out = self.o_proj(o.reshape(n, -1))[0]
        if n < t:
            out = torch.cat([out, out.new_zeros(t - n, out.shape[-1])])
        return out


# ----------------------------------------------------------------------- MoE
def _swiglu(gate, up, limit, norm=None):
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    if norm is None:
        return F.silu(gate) * up
    eps, floor, smooth = norm
    act = F.silu(gate.float()) * up.float()
    ms = act.square().mean(-1, keepdim=True)
    if smooth:
        scale = torch.rsqrt(1.0 + ms / (floor * floor))
    else:
        scale = torch.rsqrt((ms + eps).clamp(min=floor * floor))
    return (act * scale).to(gate.dtype)


class OellmMoE(nn.Module):
    """Experts sharded contiguously over the TP ranks (each rank computes its
    experts' contributions, then one all-reduce); router and shared expert
    replicated."""

    def __init__(self, config, prefix: str) -> None:
        super().__init__()
        self.tp, self.tp_rank = _tp()
        h, e, i = config.hidden_size, config.num_experts, config.moe_intermediate_size
        assert e % self.tp == 0, f"{e} experts do not divide over {self.tp} ranks"
        self.num_experts, self.local = e, e // self.tp
        self.first = self.tp_rank * self.local
        self.top_k, self.scaling = config.num_experts_per_tok, config.router_scaling_factor
        self.limit = config.activation_clamp_value
        pab = (config.rms_norm_eps, config.post_act_norm_floor, config.post_act_norm_smooth)
        self.expert_norm = pab if config.moe_expert_post_act_norm else None
        self.shared_norm = pab if config.mlp_post_act_norm else None
        self.quantile = config.router_balancing == "quantile"
        self.gate = nn.Parameter(torch.zeros(e, h))
        bias_name = "qb_beta" if self.quantile else "e_score_correction_bias"
        self.register_buffer(bias_name, torch.zeros(e, dtype=torch.float32))
        self.gate_up_proj = nn.Parameter(torch.zeros(self.local, 2 * i, h))
        self.down_proj = nn.Parameter(torch.zeros(self.local, h, i))
        for p in (self.gate_up_proj, self.down_proj):
            p.weight_loader = self._expert_loader
        si = config.shared_expert_intermediate_size
        self.shared = bool(si)
        if self.shared:
            self.shared_gate_proj = nn.Linear(h, si, bias=False)
            self.shared_up_proj = nn.Linear(h, si, bias=False)
            self.shared_down_proj = nn.Linear(si, h, bias=False)

    def _expert_loader(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded[self.first : self.first + self.local])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.gate.float())
        scores = torch.sigmoid(logits)
        if self.quantile:
            _, idx = torch.topk(logits - self.qb_beta.float(), self.top_k, dim=-1)
        else:
            _, idx = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1)
        w = scores.gather(-1, idx)
        if self.top_k > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = (w * self.scaling).to(x.dtype)
        out = torch.zeros_like(x)
        local = idx - self.first
        mine = (local >= 0) & (local < self.local)
        tok, slot = torch.nonzero(mine, as_tuple=True)
        if tok.numel():
            ex = local[tok, slot]
            order = torch.argsort(ex, stable=True)
            tok, slot, ex = tok[order], slot[order], ex[order]
            counts = torch.bincount(ex, minlength=self.local).tolist()
            start = 0
            for e, c in enumerate(counts):
                if c == 0:
                    continue
                tk, sl = tok[start : start + c], slot[start : start + c]
                start += c
                gt, up = F.linear(x[tk], self.gate_up_proj[e]).chunk(2, dim=-1)
                y = F.linear(_swiglu(gt, up, self.limit, self.expert_norm), self.down_proj[e])
                out.index_add_(0, tk, (y * w[tk, sl, None]).to(out.dtype))
        if self.tp > 1:
            out = tensor_model_parallel_all_reduce(out)
        if self.shared:
            out = out + self.shared_down_proj(
                _swiglu(self.shared_gate_proj(x), self.shared_up_proj(x), self.limit, self.shared_norm)
            )
        return out


# ----------------------------------------------------------------------- Engram
class OellmEngram(MambaBase, nn.Module):
    """Megatron ``EngramLayer`` (the export's ``OellmEngramV2``). State per request:
    the depthwise conv window ``[h, (kernel - 1) * dilation]`` and the last
    ``max(orders) - 1`` shifted token ids (short-conv metadata)."""

    def __init__(self, config, layer_idx: int, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        h = config.hidden_size
        self.h = h
        self.orders = list(config.engram_ngram_orders)
        self.num_heads = config.engram_num_heads
        self.sizes = list(config.engram_table_sizes)
        self.kernel_size = config.engram_conv_kernel_size
        self.dilation = max(self.orders)
        self.pad = (self.kernel_size - 1) * self.dilation
        self.hist = max(self.orders) - 1
        self.offsets = [0]
        for n in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + n)
        self.hash_mult = [[int(a) for a in r] for r in config.engram_hash_multipliers[str(layer_idx)]]
        self.use_token_map = bool(config.engram_token_map)
        if self.use_token_map:
            self.register_buffer("token_map", torch.zeros(config.vocab_size, dtype=torch.long))
        table_dtype = getattr(config, "engram_table_dtype", None) or "bf16"
        assert table_dtype in ("bf16", "fp32"), f"engram_table_dtype {table_dtype} not supported"
        self.dim = config.engram_embed_dim
        self.table = VocabParallelEmbedding(
            sum(self.sizes), self.dim,
            params_dtype=torch.float32 if table_dtype == "fp32" else None,
            prefix=f"{prefix}.table",
        )
        self.sign_hash = bool(getattr(config, "engram_sign_hash", False))
        if self.sign_hash:
            self.sign_mult = [[int(a) for a in r] for r in config.engram_sign_multipliers[str(layer_idx)]]
            self.sign_size = int(config.engram_sign_table_size)
            self.register_buffer("sign_table", torch.ones(self.sign_size, self.dim))
        mem = self.dim * len(self.sizes)
        self.key_proj = nn.Linear(mem, h, bias=False)
        self.value_proj = nn.Linear(mem, h, bias=False)
        self.hidden_norm = OellmRMSNorm(h, config.rms_norm_eps)
        self.key_norm = OellmRMSNorm(h, config.rms_norm_eps)
        self.signed_sqrt = config.engram_gate_signed_sqrt
        self.conv_norm = OellmRMSNorm(h, config.rms_norm_eps) if config.engram_conv_prenorm else None
        self.conv1d = nn.Conv1d(h, h, self.kernel_size, groups=h, bias=False, dilation=self.dilation)

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.prefix = prefix
        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        _register(prefix, self)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_state_dtype(self):
        return (self.model_config.dtype, torch.int64)

    def get_state_shape(self):
        return ((self.h, self.pad), (self.hist,))

    def _rows(self, shifted: torch.Tensor, s: int, mult, sizes, offsets) -> torch.Tensor:
        p = self.hist
        history = [shifted[p - i : p - i + s] for i in range(max(self.orders))]
        out, lookup = [], 0
        for n in self.orders:
            for _ in range(self.num_heads):
                hsh = torch.zeros_like(history[0])
                for i in range(n):
                    hsh = hsh ^ (history[i] * mult[lookup][i])
                out.append(torch.remainder(hsh, sizes[lookup]) + offsets[lookup])
                lookup += 1
        return torch.stack(out, dim=-1)

    def _seq_rows(self, ids: torch.Tensor, id_state: torch.Tensor):
        """ids [s] (one request), id_state [hist] -> (rows [s, L], sign rows, new id state)."""
        shifted = torch.cat([id_state, ids.long() + 1])
        s = ids.shape[0]
        rows = self._rows(shifted, s, self.hash_mult, self.sizes, self.offsets)
        signs = None
        if self.sign_hash:
            n = len(self.sizes)
            signs = self._rows(shifted, s, self.sign_mult, [self.sign_size] * n, [0] * n)
        return rows, signs, shifted[shifted.shape[0] - self.hist :]

    def _conv(self, v: torch.Tensor, state: torch.Tensor):
        """v [s, h], state [h, pad] -> (conv [s, h], new state)."""
        padded = torch.cat([state.to(v.dtype), v.transpose(0, 1)], dim=-1)
        conv = F.conv1d(padded.unsqueeze(0), self.conv1d.weight, groups=self.h, dilation=self.dilation)
        return conv[0].transpose(0, 1), padded[:, padded.shape[-1] - self.pad :]

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        m = _layer_metadata(self.prefix)
        ids = self.token_map[input_ids] if self.use_token_map else input_ids
        if m is None:  # profile run: one request without history
            spans = [(0, t, None, False)]
            conv_state = id_state = None
        else:
            conv_state, id_state = self.kv_cache
            spans = []
            nd = m.num_decode_tokens
            if nd:
                sd = m.state_indices_tensor_d.flatten()
                for j in range(nd):
                    spans.append((j, j + 1, sd[j], True))
            if m.num_prefills:
                qsl = (m.query_start_loc_p + nd).tolist()
                init = m.has_initial_states_p.tolist()
                sp = m.state_indices_tensor_p.flatten()
                for j in range(m.num_prefills):
                    spans.append((qsl[j], qsl[j + 1], sp[j], bool(init[j])))
        n = spans[-1][1]
        rows, signs, new_ids = [], [], []
        zero_ids = ids.new_zeros(self.hist, dtype=torch.long)
        for a, b, slot, has_init in spans:
            st = id_state[slot] if (slot is not None and has_init) else zero_ids
            r, sg, nid = self._seq_rows(ids[a:b], st)
            rows.append(r)
            signs.append(sg)
            new_ids.append(nid)
        rows = torch.cat(rows)
        e = self.table(rows).to(x.dtype)
        if self.sign_hash:
            e = e * self.sign_table[torch.cat(signs)].to(e.dtype)
        e = e.reshape(n, -1)
        xs = x[:n]
        kk, vv = self.key_proj(e), self.value_proj(e)
        gate = (self.hidden_norm(xs).float() * self.key_norm(kk).float()).sum(-1, keepdim=True) / self.h**0.5
        if self.signed_sqrt:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        vv = torch.sigmoid(gate).to(vv.dtype) * vv
        conv_in = vv if self.conv_norm is None else self.conv_norm(vv)
        outs = []
        zero_conv = conv_in.new_zeros(self.h, self.pad)
        for (a, b, slot, has_init), nid in zip(spans, new_ids):
            st = conv_state[slot] if (slot is not None and has_init) else zero_conv
            c, nst = self._conv(conv_in[a:b], st)
            outs.append(c)
            if slot is not None:
                conv_state[slot] = nst.to(conv_state.dtype)
                id_state[slot] = nid
        y = F.silu(torch.cat(outs)) + vv
        if n < t:
            y = torch.cat([y, y.new_zeros(t - n, y.shape[-1])])
        return y


# ----------------------------------------------------------------------- layers
class OellmAmbitiousLayer(nn.Module):
    def __init__(self, config, layer_idx: int, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.kind = config.block_types[layer_idx]
        self.ldm = config.residual == "ldm"
        if self.ldm:
            self.ldm_mixer = OellmLDMMixer(config)
        else:
            self.mhc = OellmMHC(config)
        zcg = config.layernorm_zero_centered_gamma
        if self.kind != "engram":
            self.input_layernorm = OellmRMSNorm(config.hidden_size, config.rms_norm_eps, zcg)
        if self.kind == "attention":
            self.mixer = OellmXSAAttention(config, vllm_config, f"{prefix}.mixer")
        elif self.kind == "ckda":
            self.mixer = OellmCKDA(config, vllm_config, f"{prefix}.mixer")
        elif self.kind == "moe":
            self.mixer = OellmMoE(config, f"{prefix}.mixer")
        else:
            self.mixer = OellmEngram(config, layer_idx, vllm_config, f"{prefix}.mixer")

    def _sublayer(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.kind == "engram":
            return self.mixer(x, input_ids)
        return self.mixer(self.input_layernorm(x))

    def forward(self, streams, input_ids: torch.Tensor):
        if self.ldm:
            x, r, memory, covariance = streams
            out = self._sublayer(x + r, input_ids)
            x = x + out
            r, memory, covariance = ldm_step(
                self.ldm_mixer.write_site, self.ldm_mixer.next_query, x, out, memory, covariance
            )
            return x, r, memory, covariance
        x, pending = self.mhc.read(streams)
        return self.mhc.write(streams, self._sublayer(x, input_ids), pending)


class OellmAmbitiousModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.ldm = config.residual == "ldm"
        self.ldm_boundary = OellmLDMBoundary(config) if self.ldm else None
        self.layers = nn.ModuleList(
            OellmAmbitiousLayer(config, i, vllm_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.norm = OellmRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.layernorm_zero_centered_gamma
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        assert get_pp_group().world_size == 1, "pipeline parallelism is not supported"
        assert input_ids is not None, "the Engram layers need input_ids"
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if self.ldm:
            streams = self.ldm_boundary.enter(x)
        else:
            n = self.config.mhc_expansion_rate
            streams = x.unsqueeze(-2).expand(x.shape[0], n, x.shape[-1]).contiguous()
        for layer in self.layers:
            streams = layer(streams, input_ids)
        h = streams[0] + streams[1] if self.ldm else streams.mean(dim=-2)
        return self.norm(h)


# Fields later exports added, with the defaults of the export's config class: an older export's
# bundled configuration file does not define them.
_CONFIG_DEFAULTS = {
    "residual": "mhc",
    "ldm_groups": 16,
    "ldm_key_dim": 8,
    "ldm_hidden_dim": 32,
    "ldm_alpha": 1.0,
    "layernorm_zero_centered_gamma": False,
    "router_balancing": "expert_bias",
    "moe_expert_post_act_norm": False,
    "mlp_post_act_norm": False,
    "post_act_norm_floor": 0.0,
    "post_act_norm_smooth": False,
    "engram_sign_hash": False,
    "engram_table_dtype": "bf16",
}


class OellmAmbitiousForCausalLM(nn.Module, HasInnerState, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        for key, value in _CONFIG_DEFAULTS.items():
            if getattr(config, key, None) is None:
                setattr(config, key, value)
        self.config = config
        self.vllm_config = vllm_config
        self.model = OellmAmbitiousModel(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head")
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def _ckda_kwargs(cls, vllm_config):
        return dict(vllm_config.model_config.hf_config.fla_layer["kwargs"])

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype, "float32"
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        kw = cls._ckda_kwargs(vllm_config)
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            kw.get("num_heads", 16),
            kw.get("head_dim", 128),
            conv_kernel_size=kw.get("conv_size", 4),
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    # -- weights: the export's names, mapped onto the TP-sharded modules
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        for name, w in weights:
            target, shard = self._map(name)
            if target in params:
                p = params[target]
                loader = getattr(p, "weight_loader", default_weight_loader)
                if shard is not None:
                    loader(p, w, shard)
                else:
                    loader(p, w)
                loaded.add(target)
            elif target in buffers:
                buffers[target].copy_(w.to(buffers[target].dtype))
                loaded.add(target)
            else:
                raise KeyError(f"no parameter for checkpoint tensor {name} (-> {target})")
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(f"parameters not in the checkpoint: {missing[:10]} ...")
        return loaded

    def _map(self, name: str) -> tuple[str, str | None]:
        if ".mixer." in name:
            head, tail = name.split(".mixer.", 1)
            kind = self.config.block_types[int(head.rsplit(".", 1)[-1])]
            pre = f"{head}.mixer."
            if kind == "attention":
                for proj, sh in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
                    if tail == f"{proj}.weight":
                        return pre + "qkv_proj.weight", sh
            elif kind == "ckda":
                tail = tail.removeprefix("layer.")
                tail = {
                    "f_proj.0.weight": "f_proj0.weight",
                    "f_proj.1.weight": "f_proj1.weight",
                    "g_proj.0.weight": "g_proj0.weight",
                    "g_proj.1.weight": "g_proj1.weight",
                    "g_proj.1.bias": "g_proj1.bias",
                    "o_norm.weight": "o_norm_weight",
                }.get(tail, tail)
            elif kind == "engram" and tail == "table":
                return pre + "table.weight", None
            return pre + tail, None
        return name, None
