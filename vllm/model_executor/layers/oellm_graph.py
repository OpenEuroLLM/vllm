# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded local graph islands; no CPU lookup or distributed capture inside."""

import torch

from vllm.platforms import current_platform


class DecodeGraph:
    """Stage changing inputs and replay a shape-specific local CUDA graph.

    The initial eager call warms kernels and advances state once. Capture only
    records kernels: it is not replayed on that call, so recurrent state is not
    advanced a second time. Outputs remain valid until this island's next call.
    Islands replay sequentially in model order; their private-pool scratch can
    therefore be shared. Strong output references preserve cross-island values.
    """

    def __init__(self, max_entries=8):
        self.entries = {}
        self.max_entries = max_entries
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0

    def __call__(self, fn, *args, key=()):
        if torch.cuda.is_current_stream_capturing():
            return fn(*args)
        signature = (key, tuple((x.shape, x.stride(), x.dtype, x.device) for x in args))
        entry = self.entries.get(signature)
        if entry is None:
            if len(self.entries) >= self.max_entries:
                self.fallbacks += 1
                return fn(*args)
            eager = fn(*args)
            static = tuple(x.detach().clone() for x in args)
            stream = torch.cuda.Stream(device=args[0].device)
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.stream(stream),
                torch.cuda.graph(
                    graph, stream=stream, pool=current_platform.get_global_graph_pool()
                ),
            ):
                output = fn(*static)
            torch.cuda.current_stream().wait_stream(stream)
            self.entries[signature] = (graph, static, output)
            self.captures += 1
            return eager
        graph, static, output = entry
        for dest, source in zip(static, args):
            dest.copy_(source)
        graph.replay()
        self.replays += 1
        return output
