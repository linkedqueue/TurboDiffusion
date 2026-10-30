"""Attention adapters for the H3 token layout."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str = "sdpa",
    scale: float | None = None,
    sparge_attn_topk: float = 0.5,
    sparge_attn_pre_sink_length: int = 0,
    sparge_attn_post_sink_length: int = 0,
) -> torch.Tensor:
    """Accept [batch, sequence, heads, dim] and preserve that layout."""
    if q.ndim != 4:
        raise ValueError(f"expected [B,S,H,D], got {tuple(q.shape)}")
    q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    if backend == "sparge_attn":
        try:
            from sparge_attn import spas_sage2_attn_topk_cuda
        except ImportError as exc:
            raise ImportError("Install SpargeAttn to use the H3 fullopt profiles") from exc
        out = spas_sage2_attn_topk_cuda(
            q, k, v, scale=scale, smooth_k=True, topk=float(sparge_attn_topk),
            pre_sink_length=int(sparge_attn_pre_sink_length),
            post_sink_length=int(sparge_attn_post_sink_length),
        )
    elif backend == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=False)
    else:
        raise ValueError(f"unsupported H3 attention backend: {backend}")
    return out.transpose(1, 2).contiguous()
