"""
Pure-PyTorch, memory-efficient chunked fallback for NATTEN.
Requires no Triton compiler or FlexAttention, running in strict FP32
with row-chunking to stay within GPU VRAM limits.
"""

import os
import torch
import torch.nn as nn

_DEBUG = os.environ.get("NATTEN_DEBUG", "0") == "1" or os.environ.get("TRELLIS2_NATTEN_DEBUG", "0") == "1"
_CHUNK = int(os.environ.get("NATTEN_CHUNK", os.environ.get("TRELLIS2_NAF_CHUNK", "64")))


def _pair(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _offsets(H: int, W: int, kH: int, kW: int, dH: int, dW: int, device: torch.device):
    """Precomputes coordinate offsets and validity masks for neighborhood gathering."""
    ar = torch.arange
    out = []
    for a in range(kH):
        oh = (a - kH // 2) * dH
        hs = ar(H, device=device) + oh
        vh = (hs >= 0) & (hs < H)
        hsc = hs.clamp(0, H - 1)
        for b in range(kW):
            ow = (b - kW // 2) * dW
            ws = ar(W, device=device) + ow
            vw = (ws >= 0) & (ws < W)
            wsc = ws.clamp(0, W - 1)
            out.append((hsc, wsc, vh, vw, a, b))
    return out


def na2d_qk(query: torch.Tensor, key: torch.Tensor, kernel_size, dilation=1, rpb=None, is_causal: bool = False, **kwargs) -> torch.Tensor:
    """
    Computes Q-K attention logits:
    query, key: (B, heads, H, W, D) -> logits: (B, heads, H, W, K) in FP32.
    """
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)

    # Force FP32: AMD gfx1030 / RX 6800 has no native BF16 arithmetic
    qf = query.float()
    kf = key.float()
    B, hd, H, W, D = qf.shape
    K = kH * kW

    logits = torch.empty(B, hd, H, W, K, device=qf.device, dtype=torch.float32)
    offs = _offsets(H, W, kH, kW, dH, dW, qf.device)

    for h0 in range(0, H, _CHUNK):
        h1 = min(h0 + _CHUNK, H)
        qc = qf[:, :, h0:h1]
        for o, (hsc, wsc, vh, vw, a, b) in enumerate(offs):
            kg = kf[:, :, hsc[h0:h1]][:, :, :, wsc]
            dot = (qc * kg).sum(-1)
            if rpb is not None:
                dot = dot + rpb[:, (a - kH // 2) + (kH - 1), (b - kW // 2) + (kW - 1)].view(1, hd, 1, 1)
            dot = dot.masked_fill(~vh[h0:h1][:, None] | ~vw[None, :], float("-inf"))
            logits[:, :, h0:h1, :, o] = dot

    return logits


def na2d_av(attn: torch.Tensor, value: torch.Tensor, kernel_size, dilation=1, rpb=None, **kwargs) -> torch.Tensor:
    """
    Computes attention-weighted values:
    attn: (B, heads, H, W, K) [softmaxed]
    value: (B, heads, H, W, D)
    Returns: (B, heads, H, W, D) in value.dtype
    """
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)

    af = attn.float()
    vf = value.float()
    B, hd, H, W, D = vf.shape
    out = torch.zeros(B, hd, H, W, D, device=vf.device, dtype=vf.dtype)
    offs = _offsets(H, W, kH, kW, dH, dW, vf.device)

    for h0 in range(0, H, _CHUNK):
        h1 = min(h0 + _CHUNK, H)
        ac = af[:, :, h0:h1]
        acc = torch.zeros(B, hd, h1 - h0, W, D, device=vf.device, dtype=vf.dtype)
        for o, (hsc, wsc, vh, vw, a, b) in enumerate(offs):
            valid = (vh[h0:h1][:, None] & vw[None, :]).to(vf.dtype).view(1, 1, h1 - h0, W, 1)
            vg = vf[:, :, hsc[h0:h1]][:, :, :, wsc] * valid
            w = ac[:, :, :, :, o : o + 1].to(vf.dtype)
            acc = acc + w * vg
        out[:, :, h0:h1] = acc

    return out.to(value.dtype)


def na2d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size,
    stride=1,
    dilation=1,
    is_causal: bool = False,
    scale: float = None,
    **kwargs,
) -> torch.Tensor:
    """
    Functional na2d using pure PyTorch chunked execution.
    Accepts heads-last layout (B, H, W, Heads, D).
    """
    orig_dtype = query.dtype
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)

    B, H, W, N, D = query.shape
    if scale is None:
        scale = D ** -0.5

    qh = query.permute(0, 3, 1, 2, 4).contiguous().float()
    kh = key.permute(0, 3, 1, 2, 4).contiguous().float()
    vh = value.permute(0, 3, 1, 2, 4).contiguous().float()

    logits = na2d_qk(qh, kh, (kH, kW), (dH, dW)) * scale
    attn = logits.softmax(dim=-1)
    out = na2d_av(attn, vh, (kH, kW), (dH, dW))

    # Permute back to heads-last: (B, H, W, N, D)
    return out.permute(0, 2, 3, 1, 4).to(orig_dtype)


class NeighborhoodAttention2D(nn.Module):
    """Pure-PyTorch NeighborhoodAttention2D layer with chunked execution."""

    def __init__(self, dim: int, num_heads: int, kernel_size=7, dilation=1, bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = dim // num_heads
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bchw = False
        if x.dim() == 4 and x.shape[1] == self.dim:
            bchw = True
            B, C, H, W = x.shape
            flat = x.permute(0, 2, 3, 1)
        elif x.dim() == 4 and x.shape[-1] == self.dim:
            B, H, W, C = x.shape
            flat = x
        else:
            raise ValueError(f"[natten-shim] Unsupported input shape {x.shape} for dim={self.dim}")

        flat = flat.float()
        q, k, v = self.qkv(flat).chunk(3, dim=-1)
        q, k, v = [
            t.reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
            for t in (q, k, v)
        ]
        attn = na2d_qk(q, k, self.kernel_size, self.dilation).softmax(dim=-1)
        out = na2d_av(attn, v, self.kernel_size, self.dilation).permute(0, 2, 3, 1, 4).reshape(B, H, W, self.dim)
        out = self.proj(out)
        return out.permute(0, 3, 1, 2) if bchw else out
