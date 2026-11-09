"""
FlexAttention implementation of Neighborhood Attention (NATTEN).
Specifically optimized for AMD ROCm and RDNA2 (gfx1030 / RX 6800 / 6800 XT / 6900 XT).

Key design features:
1. Exact NATTEN border rule: The attention window slides inward at boundary edges
   (sy = min(max(qy - r, 0), Ly - k)), ensuring queries near boundaries always
   attend to exactly kernel_size key tokens.
2. Dilation subgrid grouping: Reshapes dilated grids into subgrids with effective
   dilation 1, handling non-divisible dimensions gracefully.
3. FP32 Precision: Enforces FP32 computation for GPUs lacking native hardware BF16
   arithmetic (e.g. AMD gfx1030).
4. Zero-Leak Memory Management: Invokes gc.collect() after first-time compilation
   on new tensor shapes to clear PyTorch's torch.compile FlexAttention reference cycle
   (freeing ~2.3 GiB per shape compile).
"""

import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_FLEX_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _flex_compiled = torch.compile(flex_attention, dynamic=False)
    _FLEX_AVAILABLE = True
except (ImportError, AttributeError):
    _flex_compiled = None

_DEBUG = os.environ.get("NATTEN_DEBUG", "0") == "1" or os.environ.get("TRELLIS2_NATTEN_DEBUG", "0") == "1"
_masks = {}


def is_flex_available() -> bool:
    return _FLEX_AVAILABLE


def _pair(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _block_mask(B: int, H: int, W: int, kH: int, kW: int, dH: int, dW: int, device: torch.device):
    """
    Constructs a block mask for FlexAttention matching NATTEN's sliding border rule.
    Caches compiled block masks by dimension and dilation.
    """
    key = (B, H, W, kH, kW, dH, dW, str(device))
    if key in _masks:
        return _masks[key]

    Hp, Wp = -(-H // dH), -(-W // dW)
    G = dH * dW
    rH, rW = kH // 2, kW // 2

    # Tokens are regrouped so each dilation group is its own batch entry: a plain
    # dilation-1 grid of Hp x Wp. A group's real length can be one short of Hp
    # (H not divisible by dH); the window is clamped to the real length, so the
    # padded tail is never attended to.
    def mask_mod(b, h, q_idx, kv_idx):
        g = b % G
        gy, gx = g // dW, g % dW
        Ly = (H - gy + dH - 1) // dH
        Lx = (W - gx + dW - 1) // dW
        qy, qx = q_idx // Wp, q_idx % Wp
        ky, kx = kv_idx // Wp, kv_idx % Wp
        sy = torch.minimum((qy - rH).clamp(min=0), Ly - kH)
        sx = torch.minimum((qx - rW).clamp(min=0), Lx - kW)
        return (ky >= sy) & (ky < sy + kH) & (kx >= sx) & (kx < sx + kW)

    # _compile=True builds the mask block by block; the eager path would materialize a
    # dense (B*G, Q, KV) boolean tensor (~1.5 GB for 448px image, d=14).
    m = create_block_mask(mask_mod, B * G, None, Hp * Wp, Hp * Wp, device=device, _compile=True)
    _masks[key] = m
    return m


def _to_groups(x: torch.Tensor, dH: int, dW: int) -> torch.Tensor:
    """Reshape (B, H, W, N, D) -> (B * dH * dW, N, Hp * Wp, D) for subgrid attention."""
    B, H, W, N, D = x.shape
    Hp, Wp = -(-H // dH), -(-W // dW)
    x = F.pad(x, (0, 0, 0, 0, 0, Wp * dW - W, 0, Hp * dH - H))
    x = x.view(B, Hp, dH, Wp, dW, N, D).permute(0, 2, 4, 5, 1, 3, 6)
    return x.reshape(B * dH * dW, N, Hp * Wp, D)


def _from_groups(x: torch.Tensor, B: int, H: int, W: int, dH: int, dW: int) -> torch.Tensor:
    """Inverse of _to_groups: (B * dH * dW, N, Hp * Wp, D) -> (B, H, W, N, D)."""
    _, N, _, D = x.shape
    Hp, Wp = -(-H // dH), -(-W // dW)
    x = x.view(B, dH, dW, N, Hp, Wp, D).permute(0, 4, 1, 5, 2, 3, 6)
    return x.reshape(B, Hp * dH, Wp * dW, N, D)[:, :H, :W]


def na2d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size,
    stride=1,
    dilation=1,
    is_causal: bool = False,
    scale: float = None,
    additional_keys=None,
    additional_values=None,
    backend=None,
    **kwargs,
) -> torch.Tensor:
    """
    2D Neighborhood Attention via PyTorch FlexAttention.

    Args:
        query: Query tensor of shape (B, H, W, Heads, Dim)
        key: Key tensor of shape (B, H, W, Heads, Dim)
        value: Value tensor of shape (B, H, W, Heads, Dim_v)
        kernel_size: Neighborhood window size (int or (kH, kW)), must be odd
        stride: Stride (must be 1)
        dilation: Dilation factor (int or (dH, dW))
        is_causal: Causal masking (must be False)
        scale: Scaling factor (defaults to 1.0 / sqrt(Dim))

    Returns:
        Attention output tensor of shape (B, H, W, Heads, Dim_v) in query.dtype
    """
    if not _FLEX_AVAILABLE:
        raise RuntimeError("torch.nn.attention.flex_attention is not available in your PyTorch build.")

    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)

    if _pair(stride) != (1, 1) or is_causal or additional_keys is not None or additional_values is not None:
        raise NotImplementedError("[natten-flex] Only stride=1, non-causal, no additional KV is supported.")
    if kH % 2 == 0 or kW % 2 == 0:
        raise NotImplementedError(f"[natten-flex] Odd kernel_size required, got {kernel_size}")

    B, H, W, N, D = query.shape
    if H // dH < kH or W // dW < kW:
        raise ValueError(f"[natten-flex] Input {H}x{W} too small for kernel ({kH},{kW}) and dilation ({dH},{dW})")

    if scale is None:
        scale = D ** -0.5

    # gfx1030 / RX 6800 has no native BF16 arithmetic: keep attention in FP32
    orig_dtype = query.dtype
    q = _to_groups(query.float(), dH, dW)
    k = _to_groups(key.float(), dH, dW)
    v = _to_groups(value.float(), dH, dW)

    n_masks = len(_masks)
    mask = _block_mask(B, H, W, kH, kW, dH, dW, q.device)
    out = _flex_compiled(q, k, v, block_mask=mask, scale=scale)

    # Free regrouped copies before _from_groups creates the reconstructed tensor
    del q, k, v

    # When a new shape is encountered, torch.compile leaves a reference cycle
    # holding q/k/v (~2.3 GB for large shapes). Cyclic GC cleans it up immediately.
    if len(_masks) != n_masks:
        gc.collect()

    if _DEBUG:
        print(f"[natten-flex] na2d q={tuple(query.shape)} v={tuple(value.shape)} kernel=({kH},{kW}) dilation=({dH},{dW})")

    return _from_groups(out, B, H, W, dH, dW).to(orig_dtype)


class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Layer using FlexAttention backend.
    Accepts both (B, C, H, W) and (B, H, W, C) input tensor layouts.
    """

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
            raise ValueError(f"[natten-flex] Unsupported input shape {x.shape} for dim={self.dim}")

        qkv = self.qkv(flat.float())
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, H, W, self.num_heads, self.head_dim)
        k = k.reshape(B, H, W, self.num_heads, self.head_dim)
        v = v.reshape(B, H, W, self.num_heads, self.head_dim)

        out = na2d(q, k, v, kernel_size=self.kernel_size, dilation=self.dilation)
        out = self.proj(out.reshape(B, H, W, self.dim))
        return out.permute(0, 3, 1, 2) if bchw else out
