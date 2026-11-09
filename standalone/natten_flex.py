# Standalone FlexAttention replacement for `natten` on AMD ROCm (gfx1030).
# Exposes `natten.na2d` with heads-last layout (B, H, W, heads, D) and NATTEN border rule.
# Call `install()` to patch `sys.modules["natten"]` for zero-code-change NAF execution.

import gc
import importlib.machinery
import os
import sys
import types

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_DEBUG = os.environ.get("TRELLIS2_NATTEN_DEBUG", "0") == "1"
_flex = torch.compile(flex_attention, dynamic=False)
_masks = {}


def _pair(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _block_mask(B, H, W, kH, kW, dH, dW, device):
    key = (B, H, W, kH, kW, dH, dW, str(device))
    if key in _masks:
        return _masks[key]
    Hp, Wp = -(-H // dH), -(-W // dW)
    G = dH * dW
    rH, rW = kH // 2, kW // 2

    # Regroup dilation groups to batch dimension: grid of Hp x Wp.
    # NATTEN edge-sliding window: sy = min(clamp(qy - rH, min=0), Ly - kH).
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

    m = create_block_mask(mask_mod, B * G, None, Hp * Wp, Hp * Wp, device=device, _compile=True)
    _masks[key] = m
    return m


def _to_groups(x, dH, dW):
    B, H, W, N, D = x.shape
    Hp, Wp = -(-H // dH), -(-W // dW)
    x = F.pad(x, (0, 0, 0, 0, 0, Wp * dW - W, 0, Hp * dH - H))
    x = x.view(B, Hp, dH, Wp, dW, N, D).permute(0, 2, 4, 5, 1, 3, 6)
    return x.reshape(B * dH * dW, N, Hp * Wp, D)


def _from_groups(x, B, H, W, dH, dW):
    _, N, _, D = x.shape
    Hp, Wp = -(-H // dH), -(-W // dW)
    x = x.view(B, dH, dW, N, Hp, Wp, D).permute(0, 4, 1, 5, 2, 3, 6)
    return x.reshape(B, Hp * dH, Wp * dW, N, D)[:, :H, :W]


def na2d(query, key, value, kernel_size, stride=1, dilation=1, is_causal=False, scale=None,
         additional_keys=None, additional_values=None, backend=None, **tuning):
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
    if _pair(stride) != (1, 1) or is_causal or additional_keys is not None or additional_values is not None:
        raise NotImplementedError("[natten-flex] only stride=1, non-causal, no additional KV")
    if kH % 2 == 0 or kW % 2 == 0:
        raise NotImplementedError(f"[natten-flex] odd kernel_size only, got {kernel_size}")
    B, H, W, N, D = query.shape
    if H // dH < kH or W // dW < kW:
        raise ValueError(f"[natten-flex] input {H}x{W} too small for kernel ({kH},{kW}) dilation ({dH},{dW})")
    if scale is None:
        scale = D ** -0.5

    # gfx1030 / RX 6800 has no native BF16 arithmetic: keep attention in FP32
    q, k, v = (_to_groups(t.float(), dH, dW) for t in (query, key, value))
    n_masks = len(_masks)
    out = _flex(q, k, v, block_mask=_block_mask(B, H, W, kH, kW, dH, dW, q.device), scale=scale)

    del q, k, v
    if len(_masks) != n_masks:
        # torch.compile leaves reference cycle on new shapes; gc.collect frees ~2.3 GB
        gc.collect()

    if _DEBUG:
        print(f"[natten-flex] na2d q={tuple(query.shape)} v={tuple(value.shape)} kernel=({kH},{kW}) dilation=({dH},{dW})")
    return _from_groups(out, B, H, W, dH, dW).to(query.dtype)


def install():
    if sys.modules.get("natten") is not None and getattr(sys.modules["natten"], "_flex_shim", False):
        return
    mod = types.ModuleType("natten")
    mod._flex_shim = True
    mod.__spec__ = importlib.machinery.ModuleSpec("natten", loader=None)
    mod.__path__ = []  # no submodules: import natten.functional fails -> NAF takes na2d path
    mod.__version__ = "0.21.0"
    mod.__file__ = __file__
    mod.na2d = na2d
    for nm in ("has_cuda", "has_half", "has_gemm", "has_tf32", "has_64bit_indexing"):
        setattr(mod, nm, lambda *a, **k: False)
    mod.has_fp32 = lambda *a, **k: True
    mod.has_bfloat = lambda *a, **k: False
    for nm in ("set_kernel_choice", "set_memory_usage", "set_tiling", "reset_tiling", "reset_kernel_choice", "set_floating_point_ops_mode"):
        setattr(mod, nm, lambda *a, **k: None)
    sys.modules["natten"] = mod
    sys.modules.pop("natten.functional", None)
    print("[natten-flex] installed FlexAttention natten (NATTEN border rule, FP32)")
