# Standalone Pure-PyTorch chunked replacement for `natten` on AMD ROCm.
# Computes neighborhood attention in FP32 with row chunks to fit within GPU VRAM.

import importlib.machinery
import os
import sys
import types
import torch
import torch.nn as nn

_DEBUG = os.environ.get("TRELLIS2_NATTEN_DEBUG", "0") == "1"
_CHUNK = int(os.environ.get("TRELLIS2_NAF_CHUNK", "64"))


def _pair(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _offsets(H, W, kH, kW, dH, dW, device):
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


def na2d_qk(query, key, kernel_size, dilation=1, rpb=None, is_causal=False, **kw):
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
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


def na2d_av(attn, value, kernel_size, dilation=1, rpb=None, **kw):
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
    return out


class NeighborhoodAttention2D(nn.Module):
    def __init__(self, dim=None, num_heads=None, kernel_size=7, dilation=1, *a, **k):
        super().__init__()
        self.dim = k.get("dim", dim)
        self.num_heads = k.get("num_heads", num_heads)
        self.head_dim = self.dim // self.num_heads
        self.kernel_size = k.get("kernel_size", kernel_size)
        self.dilation = k.get("dilation", dilation)
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        bchw = False
        if x.dim() == 4 and x.shape[1] == self.dim:
            bchw = True
            B, C, H, W = x.shape
            flat = x.permute(0, 2, 3, 1)
        elif x.dim() == 4 and x.shape[-1] == self.dim:
            B, H, W, C = x.shape
            flat = x
        else:
            raise ValueError(f"Bad input shape {x.shape} for dim={self.dim}")
        flat = flat.float()
        q, k, v = self.qkv(flat).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous() for t in (q, k, v)]
        attn = na2d_qk(q, k, self.kernel_size, self.dilation).softmax(dim=-1)
        out = na2d_av(attn, v, self.kernel_size, self.dilation).permute(0, 2, 3, 1, 4).reshape(B, H, W, self.dim)
        out = self.proj(out)
        return out.permute(0, 3, 1, 2) if bchw else out


def install():
    if sys.modules.get("natten") is not None and getattr(sys.modules["natten"], "_pure_torch_shim", False):
        return
    mod = types.ModuleType("natten")
    mod._pure_torch_shim = True
    mod.__spec__ = importlib.machinery.ModuleSpec("natten", loader=None)
    mod.__path__ = []
    mod.__version__ = "0.14.0"
    mod.__file__ = __file__
    func = types.ModuleType("natten.functional")
    func.__spec__ = importlib.machinery.ModuleSpec("natten.functional", loader=None)
    func.__file__ = __file__
    func.na2d_qk = na2d_qk
    func.na2d_av = na2d_av
    mod.functional = func
    mod.na2d = NeighborhoodAttention2D
    mod.NeighborhoodAttention2D = NeighborhoodAttention2D
    for nm in ("has_cuda", "has_half", "has_gemm", "has_tf32", "has_64bit_indexing"):
        setattr(mod, nm, lambda *a, **k: False)
    mod.has_fp32 = lambda *a, **k: True
    mod.has_bfloat = lambda *a, **k: False
    for nm in ("set_kernel_choice", "set_memory_usage", "set_tiling", "reset_tiling", "reset_kernel_choice", "set_floating_point_ops_mode"):
        setattr(mod, nm, lambda *a, **k: None)
    sys.modules["natten"] = mod
    sys.modules["natten.functional"] = func
    print(f"[natten-shim] installed pure-torch natten (chunk={_CHUNK}, FP32 ONLY)")
