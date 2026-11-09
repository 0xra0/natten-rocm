"""
Runtime monkey-patching helper for NATTEN.
Allows transparent drop-in replacement into sys.modules['natten'] without
modifying upstream model code (e.g. valeoai/NAF, Trellis2, ComfyUI).
"""

import importlib.machinery
import sys
import types
from .flex import na2d as flex_na2d, NeighborhoodAttention2D as FlexNA2D, is_flex_available
from .shim import na2d as shim_na2d, na2d_qk, na2d_av, NeighborhoodAttention2D as ShimNA2D


def install(backend: str = "auto"):
    """
    Patches sys.modules['natten'] with this ROCm-optimized implementation.

    Args:
        backend: "auto" (default: flex if available, else shim),
                 "flex" (FlexAttention with exact border rule),
                 "shim" (Pure PyTorch chunked fallback).
    """
    if backend == "auto":
        backend = "flex" if is_flex_available() else "shim"

    mod = types.ModuleType("natten")
    mod.__spec__ = importlib.machinery.ModuleSpec("natten", loader=None)
    mod.__version__ = "0.21.0"
    mod.__file__ = __file__

    # Hardware capabilities reporting
    mod.has_cuda = lambda *a, **k: True
    mod.has_fp32 = lambda *a, **k: True
    mod.has_bfloat = lambda *a, **k: False
    mod.has_half = lambda *a, **k: False
    mod.has_gemm = lambda *a, **k: False
    mod.has_tf32 = lambda *a, **k: False
    mod.has_64bit_indexing = lambda *a, **k: True

    # No-op setters for compatibility
    for name in (
        "set_kernel_choice",
        "set_memory_usage",
        "set_tiling",
        "reset_tiling",
        "reset_kernel_choice",
        "set_floating_point_ops_mode",
    ):
        setattr(mod, name, lambda *a, **k: None)

    if backend == "flex":
        mod._backend = "flex"
        mod._flex_shim = True
        mod.na2d = flex_na2d
        mod.NeighborhoodAttention2D = FlexNA2D
        # For NAF: leave __path__ empty and remove natten.functional so NAF
        # falls back to the modern `from natten import na2d` path
        mod.__path__ = []
        sys.modules["natten"] = mod
        sys.modules.pop("natten.functional", None)
        print("[natten-rocm] Patched sys.modules['natten'] -> FlexAttention backend (NATTEN border rule, FP32)")
    elif backend == "shim":
        mod._backend = "shim"
        mod._pure_torch_shim = True
        mod.na2d = shim_na2d
        mod.NeighborhoodAttention2D = ShimNA2D
        mod.__path__ = []
        func = types.ModuleType("natten.functional")
        func.__spec__ = importlib.machinery.ModuleSpec("natten.functional", loader=None)
        func.__file__ = __file__
        func.na2d_qk = na2d_qk
        func.na2d_av = na2d_av
        mod.functional = func
        sys.modules["natten"] = mod
        sys.modules["natten.functional"] = func
        print("[natten-rocm] Patched sys.modules['natten'] -> Pure PyTorch chunked shim (FP32)")
    else:
        raise ValueError(f"Unknown backend '{backend}'. Expected 'auto', 'flex', or 'shim'.")

    return mod
