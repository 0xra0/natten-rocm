"""
natten-flex-rocm: High-performance drop-in replacement for NATTEN on AMD ROCm (gfx1030).
Supports exact NATTEN border sliding rules via PyTorch FlexAttention and pure PyTorch chunking.
"""

from .flex import is_flex_available, na2d as _flex_na2d, NeighborhoodAttention2D as _FlexNA2D
from .shim import na2d as _shim_na2d, NeighborhoodAttention2D as _ShimNA2D
from .patch import install

__version__ = "0.21.0"
__all__ = [
    "na2d",
    "NeighborhoodAttention2D",
    "install",
    "is_flex_available",
    "has_cuda",
    "has_fp32",
    "has_bfloat",
    "has_half",
    "has_gemm",
    "has_tf32",
    "has_64bit_indexing",
]

# Select default backend: prefer FlexAttention if available
if is_flex_available():
    na2d = _flex_na2d
    NeighborhoodAttention2D = _FlexNA2D
    _DEFAULT_BACKEND = "flex"
else:
    na2d = _shim_na2d
    NeighborhoodAttention2D = _ShimNA2D
    _DEFAULT_BACKEND = "shim"


def has_cuda() -> bool:
    return True


def has_fp32() -> bool:
    return True


def has_bfloat() -> bool:
    return False


def has_half() -> bool:
    return False


def has_gemm() -> bool:
    return False


def has_tf32() -> bool:
    return False


def has_64bit_indexing() -> bool:
    return True


def set_kernel_choice(*args, **kwargs):
    pass


def set_memory_usage(*args, **kwargs):
    pass


def set_tiling(*args, **kwargs):
    pass


def reset_tiling(*args, **kwargs):
    pass


def reset_kernel_choice(*args, **kwargs):
    pass


def set_floating_point_ops_mode(*args, **kwargs):
    pass
