"""
Functional interface compatible with upstream NATTEN's functional module.
Provides na2d_qk and na2d_av.
"""

from .shim import na2d_qk, na2d_av

__all__ = ["na2d_qk", "na2d_av"]
