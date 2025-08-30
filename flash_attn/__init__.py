"""
Lightweight stub of the flash_attn package to satisfy optional imports
when using non-flash attention backends (e.g., SDPA or eager).

This stub is only meant to bypass import-time checks in upstream model code.
It does not provide FlashAttention kernels. If you select a flash attention
backend at runtime, you must install the real 'flash-attn' package.
"""

__all__ = []
__version__ = "0.0.0-stub"

