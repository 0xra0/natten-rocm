"""
Benchmark measuring GPU VRAM consumption for full-image vs. row-banded execution
on high-resolution images (512x512 and 1024x1024).
"""

import sys
import torch
import natten.flex as nf

GiB = 1024 ** 3


def benchmark_vram(H=1024, W=1024, hk=64, wk=64, bands=4, dev="cuda"):
    if not torch.cuda.is_available() or dev != "cuda":
        print("[SKIP] Benchmark requires a CUDA/ROCm GPU.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    K, N, D = 9, 4, 64
    dil = (H // hk, W // wk)
    halo = (K // 2) * dil[0]

    print(f"\n--- Benchmarking VRAM: {H}x{W} (Kernel={K}x{K}, Dilation={dil}, Heads={N}, Dim={D}) ---")

    q = torch.randn(1, H, W, N, D, device=dev)
    k = torch.randn(1, H, W, N, D, device=dev)
    v = torch.randn(1, H, W, N, D, device=dev)
    base_mem = torch.cuda.memory_allocated() / GiB
    print(f"Base allocation (Tensors): {base_mem:.2f} GiB")

    # 1. Row-band tiled execution
    torch.cuda.reset_peak_memory_stats()
    band_outputs = []
    for b in range(bands):
        c0, c1 = H * b // bands, H * (b + 1) // bands
        a0, a1 = max(0, c0 - halo), min(H, c1 + halo)

        out_band = nf.na2d(q[:, a0:a1], k[:, a0:a1], v[:, a0:a1], (K, K), dilation=dil)
        band_outputs.append(out_band[:, c0 - a0 : c1 - a0])

    tiled_peak = torch.cuda.max_memory_allocated() / GiB
    print(f"Tiled ({bands} bands) Peak VRAM: {tiled_peak:.2f} GiB (Net overhead: {tiled_peak - base_mem:.2f} GiB)")

    del band_outputs
    torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark_vram(H=512, W=512, bands=2)
    benchmark_vram(H=1024, W=1024, bands=4)
