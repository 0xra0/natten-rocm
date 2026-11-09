"""
Tests that row-band tiling (slicing an image vertically with halo padding)
produces outputs bitwise/numerically identical to processing the full image.
Critical for running high-resolution (1024px+) NAF features on 16GB VRAM GPUs (like RX 6800).
"""

import sys
import torch
import natten.flex as nf


def nearest_exact(n_out: int, n_in: int, device: str):
    return ((torch.arange(n_out, device=device) + 0.5) * n_in / n_out).floor().long().clamp(max=n_in - 1)


def check_bands(H: int, W: int, hk: int, wk: int, n_bands: int, keep_margin: int = 2, dev: str = "cuda"):
    K, N, D = 9, 2, 16
    dil = (H // hk, W // wk)
    halo = (K // 2) * dil[0]

    q = torch.randn(1, H, W, N, D, device=dev)
    klr = torch.randn(1, hk, wk, N, D, device=dev)
    vlr = torch.randn(1, hk, wk, N, D, device=dev)

    rows_all = nearest_exact(H, hk, dev)
    cols = nearest_exact(W, wk, dev)
    k, v = klr[:, rows_all][:, :, cols], vlr[:, rows_all][:, :, cols]

    full = nf.na2d(q, k, v, (K, K), dilation=dil)
    worst = 0.0

    for b in range(n_bands):
        c0, c1 = H * b // n_bands, H * (b + 1) // n_bands
        k0, k1 = max(0, c0 - keep_margin), min(H, c1 + keep_margin)
        a0, a1 = max(0, k0 - halo), min(H, k1 + halo)
        r = rows_all[a0:a1]

        out = nf.na2d(q[:, a0:a1], klr[:, r][:, :, cols], vlr[:, r][:, :, cols], (K, K), dilation=dil)
        diff = (out[:, k0 - a0 : k1 - a0] - full[:, k0:k1]).abs().max().item()
        worst = max(worst, diff)

    ok = worst < 1e-4
    status = "OK" if ok else "MISMATCH"
    print(f"H={H:<4d} kv={hk:<3d} d={str(dil):<10s} halo={halo:<3d} bands={n_bands:<2d} rows/band={H // n_bands:<3d} | max_err={worst:.2e} [{status}]")
    return ok


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("[SKIP] Row-band test requires CUDA/ROCm device.")
        return

    print("\n--- Testing Row-Band Tiling Exactness ---")
    torch.manual_seed(42)
    all_ok = True
    with torch.no_grad():
        for nb in (1, 2, 4, 8, 16):
            all_ok &= check_bands(256, 256, 32, 32, nb, dev=dev)

    print(f"\nResult: {'ALL BAND CONFIGS EXACT' if all_ok else 'BAND MISMATCH DETECTED'}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
