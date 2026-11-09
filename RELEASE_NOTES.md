# Release v0.1.0: Drop-in NATTEN on AMD ROCm (gfx1030) via FlexAttention

We are pleased to announce the initial release of **natten-rocm** (`v0.1.0`), providing a high-performance, drop-in replacement for NATTEN (Neighborhood Attention Extension) on AMD ROCm GPUs, with specialized optimizations for AMD RDNA2 architecture (**gfx1030 / Radeon RX 6800, RX 6800 XT, RX 6900 XT**).

## 🚀 What's New

### 1. PyTorch FlexAttention Implementation (`natten.flex`)
- **Exact NATTEN Border Rule:** Implements the sliding window border logic (`sy = min(clamp(qy - r, 0), Ly - k)`), matching NATTEN C++ behavior and eliminating mesh dusting and boundary artifacts.
- **Subgrid Dilation Grouping:** Efficiently groups dilation grids into batch dimensions with effective dilation 1.
- **Strict FP32 Execution:** Enforces single-precision arithmetic to bypass the lack of native hardware BF16 ALUs on gfx1030.
- **Zero-Leak Memory Management:** Automatic `gc.collect()` clears the compilation reference cycle created by `torch.compile(flex_attention)`, saving ~2.3 GiB per compiled shape.

### 2. Pure-PyTorch Chunked Fallback (`natten.shim`)
- Pure PyTorch fallback requiring zero Triton or compiler dependencies.
- Chunked row processing (`_CHUNK = 64`) to execute large resolution neighborhood attention within constrained GPU VRAM.

### 3. Transparent Zero-Code-Change Monkeypatch (`natten.install()`)
- Injects directly into `sys.modules["natten"]` and configures fallback behaviors.
- Allows models such as **ValeoAI NAF** (`torch.hub.load("valeoai/NAF", "naf")`) and **ComfyUI-Trellis2** to run out-of-the-box on ROCm without editing upstream model code.

### 4. Mathematical Verification Suite
- `tests/test_flex_na2d.py`: Verifies FlexAttention na2d against double-precision C++ ground truth formulas (max error `< 3.6e-7`).
- `tests/test_bands.py`: Proves row-band tiling exactness across 2, 4, 8, and 16 vertical bands.
- `tests/compare_shim_vs_flex.py`: Isolates and quantifies boundary behavior between centered-window zero-padding and NATTEN window sliding.

## 📦 How to Install

```bash
pip install git+https://github.com/0xra0/natten-rocm.git@v0.1.0
```

Or for local development:
```bash
git clone https://github.com/0xra0/natten-rocm.git
cd natten-rocm
pip install -e .
```

## 🏷️ GitHub Release Metadata
- **Tag:** `v0.1.0`
- **Target Branch:** `main`
- **Release Title:** `v0.1.0: Drop-in NATTEN on AMD ROCm (gfx1030) via FlexAttention`
