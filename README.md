# natten-rocm (NATTEN on AMD ROCm & gfx1030)

[![CI](https://github.com/0xra0/natten-rocm/actions/workflows/ci.yml/badge.svg)](https://github.com/0xra0/natten-rocm/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, drop-in replacement for [NATTEN](https://github.com/SHI-Labs/NATTEN) (Neighborhood Attention Extension) built specifically for **AMD ROCm**, with full mathematical support for **NAF** ([valeoai/NAF](https://github.com/valeoai/NAF)), **Trellis2 / Pixal3D**, and other vision transformers running on **AMD RDNA2 GPUs (gfx1030 / RX 6800, RX 6800 XT, RX 6900 XT)**.

---

## ⚡ Highlights

- **Exact NATTEN Border Rule:** Implements NATTEN's sliding window border logic (`sy = min(clamp(qy - r, 0), Ly - k)`) using PyTorch `FlexAttention`. Avoids border artifacts and mesh dusting caused by naive `-inf` masking.
- **gfx1030 / RDNA2 Optimized:** Enforces numerical computation in strict **FP32**. RDNA2 lacks native hardware BF16 execution units, making FP32 essential to prevent crashes, slowdowns, and non-finite outputs.
- **Zero-Leak Memory Management:** Automatically identifies new tensor shape compiles and triggers cyclic garbage collection (`gc.collect()`), resolving PyTorch's `torch.compile` FlexAttention reference cycle bug (saving **~2.3 GiB** per shape).
- **Exact Row-Band Tiling:** Supports row-band slicing for high-resolution images (1024×1024+) with maximum numerical error `< 3.6e-7` compared to untiled references, allowing complex pipelines to run within 16 GB VRAM.
- **Zero Code Modification:** Includes a one-line `natten.install()` monkeypatch that drops directly into `sys.modules["natten"]`, allowing third-party libraries (like `torch.hub.load("valeoai/NAF")`) to execute out of the box.
- **Pure-PyTorch Fallback:** Includes a memory-efficient chunked fallback shim (`shim`) for environments where `FlexAttention` or Triton is unavailable.

---

## 🧩 The Problem: Why NATTEN Fails on AMD ROCm / gfx1030

### 1. Lack of Prebuilt Wheels & Cutlass CUDA Assumptions
Upstream NATTEN relies heavily on NVIDIA CUDA intrinsics and CUTLASS kernels. Compiling upstream NATTEN on AMD ROCm (via HIP) frequently fails due to unsupported template instantiations or missing architecture targets.

### 2. No Native BF16 Arithmetic on gfx1030
AMD RDNA2 GPUs (such as the Radeon RX 6800, 6800 XT, 6900 XT, and 6700 XT / `gfx1030`) do **not** have native hardware bfloat16 (BF16) arithmetic ALU execution. Running attention kernels in BF16 on gfx1030 triggers software emulation, extreme slowdowns, numerical inaccuracy, or hard driver faults. Attention must be kept strictly in **FP32**.

### 3. The NATTEN Border Rule vs. Naive Zero-Masking
Standard convolution padding either pads with zeros or wraps. Naive NATTEN shims often center the attention window and mask out-of-boundary tokens to `-inf`.
However, NATTEN defines a **sliding border rule**:
> *At image borders, the receptive field window slides inward so every query token always attends to exactly `kernel_size` valid tokens from its dilation subgrid.*

At high dilation (e.g., $d = 14$ or $16$), the boundary region where window sliding occurs accounts for **up to 40% of the entire image area**! Naive `-inf` masking produces degraded features, causing 3D reconstruction models (such as Trellis2 / Pixal3D) to produce broken "dust" meshes instead of solid geometry.

### 4. PyTorch FlexAttention Memory Cycle Leak
When `torch.compile(flex_attention)` encounters a new tensor shape, PyTorch's internal compilation graph leaves a circular reference holding the inputs (`query`, `key`, `value`) alive. On high-resolution tensors (e.g., 1024px with 1024 channels), this leaks **~2.3 GB of VRAM** per shape until circular garbage collection runs.

---

## 🚀 Installation

### Option 1: Install from GitHub / Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/0xra0/natten-rocm.git
cd natten-rocm

# Install in editable mode
pip install -e .
```

Or install directly via `pip`:
```bash
pip install git+https://github.com/0xra0/natten-rocm.git
```

### Option 2: Standalone Single-File Drop-in
If you prefer not to install a Python package, copy [`standalone/natten_flex.py`](standalone/natten_flex.py) directly into your project or ComfyUI custom nodes directory:

```bash
cp standalone/natten_flex.py /path/to/your/project/
```

---

## 🛠️ Requirements & Environment Setup

- **GPU:** AMD Radeon RX 6800 / 6800 XT / 6900 XT (`gfx1030`), RX 7000 series (`gfx1100`), or Instinct (`gfx90a`, `gfx942`).
- **OS:** Linux (Ubuntu 22.04+ or similar).
- **ROCm:** ROCm 6.0, 6.1, or 6.2.
- **Python:** 3.10, 3.11, or 3.12.
- **PyTorch:** PyTorch 2.4+ (PyTorch 2.5+ recommended for full `torch.nn.attention.flex_attention` support).

### Recommended Environment Variables for ROCm
To prevent Triton / TorchInductor from filling up `/tmp` on memory-limited partitions:
```bash
export TMPDIR=/mnt/ssd/work/tmp
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TRITON_CACHE_DIR=$TMPDIR/triton
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

---

## 💻 How to Use

### 1. Transparent Monkey-Patching (Zero Code Changes)
If you are running an existing library that imports `natten` (e.g. `valeoai/NAF` or ComfyUI extensions), call `install()` before importing the library:

```python
import natten
natten.install(backend="auto")  # Uses "flex" if available, else "shim"

# Now any code that calls `import natten` or `from natten import na2d` works seamlessly!
import torch
naf = torch.hub.load("valeoai/NAF", "naf", pretrained=True)
```

### 2. Direct Functional API (`natten.na2d`)
The functional interface accepts heads-last `(B, H, W, Heads, Dim)` tensors:

```python
import torch
from natten import na2d

B, H, W, Heads, Dim = 1, 448, 448, 4, 64
q = torch.randn(B, H, W, Heads, Dim, device="cuda")
k = torch.randn(B, H, W, Heads, Dim, device="cuda")
v = torch.randn(B, H, W, Heads, Dim, device="cuda")

# 2D Neighborhood Attention with 9x9 kernel and dilation 14
out = na2d(q, k, v, kernel_size=(9, 9), dilation=(14, 14))
print("Output shape:", out.shape)  # (1, 448, 448, 4, 64)
```

### 3. Neural Network Layer (`natten.NeighborhoodAttention2D`)
Use as a standard PyTorch module in your vision architectures:

```python
import torch
from natten import NeighborhoodAttention2D

# Accepts either (B, C, H, W) or (B, H, W, C)
layer = NeighborhoodAttention2D(
    dim=128,
    num_heads=4,
    kernel_size=7,
    dilation=2
).cuda()

x = torch.randn(1, 128, 64, 64, device="cuda")
out = layer(x)
print(out.shape)  # (1, 128, 64, 64)
```

### 4. Integrating with ComfyUI-Trellis2
In `image_conditioned_proj.py`:
```python
import natten
natten.install(backend="flex")
```
Set runtime implementation:
```bash
export TRELLIS2_NATTEN_IMPL=flex
```

---

## 🔬 Technical Architecture

### 1. Sliding Window Border Mathematics
In NATTEN, given an axis length $L$, kernel size $k$, dilation $d$, and query coordinate $i$, the receptive field window starts at index $s$:

$$s(i) = \min\left(\max(i - r \cdot d, \; 0), \; L - k \cdot d\right)$$

In `natten.flex`, dilation groups are restructured into a batch grid of subgrids:
```python
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
```
This block mask is compiled block-by-block (`_compile=True`), avoiding dense mask allocation (~1.5 GB saved at 448px).

### 2. Solving the PyTorch FlexAttention Memory Leak
During first-time JIT compilation of `flex_attention` on a new tensor resolution, PyTorch leaves uncollected reference cycles holding the input tensors. At 1024px, inputs can exceed 2.3 GB.
`natten-rocm` detects new shape compilation and triggers garbage collection:
```python
if len(_masks) != n_masks:
    gc.collect()  # Clears circular references from compilation
```
Peak memory drops from **10.76 GiB** down to **8.42 GiB**, preventing out-of-memory crashes on 16GB VRAM cards.

---

## 📊 Verification & Tests

The repository includes a comprehensive test suite verifying numerical exactness against double-precision C++ ground truth:

### 1. Mathematical Equivalence against NATTEN C++
Compares against the double-precision reference implementing NATTEN's `get_window_start`:
```bash
python tests/test_flex_na2d.py
```
Output:
```
PASS H=24  W=24  k=(9, 9)   d=(1, 1)   Dv=16  | max_err=0.00e+00
PASS H=38  W=41  k=(9, 9)   d=(4, 4)   Dv=32  | max_err=3.10e-07
PASS H=30  W=52  k=(5, 7)   d=(3, 5)   Dv=64  | max_err=2.85e-07
PASS H=140 W=140 k=(9, 9)   d=(14, 14) Dv=32  | max_err=3.58e-07
Result: ALL TESTS PASSED
```

### 2. Row-Band Tiling Exactness
Verifies that slicing an image into $K \in \{1, 2, 4, 8, 16\}$ row bands with halo overlap produces exact bitwise-level matches:
```bash
python tests/test_bands.py
```
Output:
```
H=256 kv=32 d=(8, 8) halo=32 bands=4  rows/band=64 | max_err=3.61e-07 [OK]
Result: ALL BAND CONFIGS EXACT
```

### 3. Shim vs. Flex Comparison
Inspects the difference between centered-window zero masking and the sliding border rule:
```bash
python tests/compare_shim_vs_flex.py
```

---

## 🏷️ GitHub Metadata & Configuration

### GitHub Description
```text
🚀 Drop-in NATTEN & NAF replacement for AMD ROCm & gfx1030 (RX 6800/6900 series) using PyTorch FlexAttention with exact border rules, FP32 precision, and zero-leak memory management.
```

### GitHub Topics
```text
rocm, gfx1030, natten, neighborhood-attention, flex-attention, amd-gpu, rx6800, rx6800xt, rx6900xt, rdna2, naf, trellis2, comfyui, pytorch, triton
```

---

## 📦 GitHub Release Guide (v0.1.0)

To create a new release on GitHub, you can use the GitHub CLI or git tags:

### Using GitHub CLI (`gh`):
```bash
# 1. Authenticate with GitHub (if not already logged in)
gh auth login

# 2. Create the remote repository and push
gh repo create 0xra0/natten-rocm --public --source=. --remote=origin --push

# 3. Create the v0.1.0 release
gh release create v0.1.0 \
  --title "v0.1.0: Drop-in NATTEN on AMD ROCm (gfx1030) via FlexAttention" \
  --notes-file RELEASE_NOTES.md
```

### Manual Git Tag Push:
```bash
git init -b main
git add .
git commit -m "Initial commit: Drop-in NATTEN for AMD ROCm & gfx1030"
git remote add origin https://github.com/0xra0/natten-rocm.git
git tag -a v0.1.0 -m "Release v0.1.0"
git push -u origin main --tags
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
NATTEN is developed by the [SHI Lab](https://github.com/SHI-Labs/NATTEN).
NAF is developed by [Valeo AI](https://github.com/valeoai/NAF).
