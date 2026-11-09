"""
Example: Running ValeoAI NAF on AMD ROCm (gfx1030 / RX 6800).
Demonstrates zero-code-change integration using natten.install().
"""

import os
import torch
from natten import install

# 1. Install ROCm NATTEN replacement into sys.modules
install(backend="auto")  # Prefers FlexAttention, falls back to chunked shim

# 2. Load NAF model via torch.hub (or local cache)
print("[*] Loading valeoai/NAF model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Attempt local hub cache if available, or fetch from GitHub
    cache_path = os.path.expanduser("~/.cache/torch/hub/valeoai_NAF_main")
    source = "local" if os.path.isdir(cache_path) else "github"
    hub_repo = cache_path if source == "local" else "valeoai/NAF"

    naf = torch.hub.load(hub_repo, "naf", source=source, pretrained=True).eval().float().to(device)
    print("[+] Model loaded successfully!")

    # 3. Dummy image input (B, C, H, W)
    dummy_input = torch.rand(1, 3, 512, 512, device=device)
    print(f"[*] Running forward pass on input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = naf(dummy_input)

    print(f"[+] Output shape: {output.shape}")
    print("[+] Inference completed successfully on ROCm!")

except Exception as e:
    print(f"[!] Could not run full hub demo: {e}")
    print("[*] Verifying direct na2d neighborhood attention instead:")
    import natten
    q = torch.randn(1, 64, 64, 4, 32, device=device)
    out = natten.na2d(q, q, q, kernel_size=(7, 7), dilation=(1, 1))
    print(f"[+] Direct na2d output shape: {out.shape}")
