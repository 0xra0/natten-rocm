"""
Unit tests for natten package import, module patching, and API compatibility.
"""

import sys
import unittest
import torch
import natten
from natten import install, is_flex_available


class TestNattenCompatibility(unittest.TestCase):
    def test_package_exports(self):
        self.assertTrue(hasattr(natten, "na2d"))
        self.assertTrue(hasattr(natten, "NeighborhoodAttention2D"))
        self.assertTrue(hasattr(natten, "install"))
        self.assertTrue(hasattr(natten, "has_fp32"))
        self.assertTrue(natten.has_fp32())
        self.assertFalse(natten.has_bfloat())

    def test_install_flex(self):
        if not is_flex_available():
            self.skipTest("FlexAttention not available")
        install(backend="flex")
        mod = sys.modules["natten"]
        self.assertTrue(getattr(mod, "_flex_shim", False))
        self.assertNotIn("natten.functional", sys.modules)

        # In NAF, `from natten.functional import ...` should fail so it falls back to `from natten import na2d`
        with self.assertRaises(ImportError):
            import natten.functional  # noqa: F401

        from natten import na2d
        self.assertTrue(callable(na2d))

    def test_install_shim(self):
        install(backend="shim")
        mod = sys.modules["natten"]
        self.assertTrue(getattr(mod, "_pure_torch_shim", False))
        self.assertIn("natten.functional", sys.modules)

        from natten.functional import na2d_qk, na2d_av
        self.assertTrue(callable(na2d_qk))
        self.assertTrue(callable(na2d_av))

    def test_module_forward_cpu(self):
        layer = natten.NeighborhoodAttention2D(dim=32, num_heads=4, kernel_size=7, dilation=1)
        x = torch.randn(1, 32, 16, 16)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
