# SPDX-License-Identifier: Apache-2.0
"""Regression: fp8/fp4 GDX curves must use vendor FA pages, not bf16 casts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


class GdxVendorPagesCurveTests(unittest.TestCase):
    def test_gdx_fp8_curve_bans_bf16_standin_and_requires_vendor_pages(self) -> None:
        curve = (SCRIPTS / "gdx_kv_bytes_curve.py").read_text(encoding="utf-8")
        vendor = (SCRIPTS / "gdx_kv_curve_vendor_pages.py").read_text(encoding="utf-8")
        harness = (SCRIPTS / "harness_lib.sh").read_text(encoding="utf-8")
        for banned in (
            ">> 8",
            ">> 12",
            ">>8",
            ">>12",
            "stand-in",
            "stand_in",
            "high byte",
            "_codes_for_alphabet",
        ):
            self.assertNotIn(banned, curve, msg=f"curve still has {banned!r}")
            self.assertNotIn(banned, vendor, msg=f"vendor capture still has {banned!r}")
        self.assertIn("vendor_fa_pages", curve)
        self.assertIn("--vendor-pages", curve)
        self.assertIn("no bf16 cast", curve)
        self.assertIn("gdx_kv_curve_vendor_pages.py", harness)
        self.assertIn("--vendor-pages", harness)
        self.assertIn("vendor_fa_pages_unsupported_reason", harness)
        self.assertIn("ISIRO_BENCH_MLA_LATENT_NPZ", harness)
        self.assertIn("--mla-latent", curve)
        self.assertIn("measure_curve_from_mla_latent", curve)
        # fp8/fp4 harness path uses BASELINE_MODEL_DIR, not the bf16 twin override.
        fp8_block_start = harness.index("gdx_kv_curve_vendor_pages")
        fp8_block = harness[fp8_block_start : fp8_block_start + 1800]
        self.assertIn("BASELINE_MODEL_DIR", fp8_block)
        self.assertNotIn("ISIRO_BENCH_GDX_MODEL_DIR", fp8_block)

        from gdx_kv_bytes_curve import measure_curve, measure_curve_from_vendor_pages

        denied = measure_curve(
            "fp8",
            tokens=[512, 1024],
            model_dir=Path("."),
        )
        self.assertFalse(denied.get("ok"))
        self.assertIn("vendor-pages", str(denied.get("error")))

        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"num_hidden_layers": 2}),
                encoding="utf-8",
            )
            # Low-entropy synthetic vendor e4m3 pages [L,T,H,D] (compressible).
            rng = np.random.default_rng(7)
            k = np.zeros((2, 2048, 2, 16), dtype=np.uint8)
            v = np.zeros((2, 2048, 2, 16), dtype=np.uint8)
            k[:, ::17, :, 0] = rng.integers(0, 32, size=k[:, ::17, :, 0].shape)
            v[:, ::19, :, 0] = rng.integers(0, 32, size=v[:, ::19, :, 0].shape)
            pages = root / "pages.npz"
            np.savez_compressed(pages, k=k, v=v)
            doc = measure_curve_from_vendor_pages(
                "fp8",
                tokens=[512, 1024, 2048],
                vendor_pages=pages,
                model_dir=root,
                live_window=512,
                kv_cache_dtype="fp8",
            )
            if doc.get("error") and "not importable" in str(doc.get("error")):
                self.skipTest("isiro_runtime not importable for GDX curve")
            self.assertEqual(doc.get("kv_source"), "vendor_fa_pages")
            self.assertEqual(doc.get("source"), "vendor_fa_pages")
            self.assertEqual(doc.get("kv_cache_dtype"), "fp8")
            self.assertTrue(doc.get("ok"), doc.get("error"))
            cold = [p for p in doc["points"] if int(p["cold_t"]) > 0]
            self.assertGreaterEqual(len(cold), 2)
            self.assertLess(float(cold[-1]["payload_bpv"]), 8.0)

            with self.assertRaises(TypeError):
                from gdx_kv_bytes_curve import _pack_codes_nbytes

                _pack_codes_nbytes(
                    "fp8",
                    np.zeros((1, 8, 2, 16), dtype=np.uint16),
                )

    def test_mla_latent_curve_path_a(self) -> None:
        from gdx_kv_bytes_curve import measure_curve_from_mla_latent

        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(3)
            # [L,T,D] sparse latent (path_a_mla single plane).
            lat = np.zeros((3, 2048, 64), dtype=np.uint8)
            lat[:, ::29, ::4] = rng.integers(1, 20, size=lat[:, ::29, ::4].shape)
            npz = root / "mla.npz"
            np.savez_compressed(npz, g0=lat)
            doc = measure_curve_from_mla_latent(
                "fp8",
                tokens=[512, 1024, 2048],
                mla_latent=npz,
                live_window=512,
                kv_cache_dtype="fp8_ds_mla",
            )
            if doc.get("error") and "not importable" in str(doc.get("error")):
                self.skipTest("isiro_runtime not importable for GDX curve")
            self.assertEqual(doc.get("kv_source"), "vendor_mla_latent")
            self.assertTrue(doc.get("ok"), doc.get("error"))
            cold = [p for p in doc["points"] if int(p["cold_t"]) > 0]
            self.assertGreaterEqual(len(cold), 2)
            self.assertGreater(float(cold[-1]["save_vs_fa"]), 0.0)


if __name__ == "__main__":
    unittest.main()
