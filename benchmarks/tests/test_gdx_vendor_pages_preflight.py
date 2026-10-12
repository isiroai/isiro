# SPDX-License-Identifier: Apache-2.0
"""Path-A FA page dump must refuse MLA / CSA2 / sparse-only hybrids early."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from gdx_kv_bytes_curve import _fa_dump_layer_caps, _layer_caps_from_config
from gdx_kv_curve_vendor_pages import (
    _expand_logical_block_ids,
    vendor_fa_pages_unsupported_reason,
)


class GdxVendorPagesPreflightTests(unittest.TestCase):
    def _write_cfg(self, root: Path, cfg: dict) -> Path:
        path = root / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return root

    def test_dense_gqa_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._write_cfg(
                Path(tmp),
                {"num_hidden_layers": 4, "num_key_value_heads": 8, "head_dim": 128},
            )
            self.assertIsNone(vendor_fa_pages_unsupported_reason(model))

    def test_qwen_hybrid_full_attention_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._write_cfg(
                Path(tmp),
                {
                    "num_hidden_layers": 4,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                        "linear_attention",
                    ],
                },
            )
            self.assertIsNone(vendor_fa_pages_unsupported_reason(model))
            caps, _ = _layer_caps_from_config(model)
            fa = _fa_dump_layer_caps(model, caps)
            self.assertEqual(len(fa), 1)

    def test_glm_mla_hybrid_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._write_cfg(
                Path(tmp),
                {
                    "num_hidden_layers": 4,
                    "kv_lora_rank": 512,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "deepseek_sparse_attention",
                        "linear_attention",
                    ],
                },
            )
            reason = vendor_fa_pages_unsupported_reason(model)
            self.assertIsNotNone(reason)
            self.assertIn("kv_lora_rank", str(reason))
            caps, _ = _layer_caps_from_config(model)
            self.assertEqual(_fa_dump_layer_caps(model, caps), [])

    def test_deepseek_csa2_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._write_cfg(
                Path(tmp),
                {
                    "num_hidden_layers": 8,
                    "kv_source_layer_ids": [2, 4],
                    "sliding_window": 128,
                    "num_key_value_heads": 1,
                    "head_dim": 512,
                },
            )
            reason = vendor_fa_pages_unsupported_reason(model)
            self.assertIsNotNone(reason)
            self.assertIn("kv_source_layer_ids", str(reason))
            caps, meta = _layer_caps_from_config(model)
            self.assertEqual(caps, [128] * 8)
            self.assertEqual(meta.get("n_sliding"), 8)

    def test_hybrid_manager_page_expands_to_storage_tiles(self) -> None:
        # Qwen hybrid: manager page 1568, FlashInfer tile 32 → 49 tiles/page.
        ids = _expand_logical_block_ids(
            [0, 1, 2, 3, 4, 5],
            manager_block_size=1568,
            storage_block_size=32,
            n_storage_blocks=256,
            n_logical_blocks=6,
        )
        self.assertEqual(len(ids), 256)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[48], 48)
        self.assertEqual(ids[49], 49)
        self.assertEqual(ids[1568 // 32], 49)

if __name__ == "__main__":
    unittest.main()
