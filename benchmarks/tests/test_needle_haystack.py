# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from needle_haystack import (  # noqa: E402
    NEEDLE,
    build_messages,
    insert_needle,
    load_jsonl,
    mock_rows,
    render_heatmap,
    row,
    score_hit,
    write_jsonl,
)


class NeedleHaystackTests(unittest.TestCase):
    def test_score_hit_needle_or_number(self) -> None:
        self.assertTrue(score_hit("84932017"))
        self.assertTrue(score_hit("The number is " + NEEDLE))
        self.assertFalse(score_hit("I do not know"))

    def test_insert_needle_depth(self) -> None:
        hay = "aaaaaa"
        low = insert_needle(hay, "N", 0)
        high = insert_needle(hay, "N", 100)
        self.assertTrue(low.startswith(" N ") or low.startswith("N"))
        self.assertIn("N", high)

    def test_build_messages_include_question(self) -> None:
        msgs = build_messages(depth_percent=50, approx_chars=200)
        self.assertEqual(len(msgs), 1)
        self.assertIn(NEEDLE, msgs[0]["content"])
        self.assertIn("special magic number", msgs[0]["content"])

    def test_jsonl_roundtrip_and_svg(self) -> None:
        rows = mock_rows()
        self.assertTrue(any(r["panel"] == "tic" and r["hit"] for r in rows))
        self.assertTrue(
            any(
                r["panel"] == "vanilla" and r["kv_cache_gib"] == 1.09 and not r["hit"]
                for r in rows
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "needle_haystack.jsonl"
            write_jsonl(path, rows)
            loaded = load_jsonl(path)
            self.assertEqual(len(loaded), len(rows))
            self.assertIn("kv_cache_gib", loaded[0])
            self.assertIn("response_completes", loaded[0])
            self.assertNotIn("dest_gib", loaded[0])
            self.assertNotIn("continue_ok", loaded[0])
            svg = Path(tmp) / "heat.svg"
            render_heatmap(loaded, svg)
            text = svg.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertIn("TIC", text)
            self.assertIn("Baseline", text)
            self.assertIn("#2ca02c", text)
            self.assertIn("#d62728", text)

    def test_write_report_embeds_heatmap_svg(self) -> None:
        from generate_report import write_report

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "baseline").mkdir()
            (run / "isiro").mkdir()
            write_jsonl(run / "needle_haystack.jsonl", mock_rows())
            summary = {
                "schema": "isiro-benchmark-summary-v1",
                "model": "qwen2.5-7b-instruct",
                "precision": "bf16",
                "system_id": "rtx-5090",
                "isiro_format": "v0.1.1",
                "recorded_at": "2026-08-21T22:29:00+00:00",
                "experiment_kind": "equal-batch",
                "fairness": {"passed": True},
                "footprint": {
                    "baseline": {
                        "on_disk_bytes": 1,
                        "loaded_bytes": 1,
                        "non_kv_cache_memory_bytes_reported": 1,
                    },
                    "tic": {
                        "on_disk_bytes": 1,
                        "loaded_bytes": 1,
                        "non_kv_cache_memory_bytes_reported": 1,
                    },
                },
                "correctness": {"weight_bit_exact_ok": True},
                "kv_cache": {},
            }
            (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run / "environment.json").write_text(
                json.dumps(
                    {
                        "gpu_model": "NVIDIA GeForce RTX 5090",
                        "gpu_count": 1,
                        "driver_version": "1",
                        "cuda_version": "13",
                        "vllm_version": "0.26.0",
                    }
                ),
                encoding="utf-8",
            )
            (run / "commands.json").write_text(
                json.dumps({"enforce_eager": False}), encoding="utf-8"
            )
            dest = Path(tmp) / "out" / "rtx-5090-report.md"
            write_report(run, dest)
            text = dest.read_text(encoding="utf-8")
            self.assertIn("## B. Needle-in-a-haystack", text)
            self.assertIn("### KV cache", text)
            self.assertIn("rtx-5090-report-needle-haystack.svg", text)
            self.assertTrue(
                (dest.parent / "rtx-5090-report-needle-haystack.svg").is_file()
            )

    def test_row_score_from_hit(self) -> None:
        self.assertEqual(
            row(panel="tic", kv_cache_gib=1.0, depth_percent=10, hit=True)["score"],
            1.0,
        )
        miss = row(panel="vanilla", kv_cache_gib=1.0, depth_percent=10, hit=False)
        self.assertEqual(miss["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
