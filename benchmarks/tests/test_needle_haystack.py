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

try:
    import seaborn as _seaborn  # noqa: F401
    import matplotlib as _matplotlib  # noqa: F401
    import pandas as _pandas  # noqa: F401

    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from needle_haystack import (  # noqa: E402
    NEEDLE,
    STORY_CONTEXT_TOKENS,
    build_messages,
    chars_for_context_tokens,
    context_bins_for_max_len,
    first_recall_candidates,
    insert_needle,
    kv_sweep_table_lines,
    load_jsonl,
    mock_rows,
    niah_discover_steps,
    parse_kv_cache_gib,
    render_heatmap,
    rows_for_heatmap,
    row,
    score_hit,
    story_tokens_for_max_len,
    tic_pool_smaller_than_fill,
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

    @unittest.skipUnless(HAS_PLOT, "matplotlib seaborn pandas required")
    def test_jsonl_roundtrip_and_png(self) -> None:
        rows = mock_rows()
        self.assertEqual(len(rows), 224)
        tic_sizes = {
            r["kv_cache_gib"] for r in rows if r["panel"] == "tic"
        }
        self.assertEqual(tic_sizes, {1.09})
        self.assertTrue(all(r["hit"] for r in rows if r["panel"] == "tic"))
        self.assertTrue(
            any(
                r["panel"] == "vanilla"
                and r["kv_cache_gib"] == 1.09
                and r["context_tokens"] >= 4000
                and not r["hit"]
                for r in rows
            )
        )
        depths = {r["depth_percent"] for r in rows}
        self.assertTrue({0, 50, 100}.issubset(depths))
        contexts = {r["context_tokens"] for r in rows}
        self.assertIn(1000, contexts)
        self.assertIn(32000, contexts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "needle_haystack.jsonl"
            write_jsonl(path, rows)
            loaded = load_jsonl(path)
            self.assertEqual(len(loaded), len(rows))
            self.assertIn("kv_cache_gib", loaded[0])
            self.assertIn("context_tokens", loaded[0])
            self.assertIn("response_completes", loaded[0])
            self.assertNotIn("dest_gib", loaded[0])
            self.assertNotIn("continue_ok", loaded[0])
            png = Path(tmp) / "heat.png"
            render_heatmap(loaded, png)
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 1000)

    @unittest.skipUnless(HAS_PLOT, "matplotlib seaborn pandas required")
    def test_write_report_embeds_heatmap_png(self) -> None:
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
                "kv_cache": {
                    "baseline_memory_bytes_measured": 13647233024,
                    "tic_memory_bytes_measured": 1170378588,
                    "baseline_tokens_measured": 238064,
                    "tic_tokens_measured": 20480,
                },
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
            self.assertIn("Fact retrieved (needle)", text)
            self.assertIn("Fact retrieval by KV cache (needle-in-a-haystack)", text)
            self.assertIn(
                "Figure. Fact retrieval by KV cache (needle-in-a-haystack).",
                text,
            )
            self.assertIn("### KV cache", text)
            self.assertIn("rtx-5090-report-needle-haystack.png", text)
            self.assertTrue(
                (dest.parent / "rtx-5090-report-needle-haystack.png").is_file()
            )

    def test_discover_helpers_are_measured_not_fixed(self) -> None:
        self.assertEqual(chars_for_context_tokens(1000), 4000)
        self.assertEqual(STORY_CONTEXT_TOKENS, 16000)
        self.assertEqual(context_bins_for_max_len(8192)[-1], 4000)
        self.assertNotIn(32000, context_bins_for_max_len(32768))
        self.assertIn(24000, context_bins_for_max_len(32768))
        self.assertEqual(story_tokens_for_max_len(8192), 4000)
        self.assertEqual(story_tokens_for_max_len(32768), 16000)
        cropped = rows_for_heatmap(
            [
                row(
                    panel="tic",
                    kv_cache_gib=1.75,
                    depth_percent=50,
                    hit=True,
                    context_tokens=24000,
                ),
                row(
                    panel="tic",
                    kv_cache_gib=1.75,
                    depth_percent=50,
                    hit=False,
                    context_tokens=32000,
                    response_completes=False,
                    refuse=True,
                ),
            ],
            max_model_len=32768,
        )
        self.assertEqual({int(item["context_tokens"]) for item in cropped}, {24000})
        cands = first_recall_candidates(1.09, cap=13.65)
        self.assertGreaterEqual(cands[0], 2.0)
        self.assertLess(cands[-1], 13.65)
        self.assertNotEqual(cands, [4.0])
        self.assertTrue(tic_pool_smaller_than_fill(1.09, 13.65))
        self.assertTrue(tic_pool_smaller_than_fill(3.5, 10.0))
        self.assertFalse(tic_pool_smaller_than_fill(13.0, 13.65))
        gib = parse_kv_cache_gib(
            "Available KV cache memory: 1.09 GiB\nGPU KV cache size: 12,345 tokens\n"
        )
        self.assertAlmostEqual(gib, 1.09)
        locked = parse_kv_cache_gib(
            "kv-set on W=512 pages=1 vendor_sw=4608 family=ks vendor_pool_bytes=1879048192\n"
        )
        self.assertAlmostEqual(locked, 1.75)
        same = parse_kv_cache_gib(
            "reserved 1.75 GiB memory for KV Cache as specified by "
            "kv_cache_memory_bytes config\nGPU KV cache size: 32,768 tokens\n"
        )
        self.assertAlmostEqual(same, 1.75)
        steps = "\n".join(niah_discover_steps())
        self.assertIn("scrape GPU KV cache size", steps)
        self.assertIn("first recall", steps)
        self.assertIn("any model", steps)
        self.assertNotIn("4.00", steps)
        self.assertNotIn("2.5", steps)
        self.assertNotIn("dest_gib", steps)
        table = "\n".join(
            kv_sweep_table_lines(
                [
                    row(panel="tic", kv_cache_gib=1.75, depth_percent=50, hit=True),
                    row(panel="vanilla", kv_cache_gib=1.75, depth_percent=50, hit=True),
                    row(panel="vanilla", kv_cache_gib=13.48, depth_percent=50, hit=True),
                ],
                reserved_gib=1.09,
            )
        )
        self.assertIn("| **TIC** | Fixed |", table)
        self.assertIn("Same size as TIC", table)
        self.assertIn("Default fill", table)
        self.assertIn("| **TIC** | Fixed | 1.75 |", table)
        self.assertNotIn("1.75†", table)
        self.assertNotIn("overflow to host memory", table)
        self.assertIn("Needle compare is 1.75 GiB on both sides.", table)
        self.assertIn("The Baseline serve would not start at 1.09 GiB.", table)
        self.assertNotIn("TIC reserved pool on this serve", table)
        self.assertNotIn("First size that recalls", table)
        self.assertIn("## B. Needle-in-a-haystack (context tokens × depth)", table)

    def test_runner_dry_run_lists_discover_steps(self) -> None:
        import os
        import subprocess
        import tempfile

        runner = ROOT / "scripts" / "run_needle_haystack.sh"
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "qwen2.5-7b-instruct"
            model_dir.mkdir()
            (model_dir / "common.env").write_text(
                "BASELINE_MODEL_DIR=/tmp/baseline\n"
                "TIC_MODEL_DIR=/tmp/tic\n"
                "MODEL_ID=Qwen/Qwen2.5-7B-Instruct\n"
                "VLLM_IMAGE=vllm/vllm-openai:v0.26.0\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["MODEL_DIR"] = str(model_dir)
            env["ISIRO_BENCH_CONFIG"] = str(model_dir / "common.env")
            completed = subprocess.run(
                ["bash", str(runner), "qwen2.5-7b-instruct", "--dry-run"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        out = completed.stdout
        self.assertIn("max_num_seqs=1", out)
        self.assertIn("max_model_len=32768", out)
        self.assertIn("scrape GPU KV cache size", out)
        self.assertIn("not a fixed 4.00", out)
        self.assertIn("fill scrape and compare", out)
        self.assertIn("fill grid, TIC grid, same-size grid", out)
        self.assertNotIn("dest_gib", out)
        self.assertNotIn("dest-from-live", out)
        script = (ROOT / "scripts" / "run_needle_haystack.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("TIC_PRODUCT_KV_GIB_MAX", script)
        for banned in (
            "dest_gib",
            "dest-from-live",
            "leftover",
            "tANS",
            "rANS",
            "signfold",
            "percol",
        ):
            self.assertNotIn(banned, script)

    def test_row_score_from_hit(self) -> None:
        self.assertEqual(
            row(panel="tic", kv_cache_gib=1.0, depth_percent=10, hit=True)["score"],
            1.0,
        )
        miss = row(panel="vanilla", kv_cache_gib=1.0, depth_percent=10, hit=False)
        self.assertEqual(miss["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
