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
    needle_window_tokens,
    parse_kv_cache_gib,
    parse_kv_cache_tokens,
    render_heatmap,
    rows_for_heatmap,
    row,
    score_hit,
    figure_unstarted_note,
    start_fail_rows,
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
        self.assertEqual(len(rows), 169)
        tic_sizes = {
            r["kv_cache_gib"] for r in rows if r["panel"] == "tic"
        }
        self.assertEqual(tic_sizes, {1.09})
        self.assertTrue(all(r["hit"] for r in rows if r["panel"] == "tic"))
        self.assertTrue(
            any(
                r.get("start_fail")
                and r["panel"] == "vanilla"
                and r["kv_cache_gib"] == 1.09
                for r in rows
            )
        )
        self.assertFalse(
            any(
                r["panel"] == "vanilla"
                and r["kv_cache_gib"] == 1.09
                and not r.get("start_fail")
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
            self.assertIn("The Baseline serve would not start at 1.09 GiB.", text)
            self.assertIn("### KV cache", text)
            self.assertIn("rtx-5090-report-needle-haystack.png", text)
            self.assertTrue(
                (dest.parent / "rtx-5090-report-needle-haystack.png").is_file()
            )

    def test_discover_helpers_are_measured_not_fixed(self) -> None:
        self.assertEqual(
            figure_unstarted_note(1.09),
            "Baseline did not start at 1.09 GiB, so it is not shown.",
        )
        self.assertEqual(
            figure_unstarted_note([1.09, 2.18]),
            "Baseline did not start at 1.09 GiB or 2.18 GiB, so they are not shown.",
        )
        self.assertEqual(
            figure_unstarted_note([1.09, 2.18, 4.36]),
            "Baseline did not start at 1.09 GiB, 2.18 GiB, or 4.36 GiB, so they are not shown.",
        )
        self.assertEqual(chars_for_context_tokens(1000), 4000)
        self.assertEqual(STORY_CONTEXT_TOKENS, 16000)
        self.assertEqual(context_bins_for_max_len(8192)[-1], 8000)
        self.assertIn(32000, context_bins_for_max_len(32768))
        self.assertIn(24000, context_bins_for_max_len(32768))
        self.assertEqual(story_tokens_for_max_len(8192), 8000)
        self.assertEqual(story_tokens_for_max_len(32768), 16000)
        self.assertEqual(needle_window_tokens(20480), 20480)
        self.assertEqual(needle_window_tokens(20480, cap=32768), 20480)
        self.assertEqual(needle_window_tokens(20480, cap=8192), 8192)
        qwen_bins = context_bins_for_max_len(20480)
        self.assertIn(16000, qwen_bins)
        self.assertNotIn(24000, qwen_bins)
        self.assertNotIn(32000, qwen_bins)
        self.assertEqual(story_tokens_for_max_len(20480), 16000)
        self.assertEqual(
            parse_kv_cache_tokens(
                "KV pool: 1.09 GiB (20480 tokens, last-W spec)\n"
                "ValueError: To serve at least one request with the model's "
                "max seq len (32768), (1.75 GiB KV cache is needed, which is "
                "larger than the available KV cache memory (1.09 GiB). "
                "estimated maximum model length is 20480.\n"
            ),
            20480,
        )
        cap_log = (
            "KV pool cap: 0.25 GiB (kv pool cap EngineArgs)\n"
            "KV pool cap: 0.25 GiB (4608 tokens, last-W=512, "
            "admit_seqs=8192, live~9, src=live)\n"
            "GPU KV cache size: 294,912 tokens\n"
            "estimated maximum model length is 32768.\n"
        )
        self.assertEqual(parse_kv_cache_tokens(cap_log), 4608)
        self.assertAlmostEqual(parse_kv_cache_gib(cap_log), 0.25)
        self.assertEqual(
            parse_kv_cache_tokens("GPU KV cache size: 20,480 tokens\n"),
            20480,
        )
        dest_only = (
            "KV pool: 0.25 GiB\n"
            "Using max model len 18289\n"
            "GPU KV cache size: 40,831 tokens\n"
            "estimated maximum model length is 18289.\n"
        )
        self.assertEqual(parse_kv_cache_tokens(dest_only), 512 + 4096)
        self.assertAlmostEqual(parse_kv_cache_gib(dest_only), 0.25)
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
        heat = rows_for_heatmap(
            start_fail_rows(panel="vanilla", kv_cache_gib=1.09)
            + [
                row(
                    panel="tic",
                    kv_cache_gib=1.09,
                    depth_percent=50,
                    hit=True,
                    context_tokens=16000,
                )
            ]
        )
        self.assertFalse(any(item.get("start_fail") for item in heat))
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
        product = parse_kv_cache_gib(
            "KV pool: 1.09 GiB (20480 tokens, last-W spec)\n"
            "GPU KV cache size: 20,480 tokens\n"
        )
        self.assertAlmostEqual(product, 1.09)
        steps = "\n".join(niah_discover_steps())
        self.assertIn("scrape GPU KV cache size", steps)
        self.assertIn("NIAH_MAX_MODEL_LEN", steps)
        self.assertIn("do not shrink the window", steps)
        self.assertIn("first recall", steps)
        self.assertIn("start-fail", steps)
        self.assertIn("any model", steps)
        self.assertNotIn("4.00", steps)
        self.assertNotIn("2.5", steps)
        self.assertNotIn("dest_gib", steps)
        table = "\n".join(
            kv_sweep_table_lines(
                [
                    row(
                        panel="tic",
                        kv_cache_gib=1.09,
                        depth_percent=50,
                        hit=True,
                        context_tokens=STORY_CONTEXT_TOKENS,
                    ),
                    *start_fail_rows(panel="vanilla", kv_cache_gib=1.09),
                    row(
                        panel="vanilla",
                        kv_cache_gib=2.18,
                        depth_percent=50,
                        hit=True,
                        context_tokens=STORY_CONTEXT_TOKENS,
                    ),
                    row(
                        panel="vanilla",
                        kv_cache_gib=13.48,
                        depth_percent=50,
                        hit=True,
                        context_tokens=STORY_CONTEXT_TOKENS,
                    ),
                ],
                reserved_gib=1.09,
            )
        )
        self.assertIn("| **TIC** | Fixed | 1.09 | yes | yes | yes |", table)
        self.assertIn(
            "| **Baseline** | Same size as TIC | 1.09 | no | no | no |",
            table,
        )
        self.assertIn("| **Baseline** | Larger KV | 2.18 | yes | yes | yes |", table)
        self.assertIn("Started without OOM", table)
        self.assertNotIn("Rejected or OOM", table)
        self.assertIn("Default fill", table)
        self.assertNotIn("1.75", table)
        self.assertNotIn("Needle compare is", table)
        self.assertNotIn("overflow to host memory", table)
        self.assertIn("on every setup that starts.", table)
        self.assertIn("The Baseline serve would not start at 1.09 GiB.", table)
        snapped = "\n".join(
            kv_sweep_table_lines(
                [
                    row(
                        panel="tic",
                        kv_cache_gib=1.09,
                        depth_percent=50,
                        hit=True,
                        context_tokens=STORY_CONTEXT_TOKENS,
                    ),
                    *start_fail_rows(panel="vanilla", kv_cache_gib=1.09),
                    row(
                        panel="vanilla",
                        kv_cache_gib=2.18,
                        depth_percent=50,
                        hit=True,
                        context_tokens=STORY_CONTEXT_TOKENS,
                    ),
                ],
                reserved_gib=1.0934,
            )
        )
        self.assertIn("| **Baseline** | Same size as TIC | 1.09 | no | no | no |", snapped)
        self.assertNotIn("1.0934", snapped)
        self.assertIn("The Baseline serve would not start at 1.09 GiB.", snapped)
        honest = "\n".join(
            kv_sweep_table_lines(
                [
                    row(
                        panel="tic",
                        kv_cache_gib=0.25,
                        depth_percent=50,
                        hit=True,
                        context_tokens=2000,
                    ),
                    row(
                        panel="vanilla",
                        kv_cache_gib=0.25,
                        depth_percent=50,
                        hit=True,
                        context_tokens=2000,
                    ),
                    row(
                        panel="vanilla",
                        kv_cache_gib=13.48,
                        depth_percent=50,
                        hit=True,
                        context_tokens=2000,
                    ),
                ],
                reserved_gib=0.4882,
            )
        )
        self.assertIn("| **TIC** | Fixed | 0.25 | yes | yes | yes |", honest)
        self.assertIn(
            "| **Baseline** | Same size as TIC | 0.25 | yes | yes | yes |",
            honest,
        )
        self.assertNotIn("0.4882", honest)
        self.assertNotIn("Larger KV", honest)
        self.assertNotIn("would not start at 0.49", honest)
        two = "\n".join(
            kv_sweep_table_lines(
                [
                    row(
                        panel="tic",
                        kv_cache_gib=1.09,
                        depth_percent=50,
                        hit=True,
                        context_tokens=STORY_CONTEXT_TOKENS,
                    ),
                    *start_fail_rows(panel="vanilla", kv_cache_gib=1.09),
                    *start_fail_rows(panel="vanilla", kv_cache_gib=2.18),
                    row(
                        panel="vanilla",
                        kv_cache_gib=4.36,
                        depth_percent=50,
                        hit=True,
                        context_tokens=STORY_CONTEXT_TOKENS,
                    ),
                ]
            )
        )
        self.assertIn(
            "The Baseline serve would not start at 1.09 GiB or 2.18 GiB.",
            two,
        )
        self.assertNotIn("TIC reserved pool on this serve", table)
        self.assertNotIn("did not retrieve", table)
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
        self.assertIn("max_num_seqs=product", out)
        self.assertIn("do not overlay 1", out)
        self.assertIn("max_model_len follows NIAH_MAX_MODEL_LEN", out)
        self.assertIn("not KV pool tokens", out)
        self.assertIn("not a fixed 4.00", out)
        self.assertIn("Do not grow TIC to 1.75.", out)
        self.assertIn("same lock and the same window", out)
        self.assertIn("fill scrape and compare", out)
        self.assertIn("same-size start or start-fail", out)
        self.assertNotIn("max_num_seqs=1", out)
        self.assertNotIn("dest_gib", out)
        self.assertNotIn("dest-from-live", out)
        script = (ROOT / "scripts" / "run_needle_haystack.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("TIC_PRODUCT_KV_GIB_MAX", script)
        self.assertNotIn('upsert(text, "max_num_seqs"', script)
        self.assertNotIn("NIAH_MAX_NUM_SEQS=1", script)
        self.assertIn("estimated maximum model length", script)
        self.assertIn("gib_to_bytes", script)
        self.assertNotIn("bytes_for_pool_tokens", script)
        fails = start_fail_rows(panel="vanilla", kv_cache_gib=1.09)
        self.assertTrue(all(item.get("start_fail") for item in fails))
        self.assertTrue(all(item.get("refuse") for item in fails))
        self.assertFalse(any(item.get("hit") for item in fails))
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
