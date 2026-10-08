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
    NEEDLE_BINDER,
    NEEDLES,
    NEEDLE_KEYS,
    CHAT_SEED,
    STORY_CONTEXT_TOKENS,
    LONG_CONTEXT_TOKENS,
    CHAT_MAX_TOKENS,
    TEMPLATE_SLACK_TOKENS,
    DEFAULT_DEPTHS,
    build_messages,
    chars_for_context_tokens,
    context_axis_tokens,
    context_bins_for_max_len,
    hay_tokens_for_context,
    insert_needle,
    insert_needles,
    needle_spread_depths,
    kv_sweep_table_lines,
    load_jsonl,
    advertised_max_model_len,
    context_length_bins,
    log_context_bins,
    long_needle_text,
    mock_rows,
    niah_discover_steps,
    needle_window_tokens,
    parse_kv_cache_gib,
    parse_kv_cache_tokens,
    parse_non_kv_cache_bytes,
    parse_prometheus_kv_usage_perc,
    parse_prometheus_isiro_peak_bytes,
    persist_outside_bytes,
    derived_archived_id_bytes,
    derived_kv_in_use_bytes,
    kv_in_use_bytes,
    gib_to_bytes,
    bytes_to_gib,
    resolve_non_kv_bytes,
    reserved_vram_y_bytes,
    vram_in_use_y_bytes,
    extra_prompt_tokens,
    NEEDLE_COUNT,
    figure_tagline,
    haystack_text,
    prompt_token_budget,
    _context_tick,
    render_heatmap,
    render_vram_vs_context,
    rows_for_heatmap,
    row,
    score_hit,
    score_needles,
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
        self.assertTrue(
            score_hit("a- to- er 0_0 soup with 84932017 buried in noise " * 4)
        )

    def test_window_fit_keeps_8k_bin_inside_8192(self) -> None:
        self.assertEqual(chars_for_context_tokens(1000), 4000)
        hay = hay_tokens_for_context(8000, 8192)
        self.assertLessEqual(
            hay + CHAT_MAX_TOKENS + TEMPLATE_SLACK_TOKENS, 8192
        )
        fitted = chars_for_context_tokens(8000, 8192)
        self.assertLess(fitted, chars_for_context_tokens(8000))
        self.assertEqual(
            prompt_token_budget(8192),
            8192 - CHAT_MAX_TOKENS - TEMPLATE_SLACK_TOKENS,
        )
        extra = extra_prompt_tokens(long_needle_text(n_words=128), "x")
        fitted_long = chars_for_context_tokens(8000, 8192, extra_tokens=extra)
        self.assertLess(fitted_long, chars_for_context_tokens(8000, 8192))
        self.assertIn(
            LONG_CONTEXT_TOKENS,
            context_bins_for_max_len(
                LONG_CONTEXT_TOKENS,
                (8000, LONG_CONTEXT_TOKENS),
            ),
        )
        self.assertNotIn(
            LONG_CONTEXT_TOKENS,
            context_bins_for_max_len(8192, (8000, LONG_CONTEXT_TOKENS)),
        )

    def test_score_needles_fractions(self) -> None:
        self.assertEqual(NEEDLE_COUNT, 3)
        self.assertEqual(len(NEEDLES), 3)
        self.assertEqual(NEEDLE_KEYS, ("84932017", "Zebra-419", "Copperbrook"))
        self.assertEqual(CHAT_SEED, 17)
        self.assertEqual(score_needles(""), 0.0)
        self.assertEqual(score_needles(NEEDLES[0]), 0.3333)
        self.assertEqual(score_needles(NEEDLES[0] + " " + NEEDLES[1]), 0.6667)
        self.assertEqual(score_needles(" ".join(NEEDLES)), 1.0)
        self.assertEqual(
            score_needles("The facts are 84932017, Zebra-419, and Copperbrook."),
            1.0,
        )
        self.assertEqual(score_needles("84932017 only"), 0.3333)
        self.assertEqual(
            figure_tagline(
                gpu="NVIDIA GeForce RTX 5090",
                gpu_count=1,
                model="qwen2.5-7b-instruct",
            ),
            "NVIDIA GeForce RTX 5090 x 1 | qwen2.5-7b-instruct",
        )

    def test_kv_in_use_metrics_and_derived(self) -> None:
        text = (
            "# HELP vllm:kv_cache_usage_perc GPU KV cache usage.\n"
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            "vllm:kv_cache_usage_perc{engine=\"0\"} 0.25\n"
        )
        self.assertAlmostEqual(parse_prometheus_kv_usage_perc(text), 0.25)
        gpu = (
            "vllm:gpu_cache_usage_perc 12.5\n"
        )
        self.assertAlmostEqual(parse_prometheus_kv_usage_perc(gpu), 0.125)
        self.assertEqual(
            parse_prometheus_isiro_peak_bytes("isiro_gpu_memory_peak_bytes 123456789\n"),
            123456789,
        )
        self.assertEqual(derived_archived_id_bytes(32768, 20480), 12288 * 4)
        used = derived_kv_in_use_bytes(10240, 1.09, 20480)
        self.assertEqual(used, int(round(0.5 * gib_to_bytes(1.09))))
        metered, kind = kv_in_use_bytes(
            context_tokens=10240,
            pool_gib=1.09,
            pool_tokens=20480,
            usage_perc=0.4,
        )
        self.assertEqual(kind, "metrics")
        self.assertEqual(metered, int(round(0.4 * gib_to_bytes(1.09))))
        derived, kind = kv_in_use_bytes(
            context_tokens=10240,
            pool_gib=1.09,
            pool_tokens=20480,
            usage_perc=None,
        )
        self.assertEqual(kind, "derived")
        self.assertEqual(derived, used)

    def test_parse_non_kv_and_vram_y_intercept(self) -> None:
        tight = "Total non KV cache memory: 11.76GiB\n"
        spaced = "Total non KV cache memory: 11.19 GiB\n"
        self.assertEqual(parse_non_kv_cache_bytes(tight), gib_to_bytes(11.76))
        self.assertEqual(parse_non_kv_cache_bytes(spaced), gib_to_bytes(11.19))
        self.assertIsNone(parse_non_kv_cache_bytes("Available KV cache memory: 1.09 GiB\n"))
        reported, kind = resolve_non_kv_bytes(
            reported=gib_to_bytes(11.19),
            process_bytes=gib_to_bytes(12.28),
            pool_gib=1.09,
        )
        self.assertEqual(kind, "reported")
        self.assertEqual(reported, gib_to_bytes(11.19))
        fallback, kind = resolve_non_kv_bytes(
            reported=None,
            process_bytes=gib_to_bytes(12.28),
            pool_gib=1.09,
        )
        self.assertEqual(kind, "smi_minus_pool")
        self.assertEqual(fallback, gib_to_bytes(12.28) - gib_to_bytes(1.09))
        rows = [r for r in mock_rows() if not r.get("start_fail")]
        tic = next(
            r
            for r in rows
            if r["panel"] == "tic" and int(r["context_tokens"]) == 1000
        )
        base = next(
            r
            for r in rows
            if r["panel"] == "vanilla" and int(r["context_tokens"]) == 1000
        )
        self.assertEqual(tic["non_kv_kind"], "reported")
        self.assertEqual(tic["non_kv_bytes_reported"], gib_to_bytes(11.19))
        self.assertEqual(base["non_kv_bytes_reported"], gib_to_bytes(15.34))
        self.assertLess(
            abs(vram_in_use_y_bytes(tic) - tic["non_kv_bytes_reported"]),
            gib_to_bytes(0.15),
        )
        self.assertEqual(persist_outside_bytes(tic), 0)
        self.assertEqual(
            reserved_vram_y_bytes(tic),
            tic["non_kv_bytes_reported"] + gib_to_bytes(1.09),
        )
        self.assertEqual(
            reserved_vram_y_bytes(base),
            base["non_kv_bytes_reported"] + gib_to_bytes(13.48),
        )
        filled = next(
            r
            for r in rows
            if r["panel"] == "tic" and int(r["context_tokens"]) == 32768
        )
        self.assertGreater(vram_in_use_y_bytes(filled), vram_in_use_y_bytes(tic))
        self.assertEqual(
            persist_outside_bytes(filled),
            derived_archived_id_bytes(32768, 20480),
        )
        self.assertAlmostEqual(bytes_to_gib(vram_in_use_y_bytes(filled)), 11.19 + 1.09, places=2)

    def test_smi_non_kv_locked_keeps_tic_below_baseline(self) -> None:
        from needle_haystack import locked_non_kv_bytes

        pool = gib_to_bytes(1.09)
        fill = gib_to_bytes(13.48)
        rows = []
        for t, tic_smi, van_kv in (
            (1000, gib_to_bytes(12.80), int(round(1000 / 252352 * fill))),
            (32768, gib_to_bytes(18.84), int(round(32768 / 252352 * fill))),
        ):
            rows.append(
                {
                    "panel": "tic",
                    "context_tokens": t,
                    "kv_cache_gib": 1.09,
                    "kv_bytes_in_use": min(pool, int(round(t / 20480 * pool))),
                    "non_kv_bytes_reported": tic_smi,
                    "non_kv_kind": "smi_minus_pool",
                    "persist_page_bytes": derived_archived_id_bytes(t, 20480),
                    "archived_id_bytes": derived_archived_id_bytes(t, 20480),
                }
            )
            rows.append(
                {
                    "panel": "vanilla",
                    "context_tokens": t,
                    "kv_cache_gib": 13.48,
                    "kv_bytes_in_use": van_kv,
                    "non_kv_bytes_reported": gib_to_bytes(14.75),
                    "non_kv_kind": "reported",
                    "persist_page_bytes": 0,
                    "archived_id_bytes": 0,
                }
            )
        tic_nk = locked_non_kv_bytes(rows, panel="tic", size=1.09)
        van_nk = locked_non_kv_bytes(rows, panel="vanilla", size=13.48)
        self.assertEqual(tic_nk, gib_to_bytes(12.80))
        self.assertEqual(van_nk, gib_to_bytes(14.75))
        for t in (1000, 32768):
            tic = next(r for r in rows if r["panel"] == "tic" and r["context_tokens"] == t)
            van = next(
                r for r in rows if r["panel"] == "vanilla" and r["context_tokens"] == t
            )
            self.assertLess(
                vram_in_use_y_bytes(tic, non_kv_override=tic_nk),
                vram_in_use_y_bytes(van, non_kv_override=van_nk),
            )

    def test_advertised_max_model_len_from_config(self) -> None:
        self.assertEqual(
            advertised_max_model_len({"max_position_embeddings": 32768}),
            32768,
        )
        self.assertEqual(
            advertised_max_model_len(
                {
                    "text_config": {"max_position_embeddings": 8192},
                    "max_position_embeddings": 2048,
                }
            ),
            8192,
        )
        self.assertEqual(
            advertised_max_model_len(
                {
                    "max_position_embeddings": 32768,
                    "rope_scaling": {
                        "type": "yarn",
                        "factor": 4.0,
                        "original_max_position_embeddings": 32768,
                    },
                }
            ),
            131072,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(
                json.dumps({"max_position_embeddings": 1048576}),
                encoding="utf-8",
            )
            self.assertEqual(advertised_max_model_len(Path(tmp)), 1048576)
            self.assertEqual(advertised_max_model_len(cfg), 1048576)
        self.assertIsNone(advertised_max_model_len({}))

    def test_log_context_bins_32k_and_1m(self) -> None:
        bins_32k = context_length_bins(32768)
        self.assertEqual(bins_32k[0], 1000)
        self.assertEqual(bins_32k[-1], 32768)
        self.assertEqual(len(bins_32k), 20)
        step = (32768 - 1000) / 19
        self.assertEqual(bins_32k[1], int(round(1000 + step)))
        self.assertEqual(log_context_bins(32768), bins_32k)
        bins_1m = context_length_bins(1_048_576)
        self.assertEqual(bins_1m[0], 1000)
        self.assertEqual(bins_1m[-1], 1_048_576)
        self.assertEqual(len(bins_1m), 20)
        self.assertEqual(context_bins_for_max_len(32768), bins_32k)
        self.assertEqual(_context_tick(1000), "1,000")
        self.assertEqual(_context_tick(32768), "32,768")
        self.assertEqual(_context_tick(1_048_576), "1,048,576")
        axis = context_axis_tokens(mock_rows())
        self.assertEqual(axis[0], 1000)
        self.assertEqual(axis[-1], 32768)
        self.assertEqual(axis, log_context_bins(32768))
        self.assertEqual(context_bins_for_max_len(1_048_576), bins_1m)
        self.assertEqual(DEFAULT_DEPTHS, (0, 11, 22, 33, 44, 56, 67, 78, 89, 100))
        hay = "aaaaaa"
        body = insert_needles(hay, NEEDLES, 50)
        for fact in NEEDLES:
            self.assertIn(fact, body)
        self.assertEqual(
            needle_spread_depths(3, 0),
            (0.0, 100.0 / 3.0, 200.0 / 3.0),
        )
        self.assertEqual(needle_spread_depths(3, 100), (100.0, 100.0, 100.0))
        long_hay = haystack_text(approx_chars=4000)
        at_zero = insert_needles(long_hay, NEEDLES, 0)
        at_mid = insert_needles(long_hay, NEEDLES, 50)
        at_end = insert_needles(long_hay, NEEDLES, 100)
        pos_zero = [at_zero.find(fact) for fact in NEEDLES]
        pos_mid = [at_mid.find(fact) for fact in NEEDLES]
        self.assertEqual(pos_zero, sorted(pos_zero))
        self.assertEqual(pos_mid, sorted(pos_mid))
        self.assertLess(pos_zero[0], pos_zero[1])
        self.assertLess(pos_zero[1], pos_zero[2])
        self.assertLess(pos_mid[0], pos_mid[1])
        self.assertLess(pos_mid[1], pos_mid[2])
        self.assertGreater(pos_mid[0], pos_zero[0])
        self.assertTrue(at_end.endswith(NEEDLES[-1]))
        for fact in NEEDLES:
            self.assertGreater(at_end.find(fact), len(long_hay) - 8)

    def test_long_needle_survives_depths_and_1m_hay(self) -> None:
        needle = long_needle_text(n_words=128)
        self.assertIn(NEEDLE, needle)
        self.assertIn(NEEDLE_BINDER, needle)
        self.assertGreater(len(needle.split()), 120)
        hay = "x" * 2000
        for depth in (0, 50, 100):
            body = insert_needle(hay, needle, depth)
            self.assertIn(needle, body)
        huge = "y" * 50_000
        body = insert_needle(huge, needle, 0)
        self.assertTrue(body.strip().startswith(needle.strip()[:24]) or needle in body)
        self.assertIn(NEEDLE_BINDER + "-0000", body)
        self.assertIn(NEEDLE_BINDER + "-0127", body)
        self.assertLessEqual(
            hay_tokens_for_context(LONG_CONTEXT_TOKENS, LONG_CONTEXT_TOKENS)
            + CHAT_MAX_TOKENS
            + TEMPLATE_SLACK_TOKENS,
            LONG_CONTEXT_TOKENS,
        )

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
        self.assertIn(NEEDLES[1], msgs[0]["content"])
        self.assertIn(NEEDLES[2], msgs[0]["content"])
        self.assertIn("special magic number", msgs[0]["content"])
        self.assertIn("secret city code", msgs[0]["content"])
        self.assertIn("hidden river name", msgs[0]["content"])

    @unittest.skipUnless(HAS_PLOT, "matplotlib seaborn pandas required")
    def test_jsonl_roundtrip_and_png(self) -> None:
        rows = mock_rows()
        from needle_haystack import mock_advertised_max

        bins = context_length_bins(mock_advertised_max())
        self.assertEqual(len(rows), 1 + 2 * len(DEFAULT_DEPTHS) * len(bins))
        tic_sizes = {
            r["kv_cache_gib"] for r in rows if r["panel"] == "tic"
        }
        self.assertEqual(tic_sizes, {1.09})
        self.assertTrue(all(r["hit"] for r in rows if r["panel"] == "tic"))
        vanilla_scores = {
            round(float(r["score"]), 4)
            for r in rows
            if r["panel"] == "vanilla" and not r.get("start_fail")
        }
        self.assertTrue({0.3333, 0.6667, 1.0, 0.0} <= vanilla_scores)
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
        depths = {r["depth_percent"] for r in rows if not r.get("start_fail")}
        self.assertEqual(depths, set(DEFAULT_DEPTHS))
        contexts = {r["context_tokens"] for r in rows if not r.get("start_fail")}
        self.assertIn(1000, contexts)
        self.assertIn(32768, contexts)
        self.assertNotIn(2.18, {r["kv_cache_gib"] for r in rows})
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
            vram = Path(tmp) / "vram-vs-context.png"
            render_vram_vs_context(loaded, vram)
            self.assertTrue(vram.is_file())
            self.assertGreater(vram.stat().st_size, 1000)

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
            dest = (
                Path(tmp) / "out" / "rtx-5090-report-20260821T000000Z" / "report.md"
            )
            write_report(run, dest)
            text = dest.read_text(encoding="utf-8")
            self.assertIn("## B. Needle-in-a-haystack", text)
            self.assertNotIn("Fact retrieved (needle)", text)
            self.assertNotIn("| **TIC** | Fixed |", text)
            self.assertIn("Fact retrieval by context length (3-needle haystack)", text)
            self.assertIn(
                "Figure. Fact retrieval by context length (3-needle haystack).",
                text,
            )
            self.assertIn("The Baseline serve would not start at 1.09 GiB.", text)
            self.assertIn("### KV cache", text)
            self.assertIn("needle-haystack.png", text)
            self.assertNotIn("rtx-5090-report-needle-haystack.png", text)
            self.assertTrue((dest.parent / "needle-haystack.png").is_file())
            self.assertIn("vram-vs-context.png", text)
            self.assertTrue((dest.parent / "vram-vs-context.png").is_file())
            self.assertIn(
                "Figure. VRAM usage vs context tokens.",
                text,
            )
            self.assertIn("Solid line is in-use: non-KV (weights) plus live vendor KV", text)
            self.assertIn("Not nvidia-smi reserved", text)
            self.assertIn("An X on the GPU HBM line is OOM", text)
            self.assertIn("An X in a column is OOM, same as the VRAM figure.", text)
            self.assertNotIn("Above GPU HBM (OOM)", text)
            self.assertNotIn("## Serve at recall-matched KV cache", text)
            self.assertNotIn("Not default fill", text)

    def test_mock_report_dir_has_figures(self) -> None:
        mock = ROOT / "qwen2.5-7b-instruct" / "rtx-5090-report_mock"
        text = (mock / "report.md").read_text(encoding="utf-8")
        self.assertIn("](needle-haystack.png)", text)
        self.assertIn("](vram-vs-context.png)", text)
        self.assertIn("Solid line is in-use: non-KV (weights) plus live vendor KV", text)
        self.assertIn("An X on the GPU HBM line is OOM", text)
        self.assertIn("An X in a column is OOM, same as the VRAM figure.", text)
        self.assertNotIn("Above GPU HBM (OOM)", text)
        self.assertNotIn("## Serve at recall-matched KV cache", text)
        self.assertNotIn("Not default fill", text)
        self.assertIn("| GPU memory for loaded weights | 15.34 GiB |", text)
        self.assertNotIn("| GPU memory for loaded weights | 15.34 GB |", text)
        self.assertNotIn("Larger KV", text)
        self.assertNotIn("Same size as TIC", text)
        self.assertNotIn("Fact retrieved (needle)", text)
        self.assertTrue((mock / "needle-haystack.png").is_file())
        self.assertTrue((mock / "vram-vs-context.png").is_file())
        self.assertGreater((mock / "needle-haystack.png").stat().st_size, 1000)
        self.assertGreater((mock / "vram-vs-context.png").stat().st_size, 1000)

    def test_discover_helpers_are_measured_not_fixed(self) -> None:
        self.assertEqual(chars_for_context_tokens(1000), 4000)
        self.assertEqual(STORY_CONTEXT_TOKENS, 16000)
        crop = (1000, 8000, 16000, 24000, 32000)
        self.assertEqual(context_bins_for_max_len(8192, crop)[-1], 8000)
        self.assertIn(32000, context_bins_for_max_len(32768, crop))
        self.assertIn(24000, context_bins_for_max_len(32768, crop))
        self.assertEqual(story_tokens_for_max_len(8192, bins=crop), 8000)
        self.assertEqual(story_tokens_for_max_len(32768, bins=crop), 16000)
        self.assertEqual(needle_window_tokens(20480), 20480)
        self.assertEqual(needle_window_tokens(20480, cap=32768), 20480)
        self.assertEqual(needle_window_tokens(20480, cap=8192), 8192)
        qwen_bins = context_bins_for_max_len(20480, crop)
        self.assertIn(16000, qwen_bins)
        self.assertNotIn(24000, qwen_bins)
        self.assertNotIn(32000, qwen_bins)
        self.assertEqual(story_tokens_for_max_len(20480, bins=crop), 16000)
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
        oom_col = rows_for_heatmap(
            [
                row(
                    panel="vanilla",
                    kv_cache_gib=13.48,
                    depth_percent=50,
                    hit=False,
                    score=0.0,
                    context_tokens=32768,
                    oom=True,
                    response_completes=False,
                    refuse=True,
                )
            ]
        )
        self.assertEqual({int(item["context_tokens"]) for item in oom_col}, {32768})
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
        gdx = parse_kv_cache_gib(
            "GDX FA live clamp: 449 MiB (vendor FA HBM; admit=8192 cold_arm_W=512)\n"
            "GPU KV cache size: 8,208 tokens\n"
        )
        self.assertAlmostEqual(gdx, 449 / 1024.0)
        steps = "\n".join(niah_discover_steps())
        self.assertIn("scrape GPU KV cache size", steps)
        self.assertIn("NIAH_MAX_MODEL_LEN", steps)
        self.assertIn("do not shrink the window", steps)
        self.assertNotIn("first recall", steps)
        self.assertIn("no extra KV-size grids", steps)
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
        self.assertNotIn("| **TIC** | Fixed |", table)
        self.assertNotIn("Fact retrieved (needle)", table)
        self.assertNotIn("Same size as TIC", table)
        self.assertNotIn("Larger KV", table)
        self.assertNotIn("Started without OOM", table)
        self.assertNotIn("1.75", table)
        self.assertNotIn("Needle compare is", table)
        self.assertNotIn("overflow to host memory", table)
        from needle_haystack import kv_overflow_footnote_lines

        note = "\n".join(kv_overflow_footnote_lines())
        self.assertIn("persist after vendor reclaim", note)
        self.assertNotIn("overflow to host memory", note)
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
        self.assertIn("The Baseline serve would not start at 1.09 GiB.", snapped)
        self.assertNotIn("| **Baseline** | Default fill |", snapped)
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
        self.assertNotIn("| **TIC** | Fixed |", honest)
        self.assertNotIn("| **Baseline** | Default fill |", honest)
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
        self.assertIn(
            "max_model_len follows advertised HF config.json (NIAH_MAX_MODEL_LEN overrides)",
            out,
        )
        self.assertIn("not KV pool tokens", out)
        self.assertIn("not a fixed 4.00", out)
        self.assertIn("Do not grow TIC to 1.75.", out)
        self.assertIn("same lock and the same window", out)
        self.assertIn("fill scrape and compare", out)
        self.assertIn("same-size start or start-fail", out)
        self.assertNotIn("double-until-recall", out)
        self.assertNotIn("first-recall", out)
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
