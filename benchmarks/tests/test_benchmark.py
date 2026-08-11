# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from baseline_cache import (  # noqa: E402
    REQUIRED_PROFILES,
    build_fingerprint,
    copy_baseline,
    find_reusable,
    fingerprint_complete,
)
from benchmark_lib import (  # noqa: E402
    aggregate_replicates,
    effective_bpv,
    fairness_mismatches,
    normalize_vllm_result,
    validate_capacity,
    validate_capacity_memory_story,
    validate_equal_batch,
    validate_profile,
    validate_summary,
    kv_token_capacity_ratio,
    weight_capacity_headroom,
    vllm_non_kv_cache_bytes,
    vllm_requested_memory_bytes,
    weights_and_non_kv_bytes,
)
from capacity_plan import (  # noqa: E402
    build_capacity_doc,
    concurrency_for_seqs,
    default_scale_hi,
    scale_seqs_from_kv_estimate,
    scale_seqs_from_kv_measured,
    search_max,
)
from compare_ab import (  # noqa: E402
    artifact_digest,
    build_equal_batch_block,
    fold_serve_output_match,
)
from generate_report import render, sanitize, write_report  # noqa: E402
from launch_ab import (  # noqa: E402
    resolve_system_id,
)
from serve_output_match import compare_captures  # noqa: E402


def _sample_capacity() -> dict:
    return build_capacity_doc(
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        baseline_max_num_seqs=32,
        tic_max_num_seqs=42,
        baseline_max_concurrency=32,
        tic_max_concurrency=42,
        search_lo=32,
        search_hi=45,
        graph_mode="graphs",
        seqs_scale_mode="kv_estimate",
        kv_scale_ratio_estimated=1.315,
        seqs_implied_weight_estimate=42,
        kv_scale_ratio_measured=1.1168,
        seqs_implied_kv_measured=36,
        kv_measured_source="matched_probe",
    )


def _sample_summary(
    *,
    ttft_tic_slower: bool = True,
    experiment_kind: str = "equal-batch",
) -> dict:
    baseline_ttft = 10.0
    tic_ttft = 12.0 if ttft_tic_slower else 9.0
    summary = {
        "schema": "isiro-benchmark-summary-v1",
        "run_id": "test-run",
        "recorded_at": "2026-07-18T00:00:00+00:00",
        "model": "qwen2.5-7b-instruct",
        "precision": "bf16",
        "system_id": "rtx-5090",
        "isiro_format": "v0.1.0",
        "experiment_kind": experiment_kind,
        "publish_quality": False,
        "smoke": True,
        "fairness": {"passed": True, "mismatches": []},
        "footprint": {
            "baseline": {
                "on_disk_bytes_measured": 1000,
                "gpu_process_memory_after_load_bytes_measured": 2000,
                "gpu_requested_memory_bytes_derived": 2000,
                "weights_and_non_kv_gpu_memory_bytes_derived": 1900,
            },
            "tic": {
                "on_disk_bytes_measured": 600,
                "gpu_process_memory_after_load_bytes_measured": 1800,
                "gpu_requested_memory_bytes_derived": 1800,
                "weights_and_non_kv_gpu_memory_bytes_derived": 1676,
            },
            "on_disk_savings_pct_derived": 40.0,
        },
        "correctness": {
            "integrity_ok": True,
            "bit_exact_ok": True,
            "verify_mode": "reference",
            "serve_output_match_ok": True,
            "serve_output_match_matched": 4,
            "serve_output_match_prompt_count": 4,
            "tic_sha256": "a" * 64,
            "artifact_kind": "directory",
        },
        "kv_cache": {
            "baseline_memory_bytes_measured": 100,
            "tic_memory_bytes_measured": 124,
            "baseline_tokens_measured": 238064,
            "tic_tokens_measured": 294048,
        },
        "profiles": [
            {
                "profile": "ttft-128",
                "input_tokens": 128,
                "output_tokens": 64,
                "baseline": {
                    "latency_ms": {
                        **profile("baseline", "x", 128, 64)["latency_ms"],
                        "ttft_p50": baseline_ttft,
                    },
                    "throughput": profile("baseline", "x", 128, 64)["throughput"],
                },
                "tic": {
                    "latency_ms": {
                        **profile("isiro", "x", 128, 64)["latency_ms"],
                        "ttft_p50": tic_ttft,
                    },
                    "throughput": profile("isiro", "x", 128, 64)["throughput"],
                },
            },
            {
                "profile": "generation-32-256",
                "input_tokens": 32,
                "output_tokens": 256,
                "baseline": {
                    "latency_ms": profile("baseline", "x", 32, 256)["latency_ms"],
                    "throughput": profile("baseline", "x", 32, 256)["throughput"],
                },
                "tic": {
                    "latency_ms": profile("isiro", "x", 32, 256)["latency_ms"],
                    "throughput": profile("isiro", "x", 32, 256)["throughput"],
                },
            },
        ],
    }
    if experiment_kind == "capacity":
        capacity = _sample_capacity()
        capacity["decode_output_tokens_per_sec"] = {
            "baseline": 1000.0,
            "tic": 1400.0,
        }
        capacity["throughput_ratio_derived"] = 1.4
        capacity["efficiency_tok_per_s_per_non_kv_gb_derived"] = {
            "baseline": 100.0,
            "tic": 200.0,
        }
        capacity["kv_token_capacity_ratio_measured"] = round(
            kv_token_capacity_ratio(
                summary["kv_cache"]["baseline_tokens_measured"],
                summary["kv_cache"]["tic_tokens_measured"],
            ),
            4,
        )
        capacity["kv_cache_memory_ratio_measured"] = round(
            float(summary["kv_cache"]["tic_memory_bytes_measured"])
            / float(summary["kv_cache"]["baseline_memory_bytes_measured"]),
            4,
        )
        summary["capacity"] = capacity
        summary["equal_batch"] = build_equal_batch_block(
            max_num_seqs=32,
            max_concurrency=32,
            baseline_tps=1000.0,
            tic_tps=950.0,
            baseline_latency_ms=profile("baseline", "x", 32, 256)["latency_ms"],
            tic_latency_ms=profile("isiro", "x", 32, 256)["latency_ms"],
        )
    elif experiment_kind == "equal-batch":
        summary["equal_batch"] = build_equal_batch_block(
            max_num_seqs=64,
            max_concurrency=32,
            baseline_tps=1000.0,
            tic_tps=900.0,
            baseline_latency_ms=profile("baseline", "x", 32, 256)["latency_ms"],
            tic_latency_ms=profile("isiro", "x", 32, 256)["latency_ms"],
        )
    return summary


def _write_run(run: Path, summary: dict) -> None:
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "environment.json").write_text(
        json.dumps({"gpu_model": "RTX 5090", "gpu_count": 1}),
        encoding="utf-8",
    )
    (run / "commands.json").write_text(
        json.dumps(
            {
                "baseline_serve": ["vllm", "serve", "--enforce-eager"],
                "tic_serve": ["isiro", "serve"],
                "enforce_eager": True,
            }
        ),
        encoding="utf-8",
    )


def profile(variant: str, name: str, input_tokens: int, output_tokens: int) -> dict:
    return {
        "schema": "isiro-benchmark-profile-v1",
        "run_id": "test-run",
        "variant": variant,
        "profile": name,
        "recorded_at": "2026-07-18T00:00:00+00:00",
        "publish_quality": True,
        "smoke": False,
        "workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "serve_config": {
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
            "max_num_seqs": 256,
            "prefix_caching": False,
            "enforce_eager": True,
            "trust_remote_code": True,
            "flashinfer_autotune": False,
        },
        "bench_config": {
            "backend": "openai-chat",
            "endpoint": "/chat/completions",
            "model_id": "model",
            "dataset_name": "random",
            "num_prompts": 200,
            "num_warmups": 50,
            "request_rate": "inf",
            "max_concurrency": 32,
            "burstiness": 1.0,
            "seed": 17,
        },
        "latency_ms": {
            "ttft_p50": 10.0 if variant == "baseline" else 9.0,
            "ttft_p95": 15.0 if variant == "baseline" else 14.0,
            "ttft_p99": 20.0 if variant == "baseline" else 19.0,
            "itl_p50": 2.0,
            "itl_p95": 3.0,
            "itl_p99": 4.0,
            "tpot_p50": 2.1,
            "tpot_p95": 3.1,
            "tpot_p99": 4.1,
            "e2e_p50": 140.0,
            "e2e_p95": 180.0,
            "e2e_p99": 220.0,
        },
        "throughput": {
            "output_tokens_per_sec": 1000.0 if variant == "baseline" else 1100.0,
            "requests_per_sec": 16.0,
            "total_output_tokens": 12800,
        },
        "load_errors": 0,
        "replicates": 3,
    }


def _fingerprint_args(**overrides):
    class NS:
        pass

    args = NS()
    args.vllm_image = "vllm/vllm-openai:v0.25.1"
    args.model_id = "Qwen/Qwen2.5-7B-Instruct"
    args.precision = "bf16"
    args.baseline_model = overrides.get("baseline_model")
    args.max_model_len = "4096"
    args.gpu_memory_utilization = "0.90"
    args.tensor_parallel_size = "1"
    args.max_num_seqs = "256"
    args.enforce_eager = "true"
    args.seed = "17"
    args.warmups = "5"
    args.prompts = "12"
    args.replicates = "1"
    args.max_concurrency = "32"
    args.publish_quality = False
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class BenchmarkTests(unittest.TestCase):
    def test_normalize_requires_and_extracts_p95(self) -> None:
        raw = json.loads(
            (ROOT / "tests/fixtures/vllm-result.json").read_text(encoding="utf-8")
        )
        result = normalize_vllm_result(raw)
        self.assertEqual(result["latency_ms"]["ttft_p95"], 15.0)
        self.assertEqual(result["latency_ms"]["itl_p99"], 4.0)

    def test_normalize_refuses_missing_p95(self) -> None:
        raw = json.loads(
            (ROOT / "tests/fixtures/vllm-result.json").read_text(encoding="utf-8")
        )
        raw["percentiles_ttft_ms"] = [[50, 10.0], [99, 20.0]]
        with self.assertRaisesRegex(ValueError, "p95"):
            normalize_vllm_result(raw)

    def test_replicate_aggregation_uses_median(self) -> None:
        raw = json.loads(
            (ROOT / "tests/fixtures/vllm-result.json").read_text(encoding="utf-8")
        )
        items = []
        for throughput in (900.0, 1100.0, 1000.0):
            current = dict(raw)
            current["output_throughput"] = throughput
            items.append(normalize_vllm_result(current))
        result = aggregate_replicates(items)
        self.assertEqual(result["throughput"]["output_tokens_per_sec"], 1000.0)

    def test_fairness_reports_specific_mismatch(self) -> None:
        baseline = profile("baseline", "ttft-128", 128, 64)
        tic = profile("isiro", "ttft-128", 128, 64)
        tic["bench_config"]["seed"] = 99
        env = {
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "driver_version": "1",
            "cuda_version": "1",
            "vllm_version": "0.25.1",
            "substrate": "docker",
        }
        mismatches = fairness_mismatches(baseline, tic, env, env)
        self.assertEqual(len(mismatches), 1)
        self.assertIn("bench_config.seed", mismatches[0])

    def test_fairness_ignores_substrate_and_image_digest(self) -> None:
        baseline = profile("baseline", "ttft-128", 128, 64)
        tic = profile("isiro", "ttft-128", 128, 64)
        baseline_env = {
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "driver_version": "1",
            "cuda_version": "1",
            "vllm_version": "0.25.1",
            "substrate": "docker",
            "image_digest": "sha256:baseline",
        }
        tic_env = {
            **baseline_env,
            "substrate": "host_isiro",
            "image_digest": "",
        }
        self.assertEqual(
            fairness_mismatches(baseline, tic, baseline_env, tic_env), []
        )

    def test_fairness_equal_batch_requires_matched_concurrency(self) -> None:
        baseline = profile("baseline", "generation-32-256", 32, 256)
        tic = profile("isiro", "generation-32-256", 32, 256)
        tic["bench_config"]["max_concurrency"] = 64
        tic["serve_config"]["max_num_seqs"] = 128
        env = {
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "driver_version": "1",
            "cuda_version": "1",
            "vllm_version": "0.25.1",
            "substrate": "docker",
        }
        mismatches = fairness_mismatches(
            baseline, tic, env, env, experiment_kind="equal-batch"
        )
        self.assertTrue(any("max_concurrency" in item for item in mismatches))
        self.assertTrue(any("max_num_seqs" in item for item in mismatches))

    def test_fairness_capacity_allows_differing_batch(self) -> None:
        baseline = profile("baseline", "generation-32-256", 32, 256)
        tic = profile("isiro", "generation-32-256", 32, 256)
        tic["bench_config"]["max_concurrency"] = 64
        tic["serve_config"]["max_num_seqs"] = 128
        env = {
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "driver_version": "1",
            "cuda_version": "1",
            "vllm_version": "0.25.1",
            "substrate": "docker",
        }
        self.assertEqual(
            fairness_mismatches(
                baseline, tic, env, env, experiment_kind="capacity"
            ),
            [],
        )

    def test_capacity_schema_and_search_helpers(self) -> None:
        doc = _sample_capacity()
        self.assertEqual(validate_capacity(doc), [])
        self.assertEqual(concurrency_for_seqs(96, 32), 32)
        self.assertEqual(concurrency_for_seqs(16, 32), 16)
        self.assertEqual(search_max(1, 10, lambda n: n <= 7), 7)
        self.assertEqual(search_max(1, 10, lambda n: False), 0)
        bad = dict(doc)
        bad["experiment_kind"] = "equal-batch"
        self.assertTrue(validate_capacity(bad))

    def test_capacity_search_hi_first(self) -> None:
        from capacity_plan import search_max_hi_first

        probes: list[int] = []

        def probe(n: int) -> bool:
            probes.append(n)
            return n <= 200

        self.assertEqual(search_max_hi_first(1, 256, probe), 200)
        self.assertEqual(probes[0], 256)
        probes.clear()

        def always_ok(n: int) -> bool:
            probes.append(n)
            return True

        self.assertEqual(search_max_hi_first(1, 128, always_ok), 128)
        self.assertEqual(probes, [128])

    def test_weight_capacity_headroom_is_sixteen_over_bpv(self) -> None:
        self.assertAlmostEqual(effective_bpv(1000, 600), 9.6)
        self.assertAlmostEqual(weight_capacity_headroom(9.6), 16 / 9.6, places=4)
        self.assertAlmostEqual(weight_capacity_headroom(11.41), 16 / 11.41, places=4)

    def test_kv_token_capacity_ratio_from_sample_tokens(self) -> None:
        self.assertAlmostEqual(
            kv_token_capacity_ratio(238064, 294048),
            294048 / 238064,
            places=4,
        )
        with self.assertRaises(ValueError):
            kv_token_capacity_ratio(0, 100)

    def test_scale_seqs_from_kv_estimate(self) -> None:
        # ~12.71 GiB KV + ~4 GiB freed weights ≈ 1.31x → 32 scales above baseline
        kv_b = int(12.71 * 1024**3)
        w_b = 14 * 1024**3
        w_t = 10 * 1024**3
        hi = default_scale_hi(32, w_b, w_t)
        self.assertEqual(hi, round(32 * 14 / 10))
        scaled = scale_seqs_from_kv_estimate(32, kv_b, w_b, w_t, hi)
        freed = w_b - w_t
        expected = max(32, min(hi, round(32 * (kv_b + freed) / kv_b)))
        self.assertEqual(scaled, expected)
        self.assertGreater(scaled, 32)
        self.assertEqual(
            scale_seqs_from_kv_estimate(32, kv_b, w_b, w_t, 32), 32
        )

    def test_scale_seqs_from_kv_measured(self) -> None:
        # Remeter sample: 238064 → 265872 tokens ≈ 1.12x → 36
        self.assertEqual(
            scale_seqs_from_kv_measured(32, 238064, 265872, 45), 36
        )
        self.assertEqual(
            scale_seqs_from_kv_measured(32, 238064, 265872, 32), 32
        )
        self.assertEqual(
            scale_seqs_from_kv_measured(32, 100, 50, 45), 32
        )

    def test_equal_batch_block_has_raw_ratio_only(self) -> None:
        self.assertAlmostEqual(effective_bpv(1000, 600), 9.6)
        block = build_equal_batch_block(
            max_num_seqs=32,
            max_concurrency=32,
            baseline_tps=1000.0,
            tic_tps=750.0,
        )
        self.assertEqual(validate_equal_batch(block), [])
        self.assertEqual(block["throughput_ratio_derived"], 0.75)
        self.assertNotIn(
            "throughput_ratio_footprint_adjusted_derived", block
        )
        self.assertEqual(
            weights_and_non_kv_bytes(2000, 100),
            1900,
        )
        self.assertEqual(
            vllm_non_kv_cache_bytes(
                requested_bytes=2000, available_kv_bytes=100
            ),
            1900,
        )
        self.assertEqual(
            vllm_non_kv_cache_bytes(non_kv_reported=1234),
            1234,
        )
        self.assertEqual(
            vllm_requested_memory_bytes(10_000, 0.9),
            9000,
        )
        # vLLM format_gib often prints "11.76GiB" with no space before GiB.
        sample = (
            "DEBUG worker requested memory: 28.22GiB\n"
            "DEBUG Memory profiling takes 1.21 seconds. "
            "Total non KV cache memory: 11.76GiB; "
            "torch peak memory increase: 0.71GiB; "
            "non-torch forward increase memory: 0.21GiB; "
            "weights memory: 10.84GiB.\n"
            "INFO Available KV cache memory: 16.46 GiB\n"
        )
        import re

        non_kv = re.search(
            r"Total non KV cache memory:\s*([0-9.]+)\s*GiB", sample
        )
        req = re.search(
            r"worker requested memory:\s*([0-9.]+)\s*GiB", sample
        )
        kv = re.search(
            r"Available KV cache memory:\s*([0-9.]+)\s*GiB", sample
        )
        self.assertIsNotNone(non_kv)
        self.assertIsNotNone(req)
        self.assertIsNotNone(kv)
        self.assertAlmostEqual(float(non_kv.group(1)), 11.76)
        self.assertAlmostEqual(
            float(non_kv.group(1)) + float(kv.group(1)),
            float(req.group(1)),
        )
        story_ok = validate_capacity_memory_story(
            _sample_summary(experiment_kind="capacity")
        )
        self.assertEqual(story_ok, [])
        asym = _sample_summary(experiment_kind="capacity")
        asym["capacity"]["kv_measured_source"] = "timed_asymmetric"
        asym_errs = validate_capacity_memory_story(asym)
        self.assertTrue(
            any("matched_probe" in e for e in asym_errs),
            asym_errs,
        )
        lose = _sample_summary(experiment_kind="capacity")
        lose["kv_cache"]["tic_memory_bytes_measured"] = 90
        lose_errs = validate_capacity_memory_story(lose)
        self.assertTrue(any("§B fail" in e and "KV memory" in e for e in lose_errs), lose_errs)

    def test_equal_batch_reuse_is_mode_agnostic(self) -> None:
        text = (SCRIPTS / "harness_lib.sh").read_text(encoding="utf-8")
        start = text.index("run_equal_batch_transparency()")
        end = text.index(
            '\nif [[ "${EXPERIMENT_KIND}" == "capacity" && "${CAPACITY_EQUAL_BATCH}"',
            start,
        )
        body = text[start:end]
        self.assertIn("equal-batch baseline: reused capacity timed profiles", body)
        self.assertIn("equal-batch TIC: reused capacity timed profiles", body)
        self.assertIn("equal-batch baseline: cold start", body)
        self.assertIn("equal-batch TIC: cold start", body)
        self.assertIn("copy_equal_batch_profile_from_capacity", body)
        for banned in ("both-graph", "--graph-on", "--graph-off", "BOTH_GRAPH"):
            self.assertNotIn(banned, body)
        # Op-point match only (seqs + concurrency), same for every README mode.
        self.assertIn('"${matched_seqs}" == "${capacity_base_seqs}"', body)
        self.assertIn('"${matched_seqs}" == "${capacity_tic_seqs}"', body)
        self.assertIn('"${matched_conc}" == "${capacity_base_conc}"', body)
        self.assertIn('"${matched_conc}" == "${capacity_tic_conc}"', body)
        launch = (SCRIPTS / "launch_ab.py").read_text(encoding="utf-8")
        self.assertIn('("eager", False, True)', launch)
        self.assertIn('("graphs", True, False)', launch)

    def test_profile_validation(self) -> None:
        self.assertEqual(validate_profile(profile("baseline", "x", 128, 64)), [])

    def test_directory_digest_is_stable_and_path_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a").write_bytes(b"same")
            first = artifact_digest(root)
            (root / "a").rename(root / "b")
            second = artifact_digest(root)
            self.assertNotEqual(first, second)

    def test_report_render_basics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            _write_run(run, _sample_summary(ttft_tic_slower=True))
            report = render(run)
            self.assertNotIn("## Optional: Layer GEMM", report)
            self.assertIn("## Correctness", report)
            self.assertNotIn("## A. Correctness", report)
            self.assertIn("## 1C. Generation (input 32 / output 256)", report)
            self.assertIn("## 1D. TTFT", report)
            self.assertNotIn("## E. Prefill TTFT", report)
            self.assertIn("## 1E. Equal batch", report)
            self.assertNotIn("## 1E. Equal-batch", report)
            self.assertNotIn("capacity benefit not active", report)
            self.assertIn("Output tok/s", report)
            self.assertIn("Tooling: **`vllm bench serve`**.", report)
            self.assertIn("| Graph modes | OFF (eager) |", report)
            self.assertNotIn(
                "Graph mode: `eager` (product default is eager; graphs is opt-in).",
                report,
            )
            self.assertNotIn("Smaller on-disk weights free", report)
            self.assertNotIn("Experiment:", report)
            self.assertIn("Serve output match", report)
            self.assertNotIn("greedy vLLM", report)
            self.assertIn("`isiro serve … --target vllm`", report)
            self.assertIn("Baseline", report)
            self.assertNotIn("E2E latency", report)
            self.assertNotIn("Footprint-adjusted tok/s ratio", report)
            self.assertNotIn("matched A/B", report)
            self.assertNotIn("ISIRO_SERVE_CUDA_GRAPHS", report)
            self.assertNotIn("Fairness gate", report)
            self.assertIn("Fairness check", report)

    def test_report_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            summary = _sample_summary(ttft_tic_slower=True, experiment_kind="capacity")
            summary["publish_quality"] = True
            summary["smoke"] = False
            _write_run(run, summary)
            report = render(run)
            self.assertNotIn("## Optional: Layer GEMM", report)
            self.assertNotIn("## How to read this report", report)
            self.assertNotIn("## Limitations", report)
            self.assertNotIn("Review refresh", report)
            self.assertNotIn("Why lower on-disk", report)
            self.assertIn(
                "† Derived: output tok/s ÷ Non-KV GPU memory (GB).",
                report,
            )
            self.assertIn("Physical meaning:", report)
            self.assertIn("† tok/s per Non-KV GB", report)
            self.assertNotIn("† tok/s per weight GB", report)
            self.assertIn("## Correctness", report)
            self.assertIn("## 1A. GPU memory", report)
            self.assertIn("## 1B. Capacity", report)
            self.assertIn("## 1C. Generation (input 32 / output 256)", report)
            self.assertIn("## 1D. TTFT", report)
            self.assertNotIn("## E. Prefill TTFT", report)
            self.assertIn("## 1E. Equal batch", report)
            self.assertIn("## Config", report)
            self.assertNotIn("## G. Config", report)
            self.assertNotIn("## G. Config / fairness / reproduce", report)
            self.assertIn("`max_num_seqs`", report)
            self.assertIn(
                "TIC `max_num_seqs` = round(baseline × estimated KV ratio)",
                report,
            )
            self.assertIn(
                "KV token capacity is vLLM-reported (`GPU KV cache size`",
                report,
            )
            self.assertNotIn("weight-estimate KV-scaled timed point", report)
            self.assertNotIn("TIC concurrency scale", report)
            self.assertNotIn("SKU / `system_id`", report)
            self.assertNotIn("| Capacity scaling |", report)
            self.assertIn("| System |", report)
            self.assertIn("| Graph modes |", report)
            self.assertIn("| Loaded model size |", report)
            self.assertIn("Norm savings %", report)
            self.assertIn(
                "Norm savings % scales TIC to the Baseline total GPU",
                report,
            )
            self.assertIn(
                "Non-KV and KV cache are vLLM-reported",
                report,
            )
            self.assertIn("nvidia-smi process usage", report)
            self.assertIn("% (", report)
            self.assertRegex(report, r"\d+\.\d+% \(\d+\.\d+x\)")
            self.assertNotIn("% more", report)
            self.assertNotIn("Model load (reported)", report)
            self.assertNotIn("| Model load |", report)
            self.assertNotIn("torch allocated", report)
            self.assertNotIn("torch reserved", report)
            self.assertNotIn("GPU process at ready", report)
            self.assertNotIn("Bench `max_concurrency`", report)
            self.assertNotIn("Weight-capacity headroom", report)
            self.assertNotIn("Concurrency implied by weight estimate", report)
            self.assertNotIn("Concurrency implied by measured KV", report)
            self.assertNotIn("KV cache dtype", report)
            self.assertNotIn("KV element bpv", report)
            self.assertNotIn("† KV element bytes ratio", report)
            self.assertIn("**1.31x**", report)  # batch_ratio 42/32
            self.assertIn("† tok/s per Non-KV GB", report)
            self.assertNotIn("† tok/s per weight GB", report)
            self.assertNotIn("† tok/s per KV GB", report)
            self.assertNotIn("## Optional: Layer GEMM", report)
            self.assertNotIn("Footprint-adjusted tok/s ratio", report)
            self.assertNotIn("queueing and prefill contention", report)

    def test_publish_summary_ok(self) -> None:
        summary = _sample_summary()
        summary["publish_quality"] = True
        summary["smoke"] = False
        self.assertEqual(validate_summary(summary), [])

    def test_publish_summary_reference_requires_bit_exact(self) -> None:
        summary = _sample_summary()
        summary["publish_quality"] = True
        summary["smoke"] = False
        summary["correctness"]["verify_mode"] = "reference"
        summary["correctness"].pop("bit_exact_ok", None)
        errors = validate_summary(summary)
        self.assertTrue(any("bit_exact_ok" in item for item in errors))
        summary["correctness"]["bit_exact_ok"] = True
        self.assertEqual(validate_summary(summary), [])

    def test_system_id_resolve_and_reject(self) -> None:
        self.assertEqual(resolve_system_id("a100", None), "a100")
        self.assertEqual(resolve_system_id(None, "h100"), "h100")
        with self.assertRaises(SystemExit):
            resolve_system_id("mi300", None)

    def test_launch_requires_model(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "--dry-run",
                "--system-id",
                "rtx-5090",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(ROOT.parent),
        )
        self.assertNotEqual(result.returncode, 0)

    def test_launch_dry_meta_resolves_qwen_harness(self) -> None:
        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "ISIRO_BENCH_LAUNCH_DRY_META": "1",
        }
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "qwen2.5-7b-instruct",
                "--dry-run",
                "--system-id",
                "rtx-5090",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(ROOT.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scripts/run_harness.sh", result.stdout)
        self.assertIn("MODEL_SLUG=qwen2.5-7b-instruct", result.stdout)
        self.assertIn("system_id=rtx-5090", result.stdout)
        self.assertIn("ISIRO_VERIFY_REFERENCE=1", result.stdout)
        self.assertIn("ISIRO_BENCH_EQUAL_BATCH=1", result.stdout)
        self.assertIn("--graph-on", result.stdout)

    def test_launch_both_graph_modes_dry_meta(self) -> None:
        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "ISIRO_BENCH_LAUNCH_DRY_META": "1",
        }
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "qwen2.5-7b-instruct",
                "--dry-run",
                "--system-id",
                "rtx-5090",
                "--both-graph-modes",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(ROOT.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("eager:", result.stdout)
        self.assertIn("graphs:", result.stdout)
        self.assertIn("--graph-off", result.stdout)
        self.assertIn("--graph-on", result.stdout)

    def test_launch_rejects_both_with_graphs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "qwen2.5-7b-instruct",
                "--dry-run",
                "--system-id",
                "rtx-5090",
                "--both-graph-modes",
                "--graph-on",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(ROOT.parent),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pass only one of", result.stderr)

    def test_harness_docs_omit_isiro_core(self) -> None:
        banned = "isiro" + "-core"
        skip_names = {"common.env", "common.env.fp8"}
        for path in (ROOT).rglob("*"):
            if not path.is_file():
                continue
            if path.name in skip_names:
                continue
            rel = path.as_posix()
            if "scratch" in rel or "/private/" in rel or "/tests/" in rel:
                continue
            if path.suffix not in {".md", ".sh", ".py", ".json", ".example"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn(banned, text, msg=str(path))

    def test_report_sanitize_redacts_identity_and_paths(self) -> None:
        scratch = Path("/tmp/scratch-run")
        doc = {
            "gpus": [
                {
                    "name": "RTX 5090",
                    "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                    "pci_bus_id": "0000:01:00.0",
                }
            ],
            "command": [
                f"{Path.home()}/models/baseline",
                "/opt/vendor/bin/vllm",
            ],
            "note": f"see {scratch}/logs/baseline-serve.log",
        }
        cleaned = sanitize(doc, scratch)
        self.assertEqual(cleaned["gpus"][0]["uuid"], "<REDACTED>")
        self.assertEqual(cleaned["gpus"][0]["pci_bus_id"], "<REDACTED>")
        self.assertEqual(cleaned["command"][0], "<OPERATOR_PATH>/baseline")
        self.assertEqual(cleaned["command"][1], "<OPERATOR_PATH>/vllm")
        self.assertIn("<SCRATCH_RUN>", cleaned["note"])
        deep = sanitize(
            {
                "mount": (
                    f"{Path.home()}/repos/isiroai/private-tree/models/"
                    "Qwen2.5-7B-Instruct:/model:ro"
                )
            },
            scratch,
        )
        self.assertEqual(deep["mount"], "<OPERATOR_PATH>/Qwen2.5-7B-Instruct:/model:ro")
        self.assertNotIn("private-tree", deep["mount"])
        api = sanitize({"endpoint": "/v1/completions"}, scratch)
        self.assertEqual(api["endpoint"], "/v1/completions")

    def test_write_report_only_to_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            run = repo / "scratch" / "run"
            (run / "baseline").mkdir(parents=True)
            (run / "isiro").mkdir()
            summary = _sample_summary(ttft_tic_slower=True, experiment_kind="capacity")
            summary["publish_quality"] = True
            summary["smoke"] = False
            (run / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            env = {
                "schema": "isiro-benchmark-environment-v1",
                "system_id": "rtx-5090",
                "gpu_model": "NVIDIA GeForce RTX 5090",
                "gpu_count": 1,
                "driver_version": "590.44.01",
                "cuda_version": "13.1",
                "vllm_version": "0.26.0",
                "substrate": "docker",
                "variant_runtime": {
                    "baseline": {"substrate": "docker"},
                    "tic": {"substrate": "host_isiro"},
                },
            }
            (run / "environment.json").write_text(
                json.dumps(env), encoding="utf-8"
            )
            (run / "commands.json").write_text(
                json.dumps(
                    {
                        "baseline_serve": ["vllm", "serve"],
                        "tic_serve": ["isiro", "serve"],
                        "enforce_eager": False,
                    }
                ),
                encoding="utf-8",
            )
            (run / "verify.json").write_text("{}", encoding="utf-8")
            for variant in ("baseline", "isiro"):
                (run / variant / "generation-32-256.json").write_text(
                    json.dumps(profile(variant, "generation-32-256", 32, 256)),
                    encoding="utf-8",
                )
            dest = (
                repo
                / "benchmarks"
                / summary["model"].lower()
                / f"{summary['system_id']}-report-20260811T120000Z.md"
            )
            write_report(run, dest)
            self.assertTrue(dest.is_file())
            names = {p.name for p in dest.parent.iterdir()}
            self.assertEqual(names, {"rtx-5090-report-20260811T120000Z.md"})
            self.assertFalse((dest.parent / "report.md").exists())
            self.assertFalse((dest.parent / "tic").exists())
            self.assertFalse((dest.parent / "isiro").exists())
            self.assertFalse((dest.parent / "summary.json").exists())

    def test_public_methodology_omits_research_claims(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        private_tree = "isiro" + "-core"
        for banned in ("NUS", "arXiv", "10.65", "tANS", private_tree):
            self.assertNotIn(banned, text)
        self.assertNotIn("Measured vs derived", text)
        self.assertNotIn("Footprint-adjusted", text)
        self.assertNotIn("Publish-quality", text)
        self.assertNotIn("Do not scale", text)
        self.assertNotIn("upstream", text)
        self.assertIn("vllm bench serve", text)
        self.assertIn("{model}/{system_id}-report-<UTC>.md", text)
        self.assertIn("benchmarks/{model}/{system_id}-report-<UTC>.md", text)
        self.assertNotIn("benchmarks/scratch/<run-id>/report.md", text)
        self.assertIn("benchmarks/scratch/", text)
        self.assertNotIn("Rename to `report.md`", text)
        self.assertIn("benchmarks/run_ab.sh {model}\n", text)
        self.assertIn("benchmarks/run_ab.sh {model} --graph-off", text)
        self.assertIn("benchmarks/run_ab.sh {model} --both-graph-modes", text)
        self.assertNotIn("benchmarks/run_ab.sh {model} --smoke", text)
        self.assertIn("benchmarks/common.env.example", text)
        self.assertIn("BASELINE_MODEL_DIR", text)
        self.assertIn("TIC_MODEL_DIR", text)
        self.assertIn("Graph ON", text)
        self.assertIn("directory name under `benchmarks/`", text)
        self.assertNotIn("# set BASELINE_MODEL_DIR", text)
        self.assertNotIn("export PATH=", text)
        self.assertNotIn("ISIRO_VERIFY_REFERENCE=1 ISIRO_BENCH_EQUAL_BATCH=1", text)
        self.assertIn("## Prerequisites", text)
        self.assertIn("isiro.ai/compiler", text)
        self.assertNotIn("ISIRO_VERIFY_REFERENCE=0", text)
        self.assertNotIn("## Defaults", text)
        self.assertNotIn("## Results", text)
        self.assertNotIn("promote_run", text)
        self.assertNotIn("ISIRO_SERVE_CUDA_GRAPHS", text)
        self.assertNotIn("results/published", text)
        self.assertNotIn("decode-32-256", text)
        self.assertIn("correctness", text.lower())
        self.assertIn("memory", text.lower())
        self.assertIn("capacity", text.lower())
        self.assertIn("throughput", text.lower())
        self.assertIn("generation", text.lower())
        self.assertIn("latency", text.lower())

    def test_scripts_omit_invent_layout_jargon(self) -> None:
        private_tree = "isiro" + "-core"
        banned = ("ms+E", "dense-E", "coded-E", "Path A", private_tree)
        targets = [ROOT / "README.md", SCRIPTS / "harness_lib.sh"]
        targets.extend(sorted(SCRIPTS.glob("*.py")))
        for path in targets:
            text = path.read_text(encoding="utf-8")
            for term in banned:
                self.assertNotIn(
                    term,
                    text,
                    msg=f"{path.relative_to(ROOT)} contains banned term {term!r}",
                )

    def test_baseline_fingerprint_stable_and_incomplete_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "a.safetensors").write_bytes(b"abc")
            first = build_fingerprint(_fingerprint_args(baseline_model=model))
            second = build_fingerprint(_fingerprint_args(baseline_model=model))
            self.assertEqual(first["fingerprint"], second["fingerprint"])

            source = root / "source-run"
            source.mkdir()
            (source / "baseline_fingerprint.json").write_text(
                json.dumps(first), encoding="utf-8"
            )
            self.assertFalse(fingerprint_complete(source))
            self.assertIsNone(
                find_reusable(
                    root,
                    first["fingerprint"],
                    require_publish_quality=False,
                )
            )

            (source / "baseline").mkdir()
            (source / "baseline_environment.json").write_text("{}", encoding="utf-8")
            (source / "baseline_serve_config.json").write_text("{}", encoding="utf-8")
            for name in REQUIRED_PROFILES:
                (source / "baseline" / f"{name}.json").write_text(
                    "{}", encoding="utf-8"
                )
            self.assertTrue(fingerprint_complete(source))
            hit = find_reusable(
                root,
                first["fingerprint"],
                require_publish_quality=False,
            )
            self.assertEqual(hit, source)

            dest = root / "dest-run"
            copy_baseline(source, dest)
            self.assertTrue(fingerprint_complete(dest))

    def test_serve_output_match_compare_token_ids(self) -> None:
        capture = {
            "schema": "isiro-benchmark-output-match-v1",
            "kind": "capture",
            "seed": 17,
            "max_tokens": 32,
            "prompts": [
                {"index": 0, "prompt": "a", "token_ids": [1, 2, 3], "text": "x"},
                {"index": 1, "prompt": "b", "token_ids": [4, 5], "text": "y"},
            ],
        }
        ok = compare_captures(capture, capture)
        self.assertTrue(ok["serve_output_match_ok"])
        self.assertEqual(ok["matched"], 2)
        bad = dict(capture)
        bad["prompts"] = [
            capture["prompts"][0],
            {**capture["prompts"][1], "token_ids": [9, 9]},
        ]
        fail = compare_captures(capture, bad)
        self.assertFalse(fail["serve_output_match_ok"])
        self.assertEqual(fail["matched"], 1)

    def test_report_dual_graph_companion_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            eager = root / "eager"
            graphs = root / "graphs"
            eager.mkdir()
            graphs.mkdir()
            summary = _sample_summary(ttft_tic_slower=True, experiment_kind="capacity")
            summary["publish_quality"] = True
            summary["smoke"] = False
            _write_run(eager, summary)
            _write_run(graphs, summary)
            (graphs / "commands.json").write_text(
                json.dumps(
                    {
                        "baseline_serve": ["vllm", "serve"],
                        "tic_serve": ["isiro", "serve"],
                        "enforce_eager": False,
                    }
                ),
                encoding="utf-8",
            )
            report = render(eager, companion_run_dir=graphs)
            self.assertIn("Tooling: **`vllm bench serve`**.", report)
            self.assertIn("# ISIRO Benchmark Report: `qwen2.5-7b-instruct`", report)
            self.assertIn("`bf16` | `rtx-5090` |", report)
            self.assertRegex(report, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC")
            self.assertIn("## Graph ON (CUDA graphs)", report)
            self.assertIn("## Graph OFF (eager)", report)
            self.assertNotIn("<details>", report)
            self.assertNotIn("<summary>", report)
            self.assertNotIn("ISIRO_SERVE_CUDA_GRAPHS", report)
            self.assertNotIn("matched A/B", report)
            self.assertIn("CUDA graphs on (product default; `--graph-on`).", report)
            self.assertIn("CUDA graphs off (`--graph-off`).", report)
            # Dual mode uses full 1A-1E / 2A-2E tables (both visible).
            self.assertEqual(
                report.count("## 1C. Generation (input 32 / output 256)"), 1
            )
            self.assertEqual(
                report.count("## 2C. Generation (input 32 / output 256)"), 1
            )
            self.assertLess(
                report.index("## Graph ON (CUDA graphs)"),
                report.index("## Graph OFF (eager)"),
            )
            self.assertLess(
                report.index("## Graph OFF (eager)"),
                report.index("## Config"),
            )
            self.assertEqual(report.count("## 1D. TTFT"), 1)
            self.assertEqual(report.count("## 2D. TTFT"), 1)
            self.assertIn(
                "† Derived: output tok/s ÷ Non-KV GPU memory (GB).",
                report,
            )
            self.assertIn("Physical meaning:", report)
            self.assertIn(
                "TTFT is separated from generation (1C) because it is more ",
                report,
            )
            self.assertIn(
                "TTFT is separated from generation (2C) because it is more ",
                report,
            )
            self.assertNotIn("opt-in CUDA graphs) - matched A/B", report)
            self.assertNotIn("Reproduce:", report)
            self.assertNotIn("Baseline server:", report)
            self.assertNotIn("TIC server:", report)
            self.assertNotIn("--both-graph-modes", report)
            self.assertIn("Weight bit-exactness", report)
            self.assertIn("4/4 prompts, temp=0, token IDs equal", report)
            self.assertIn(
                "ON (CUDA graphs); OFF (eager)",
                report,
            )

    def test_fold_serve_output_match_isiro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            capture = {
                "prompts": [
                    {"prompt": "a", "token_ids": [1, 2]},
                    {"prompt": "b", "token_ids": [3, 4]},
                ],
            }
            (run / "output_match_baseline.json").write_text(
                json.dumps(capture), encoding="utf-8"
            )
            (run / "output_match_isiro.json").write_text(
                json.dumps(capture), encoding="utf-8"
            )
            correctness: dict = {}
            fold_serve_output_match(run, correctness)
            self.assertTrue(correctness["serve_output_match_ok"])
            self.assertEqual(correctness["serve_output_match_matched"], 2)
            self.assertEqual(correctness["serve_output_match_prompt_count"], 2)
            self.assertTrue((run / "output_match.json").is_file())
            compare_src = (SCRIPTS / "compare_ab.py").read_text(encoding="utf-8")
            self.assertIn("output_match_isiro.json", compare_src)
            self.assertNotIn("output_match_tic.json", compare_src)

    def test_report_serve_output_match_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            summary = _sample_summary(ttft_tic_slower=True)
            summary["correctness"].pop("serve_output_match_ok", None)
            summary["correctness"].pop("serve_output_match_matched", None)
            summary["correctness"].pop("serve_output_match_prompt_count", None)
            _write_run(run, summary)
            report = render(run)
            self.assertIn(
                "| Serve output match | - | not run |",
                report,
            )
            self.assertNotIn("| Serve output match | not run |", report)
            self.assertNotIn("| Verify mode |", report)
            self.assertNotIn("greedy vLLM", report)


if __name__ == "__main__":
    unittest.main()
