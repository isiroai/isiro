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
    resolve_fp8_baseline_quantization,
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
    align_seqs_to_cudagraph_stride,
    build_capacity_doc,
    coded_kv_token_capacity,
    concurrency_for_seqs,
    default_scale_hi,
    gdx_capacity_scale,
    gdx_cold_savings_from_bpv,
    scale_seqs_from_kv_estimate,
    scale_seqs_from_kv_measured,
    search_max,
)
from compare_ab import (  # noqa: E402
    artifact_digest,
    build_equal_batch_block,
    fold_gdx_kv_bitexact,
    fold_serve_output_match,
)
from gdx_kv_bitexact import run_meter as gdx_kv_run_meter  # noqa: E402
from attention_ab import parse_serve_attention  # noqa: E402
from generate_report import (  # noqa: E402
    _graph_mode,
    bytes_gb,
    normalized_gpu_gain_pct,
    pct_of_baseline,
    render,
    sanitize,
    vllm_visible_budget,
    write_report,
)
from launch_ab import (  # noqa: E402
    resolve_system_id,
)
from probe_isiro_versions import (  # noqa: E402
    resolve_versions,
    runtime_version_from_help,
    tic_compiler_version,
)
from serve_output_match import (  # noqa: E402
    compare_captures,
    greedy_chat_payload,
    greedy_protocol_errors,
    thinking_controls_present,
)


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

    def test_fairness_ignores_attention_backend(self) -> None:
        baseline = profile("baseline", "ttft-128", 128, 64)
        tic = profile("isiro", "ttft-128", 128, 64)
        baseline["serve_config"]["attention_backend"] = "FLASH_ATTN"
        baseline["serve_config"]["isiro_attn"] = "unset"
        tic["serve_config"]["attention_backend"] = "CUSTOM"
        tic["serve_config"]["isiro_attn"] = "unset"
        env = {
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "driver_version": "1",
            "cuda_version": "1",
            "vllm_version": "0.25.1",
        }
        self.assertEqual(fairness_mismatches(baseline, tic, env, env), [])

    def test_fairness_rejects_cudagraph_compile_mismatch(self) -> None:
        baseline = profile("baseline", "generation-32-256", 32, 256)
        tic = profile("isiro", "generation-32-256", 32, 256)
        baseline["serve_config"]["cudagraph_mode"] = "FULL_AND_PIECEWISE"
        baseline["serve_config"]["compilation_mode"] = "VLLM_COMPILE"
        tic["serve_config"]["cudagraph_mode"] = "FULL_DECODE_ONLY"
        tic["serve_config"]["compilation_mode"] = "DYNAMO_TRACE_ONCE"
        env = {
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "driver_version": "1",
            "cuda_version": "1",
            "vllm_version": "0.31.0",
        }
        mismatches = fairness_mismatches(baseline, tic, env, env)
        self.assertTrue(any("cudagraph_mode" in m for m in mismatches))
        self.assertTrue(any("compilation_mode" in m for m in mismatches))
        # Missing on one side: do not fail (older logs).
        del tic["serve_config"]["cudagraph_mode"]
        del tic["serve_config"]["compilation_mode"]
        self.assertEqual(fairness_mismatches(baseline, tic, env, env), [])

    def test_parse_serve_attention(self) -> None:
        parsed = parse_serve_attention(
            "Using AttentionBackendEnum.CUSTOM\n"
            "ready: CUSTOM vendor-family drop-in\n"
        )
        self.assertEqual(parsed["attention_backend"], "CUSTOM")

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
        tic["bench_config"]["num_prompts"] = 64
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

    def test_fairness_equal_batch_rejects_reused_baseline_run_id(self) -> None:
        baseline = profile("baseline", "generation-32-256", 32, 256)
        tic = profile("isiro", "generation-32-256", 32, 256)
        baseline["run_id"] = "old-path-a-ttft"
        tic["run_id"] = "fresh-equal-batch"
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
        self.assertTrue(any("run_id" in item for item in mismatches), mismatches)
        # Capacity may reuse baseline artifacts under SKIP_BASELINE.
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

    def test_pct_of_baseline_from_token_ratio(self) -> None:
        self.assertEqual(pct_of_baseline(20480 / 238064), "9% of Baseline")
        self.assertEqual(pct_of_baseline(294048 / 238064), "124% of Baseline")
        self.assertEqual(pct_of_baseline(None), "")

    def test_probe_isiro_versions_auto_and_pin(self) -> None:
        import struct

        header = json.dumps(
            {"__metadata__": {"codec": "v0.1.1", "tic_format": "v0.1.0"}}
        ).encode()
        body = (
            b"TIC\x00"
            + (b"\x00" * 8)
            + header
            + struct.pack("<Q", len(header))
            + b"THDR"
        )
        with tempfile.TemporaryDirectory() as temp:
            tic = Path(temp) / "model.tic"
            tic.write_bytes(body)
            self.assertEqual(tic_compiler_version(tic), "v0.1.1")
            self.assertEqual(
                runtime_version_from_help("ISIRO Runtime v0.1.1\n\nUsage"),
                "v0.1.1",
            )
            compiler, runtime = resolve_versions(
                tic_path=tic,
                format_override="auto",
                runtime_override="v0.1.0",
            )
            self.assertEqual((compiler, runtime), ("v0.1.1", "v0.1.0"))
            pinned, _ = resolve_versions(
                tic_path=tic,
                format_override="v0.1.0",
                runtime_override="v0.1.0",
            )
            self.assertEqual(pinned, "v0.1.0")

    def test_kv_norm_uses_vllm_budget_not_nvidia_smi(self) -> None:
        b_non, t_non = 16.65e9, 12.79e9
        b_kv, t_kv = 13.65e9, 17.51e9
        b_smi, t_smi = 30.33e9, 32.05e9
        budget_b = vllm_visible_budget(b_non, b_kv)
        budget_t = vllm_visible_budget(t_non, t_kv)
        kv_vllm = normalized_gpu_gain_pct(
            b_kv, t_kv, baseline_total=budget_b, tic_total=budget_t
        )
        kv_smi = normalized_gpu_gain_pct(
            b_kv, t_kv, baseline_total=b_smi, tic_total=t_smi
        )
        self.assertIsNotNone(kv_vllm)
        self.assertIsNotNone(kv_smi)
        self.assertGreater(kv_vllm, 27.0)
        self.assertLess(kv_smi, 22.0)
        self.assertAlmostEqual(kv_vllm, (17.51 / 13.65 - 1.0) * 100.0, places=1)
        self.assertAlmostEqual(bytes_gb(1e9), "1.00 GB")

    def test_graph_mode_from_enforce_eager_flag(self) -> None:
        self.assertEqual(_graph_mode({"enforce_eager": False}), "graphs")
        self.assertEqual(_graph_mode({"enforce_eager": True}), "eager")

    def test_scale_seqs_from_kv_estimate(self) -> None:
        # ~12.71 GiB KV + ~4 GiB freed weights ≈ 1.31x → 32 scales above baseline
        kv_b = int(12.71 * 1024**3)
        w_b = 14 * 1024**3
        w_t = 10 * 1024**3
        hi = default_scale_hi(32, w_b, w_t)
        self.assertEqual(hi, round(32 * 14 / 10))
        scaled = scale_seqs_from_kv_estimate(32, kv_b, w_b, w_t, hi)
        freed = w_b - w_t
        expected = align_seqs_to_cudagraph_stride(
            max(32, min(hi, round(32 * (kv_b + freed) / kv_b))), hi
        )
        self.assertEqual(scaled, expected)
        self.assertGreater(scaled, 32)
        self.assertEqual(
            scale_seqs_from_kv_estimate(32, kv_b, w_b, w_t, 32), 32
        )

    def test_coded_kv_token_capacity_hot_cold(self) -> None:
        # Asymptotic: s=0.42 → 1/0.58 ≈ 1.724; with W=0 equals raw scale.
        self.assertAlmostEqual(gdx_capacity_scale(0.42), 1.0 / 0.58, places=6)
        self.assertEqual(
            coded_kv_token_capacity(10000, savings_s=0.42, live_window=0),
            round(10000 / 0.58),
        )
        # Hot correction: T_coded = (T_adm - W*s)/(1-s) < T_adm/(1-s).
        raw = coded_kv_token_capacity(10000, savings_s=0.42, live_window=0)
        hot = coded_kv_token_capacity(10000, savings_s=0.42, live_window=512)
        self.assertLess(hot, raw)
        self.assertEqual(
            hot,
            round((10000 - 512 * 0.42) / 0.58),
        )
        # Baseline arm: no GDX.
        self.assertEqual(
            coded_kv_token_capacity(
                47396, savings_s=0.42, live_window=512, apply_gdx=False
            ),
            47396,
        )
        self.assertAlmostEqual(
            gdx_cold_savings_from_bpv(6.603098788094933, 16.0),
            1.0 - 6.603098788094933 / 16.0,
            places=9,
        )

    def test_scale_seqs_from_kv_measured(self) -> None:
        # Remeter sample: 238064 → 265872 ≈ 1.12x → 36, align up to stride-8 (40)
        self.assertEqual(
            scale_seqs_from_kv_measured(32, 238064, 265872, 45), 40
        )
        self.assertEqual(
            scale_seqs_from_kv_measured(32, 238064, 265872, 32), 32
        )
        self.assertEqual(
            scale_seqs_from_kv_measured(32, 100, 50, 45), 32
        )
        # 5090 GDX remeter: 235120 → 285056 ≈ 1.21x → 39, align to 40
        self.assertEqual(
            scale_seqs_from_kv_measured(32, 235120, 285056, 45), 40
        )
        self.assertEqual(align_seqs_to_cudagraph_stride(39, 45), 40)
        self.assertEqual(align_seqs_to_cudagraph_stride(36, 45), 40)
        self.assertEqual(align_seqs_to_cudagraph_stride(40, 45), 40)
        # Cannot grow past scale_hi: leave off-stride (product serve also aligns).
        self.assertEqual(align_seqs_to_cudagraph_stride(39, 39), 39)

    def test_prefill_wave_cliff_under_vendor_2048(self) -> None:
        """Capacity TTFT cliff: fair-share 2048/n < prompt → second prefill wave."""
        vendor = 2048
        # Measured chat-templated length for random input_len=32 on this A/B.
        prompt = 61
        for n, waves in ((32, 1), (33, 1), (34, 2), (40, 2)):
            per_seq = vendor // n
            need = 1 if per_seq >= prompt else 2
            self.assertEqual(need, waves, msg=f"conc={n} per_seq={per_seq}")
        self.assertGreater(40 * prompt, vendor)
        self.assertLessEqual(32 * prompt, vendor)

    def test_one_wave_batched_tokens_capacity_cliff(self) -> None:
        from capacity_plan import one_wave_batched_tokens

        # Metered ~61 tok/prompt → pad=30; conc=32/33 fit vendor 2048.
        self.assertIsNone(one_wave_batched_tokens(32, 32))
        self.assertIsNone(one_wave_batched_tokens(33, 32))
        # conc=34/40 need a second wave; harness raises budget (256-aligned).
        self.assertEqual(one_wave_batched_tokens(34, 32), 2304)
        self.assertEqual(one_wave_batched_tokens(40, 32), 2560)

    def test_harness_clamps_concurrency_to_max_num_seqs(self) -> None:
        """Stale capacity BENCH_MAX_CONCURRENCY=40 must not overload seqs=32."""
        harness = (
            Path(__file__).resolve().parents[1] / "scripts" / "harness_lib.sh"
        )
        text = harness.read_text(encoding="utf-8")
        self.assertIn("cannot exceed max_num_seqs", text)
        self.assertIn("clamp BENCH_MAX_CONCURRENCY", text)

    def test_harness_prompts_match_concurrency_avoids_ttft_tail(self) -> None:
        """40@32 second-wave singles land on fused pf (~189ms); pack prompts=conc."""
        harness = (
            Path(__file__).resolve().parents[1] / "scripts" / "harness_lib.sh"
        )
        text = harness.read_text(encoding="utf-8")
        self.assertIn("PROMPTS_MATCH_CONCURRENCY", text)
        self.assertIn("timed_num_prompts", text)
        self.assertIn('EQUALIZE_INPUT_TOKENS="${EQUALIZE_INPUT_TOKENS:-32}"', text)
        # One-wave: batched tokens + dual compile [2048,N]. Omitting
        # compilation_config fattens to [(1,N)]; [2048]-only still grows N.
        self.assertIn("max_num_batched_tokens", text)
        prep_start = text.index("prepare_tic_bundle()")
        prep = text[prep_start : prep_start + 8000]
        self.assertIn("compile_ranges_endpoints", prep)
        self.assertIn("FULL_AND_PIECEWISE", prep)
        self.assertIn('drop_key(text, "compilation_config")', prep)
        self.assertIn("mb > 2048", prep)
        self.assertIn("[2048,\" + str(mb)", prep)
        # Product serve must not setdefault this knob.
        vllm_py = (
            Path(__file__).resolve().parents[3]
            / "isiro-core"
            / "src"
            / "isiro_runtime"
            / "targets"
            / "vllm.py"
        )
        if not vllm_py.is_file():
            vllm_py = Path("/home/kunle/repos/isiroai/isiro-core/src/isiro_runtime/targets/vllm.py")
        text = vllm_py.read_text(encoding="utf-8")
        self.assertNotIn(
            'setdefault("max_num_batched_tokens"',
            text,
        )
        self.assertNotIn(
            "setdefault('max_num_batched_tokens'",
            text,
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
        self.assertIn("ISIRO_BENCH_SKIP_BASELINE=1; no REF profile", body)
        self.assertIn("SKIP cold start; reused", body)
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
            self.assertIn("## D. Generation (recall-matched KV, input 32 / output 256)", report)
            self.assertNotIn("## D. TTFT", report)
            self.assertNotIn("## E. Prefill TTFT", report)
            self.assertIn("## C. Equal batch", report)
            self.assertNotIn("## D. Capacity", report)
            self.assertNotIn("## 1E. Equal-batch", report)
            self.assertNotIn("capacity benefit not active", report)
            self.assertIn("Output tok/s", report)
            self.assertIn("ITL (ms) p50 / p95 / p99", report)
            self.assertIn("Tooling:", report)
            self.assertIn("- `vllm bench serve` (tok/s, ITL)", report)
            self.assertIn("| CUDA graphs | off (eager) |", report)
            self.assertNotIn(
                "Graph mode: `eager` (product default is eager; graphs is opt-in).",
                report,
            )
            self.assertNotIn("Smaller on-disk weights free", report)
            self.assertNotIn("Experiment:", report)
            self.assertIn("Serve output match", report)
            self.assertNotIn("greedy vLLM", report)
            self.assertIn("`isiro serve … --target vllm`", report)
            self.assertIn("| Metric | Baseline | Isiro |", report)
            self.assertIn("**Isiro:**", report)
            self.assertIn("Baseline", report)
            self.assertNotIn("E2E latency", report)
            self.assertNotIn("Footprint-adjusted tok/s ratio", report)
            self.assertNotIn("matched A/B", report)
            self.assertNotIn("ISIRO_SERVE_CUDA_GRAPHS", report)
            self.assertNotIn("Fairness gate", report)
            self.assertIn("Fairness check", report)

    def test_report_records_product_attention(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            summary = _sample_summary(ttft_tic_slower=True)
            summary["attention"] = {
                "baseline": {"attention_backend": "FLASH_ATTN", "isiro_attn": None},
                "tic": {"attention_backend": "CUSTOM", "isiro_attn": "unset"},
            }
            _write_run(run, summary)
            report = render(run)
            self.assertIn("| Isiro attention | `CUSTOM` (`ISIRO_ATTN` unset) |", report)
            self.assertIn("| Baseline attention | `FLASH_ATTN` |", report)
            self.assertNotIn("## F. Same TIC", report)

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
            self.assertNotIn("† tok/s per Non-KV GB", report)
            self.assertNotIn("† tok/s per weight GB", report)
            self.assertIn("## Correctness", report)
            self.assertIn("## A. GPU memory", report)
            self.assertIn("### Weights", report)
            self.assertIn("## C. Equal batch", report)
            self.assertIn("## D. Capacity", report)
            self.assertIn("### Generation (recall-matched KV, input 32 / output 256)", report)
            self.assertNotIn("## E. Prefill TTFT", report)
            self.assertNotIn("## C. Capacity", report)
            self.assertIn("## Config", report)
            self.assertNotIn("## G. Config", report)
            self.assertNotIn("## G. Config / fairness / reproduce", report)
            self.assertIn("| Concurrent requests |", report)
            self.assertIn(
                "Isiro concurrent requests = round(Baseline × estimated KV",
                report,
            )
            self.assertIn("### Available KV", report)
            self.assertIn("| Available KV |", report)
            self.assertIn("### KV tokens capacity", report)
            self.assertIn("| KV tokens capacity |", report)
            self.assertNotIn("% of Baseline", report)
            self.assertNotIn("### KV cache", report)
            self.assertNotIn("| GPU memory reserved for KV |", report)
            self.assertNotIn("weight-estimate KV-scaled timed point", report)
            self.assertNotIn("TIC concurrency scale", report)
            self.assertNotIn("SKU / `system_id`", report)
            self.assertNotIn("| Capacity scaling |", report)
            self.assertIn("| System |", report)
            self.assertIn("| CUDA graphs |", report)
            self.assertIn("| GPU memory for loaded weights |", report)
            self.assertNotIn("## Serve at recall-matched KV cache", report)
            self.assertNotIn("Not default fill", report)
            self.assertIn("| Compiler |", report)
            self.assertIn("| Runtime |", report)
            self.assertIn("Savings %", report)
            self.assertIn(
                "Savings % uses `1 - Isiro/Baseline` for loaded weights.",
                report,
            )
            self.assertNotIn(
                "For KV it uses `Isiro/Baseline - 1` as a % gain",
                report,
            )
            self.assertNotIn("Savings % uses `1 - TIC/Baseline`.", report)
            self.assertNotIn("GPU memory excluding KV cache", report)
            self.assertNotIn("nvidia-smi process usage", report)
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
            self.assertIn("**+31%**", report)  # batch_ratio 42/32 → +31%
            self.assertNotIn("† tok/s per Non-KV GB", report)
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
        self.assertEqual(
            resolve_system_id("rtx-pro-6000-blackwell", None),
            "rtx-pro-6000-blackwell",
        )
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
        env.pop("ISIRO_VERIFY_REFERENCE", None)
        env.pop("ISIRO_BENCH_EQUAL_BATCH", None)
        env.pop("ISIRO_BENCH_SKIP_NEEDLE", None)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "qwen2.5-7b-instruct",
                "--dry-run",
                "--system-id",
                "rtx-5090",
                "--config",
                str(ROOT / "common.env.example"),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(ROOT.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scripts/run_harness.sh", result.stdout)
        self.assertIn("scripts/run_needle_haystack.sh", result.stdout)
        self.assertIn("needle:", result.stdout)
        self.assertIn("MODEL_SLUG=qwen2.5-7b-instruct", result.stdout)
        self.assertIn("system_id=rtx-5090", result.stdout)
        self.assertIn("ISIRO_VERIFY_REFERENCE=1", result.stdout)
        self.assertIn("ISIRO_BENCH_EQUAL_BATCH=1", result.stdout)
        self.assertIn("graphs:", result.stdout)
        self.assertNotIn("--graph-on", result.stdout)

    def test_launch_dry_meta_skip_baseline_disables_equal_batch(self) -> None:
        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "ISIRO_BENCH_LAUNCH_DRY_META": "1",
            "ISIRO_BENCH_SKIP_BASELINE": "1",
        }
        env.pop("ISIRO_BENCH_EQUAL_BATCH", None)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "qwen2.5-7b-instruct",
                "--dry-run",
                "--system-id",
                "rtx-5090",
                "--config",
                str(ROOT / "common.env.example"),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(ROOT.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ISIRO_BENCH_EQUAL_BATCH=0", result.stdout)
        self.assertNotIn("ISIRO_BENCH_EQUAL_BATCH=1", result.stdout)

    def test_launch_dry_meta_skip_needle_omits_needle_phase(self) -> None:
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
                "--skip-needle",
                "--system-id",
                "rtx-5090",
                "--config",
                str(ROOT / "common.env.example"),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(ROOT.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("needle:", result.stdout)
        self.assertIn("scripts/run_harness.sh", result.stdout)

    def test_launch_rejects_skip_and_needle_only(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "qwen2.5-7b-instruct",
                "--dry-run",
                "--skip-needle",
                "--needle-only",
                "--system-id",
                "rtx-5090",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(ROOT.parent),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pass only one of", result.stderr)

    def test_launch_rejects_unknown_both_graph_flag(self) -> None:
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
            cwd=str(ROOT.parent),
        )
        self.assertNotEqual(result.returncode, 0)

    def test_launch_rejects_graph_on_and_off(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "launch_ab.py"),
                "qwen2.5-7b-instruct",
                "--dry-run",
                "--system-id",
                "rtx-5090",
                "--graph-on",
                "--graph-off",
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
                / f"{summary['system_id']}-report-20260811T120000Z"
                / "report.md"
            )
            write_report(run, dest)
            self.assertTrue(dest.is_file())
            names = {p.name for p in dest.parent.iterdir()}
            self.assertEqual(names, {"report.md"})
            self.assertFalse((dest.parent / "summary.json").exists())
            self.assertFalse((dest.parent / "tic").exists())
            self.assertFalse((dest.parent / "isiro").exists())

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
        self.assertIn("{model}/{system_id}-report-<UTC>/", text)
        self.assertIn("benchmarks/{model}/{system_id}-report-<UTC>/", text)
        self.assertIn("benchmarks/{model}/{system_id}-report/", text)
        self.assertNotIn("benchmarks/scratch/<run-id>/report.md", text)
        self.assertIn("benchmarks/scratch/", text)
        self.assertNotIn("Rename to `report.md`", text)
        self.assertIn("benchmarks/run_ab.sh {model}\n", text)
        self.assertIn("# matches the report (Graph ON default)", text)
        self.assertIn("benchmarks/run_ab.sh {model} --enforce-eager", text)
        self.assertIn("--graph-off is an alias", text)
        self.assertNotIn("run_needle_haystack.sh", text)
        self.assertNotIn("benchmarks/run_ab.sh {model} --both-graph-modes", text)
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
        banned = (
            "ms+E",
            "dense-E",
            "coded-E",
            "Path A",
            private_tree,
            "dest_gib",
            "dest-from-live",
            "continue_ok",
            "leftover",
            "tANS",
            "rANS",
            "signfold",
            "percol",
            "working-set",
        )
        targets = [ROOT / "README.md", SCRIPTS / "harness_lib.sh"]
        targets.extend(sorted(SCRIPTS.glob("*.py")))
        targets.append(SCRIPTS / "run_needle_haystack.sh")
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
            "enable_thinking": False,
            "prompts": [
                {
                    "index": 0,
                    "prompt": "a",
                    "token_ids": [1, 2, 3],
                    "text": "x",
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "prompt": "b",
                    "token_ids": [4, 5],
                    "text": "y",
                    "finish_reason": "stop",
                },
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

    def test_report_single_graph_on_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            summary = _sample_summary(ttft_tic_slower=True, experiment_kind="capacity")
            summary["publish_quality"] = True
            summary["smoke"] = False
            _write_run(run, summary)
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
            report = render(run)
            self.assertIn("Tooling:", report)
            self.assertIn("- `vllm bench serve` (tok/s, ITL)", report)
            self.assertIn("# ISIRO Benchmark Report: `qwen2.5-7b-instruct`", report)
            self.assertIn("`bf16` | `rtx-5090` |", report)
            self.assertRegex(report, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC")
            self.assertNotIn("## Graph ON (CUDA graphs)", report)
            self.assertNotIn("## Graph OFF (eager)", report)
            self.assertNotIn("<details>", report)
            self.assertNotIn("ISIRO_SERVE_CUDA_GRAPHS", report)
            self.assertNotIn("--both-graph-modes", report)
            self.assertEqual(report.count("## A. GPU memory"), 1)
            self.assertEqual(report.count("## C. Equal batch"), 1)
            self.assertEqual(report.count("## D. Capacity"), 1)
            self.assertEqual(report.count("### Generation"), 1)
            self.assertNotIn("## C. Capacity", report)
            self.assertIn("ITL (ms) p50 / p95 / p99", report)
            self.assertNotIn("† tok/s per Non-KV GB", report)
            self.assertNotIn("Reproduce:", report)
            self.assertIn("Weight bit-exactness", report)
            self.assertIn("4/4 prompts, temp=0, token IDs equal", report)
            self.assertIn("| CUDA graphs | on (eager prefill, graph decode) |", report)
            self.assertIn("| KV cache | product KV cache size |", report)

    def test_greedy_chat_payload_disables_thinking(self) -> None:
        payload = greedy_chat_payload(
            model="Qwen/Qwen3.8-27B",
            prompt="ping",
            seed=17,
            max_tokens=32,
        )
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 17)
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertTrue(thinking_controls_present(payload))

    def test_greedy_protocol_rejects_thinking_dump_even_if_ids_match(self) -> None:
        dump = "We need to reply with exactly one word in lowercase: hello. <think>"
        ids = list(range(32))
        capture = {
            "max_tokens": 32,
            "enable_thinking": True,
            "prompts": [
                {
                    "index": 3,
                    "prompt": "hello",
                    "token_ids": ids,
                    "text": dump,
                    "finish_reason": "length",
                }
            ],
        }
        errors = greedy_protocol_errors(capture)
        self.assertTrue(errors)
        matched = compare_captures(capture, capture)
        self.assertTrue(matched["details"][0]["ok"])
        self.assertFalse(matched["protocol_ok"])
        self.assertFalse(matched["serve_output_match_ok"])

    def test_report_compiler_runtime_can_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            summary = _sample_summary()
            summary["isiro_compiler"] = "v0.1.0"
            summary["isiro_format"] = "v0.1.0"
            summary["isiro_runtime"] = "v0.1.1"
            _write_run(run, summary)
            report = render(run)
            self.assertIn(
                "`bf16` | `rtx-5090` | compiler `v0.1.0` | runtime `v0.1.1` |",
                report,
            )
            self.assertIn("| Compiler | `v0.1.0` |", report)
            self.assertIn("| Runtime | `v0.1.1` |", report)

    def test_report_skips_equal_batch_when_redundant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            summary = _sample_summary(ttft_tic_slower=True, experiment_kind="capacity")
            summary["equal_batch"] = build_equal_batch_block(
                max_num_seqs=32,
                max_concurrency=32,
                baseline_tps=1000.0,
                tic_tps=1400.0,
                baseline_latency_ms=profile("baseline", "x", 32, 256)["latency_ms"],
                tic_latency_ms=profile("isiro", "x", 32, 256)["latency_ms"],
            )
            _write_run(run, summary)
            report = render(run)
            self.assertNotIn("## C. Equal batch", report)
            self.assertNotIn("Not equal batch (C).", report)
            self.assertIn("## C. Capacity", report)
            self.assertIn("### Generation", report)

    def test_report_gdx_kv_bitexact_and_lean_intro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            summary = _sample_summary(
                ttft_tic_slower=True, experiment_kind="capacity"
            )
            summary["correctness"]["kv_bit_exact_ok"] = True
            summary["correctness"]["kv_bit_exact_alphabet"] = "bf16"
            # Bitexact bpv → coded capacity fold (no curve needed).
            summary["correctness"]["kv_bit_exact_avg_bpv"] = 6.603098788094933
            summary["correctness"]["kv_bit_exact_fa_bpv"] = 16.0
            summary["footprint"]["baseline"]["model_loading_bytes_reported"] = (
                19 * 1024**3
            )
            summary["footprint"]["tic"]["model_loading_bytes_reported"] = (
                15 * 1024**3
            )
            _write_run(run, summary)
            (run / "commands.json").write_text(
                json.dumps(
                    {
                        "baseline_serve": ["vllm", "serve"],
                        "tic_serve": ["isiro", "serve"],
                        "isiro_gdx_kv": "bf16",
                        "kv_cache_dtype": "bfloat16",
                        "enforce_eager": True,
                    }
                ),
                encoding="utf-8",
            )
            report = render(run)
            self.assertIn("**Isiro:**", report)
            self.assertIn("`ISIRO_GDX=bf16`", report)
            self.assertIn("KV bit-exactness", report)
            self.assertIn("GDX `bf16` cold roundtrip", report)
            self.assertIn("Available KV", report)
            self.assertIn("KV tokens capacity", report)
            self.assertIn("KV vs context", report)
            # Available KV is GiB-only (no parenthetical vendor slots).
            self.assertRegex(
                report,
                r"\| Available KV \| [0-9.]+ GiB \|",
            )
            self.assertNotRegex(
                report,
                r"\| Available KV \| [0-9.]+ GiB \([0-9]+ tokens\) \|",
            )
            self.assertIn("scale 1/(1-s)", report)
            self.assertIn("hot-corrected", report)
            self.assertNotIn("coding-rate", report)
            self.assertNotIn("~29-31%", report)
            self.assertNotIn("ISIRO_GDX_C=fp8", report)
            self.assertNotIn("TIC + KV compress", report)
            self.assertNotIn("ISIRO invent lock", report)
            self.assertNotIn("| Metric | Baseline | TIC |", report)
            # Capacity section no longer duplicates vendor-slot KV tokens row.
            self.assertIn("## D. Capacity", report)
            cap = report.split("## D. Capacity", 1)[1]
            cap_head = cap.split("### Generation", 1)[0]
            self.assertIn("Concurrent requests", cap_head)
            self.assertNotIn("| KV tokens capacity |", cap_head)
            # Coded capacity lives under Section A.
            self.assertIn("### KV tokens capacity", report.split("## D. Capacity")[0])

    def test_fold_gdx_kv_bitexact_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            correctness: dict = {}
            missing = fold_gdx_kv_bitexact(run, correctness, required=True)
            self.assertTrue(missing)
            (run / "gdx_kv_bitexact.json").write_text(
                json.dumps(
                    {
                        "schema": "isiro-benchmark-gdx-kv-bitexact-v1",
                        "alphabet": "bf16",
                        "kv_bit_exact_ok": True,
                        "n_values": 64,
                        "avg_bpv": 6.5,
                        "fa_bpv": 16.0,
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )
            correctness = {}
            self.assertEqual(
                fold_gdx_kv_bitexact(run, correctness, required=True), []
            )
            self.assertEqual(correctness.get("kv_bit_exact_avg_bpv"), 6.5)
            self.assertEqual(correctness.get("kv_bit_exact_fa_bpv"), 16.0)
            self.assertTrue(correctness["kv_bit_exact_ok"])
            self.assertEqual(correctness["kv_bit_exact_alphabet"], "bf16")

    def test_gdx_kv_bitexact_meter_bf16_when_overlay_present(self) -> None:
        site = Path.home() / ".isiro" / "serve-site" / "isiro_runtime"
        if not site.is_dir():
            self.skipTest("product serve-site overlay not installed")
        doc = gdx_kv_run_meter("bf16", seed=11)
        self.assertTrue(doc.get("kv_bit_exact_ok"), msg=doc.get("error"))
        self.assertEqual(doc.get("alphabet"), "bf16")

    def test_fold_serve_output_match_isiro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            capture = {
                "max_tokens": 32,
                "enable_thinking": False,
                "prompts": [
                    {
                        "index": 0,
                        "prompt": "a",
                        "token_ids": [1, 2],
                        "text": "x",
                        "finish_reason": "stop",
                    },
                    {
                        "index": 1,
                        "prompt": "b",
                        "token_ids": [3, 4],
                        "text": "y",
                        "finish_reason": "stop",
                    },
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

    def test_stock_vllm_story_tco_derived_and_empty_report(self) -> None:
        from stock_vllm_story import (
            STORY_SCHEMA,
            derive_tco_point,
            empty_story,
            render_story_markdown,
            validate_story,
        )

        doc = empty_story()
        self.assertEqual(doc["schema"], STORY_SCHEMA)
        self.assertEqual(validate_story(doc), [])
        md = "\n".join(render_story_markdown(doc))
        self.assertIn("no measured points", md)
        self.assertNotIn("90%", md)
        p = derive_tco_point(concurrency=8, output_tok_s=1000.0, gpu_hour_usd=3.6)
        self.assertEqual(p["kind"], "derived")
        self.assertAlmostEqual(p["cost_usd_per_1m_output_tokens"], 1.0)
        from stock_vllm_story import archived_id_bytes, vram_stack_bytes

        self.assertEqual(archived_id_bytes(1_000_000, 1), 4_000_000)
        stack = vram_stack_bytes(
            weight_bytes=11_000_000_000,
            kv_pool_bytes=1_090_000_000,
            archived_id_bytes=archived_id_bytes(1_000_000, 1),
        )
        self.assertEqual(stack["archived_id_bytes"], 4_000_000)
        self.assertEqual(
            stack["stack_bytes"],
            11_000_000_000 + 1_090_000_000 + 4_000_000,
        )
        doc["graph1_vram_vs_context"]["points"] = [
            {
                "context_tokens": 1_000_000,
                "baseline_kv_bytes": 1,
                "tic_weight_bytes": stack["weight_bytes"],
                "tic_kv_pool_bytes": stack["kv_pool_bytes"],
                "tic_archived_id_bytes": stack["archived_id_bytes"],
                "tic_persist_page_bytes": 0,
                "baseline_smi_bytes": 2,
                "tic_smi_bytes": 3,
                "oom_baseline": True,
                "oom_tic": False,
            }
        ]
        md1 = "\n".join(render_story_markdown(doc))
        self.assertIn("T weights", md1)
        self.assertIn("T ids", md1)
        self.assertIn("Graph 1a", md1)
        from stock_vllm_story import (
            graph1_composition_row,
            util_fill_smi_is_failure,
            inactive_kv_bytes,
        )

        self.assertTrue(
            util_fill_smi_is_failure(
                tic_smi_bytes=31_000_000_000,
                dest_lock_ready_ceiling_bytes=28_000 * 1024 * 1024,
            )
        )
        self.assertFalse(
            util_fill_smi_is_failure(
                tic_smi_bytes=14_000_000_000,
                dest_lock_ready_ceiling_bytes=28_000 * 1024 * 1024,
            )
        )
        doc["graph1_composition"]["points"] = [
            graph1_composition_row(
                context_tokens=1_000_000,
                weight_bytes=stack["weight_bytes"],
                kv_pool_bytes=stack["kv_pool_bytes"],
                archived_id_bytes=stack["archived_id_bytes"],
                baseline_kv_bytes=inactive_kv_bytes(1_000_000, 1),
            )
        ]
        md1b = "\n".join(render_story_markdown(doc))
        self.assertIn("Graph 1b", md1b)
        self.assertIn("derived", md1b)
        doc["graph2_toks_vs_concurrency"]["points"] = [
            {"concurrency": 1, "baseline_tok_s": 100.0, "tic_tok_s": 100.0}
        ]
        md2 = "\n".join(render_story_markdown(doc))
        self.assertIn("Graph 2", md2)

    def test_gdx_kv_layer_caps_from_hybrid_config(self) -> None:
        from gdx_kv_bytes_curve import (
            _layer_caps_from_config,
            _layer_fa_nbytes,
            _vendor_eff_t,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "num_hidden_layers": 6,
                            "sliding_window": 1024,
                            "layer_types": [
                                "sliding_attention",
                                "sliding_attention",
                                "full_attention",
                                "sliding_attention",
                                "sliding_attention",
                                "full_attention",
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            caps, meta = _layer_caps_from_config(root)
        self.assertEqual(meta.get("fa_twin_mode"), "vendor_layer_caps")
        self.assertEqual(meta.get("sliding_window"), 1024)
        self.assertEqual(meta.get("n_sliding"), 4)
        self.assertEqual(meta.get("n_full"), 2)
        self.assertEqual(caps, [1024, 1024, None, 1024, 1024, None])
        self.assertEqual(_vendor_eff_t(8192, 1024), 1024)
        self.assertEqual(_vendor_eff_t(512, 1024), 512)
        self.assertEqual(_vendor_eff_t(8192, None), 8192)
        # Hybrid FA twin at T=8192, bf16, H=8, D=256: 0.8125 GiB for
        # Gemma-shaped 40 sliding + 8 full scales down here to 6 layers.
        fa = 0
        for cap in caps:
            fa += _layer_fa_nbytes(
                eff_t=_vendor_eff_t(8192, cap),
                kv_heads=8,
                head_dim=256,
                fa_bpv=16.0,
            )
        # 4*1024 + 2*8192 = 20480 token-layers; *2*8*256*2 bytes
        self.assertEqual(fa, 20480 * 2 * 8 * 256 * 2)

    def test_gdx_kv_layer_caps_full_sequence_without_layer_types(self) -> None:
        from gdx_kv_bytes_curve import _layer_caps_from_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"num_hidden_layers": 3}),
                encoding="utf-8",
            )
            caps, meta = _layer_caps_from_config(root)
        self.assertEqual(meta.get("fa_twin_mode"), "full_sequence")
        self.assertEqual(caps, [None, None, None])

    def test_gdx_kv_bytes_curve_cold_grows_with_t(self) -> None:
        import os
        from pathlib import Path

        import torch

        from gdx_kv_bytes_curve import measure_curve

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for real KV curve")
        model_dir = Path(
            os.environ.get("BASELINE_MODEL_DIR")
            or os.environ.get("ISIRO_BENCH_MODEL_DIR")
            or ""
        )
        if not model_dir.is_dir():
            self.skipTest("BASELINE_MODEL_DIR / ISIRO_BENCH_MODEL_DIR required")
        doc = measure_curve(
            "bf16",
            tokens=[512, 1024, 2048],
            model_dir=model_dir,
            live_window=512,
        )
        if doc.get("error") and "not importable" in str(doc.get("error")):
            self.skipTest("isiro_runtime not importable for GDX curve")
        self.assertTrue(doc.get("ok"), doc.get("error"))
        self.assertEqual(doc.get("source"), "real_model_kv")
        cold = [p for p in doc["points"] if int(p["cold_t"]) > 0]
        self.assertGreaterEqual(len(cold), 2)
        self.assertGreater(
            int(cold[-1]["gdx_cold_nbytes"]),
            2 * int(cold[0]["gdx_cold_nbytes"]),
        )
        self.assertGreater(
            int(cold[-1]["gdx_total_nbytes"]),
            int(cold[0]["gdx_total_nbytes"]),
        )
        self.assertLess(float(cold[-1]["payload_bpv"]), 16.0)
        self.assertGreater(float(cold[-1]["save_vs_fa"]), 0.0)

    def test_gdx_kv_bytes_curve_skips_empty_hybrid_cache_layers(self) -> None:
        """Qwen3.5 GDN / linear_attention leave empty DynamicCache slots."""
        src = (SCRIPTS / "gdx_kv_bytes_curve.py").read_text(encoding="utf-8")
        self.assertIn("Hybrid stacks", src)
        self.assertIn("no FA past_key_values layers with keys/values", src)
        self.assertIn("hf_fa_layers", src)
        self.assertNotIn(
            '"DynamicCache layer missing keys/values"',
            src,
        )

    def test_gdx_kv_bytes_curve_legend_has_no_gdx_and_weights_panel(self) -> None:
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "gdx_kv_bytes_curve.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Isiro GDX", src)
        self.assertNotIn("gaussian", src.lower())
        self.assertNotIn("_bf16_page", src)
        self.assertIn("real_model_kv", src)
        self.assertIn("vendor_layer_caps", src)
        self.assertIn("_layer_caps_from_config", src)
        self.assertIn("Baseline (weights + KV)", src)
        self.assertIn("Isiro (weights + KV)", src)
        self.assertIn("Baseline (KV)", src)
        self.assertIn("Isiro (KV)", src)
        self.assertIn('loc="lower left"', src)
        self.assertIn("KV (GiB)", src)
        self.assertIn("2, 1", src)  # 2x1 subplot layout

    def test_bench_sse_patch_has_no_chunk_strip(self) -> None:
        patch = SCRIPTS / "patches" / "endpoint_request_func.py"
        self.assertTrue(patch.is_file(), patch)
        body = patch.read_text(encoding="utf-8")
        self.assertIn("ISIRO patch", body)
        self.assertNotIn("chunk_bytes.strip()", body)

    def test_run_bench_dry_run_mounts_sse_patch(self) -> None:
        from run_bench import _SSE_PATCH, _VLLM_SSE_IN_IMAGE, bench_command

        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            raw.mkdir()
            serve = Path(temp) / "serve.json"
            serve.write_text("{}", encoding="utf-8")
            out = Path(temp) / "out.json"
            args = __import__("argparse").Namespace(
                variant="isiro",
                profile="ttft-2048",
                run_id="t",
                model="m",
                tokenizer=None,
                api_url="http://127.0.0.1:8000",
                backend="openai-chat",
                endpoint="/chat/completions",
                input_tokens=2048,
                output_tokens=64,
                num_prompts=8,
                num_warmups=0,
                max_concurrency=8,
                seed=17,
                replicates=1,
                serve_config=serve,
                raw_dir=raw,
                out=out,
                vllm_image="vllm/vllm-openai:v0.31.0",
                dry_run=True,
                raw_input=[],
                extra_body='{"chat_template_kwargs":{"enable_thinking":false}}',
            )
            cmd = bench_command(args, raw / "isiro-ttft-2048-r1.json")
            joined = " ".join(cmd)
            self.assertIn(str(_SSE_PATCH.resolve()), joined)
            self.assertIn(_VLLM_SSE_IN_IMAGE, joined)
            self.assertIn("--extra-body", cmd)


class ResolveFp8BaselineQuantizationTests(unittest.TestCase):
    def test_compressed_tensors_from_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_fp8_baseline_quantization(root),
                "compressed-tensors",
            )

    def test_modelopt_from_hf_quant_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hf_quant_config.json").write_text(
                json.dumps({"quantization": {"quant_algo": "FP8"}}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_fp8_baseline_quantization(root), "modelopt")

    def test_native_hub_fp8_quant_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "fmt": "e4m3",
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_fp8_baseline_quantization(root), "fp8")

    def test_plain_bf16_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_fp8_baseline_quantization(root)


class GdxMeterEnvTests(unittest.TestCase):
    def test_requirements_pin_kernels_band(self) -> None:
        text = (ROOT / "requirements-gdx-meters.txt").read_text(encoding="utf-8")
        self.assertIn("kernels>=0.17.0,<0.18.0", text)
        self.assertIn("transformers", text)
        self.assertIn("numpy", text)

    def test_harness_calls_ensure_before_meters(self) -> None:
        src = (SCRIPTS / "harness_lib.sh").read_text(encoding="utf-8")
        self.assertIn("ensure_gdx_meter_env.py", src)
        self.assertLess(
            src.index("gdx_meter_env: ensuring pinned host meter python"),
            src.index("gdx_kv_bitexact: measuring cold roundtrip"),
        )

    def test_preflight_fails_closed_without_kernels(self) -> None:
        from ensure_gdx_meter_env import preflight_python

        # System python3 almost never has the meter stack; must fail closed.
        report = preflight_python(Path(sys.executable), need_cuda=False)
        self.assertFalse(report.ok)
        self.assertTrue(
            any("kernels" in e for e in report.errors),
            report.errors,
        )

    def test_ensure_dry_run_prints_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "bench-venv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "ensure_gdx_meter_env.py"),
                    "--venv",
                    str(venv),
                    "--requirements",
                    str(ROOT / "requirements-gdx-meters.txt"),
                    "--dry-run",
                    "--print-python",
                    "--allow-no-cuda",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bin/python", result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
