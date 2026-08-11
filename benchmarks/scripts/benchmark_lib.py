# SPDX-License-Identifier: Apache-2.0
"""Shared, dependency-free helpers for the public benchmark harness."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

PROFILE_SCHEMA = "isiro-benchmark-profile-v1"
SUMMARY_SCHEMA = "isiro-benchmark-summary-v1"

FAIR_BENCH_KEYS = (
    "backend",
    "endpoint",
    "model_id",
    "dataset_name",
    "num_prompts",
    "num_warmups",
    "request_rate",
    "max_concurrency",
    "burstiness",
    "seed",
)
# Capacity mode allows per-side concurrency (the win axis).
FAIR_BENCH_KEYS_CAPACITY = tuple(
    key for key in FAIR_BENCH_KEYS if key != "max_concurrency"
)
FAIR_SERVE_KEYS = (
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "max_num_seqs",
    "prefix_caching",
    "enforce_eager",
    "trust_remote_code",
    "flashinfer_autotune",
)
FAIR_SERVE_KEYS_CAPACITY = tuple(
    key for key in FAIR_SERVE_KEYS if key != "max_num_seqs"
)
CAPACITY_SCHEMA = "isiro-benchmark-capacity-v1"
EQUAL_BATCH_SCHEMA = "isiro-benchmark-equal-batch-v1"
DENSE_BPV = 16.0
# Substrate (docker vs host_isiro) is recorded and labeled, not force-equal.
# Image digest is only meaningful for the docker side.
FAIR_ENV_KEYS = (
    "gpu_model",
    "gpu_count",
    "driver_version",
    "cuda_version",
    "vllm_version",
)


def load_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile_value(
    raw: dict[str, Any],
    metric: str,
    percentile: int,
) -> float | None:
    if percentile == 50:
        median_key = f"median_{metric}_ms"
        if raw.get(median_key) is not None:
            return round(float(raw[median_key]), 3)
    for item in raw.get(f"percentiles_{metric}_ms") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            if float(item[0]) == float(percentile):
                return round(float(item[1]), 3)
    fallback = raw.get(f"p{percentile}_{metric}_ms")
    return None if fallback is None else round(float(fallback), 3)


def normalize_vllm_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a vLLM result while requiring genuine p50/p95/p99 values."""
    latency: dict[str, float | None] = {}
    for metric in ("ttft", "itl", "tpot", "e2el"):
        output_name = "e2e" if metric == "e2el" else metric
        for percentile in (50, 95, 99):
            value = _percentile_value(raw, metric, percentile)
            if metric != "e2el" and value is None:
                raise ValueError(
                    f"vLLM result is missing {metric} p{percentile}; "
                    "run with --metric-percentiles 50,95,99"
                )
            latency[f"{output_name}_p{percentile}"] = value
    return {
        "latency_ms": latency,
        "throughput": {
            "output_tokens_per_sec": round(
                float(raw.get("output_throughput") or 0.0), 4
            ),
            "requests_per_sec": round(
                float(raw.get("request_throughput") or 0.0), 4
            ),
            "total_output_tokens": int(
                raw.get("total_output")
                or raw.get("total_output_tokens")
                or 0
            ),
        },
        "load_errors": int(raw.get("failed") or 0),
        "raw_summary": {
            "completed": int(raw.get("completed") or 0),
            "failed": int(raw.get("failed") or 0),
            "duration_sec": round(float(raw.get("duration") or 0.0), 3),
            "total_input_tokens": int(
                raw.get("total_input") or raw.get("total_input_tokens") or 0
            ),
            "total_output_tokens": int(
                raw.get("total_output")
                or raw.get("total_output_tokens")
                or 0
            ),
        },
    }


def _median(values: Iterable[float], digits: int) -> float:
    return round(float(statistics.median(values)), digits)


def aggregate_replicates(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("at least one replicate is required")
    latency_keys = tuple(items[0]["latency_ms"])
    throughput_keys = ("output_tokens_per_sec", "requests_per_sec")
    latency = {
        key: (
            None
            if all(item["latency_ms"].get(key) is None for item in items)
            else _median(
                [
                    float(item["latency_ms"][key])
                    for item in items
                    if item["latency_ms"].get(key) is not None
                ],
                3,
            )
        )
        for key in latency_keys
    }
    throughput = {
        key: _median(
            [float(item["throughput"].get(key) or 0.0) for item in items],
            4,
        )
        for key in throughput_keys
    }
    throughput["total_output_tokens"] = int(
        statistics.median(
            [int(item["throughput"].get("total_output_tokens") or 0) for item in items]
        )
    )
    return {
        "latency_ms": latency,
        "throughput": throughput,
        "load_errors": max(int(item.get("load_errors") or 0) for item in items),
    }


def fairness_mismatches(
    baseline: dict[str, Any],
    tic: dict[str, Any],
    baseline_environment: dict[str, Any],
    tic_environment: dict[str, Any],
    experiment_kind: str = "equal-batch",
) -> list[str]:
    """Compare paired profiles. Capacity mode allows differing concurrency."""
    mismatches: list[str] = []
    if experiment_kind not in ("equal-batch", "capacity"):
        mismatches.append(
            f"experiment_kind must be equal-batch or capacity, got {experiment_kind!r}"
        )
        experiment_kind = "equal-batch"
    bench_keys = (
        FAIR_BENCH_KEYS_CAPACITY
        if experiment_kind == "capacity"
        else FAIR_BENCH_KEYS
    )
    serve_keys = (
        FAIR_SERVE_KEYS_CAPACITY
        if experiment_kind == "capacity"
        else FAIR_SERVE_KEYS
    )
    for block, keys in (
        ("workload", ("input_tokens", "output_tokens")),
        ("bench_config", bench_keys),
        ("serve_config", serve_keys),
    ):
        for key in keys:
            left = baseline.get(block, {}).get(key)
            right = tic.get(block, {}).get(key)
            if left != right:
                mismatches.append(
                    f"{block}.{key}: baseline={left!r}, tic={right!r}"
                )
    for key in FAIR_ENV_KEYS:
        left = baseline_environment.get(key)
        right = tic_environment.get(key)
        if left != right:
            mismatches.append(
                f"environment.{key}: baseline={left!r}, tic={right!r}"
            )
    return mismatches


def effective_bpv(baseline_bytes: int | float, tic_bytes: int | float) -> float:
    """Effective compressed bpv vs BF16 dense (16 bpv) from measured weight bytes."""
    baseline = float(baseline_bytes)
    tic = float(tic_bytes)
    if baseline <= 0:
        raise ValueError("baseline_bytes must be positive")
    if tic <= 0:
        raise ValueError("tic_bytes must be positive")
    return DENSE_BPV * (tic / baseline)


def weights_and_non_kv_bytes(
    process_bytes: int | float | None,
    kv_bytes: int | float | None,
) -> int | None:
    """Audit-only: nvidia-smi process bytes minus vLLM Available KV."""
    if process_bytes is None or kv_bytes is None:
        return None
    process = int(process_bytes)
    kv = int(kv_bytes)
    if process < 0 or kv < 0:
        raise ValueError("process_bytes and kv_bytes must be non-negative")
    return process - kv


def vllm_requested_memory_bytes(
    total_bytes: int | float | None,
    gpu_memory_utilization: float | None,
) -> int | None:
    """Match vLLM ``request_memory``: ceil(total * util)."""
    if total_bytes is None or gpu_memory_utilization is None:
        return None
    total = int(total_bytes)
    util = float(gpu_memory_utilization)
    if total < 0 or util < 0:
        raise ValueError("total_bytes and gpu_memory_utilization must be non-negative")
    return math.ceil(total * util)


def vllm_non_kv_cache_bytes(
    *,
    non_kv_reported: int | float | None = None,
    requested_bytes: int | float | None = None,
    available_kv_bytes: int | float | None = None,
) -> int | None:
    """vLLM ``MemoryProfilingResult.non_kv_cache_memory``.

    Prefer the DEBUG scrape (``Total non KV cache memory``). Otherwise
    invert Available KV: requested - available_kv (cudagraph estimate
    applied is 0 unless ``VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS``).
    """
    if non_kv_reported is not None:
        value = int(non_kv_reported)
        if value < 0:
            raise ValueError("non_kv_reported must be non-negative")
        return value
    if requested_bytes is None or available_kv_bytes is None:
        return None
    requested = int(requested_bytes)
    kv = int(available_kv_bytes)
    if requested < 0 or kv < 0:
        raise ValueError("requested_bytes and available_kv_bytes must be non-negative")
    return requested - kv


def validate_gpu_memory_split(
    *,
    kv_bytes: int | float | None,
    non_kv_bytes: int | float | None,
    requested_bytes: int | float | None = None,
    process_bytes: int | float | None = None,
    epsilon_bytes: int = 64 * 1024 * 1024,
) -> list[str]:
    """Fail closed when vLLM requested (or legacy process) disagrees with non-KV + KV."""
    errors: list[str] = []
    if kv_bytes is None or non_kv_bytes is None:
        errors.append("missing GPU memory split fields (kv, non-KV)")
        return errors
    kv = int(kv_bytes)
    non_kv = int(non_kv_bytes)
    if requested_bytes is not None:
        requested = int(requested_bytes)
        if abs((non_kv + kv) - requested) > int(epsilon_bytes):
            errors.append(
                "vLLM requested memory must equal non_kv_cache + Available KV "
                f"(requested={requested}, non_kv={non_kv}, kv={kv})"
            )
        return errors
    if process_bytes is None:
        errors.append("missing GPU memory split fields (requested or process, kv, non-KV)")
        return errors
    process = int(process_bytes)
    if abs((non_kv + kv) - process) > int(epsilon_bytes):
        errors.append(
            "GPU process after load must equal weights_and_non_kv + KV "
            f"(process={process}, non_kv={non_kv}, kv={kv})"
        )
    return errors


def validate_capacity_memory_story(summary: dict[str, Any]) -> list[str]:
    """Publish gates for GPU memory split (§B).

    Prefer vLLM identity: requested ≈ non-KV + Available KV. Capacity
    publish requires a matched-seqs KV probe and a hard §B win: TIC KV
    GB (and tokens) strictly above baseline, and TIC non-KV strictly
    below baseline. Timed asymmetric reads alone are not enough to
    publish (pf can look like a weight win while losing measured KV).
    """
    errors: list[str] = []
    footprint = summary.get("footprint") or {}
    kv = summary.get("kv_cache") or {}
    for side, process_key, kv_key, non_kv_key in (
        (
            "baseline",
            "gpu_process_memory_after_load_bytes_measured",
            "baseline_memory_bytes_measured",
            "weights_and_non_kv_gpu_memory_bytes_derived",
        ),
        (
            "tic",
            "gpu_process_memory_for_non_kv_bytes_measured",
            "tic_memory_bytes_measured",
            "weights_and_non_kv_gpu_memory_bytes_derived",
        ),
    ):
        side_fp = footprint.get(side) or {}
        process_bytes = side_fp.get(process_key)
        if side == "tic" and process_bytes is None:
            process_bytes = side_fp.get(
                "gpu_process_memory_after_ready_bytes_measured"
            ) or side_fp.get("gpu_process_memory_after_load_bytes_measured")
        requested_bytes = side_fp.get(
            "gpu_requested_memory_bytes_reported"
        ) or side_fp.get("gpu_requested_memory_bytes_derived")
        errors.extend(
            f"{side}: {item}"
            for item in validate_gpu_memory_split(
                process_bytes=process_bytes,
                kv_bytes=kv.get(kv_key),
                non_kv_bytes=side_fp.get(non_kv_key),
                requested_bytes=requested_bytes,
            )
        )
    if str(summary.get("experiment_kind") or "") != "capacity":
        return errors
    capacity = summary.get("capacity") or {}
    kv_source = str(capacity.get("kv_measured_source") or "")
    if kv_source != "matched_probe":
        errors.append(
            "capacity publish requires capacity.kv_measured_source="
            "matched_probe (run with ISIRO_BENCH_KV_MEASURED=1; "
            f"got {kv_source!r})"
        )
        return errors
    b_non = (footprint.get("baseline") or {}).get(
        "weights_and_non_kv_gpu_memory_bytes_derived"
    )
    t_non = (footprint.get("tic") or {}).get(
        "weights_and_non_kv_gpu_memory_bytes_derived"
    )
    b_kv = kv.get("baseline_memory_bytes_measured")
    t_kv = kv.get("tic_memory_bytes_measured")
    b_tok = kv.get("baseline_tokens_measured")
    t_tok = kv.get("tic_tokens_measured")
    if (
        isinstance(b_non, (int, float))
        and isinstance(t_non, (int, float))
        and float(t_non) >= float(b_non)
    ):
        errors.append(
            "§B fail: TIC non-KV GPU memory must be strictly below baseline "
            f"(got tic={t_non} baseline={b_non})"
        )
    if (
        isinstance(b_kv, (int, float))
        and isinstance(t_kv, (int, float))
        and float(t_kv) <= float(b_kv)
    ):
        errors.append(
            "§B fail: TIC KV memory must be strictly above baseline "
            f"(got tic={t_kv} baseline={b_kv})"
        )
    if (
        isinstance(b_tok, (int, float))
        and isinstance(t_tok, (int, float))
        and float(t_tok) <= float(b_tok)
    ):
        errors.append(
            "§B fail: TIC KV tokens must be strictly above baseline "
            f"(got tic={t_tok} baseline={b_tok})"
        )
    return errors


def weight_capacity_headroom(measured_bpv: float) -> float:
    """Dense BF16 weight bytes / TIC weight bytes = 16 / measured_bpv.

    Derived capacity headroom from measured on-disk compression. Not a measured
    max_num_seqs: CUDA graphs and allocator effects can diverge from this ratio.
    """
    if measured_bpv <= 0:
        raise ValueError("measured_bpv must be positive")
    return DENSE_BPV / float(measured_bpv)


def kv_token_capacity_ratio(baseline_tokens: float, tic_tokens: float) -> float:
    """TIC / baseline KV tokens from vLLM startup accounting."""
    baseline = float(baseline_tokens)
    tic = float(tic_tokens)
    if baseline <= 0:
        raise ValueError("baseline_tokens must be positive")
    if tic < 0:
        raise ValueError("tic_tokens must be non-negative")
    return tic / baseline


def validate_capacity(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if doc.get("schema") != CAPACITY_SCHEMA:
        errors.append(f"schema must be {CAPACITY_SCHEMA}")
    if doc.get("experiment_kind") != "capacity":
        errors.append("experiment_kind must be capacity")
    for key in (
        "gpu_memory_utilization",
        "max_model_len",
        "baseline",
        "tic",
        "search_lo",
        "search_hi",
    ):
        if key not in doc:
            errors.append(f"missing {key}")
    for side in ("baseline", "tic"):
        block = doc.get(side) or {}
        for key in ("max_num_seqs", "max_concurrency"):
            value = block.get(key)
            if not isinstance(value, int) or value < 1:
                errors.append(f"{side}.{key} must be a positive int")
    return errors


def validate_equal_batch(doc: dict[str, Any]) -> list[str]:
    """Validate bundled equal-batch transparency block (capacity publish)."""
    errors: list[str] = []
    if doc.get("schema") != EQUAL_BATCH_SCHEMA:
        errors.append(f"schema must be {EQUAL_BATCH_SCHEMA}")
    if doc.get("label") != "capacity benefit not active":
        errors.append("label must be 'capacity benefit not active'")
    for key in ("max_num_seqs", "max_concurrency"):
        value = doc.get(key)
        if not isinstance(value, int) or value < 1:
            errors.append(f"{key} must be a positive int")
    tps = doc.get("decode_output_tokens_per_sec") or {}
    for side in ("baseline", "tic"):
        value = tps.get(side)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            errors.append(f"decode_output_tokens_per_sec.{side} must be > 0")
    if doc.get("throughput_ratio_derived") is None:
        errors.append("missing throughput_ratio_derived")
    return errors


def validate_profile(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "run_id",
        "variant",
        "profile",
        "recorded_at",
        "workload",
        "serve_config",
        "bench_config",
        "latency_ms",
        "throughput",
        "replicates",
    )
    if doc.get("schema") != PROFILE_SCHEMA:
        errors.append(f"schema must be {PROFILE_SCHEMA}")
    errors.extend(f"missing {key}" for key in required if key not in doc)
    if doc.get("variant") not in ("baseline", "isiro"):
        errors.append("variant must be baseline or isiro")
    for metric in ("ttft", "itl", "tpot"):
        for percentile in (50, 95, 99):
            if doc.get("latency_ms", {}).get(f"{metric}_p{percentile}") is None:
                errors.append(f"missing latency_ms.{metric}_p{percentile}")
    if int(doc.get("load_errors") or 0) != 0:
        errors.append("load_errors must be zero")
    return errors


def validate_summary(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "run_id",
        "recorded_at",
        "model",
        "precision",
        "system_id",
        "isiro_format",
        "fairness",
        "footprint",
        "correctness",
        "profiles",
    )
    if doc.get("schema") != SUMMARY_SCHEMA:
        errors.append(f"schema must be {SUMMARY_SCHEMA}")
    errors.extend(f"missing {key}" for key in required if key not in doc)
    correctness = doc.get("correctness") or {}
    digest = correctness.get("tic_sha256", "")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        errors.append("correctness.tic_sha256 must be lowercase SHA-256")
    verify_mode = correctness.get("verify_mode")
    if verify_mode not in ("integrity", "reference"):
        # Legacy summaries used bit_exact_ok alone (reference verify).
        if correctness.get("bit_exact_ok") is True and "integrity_ok" not in correctness:
            verify_mode = "reference"
        else:
            errors.append(
                "correctness.verify_mode must be 'integrity' or 'reference'"
            )
    integrity_ok = correctness.get("integrity_ok")
    if integrity_ok is None and correctness.get("bit_exact_ok") is True:
        integrity_ok = True
    if integrity_ok is not True:
        errors.append("correctness.integrity_ok must be true")
    if verify_mode == "reference" and correctness.get("bit_exact_ok") is not True:
        errors.append(
            "correctness.bit_exact_ok must be true when verify_mode is reference"
        )
    if not doc.get("fairness", {}).get("passed"):
        errors.append("fairness.passed must be true")
    if not doc.get("profiles"):
        errors.append("profiles must not be empty")
    return errors
