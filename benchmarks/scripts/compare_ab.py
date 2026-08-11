#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate baseline/TIC comparability and build the paired summary JSON."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_lib import (
    EQUAL_BATCH_SCHEMA,
    SUMMARY_SCHEMA,
    fairness_mismatches,
    load_json,
    validate_capacity,
    validate_equal_batch,
    validate_profile,
    kv_token_capacity_ratio,
    vllm_non_kv_cache_bytes,
    weights_and_non_kv_bytes,
    write_json,
)
from capacity_plan import scale_seqs_from_kv_measured
from serve_output_match import compare_captures


def fold_serve_output_match(
    run_dir: Path, correctness: dict[str, Any]
) -> None:
    """Fold greedy serve captures into summary correctness (baseline + isiro)."""
    baseline_match = run_dir / "output_match_baseline.json"
    isiro_match = run_dir / "output_match_isiro.json"
    if baseline_match.is_file() and isiro_match.is_file():
        match_doc = compare_captures(load_json(baseline_match), load_json(isiro_match))
        write_json(run_dir / "output_match.json", match_doc)
        correctness["serve_output_match_ok"] = bool(
            match_doc.get("serve_output_match_ok")
        )
        correctness["serve_output_match_matched"] = int(match_doc.get("matched") or 0)
        correctness["serve_output_match_prompt_count"] = int(
            match_doc.get("prompt_count") or 0
        )
        return
    legacy = run_dir / "output_match.json"
    if legacy.is_file():
        match_doc = load_json(legacy)
        correctness["serve_output_match_ok"] = bool(
            match_doc.get("serve_output_match_ok")
        )
        correctness["serve_output_match_matched"] = int(match_doc.get("matched") or 0)
        correctness["serve_output_match_prompt_count"] = int(
            match_doc.get("prompt_count") or 0
        )


def artifact_digest(path: Path) -> str:
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(child.stat().st_size.to_bytes(8, "big"))
        with child.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def weight_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    weight_files = sorted(path.rglob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"no .safetensors weights found under {path}")
    return sum(item.stat().st_size for item in weight_files)


def profile_map(directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        doc = load_json(path)
        if doc.get("schema") != "isiro-benchmark-profile-v1":
            continue
        result[str(doc["profile"])] = (path, doc)
    return result


def percent_savings(baseline: int | None, tic: int | None) -> float | None:
    if baseline is None or tic is None or baseline <= 0:
        return None
    return round((1.0 - tic / baseline) * 100.0, 3)


def _decode_tps_pair(
    baseline_doc: dict[str, Any], tic_doc: dict[str, Any]
) -> tuple[float, float]:
    return (
        float(baseline_doc["throughput"]["output_tokens_per_sec"]),
        float(tic_doc["throughput"]["output_tokens_per_sec"]),
    )


def build_equal_batch_block(
    *,
    max_num_seqs: int,
    max_concurrency: int,
    baseline_tps: float,
    tic_tps: float,
    baseline_latency_ms: dict[str, Any] | None = None,
    tic_latency_ms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_ratio = round(tic_tps / baseline_tps, 4) if baseline_tps > 0 else None
    block: dict[str, Any] = {
        "schema": EQUAL_BATCH_SCHEMA,
        "label": "capacity benefit not active",
        "max_num_seqs": max_num_seqs,
        "max_concurrency": max_concurrency,
        "decode_output_tokens_per_sec": {
            "baseline": baseline_tps,
            "tic": tic_tps,
        },
        "throughput_ratio_derived": raw_ratio,
    }
    if baseline_latency_ms is not None and tic_latency_ms is not None:
        block["latency_ms"] = {
            "baseline": baseline_latency_ms,
            "tic": tic_latency_ms,
        }
    return block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--tic-artifact", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--system-id", default="rtx-5090")
    parser.add_argument("--isiro-format", default="v0.1.0")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--experiment-kind",
        choices=("equal-batch", "capacity"),
        default=None,
        help="Override experiment kind (default: capacity.json or equal-batch).",
    )
    args = parser.parse_args()

    capacity_path = args.run_dir / "capacity.json"
    capacity_doc: dict[str, Any] | None = None
    if capacity_path.is_file():
        capacity_doc = load_json(capacity_path)
    experiment_kind = args.experiment_kind
    if experiment_kind is None:
        if capacity_doc is not None:
            experiment_kind = "capacity"
        else:
            experiment_kind = "equal-batch"
    baseline_profiles = profile_map(args.run_dir / "baseline")
    tic_profiles = profile_map(args.run_dir / "isiro")
    errors: list[str] = []
    if set(baseline_profiles) != set(tic_profiles):
        errors.append(
            f"profile sets differ: baseline={sorted(baseline_profiles)}, "
            f"tic={sorted(tic_profiles)}"
        )
    if experiment_kind == "capacity":
        if capacity_doc is None:
            errors.append("capacity mode requires capacity.json")
        else:
            errors.extend(
                f"capacity: {item}" for item in validate_capacity(capacity_doc)
            )
    baseline_env = load_json(args.run_dir / "baseline_environment.json")
    tic_env = load_json(args.run_dir / "isiro_environment.json")
    profile_rows: list[dict[str, Any]] = []
    for name in sorted(set(baseline_profiles) & set(tic_profiles)):
        baseline_path, baseline = baseline_profiles[name]
        tic_path, tic = tic_profiles[name]
        errors.extend(
            f"{name} baseline: {item}" for item in validate_profile(baseline)
        )
        errors.extend(f"{name} tic: {item}" for item in validate_profile(tic))
        errors.extend(
            f"{name}: {item}"
            for item in fairness_mismatches(
                baseline,
                tic,
                baseline_env,
                tic_env,
                experiment_kind=experiment_kind,
            )
        )
        profile_rows.append(
            {
                "profile": name,
                "input_tokens": baseline["workload"]["input_tokens"],
                "output_tokens": baseline["workload"]["output_tokens"],
                "baseline_ref": str(baseline_path.relative_to(args.run_dir)),
                "tic_ref": str(tic_path.relative_to(args.run_dir)),
                "baseline": {
                    "latency_ms": baseline["latency_ms"],
                    "throughput": baseline["throughput"],
                },
                "tic": {
                    "latency_ms": tic["latency_ms"],
                    "throughput": tic["throughput"],
                },
            }
        )

    verify = load_json(args.run_dir / "verify.json")
    baseline_bytes = weight_size(args.baseline_model)
    tic_bytes = weight_size(args.tic_artifact)
    baseline_memory = baseline_env.get(
        "gpu_process_memory_after_load_bytes_measured"
    )
    tic_memory = tic_env.get("gpu_process_memory_after_load_bytes_measured")
    baseline_loading = baseline_env.get("model_loading_bytes_reported")
    tic_loading = tic_env.get("model_loading_bytes_reported")
    publish_quality = all(
        doc.get("publish_quality", False)
        for _, doc in [*baseline_profiles.values(), *tic_profiles.values()]
    )
    smoke = any(
        doc.get("smoke", False)
        for _, doc in [*baseline_profiles.values(), *tic_profiles.values()]
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "run_id": args.run_dir.name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "precision": args.precision,
        "system_id": args.system_id,
        "isiro_format": args.isiro_format,
        "experiment_kind": experiment_kind,
        "publish_quality": publish_quality,
        "smoke": smoke,
        "fairness": {"passed": not errors, "mismatches": errors},
        "footprint": {
            "baseline": {
                "on_disk_bytes_measured": baseline_bytes,
                "gpu_process_memory_after_load_bytes_measured": baseline_memory,
                "model_loading_bytes_reported": baseline_loading,
            },
            "tic": {
                "on_disk_bytes_measured": tic_bytes,
                "gpu_process_memory_after_load_bytes_measured": tic_memory,
                "model_loading_bytes_reported": tic_loading,
                "gpu_process_memory_after_ready_bytes_measured": tic_env.get(
                    "gpu_process_memory_after_ready_bytes_measured"
                ),
            },
            "on_disk_savings_pct_derived": percent_savings(
                baseline_bytes, tic_bytes
            ),
        },
        "correctness": {
            "integrity_ok": bool(
                verify.get(
                    "integrity_ok",
                    verify.get("bit_exact_ok"),
                )
            ),
            "verify_mode": str(verify.get("verify_mode") or "integrity"),
            "tic_sha256": artifact_digest(args.tic_artifact),
            "artifact_kind": (
                "directory" if args.tic_artifact.is_dir() else "file"
            ),
            "verify_command": verify.get("command", []),
        },
        "kv_cache": {
            "baseline_memory_bytes_measured": baseline_env.get(
                "kv_cache_memory_bytes_measured"
            ),
            "tic_memory_bytes_measured": tic_env.get(
                "kv_cache_memory_bytes_measured"
            ),
            "baseline_tokens_measured": baseline_env.get(
                "kv_cache_tokens_measured"
            ),
            "tic_tokens_measured": tic_env.get("kv_cache_tokens_measured"),
        },
        "profiles": profile_rows,
        "environment_ref": "environment.json",
        "commands_ref": "commands.json",
    }
    def _footprint_non_kv(
        *,
        env: dict[str, Any],
        side_fp: dict[str, Any],
        process_bytes: int | float | None,
        kv_bytes: int | float | None,
    ) -> None:
        """§B non-KV = vLLM non_kv_cache_memory (not process − KV)."""
        requested = env.get("gpu_requested_memory_bytes_reported")
        if env.get("non_kv_cache_memory_bytes_derived") is not None and requested is None:
            # Capture already inverted from vLLM-logged requested − Available KV.
            requested = (
                int(env["non_kv_cache_memory_bytes_derived"]) + int(kv_bytes)
                if kv_bytes is not None
                else None
            )
        if requested is not None:
            side_fp["gpu_requested_memory_bytes_reported"] = int(requested)
            side_fp["gpu_requested_memory_bytes_derived"] = int(requested)
        reported = env.get("non_kv_cache_memory_bytes_reported")
        if reported is None:
            reported = env.get("non_kv_cache_memory_bytes_derived")
        if reported is not None:
            side_fp["non_kv_cache_memory_bytes_reported"] = int(reported)
        non_kv = vllm_non_kv_cache_bytes(
            non_kv_reported=reported,
            requested_bytes=requested,
            available_kv_bytes=kv_bytes,
        )
        side_fp["weights_and_non_kv_gpu_memory_bytes_derived"] = non_kv
        # Audit residual only; do not use for §B gates.
        side_fp["weights_and_non_kv_from_process_bytes_derived"] = (
            weights_and_non_kv_bytes(process_bytes, kv_bytes)
        )

    _footprint_non_kv(
        env=baseline_env,
        side_fp=summary["footprint"]["baseline"],
        process_bytes=baseline_memory,
        kv_bytes=summary["kv_cache"].get("baseline_memory_bytes_measured"),
    )
    tic_process_for_non_kv = (
        tic_env.get("gpu_process_memory_after_ready_bytes_measured")
        or tic_memory
    )
    summary["footprint"]["tic"][
        "gpu_process_memory_for_non_kv_bytes_measured"
    ] = tic_process_for_non_kv
    _footprint_non_kv(
        env=tic_env,
        side_fp=summary["footprint"]["tic"],
        process_bytes=tic_process_for_non_kv,
        kv_bytes=summary["kv_cache"].get("tic_memory_bytes_measured"),
    )
    if capacity_doc is not None:
        summary["capacity"] = capacity_doc
        # vLLM-reported KV capacity at startup (same accounting that gates serve).
        kv_base = summary["kv_cache"].get("baseline_tokens_measured")
        kv_tic = summary["kv_cache"].get("tic_tokens_measured")
        if (
            isinstance(kv_base, (int, float))
            and isinstance(kv_tic, (int, float))
            and float(kv_base) > 0
        ):
            summary["capacity"]["kv_token_capacity_ratio_measured"] = round(
                kv_token_capacity_ratio(kv_base, kv_tic), 4
            )
            # Default path: imply measured-KV concurrency from timed-serve KV
            # (asymmetric max_num_seqs). Opt-in matched_probe already set these.
            if (
                str(summary["capacity"].get("seqs_scale_mode") or "")
                == "kv_estimate"
                and summary["capacity"].get("seqs_implied_kv_measured") is None
            ):
                b_seqs = (summary["capacity"].get("baseline") or {}).get(
                    "max_num_seqs"
                )
                scale_hi = summary["capacity"].get("search_hi")
                if (
                    isinstance(b_seqs, int)
                    and b_seqs >= 1
                    and isinstance(scale_hi, int)
                    and scale_hi >= b_seqs
                ):
                    implied = scale_seqs_from_kv_measured(
                        b_seqs, kv_base, kv_tic, scale_hi
                    )
                    summary["capacity"]["seqs_implied_kv_measured"] = implied
                    summary["capacity"]["kv_scale_ratio_measured"] = round(
                        float(kv_tic) / float(kv_base), 4
                    )
                    summary["capacity"]["kv_measured_source"] = "timed_asymmetric"
        kv_mem_base = summary["kv_cache"].get("baseline_memory_bytes_measured")
        kv_mem_tic = summary["kv_cache"].get("tic_memory_bytes_measured")
        if (
            isinstance(kv_mem_base, (int, float))
            and isinstance(kv_mem_tic, (int, float))
            and float(kv_mem_base) > 0
        ):
            summary["capacity"]["kv_cache_memory_ratio_measured"] = round(
                float(kv_mem_tic) / float(kv_mem_base), 4
            )
        decode = next(
            (
                row
                for row in profile_rows
                if int(row["output_tokens"]) > 64
            ),
            None,
        )
        if decode is not None:
            b_tps = float(
                decode["baseline"]["throughput"]["output_tokens_per_sec"]
            )
            t_tps = float(decode["tic"]["throughput"]["output_tokens_per_sec"])
            summary["capacity"]["decode_output_tokens_per_sec"] = {
                "baseline": b_tps,
                "tic": t_tps,
            }
            if b_tps > 0:
                summary["capacity"]["throughput_ratio_derived"] = round(
                    t_tps / b_tps, 4
                )
            b_non = summary["footprint"]["baseline"].get(
                "weights_and_non_kv_gpu_memory_bytes_derived"
            )
            t_non = summary["footprint"]["tic"].get(
                "weights_and_non_kv_gpu_memory_bytes_derived"
            )
            if (
                isinstance(b_non, (int, float))
                and isinstance(t_non, (int, float))
                and float(b_non) > 0
                and float(t_non) > 0
            ):
                gb = 1e9
                summary["capacity"][
                    "efficiency_tok_per_s_per_non_kv_gb_derived"
                ] = {
                    "baseline": round(b_tps / (float(b_non) / gb), 4),
                    "tic": round(t_tps / (float(t_non) / gb), 4),
                }

    equal_batch_meta_path = args.run_dir / "equal_batch.json"
    equal_batch_root = args.run_dir / "equal_batch"
    if equal_batch_root.is_dir():
        eb_baseline = profile_map(equal_batch_root / "baseline")
        eb_tic = profile_map(equal_batch_root / "isiro")
        if set(eb_baseline) != set(eb_tic):
            errors.append(
                "equal_batch profile sets differ: "
                f"baseline={sorted(eb_baseline)}, isiro={sorted(eb_tic)}"
            )
        elif "generation-32-256" not in eb_baseline:
            errors.append("equal_batch missing generation-32-256 profiles")
        else:
            b_path, b_doc = eb_baseline["generation-32-256"]
            t_path, t_doc = eb_tic["generation-32-256"]
            errors.extend(
                f"equal_batch baseline: {item}"
                for item in validate_profile(b_doc)
            )
            errors.extend(
                f"equal_batch tic: {item}" for item in validate_profile(t_doc)
            )
            errors.extend(
                f"equal_batch: {item}"
                for item in fairness_mismatches(
                    b_doc,
                    t_doc,
                    baseline_env,
                    tic_env,
                    experiment_kind="equal-batch",
                )
            )
            b_tps, t_tps = _decode_tps_pair(b_doc, t_doc)
            meta: dict[str, Any] = {}
            if equal_batch_meta_path.is_file():
                meta = load_json(equal_batch_meta_path)
            max_seqs = int(
                meta.get("max_num_seqs")
                or b_doc.get("serve_config", {}).get("max_num_seqs")
                or 0
            )
            max_conc = int(
                meta.get("max_concurrency")
                or b_doc.get("bench_config", {}).get("max_concurrency")
                or 0
            )
            summary["equal_batch"] = build_equal_batch_block(
                max_num_seqs=max_seqs,
                max_concurrency=max_conc,
                baseline_tps=b_tps,
                tic_tps=t_tps,
                baseline_latency_ms=b_doc.get("latency_ms"),
                tic_latency_ms=t_doc.get("latency_ms"),
            )
            summary["equal_batch"]["baseline_ref"] = str(
                b_path.relative_to(args.run_dir)
            )
            summary["equal_batch"]["tic_ref"] = str(
                t_path.relative_to(args.run_dir)
            )
            for item in validate_equal_batch(summary["equal_batch"]):
                errors.append(f"equal_batch: {item}")
    elif experiment_kind == "equal-batch":
        # Standalone equal-batch mode: transparency block from main decode.
        decode_name = next(
            (
                name
                for name, (_, doc) in baseline_profiles.items()
                if int(doc["workload"]["output_tokens"]) > 64
            ),
            None,
        )
        if decode_name is not None:
            _, b_doc = baseline_profiles[decode_name]
            _, t_doc = tic_profiles[decode_name]
            b_tps, t_tps = _decode_tps_pair(b_doc, t_doc)
            summary["equal_batch"] = build_equal_batch_block(
                max_num_seqs=int(b_doc["serve_config"]["max_num_seqs"]),
                max_concurrency=int(b_doc["bench_config"]["max_concurrency"]),
                baseline_tps=b_tps,
                tic_tps=t_tps,
                baseline_latency_ms=b_doc.get("latency_ms"),
                tic_latency_ms=t_doc.get("latency_ms"),
            )

    # Equal-batch transparency is eng-only; not required for publish capacity.

    verify_mode = summary["correctness"]["verify_mode"]
    if verify_mode == "reference":
        summary["correctness"]["bit_exact_ok"] = bool(verify.get("bit_exact_ok"))
    elif "bit_exact_ok" in verify:
        # Preserve an explicit eng field when present; do not invent PASS.
        summary["correctness"]["bit_exact_ok"] = bool(verify.get("bit_exact_ok"))

    # Optional greedy serve output match (token IDs), captured while servers are up.
    fold_serve_output_match(args.run_dir, summary["correctness"])

    summary["fairness"] = {"passed": not errors, "mismatches": errors}
    environment = dict(baseline_env)
    environment["variant_runtime"] = {
        "baseline": {
            "substrate": baseline_env.get("substrate"),
            "vllm_version": baseline_env.get("vllm_version"),
            "image_tag": baseline_env.get("image_tag"),
            "image_digest": baseline_env.get("image_digest"),
        },
        "tic": {
            "substrate": tic_env.get("substrate"),
            "vllm_version": tic_env.get("vllm_version"),
            "image_tag": tic_env.get("image_tag"),
            "image_digest": tic_env.get("image_digest"),
            "isiro_git_sha": tic_env.get("isiro_git_sha"),
            # Scratch audit only; not fairness keys and not rendered in report.md.
            "runtime_identity": tic_env.get("runtime_identity") or {},
            "compute_apps_empty": tic_env.get("compute_apps_empty"),
            "host_load": tic_env.get("host_load") or {},
            "gpu_clocks_power_temp": tic_env.get("gpu_clocks_power_temp") or [],
        },
    }
    write_json(args.run_dir / "environment.json", environment)
    out = args.out or args.run_dir / "summary.json"
    write_json(out, summary)
    if errors:
        for error in errors:
            print(error)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
