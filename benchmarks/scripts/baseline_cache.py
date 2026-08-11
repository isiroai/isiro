#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fingerprint and reuse prior baseline serve/bench artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from benchmark_lib import load_json, write_json

# Public default profile set (decode throughput). TTFT profiles are eng-only.
REQUIRED_PROFILES = (
    "generation-32-256",
)


def weight_inventory(model_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*.safetensors")):
        rows.append(
            {
                "rel": path.relative_to(model_dir).as_posix(),
                "size": path.stat().st_size,
            }
        )
    return rows


def build_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema": "isiro-benchmark-baseline-fingerprint-v1",
        "vllm_image": args.vllm_image,
        "model_id": args.model_id,
        "precision": str(args.precision),
        "baseline_model_inventory": weight_inventory(Path(args.baseline_model)),
        "serve": {
            "max_model_len": int(args.max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "max_num_seqs": int(args.max_num_seqs),
            "prefix_caching": False,
            "enforce_eager": str(args.enforce_eager) == "true",
            "trust_remote_code": True,
            "flashinfer_autotune": False,
        },
        "bench": {
            "seed": int(args.seed),
            "warmups": int(args.warmups),
            "prompts": int(args.prompts),
            "replicates": int(args.replicates),
            "max_concurrency": int(args.max_concurrency),
            "profiles": list(REQUIRED_PROFILES),
        },
        "publish_quality": bool(args.publish_quality),
        "smoke": not bool(args.publish_quality),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["fingerprint"] = digest
    return payload


def fingerprint_complete(run_dir: Path) -> bool:
    if not (run_dir / "baseline_fingerprint.json").is_file():
        return False
    if not (run_dir / "baseline_environment.json").is_file():
        return False
    if not (run_dir / "baseline_serve_config.json").is_file():
        return False
    for name in REQUIRED_PROFILES:
        if not (run_dir / "baseline" / f"{name}.json").is_file():
            return False
    return True


def find_reusable(
    scratch_root: Path,
    fingerprint: str,
    *,
    require_publish_quality: bool,
) -> Path | None:
    if not scratch_root.is_dir():
        return None
    candidates = sorted(
        (
            path
            for path in scratch_root.iterdir()
            if path.is_dir() and fingerprint_complete(path)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in candidates:
        doc = load_json(path / "baseline_fingerprint.json")
        if doc.get("fingerprint") != fingerprint:
            continue
        if require_publish_quality and not doc.get("publish_quality"):
            continue
        if not require_publish_quality and doc.get("publish_quality"):
            # Smoke may reuse a publish-quality baseline with the same knobs.
            pass
        # Smoke fingerprint includes smoke=true / publish_quality=false, so
        # publish rows never match smoke fingerprints and vice versa except
        # when bench counts differ. Extra guard: refuse smoke-into-publish.
        if require_publish_quality and doc.get("smoke"):
            continue
        return path
    return None


def copy_baseline(source: Path, destination: Path) -> None:
    if not fingerprint_complete(source):
        raise ValueError(f"incomplete baseline cache source: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "baseline").mkdir(parents=True, exist_ok=True)
    for name in (
        "baseline_fingerprint.json",
        "baseline_environment.json",
        "baseline_serve_config.json",
        "output_match_baseline.json",
    ):
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)
    for profile in REQUIRED_PROFILES:
        shutil.copy2(
            source / "baseline" / f"{profile}.json",
            destination / "baseline" / f"{profile}.json",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write")
    write.add_argument("--out", type=Path, required=True)
    write.add_argument("--vllm-image", required=True)
    write.add_argument("--model-id", required=True)
    write.add_argument(
        "--precision",
        default="bf16",
        choices=("bf16", "fp8"),
        help="Matched A/B weight precision (fingerprint key).",
    )
    write.add_argument("--baseline-model", type=Path, required=True)
    write.add_argument("--max-model-len", required=True)
    write.add_argument("--gpu-memory-utilization", required=True)
    write.add_argument("--tensor-parallel-size", required=True)
    write.add_argument("--max-num-seqs", required=True)
    write.add_argument(
        "--enforce-eager",
        default="true",
        choices=("true", "false"),
        help="Matched A/B CUDA-graph mode (true=eager, false=graphs ON).",
    )
    write.add_argument("--seed", required=True)
    write.add_argument("--warmups", required=True)
    write.add_argument("--prompts", required=True)
    write.add_argument("--replicates", required=True)
    write.add_argument("--max-concurrency", required=True)
    write.add_argument("--publish-quality", action="store_true")

    find = sub.add_parser("find")
    find.add_argument("--scratch-root", type=Path, required=True)
    find.add_argument("--fingerprint-file", type=Path, required=True)
    find.add_argument("--require-publish-quality", action="store_true")

    copy = sub.add_parser("copy")
    copy.add_argument("--from-run", type=Path, required=True)
    copy.add_argument("--to-run", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "write":
        doc = build_fingerprint(args)
        write_json(args.out, doc)
        print(doc["fingerprint"])
        return 0
    if args.command == "find":
        doc = load_json(args.fingerprint_file)
        hit = find_reusable(
            args.scratch_root,
            str(doc["fingerprint"]),
            require_publish_quality=bool(args.require_publish_quality),
        )
        if hit is None:
            return 1
        print(hit)
        return 0
    copy_baseline(args.from_run, args.to_run)
    print(args.to_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
