#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run vLLM's serving benchmark and write a compact normalized profile."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_lib import (
    PROFILE_SCHEMA,
    aggregate_replicates,
    normalize_vllm_result,
    write_json,
)


def bench_prefix(*, vllm_image: str | None, raw_dir: Path) -> list[str]:
    """Host `vllm` if present; else `docker run … --entrypoint vllm <VLLM_IMAGE>`."""
    binary = shutil.which("vllm")
    if binary:
        return [binary]
    if not vllm_image:
        raise RuntimeError(
            "vllm CLI not on PATH; pass --vllm-image "
            "(same image as baseline serve / VLLM_IMAGE)"
        )
    mount = raw_dir.resolve()
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{mount}:{mount}",
        "-w",
        str(mount),
        "--entrypoint",
        "vllm",
        vllm_image,
    ]


def executable() -> list[str]:
    """Backward-compatible helper for tests; prefer bench_prefix in new code."""
    binary = shutil.which("vllm")
    return [binary] if binary else [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def bench_command(args: argparse.Namespace, result_path: Path) -> list[str]:
    base_url = args.api_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    result_path = result_path.resolve()
    prefix = bench_prefix(vllm_image=args.vllm_image, raw_dir=args.raw_dir)
    return [
        *prefix,
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--base-url",
        base_url,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.num_warmups),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(args.max_concurrency),
        "--burstiness",
        "1.0",
        "--seed",
        str(args.seed),
        "--random-input-len",
        str(args.input_tokens),
        "--random-output-len",
        str(args.output_tokens),
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--result-filename",
        str(result_path),
        "--metadata",
        f"variant={args.variant}",
        "--metadata",
        f"profile={args.profile}",
    ]


def run_once(command: list[str], output: Path) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"vllm bench serve failed: {message}")
    if not output.is_file():
        raise RuntimeError(f"vllm did not write {output}")
    raw = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{output}: expected a JSON object")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "isiro"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", default="openai-chat")
    parser.add_argument("--endpoint", default="/chat/completions")
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--num-warmups", type=int, default=50)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--serve-config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--vllm-image",
        default=None,
        help="Docker image with vllm CLI when host vllm is not on PATH (VLLM_IMAGE).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--raw-input",
        type=Path,
        action="append",
        default=[],
        help="Normalize existing vLLM JSON instead of invoking vLLM",
    )
    args = parser.parse_args()

    serve_config = json.loads(args.serve_config.read_text(encoding="utf-8"))
    raw_paths: list[Path] = []
    commands: list[list[str]] = []
    if args.raw_input:
        raw_paths = args.raw_input
    else:
        for index in range(args.replicates):
            path = args.raw_dir / f"{args.variant}-{args.profile}-r{index + 1}.json"
            raw_paths.append(path)
            commands.append(bench_command(args, path))

    if args.dry_run:
        for command in commands:
            print(json.dumps(command))
        return 0

    normalized: list[dict[str, Any]] = []
    if args.raw_input:
        for path in raw_paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            normalized.append(normalize_vllm_result(raw))
    else:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        for command, path in zip(commands, raw_paths):
            normalized.append(normalize_vllm_result(run_once(command, path)))

    combined = aggregate_replicates(normalized)
    profile = {
        "schema": PROFILE_SCHEMA,
        "run_id": args.run_id,
        "variant": args.variant,
        "profile": args.profile,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "publish_quality": True,
        "smoke": False,
        "workload": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
        },
        "serve_config": serve_config,
        "bench_config": {
            "backend": args.backend,
            "endpoint": args.endpoint,
            "model_id": args.model,
            "dataset_name": "random",
            "num_prompts": args.num_prompts,
            "num_warmups": args.num_warmups,
            "request_rate": "inf",
            "max_concurrency": args.max_concurrency,
            "burstiness": 1.0,
            "seed": args.seed,
        },
        **combined,
        "replicates": len(normalized),
        "replicate_summaries": normalized,
        "aggregate_summary": combined,
        "bench_commands": commands,
    }
    write_json(args.out, profile)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
