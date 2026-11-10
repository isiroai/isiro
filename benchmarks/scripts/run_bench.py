#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run vLLM's serving benchmark and write a compact normalized profile."""

from __future__ import annotations

import argparse
import json
import os
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

# vllm 0.31 bench client strips each iter_any() SSE chunk; that drops framing
# under concurrent chat and yields intermittent "Never received a valid chunk"
# (load_errors). Overlay the patched module in docker bench runs.
_SSE_PATCH = (
    Path(__file__).resolve().parent / "patches" / "endpoint_request_func.py"
)
_VLLM_SSE_IN_IMAGE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/lib/"
    "endpoint_request_func.py"
)
_GUTENBERG_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "pride_and_prejudice.txt"
)


def _bench_dataset_name() -> str:
    raw = (os.environ.get("ISIRO_BENCH_DATASET") or "random").strip().lower()
    if raw in ("gutenberg", "gutenberg_1342", "pride_and_prejudice"):
        return "gutenberg"
    return "random"


def _gutenberg_prefix_text(*, tokenizer_dir: str | None, n_tokens: int) -> str:
    """Decode the first ``n_tokens`` of Gutenberg #1342. Fail closed if missing."""
    if not _GUTENBERG_FIXTURE.is_file():
        raise RuntimeError(
            f"Gutenberg fixture required: {_GUTENBERG_FIXTURE} "
            "(Pride and Prejudice / gutenberg_1342)"
        )
    text = _GUTENBERG_FIXTURE.read_text(encoding="utf-8")
    tok_json = None
    if tokenizer_dir:
        cand = Path(tokenizer_dir) / "tokenizer.json"
        if cand.is_file():
            tok_json = cand
    if tok_json is None:
        raise RuntimeError(
            "Gutenberg bench needs a local tokenizer.json "
            "(pass --tokenizer <model-dir>)"
        )
    try:
        from tokenizers import Tokenizer
    except ImportError:
        py = (os.environ.get("ISIRO_BENCH_PYTHON") or "").strip()
        if not py:
            raise RuntimeError(
                "Gutenberg bench needs tokenizers, or ISIRO_BENCH_PYTHON "
                "pointing at a venv that has it"
            )
        dest = Path("/tmp") / "isiro_gutenberg_prefix.txt"
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "from tokenizers import Tokenizer\n"
            "fix, tokp, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]\n"
            "tok = Tokenizer.from_file(tokp)\n"
            "ids = tok.encode(Path(fix).read_text(encoding='utf-8')).ids\n"
            "if len(ids) < n:\n"
            "    raise SystemExit(f'gutenberg tokens {len(ids)} < {n}')\n"
            "Path(out).write_text(tok.decode(ids[:n], skip_special_tokens=True), "
            "encoding='utf-8')\n"
        )
        completed = subprocess.run(
            [
                py,
                "-c",
                script,
                str(_GUTENBERG_FIXTURE),
                str(tok_json),
                str(int(n_tokens)),
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Gutenberg token cut failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        return dest.read_text(encoding="utf-8")
    tok = Tokenizer.from_file(str(tok_json))
    ids = tok.encode(text).ids
    if len(ids) < int(n_tokens):
        raise RuntimeError(
            f"Gutenberg fixture has {len(ids)} tokens, need {int(n_tokens)}"
        )
    return tok.decode(ids[: int(n_tokens)], skip_special_tokens=True)


def write_gutenberg_jsonl(
    dest: Path,
    *,
    tokenizer_dir: str | None,
    n_tokens: int,
    n_prompts: int,
    output_tokens: int,
) -> Path:
    """JSONL for vLLM ``--dataset-name custom``. One Gutenberg prefix per prompt."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    prompt = _gutenberg_prefix_text(tokenizer_dir=tokenizer_dir, n_tokens=n_tokens)
    with dest.open("w", encoding="utf-8") as fh:
        for _ in range(max(1, int(n_prompts))):
            fh.write(
                json.dumps(
                    {"prompt": prompt, "output_tokens": int(output_tokens)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return dest


def _pandas_site_packages() -> Path | None:
    """Host pandas for the vLLM bench image (custom JSONL loader)."""
    candidates: list[Path] = []
    py = (os.environ.get("ISIRO_BENCH_PYTHON") or "").strip()
    if py:
        candidates.append(Path(py).resolve().parent.parent / "lib")
    candidates.append(Path.home() / ".isiro" / "bench-venv" / "lib")
    for lib in candidates:
        hits = sorted(lib.glob("python*/site-packages/pandas"))
        if hits:
            return hits[0].parent
    return None


def bench_prefix(
    *,
    vllm_image: str | None,
    raw_dir: Path,
    extra_mounts: list[Path] | None = None,
    need_pandas: bool = False,
) -> list[str]:
    """Host `vllm` if present; else `docker run … --entrypoint vllm <VLLM_IMAGE>`."""
    binary = shutil.which("vllm")
    # Prefer docker when an image is set so the SSE client patch mounts.
    # Host `vllm` skips the overlay and can still hit load_errors under load.
    if binary and not vllm_image:
        return [binary]
    if not vllm_image:
        raise RuntimeError(
            "vllm CLI not on PATH; pass --vllm-image "
            "(same image as baseline serve / VLLM_IMAGE)"
        )
    mount = raw_dir.resolve()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{mount}:{mount}",
        "-w",
        str(mount),
    ]
    if _SSE_PATCH.is_file():
        cmd.extend(["-v", f"{_SSE_PATCH.resolve()}:{_VLLM_SSE_IN_IMAGE}:ro"])
    for extra in extra_mounts or []:
        resolved = extra.resolve()
        if resolved.is_dir() or resolved.is_file():
            cmd.extend(["-v", f"{resolved}:{resolved}:ro"])
    # Prefer --tokenizer on a local dir (see BENCH_TOKENIZER). Do not pass Hub
    # tokens on the docker argv (dry-run / logs would echo them).
    if need_pandas:
        site = _pandas_site_packages()
        if site is None:
            raise RuntimeError(
                "Gutenberg custom dataset needs host pandas "
                "(ISIRO_BENCH_PYTHON / ~/.isiro/bench-venv)"
            )
        cmd.extend(["-e", "PYTHONPATH=/opt/isiro_pandas"])
        for name in ("pandas", "dateutil"):
            src = site / name
            if src.is_dir():
                cmd.extend(["-v", f"{src.resolve()}:/opt/isiro_pandas/{name}:ro"])
        six_py = site / "six.py"
        if six_py.is_file():
            cmd.extend(["-v", f"{six_py.resolve()}:/opt/isiro_pandas/six.py:ro"])
    cmd.extend(["--entrypoint", "vllm", vllm_image])
    return cmd

def executable() -> list[str]:
    """Backward-compatible helper for tests; prefer bench_prefix in new code."""
    binary = shutil.which("vllm")
    return [binary] if binary else [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def bench_command(args: argparse.Namespace, result_path: Path) -> list[str]:
    base_url = args.api_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    result_path = result_path.resolve()
    tok = (getattr(args, "tokenizer", None) or "").strip() or None
    extra_mounts: list[Path] = []
    if tok and Path(tok).is_dir():
        extra_mounts.append(Path(tok))
    dataset = _bench_dataset_name()
    gutenberg_jsonl: Path | None = None
    if dataset == "gutenberg":
        gutenberg_jsonl = args.raw_dir / "gutenberg_1342.jsonl"
        write_gutenberg_jsonl(
            gutenberg_jsonl,
            tokenizer_dir=tok,
            n_tokens=int(args.input_tokens),
            n_prompts=max(int(args.num_prompts), int(args.num_warmups) + 1),
            output_tokens=int(args.output_tokens),
        )
        extra_mounts.append(gutenberg_jsonl)
    prefix = bench_prefix(
        vllm_image=args.vllm_image,
        raw_dir=args.raw_dir,
        extra_mounts=extra_mounts,
        need_pandas=(dataset == "gutenberg"),
    )
    cmd = [
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
    ]
    if tok:
        cmd.extend(["--tokenizer", tok])
    cmd.extend(
        [
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
        ]
    )
    if dataset == "gutenberg":
        assert gutenberg_jsonl is not None
        cmd.extend(
            [
                "--dataset-name",
                "custom",
                "--dataset-path",
                str(gutenberg_jsonl.resolve()),
                "--custom-output-len",
                str(args.output_tokens),
                "--skip-chat-template",
            ]
        )
    else:
        cmd.extend(
            [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(args.input_tokens),
                "--random-output-len",
                str(args.output_tokens),
            ]
        )
    cmd.extend(
        [
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
    )
    # Optional JSON body merge for chat (e.g. Qwen3 thinking off). Same for
    # baseline and TIC so A/B stays fair. Env: ISIRO_BENCH_EXTRA_BODY.
    extra = (getattr(args, "extra_body", None) or "").strip() or (
        os.environ.get("ISIRO_BENCH_EXTRA_BODY") or ""
    ).strip()
    if extra:
        json.loads(extra)  # fail closed on bad JSON before docker
        cmd.extend(["--extra-body", extra])
    return cmd


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
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Local tokenizer dir (avoids gated Hub fetch for meta-llama/*).",
    )
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
    parser.add_argument(
        "--extra-body",
        default=None,
        help=(
            "JSON object passed to vllm bench --extra-body (or set "
            "ISIRO_BENCH_EXTRA_BODY). Example: "
            '\'{"chat_template_kwargs":{"enable_thinking":false}}\''
        ),
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
            "dataset_name": _bench_dataset_name(),
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
