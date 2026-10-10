#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dispatch a local model A/B benchmark with allowed system ids."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_SYSTEM_IDS = ("rtx-5090", "a100", "h100")
GPU_NAME_MAP = (
    ("geforce rtx 5090", "rtx-5090"),
    ("rtx 5090", "rtx-5090"),
    ("a100", "a100"),
    ("h100", "h100"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def detect_system_id() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    names = [line.strip().lower() for line in completed.stdout.splitlines() if line.strip()]
    if not names:
        return None
    matched: set[str] = set()
    for name in names:
        for needle, system_id in GPU_NAME_MAP:
            if needle in name:
                matched.add(system_id)
                break
    if len(matched) == 1:
        return next(iter(matched))
    return None


def resolve_system_id(explicit: str | None, config_system_id: str | None) -> str:
    if explicit:
        system_id = explicit
    elif config_system_id:
        system_id = config_system_id
    else:
        detected = detect_system_id()
        if detected is None:
            raise SystemExit(
                "could not auto-detect SYSTEM_ID; pass --system-id "
                f"({', '.join(ALLOWED_SYSTEM_IDS)})"
            )
        system_id = detected
    if system_id not in ALLOWED_SYSTEM_IDS:
        raise SystemExit(
            f"unknown system id {system_id!r}; allowed: {', '.join(ALLOWED_SYSTEM_IDS)}"
        )
    return system_id


def peek_config_system_id(config: Path | None) -> str | None:
    if config is None or not config.is_file():
        return None
    for raw in config.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("SYSTEM_ID="):
            return line.split("=", 1)[1].strip().strip("\"'") or None
    return None


def peek_config_truthy(config: Path | None, key: str) -> bool:
    """True when KEY=1/true/yes/on appears in the env-style config file."""
    if config is None or not config.is_file():
        return False
    prefix = f"{key}="
    for raw in config.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or not line.startswith(prefix):
            continue
        value = line.split("=", 1)[1].strip().strip("\"'").lower()
        # Expand trivial ${VAR:-default} forms used in local common.env copies.
        if value.startswith("${") and ":-" in value and value.endswith("}"):
            value = value[2:-1].split(":-", 1)[1].strip().lower()
        return value in ("1", "true", "yes", "on")
    return False


def _build_command(
    run_harness: Path,
    *,
    dry_run: bool,
    graph_on: bool,
    graph_off: bool,
    mode: str | None,
    reuse_baseline: bool,
    no_reuse_baseline: bool,
    profiles: str | None,
) -> list[str]:
    command = ["bash", str(run_harness)]
    if dry_run:
        command.append("--dry-run")
    if graph_off:
        command.append("--enforce-eager")
    if mode:
        command.append(f"--mode={mode}")
    if reuse_baseline:
        command.append("--reuse-baseline")
    if no_reuse_baseline:
        command.append("--no-reuse-baseline")
    if profiles:
        command.append(f"--profiles={profiles}")
    return command


def _report_stamp(run_dir: Path, system_id: str) -> str:
    """UTC stamp from scratch run id `{UTC}-{system_id}`."""
    stamp = run_dir.name
    suffix = f"-{system_id}"
    if stamp.endswith(suffix):
        return stamp[: -len(suffix)]
    return stamp


def _report_python() -> str:
    candidates = [sys.executable, "/tmp/isiro-plot-venv/bin/python"]
    for py in candidates:
        if not py or not Path(py).is_file():
            continue
        check = subprocess.run(
            [py, "-c", "import matplotlib, seaborn, pandas"],
            check=False,
            capture_output=True,
        )
        if check.returncode == 0:
            return py
    return sys.executable


def _write_model_report(
    *,
    root: Path,
    model_dir: Path,
    system_id: str,
    run_dir: Path,
) -> Path:
    stamp = _report_stamp(run_dir, system_id)
    report_dir = model_dir / f"{system_id}-report-{stamp}"
    report_out = report_dir / "report.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _report_python(),
        str(root / "benchmarks" / "scripts" / "generate_report.py"),
        "--run-dir",
        str(run_dir),
        "--out",
        str(report_out),
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"generate_report failed with exit {completed.returncode}"
        )
    return report_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run needle-in-a-haystack, then the public capacity-primary A/B "
            "(Graph ON by default). Pass --enforce-eager for full eager "
            "(--graph-off is an alias)."
        )
    )
    parser.add_argument(
        "model",
        help="Model directory under benchmarks/ (for example qwen2.5-7b-instruct)",
    )
    parser.add_argument("--system-id", choices=ALLOWED_SYSTEM_IDS, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--graph-on",
        action="store_true",
        help=argparse.SUPPRESS,  # unused; default is already Graph ON
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Full eager A/B (same meaning as vLLM --enforce-eager).",
    )
    parser.add_argument(
        "--graph-off",
        action="store_true",
        help=argparse.SUPPRESS,  # alias for --enforce-eager
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help=argparse.SUPPRESS,  # alias for default Graph ON
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help=argparse.SUPPRESS,  # alias for --enforce-eager
    )
    parser.add_argument(
        "--mode",
        choices=("capacity", "equal-batch"),
        default=None,
        help=argparse.SUPPRESS,  # eng-only; public default is capacity
    )
    parser.add_argument("--reuse-baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-reuse-baseline", action="store_true")
    parser.add_argument(
        "--profiles",
        default=None,
        help="Optional profile list (default: generation-32-256).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ISIRO_BENCH_CONFIG path (default: model common.env)",
    )
    parser.add_argument(
        "--skip-needle",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--needle-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    graph_on = bool(args.graph_on or args.graphs)
    graph_off = bool(args.enforce_eager or args.graph_off or args.eager)
    if graph_on and graph_off:
        raise SystemExit("pass only one of --graph-on and --enforce-eager")
    if args.reuse_baseline and args.no_reuse_baseline:
        raise SystemExit("pass only one of --reuse-baseline and --no-reuse-baseline")
    if args.skip_needle and args.needle_only:
        raise SystemExit("pass only one of --skip-needle and --needle-only")

    root = repo_root()
    model = args.model
    model_dir = root / "benchmarks" / model
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")
    run_harness = root / "benchmarks" / "scripts" / "run_harness.sh"
    run_needle = root / "benchmarks" / "scripts" / "run_needle_haystack.sh"
    if not run_harness.is_file():
        raise SystemExit(f"harness entry not found: {run_harness}")
    if not run_needle.is_file():
        raise SystemExit(f"needle entry not found: {run_needle}")

    config = args.config
    if config is None:
        env_config = os.environ.get("ISIRO_BENCH_CONFIG", "").strip()
        config = Path(env_config) if env_config else model_dir / "common.env"
    if not config.is_file():
        raise SystemExit(
            f"missing {config}; copy benchmarks/common.env.example to "
            f"{model_dir / 'common.env'} and set model paths"
        )
    skip_needle = bool(args.skip_needle)
    if not skip_needle and not args.needle_only:
        env_skip = os.environ.get("ISIRO_BENCH_SKIP_NEEDLE", "").strip().lower()
        if env_skip in ("1", "true", "yes", "on") or peek_config_truthy(
            config, "ISIRO_BENCH_SKIP_NEEDLE"
        ):
            skip_needle = True
    system_id = resolve_system_id(args.system_id, peek_config_system_id(config))

    child_env = os.environ.copy()
    child_env["SYSTEM_ID"] = system_id
    child_env["ISIRO_BENCH_CONFIG"] = str(config)
    child_env["MODEL_DIR"] = str(model_dir.resolve())
    child_env["MODEL_SLUG"] = model
    # Defaults match published Graph ON report quality.
    child_env.setdefault("ISIRO_VERIFY_REFERENCE", "1")
    # Skip-baseline remeters: do not append equal-batch transparency (that
    # cold-starts another baseline/Isiro serve after capacity and hides the
    # unlock signal for tens of minutes).
    _skip_base = child_env.get("ISIRO_BENCH_SKIP_BASELINE", "").strip().lower()
    if _skip_base in ("1", "true", "yes", "on"):
        child_env.setdefault("ISIRO_BENCH_EQUAL_BATCH", "0")
    else:
        child_env.setdefault("ISIRO_BENCH_EQUAL_BATCH", "1")

    if graph_off:
        runs = (("eager", False, True),)
    else:
        runs = (("graphs", True, False),)

    scratch_root = root / "benchmarks" / "scratch"
    existing = os.environ.get("ISIRO_BENCH_RUN_DIR", "").strip()
    if existing:
        run_dir = Path(existing)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = scratch_root / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{system_id}"
        )
    child_env["ISIRO_BENCH_RUN_DIR"] = str(run_dir)

    needle_cmd = ["bash", str(run_needle), model]
    if args.dry_run:
        needle_cmd.append("--dry-run")

    dry_meta = args.dry_run and os.environ.get("ISIRO_BENCH_LAUNCH_DRY_META") == "1"
    if dry_meta:
        print(str(run_harness))
        print(f"system_id={system_id}")
        print(f"MODEL_DIR={child_env['MODEL_DIR']}")
        print(f"MODEL_SLUG={child_env['MODEL_SLUG']}")
        print(
            "defaults: "
            f"ISIRO_VERIFY_REFERENCE={child_env.get('ISIRO_VERIFY_REFERENCE')} "
            f"ISIRO_BENCH_EQUAL_BATCH={child_env.get('ISIRO_BENCH_EQUAL_BATCH')}"
        )
        if not skip_needle:
            print(f"needle: {' '.join(needle_cmd)}")
        if not args.needle_only:
            for label, on, off in runs:
                cmd = _build_command(
                    run_harness,
                    dry_run=True,
                    graph_on=on,
                    graph_off=off,
                    mode=args.mode,
                    reuse_baseline=args.reuse_baseline,
                    no_reuse_baseline=args.no_reuse_baseline,
                    profiles=args.profiles,
                )
                print(f"{label}: {' '.join(cmd)}")
        return 0

    if not skip_needle:
        print(
            f"launch needle model={model} system_id={system_id} "
            f"command={' '.join(needle_cmd)}",
            file=sys.stderr,
        )
        needle_done = subprocess.run(needle_cmd, check=False, env=child_env)
        if needle_done.returncode != 0:
            print(
                f"launch needle failed with exit {needle_done.returncode}",
                file=sys.stderr,
            )
            return int(needle_done.returncode)
        if args.needle_only:
            return 0

    run_dirs: dict[str, Path] = {}
    rc = 0
    for label, on, off in runs:
        command = _build_command(
            run_harness,
            dry_run=args.dry_run,
            graph_on=on,
            graph_off=off,
            mode=args.mode,
            reuse_baseline=args.reuse_baseline,
            no_reuse_baseline=args.no_reuse_baseline,
            profiles=args.profiles,
        )
        print(
            f"launch mode={label} model={model} system_id={system_id} "
            f"config={config} command={' '.join(command)}",
            file=sys.stderr,
        )
        completed = subprocess.run(command, check=False, env=child_env)
        if completed.returncode != 0:
            rc = int(completed.returncode)
            print(
                f"launch mode={label} failed with exit {rc}",
                file=sys.stderr,
            )
            return rc
        marker = scratch_root / f".last_run_{label}"
        if marker.is_file():
            parsed = Path(marker.read_text(encoding="utf-8").strip())
            if parsed.is_dir():
                run_dirs[label] = parsed

    if not args.dry_run and run_dirs:
        label, run_dir = next(iter(run_dirs.items()))
        try:
            report_out = _write_model_report(
                root=root,
                model_dir=model_dir,
                system_id=system_id,
                run_dir=run_dir,
            )
        except RuntimeError as exc:
            print(f"report write failed for mode={label}: {exc}", file=sys.stderr)
            return 2
        print(f"model report: {report_out}", file=sys.stderr)
        print(f"ISIRO_BENCH_REPORT={report_out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
