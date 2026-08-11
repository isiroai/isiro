#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dispatch a local model A/B benchmark with allowed system ids."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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
            return line.split("=", 1)[1].strip() or None
    return None


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
    if graph_on:
        command.append("--graph-on")
    if graph_off:
        command.append("--graph-off")
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


def _write_model_report(
    *,
    root: Path,
    model_dir: Path,
    system_id: str,
    run_dir: Path,
    companion_run_dir: Path | None = None,
) -> Path:
    stamp = _report_stamp(run_dir, system_id)
    report_out = model_dir / f"{system_id}-report-{stamp}.md"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(root / "benchmarks" / "scripts" / "generate_report.py"),
        "--run-dir",
        str(run_dir),
        "--out",
        str(report_out),
    ]
    if companion_run_dir is not None:
        cmd.extend(["--companion-run-dir", str(companion_run_dir)])
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"generate_report failed with exit {completed.returncode}"
        )
    return report_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public capacity-primary A/B (Graph ON by default). "
            "Pass --graph-off for Graph OFF, or --both-graph-modes for both."
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
        help="Outer CUDA graphs ON (product default; explicit matched A/B).",
    )
    parser.add_argument(
        "--graph-off",
        action="store_true",
        help="Matched A/B with outer graphs OFF.",
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help=argparse.SUPPRESS,  # alias for --graph-on
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help=argparse.SUPPRESS,  # alias for --graph-off
    )
    parser.add_argument(
        "--both-graph-modes",
        action="store_true",
        help=(
            "Run Graph ON and Graph OFF A/Bs. Report shows Graph ON then Graph OFF."
        ),
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
    args = parser.parse_args(argv)

    graph_on = bool(args.graph_on or args.graphs)
    graph_off = bool(args.graph_off or args.eager)
    mode_flags = sum(bool(x) for x in (graph_on, graph_off, args.both_graph_modes))
    if mode_flags > 1:
        raise SystemExit(
            "pass only one of --graph-on, --graph-off, and --both-graph-modes"
        )
    if args.reuse_baseline and args.no_reuse_baseline:
        raise SystemExit("pass only one of --reuse-baseline and --no-reuse-baseline")

    root = repo_root()
    model = args.model
    model_dir = root / "benchmarks" / model
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")
    run_harness = root / "benchmarks" / "scripts" / "run_harness.sh"
    if not run_harness.is_file():
        raise SystemExit(f"harness entry not found: {run_harness}")

    config = args.config
    if config is None:
        env_config = os.environ.get("ISIRO_BENCH_CONFIG", "").strip()
        config = Path(env_config) if env_config else model_dir / "common.env"
    if not config.is_file():
        raise SystemExit(
            f"missing {config}; copy benchmarks/common.env.example to "
            f"{model_dir / 'common.env'} and set model paths"
        )
    system_id = resolve_system_id(args.system_id, peek_config_system_id(config))

    child_env = os.environ.copy()
    child_env["SYSTEM_ID"] = system_id
    child_env["ISIRO_BENCH_CONFIG"] = str(config)
    child_env["MODEL_DIR"] = str(model_dir.resolve())
    child_env["MODEL_SLUG"] = model
    # Defaults match published Graph ON report quality.
    child_env.setdefault("ISIRO_VERIFY_REFERENCE", "1")
    child_env.setdefault("ISIRO_BENCH_EQUAL_BATCH", "1")

    if args.both_graph_modes:
        # Separate scratch dirs; baseline cache keys include enforce_eager.
        # generate_report leads with Graph ON, then Graph OFF.
        # Labels stay eager/graphs for .last_run_* markers.
        runs = (
            ("eager", False, True),
            ("graphs", True, False),
        )
    elif graph_on:
        runs = (("graphs", True, False),)
    elif graph_off:
        runs = (("eager", False, True),)
    else:
        runs = (("graphs", True, False),)

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

    scratch_root = root / "benchmarks" / "scratch"
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

    if args.both_graph_modes and not args.dry_run:
        eager_dir = run_dirs.get("eager")
        graphs_dir = run_dirs.get("graphs")
        if eager_dir is None or graphs_dir is None:
            print(
                "both-graph-modes finished but missing scratch run dirs "
                f"(eager={eager_dir} graphs={graphs_dir})",
                file=sys.stderr,
            )
            return 2
        marker = eager_dir / "graphs_companion_run.txt"
        marker.write_text(str(graphs_dir.resolve()) + "\n", encoding="utf-8")
        try:
            report_out = _write_model_report(
                root=root,
                model_dir=model_dir,
                system_id=system_id,
                run_dir=eager_dir,
                companion_run_dir=graphs_dir,
            )
        except RuntimeError as exc:
            print(f"dual-graph report merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged dual-graph report: {report_out}", file=sys.stderr)
        print(f"ISIRO_BENCH_REPORT={report_out}")
    elif not args.dry_run and run_dirs:
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
