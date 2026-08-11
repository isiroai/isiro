#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture reproducibility metadata without collecting secrets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_lib import sha256_file, write_json


def command(argv: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gpu_data() -> tuple[list[dict[str, Any]], str, str]:
    query = (
        "name,uuid,memory.total,pci.bus_id,power.limit,driver_version,"
        "clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,"
        "pstate"
    )
    raw = command(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 6:
            continue
        row: dict[str, Any] = {
            "name": fields[0],
            # Identity fields stay out of published JSON; promote redacts too.
            "uuid": fields[1],
            "memory_total_mib": int(float(fields[2])),
            "pci_bus_id": fields[3],
            "power_limit_w": float(fields[4]),
        }
        if len(fields) >= 11:
            row["clocks_graphics_mhz"] = _float_or_none(fields[6])
            row["clocks_memory_mhz"] = _float_or_none(fields[7])
            row["power_draw_w"] = _float_or_none(fields[8])
            row["temperature_c"] = _float_or_none(fields[9])
            row["pstate"] = fields[10]
        rows.append(row)
    driver = ""
    if raw.splitlines() and len(raw.splitlines()[0].split(",")) >= 6:
        driver = raw.splitlines()[0].split(",")[5].strip()
    status = command(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", status)
    cuda = cuda_match.group(1) if cuda_match else ""
    return rows, driver, cuda


def host_load() -> dict[str, Any]:
    loadavg = Path("/proc/loadavg")
    if not loadavg.is_file():
        return {}
    parts = loadavg.read_text(encoding="utf-8").split()
    if len(parts) < 3:
        return {}
    return {
        "load_1m": _float_or_none(parts[0]),
        "load_5m": _float_or_none(parts[1]),
        "load_15m": _float_or_none(parts[2]),
    }


def compute_apps_empty() -> bool:
    raw = command(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    )
    pids = [line.strip() for line in raw.splitlines() if line.strip()]
    return len(pids) == 0


def probe_fused_extension() -> dict[str, Any]:
    """Resolve and hash the installed fused CUDA extension used by ``isiro serve``."""
    out: dict[str, Any] = {
        "fused_ext_so_path": "",
        "fused_ext_so_sha256": "",
        "fused_ext_source": "",
        "fused_compile_flags": {},
    }
    path: Path | None = None
    source = ""
    # Installed module names from the runtime wheel (host lookup only).
    _module_candidates = (
        "tic_nvidia_ext_pf",
        "isiro_runtime.native.tic_nvidia_ext_pf",
    )
    try:
        import importlib.util

        for mod_name in _module_candidates:
            spec = importlib.util.find_spec(mod_name)
            if spec is not None and spec.origin and Path(spec.origin).is_file():
                path = Path(spec.origin)
                source = "installed_module" if "." not in mod_name else "aot_native"
                break
    except Exception:
        path = None
    if path is None:
        # Editable/dev fallback: isiro_runtime package next to native/.
        try:
            import isiro_runtime

            candidate = (
                Path(isiro_runtime.__file__).resolve().parent
                / "native"
                / "tic_nvidia_ext_pf.so"
            )
            if candidate.is_file():
                path = candidate
                source = "aot_native"
        except Exception:
            pass
    if path is None:
        return out
    out["fused_ext_so_path"] = str(path)
    out["fused_ext_so_sha256"] = sha256_file(path)
    out["fused_ext_source"] = source
    flags_path = Path(str(path) + ".flags.json")
    if flags_path.is_file():
        try:
            flags = json.loads(flags_path.read_text(encoding="utf-8"))
            if isinstance(flags, dict):
                out["fused_compile_flags"] = flags
        except (OSError, ValueError):
            pass
    return out


def resolve_runtime_git_sha(explicit: Path | None) -> str:
    """Git SHA of the installed runtime tree that provides the TIC extension."""
    if explicit is not None:
        sha = repo_git_sha(explicit)
        if sha:
            return sha
    try:
        import isiro_runtime

        root = Path(isiro_runtime.__file__).resolve().parents[2]
        return repo_git_sha(root)
    except Exception:
        return ""


def server_version(url: str) -> str:
    if not url:
        return ""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict):
            return str(payload.get("version") or payload.get("vllm_version") or "")
    except (OSError, ValueError):
        return ""
    return ""


def image_digest(image: str) -> str:
    if not image:
        return ""
    raw = command(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"]
    )
    try:
        digests = json.loads(raw)
        return str(digests[0]) if digests else ""
    except (ValueError, TypeError):
        return ""


def repo_git_sha(repo: Path | None) -> str:
    if repo is None:
        return ""
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        # Still try; git may resolve via worktree.
        pass
    return command(["git", "rev-parse", "HEAD"], cwd=repo)


def build_environment(args: argparse.Namespace) -> dict[str, Any]:
    gpus, driver, cuda_detail = gpu_data()
    memory_kib = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_kib = int(line.split()[1])
                break
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    substrate = args.substrate
    image_tag = args.image if substrate == "docker" else ""
    digest = image_digest(args.image) if substrate == "docker" else ""
    load = host_load()
    env: dict[str, Any] = {
        "schema": "isiro-benchmark-environment-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "system_id": args.system_id,
        "substrate": substrate,
        "gpu_model": gpus[0]["name"] if gpus else "",
        "gpu_count": len(gpus),
        "gpus": gpus,
        "driver_version": driver,
        "cuda_version": os.environ.get("CUDA_VERSION") or cuda_detail,
        "cpu_model": cpu_model,
        "cpu_logical_count": os.cpu_count(),
        "ram_total_bytes": memory_kib * 1024,
        "numa_summary": command(["lscpu", "-p=NODE,CPU"]),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "docker_version": command(["docker", "--version"]),
        "container_runtime": command(["docker", "info", "--format", "{{.Driver}}"]),
        "image_tag": image_tag,
        "image_digest": digest,
        "vllm_version": server_version(args.version_url),
        "isiro_version": args.isiro_version,
        "isiro_git_sha": repo_git_sha(args.repo),
        "quiet_host_expected": bool(args.quiet_host),
        "compute_apps_empty": compute_apps_empty(),
        "host_load": load,
    }
    if gpus:
        env["gpu_clocks_power_temp"] = [
            {
                "name": gpu.get("name"),
                "clocks_graphics_mhz": gpu.get("clocks_graphics_mhz"),
                "clocks_memory_mhz": gpu.get("clocks_memory_mhz"),
                "power_draw_w": gpu.get("power_draw_w"),
                "temperature_c": gpu.get("temperature_c"),
                "pstate": gpu.get("pstate"),
            }
            for gpu in gpus
        ]
    # Scratch-only runtime identity for TIC / AOT audits (not fairness keys).
    runtime_identity: dict[str, Any] = {}
    if args.runtime_repo is not None or substrate == "host_isiro":
        runtime_sha = resolve_runtime_git_sha(args.runtime_repo)
        if runtime_sha:
            runtime_identity["runtime_git_sha"] = runtime_sha
    if args.tic_model is not None:
        tic_path = args.tic_model
        if tic_path.is_dir():
            tic_path = tic_path / "model.tic"
        if tic_path.is_file():
            runtime_identity["model_tic_sha256"] = sha256_file(tic_path)
            runtime_identity["model_tic_path"] = str(tic_path)
    if substrate == "host_isiro" or args.probe_fused_extension:
        runtime_identity.update(probe_fused_extension())
    if runtime_identity:
        env["runtime_identity"] = runtime_identity
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--system-id", default="rtx-5090")
    parser.add_argument(
        "--substrate",
        choices=("docker", "host_isiro"),
        required=True,
    )
    parser.add_argument("--image", default="")
    parser.add_argument("--version-url", default="")
    parser.add_argument("--isiro-version", default="")
    parser.add_argument("--repo", type=Path)
    parser.add_argument(
        "--quiet-host",
        action="store_true",
        help="Caller verified the GPU had no other compute apps before capture",
    )
    parser.add_argument(
        "--runtime-repo",
        type=Path,
        help="Optional installed-runtime checkout path for runtime_git_sha (scratch audit)",
    )
    parser.add_argument(
        "--tic-model",
        type=Path,
        help="Path to model.tic or its containing directory (scratch audit)",
    )
    parser.add_argument(
        "--probe-fused-extension",
        action="store_true",
        help="Resolve and hash the installed fused CUDA extension even for non-host substrates",
    )
    args = parser.parse_args()
    if args.substrate == "docker" and not args.image:
        raise SystemExit("docker substrate requires --image")
    write_json(args.out, build_environment(args))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
