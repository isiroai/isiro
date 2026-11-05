#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap + preflight the host Python used for Isiro KV cold meters.

Serve A/B stays in the pinned ``VLLM_IMAGE`` Docker container. Isiro KV bitexact
and the KV-bytes curve run on the host (CUDA torch + transformers). Remotes
drift: a fresh ``pip install transformers`` can pull a release that needs
``kernels`` for Hub FP8 finegrained linear, then the curve dies after verify.

Top-team pattern this follows:
  - committed pin file (``requirements-isiro-kv-meters.txt``)
  - managed venv (default ``~/.isiro/bench-venv``)
  - fail-closed preflight *before* the long serve loop
  - optional self-heal install into the managed venv

Does not modify the vLLM image. SSH vs local is the same path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


@dataclass
class PreflightReport:
    ok: bool
    python: str
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        lines = [f"python={self.python}", f"ok={self.ok}"]
        for n in self.notes:
            lines.append(f"note={n}")
        for e in self.errors:
            lines.append(f"error={e}")
        return lines


def preflight_python(python: Path, *, need_cuda: bool = True) -> PreflightReport:
    """Import-check meter deps in ``python`` (no network)."""
    py = str(python)
    report = PreflightReport(ok=True, python=py)
    if not python.is_file() and not python.is_symlink():
        # Allow bare name resolved later; here we expect an executable path.
        if not Path(py).exists():
            report.ok = False
            report.errors.append(f"python not found: {py}")
            return report

    probe = r"""
import importlib
import sys
errs = []
notes = []
def need(mod):
    try:
        return importlib.import_module(mod)
    except Exception as exc:
        errs.append(f"{mod}: {exc}")
        return None
numpy = need("numpy")
torch = need("torch")
transformers = need("transformers")
kernels = need("kernels")
need("matplotlib")
if torch is not None:
    notes.append(f"torch={getattr(torch, '__version__', '?')}")
    cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    notes.append(f"torch.cuda.is_available={cuda}")
    if not cuda:
        errs.append("torch.cuda.is_available() is False (curve needs CUDA)")
if transformers is not None:
    notes.append(f"transformers={getattr(transformers, '__version__', '?')}")
if kernels is not None:
    kv = str(getattr(kernels, "__version__", "0"))
    notes.append(f"kernels={kv}")
    m = __import__("re").match(r"(\d+)\.(\d+)(?:\.(\d+))?", kv)
    if not m:
        errs.append(f"kernels version unparseable: {kv!r}")
    else:
        ver = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
        if not ((0, 17, 0) <= ver < (0, 18, 0)):
            errs.append(
                f"kernels {kv} outside required "
                f"[0.17.0, 0.18.0) for transformers Hub FP8 finegrained"
            )
else:
    # Newer transformers Hub FP8 needs kernels; fail closed always when
    # metering (pin file installs it). Missing import is an error.
    errs.append(
        "kernels not importable; Hub FP8 curve needs "
        "kernels>=0.17.0,<0.18.0 (see requirements-isiro-kv-meters.txt)"
    )
# isiro_runtime is optional here: harness overlay may be on PYTHONPATH later.
try:
    import isiro_runtime  # noqa: F401
    notes.append("isiro_runtime=ok")
except Exception:
    notes.append("isiro_runtime=pending (harness sets serve-site / PYTHONPATH)")
print("NOTES:" + "\t".join(notes))
print("ERRS:" + "\t".join(errs))
sys.exit(1 if errs else 0)
"""
    if not need_cuda:
        probe = probe.replace(
            'errs.append("torch.cuda.is_available() is False (curve needs CUDA)")',
            "notes.append('cuda_check_skipped=1')",
        )
    try:
        completed = subprocess.run(
            [py, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.ok = False
        report.errors.append(str(exc))
        return report
    for line in (completed.stdout or "").splitlines():
        if line.startswith("NOTES:"):
            report.notes.extend([x for x in line[6:].split("\t") if x])
        elif line.startswith("ERRS:"):
            report.errors.extend([x for x in line[5:].split("\t") if x])
    if completed.returncode != 0 and not report.errors:
        err = (completed.stderr or completed.stdout or "preflight failed").strip()
        report.errors.append(err[:500])
    report.ok = not report.errors
    return report


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    if dry_run:
        return
    # Pip progress goes to stdout; harness captures stdout for --print-python.
    subprocess.run(cmd, check=True, stdout=sys.stderr)


def ensure_venv(venv: Path, *, dry_run: bool = False) -> Path:
    py = venv / "bin" / "python"
    if py.is_file():
        return py
    _run([sys.executable, "-m", "venv", str(venv)], dry_run=dry_run)
    if dry_run:
        return py
    if not py.is_file():
        raise RuntimeError(
            f"venv missing {py}; on Debian/Ubuntu install python3-venv "
            f"(e.g. sudo apt-get install -y python3.12-venv)"
        )
    return py


def pip_install(
    python: Path,
    args: list[str],
    *,
    dry_run: bool = False,
) -> None:
    _run([str(python), "-m", "pip", "install", "-U", "pip", "wheel"], dry_run=dry_run)
    _run([str(python), "-m", "pip", "install", *args], dry_run=dry_run)


def ensure_torch(
    python: Path,
    *,
    index_url: str,
    dry_run: bool = False,
) -> None:
    report = preflight_python(python, need_cuda=True)
    if any(n.startswith("torch=") for n in report.notes) and not any(
        "torch.cuda.is_available=False" in e or "torch.cuda.is_available() is False" in e
        for e in report.errors
    ):
        if any(n == "torch.cuda.is_available=True" for n in report.notes):
            return
    print(
        f"isiro_kv_meter_env: installing torch from {index_url}",
        file=sys.stderr,
    )
    pip_install(
        python,
        ["torch", "--index-url", index_url],
        dry_run=dry_run,
    )


def default_venv() -> Path:
    env = (os.environ.get("ISIRO_BENCH_VENV") or "").strip()
    if env:
        return Path(env)
    return Path.home() / ".isiro" / "bench-venv"


def default_requirements() -> Path:
    return Path(__file__).resolve().parents[1] / "requirements-isiro-kv-meters.txt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Existing interpreter to preflight (e.g. ISIRO_BENCH_PYTHON)",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=None,
        help="Managed venv path (default: $ISIRO_BENCH_VENV or ~/.isiro/bench-venv)",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=None,
        help="Pin file (default: benchmarks/requirements-isiro-kv-meters.txt)",
    )
    parser.add_argument(
        "--torch-index-url",
        default=(
            os.environ.get("ISIRO_BENCH_TORCH_INDEX") or _DEFAULT_TORCH_INDEX
        ).strip(),
        help="PyTorch wheel index for CUDA builds",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Preflight only; do not create/install",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print install actions without running them",
    )
    parser.add_argument(
        "--print-python",
        action="store_true",
        help="Print the interpreter path on stdout (harness capture)",
    )
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Preflight without requiring torch.cuda (unit tests)",
    )
    args = parser.parse_args(argv)

    req = args.requirements or default_requirements()
    if not req.is_file():
        print(f"isiro_kv_meter_env: FAIL missing requirements {req}", file=sys.stderr)
        return 2

    need_cuda = not args.allow_no_cuda

    if args.python is not None:
        report = preflight_python(Path(args.python), need_cuda=need_cuda)
        for line in report.as_lines():
            print(f"isiro_kv_meter_env: {line}", file=sys.stderr)
        if report.ok:
            if args.print_python:
                print(report.python)
            return 0
        print(
            "isiro_kv_meter_env: ISIRO_BENCH_PYTHON failed preflight; "
            "falling back to managed venv"
            if not args.check_only
            else "isiro_kv_meter_env: FAIL preflight (check-only)",
            file=sys.stderr,
        )
        if args.check_only:
            return 2

    if args.check_only and args.python is None:
        # Check managed venv if present.
        venv_py = (args.venv or default_venv()) / "bin" / "python"
        report = preflight_python(venv_py, need_cuda=need_cuda)
        for line in report.as_lines():
            print(f"isiro_kv_meter_env: {line}", file=sys.stderr)
        if args.print_python and report.ok:
            print(report.python)
        return 0 if report.ok else 2

    venv = args.venv or default_venv()
    try:
        py = ensure_venv(venv, dry_run=args.dry_run)
        pip_install(py, ["-r", str(req)], dry_run=args.dry_run)
        ensure_torch(py, index_url=str(args.torch_index_url), dry_run=args.dry_run)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"isiro_kv_meter_env: FAIL bootstrap: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.print_python:
            print(str(py))
        return 0

    report = preflight_python(py, need_cuda=need_cuda)
    for line in report.as_lines():
        print(f"isiro_kv_meter_env: {line}", file=sys.stderr)
    if not report.ok:
        print("isiro_kv_meter_env: FAIL after install", file=sys.stderr)
        return 2
    if args.print_python:
        print(report.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
