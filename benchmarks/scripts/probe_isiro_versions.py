#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resolve compiler (on-disk .tic) and runtime (isiro --help) semvers."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

THDR_MAGIC = b"THDR"
TSIG_MAGIC = b"TSIG"
TIC_MAGIC = b"TIC\x00"
THDR_TRAILER_SIZE = 12
TSIG_FOOTER_SIZE = 72
PREAMBLE_SIZE = 12
RUNTIME_RE = re.compile(r"ISIRO Runtime (v[0-9][^\s]*)")


def _needs_probe(value: str) -> bool:
    return value.strip().lower() in ("", "auto")


def tic_compiler_version(path: Path) -> str:
    file_size = path.stat().st_size
    if file_size < PREAMBLE_SIZE + THDR_TRAILER_SIZE:
        raise ValueError(f"TIC file too small: {path}")
    with path.open("rb") as fh:
        fh.seek(file_size - 4)
        signed = fh.read(4) == TSIG_MAGIC
        content_size = file_size - TSIG_FOOTER_SIZE if signed else file_size
        fh.seek(content_size - THDR_TRAILER_SIZE)
        trailer = fh.read(THDR_TRAILER_SIZE)
        if trailer[-4:] != THDR_MAGIC:
            raise ValueError(f"missing THDR trailer: {path}")
        (header_size,) = struct.unpack("<Q", trailer[:8])
        header_start = content_size - THDR_TRAILER_SIZE - header_size
        if header_start < PREAMBLE_SIZE:
            raise ValueError(f"header overlaps preamble: {path}")
        fh.seek(header_start)
        header = json.loads(fh.read(header_size).decode("utf-8"))
        fh.seek(0)
        if fh.read(4) != TIC_MAGIC:
            raise ValueError(f"not a TIC file: {path}")
    meta = header.get("__metadata__")
    if not isinstance(meta, dict):
        meta = {}
    for blob in (meta, header):
        raw = blob.get("tic_format") or blob.get("format")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raise ValueError(f"no tic_format in header: {path}")


def runtime_version_from_help(text: str) -> str:
    match = RUNTIME_RE.search(text)
    if match is None:
        raise ValueError("isiro --help did not print ISIRO Runtime vX")
    return match.group(1)


def probe_runtime(isiro_bin: str | None) -> str:
    exe = isiro_bin or shutil.which("isiro") or str(Path.home() / ".isiro" / "bin" / "isiro")
    completed = subprocess.run(
        [exe, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return runtime_version_from_help(text)


def resolve_versions(
    *,
    tic_path: Path | None,
    format_override: str,
    runtime_override: str,
    isiro_bin: str | None = None,
) -> tuple[str, str]:
    if _needs_probe(format_override):
        if tic_path is None or not tic_path.is_file():
            raise ValueError("ISIRO_FORMAT=auto needs a readable model.tic")
        compiler = tic_compiler_version(tic_path)
    else:
        compiler = format_override.strip()
    if _needs_probe(runtime_override):
        runtime = probe_runtime(isiro_bin)
    else:
        runtime = runtime_override.strip()
    return compiler, runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tic", type=Path)
    parser.add_argument(
        "--format-override",
        default=os.environ.get("ISIRO_FORMAT", "auto"),
    )
    parser.add_argument(
        "--runtime-override",
        default=os.environ.get("ISIRO_RUNTIME", "auto"),
    )
    parser.add_argument("--isiro-bin")
    args = parser.parse_args()
    compiler, runtime = resolve_versions(
        tic_path=args.tic,
        format_override=args.format_override,
        runtime_override=args.runtime_override,
        isiro_bin=args.isiro_bin,
    )
    print(f"ISIRO_FORMAT={compiler}")
    print(f"ISIRO_RUNTIME={runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
