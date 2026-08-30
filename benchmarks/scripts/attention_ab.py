#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse serve attention backend from an isiro/vLLM serve log."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

# Recorded, not a FAIR_SERVE_KEYS field. Baseline FLASH_ATTN vs TIC CUSTOM
# must not fail public TIC vs vLLM fairness.
_BACKEND_RES = (
    re.compile(r"Using AttentionBackendEnum\.([A-Za-z0-9_]+)"),
    re.compile(r"Using ([A-Za-z0-9_]+) attention backend"),
    re.compile(r"attention_backend[=:\s'\"]+([A-Za-z0-9_]+)"),
)


def parse_serve_attention(text: str) -> dict[str, Any]:
    """Best-effort backend name from a serve log. Missing is null, not an error."""
    backend: str | None = None
    for pattern in _BACKEND_RES:
        match = pattern.search(text)
        if match:
            backend = str(match.group(1))
            break
    if backend is None and "CUSTOM vendor-family drop-in" in text:
        backend = "CUSTOM"
    elif backend is None and "CUSTOM invent KV" in text:
        backend = "CUSTOM"
    return {"attention_backend": backend}


def requested_isiro_attn(env: dict[str, str] | None = None) -> str:
    value = (env or os.environ).get("ISIRO_ATTN")
    if value is None or value.strip() == "":
        return "unset"
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    doc = parse_serve_attention(text)
    doc["isiro_attn"] = requested_isiro_attn()
    payload = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
