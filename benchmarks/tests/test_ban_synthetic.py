"""CI: public isiro dens/report trees must not ship synthetic KV makers."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = (ROOT / "benchmarks" / "tests",)

BANNED = [
    re.compile(r"torch\.randn\s*\([^\)]*n_kv"),
    re.compile(r"torch\.randn\s*\([^\)]*Hkv"),
    re.compile(r"standard_normal\s*\(.*T"),
    re.compile(r"synthetic[_\s-]?peaky", re.I),
    re.compile(r"anchor.?plant", re.I),
]

ALLOW = {Path(__file__).resolve()}


def test_no_synthetic_kv_in_benchmark_tests() -> None:
    offenders: list[str] = []
    for base in SCAN_DIRS:
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if path.resolve() in ALLOW:
                continue
            text = path.read_text(encoding="utf-8")
            for pat in BANNED:
                if pat.search(text):
                    offenders.append(
                        f"{path.relative_to(ROOT)}: /{pat.pattern}/"
                    )
    assert not offenders, (
        "synthetic KV patterns banned in benchmark tests:\n" + "\n".join(offenders)
    )
