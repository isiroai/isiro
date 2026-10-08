#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure GDX KV cold roundtrip bit-exactness for the public A/B report.

Imports the product GDX atom from the serve overlay (or ISIRO_RUNTIME_SRC /
PYTHONPATH) and encodes/decodes a small synthetic page. Writes a JSON meter
for compare_ab / generate_report. Fail-closed: missing import or mismatch is
not PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCHEMA = "isiro-benchmark-gdx-kv-bitexact-v1"


def _serve_site_candidates() -> list[Path]:
    home = Path.home()
    env = (os.environ.get("ISIRO_SERVE_SITE") or "").strip()
    out: list[Path] = []
    if env:
        out.append(Path(env))
    out.append(home / ".isiro" / "serve-site")
    runtime_src = (os.environ.get("ISIRO_RUNTIME_SRC") or "").strip()
    if runtime_src:
        out.append(Path(runtime_src))
    return out


def _ensure_runtime_import() -> str | None:
    """Return None on success, else an error string."""
    try:
        import isiro_runtime  # noqa: F401

        return None
    except ImportError:
        pass
    for root in _serve_site_candidates():
        if not root.is_dir():
            continue
        marker = root / "isiro_runtime"
        if not marker.is_dir():
            continue
        text = str(root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
        try:
            import isiro_runtime  # noqa: F401

            return None
        except ImportError:
            continue
    return (
        "isiro_runtime not importable; set ISIRO_SERVE_SITE or install the "
        "product overlay under ~/.isiro/serve-site"
    )


def _roundtrip_bf16(seed: int = 11) -> dict[str, Any]:
    import numpy as np
    from isiro_runtime.ops.tic.native_compute.gdx.atoms.bf16_o1 import (
        bf16_to_u16,
        decode_gdx_bf16_o1,
        encode_gdx_bf16_o1,
    )

    rng = np.random.default_rng(seed)
    vals = rng.normal(size=(1, 64, 2, 32)).astype(np.float32)
    u16 = bf16_to_u16(vals)
    packed = encode_gdx_bf16_o1(vals)
    out = decode_gdx_bf16_o1(packed)
    ok = bool(np.array_equal(out, u16))
    bpv = float(packed.avg_bpv_meter())
    # Entropy win vs naive bf16 FA twin (16 bpv).
    if ok and not (bpv < 16.0):
        ok = False
    return {
        "alphabet": "bf16",
        "kv_bit_exact_ok": ok,
        "n_values": int(u16.size),
        "avg_bpv": bpv,
        "fa_bpv": 16.0,
        "coding_win": bool(bpv < 16.0),
    }


def _roundtrip_fp8(seed: int = 11) -> dict[str, Any]:
    import numpy as np
    from isiro_runtime.ops.tic.native_compute.gdx.atoms.e4m3_msb_o1 import (
        decode_gdx_e4m3_msb_o1,
        encode_gdx_e4m3_msb_o1,
    )

    rng = np.random.default_rng(seed)
    # Synthetic e4m3 codes as uint8 page bytes.
    codes = rng.integers(0, 256, size=(1, 64, 2, 32), dtype=np.uint8)
    packed = encode_gdx_e4m3_msb_o1(codes)
    out = decode_gdx_e4m3_msb_o1(packed)
    ok = bool(np.array_equal(out, codes))
    bpv = float(packed.avg_bpv_meter())
    if ok and not (bpv < 8.0):
        ok = False
    return {
        "alphabet": "fp8",
        "kv_bit_exact_ok": ok,
        "n_values": int(codes.size),
        "avg_bpv": bpv,
        "fa_bpv": 8.0,
        "coding_win": bool(bpv < 8.0),
    }


def _roundtrip_fp4(seed: int = 11) -> dict[str, Any]:
    import numpy as np
    from isiro_runtime.ops.tic.native_compute.gdx.atoms.e2m1_o1 import (
        decode_gdx_e2m1_o1,
        encode_gdx_e2m1_o1,
    )

    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(1, 64, 2, 32), dtype=np.uint8)
    packed = encode_gdx_e2m1_o1(codes)
    out = decode_gdx_e2m1_o1(packed)
    ok = bool(np.array_equal(out, codes))
    bpv = float(packed.avg_bpv_meter())
    if ok and not (bpv < 4.0):
        ok = False
    return {
        "alphabet": "fp4",
        "kv_bit_exact_ok": ok,
        "n_values": int(codes.size),
        "avg_bpv": bpv,
        "fa_bpv": 4.0,
        "coding_win": bool(bpv < 4.0),
    }


def run_meter(alphabet: str, *, seed: int = 11) -> dict[str, Any]:
    alphabet = alphabet.strip().lower()
    err = _ensure_runtime_import()
    if err is not None:
        return {
            "schema": SCHEMA,
            "alphabet": alphabet,
            "kv_bit_exact_ok": False,
            "error": err,
        }
    try:
        if alphabet == "bf16":
            body = _roundtrip_bf16(seed=seed)
        elif alphabet in ("fp8", "fp8_e4m3"):
            body = _roundtrip_fp8(seed=seed)
        elif alphabet in ("fp4", "nvfp4", "e2m1"):
            body = _roundtrip_fp4(seed=seed)
        else:
            return {
                "schema": SCHEMA,
                "alphabet": alphabet,
                "kv_bit_exact_ok": False,
                "error": f"unsupported GDX alphabet {alphabet!r}",
            }
    except Exception as exc:  # noqa: BLE001 - meter must fail-closed
        return {
            "schema": SCHEMA,
            "alphabet": alphabet,
            "kv_bit_exact_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"schema": SCHEMA, **body, "error": None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphabet",
        default=os.environ.get("ISIRO_BENCH_GDX")
        or os.environ.get("ISIRO_BENCH_GDX_KV")
        or "bf16",
        help="GDX alphabet (bf16, fp8, fp4)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write JSON meter here",
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    doc = run_meter(str(args.alphabet), seed=int(args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ok = bool(doc.get("kv_bit_exact_ok"))
    if ok:
        print(
            f"gdx_kv_bitexact: PASS alphabet={doc.get('alphabet')} "
            f"n={doc.get('n_values')} bpv={doc.get('avg_bpv')}",
            file=sys.stderr,
        )
        return 0
    print(
        f"gdx_kv_bitexact: FAIL alphabet={doc.get('alphabet')} "
        f"error={doc.get('error')}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
