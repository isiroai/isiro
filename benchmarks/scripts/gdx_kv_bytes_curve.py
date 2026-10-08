#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot cold GDX KV bytes vs naive FA twin across context length T.

Fail-closed when coded cold bytes do not beat FA twin at every measured T
(entropy-style win). Writes PNG + JSON meter for the public GDX report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "isiro-benchmark-gdx-kv-bytes-curve-v1"

# Qwen2.5-7B-ish KV geometry for the published curve (not a second product lock).
_DEFAULT_LAYERS = 28
_DEFAULT_KV_HEADS = 4
_DEFAULT_HEAD_DIM = 128
_DEFAULT_W = 512


def _ensure_runtime_import() -> str | None:
    try:
        import isiro_runtime  # noqa: F401

        return None
    except ImportError:
        pass
    home = Path.home()
    for root in (
        Path(os.environ.get("ISIRO_SERVE_SITE") or ""),
        home / ".isiro" / "serve-site",
    ):
        if not root.is_dir():
            continue
        text = str(root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
        try:
            import isiro_runtime  # noqa: F401

            return None
        except ImportError:
            continue
    return "isiro_runtime not importable for GDX KV-bytes curve"


def _fa_bpv(alphabet: str) -> float:
    a = alphabet.strip().lower()
    if a in ("fp8", "fp8_e4m3"):
        return 8.0
    if a in ("fp4", "nvfp4", "e2m1"):
        return 4.0
    return 16.0


def _smooth_bf16_page(T: int, H: int, D: int, seed: int):
    import numpy as np
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + int(T))
    base = torch.randn(1, 1, H, D, generator=g, dtype=torch.bfloat16)
    t = torch.linspace(0, 1, T).view(T, 1, 1)
    drift = (0.04 * t).to(torch.bfloat16)
    bf = (base + drift).expand(1, T, H, D).contiguous()
    return bf.view(torch.uint16).numpy()


def _smooth_fp8_page(T: int, H: int, D: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(int(seed) + int(T))
    # Correlated walk in e4m3 code space (not i.i.d. noise).
    out = np.empty((1, T, H, D), dtype=np.uint8)
    for h in range(H):
        for d in range(D):
            x = int(rng.integers(64, 192))
            for t in range(T):
                if rng.random() < 0.85:
                    x = (x + int(rng.integers(-2, 3))) & 255
                else:
                    x = int(rng.integers(64, 192))
                out[0, t, h, d] = x
    return out


def _pack_nbytes(alphabet: str, page) -> tuple[int, float]:
    """Return (packed_nbytes, payload_bpv) for one [1,T,H,D] page."""
    a = alphabet.strip().lower()
    if a in ("bf16", "bfloat16"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.bf16_o1 import (
            encode_gdx_bf16_o1,
            packed_nbytes_bf16_o1,
        )

        ra = encode_gdx_bf16_o1(page)
        return int(packed_nbytes_bf16_o1(ra)), float(ra.avg_bpv_meter())
    if a in ("fp8", "fp8_e4m3"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.e4m3_msb_o1 import (
            encode_gdx_e4m3_msb_o1,
        )

        ra = encode_gdx_e4m3_msb_o1(page)
        n = int(ra.n_values)
        packed = int(ra.streams.nbytes + ra.offsets.nbytes + ra.freq_blob.nbytes)
        bpv = float(packed * 8.0 / max(n, 1))
        return packed, bpv
    if a in ("fp4", "nvfp4", "e2m1"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.e2m1_o1 import (
            encode_gdx_e2m1_o1,
        )

        ra = encode_gdx_e2m1_o1(page)
        n = int(ra.n_values)
        packed = int(ra.streams.nbytes + ra.offsets.nbytes + ra.freq_blob.nbytes)
        bpv = float(packed * 8.0 / max(n, 1))
        return packed, bpv
    raise ValueError(f"unsupported alphabet {alphabet!r}")


def measure_curve(
    alphabet: str,
    *,
    tokens: list[int],
    layers: int = _DEFAULT_LAYERS,
    kv_heads: int = _DEFAULT_KV_HEADS,
    head_dim: int = _DEFAULT_HEAD_DIM,
    live_window: int = _DEFAULT_W,
    seed: int = 17,
) -> dict[str, Any]:
    err = _ensure_runtime_import()
    if err is not None:
        return {"schema": SCHEMA, "ok": False, "error": err, "points": []}
    fa_bpv = _fa_bpv(alphabet)
    points: list[dict[str, Any]] = []
    a = alphabet.strip().lower()
    for T in tokens:
        cold_t = max(0, int(T) - int(live_window))
        if cold_t < 1:
            # In-window: no cold mass; FA twin for hot only (parity).
            fa_hot = int(
                (int(T) * layers * 2 * kv_heads * head_dim * fa_bpv + 7) // 8
            )
            points.append(
                {
                    "T": int(T),
                    "cold_t": 0,
                    "fa_twin_nbytes": fa_hot,
                    "gdx_cold_nbytes": 0,
                    "gdx_total_nbytes": fa_hot,
                    "save_vs_fa": 0.0,
                    "payload_bpv": None,
                }
            )
            continue
        if a in ("bf16", "bfloat16"):
            page = _smooth_bf16_page(cold_t, kv_heads, head_dim, seed)
        elif a in ("fp8", "fp8_e4m3"):
            page = _smooth_fp8_page(cold_t, kv_heads, head_dim, seed)
        else:
            page = _smooth_fp8_page(cold_t, kv_heads, head_dim, seed) % 16
        packed_one, bpv = _pack_nbytes(a, page)
        # K and V per layer.
        gdx_cold = int(packed_one) * 2 * int(layers)
        fa_cold = int(
            (cold_t * layers * 2 * kv_heads * head_dim * fa_bpv + 7) // 8
        )
        fa_hot = int(
            (min(int(T), int(live_window)) * layers * 2 * kv_heads * head_dim * fa_bpv
             + 7)
            // 8
        )
        fa_all = fa_cold + fa_hot
        gdx_total = gdx_cold + fa_hot
        save = 1.0 - (float(gdx_total) / float(max(fa_all, 1)))
        points.append(
            {
                "T": int(T),
                "cold_t": int(cold_t),
                "fa_twin_nbytes": int(fa_all),
                "gdx_cold_nbytes": int(gdx_cold),
                "gdx_total_nbytes": int(gdx_total),
                "save_vs_fa": float(save),
                "payload_bpv": float(bpv),
            }
        )
    cold_pts = [p for p in points if int(p["cold_t"]) > 0]
    ok = bool(cold_pts) and all(float(p["save_vs_fa"]) > 0.0 for p in cold_pts)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "alphabet": a if a not in ("fp8_e4m3",) else "fp8",
        "fa_bpv": fa_bpv,
        "live_window": int(live_window),
        "layers": int(layers),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "points": points,
        "error": None if ok else "GDX cold bytes did not beat FA twin (no entropy win)",
    }


def render_png(doc: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = list(doc.get("points") or [])
    if not pts:
        raise RuntimeError("no points to plot")
    xs = [int(p["T"]) for p in pts]
    fa = [float(p["fa_twin_nbytes"]) / (1024**3) for p in pts]
    gdx = [float(p["gdx_total_nbytes"]) / (1024**3) for p in pts]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ax.plot(xs, fa, color="#4a4a4a", lw=1.8, label="Naive FA KV (hot+cold)")
    ax.plot(xs, gdx, color="#1f6f5b", lw=2.2, label="GDX (hot FA + cold o1)")
    ax.set_xlabel("Context tokens T")
    ax.set_ylabel("KV bytes (GiB)")
    ax.set_title(
        f"GDX {doc.get('alphabet')} KV bytes vs T "
        f"(W={doc.get('live_window')})"
    )
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphabet",
        default=os.environ.get("ISIRO_BENCH_GDX")
        or os.environ.get("ISIRO_BENCH_GDX_KV")
        or "bf16",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument(
        "--tokens",
        default="512,1024,2048,4096,8192",
        help="Comma-separated context lengths",
    )
    parser.add_argument("--live-window", type=int, default=_DEFAULT_W)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    tokens = [int(x) for x in str(args.tokens).split(",") if x.strip()]
    doc = measure_curve(
        str(args.alphabet),
        tokens=tokens,
        live_window=int(args.live_window),
        seed=int(args.seed),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if doc.get("ok"):
        render_png(doc, args.out_png)
        print(
            f"gdx_kv_bytes_curve: PASS alphabet={doc.get('alphabet')} "
            f"points={len(doc.get('points') or [])} png={args.out_png}",
            file=sys.stderr,
        )
        return 0
    print(
        f"gdx_kv_bytes_curve: FAIL error={doc.get('error')}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
