#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot cold GDX KV bytes vs naive FA twin across context length T.

Hot window stays FA; cold prefix is GDX o1. Fail-closed when max-T coded
bytes do not beat the FA twin, or when cold nbytes stay flat as T grows
(over-compressible synthetic). Writes PNG + JSON for the public GDX report.
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


def _bf16_page(T: int, H: int, D: int, seed: int):
    """Gaussian bf16 page (same family as gdx_kv_bitexact).

    A near-constant drift page over-compresses to ~0 bpv and makes the GDX
    curve look flat (hot FA only). Real KV-like mass must grow with cold T.
    """
    import numpy as np

    rng = np.random.default_rng(int(seed) + int(T))
    vals = rng.normal(size=(1, int(T), int(H), int(D))).astype(np.float32)
    # Pack as uint16 bit patterns; encode_gdx_bf16_o1 accepts float or u16.
    return vals


def _fp8_page(T: int, H: int, D: int, seed: int):
    """I.i.d. e4m3 code bytes (same family as gdx_kv_bitexact)."""
    import numpy as np

    rng = np.random.default_rng(int(seed) + int(T))
    return rng.integers(0, 256, size=(1, int(T), int(H), int(D)), dtype=np.uint8)


def _pack_nbytes(alphabet: str, page) -> tuple[int, float]:
    """Return (packed_nbytes, payload_bpv) for one [1,T,H,D] page."""
    a = alphabet.strip().lower()
    if a in ("bf16", "bfloat16"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.bf16_o1 import (
            encode_gdx_bf16_o1,
            packed_nbytes_bf16_o1,
        )

        ra = encode_gdx_bf16_o1(page)
        packed = int(packed_nbytes_bf16_o1(ra))
        # Payload bpv (bytes on the wire), not Shannon meter.
        bpv = float(packed * 8.0 / max(int(ra.n_values), 1))
        return packed, bpv
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
            page = _bf16_page(cold_t, kv_heads, head_dim, seed)
        elif a in ("fp8", "fp8_e4m3"):
            page = _fp8_page(cold_t, kv_heads, head_dim, seed)
        else:
            page = _fp8_page(cold_t, kv_heads, head_dim, seed) % 16
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
    err: str | None = None
    ok = False
    if not cold_pts:
        err = "GDX cold bytes curve has no cold-T points"
    else:
        last = cold_pts[-1]
        # Early T (cold just past W) can lose slightly to pack tables; require
        # a clear win at max T and a rising cold curve (not hot-only flat).
        if float(last["save_vs_fa"]) <= 0.0:
            err = "GDX cold+hot bytes did not beat FA twin at max T"
        elif float(last.get("payload_bpv") or fa_bpv) >= float(fa_bpv):
            err = (
                "GDX payload bpv at max T did not beat FA twin "
                f"({last.get('payload_bpv')} >= {fa_bpv})"
            )
        elif len(cold_pts) >= 2:
            first = cold_pts[0]
            t_ratio = float(last["cold_t"]) / float(max(int(first["cold_t"]), 1))
            b_ratio = float(last["gdx_cold_nbytes"]) / float(
                max(int(first["gdx_cold_nbytes"]), 1)
            )
            if t_ratio >= 4.0 and b_ratio < 2.0:
                err = (
                    "GDX cold nbytes did not grow with T "
                    f"(cold_t x{t_ratio:.1f}, bytes x{b_ratio:.2f}); "
                    "synthetic page is over-compressible"
                )
            else:
                ok = True
        else:
            ok = True
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
        "error": err,
    }


def render_png(
    doc: dict[str, Any],
    out: Path,
    *,
    gpu: str = "",
    gpu_count: int | None = None,
    model: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from needle_haystack import figure_tagline

    pts = list(doc.get("points") or [])
    if not pts:
        raise RuntimeError("no points to plot")
    xs = [int(p["T"]) for p in pts]
    fa = [float(p["fa_twin_nbytes"]) / (1024**3) for p in pts]
    gdx = [float(p["gdx_total_nbytes"]) / (1024**3) for p in pts]
    tagline = figure_tagline(gpu=gpu, gpu_count=gpu_count, model=model)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    # Match the public VRAM figure palette: Baseline red, Isiro blue.
    ax.plot(xs, fa, color="#c0392b", lw=1.9, label="Baseline KV")
    ax.plot(xs, gdx, color="#1f77b4", lw=2.2, label="Isiro GDX KV")
    ax.set_xlabel("Context tokens T")
    ax.set_ylabel("KV bytes (GiB)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    # Header above axes; equal outer pad on all four sides via savefig.
    fig.suptitle("KV bytes vs context", y=0.97, fontsize=13, va="top")
    top = 0.90
    if tagline:
        fig.text(
            0.5,
            0.925,
            tagline,
            ha="center",
            va="top",
            fontsize=9,
            color="#444444",
        )
        top = 0.89
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=top)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.35)
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
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID")
        or os.environ.get("ISIRO_BENCH_MODEL")
        or "",
        help="Model name for the figure tagline",
    )
    parser.add_argument(
        "--gpu-model",
        default=os.environ.get("ISIRO_BENCH_GPU_MODEL") or "",
        help="GPU name for the figure tagline",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=int(os.environ.get("ISIRO_BENCH_GPU_COUNT") or "0"),
        help="GPU count for the figure tagline (e.g. 1 for 'x 1')",
    )
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
        render_png(
            doc,
            args.out_png,
            gpu=str(args.gpu_model or ""),
            gpu_count=int(args.gpu_count) or None,
            model=str(args.model or ""),
        )
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
