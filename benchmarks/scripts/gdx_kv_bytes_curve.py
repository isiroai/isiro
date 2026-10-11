#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot cold GDX KV bytes vs FA twin across context length T.

Uses real model KV from a real token stream (Project Gutenberg text).
One forward at max T; each T slices that cache. Hot window stays FA; cold
prefix is GDX o1. Writes PNG + JSON for the public report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "isiro-benchmark-gdx-kv-bytes-curve-v2"

_DEFAULT_W = 512
# Public-domain long text (Project Gutenberg #1342). Cached under the out dir.
_REAL_TEXT_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
_REAL_TEXT_NAME = "pride_and_prejudice.txt"


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


def _load_real_text(*, text_file: Path | None, cache_dir: Path) -> str:
    """Load a real long document (fixture / cache / Gutenberg download)."""
    if text_file is not None and text_file.is_file():
        return text_file.read_text(encoding="utf-8", errors="replace")
    fixture = Path(__file__).resolve().parent / "fixtures" / _REAL_TEXT_NAME
    if fixture.is_file():
        return fixture.read_text(encoding="utf-8", errors="replace")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / _REAL_TEXT_NAME
    if cached.is_file() and cached.stat().st_size > 10_000:
        return cached.read_text(encoding="utf-8", errors="replace")
    from urllib.request import urlopen

    with urlopen(_REAL_TEXT_URL, timeout=60) as resp:  # noqa: S310
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    if len(text) < 10_000:
        raise RuntimeError("real text download too short for KV curve")
    cached.write_text(text, encoding="utf-8")
    return text


def _pack_codes_nbytes(alphabet: str, codes) -> tuple[int, float]:
    """Pack one [1,T,H,D] code stack → (packed_nbytes, payload_bpv)."""
    import numpy as np

    a = alphabet.strip().lower()
    if a in ("bf16", "bfloat16"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.bf16_o1 import (
            encode_gdx_bf16_o1,
            packed_nbytes_bf16_o1,
        )

        ra = encode_gdx_bf16_o1(codes)
        packed = int(packed_nbytes_bf16_o1(ra))
        bpv = float(packed * 8.0 / max(int(ra.n_values), 1))
        return packed, bpv
    if a in ("fp8", "fp8_e4m3"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.e4m3_msb_o1 import (
            encode_gdx_e4m3_msb_o1,
        )

        u8 = np.ascontiguousarray(codes, dtype=np.uint8)
        ra = encode_gdx_e4m3_msb_o1(u8)
        n = int(ra.n_values)
        packed = int(ra.streams.nbytes + ra.offsets.nbytes + ra.freq_blob.nbytes)
        bpv = float(packed * 8.0 / max(n, 1))
        return packed, bpv
    if a in ("fp4", "nvfp4", "e2m1"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.e2m1_o1 import (
            encode_gdx_e2m1_o1,
        )

        u8 = np.ascontiguousarray(codes, dtype=np.uint8)
        ra = encode_gdx_e2m1_o1(u8)
        n = int(ra.n_values)
        packed = int(ra.streams.nbytes + ra.offsets.nbytes + ra.freq_blob.nbytes)
        bpv = float(packed * 8.0 / max(n, 1))
        return packed, bpv
    raise ValueError(f"unsupported alphabet {alphabet!r}")


def _kv_layer_to_lthd(k_or_v) -> Any:
    """HF cache [B,H,T,D] → numpy [1,T,H,D] uint16 (bf16 bits) or uint8."""
    import numpy as np
    import torch

    t = k_or_v.detach()
    if t.ndim != 4:
        raise RuntimeError(f"expected KV [B,H,T,D], got {tuple(t.shape)}")
    # [B,H,T,D] → [B,T,H,D]
    t = t.transpose(1, 2).contiguous()
    if t.dtype == torch.bfloat16:
        u16 = t.view(torch.uint16).cpu().numpy()
        return u16[:1]
    if t.dtype == torch.float16:
        u16 = t.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
        return u16[:1]
    if t.dtype == torch.uint8:
        return t.cpu().numpy()[:1]
    u16 = t.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    return u16[:1]


def _codes_for_alphabet(page_lthd, alphabet: str):
    """Map bf16-bit page to the deposit alphabet codes."""
    import numpy as np

    a = alphabet.strip().lower()
    if a in ("bf16", "bfloat16"):
        return page_lthd
    if a in ("fp8", "fp8_e4m3"):
        # Real vendor path stores e4m3; for the curve, take the high byte of
        # the bf16 word as a stand-in code plane (not a second silent quant
        # on the serve path; measurement-only).
        u = np.ascontiguousarray(page_lthd, dtype=np.uint16)
        return ((u.astype(np.uint32) >> 8) & 0xFF).astype(np.uint8)
    if a in ("fp4", "nvfp4", "e2m1"):
        u = np.ascontiguousarray(page_lthd, dtype=np.uint16)
        return ((u.astype(np.uint32) >> 12) & 0xF).astype(np.uint8)
    raise ValueError(f"unsupported alphabet {alphabet!r}")


def measure_curve(
    alphabet: str,
    *,
    tokens: list[int],
    model_dir: str | Path,
    text_file: Path | None = None,
    live_window: int = _DEFAULT_W,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Real-KV curve: one model forward, slice per T, pack cold prefix."""
    err = _ensure_runtime_import()
    if err is not None:
        return {"schema": SCHEMA, "ok": False, "error": err, "points": []}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": f"transformers/torch required for real KV curve: {exc}",
            "points": [],
        }
    if not torch.cuda.is_available():
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": "CUDA required for real KV curve",
            "points": [],
        }
    model_path = Path(model_dir)
    if not model_path.is_dir():
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": f"model_dir missing: {model_path}",
            "points": [],
        }
    tok_list = sorted({int(t) for t in tokens if int(t) > 0})
    if not tok_list:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": "no positive token lengths",
            "points": [],
        }
    max_t = int(tok_list[-1])
    cache = cache_dir or (Path.cwd() / ".gdx_kv_curve_cache")
    try:
        text = _load_real_text(text_file=text_file, cache_dir=cache)
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": f"real text load failed: {exc}",
            "points": [],
        }

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    # Cap the text we feed the tokenizer; repeat after encode if still short.
    # (~4 chars/token is enough headroom for English prose.)
    chunk = text[: max(max_t * 8, 50_000)]
    encoded = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"]
    if int(ids.shape[-1]) < max_t:
        # Repeat the real document until we cover max_t (still real tokens).
        reps = (max_t // int(ids.shape[-1])) + 2
        ids = ids.repeat(1, reps)
    ids = ids[:, :max_t].contiguous().cuda()

    # Prefer checkpoint dtype (Hub FP8 stays float8). Forcing bf16 breaks
    # compressed-tensors weight/activation pairs.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.eval().cuda()
    # Prefer the inner stack (no lm_head). Large-vocab CausalLM forwards
    # materialize logits [B,T,V] and OOM on 12B-class models at T=8k.
    forward_mod = model
    for attr in ("language_model", "model", "transformer"):
        inner = getattr(model, attr, None)
        if inner is not None and callable(getattr(inner, "forward", None)):
            forward_mod = inner
            break
    with torch.inference_mode():
        try:
            out = forward_mod(input_ids=ids, use_cache=True)
        except RuntimeError:
            # Some multimodal stacks need the outer CausalLM entry.
            out = model(input_ids=ids, use_cache=True)
    pkv = getattr(out, "past_key_values", None)
    if pkv is None and isinstance(out, (tuple, list)) and len(out) >= 2:
        pkv = out[1]
    # Drop weights before cold packing; only KV pages are needed next.
    del model, forward_mod, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if pkv is None:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": "model returned no past_key_values",
            "points": [],
        }
    # HF Cache: prefer DynamicCache.layers; legacy (k,v) tuples as fallback.
    layers_kv: list[tuple[Any, Any]] = []
    if hasattr(pkv, "layers") and pkv.layers:
        for layer in pkv.layers:
            k_t = getattr(layer, "keys", None)
            v_t = getattr(layer, "values", None)
            if k_t is None or v_t is None:
                return {
                    "schema": SCHEMA,
                    "ok": False,
                    "error": "DynamicCache layer missing keys/values",
                    "points": [],
                }
            layers_kv.append((k_t, v_t))
    elif hasattr(pkv, "to_legacy_cache"):
        legacy = pkv.to_legacy_cache()
        for entry in legacy:
            layers_kv.append((entry[0], entry[1]))
    else:
        for entry in list(pkv):
            layers_kv.append((entry[0], entry[1]))
    n_layers = len(layers_kv)
    k0 = layers_kv[0][0]
    # [B, H, T, D]
    kv_heads = int(k0.shape[1])
    head_dim = int(k0.shape[-1])
    fa_bpv = _fa_bpv(alphabet)
    a = alphabet.strip().lower()
    points: list[dict[str, Any]] = []
    for T in tok_list:
        cold_t = max(0, int(T) - int(live_window))
        fa_hot_t = min(int(T), int(live_window))
        fa_hot = int((fa_hot_t * n_layers * 2 * kv_heads * head_dim * fa_bpv + 7) // 8)
        if cold_t < 1:
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
        gdx_cold = 0
        bpv_acc = 0.0
        n_pack = 0
        for k_t, v_t in layers_kv:
            k_page = _kv_layer_to_lthd(k_t[:, :, :cold_t, :])
            v_page = _kv_layer_to_lthd(v_t[:, :, :cold_t, :])
            for page in (k_page, v_page):
                codes = _codes_for_alphabet(page, a)
                packed, bpv = _pack_codes_nbytes(a, codes)
                gdx_cold += int(packed)
                bpv_acc += float(bpv)
                n_pack += 1
        fa_cold = int(
            (cold_t * n_layers * 2 * kv_heads * head_dim * fa_bpv + 7) // 8
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
                "payload_bpv": float(bpv_acc / max(n_pack, 1)),
            }
        )
    # Free GPU before plot (weights already dropped after forward).
    del pkv, layers_kv, ids
    torch.cuda.empty_cache()

    cold_pts = [p for p in points if int(p["cold_t"]) > 0]
    err_msg: str | None = None
    ok = False
    if not cold_pts:
        err_msg = "GDX cold bytes curve has no cold-T points"
    else:
        last = cold_pts[-1]
        if float(last["save_vs_fa"]) <= 0.0:
            err_msg = "GDX cold+hot bytes did not beat FA twin at max T"
        elif float(last.get("payload_bpv") or fa_bpv) >= float(fa_bpv):
            err_msg = (
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
                err_msg = (
                    "GDX cold nbytes did not grow with T "
                    f"(cold_t x{t_ratio:.1f}, bytes x{b_ratio:.2f})"
                )
            else:
                ok = True
        else:
            ok = True
    return {
        "schema": SCHEMA,
        "ok": ok,
        "source": "real_model_kv",
        "text": "gutenberg_1342" if text_file is None else str(text_file),
        # Basename only: public tree must not record private checkout paths.
        "model_dir": model_path.name,
        "alphabet": a if a not in ("fp8_e4m3",) else "fp8",
        "fa_bpv": fa_bpv,
        "live_window": int(live_window),
        "layers": int(n_layers),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "points": points,
        "error": err_msg,
    }


def render_png(
    doc: dict[str, Any],
    out: Path,
    *,
    gpu: str = "",
    gpu_count: int | None = None,
    model: str = "",
) -> None:
    """2x1: KV-only (top) and weights+KV (bottom). No GDX in legend labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from needle_haystack import figure_tagline

    pts = list(doc.get("points") or [])
    if not pts:
        raise RuntimeError("no points to plot")
    xs = [int(p["T"]) for p in pts]
    gib = float(1024**3)
    fa = [float(p["fa_twin_nbytes"]) / gib for p in pts]
    isiro_kv = [float(p["gdx_total_nbytes"]) / gib for p in pts]
    b_w = doc.get("baseline_weight_bytes")
    t_w = doc.get("isiro_weight_bytes")
    has_weights = (
        isinstance(b_w, (int, float))
        and isinstance(t_w, (int, float))
        and float(b_w) > 0
        and float(t_w) > 0
    )
    tagline = figure_tagline(gpu=gpu, gpu_count=gpu_count, model=model)
    if has_weights:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(7.2, 7.6), dpi=140, sharex=False
        )
        ax0.plot(xs, fa, color="#c0392b", lw=1.9, label="Baseline (KV)")
        ax0.plot(xs, isiro_kv, color="#1f77b4", lw=2.2, label="Isiro (KV)")
        ax0.set_title("KV vs context", fontsize=12, pad=8)
        ax0.set_xlabel("Context tokens T")
        ax0.set_ylabel("KV (GiB)")
        ax0.set_ylim(bottom=0.0)
        ax0.tick_params(axis="x", labelbottom=True)
        ax0.grid(True, alpha=0.28)
        ax0.legend(frameon=False, loc="upper left")
        b_w_gib = float(b_w) / gib
        t_w_gib = float(t_w) / gib
        ax1.plot(
            xs,
            [b_w_gib + y for y in fa],
            color="#c0392b",
            lw=1.9,
            label="Baseline (weights + KV)",
        )
        ax1.plot(
            xs,
            [t_w_gib + y for y in isiro_kv],
            color="#1f77b4",
            lw=2.2,
            label="Isiro (weights + KV)",
        )
        ax1.set_title("Weights + KV vs context", fontsize=12, pad=8)
        ax1.set_xlabel("Context tokens T")
        ax1.set_ylabel("Weights + KV (GiB)")
        ax1.set_ylim(bottom=0.0)
        ax1.tick_params(axis="x", labelbottom=True)
        ax1.grid(True, alpha=0.28)
        ax1.legend(frameon=False, loc="lower left")
        # Tagline under both panels so it cannot overlap the top subtitle.
        bottom = 0.10
        if tagline:
            fig.text(
                0.5,
                0.02,
                tagline,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#444444",
            )
            bottom = 0.12
        fig.subplots_adjust(
            left=0.12, right=0.88, bottom=bottom, top=0.96, hspace=0.40
        )
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
        ax.plot(xs, fa, color="#c0392b", lw=1.9, label="Baseline (KV)")
        ax.plot(xs, isiro_kv, color="#1f77b4", lw=2.2, label="Isiro (KV)")
        ax.set_title("KV vs context", fontsize=12, pad=8)
        ax.set_xlabel("Context tokens T")
        ax.set_ylabel("KV (GiB)")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=False)
        bottom = 0.14
        if tagline:
            fig.text(
                0.5,
                0.02,
                tagline,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#444444",
            )
            bottom = 0.18
        fig.subplots_adjust(left=0.12, right=0.88, bottom=bottom, top=0.90)
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
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            os.environ.get("BASELINE_MODEL_DIR")
            or os.environ.get("ISIRO_BENCH_MODEL_DIR")
            or ""
        ),
        help="HF model dir for real KV capture (required)",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Optional real long text file (default: Gutenberg #1342 cache)",
    )
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
    parser.add_argument(
        "--baseline-weight-bytes",
        type=int,
        default=int(os.environ.get("ISIRO_BENCH_BASELINE_WEIGHT_BYTES") or "0"),
        help="Loaded-weight GPU bytes for Baseline (weights+KV panel)",
    )
    parser.add_argument(
        "--isiro-weight-bytes",
        type=int,
        default=int(os.environ.get("ISIRO_BENCH_ISIRO_WEIGHT_BYTES") or "0"),
        help="Loaded-weight GPU bytes for Isiro (weights+KV panel)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary.json; reads footprint model_loading_bytes_reported",
    )
    args = parser.parse_args(argv)
    tokens = [int(x) for x in str(args.tokens).split(",") if x.strip()]
    if not args.model_dir or not Path(args.model_dir).is_dir():
        print(
            "gdx_kv_bytes_curve: FAIL error=--model-dir required (real KV)",
            file=sys.stderr,
        )
        return 2
    doc = measure_curve(
        str(args.alphabet),
        tokens=tokens,
        model_dir=Path(args.model_dir),
        text_file=args.text_file,
        live_window=int(args.live_window),
        cache_dir=args.out_json.parent / ".gdx_kv_curve_cache",
    )
    b_w = int(args.baseline_weight_bytes or 0)
    t_w = int(args.isiro_weight_bytes or 0)
    if args.summary_json is not None and args.summary_json.is_file():
        try:
            summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
            fp = summary.get("footprint") or {}
            b_fp = (fp.get("baseline") or {}).get("model_loading_bytes_reported")
            t_fp = (fp.get("tic") or {}).get("model_loading_bytes_reported")
            if isinstance(b_fp, (int, float)) and int(b_fp) > 0:
                b_w = int(b_fp)
            if isinstance(t_fp, (int, float)) and int(t_fp) > 0:
                t_w = int(t_fp)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if b_w > 0 and t_w > 0:
        doc["baseline_weight_bytes"] = b_w
        doc["isiro_weight_bytes"] = t_w
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
            f"source={doc.get('source')} points={len(doc.get('points') or [])} "
            f"png={args.out_png}",
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
