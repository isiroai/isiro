#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot cold GDX KV bytes vs FA twin across context length T.

Uses real KV from a real token stream (Project Gutenberg text). Hot window
stays FA; cold prefix is GDX o1. Writes PNG + JSON for the public report.

KV sources (fail-closed):
- ``bf16`` alphabet: HF forward on a dense/bf16-capable weight tree (real bf16
  bit patterns; matches product bf16 deposit bitcast).
- ``fp8`` / ``fp4`` alphabets: vendor FA page bytes from
  ``gdx_kv_curve_vendor_pages.py`` (same ``kv_cache_dtype`` as serve). Product
  GDX compresses those pages only; there is no bf16→e4m3/e2m1 cast.

FA twin and GDX cold lengths follow vendor layer caps from config
(``layer_types`` + ``sliding_window``): sliding layers use
``min(T, window)``, full layers use ``T``. That matches vLLM hybrid
AttentionSpec geometry better than naive ``T * all_layers``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "isiro-benchmark-gdx-kv-bytes-curve-v3"

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


def _is_packed_fa_layer_type(name: str) -> bool:
    """Dense / sliding FA only (exclude KDA, Mamba, MLA, sparse)."""
    s = str(name).lower()
    if "linear" in s or "mamba" in s or s in ("gdn", "ssm", "kda"):
        return False
    if "sparse" in s or "mla" in s:
        return False
    if "full" in s or "sliding" in s or s in ("attention", "full_attention"):
        return True
    return False


def _fa_dump_layer_caps(
    model_path: Path, layer_caps: list[int | None]
) -> list[int | None]:
    """Caps matching a vendor packed-FA page dump (skip KDA/Mamba/MLA/sparse)."""
    cfg_path = model_path / "config.json"
    if not cfg_path.is_file():
        return []
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    layer_types = tc.get("layer_types") or cfg.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        return []
    fa_caps: list[int | None] = []
    for i, lt in enumerate(layer_types):
        if not _is_packed_fa_layer_type(str(lt)):
            continue
        if i < len(layer_caps):
            fa_caps.append(layer_caps[i])
        else:
            fa_caps.append(None)
    return fa_caps


def _layer_caps_from_config(model_path: Path) -> tuple[list[int | None], dict[str, Any]]:
    """Per-layer vendor KV length cap from Hub config.

    ``None`` means full-sequence KV (no sliding cap). An int is the sliding
    window (vendor keeps at most that many tokens for the layer).
    """
    cfg_path = model_path / "config.json"
    meta: dict[str, Any] = {
        "fa_twin_mode": "full_sequence",
        "sliding_window": None,
        "n_sliding": 0,
        "n_full": 0,
    }
    if not cfg_path.is_file():
        return [], meta
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], meta
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    layer_types = tc.get("layer_types") or cfg.get("layer_types")
    sliding = tc.get("sliding_window")
    if sliding is None:
        sliding = cfg.get("sliding_window")
    n_layers = int(tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    sw: int | None
    if isinstance(sliding, (int, float)) and int(sliding) > 0:
        sw = int(sliding)
    elif isinstance(sliding, list) and sliding:
        sw = int(sliding[0]) if int(sliding[0]) > 0 else None
    else:
        sw = None
    if not isinstance(layer_types, list) or not layer_types:
        # Global sliding_window with no layer_types (e.g. CSA2 Flash): every
        # layer is window-capped. Dense GQA with no SW keeps full sequence.
        if sw is not None and n_layers > 0:
            caps = [sw] * n_layers
            meta = {
                "fa_twin_mode": "vendor_layer_caps",
                "sliding_window": sw,
                "n_sliding": n_layers,
                "n_full": 0,
            }
            return caps, meta
        caps = [None] * max(n_layers, 0)
        meta["n_full"] = len(caps)
        return caps, meta
    caps = []
    n_slide = 0
    n_full = 0
    for i, lt in enumerate(layer_types):
        name = str(lt).lower()
        if sw is not None and "sliding" in name:
            # Optional per-index window when config lists them.
            if isinstance(sliding, list) and i < len(sliding) and int(sliding[i]) > 0:
                caps.append(int(sliding[i]))
            else:
                caps.append(sw)
            n_slide += 1
        else:
            caps.append(None)
            n_full += 1
    if n_layers > len(caps):
        caps.extend([None] * (n_layers - len(caps)))
        n_full += n_layers - len(layer_types)
    meta = {
        "fa_twin_mode": "vendor_layer_caps" if n_slide else "full_sequence",
        "sliding_window": sw,
        "n_sliding": n_slide,
        "n_full": n_full,
    }
    return caps, meta


def _vendor_eff_t(seq_t: int, cap: int | None) -> int:
    """Tokens a vendor FA layer retains at logical sequence length ``seq_t``."""
    t = max(0, int(seq_t))
    if cap is None:
        return t
    return min(t, max(0, int(cap)))


def _layer_fa_nbytes(
    *,
    eff_t: int,
    kv_heads: int,
    head_dim: int,
    fa_bpv: float,
) -> int:
    """K+V FA bytes for one layer at ``eff_t`` tokens."""
    return int((int(eff_t) * 2 * int(kv_heads) * int(head_dim) * float(fa_bpv) + 7) // 8)


def _load_real_text(*, text_file: Path | None, cache_dir: Path) -> str:
    """Load a real long document (fixture / cache / Gutenberg download)."""
    if text_file is not None and text_file.is_file():
        return text_file.read_text(encoding="utf-8", errors="replace")
    # Prefer benchmarks/fixtures (repo layout); keep scripts/fixtures as alias.
    here = Path(__file__).resolve().parent
    for fixture in (
        here.parent / "fixtures" / _REAL_TEXT_NAME,
        here / "fixtures" / _REAL_TEXT_NAME,
    ):
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


def _require_uint8_codes(codes, *, where: str):
    """Fail-closed: fp8/fp4 pack path accepts vendor uint8 codes only."""
    import numpy as np

    arr = np.asarray(codes)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"{where}: expected vendor uint8 FA codes, got dtype={arr.dtype}"
        )
    return np.ascontiguousarray(arr)


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

        u8 = _require_uint8_codes(codes, where="pack_fp8")
        ra = encode_gdx_e4m3_msb_o1(u8)
        n = int(ra.n_values)
        packed = int(ra.streams.nbytes + ra.offsets.nbytes + ra.freq_blob.nbytes)
        bpv = float(packed * 8.0 / max(n, 1))
        return packed, bpv
    if a in ("fp4", "nvfp4", "e2m1"):
        from isiro_runtime.ops.tic.native_compute.gdx.atoms.e2m1_o1 import (
            encode_gdx_e2m1_o1,
        )

        u8 = _require_uint8_codes(codes, where="pack_fp4")
        ra = encode_gdx_e2m1_o1(u8)
        n = int(ra.n_values)
        packed = int(ra.streams.nbytes + ra.offsets.nbytes + ra.freq_blob.nbytes)
        bpv = float(packed * 8.0 / max(n, 1))
        return packed, bpv
    raise ValueError(f"unsupported alphabet {alphabet!r}")


def _kv_layer_to_lthd(k_or_v) -> Any:
    """HF cache [B,H,T,D] → numpy [1,T,H,D] uint16 (bf16 bits)."""
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
    raise RuntimeError(
        f"bf16 curve fail-closed: expected bf16/fp16 HF KV, got dtype={t.dtype}"
    )


def _gate_curve_points(
    points: list[dict[str, Any]],
    *,
    fa_bpv: float,
) -> tuple[bool, str | None]:
    cold_pts = [p for p in points if int(p["cold_t"]) > 0]
    if not cold_pts:
        return False, "GDX cold bytes curve has no cold-T points"
    last = cold_pts[-1]
    if float(last["save_vs_fa"]) <= 0.0:
        return False, "GDX cold+hot bytes did not beat FA twin at max T"
    if float(last.get("payload_bpv") or fa_bpv) >= float(fa_bpv):
        return False, (
            "GDX payload bpv at max T did not beat FA twin "
            f"({last.get('payload_bpv')} >= {fa_bpv})"
        )
    if len(cold_pts) >= 2:
        first = cold_pts[0]
        t_ratio = float(last["cold_t"]) / float(max(int(first["cold_t"]), 1))
        b_ratio = float(last["gdx_cold_nbytes"]) / float(
            max(int(first["gdx_cold_nbytes"]), 1)
        )
        if t_ratio >= 4.0 and b_ratio < 2.0:
            return False, (
                "GDX cold nbytes did not grow with T "
                f"(cold_t x{t_ratio:.1f}, bytes x{b_ratio:.2f})"
            )
    return True, None


def measure_curve_from_vendor_pages(
    alphabet: str,
    *,
    tokens: list[int],
    vendor_pages: str | Path,
    model_dir: str | Path | None = None,
    live_window: int = _DEFAULT_W,
    kv_cache_dtype: str = "",
) -> dict[str, Any]:
    """Curve from vendor FA page dump (fp8/fp4 SSOT)."""
    import numpy as np

    err = _ensure_runtime_import()
    if err is not None:
        return {"schema": SCHEMA, "ok": False, "error": err, "points": []}
    a = alphabet.strip().lower()
    if a not in ("fp8", "fp8_e4m3", "fp4", "nvfp4", "e2m1"):
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": f"vendor pages require fp8/fp4 alphabet, got {alphabet!r}",
            "points": [],
        }
    path = Path(vendor_pages)
    if not path.is_file():
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": f"vendor pages missing: {path}",
            "points": [],
        }
    try:
        blob = np.load(path, allow_pickle=False)
        k_all = np.asarray(blob["k"])
        v_all = np.asarray(blob["v"])
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": f"vendor pages load failed: {exc}",
            "points": [],
        }
    if k_all.dtype != np.uint8 or v_all.dtype != np.uint8:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": (
                "vendor pages fail-closed: expected uint8 FA codes, "
                f"got k={k_all.dtype} v={v_all.dtype}"
            ),
            "points": [],
        }
    if k_all.ndim != 4 or v_all.ndim != 4 or k_all.shape != v_all.shape:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": f"vendor pages shape invalid k={k_all.shape} v={v_all.shape}",
            "points": [],
        }
    n_layers, max_t_pages, kv_heads, head_dim = (int(x) for x in k_all.shape)
    tok_list = sorted({int(t) for t in tokens if int(t) > 0})
    if not tok_list:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": "no positive token lengths",
            "points": [],
        }
    if int(tok_list[-1]) > max_t_pages:
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": (
                f"tokens max {tok_list[-1]} exceeds vendor pages T={max_t_pages}"
            ),
            "points": [],
        }
    model_path = Path(model_dir) if model_dir else path.parent
    layer_caps, cap_meta = _layer_caps_from_config(model_path)
    # Vendor FA dumps omit Mamba/GDN / linear_attn slabs. Keep only FA-capable
    # config layers (full / sliding), or fall back to full-sequence per page.
    fa_caps = _fa_dump_layer_caps(model_path, layer_caps)
    if len(fa_caps) == n_layers:
        layer_caps = fa_caps
        cap_meta = {
            **cap_meta,
            "fa_twin_mode": "vendor_fa_pages",
            "n_full": sum(1 for c in layer_caps if c is None),
            "n_sliding": sum(1 for c in layer_caps if c is not None),
        }
    elif len(layer_caps) < n_layers:
        layer_caps = list(layer_caps) + [None] * (n_layers - len(layer_caps))
    elif len(layer_caps) > n_layers:
        layer_caps = [None] * n_layers
        cap_meta = {
            **cap_meta,
            "fa_twin_mode": "vendor_fa_pages",
            "n_full": n_layers,
            "n_sliding": 0,
        }
    fa_bpv = _fa_bpv(a)
    points: list[dict[str, Any]] = []
    for T in tok_list:
        fa_hot = 0
        fa_cold = 0
        gdx_cold = 0
        cold_tokens = 0
        bpv_acc = 0.0
        n_pack = 0
        for li in range(n_layers):
            cap = layer_caps[li] if li < len(layer_caps) else None
            eff_t = _vendor_eff_t(int(T), cap)
            if eff_t < 1:
                continue
            # Vendor dump is a full-sequence prefix; sliding layers keep the
            # last eff_t tokens of that prefix (same as HF slice path).
            t_end = int(T)
            t_start = t_end - eff_t
            k_vis = k_all[li, t_start:t_end, :, :]
            v_vis = v_all[li, t_start:t_end, :, :]
            hot_t = min(eff_t, int(live_window))
            cold_t_layer = max(0, eff_t - int(live_window))
            fa_hot += _layer_fa_nbytes(
                eff_t=hot_t, kv_heads=kv_heads, head_dim=head_dim, fa_bpv=fa_bpv
            )
            fa_cold += _layer_fa_nbytes(
                eff_t=cold_t_layer,
                kv_heads=kv_heads,
                head_dim=head_dim,
                fa_bpv=fa_bpv,
            )
            cold_tokens += int(cold_t_layer)
            if cold_t_layer < 1:
                continue
            for page in (k_vis[:cold_t_layer], v_vis[:cold_t_layer]):
                codes = _require_uint8_codes(
                    page.reshape(1, cold_t_layer, kv_heads, head_dim),
                    where="vendor_page",
                )
                packed, bpv = _pack_codes_nbytes(a, codes)
                gdx_cold += int(packed)
                bpv_acc += float(bpv)
                n_pack += 1
        fa_all = int(fa_cold + fa_hot)
        gdx_total = int(gdx_cold + fa_hot)
        if cold_tokens < 1:
            points.append(
                {
                    "T": int(T),
                    "cold_t": 0,
                    "fa_twin_nbytes": fa_all,
                    "gdx_cold_nbytes": 0,
                    "gdx_total_nbytes": fa_all,
                    "save_vs_fa": 0.0,
                    "payload_bpv": None,
                }
            )
            continue
        save = 1.0 - (float(gdx_total) / float(max(fa_all, 1)))
        points.append(
            {
                "T": int(T),
                "cold_t": int(cold_tokens),
                "fa_twin_nbytes": int(fa_all),
                "gdx_cold_nbytes": int(gdx_cold),
                "gdx_total_nbytes": int(gdx_total),
                "save_vs_fa": float(save),
                "payload_bpv": float(bpv_acc / max(n_pack, 1)),
            }
        )
    ok, err_msg = _gate_curve_points(points, fa_bpv=fa_bpv)
    alphabet_out = "fp8" if a in ("fp8_e4m3",) else (
        "fp4" if a in ("nvfp4", "e2m1") else a
    )
    return {
        "schema": SCHEMA,
        "ok": ok,
        "source": "vendor_fa_pages",
        "kv_source": "vendor_fa_pages",
        "kv_cache_dtype": str(kv_cache_dtype or "").strip() or None,
        "text": "gutenberg_1342",
        "model_dir": model_path.name if model_path.is_dir() else path.stem,
        "alphabet": alphabet_out,
        "fa_bpv": fa_bpv,
        "live_window": int(live_window),
        "layers": int(n_layers),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "fa_twin_mode": cap_meta.get("fa_twin_mode"),
        "sliding_window": cap_meta.get("sliding_window"),
        "n_sliding_layers": cap_meta.get("n_sliding"),
        "n_full_layers": cap_meta.get("n_full"),
        "points": points,
        "error": err_msg,
    }


def measure_curve(
    alphabet: str,
    *,
    tokens: list[int],
    model_dir: str | Path,
    text_file: Path | None = None,
    live_window: int = _DEFAULT_W,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Real-KV curve (bf16 alphabet): one HF forward, slice per T, pack cold."""
    a0 = alphabet.strip().lower()
    if a0 in ("fp8", "fp8_e4m3", "fp4", "nvfp4", "e2m1"):
        return {
            "schema": SCHEMA,
            "ok": False,
            "error": (
                f"alphabet {alphabet!r} requires --vendor-pages "
                "(vendor FA e4m3/nvfp4 storage; no bf16 cast)"
            ),
            "points": [],
        }
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
    layer_caps, cap_meta = _layer_caps_from_config(model_path)
    if len(layer_caps) < n_layers:
        layer_caps = list(layer_caps) + [None] * (n_layers - len(layer_caps))
    elif len(layer_caps) > n_layers:
        layer_caps = layer_caps[:n_layers]
    points: list[dict[str, Any]] = []
    for T in tok_list:
        fa_hot = 0
        fa_cold = 0
        gdx_cold = 0
        cold_tokens = 0
        bpv_acc = 0.0
        n_pack = 0
        for li, (k_t, v_t) in enumerate(layers_kv):
            cap = layer_caps[li] if li < len(layer_caps) else None
            theo_eff = _vendor_eff_t(int(T), cap)
            cache_t = int(k_t.shape[2])
            # From a max-T forward: prefer the logical prefix [:T]. Sliding
            # layers keep the last theo_eff tokens of that prefix; full
            # layers keep the whole prefix. If HF already windowed the
            # layer (cache_t < T), fall back to the tail of what remains.
            if cache_t >= int(T):
                t_end = int(T)
                t_start = t_end - theo_eff
                k_vis = k_t[:, :, t_start:t_end, :]
                v_vis = v_t[:, :, t_start:t_end, :]
                eff_t = theo_eff
            else:
                eff_t = min(theo_eff, cache_t)
                if eff_t < 1:
                    continue
                k_vis = k_t[:, :, cache_t - eff_t : cache_t, :]
                v_vis = v_t[:, :, cache_t - eff_t : cache_t, :]
            if eff_t < 1:
                continue
            lh = int(k_vis.shape[1])
            ld = int(k_vis.shape[-1])
            hot_t = min(eff_t, int(live_window))
            cold_t_layer = max(0, eff_t - int(live_window))
            fa_hot += _layer_fa_nbytes(
                eff_t=hot_t, kv_heads=lh, head_dim=ld, fa_bpv=fa_bpv
            )
            fa_cold += _layer_fa_nbytes(
                eff_t=cold_t_layer, kv_heads=lh, head_dim=ld, fa_bpv=fa_bpv
            )
            cold_tokens += int(cold_t_layer)
            if cold_t_layer < 1:
                continue
            # Cold = older side of the vendor-visible window.
            k_page = _kv_layer_to_lthd(k_vis[:, :, :cold_t_layer, :])
            v_page = _kv_layer_to_lthd(v_vis[:, :, :cold_t_layer, :])
            for page in (k_page, v_page):
                packed, bpv = _pack_codes_nbytes(a, page)
                gdx_cold += int(packed)
                bpv_acc += float(bpv)
                n_pack += 1
        fa_all = int(fa_cold + fa_hot)
        gdx_total = int(gdx_cold + fa_hot)
        if cold_tokens < 1:
            points.append(
                {
                    "T": int(T),
                    "cold_t": 0,
                    "fa_twin_nbytes": fa_all,
                    "gdx_cold_nbytes": 0,
                    "gdx_total_nbytes": fa_all,
                    "save_vs_fa": 0.0,
                    "payload_bpv": None,
                }
            )
            continue
        save = 1.0 - (float(gdx_total) / float(max(fa_all, 1)))
        points.append(
            {
                "T": int(T),
                "cold_t": int(cold_tokens),
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

    ok, err_msg = _gate_curve_points(points, fa_bpv=fa_bpv)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "source": "real_model_kv",
        "kv_source": "hf_bf16_kv",
        # Basename only: public tree must not record private checkout paths.
        "text": (
            "gutenberg_1342"
            if text_file is None
            else Path(text_file).name
        ),
        "model_dir": model_path.name,
        "alphabet": a if a not in ("fp8_e4m3",) else "fp8",
        "fa_bpv": fa_bpv,
        "live_window": int(live_window),
        "layers": int(n_layers),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "fa_twin_mode": cap_meta.get("fa_twin_mode"),
        "sliding_window": cap_meta.get("sliding_window"),
        "n_sliding_layers": cap_meta.get("n_sliding"),
        "n_full_layers": cap_meta.get("n_full"),
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
    weight_precision: str = "",
    kv_precision: str = "",
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
    w_prec = str(weight_precision or doc.get("weight_precision") or "").strip()
    kv_prec = str(
        kv_precision
        or doc.get("kv_precision")
        or doc.get("alphabet")
        or ""
    ).strip()
    tagline = figure_tagline(
        gpu=gpu,
        gpu_count=gpu_count,
        model=model,
        weight_precision=w_prec,
        kv_precision=kv_prec,
    )
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
    parser.add_argument(
        "--weight-precision",
        default=os.environ.get("PRECISION")
        or os.environ.get("ISIRO_BENCH_PRECISION")
        or "",
        help="Weight precision for the figure tagline (e.g. fp8, bf16)",
    )
    parser.add_argument(
        "--kv-precision",
        default="",
        help="KV precision for the figure tagline (default: --alphabet)",
    )
    parser.add_argument(
        "--vendor-pages",
        type=Path,
        default=None,
        help="npz from gdx_kv_curve_vendor_pages.py (required for fp8/fp4)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=os.environ.get("SERVE_KV_CACHE_DTYPE") or "",
        help="Recorded vendor kv_cache_dtype (fp8 / nvfp4)",
    )
    args = parser.parse_args(argv)
    tokens = [int(x) for x in str(args.tokens).split(",") if x.strip()]
    alphabet = str(args.alphabet)
    a_norm = alphabet.strip().lower()
    needs_vendor = a_norm in ("fp8", "fp8_e4m3", "fp4", "nvfp4", "e2m1")
    if needs_vendor:
        if args.vendor_pages is None or not Path(args.vendor_pages).is_file():
            print(
                "gdx_kv_bytes_curve: FAIL error=--vendor-pages required for "
                f"alphabet={alphabet!r} (vendor FA pages; no bf16 cast)",
                file=sys.stderr,
            )
            return 2
        model_for_caps = (
            Path(args.model_dir)
            if args.model_dir and Path(args.model_dir).is_dir()
            else Path(args.vendor_pages).parent
        )
        doc = measure_curve_from_vendor_pages(
            alphabet,
            tokens=tokens,
            vendor_pages=Path(args.vendor_pages),
            model_dir=model_for_caps,
            live_window=int(args.live_window),
            kv_cache_dtype=str(args.kv_cache_dtype or ""),
        )
    else:
        if not args.model_dir or not Path(args.model_dir).is_dir():
            print(
                "gdx_kv_bytes_curve: FAIL error=--model-dir required (real KV)",
                file=sys.stderr,
            )
            return 2
        doc = measure_curve(
            alphabet,
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
    w_prec = str(args.weight_precision or "").strip()
    kv_prec = str(args.kv_precision or args.alphabet or "").strip()
    if w_prec:
        doc["weight_precision"] = w_prec
    if kv_prec:
        doc["kv_precision"] = kv_prec
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
            weight_precision=w_prec,
            kv_precision=kv_prec,
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
