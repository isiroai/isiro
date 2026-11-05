#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-a-haystack grid: JSONL probe + heatmap (TIC vs baseline).

Protocol is multi-needle (Kamradt / RULER / EvalScope style).
Each panel is one KV cache size that started. X is context tokens. Y is needle depth %.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

NEEDLE = "The special magic number is 84932017."
NEEDLES: tuple[str, ...] = (
    NEEDLE,
    "The secret city code is Zebra-419.",
    "The hidden river name is Copperbrook.",
)
# Kamradt / RULER fragments: unique values, not wrapping sentences.
NEEDLE_KEYS: tuple[str, ...] = ("84932017", "Zebra-419", "Copperbrook")
NEEDLE_BINDER = "ISIRO-NIAH-QX7K29"
QUESTION = (
    "List the special magic number, the secret city code, and the hidden river name."
)
HAYSTACK_LINE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "We like to go to the park and sit on a bench. "
)

DEFAULT_DEPTHS = (0, 11, 22, 33, 44, 56, 67, 78, 89, 100)
CONTEXT_BIN_COUNT = 20
CONTEXT_BIN_MIN = 1000
LONG_CONTEXT_TOKENS = 1_000_000
# Table A folds the grid at this bin when the model window allows it.
STORY_CONTEXT_TOKENS = 16000
CHARS_PER_TOKEN = 4
# English hay is denser than 4 chars/token. Fitted hay uses this so an 8k
# bin plus chat template plus a short reply stays inside max_model_len.
HAY_CHARS_PER_TOKEN_FIT = 2.7
CHAT_MAX_TOKENS = 128
CHAT_SEED = 17
COMPLETION_STORE_CHARS = 4000
TOKEN_ID_BYTES = 4
TEMPLATE_SLACK_TOKENS = 160
# Haystack bin plus needle, question, chat template, and a short reply.
WINDOW_RESERVE_TOKENS = 1024
NEEDLE_COUNT = len(NEEDLES)
FIGURE_TITLE = (
    f"Fact retrieval by context length ({NEEDLE_COUNT}-needle haystack)"
)
VRAM_TITLE = "VRAM usage vs context"
VRAM_TAGLINE_AXES_GAP_PT = 16.0


def figure_tagline(
    rows: Iterable[dict[str, Any]] | None = None,
    *,
    gpu: str = "",
    gpu_count: int | None = None,
    model: str = "",
    weight_precision: str = "",
    kv_precision: str = "",
) -> str:
    """GPU box, model, and optional weight/KV precisions under the figure."""
    gpu_s = str(gpu or "").strip()
    model_s = str(model or "").strip()
    w_s = str(weight_precision or "").strip()
    kv_s = str(kv_precision or "").strip()
    count: int | None = int(gpu_count) if gpu_count else None
    if rows is not None:
        for item in rows:
            if not gpu_s:
                gpu_s = str(item.get("gpu_model") or "").strip()
            if count is None:
                raw_n = item.get("gpu_count")
                try:
                    if raw_n is not None and int(raw_n) > 0:
                        count = int(raw_n)
                except (TypeError, ValueError):
                    pass
            if not model_s:
                model_s = str(
                    item.get("model") or item.get("model_id") or ""
                ).strip()
            if not w_s:
                w_s = str(
                    item.get("weight_precision")
                    or item.get("precision")
                    or ""
                ).strip()
            if not kv_s:
                kv_s = str(
                    item.get("kv_precision")
                    or item.get("kv_cache_dtype")
                    or item.get("isiro_kv")
                    or ""
                ).strip()
            if gpu_s and model_s:
                break
    if gpu_s and count is not None and count > 0:
        gpu_s = f"{gpu_s} x {count}"
    parts = [gpu_s, model_s]
    if w_s:
        parts.append(f"weights {w_s}")
    if kv_s:
        parts.append(f"KV {kv_s}")
    return " | ".join(part for part in parts if part)


def format_gib_cell(size: float) -> str:
    """Two decimal GiB. Do not print 0.4882."""
    return f"{round(float(size), 2):.2f}"


def format_gib_list(sizes: Iterable[float]) -> str:
    vals = [
        f"{format_gib_cell(s)} GiB"
        for s in sorted({round(float(s), 2) for s in sizes})
    ]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]} or {vals[1]}"
    return f"{', '.join(vals[:-1])}, or {vals[-1]}"


def _start_fail_gibs(rows: Iterable[dict[str, Any]]) -> list[float]:
    return sorted(
        {
            round(float(item["kv_cache_gib"]), 4)
            for item in rows
            if item.get("start_fail") and "kv_cache_gib" in item
        }
    )


def needle_keys(needles: Iterable[str] | None = None) -> tuple[str, ...]:
    items = tuple(needles) if needles is not None else NEEDLES
    if items == NEEDLES:
        return NEEDLE_KEYS
    keys: list[str] = []
    for needle, mapped in zip(items, NEEDLE_KEYS):
        if needle in NEEDLES:
            keys.append(mapped)
            continue
        digits = "".join(ch for ch in needle if ch.isdigit())
        keys.append(digits or needle.strip())
    if len(keys) < len(items):
        keys.extend(items[len(keys) :])
    return tuple(keys)


def score_key_hit(completion: str, key: str) -> bool:
    """Case-insensitive substring of a unique fact value (Kamradt fragment)."""
    fragment = (key or "").strip()
    if not fragment:
        return False
    return fragment.casefold() in (completion or "").casefold()


def score_hit(completion: str, needle: str = NEEDLE) -> bool:
    """Full needle sentence, or the unique key / digits in the reply."""
    text = (completion or "").strip()
    if not text:
        return False
    if needle in text:
        return True
    key = ""
    if needle in NEEDLES:
        key = NEEDLE_KEYS[NEEDLES.index(needle)]
    elif needle in NEEDLE_KEYS:
        key = needle
    if key and score_key_hit(text, key):
        return True
    digits = "".join(ch for ch in needle if ch.isdigit())
    if digits and score_key_hit(text, digits):
        return True
    if not digits or digits not in text:
        return False
    compact = "".join(text.split()).rstrip(".")
    if compact == digits:
        return True
    remainder_text = text.replace(digits, " ")
    words = [w for w in remainder_text.split() if any(ch.isalnum() for ch in w)]
    return len(words) <= 12


def score_needles(
    completion: str, needles: Iterable[str] = NEEDLES
) -> float:
    items = tuple(needles) or (NEEDLE,)
    keys = needle_keys(items)
    hits = sum(1 for key in keys if score_key_hit(completion, key))
    if hits == 0:
        hits = sum(1 for needle in items if score_hit(completion, needle=needle))
        return round(hits / float(len(items)), 4)
    return round(hits / float(len(keys)), 4)


def long_needle_text(*, n_words: int = 128, needle: str = NEEDLE) -> str:
    """Unique multi-token needle. Harder to hallucinate than eight digits."""
    n = max(1, int(n_words))
    codes = " ".join(f"{NEEDLE_BINDER}-{i:04d}" for i in range(n))
    return f"{needle} Binder {codes}."


def prompt_token_budget(
    max_model_len: int,
    *,
    max_tokens: int = CHAT_MAX_TOKENS,
    slack: int = TEMPLATE_SLACK_TOKENS,
) -> int:
    """Hay tokens that still leave room for template and the reply."""
    return max(1, int(max_model_len) - max(0, int(max_tokens)) - max(0, int(slack)))


def advertised_max_model_len(source: Any) -> int | None:
    """Advertised context window from HF config.json. Not safetensors.

    Reads max_position_embeddings. If rope_scaling has
    original_max_position_embeddings and factor, take the larger of that
    product and max_position_embeddings. Unwraps text_config.
    """
    cfg: Mapping[str, Any] | None = None
    if isinstance(source, Mapping):
        cfg = dict(source)
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "config.json"
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        cfg = raw
    text = cfg.get("text_config")
    if isinstance(text, Mapping):
        merged = dict(cfg)
        merged.update(text)
        cfg = merged
    candidates: list[int] = []
    for key in ("max_position_embeddings", "max_sequence_length"):
        raw_m = cfg.get(key)
        try:
            if raw_m is not None and int(raw_m) > 0:
                candidates.append(int(raw_m))
        except (TypeError, ValueError):
            pass
    rope = cfg.get("rope_scaling")
    if isinstance(rope, Mapping):
        try:
            orig = rope.get("original_max_position_embeddings")
            factor = rope.get("factor")
            if orig is not None and factor is not None:
                scaled = int(round(float(orig) * float(factor)))
                if scaled > 0:
                    candidates.append(scaled)
        except (TypeError, ValueError):
            pass
    return max(candidates) if candidates else None


def extra_prompt_tokens(*parts: str) -> int:
    """Word count undercounts hyphenated binder codes. Use char density too."""
    n = 0
    for part in parts:
        text = str(part or "")
        n += max(len(text.split()), int(len(text) / HAY_CHARS_PER_TOKEN_FIT))
    return n


def hay_tokens_for_context(
    context_tokens: int,
    max_model_len: int | None = None,
    *,
    extra_tokens: int = 0,
) -> int:
    n = max(1, int(context_tokens))
    if max_model_len is not None and int(max_model_len) > 0:
        n = min(n, prompt_token_budget(int(max_model_len)))
    n = max(1, n - max(0, int(extra_tokens)))
    return n


def chars_for_context_tokens(
    context_tokens: int,
    max_model_len: int | None = None,
    *,
    extra_tokens: int = 0,
) -> int:
    n = hay_tokens_for_context(
        context_tokens, max_model_len, extra_tokens=extra_tokens
    )
    if max_model_len is not None and int(max_model_len) > 0:
        return max(1, int(n * HAY_CHARS_PER_TOKEN_FIT))
    return max(1, int(n) * CHARS_PER_TOKEN)


def parse_kv_cache_gib(log_text: str) -> float:
    """Measured KV pool in GiB from a serve log.

    Prefer the kv pool cap print (``KV pool cap:`` / ``KV pool:``). Fall
    back to vLLM Available / reserved.
    """
    import re

    kv_pool = re.search(r"KV pool(?: cap)?:\s*([0-9.]+)\s*GiB", log_text)
    if kv_pool:
        return float(kv_pool.group(1))
    match = re.search(
        r"Available KV cache memory:\s*([0-9.]+)\s*GiB",
        log_text,
    )
    if match:
        return float(match.group(1))
    reserved = re.search(
        r"reserved\s+([0-9.]+)\s*GiB memory for KV Cache",
        log_text,
    )
    if reserved:
        return float(reserved.group(1))
    # Legacy Isiro KV admit-clamp print (retired; prefer Available / reserved above).
    gdx_mib = re.search(r"Isiro KV FA live clamp:\s*([0-9.]+)\s*MiB", log_text)
    if gdx_mib:
        return float(gdx_mib.group(1)) / 1024.0
    pool = re.search(r"vendor_pool_bytes=(\d+)", log_text)
    if pool:
        return int(pool.group(1)) / float(1024**3)
    locked = re.search(r"kv_cache_memory_bytes['\"]?\s*[:=]\s*(\d+)", log_text)
    if locked:
        return int(locked.group(1)) / float(1024**3)
    raise ValueError("Available KV cache memory not found in serve log")


def parse_non_kv_cache_bytes(log_text: str) -> int | None:
    """vLLM Total non KV cache memory. None if the line is missing."""
    import re

    match = re.search(
        r"Total non KV cache memory:\s*([0-9.]+)\s*GiB",
        log_text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return gib_to_bytes(float(match.group(1)))


def resolve_non_kv_bytes(
    *,
    reported: int | None,
    process_bytes: int | None,
    pool_gib: float,
) -> tuple[int, str]:
    if reported is not None and int(reported) > 0:
        return int(reported), "reported"
    pool = gib_to_bytes(pool_gib)
    if process_bytes is not None and int(process_bytes) > pool:
        return int(process_bytes) - pool, "smi_minus_pool"
    return 0, "missing"


def locked_non_kv_bytes(rows: list[dict[str, Any]], *, panel: str, size: float) -> int:
    """Weights intercept. Per-bin nvidia-smi is not non-KV."""
    want = round(float(size), 4)
    alias = _panel_alias(panel) or panel
    items: list[dict[str, Any]] = []
    for item in rows:
        if _panel_alias(str(item.get("panel") or "")) != alias:
            continue
        if round(_kv_cache_gib(item), 4) != want:
            continue
        items.append(item)
    if not items:
        return 0
    reported = [
        int(r["non_kv_bytes_reported"])
        for r in items
        if str(r.get("non_kv_kind") or "") == "reported"
        and isinstance(r.get("non_kv_bytes_reported"), (int, float))
        and int(r["non_kv_bytes_reported"]) > 0
    ]
    if reported:
        return min(reported)
    smi = [
        int(r["non_kv_bytes_reported"])
        for r in items
        if isinstance(r.get("non_kv_bytes_reported"), (int, float))
        and int(r["non_kv_bytes_reported"]) > 0
    ]
    return min(smi) if smi else 0


def vram_in_use_y_bytes(
    item: dict[str, Any], *, non_kv_override: int | None = None
) -> int:
    """Weights/non-KV plus live pool blocks plus persist/ids. Not nvidia-smi."""
    kv = item.get("kv_bytes_in_use")
    kv_n = int(kv) if isinstance(kv, (int, float)) else 0
    if non_kv_override is not None:
        non_n = int(non_kv_override)
    else:
        non_kv = item.get("non_kv_bytes_reported")
        non_n = int(non_kv) if isinstance(non_kv, (int, float)) else 0
    return non_n + kv_n + persist_outside_bytes(item)


def persist_outside_bytes(item: dict[str, Any]) -> int:
    """TokenID / persist pages outside vendor usage_perc. Not dump-all KV."""
    raw_p = item.get("persist_page_bytes")
    if isinstance(raw_p, (int, float)) and int(raw_p) > 0:
        return int(raw_p)
    raw_a = item.get("archived_id_bytes")
    if isinstance(raw_a, (int, float)) and int(raw_a) > 0:
        return int(raw_a)
    alias = _panel_alias(str(item.get("panel") or ""))
    if alias != "tic":
        return 0
    t_tokens = int(item.get("context_tokens") or 0)
    w_tokens = int(item.get("kv_pool_tokens") or 0)
    return derived_archived_id_bytes(t_tokens, w_tokens)


def derived_archived_id_bytes(context_tokens: int, pool_tokens: int) -> int:
    """uint32 ids past the lock. Derived, not nvidia-smi."""
    extra = max(0, int(context_tokens) - max(0, int(pool_tokens)))
    return extra * TOKEN_ID_BYTES


def reserved_vram_y_bytes(item: dict[str, Any]) -> int:
    """Non-KV plus that serve's reserved KV pool."""
    non_kv = item.get("non_kv_bytes_reported")
    non_n = int(non_kv) if isinstance(non_kv, (int, float)) else 0
    return non_n + gib_to_bytes(_kv_cache_gib(item))


# Dest lock_seqs=1: (W + slack) slots. Serve prints GiB only.
_DEST_ONE_WINDOW = 512
_DEST_ONE_WINDOW_SLACK = 4096


def parse_kv_cache_tokens(log_text: str) -> int:
    """Kv pool cap slot count. Do not use vLLM GPU tokens when the cap printed."""
    import re

    pool = re.search(
        r"KV pool(?: cap)?:\s*[0-9.]+\s*GiB\s*\(([0-9,]+)\s*tokens",
        log_text,
    )
    if pool:
        return int(pool.group(1).replace(",", ""))
    if re.search(r"KV pool(?: cap)?:\s*[0-9.]+\s*GiB", log_text):
        # One-window pool. vLLM GPU token count is last-W-inflated.
        # estimated max_model_len is the last-W boot guard, not pool slots.
        w = _DEST_ONE_WINDOW
        env_w = re.search(r"ISIRO_KV_SET_WINDOW[=:\s]+(\d+)", log_text)
        if env_w:
            w = max(1, int(env_w.group(1)))
        return w + _DEST_ONE_WINDOW_SLACK
    gpu = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", log_text)
    if gpu:
        return int(gpu.group(1).replace(",", ""))
    est = re.search(r"estimated maximum model length is\s*([0-9,]+)", log_text)
    if est:
        return int(est.group(1).replace(",", ""))
    raise ValueError("GPU KV token count not found in serve log")


def needle_window_tokens(pool_tokens: int, cap: int | None = None) -> int:
    """Advertise a window the GPU pool can start. Do not grow the lock."""
    window = max(1, int(pool_tokens))
    if cap is not None and int(cap) > 0:
        window = min(window, int(cap))
    return window


def gib_to_bytes(kv_cache_gib: float) -> int:
    return int(round(float(kv_cache_gib) * (1024**3)))


def bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def derived_kv_in_use_bytes(
    context_tokens: int,
    pool_gib: float,
    pool_tokens: int,
) -> int:
    """One-request occupancy of the reserved pool. Derived, not a meter."""
    pool_bytes = gib_to_bytes(pool_gib)
    slots = max(1, int(pool_tokens))
    frac = min(1.0, max(0, int(context_tokens)) / float(slots))
    return int(round(frac * pool_bytes))


def kv_in_use_bytes(
    *,
    context_tokens: int,
    pool_gib: float,
    pool_tokens: int,
    usage_perc: float | None,
) -> tuple[int, str]:
    pool_bytes = gib_to_bytes(pool_gib)
    if usage_perc is not None and float(usage_perc) > 0.001:
        frac = min(1.0, max(0.0, float(usage_perc)))
        return int(round(frac * pool_bytes)), "metrics"
    if float(pool_gib) > 0 and int(pool_tokens) > 0:
        return (
            derived_kv_in_use_bytes(context_tokens, pool_gib, pool_tokens),
            "derived",
        )
    return 0, "derived"


def parse_prometheus_kv_usage_perc(text: str) -> float | None:
    """Peak 0-1 usage from vLLM /metrics text."""
    import re

    names = (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm_kv_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
    )
    best: float | None = None
    for name in names:
        pattern = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$"
        for match in re.finditer(pattern, text, re.MULTILINE):
            val = float(match.group(1))
            if val > 1.0:
                val = val / 100.0
            val = min(1.0, max(0.0, val))
            best = val if best is None else max(best, val)
    return best


def parse_prometheus_isiro_peak_bytes(text: str) -> int | None:
    """isiro_gpu_memory_peak_bytes from serve /metrics. Allocated, not smi."""
    import re

    pattern = r"^isiro_gpu_memory_peak_bytes(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$"
    best: int | None = None
    for match in re.finditer(pattern, text, re.MULTILINE):
        val = int(float(match.group(1)))
        if val > 0:
            best = val if best is None else max(best, val)
    return best


def context_length_bins(
    cap: int,
    *,
    n: int = CONTEXT_BIN_COUNT,
    lo: int = CONTEXT_BIN_MIN,
) -> tuple[int, ...]:
    """Kamradt / EvalScope linspace: n context lengths from lo to cap inclusive."""
    hi = max(1, int(cap))
    start = min(max(1, int(lo)), hi)
    count = max(2, int(n))
    if hi <= start:
        return (hi,)
    raw: list[int] = []
    span = float(hi - start)
    for i in range(count):
        t = i / (count - 1)
        raw.append(int(round(start + t * span)))
    ordered: list[int] = []
    for value in raw:
        clipped = min(hi, max(start, value))
        if not ordered or ordered[-1] != clipped:
            ordered.append(clipped)
    if ordered[0] != start:
        ordered = [start] + [v for v in ordered if v > start]
    if ordered[-1] != hi:
        ordered = [v for v in ordered if v < hi] + [hi]
    return tuple(ordered)


def log_context_bins(
    cap: int,
    *,
    n: int = CONTEXT_BIN_COUNT,
    lo: int = CONTEXT_BIN_MIN,
) -> tuple[int, ...]:
    return context_length_bins(cap, n=n, lo=lo)


MOCK_ADVERTISED_CONFIG = {"max_position_embeddings": 32768}


def mock_advertised_max() -> int:
    """Qwen2.5-7B-Instruct advertised window. Same parser as live HF config.json."""
    return int(advertised_max_model_len(MOCK_ADVERTISED_CONFIG) or 32768)


DEFAULT_CONTEXT_TOKENS = context_length_bins(mock_advertised_max())


def context_axis_tokens(
    rows: Iterable[dict[str, Any]],
    max_model_len: int | None = None,
) -> tuple[int, ...]:
    """Sorted context bins for both NIAH and VRAM x-axes."""
    cap = None
    if isinstance(max_model_len, (int, float)) and int(max_model_len) > 0:
        cap = int(max_model_len)
    found: set[int] = set()
    for item in rows:
        if item.get("start_fail"):
            continue
        tokens = int(item.get("context_tokens") or 0)
        if tokens <= 0:
            continue
        if cap is not None and tokens > cap:
            continue
        found.add(tokens)
    return tuple(sorted(found))


def context_bins_for_max_len(
    max_model_len: int,
    bins: Iterable[int] | None = None,
) -> tuple[int, ...]:
    cap = max(1, int(max_model_len))
    if bins is not None:
        kept = tuple(int(b) for b in bins if int(b) <= cap)
        return kept or (cap,)
    return context_length_bins(cap)


def rows_for_heatmap(
    rows: Iterable[dict[str, Any]],
    max_model_len: int | None = None,
) -> list[dict[str, Any]]:
    """Drop bins that cannot fit the window, or that never completed."""
    items = [item for item in rows if not item.get("start_fail")]
    if isinstance(max_model_len, (int, float)) and int(max_model_len) > 0:
        cap = int(max_model_len)
        items = [
            item
            for item in items
            if int(item.get("context_tokens") or 0) <= cap
        ]
    by_ctx: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if "context_tokens" not in item:
            continue
        by_ctx.setdefault(int(item["context_tokens"]), []).append(item)
    drop = {
        ctx
        for ctx, group in by_ctx.items()
        if group
        and all(not item.get("oom") for item in group)
        and all(
            (not item.get("response_completes", True))
            or item.get("refuse")
            for item in group
        )
    }
    if not drop:
        return items
    return [
        item
        for item in items
        if int(item.get("context_tokens") or 0) not in drop
    ]


def story_tokens_for_max_len(
    max_model_len: int,
    *,
    story: int = STORY_CONTEXT_TOKENS,
    bins: Iterable[int] = DEFAULT_CONTEXT_TOKENS,
) -> int:
    cap = max(1, int(max_model_len))
    target = min(int(story), cap)
    kept = context_bins_for_max_len(cap, bins)
    at_or_below = [b for b in kept if b <= target]
    return int(at_or_below[-1] if at_or_below else kept[0])


def tic_pool_smaller_than_fill(
    tic_gib: float,
    fill_gib: float,
    *,
    max_ratio: float = 0.85,
) -> bool:
    """True when product TIC KV is smaller than Baseline default-fill KV."""
    tic = float(tic_gib)
    fill = float(fill_gib)
    return fill > 0 and tic > 0 and tic < fill * float(max_ratio)


def niah_discover_steps() -> list[str]:
    return [
        "TIC product serve at NIAH_MAX_MODEL_LEN; scrape KV pool GiB (do not shrink the window); full grid (1 serve)",
        "Baseline default fill; scrape GPU KV cache size; full grid (1 serve)",
        "Refuse if TIC KV is not smaller than Baseline fill (any model)",
        "Baseline same size as TIC; record start-fail if serve does not start; no extra KV-size grids",
    ]


def haystack_text(*, approx_chars: int) -> str:
    n = max(1, int(approx_chars))
    block = HAYSTACK_LINE
    reps = max(1, (n + len(block) - 1) // len(block))
    return (block * reps)[:n]


def insert_needle(haystack: str, needle: str, depth_percent: float) -> str:
    depth = min(100.0, max(0.0, float(depth_percent)))
    if not haystack:
        return needle
    idx = int(round((len(haystack) * depth) / 100.0))
    idx = min(len(haystack), max(0, idx))
    return haystack[:idx] + " " + needle + " " + haystack[idx:]


def needle_spread_depths(n: int, depth_percent: float) -> tuple[float, ...]:
    """First needle at depth_percent; remaining even through the rest of the hay.

    Same interval as EvalScope / Kamradt: (100 - depth) / N.
    Depth 100 places every needle at the end.
    """
    count = max(1, int(n))
    depth = min(100.0, max(0.0, float(depth_percent)))
    if depth >= 100.0:
        return tuple(100.0 for _ in range(count))
    interval = (100.0 - depth) / float(count)
    return tuple(depth + i * interval for i in range(count))


def _snap_to_sentence(haystack: str, idx: int) -> int:
    idx = min(len(haystack), max(0, int(idx)))
    if idx <= 0 or idx >= len(haystack):
        return idx
    period = haystack.rfind(". ", 0, idx)
    if period >= 0:
        return period + 2
    return idx


def insert_needles(
    haystack: str,
    needles: Iterable[str],
    depth_percent: float,
) -> str:
    facts = [str(item) for item in needles]
    if not facts:
        return haystack
    depths = needle_spread_depths(len(facts), depth_percent)
    if not haystack or all(d >= 100.0 for d in depths):
        joined = " ".join(facts)
        if not haystack:
            return joined
        return haystack.rstrip() + " " + joined
    placements: list[tuple[int, str]] = []
    for fact, depth in zip(facts, depths):
        raw = int(round((len(haystack) * depth) / 100.0))
        placements.append((_snap_to_sentence(haystack, raw), fact))
    out = haystack
    for idx, fact in sorted(placements, key=lambda item: item[0], reverse=True):
        out = out[:idx] + " " + fact + " " + out[idx:]
    return out


def build_messages(
    *,
    depth_percent: float,
    approx_chars: int,
    needle: str | None = None,
    needles: Iterable[str] | None = None,
    question: str = QUESTION,
) -> list[dict[str, str]]:
    hay = haystack_text(approx_chars=approx_chars)
    facts = tuple(needles) if needles is not None else (
        (needle,) if needle else NEEDLES
    )
    body = insert_needles(hay, facts, depth_percent)
    return [
        {
            "role": "user",
            "content": body + "\n\n" + question,
        }
    ]


def _kv_cache_gib(item: dict[str, Any]) -> float:
    return float(item["kv_cache_gib"])


def _has_kv_cache_gib(item: dict[str, Any]) -> bool:
    return "kv_cache_gib" in item


def row(
    *,
    panel: str,
    kv_cache_gib: float,
    depth_percent: float,
    hit: bool,
    context_tokens: int | None = None,
    response_completes: bool = True,
    refuse: bool = False,
    oom: bool = False,
    extra: dict[str, Any] | None = None,
    score: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "panel": str(panel),
        "kv_cache_gib": float(kv_cache_gib),
        "depth_percent": float(depth_percent),
        "hit": bool(hit),
        "score": (
            round(float(score), 4)
            if score is not None
            else (1.0 if hit else 0.0)
        ),
        "response_completes": bool(response_completes),
        "refuse": bool(refuse),
        "oom": bool(oom),
    }
    if context_tokens is not None:
        out["context_tokens"] = int(context_tokens)
    if extra:
        out.update(extra)
    return out


def start_fail_rows(
    *,
    panel: str,
    kv_cache_gib: float,
    story_tokens: int = STORY_CONTEXT_TOKENS,
) -> list[dict[str, Any]]:
    """One story row when Baseline does not start at this lock. Not a recall miss."""
    return [
        row(
            panel=panel,
            kv_cache_gib=kv_cache_gib,
            depth_percent=50,
            hit=False,
            context_tokens=int(story_tokens),
            response_completes=False,
            refuse=True,
            extra={"start_fail": True},
        )
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )


def _chat_completion(
    url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 120.0,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": CHAT_MAX_TOKENS,
            "temperature": 0,
            "seed": CHAT_SEED,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        err = f"needle haystack chat HTTP {exc.code}"
        if detail:
            err = f"{err}: {detail}"
        raise RuntimeError(err) from exc
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("content") or "")


def scrape_gpu_process_bytes() -> int | None:
    """Sum nvidia-smi compute-app used memory. None if unavailable."""
    import subprocess

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    total_mib = 0.0
    for line in completed.stdout.splitlines():
        text = line.strip().replace(" MiB", "")
        if not text:
            continue
        try:
            total_mib += float(text)
        except ValueError:
            continue
    if total_mib <= 0:
        return None
    return int(total_mib * 1024 * 1024)


def scrape_gpu_hbm_bytes() -> int | None:
    """GPU total HBM. None if nvidia-smi is unavailable."""
    import subprocess

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    total_mib = 0.0
    for line in completed.stdout.splitlines():
        text = line.strip().replace(" MiB", "")
        if not text:
            continue
        try:
            total_mib += float(text)
        except ValueError:
            continue
    if total_mib <= 0:
        return None
    return int(total_mib * 1024 * 1024)


def scrape_vllm_kv_usage_perc(url: str, timeout: float = 2.0) -> float | None:
    metrics_url = url.rstrip("/") + "/metrics"
    try:
        req = urllib.request.Request(metrics_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    return parse_prometheus_kv_usage_perc(body)


def scrape_serve_metrics(url: str, timeout: float = 2.0) -> tuple[float | None, int | None]:
    """KV usage fraction and optional isiro allocated peak bytes."""
    metrics_url = url.rstrip("/") + "/metrics"
    try:
        req = urllib.request.Request(metrics_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None, None
    return parse_prometheus_kv_usage_perc(body), parse_prometheus_isiro_peak_bytes(body)


def _peak_kv_usage_perc(url: str, fn, poll_s: float = 0.05):
    """Run fn while sampling /metrics. Returns (fn result, peak usage, peak alloc)."""
    import threading

    usage0, alloc0 = scrape_serve_metrics(url)
    peak: list[float | None] = [usage0]
    alloc: list[int | None] = [alloc0]
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            perc, peak_b = scrape_serve_metrics(url)
            if perc is not None:
                peak[0] = perc if peak[0] is None else max(peak[0], perc)
            if peak_b is not None:
                alloc[0] = peak_b if alloc[0] is None else max(alloc[0], peak_b)
            stop.wait(poll_s)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        return fn(), peak[0], alloc[0]
    finally:
        stop.set()
        thread.join(timeout=2.0)
        perc, peak_b = scrape_serve_metrics(url)
        if perc is not None:
            peak[0] = perc if peak[0] is None else max(peak[0], perc)
        if peak_b is not None:
            alloc[0] = peak_b if alloc[0] is None else max(alloc[0], peak_b)


def _one_probe(
    *,
    url: str,
    model: str,
    panel: str,
    kv_cache_gib: float,
    depth: float,
    context_tokens: int,
    timeout: float,
    max_model_len: int = 0,
    needles: Iterable[str] = NEEDLES,
    question: str = QUESTION,
) -> dict[str, Any]:
    cap = int(max_model_len) if int(max_model_len) > 0 else None
    facts = tuple(needles) or (NEEDLE,)
    extra_tokens = extra_prompt_tokens(*facts, question)
    hay_chars = chars_for_context_tokens(
        context_tokens, cap, extra_tokens=extra_tokens
    )
    messages = build_messages(
        depth_percent=float(depth),
        approx_chars=hay_chars,
        needles=facts,
        question=question,
    )
    refuse = False
    oom = False
    window_overflow = False
    response_completes = True
    completion = ""
    err = ""
    try:
        completion = _chat_completion(
            url, model=model, messages=messages, timeout=timeout
        )
    except Exception as exc:
        response_completes = False
        err = str(exc)
        text = err.lower()
        refuse = "refuse" in text or "rejected" in text or "bad request" in text
        oom = "out of memory" in text or "oom" in text
        window_overflow = (
            "maximum context length" in text or "input_tokens" in text
        )
        if window_overflow:
            refuse = True
    extra: dict[str, Any] = {
        "completion_chars": len(completion),
        "completion": (completion or "")[:COMPLETION_STORE_CHARS],
        "hay_chars": hay_chars,
        "hay_tokens": hay_tokens_for_context(
            context_tokens, cap, extra_tokens=extra_tokens
        ),
        "needle_count": len(facts),
        "needle_depths": list(needle_spread_depths(len(facts), float(depth))),
        "score_keys": list(needle_keys(facts)),
        "seed": CHAT_SEED,
        "model": str(model),
    }
    if err:
        extra["error"] = err[:500]
    if window_overflow:
        extra["window_overflow"] = True
    score = score_needles(completion, facts)
    return row(
        panel=panel,
        kv_cache_gib=kv_cache_gib,
        depth_percent=float(depth),
        context_tokens=int(context_tokens),
        hit=score >= 1.0,
        score=score,
        response_completes=response_completes,
        refuse=refuse,
        oom=oom,
        extra=extra,
    )


def probe_grid(
    *,
    url: str,
    model: str,
    panel: str,
    kv_cache_gib: float,
    depths: Iterable[float] = DEFAULT_DEPTHS,
    context_tokens: Iterable[int] = DEFAULT_CONTEXT_TOKENS,
    timeout: float = 300.0,
    max_model_len: int = 0,
    needle: str = NEEDLE,
    question: str = QUESTION,
    pool_tokens: int = 0,
    gpu_memory_utilization: float = 0.90,
    non_kv_bytes: int = 0,
) -> list[dict[str, Any]]:
    facts = (needle,) if needle != NEEDLE else NEEDLES
    rows: list[dict[str, Any]] = []
    hbm_bytes = scrape_gpu_hbm_bytes()
    util = float(gpu_memory_utilization)
    reported_non_kv = int(non_kv_bytes) if int(non_kv_bytes) > 0 else None
    gpu_once = scrape_gpu_process_bytes()
    non_kv, non_kv_kind = resolve_non_kv_bytes(
        reported=reported_non_kv,
        process_bytes=gpu_once,
        pool_gib=float(kv_cache_gib),
    )
    for tokens in context_tokens:
        gpu_bytes = scrape_gpu_process_bytes()
        depth_list = [float(d) for d in depths]
        usage_perc: float | None = None
        alloc_peak: int | None = None
        bin_rows: list[dict[str, Any]] = []
        for i, depth in enumerate(depth_list):
            if i == 0:
                item, usage_perc, alloc_peak = _peak_kv_usage_perc(
                    url,
                    lambda d=depth: _one_probe(
                        url=url,
                        model=model,
                        panel=panel,
                        kv_cache_gib=kv_cache_gib,
                        depth=d,
                        context_tokens=int(tokens),
                        timeout=timeout,
                        max_model_len=max_model_len,
                        needles=facts,
                        question=question,
                    ),
                )
            else:
                item = _one_probe(
                    url=url,
                    model=model,
                    panel=panel,
                    kv_cache_gib=kv_cache_gib,
                    depth=float(depth),
                    context_tokens=int(tokens),
                    timeout=timeout,
                    max_model_len=max_model_len,
                    needles=facts,
                    question=question,
                )
            bin_rows.append(item)
        in_use, kind = kv_in_use_bytes(
            context_tokens=int(tokens),
            pool_gib=float(kv_cache_gib),
            pool_tokens=int(pool_tokens),
            usage_perc=usage_perc,
        )
        for item in bin_rows:
            if gpu_bytes is not None:
                item["gpu_process_bytes_measured"] = gpu_bytes
            if hbm_bytes is not None:
                item["gpu_hbm_bytes"] = hbm_bytes
            item["gpu_memory_utilization"] = util
            item["kv_bytes_in_use"] = in_use
            item["kv_in_use_kind"] = kind
            item["kv_pool_tokens"] = int(pool_tokens)
            item["non_kv_bytes_reported"] = non_kv
            item["non_kv_kind"] = non_kv_kind
            ids_b = derived_archived_id_bytes(int(tokens), int(pool_tokens))
            persist_b = 0
            persist_kind = "derived"
            if (
                _panel_alias(panel) == "tic"
                and alloc_peak is not None
                and int(alloc_peak) > 0
            ):
                floor = non_kv + gib_to_bytes(float(kv_cache_gib))
                extra_alloc = int(alloc_peak) - floor
                if extra_alloc > 0:
                    persist_b = extra_alloc
                    persist_kind = "metrics"
            if persist_b <= 0 and _panel_alias(panel) == "tic":
                persist_b = ids_b
                persist_kind = "derived"
            item["archived_id_bytes"] = ids_b
            item["persist_page_bytes"] = persist_b
            item["persist_kind"] = persist_kind
            if alloc_peak is not None:
                item["isiro_gpu_memory_peak_bytes"] = int(alloc_peak)
            rows.append(item)
    return rows


def probe_story(
    *,
    url: str,
    model: str,
    panel: str,
    kv_cache_gib: float,
    depths: Iterable[float] = DEFAULT_DEPTHS,
    story_tokens: int = STORY_CONTEXT_TOKENS,
    timeout: float = 300.0,
    max_model_len: int = 0,
    needle: str = NEEDLE,
    question: str = QUESTION,
    pool_tokens: int = 0,
    gpu_memory_utilization: float = 0.90,
    non_kv_bytes: int = 0,
) -> list[dict[str, Any]]:
    return probe_grid(
        url=url,
        model=model,
        panel=panel,
        kv_cache_gib=kv_cache_gib,
        depths=depths,
        context_tokens=(int(story_tokens),),
        timeout=timeout,
        max_model_len=max_model_len,
        needle=needle,
        question=question,
        pool_tokens=pool_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        non_kv_bytes=non_kv_bytes,
    )


def story_recalls(
    rows: list[dict[str, Any]],
    *,
    story_tokens: int = STORY_CONTEXT_TOKENS,
) -> bool:
    target = int(story_tokens)
    return any(
        item.get("hit") and int(item.get("context_tokens", target)) == target
        for item in rows
    )


def _panel_label(panel: str) -> str:
    name = str(panel or "").lower()
    if name in ("tic", "isiro"):
        return "Isiro"
    if name in ("vanilla", "baseline"):
        return "Baseline"
    return str(panel)


def _panel_alias(name: str) -> str | None:
    key = str(name or "").lower()
    if key in ("tic", "isiro"):
        return "tic"
    if key in ("vanilla", "baseline"):
        return "baseline"
    return None


def _setup_groups(rows: list[dict[str, Any]]) -> list[tuple[str, float, list[dict[str, Any]]]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for item in rows:
        alias = _panel_alias(str(item.get("panel") or ""))
        if (
            alias is None
            or not _has_kv_cache_gib(item)
            or "depth_percent" not in item
            or item.get("start_fail")
        ):
            continue
        key = (alias, round(_kv_cache_gib(item), 4))
        groups.setdefault(key, []).append(item)
    return [
        (panel, size, groups[key])
        for key in sorted(groups, key=lambda item: (0 if item[0] == "tic" else 1, item[1]))
        for panel, size in (key,)
    ]


def _pivot_context(items: list[dict[str, Any]]):
    import pandas as pd

    if not items or "context_tokens" not in items[0]:
        raise ValueError("rows need context_tokens for the heatmap")
    frame = pd.DataFrame(
        {
            "depth": [round(float(item["depth_percent"]), 4) for item in items],
            "context": [int(item["context_tokens"]) for item in items],
            "score": [
                float(item.get("score", 1.0 if item.get("hit") else 0.0))
                for item in items
            ],
        }
    )
    pivot = frame.pivot_table(
        index="depth", columns="context", values="score", aggfunc="mean"
    )
    return pivot.sort_index(ascending=True)


def _context_tick(value: float) -> str:
    return f"{int(value):,}"


def _niah_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    # Same stops as EvalScope / Kamradt NIAH figures.
    return LinearSegmentedColormap.from_list(
        "niah", ["#F0496E", "#EBB839", "#0CD79F"]
    )


_VRAM_COLORS = {"tic": "#1b6ca8", "baseline": "#c45c26"}


def render_heatmap(
    rows: list[dict[str, Any]],
    dest: Path,
    max_model_len: int | None = None,
    *,
    gpu: str = "",
    gpu_count: int | None = None,
    model: str = "",
) -> Path:
    """JSONL pivot to seaborn heatmap. Product path is PNG."""
    tagline = figure_tagline(rows, gpu=gpu, gpu_count=gpu_count, model=model)
    axis = context_axis_tokens(rows, max_model_len)
    rows = rows_for_heatmap(rows, max_model_len)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit(
            "heatmap needs matplotlib, seaborn, and pandas; "
            "install matplotlib seaborn pandas"
        ) from exc

    dest = Path(dest)
    setups = _setup_groups(rows)
    if not setups:
        raise ValueError("rows need panel, kv_cache_gib, and depth_percent")
    pivots = [_pivot_context(items) for _, _, items in setups]
    cmap = _niah_cmap()
    n = len(setups)
    any_oom = any(item.get("oom") for item in rows)
    fig = plt.figure(figsize=(11.2, 5.15 * n + (1.15 if tagline else 0.95)))
    spec = fig.add_gridspec(
        n,
        2,
        width_ratios=[1.0, 0.038],
        wspace=0.10,
        hspace=0.34,
    )
    for i, ((panel, _size, items), pivot) in enumerate(zip(setups, pivots)):
        if axis:
            pivot = pivot.reindex(columns=list(axis))
        oom_cols = {
            int(item["context_tokens"])
            for item in items
            if item.get("oom") and item.get("context_tokens") is not None
        }
        annot = pivot.copy().astype(object)
        plot = pivot.astype(float)
        for col in list(pivot.columns):
            if int(col) in oom_cols:
                plot[col] = float("nan")
                annot[col] = ""
            else:
                annot[col] = pivot[col].map(
                    lambda v: "" if v != v else f"{float(v):.1f}"
                )
        ax = fig.add_subplot(spec[i, 0])
        cax = fig.add_subplot(spec[i, 1])
        mesh = sns.heatmap(
            plot,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            annot=annot,
            fmt="",
            annot_kws={"size": 7.2},
            cbar=True,
            cbar_ax=cax,
            cbar_kws={"label": "Score"},
            linewidths=0.35,
            linecolor="#f7f7f7",
            xticklabels=[_context_tick(col) for col in pivot.columns],
            yticklabels=[f"{idx:g}" for idx in pivot.index],
            square=False,
        )
        for artist in mesh.collections:
            artist.set_edgecolor("#f7f7f7")
            artist.set_linewidth(0.35)
        color = _VRAM_COLORS.get(panel, "#333333")
        col_list = list(pivot.columns)
        row_n = len(pivot.index)
        for j, col in enumerate(col_list):
            if int(col) not in oom_cols:
                continue
            for i_row in range(row_n):
                ax.scatter(
                    j + 0.5,
                    i_row + 0.5,
                    marker="x",
                    s=42,
                    color=color,
                    linewidths=1.3,
                    zorder=5,
                )
        ax.set_title(_panel_label(panel), fontsize=11, pad=8)
        ax.set_xlabel("Context (tokens)")
        ax.set_ylabel("Needle depth (%)")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax.tick_params(labelsize=8)
    fig.suptitle(FIGURE_TITLE, y=0.995, fontsize=12, va="top")
    top = 0.93
    if tagline:
        fig.text(
            0.49,
            0.962,
            tagline,
            ha="center",
            va="top",
            fontsize=9,
            color="#444444",
        )
        top = 0.90
    bottom = 0.14 if any_oom else 0.08
    if any_oom:
        from matplotlib.lines import Line2D

        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=_VRAM_COLORS["baseline"],
                    marker="x",
                    ls="none",
                    ms=8,
                    mew=1.6,
                    markeredgecolor=_VRAM_COLORS["baseline"],
                    markerfacecolor=_VRAM_COLORS["baseline"],
                    label="Out of memory (OOM)",
                )
            ],
            loc="upper center",
            bbox_to_anchor=(0.49, 0.02),
            frameon=True,
            fancybox=False,
            edgecolor="#c8c8c8",
            fontsize=9,
        )
    fig.subplots_adjust(left=0.08, right=0.90, top=top, bottom=bottom)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return dest


def render_vram_vs_context(
    rows: list[dict[str, Any]],
    dest: Path,
    max_model_len: int | None = None,
    *,
    gpu: str = "",
    gpu_count: int | None = None,
    model: str = "",
) -> Path:
    """Non-KV plus KV occupancy vs context. Dashed non-KV plus pool."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise SystemExit(
            "vram figure needs matplotlib; install matplotlib"
        ) from exc

    dest = Path(dest)
    series: dict[tuple[str, float], dict[int, int]] = {}
    reserved_by_key: dict[tuple[str, float], int] = {}
    oom_by_key: dict[tuple[str, float], set[int]] = {}
    oom_tokens: set[int] = set()
    hbm_bytes = 0
    util = 0.90
    for item in rows:
        if item.get("start_fail"):
            continue
        alias = _panel_alias(str(item.get("panel") or ""))
        if alias is None or not _has_kv_cache_gib(item):
            continue
        tokens = int(item.get("context_tokens") or 0)
        if tokens <= 0:
            continue
        raw_hbm = item.get("gpu_hbm_bytes")
        if isinstance(raw_hbm, (int, float)) and raw_hbm > 0:
            hbm_bytes = max(hbm_bytes, int(raw_hbm))
        raw_util = item.get("gpu_memory_utilization")
        if isinstance(raw_util, (int, float)) and raw_util > 0:
            util = float(raw_util)
        key = (alias, round(_kv_cache_gib(item), 4))
        non_kv_lock = locked_non_kv_bytes(
            rows, panel=alias, size=_kv_cache_gib(item)
        )
        reserved_by_key[key] = non_kv_lock + gib_to_bytes(_kv_cache_gib(item))
        used = vram_in_use_y_bytes(item, non_kv_override=non_kv_lock)
        if item.get("oom"):
            oom_by_key.setdefault(key, set()).add(tokens)
            series.setdefault(key, {})
            series[key].setdefault(tokens, used)
            continue
        series.setdefault(key, {})
        series[key].setdefault(tokens, used)

    if hbm_bytes <= 0:
        hbm_bytes = gib_to_bytes(32.0)
    hbm_gib = bytes_to_gib(hbm_bytes)
    usable_gib = hbm_gib * util

    dest.parent.mkdir(parents=True, exist_ok=True)
    ymax = hbm_gib * 1.08
    tagline = figure_tagline(rows, gpu=gpu, gpu_count=gpu_count, model=model)
    axis = context_axis_tokens(rows, max_model_len)
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.axhspan(usable_gib, hbm_gib, color="#c0392b", alpha=0.12, zorder=0)
    ax.axhspan(hbm_gib, ymax, color="#c0392b", alpha=0.38, zorder=0)
    ax.axhline(usable_gib, color="#4a4a4a", ls=":", lw=1.4, zorder=2)
    tick_xs: list[int] = []
    oom_legend_color = _VRAM_COLORS["baseline"]
    for (panel, size), points in sorted(
        series.items(),
        key=lambda item: (0 if item[0][0] == "tic" else 1, item[0][1]),
    ):
        color = _VRAM_COLORS.get(panel, "#333333")
        skip = oom_by_key.get((panel, size), set())
        xs = sorted(points)
        if xs:
            ys = [
                hbm_gib if x in skip else bytes_to_gib(points[x])
                for x in xs
            ]
            tick_xs.extend(xs)
            ax.plot(
                xs,
                ys,
                color=color,
                marker="None",
                lw=2.0,
                zorder=3,
            )
            good = [x for x in xs if x not in skip]
            if good:
                ax.scatter(
                    good,
                    [bytes_to_gib(points[x]) for x in good],
                    marker="o",
                    s=28,
                    color=color,
                    zorder=4,
                    linewidths=0,
                )
            oom_xs = [x for x in xs if x in skip]
            if oom_xs:
                oom_legend_color = color
                ax.scatter(
                    oom_xs,
                    [hbm_gib] * len(oom_xs),
                    marker="x",
                    s=70,
                    color=color,
                    zorder=5,
                    linewidths=1.6,
                )
        ax.axhline(
            bytes_to_gib(reserved_by_key.get((panel, size), gib_to_bytes(size))),
            color=color,
            ls="--",
            lw=1.25,
            alpha=0.85,
            zorder=2,
        )
    ax.set_xlabel("Context (tokens)", fontsize=11)
    ax.set_ylabel("VRAM usage (GiB)", fontsize=11)
    ax.set_xscale("linear")
    ax.set_ylim(0.0, ymax)
    fig_h = float(fig.get_size_inches()[1])
    title_h = 13.0 / (72.0 * fig_h)
    tag_h = 9.0 / (72.0 * fig_h)
    title_tag_gap = 4.0 / (72.0 * fig_h)
    axes_gap = VRAM_TAGLINE_AXES_GAP_PT / (72.0 * fig_h)
    title_y = 0.99
    fig.suptitle(VRAM_TITLE, y=title_y, fontsize=13, va="top")
    top = 0.90
    if tagline:
        tagline_y = title_y - title_h - title_tag_gap
        fig.text(
            0.535,
            tagline_y,
            tagline,
            ha="center",
            va="top",
            fontsize=9,
            color="#444444",
        )
        top = tagline_y - tag_h - axes_gap
    unique_ticks = list(axis) if axis else sorted(set(tick_xs))
    if unique_ticks:
        ax.set_xticks(unique_ticks)
        ax.set_xticklabels(
            [_context_tick(x) for x in unique_ticks],
            rotation=45,
            ha="right",
        )
        lo, hi = unique_ticks[0], unique_ticks[-1]
        if hi > lo:
            pad = 0.03 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
        elif lo > 0:
            ax.set_xlim(lo / 1.12, hi * 1.12)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(
        axis="x",
        which="minor",
        bottom=False,
        top=False,
        length=0,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=8,
        direction="out",
        top=False,
        right=False,
        length=3.5,
        width=0.8,
        pad=4,
    )
    ax.tick_params(axis="x", which="major", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.28, color="#b8b8b8", lw=0.6, zorder=0)
    ax.grid(True, axis="x", which="major", alpha=0.16, color="#b8b8b8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#222222")
    legend_items = [
        Line2D([0], [0], color=_VRAM_COLORS["tic"], lw=2, marker="o", ms=5, label="Isiro"),
        Line2D(
            [0],
            [0],
            color=_VRAM_COLORS["baseline"],
            lw=2,
            marker="o",
            ms=5,
            label="Baseline",
        ),
        Line2D(
            [0],
            [0],
            color=_VRAM_COLORS["tic"],
            lw=1.25,
            ls="--",
            label="Isiro reserved",
        ),
        Line2D(
            [0],
            [0],
            color=_VRAM_COLORS["baseline"],
            lw=1.25,
            ls="--",
            label="Baseline reserved",
        ),
        Line2D([0], [0], color="#4a4a4a", lw=1.4, ls=":", label="Usable VRAM"),
        Patch(facecolor="#c0392b", alpha=0.12, label="Outside usable VRAM"),
        Patch(facecolor="#c0392b", alpha=0.38, label="Above GPU HBM"),
        Line2D(
            [0],
            [0],
            color=oom_legend_color,
            marker="x",
            markeredgecolor=oom_legend_color,
            markerfacecolor=oom_legend_color,
            ls="none",
            ms=8,
            mew=1.6,
            label="Out of memory (OOM)",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#c8c8c8",
        framealpha=1.0,
        fontsize=9,
        handlelength=2.4,
        columnspacing=1.4,
        borderpad=0.7,
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=top, bottom=0.32)
    fig.savefig(dest, dpi=160, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return dest


def mock_rows() -> list[dict[str, Any]]:
    """Qwen-class story: product TIC lock, Baseline start-fail, default fill."""
    tic_gib = 1.09
    fill_gib = 13.48
    tic_tokens = 20480
    fill_tokens = 238064
    hbm_bytes = gib_to_bytes(32.0)
    util = 0.90
    bins = context_length_bins(mock_advertised_max())
    last = bins[-1]
    prior = [b for b in bins if b < last]
    rows: list[dict[str, Any]] = start_fail_rows(
        panel="vanilla", kv_cache_gib=tic_gib
    )
    for panel, kv_cache_gib, pool_tokens in (
        ("tic", tic_gib, tic_tokens),
        ("vanilla", fill_gib, fill_tokens),
    ):
        for depth in DEFAULT_DEPTHS:
            for context_tokens in bins:
                oom = panel == "vanilla" and context_tokens == last
                if oom:
                    score = 0.0
                elif panel == "tic":
                    score = 1.0
                elif prior and context_tokens == prior[-1]:
                    score = 0.3333 if depth >= 44 else 0.6667
                elif len(prior) >= 2 and context_tokens == prior[-2]:
                    score = 0.6667 if depth >= 56 else 1.0
                else:
                    score = 1.0
                used, kind = kv_in_use_bytes(
                    context_tokens=context_tokens,
                    pool_gib=kv_cache_gib,
                    pool_tokens=pool_tokens,
                    usage_perc=None,
                )
                item = row(
                    panel=panel,
                    kv_cache_gib=kv_cache_gib,
                    depth_percent=depth,
                    context_tokens=context_tokens,
                    hit=score >= 1.0,
                    score=score,
                    oom=oom,
                    response_completes=not oom,
                    refuse=oom,
                )
                item["gpu_process_bytes_measured"] = int(
                    (12.28 if panel == "tic" else 28.05) * (1024**3)
                )
                item["gpu_hbm_bytes"] = hbm_bytes
                item["gpu_memory_utilization"] = util
                item["kv_bytes_in_use"] = used
                item["kv_in_use_kind"] = kind
                item["kv_pool_tokens"] = pool_tokens
                ids_b = derived_archived_id_bytes(context_tokens, pool_tokens)
                item["archived_id_bytes"] = ids_b
                item["persist_page_bytes"] = (
                    ids_b if panel == "tic" else 0
                )
                item["persist_kind"] = "derived"
                item["non_kv_bytes_reported"] = gib_to_bytes(
                    11.19 if panel == "tic" else 15.34
                )
                item["non_kv_kind"] = "reported"
                item["gpu_model"] = "NVIDIA GeForce RTX 5090"
                item["gpu_count"] = 1
                item["model"] = "qwen2.5-7b-instruct"
                rows.append(item)
    for item in rows:
        item.setdefault("gpu_model", "NVIDIA GeForce RTX 5090")
        item.setdefault("gpu_count", 1)
        item.setdefault("model", "qwen2.5-7b-instruct")
    return rows


def kv_sweep_table_lines(
    rows: list[dict[str, Any]],
    *,
    reserved_gib: float | None = None,
) -> list[str]:
    """Section B intro only. Recall and OOM live in the figures."""
    del reserved_gib
    if not any(_has_kv_cache_gib(item) for item in rows):
        return []
    intro = (
        "Same long prompt, needles, and question on every setup that starts."
    )
    fail_gibs = _start_fail_gibs(rows)
    if fail_gibs:
        intro = (
            f"{intro} The Baseline serve would not start at "
            f"{format_gib_list(fail_gibs)}."
        )
    return [
        "## B. Needle-in-a-haystack (context tokens × depth)",
        "",
        intro,
        "",
    ]


def kv_overflow_footnote_lines(*, gib_only: bool = False) -> list[str]:
    """Short honesty note when a dagger is still used (legacy callers)."""
    _ = gib_only
    body = (
        "† Available KV is the vendor reserved pool (GiB). KV tokens capacity "
        "uses Isiro KV cold packing on that budget: scale 1/(1-s) with a dense hot "
        "window. Example: s=0.42 → 1/0.58 ≈ 1.72 → +72% tokens."
    )
    return [body, ""]


def report_lines(
    image_rel: str,
    *,
    heading: str | None = "## B. Needle-in-a-haystack (context tokens × depth)",
) -> list[str]:
    lines: list[str] = []
    if heading:
        lines.extend([heading, ""])
    lines.extend(
        [
            (
                "X is context tokens. "
                "Y is needle depth % (0 = start of prompt, 100 = end). "
                "Color is recall. "
                "An X in a column is OOM, same as the VRAM figure."
            ),
            "",
            f"Figure. {FIGURE_TITLE}.",
            "",
            f"![{FIGURE_TITLE}]({image_rel})",
            "",
        ]
    )
    return lines


def vram_report_lines(image_rel: str) -> list[str]:
    return [
        "Figure. VRAM usage vs context tokens.",
        "",
        (
            "Solid line is in-use: non-KV (weights) plus live vendor KV "
            "(usage_perc times pool, or derived T/pool) plus TokenID/persist "
            "outside the pool. "
            "Dashed lines are reserved pools (non-KV plus the full KV lock or fill). "
            "The dotted line is usable VRAM. "
            "The light red band is above usable VRAM. "
            "The darker red band is above GPU HBM. "
            "An X on the GPU HBM line is OOM at that context. "
            "Not nvidia-smi. TIC in-use stays weights plus the 1.09 GiB lock "
            "plus TokenIDs past the lock. Baseline in-use grows with fill occupancy."
        ),
        "",
        f"![VRAM usage vs context tokens]({image_rel})",
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Needle-in-a-haystack grid probe and plot"
    )
    parser.add_argument("--jsonl", type=Path, help="Read or write JSONL rows")
    parser.add_argument("--out", type=Path, help="Heatmap PNG (seaborn)")
    parser.add_argument(
        "--vram-out",
        type=Path,
        help="VRAM usage vs context PNG",
    )
    parser.add_argument("--mock", action="store_true", help="Render mock Qwen-class rows")
    parser.add_argument("--probe-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--gpu-model",
        default="",
        help="GPU name for the figure tagline, e.g. NVIDIA GeForce RTX 5090",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=0,
        help="GPU count for the figure tagline (e.g. 1 for 'x 1')",
    )
    parser.add_argument("--panel", default="tic")
    parser.add_argument("--kv-cache-gib", type=float, default=0.0)
    parser.add_argument(
        "--context-tokens",
        default="",
        help="Comma-separated context bins (default: 1k to advertised max)",
    )
    parser.add_argument("--story-only", action="store_true")
    parser.add_argument("--story-tokens", type=int, default=STORY_CONTEXT_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--pool-tokens", type=int, default=0)
    parser.add_argument(
        "--non-kv-bytes",
        type=int,
        default=0,
        help="vLLM Total non KV cache memory from the serve log",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--advertised-max",
        type=Path,
        help="Print advertised max_position_embeddings from HF config.json",
    )
    parser.add_argument("--plan", action="store_true", help="Print discover steps")
    parser.add_argument("--parse-log", type=Path, help="Print scraped KV GiB from a serve log")
    parser.add_argument(
        "--parse-log-tokens",
        type=Path,
        help="Print scraped GPU KV token count from a serve log",
    )
    parser.add_argument(
        "--parse-log-non-kv",
        type=Path,
        help="Print scraped Total non KV cache memory bytes from a serve log",
    )
    parser.add_argument(
        "--record-start-fail",
        action="store_true",
        help="Append a Baseline start-fail story row (serve did not start)",
    )
    args = parser.parse_args(argv)

    if args.advertised_max is not None:
        found = advertised_max_model_len(args.advertised_max)
        print("" if found is None else str(found))
        return 0
    if args.plan:
        for line in niah_discover_steps():
            print(line)
        return 0
    if args.parse_log:
        print(f"{parse_kv_cache_gib(args.parse_log.read_text(encoding='utf-8', errors='replace')):.4f}")
        return 0
    if args.parse_log_tokens:
        print(
            parse_kv_cache_tokens(
                args.parse_log_tokens.read_text(encoding="utf-8", errors="replace")
            )
        )
        return 0
    if args.parse_log_non_kv:
        parsed = parse_non_kv_cache_bytes(
            args.parse_log_non_kv.read_text(encoding="utf-8", errors="replace")
        )
        print(0 if parsed is None else parsed)
        return 0
    if args.record_start_fail:
        if args.kv_cache_gib <= 0 or not args.jsonl:
            raise SystemExit("--record-start-fail needs --kv-cache-gib and --jsonl")
        rows = start_fail_rows(
            panel=args.panel,
            kv_cache_gib=args.kv_cache_gib,
            story_tokens=int(args.story_tokens),
        )
        existing = load_jsonl(args.jsonl) if args.jsonl.is_file() else []
        write_jsonl(args.jsonl, existing + rows)
        return 0

    rows: list[dict[str, Any]] = []
    if args.mock:
        rows = mock_rows()
    elif args.probe_url:
        if not args.model or args.kv_cache_gib <= 0:
            raise SystemExit("--probe-url needs --model and --kv-cache-gib")
        if args.context_tokens.strip():
            tokens = tuple(
                int(part) for part in args.context_tokens.split(",") if part.strip()
            )
            if args.max_model_len > 0:
                tokens = context_bins_for_max_len(args.max_model_len, tokens)
        elif args.max_model_len > 0:
            tokens = context_bins_for_max_len(args.max_model_len)
        else:
            tokens = DEFAULT_CONTEXT_TOKENS
        story_tokens = int(args.story_tokens)
        if args.max_model_len > 0:
            story_tokens = story_tokens_for_max_len(
                args.max_model_len, story=story_tokens, bins=tokens
            )
        if args.story_only:
            rows = probe_story(
                url=args.probe_url,
                model=args.model,
                panel=args.panel,
                kv_cache_gib=args.kv_cache_gib,
                story_tokens=story_tokens,
                max_model_len=args.max_model_len,
                pool_tokens=int(args.pool_tokens),
                gpu_memory_utilization=float(args.gpu_memory_utilization),
                non_kv_bytes=int(args.non_kv_bytes),
            )
        else:
            rows = probe_grid(
                url=args.probe_url,
                model=args.model,
                panel=args.panel,
                kv_cache_gib=args.kv_cache_gib,
                context_tokens=tokens,
                max_model_len=args.max_model_len,
                pool_tokens=int(args.pool_tokens),
                gpu_memory_utilization=float(args.gpu_memory_utilization),
                non_kv_bytes=int(args.non_kv_bytes),
            )
        if args.jsonl:
            existing = load_jsonl(args.jsonl) if args.jsonl.is_file() else []
            write_jsonl(args.jsonl, existing + rows)
            rows = load_jsonl(args.jsonl)
    elif args.jsonl and args.jsonl.is_file():
        rows = load_jsonl(args.jsonl)
    else:
        raise SystemExit("pass --mock, --jsonl, or --probe-url")

    if args.jsonl and args.mock:
        write_jsonl(args.jsonl, rows)
    max_len = int(args.max_model_len) if int(args.max_model_len) > 0 else None
    if args.out:
        render_heatmap(
            rows,
            args.out,
            max_model_len=max_len,
            gpu=args.gpu_model,
            gpu_count=args.gpu_count or None,
            model=args.model,
        )
        print(args.out)
    if args.vram_out:
        render_vram_vs_context(
            rows,
            args.vram_out,
            max_model_len=max_len,
            gpu=args.gpu_model,
            gpu_count=args.gpu_count or None,
            model=args.model,
        )
        print(args.vram_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
