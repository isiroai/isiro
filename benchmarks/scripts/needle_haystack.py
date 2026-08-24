#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-a-haystack grid: JSONL probe + heatmap (TIC vs baseline).

Protocol is single-needle (Kamradt / RULER style).
Each panel is one KV cache size that started. X is context tokens. Y is needle depth %.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

NEEDLE = "The special magic number is 84932017."
QUESTION = "What is the special magic number? Reply with the number only."
HAYSTACK_LINE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "We like to go to the park and sit on a bench. "
)

DEFAULT_DEPTHS = (0, 10, 25, 50, 75, 90, 100)
DEFAULT_CONTEXT_TOKENS = (1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000)
# Table A folds the grid at this bin when the model window allows it.
STORY_CONTEXT_TOKENS = 16000
CHARS_PER_TOKEN = 4
# Haystack bin plus needle, question, chat template, and a short reply.
WINDOW_RESERVE_TOKENS = 1024
FIGURE_TITLE = (
    "Fact retrieval by KV cache and context length (needle-in-a-haystack)"
)
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


def figure_unstarted_note(kv_cache_gib: float | Iterable[float]) -> str:
    if isinstance(kv_cache_gib, (int, float)):
        sizes = [float(kv_cache_gib)]
    else:
        sizes = [float(item) for item in kv_cache_gib]
    label = format_gib_list(sizes)
    unique = sorted({round(float(item), 4) for item in sizes})
    if len(unique) <= 1:
        return f"Baseline did not start at {label}, so it is not shown."
    return f"Baseline did not start at {label}, so they are not shown."


def _start_fail_gibs(rows: Iterable[dict[str, Any]]) -> list[float]:
    return sorted(
        {
            round(float(item["kv_cache_gib"]), 4)
            for item in rows
            if item.get("start_fail") and "kv_cache_gib" in item
        }
    )


def score_hit(completion: str, needle: str = NEEDLE) -> bool:
    """Exact needle, or the magic number if the needle contains one."""
    text = (completion or "").strip()
    if not text:
        return False
    if needle in text:
        return True
    digits = "".join(ch for ch in needle if ch.isdigit())
    return bool(digits) and digits in text


def chars_for_context_tokens(context_tokens: int) -> int:
    return max(1, int(context_tokens) * CHARS_PER_TOKEN)


def parse_kv_cache_gib(log_text: str) -> float:
    """Measured KV pool in GiB from a serve log.

    Prefer dest ``KV pool:`` GiB. Fall back to vLLM Available / reserved.
    """
    import re

    kv_pool = re.search(r"KV pool:\s*([0-9.]+)\s*GiB", log_text)
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
    pool = re.search(r"vendor_pool_bytes=(\d+)", log_text)
    if pool:
        return int(pool.group(1)) / float(1024**3)
    locked = re.search(r"kv_cache_memory_bytes['\"]?\s*[:=]\s*(\d+)", log_text)
    if locked:
        return int(locked.group(1)) / float(1024**3)
    raise ValueError("Available KV cache memory not found in serve log")


# Dest lock_seqs=1: (W + slack) slots. Serve prints GiB only.
_DEST_ONE_WINDOW = 512
_DEST_ONE_WINDOW_SLACK = 4096


def parse_kv_cache_tokens(log_text: str) -> int:
    """Dest slot count, or vLLM GPU tokens when dest did not print."""
    import re

    pool = re.search(
        r"KV pool:\s*[0-9.]+\s*GiB\s*\(([0-9,]+)\s*tokens",
        log_text,
    )
    if pool:
        return int(pool.group(1).replace(",", ""))
    if re.search(r"KV pool:\s*[0-9.]+\s*GiB", log_text):
        # One-window dest. vLLM GPU token count is last-W-inflated.
        # estimated max_model_len is the last-W boot guard, not dest slots.
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


def context_bins_for_max_len(
    max_model_len: int,
    bins: Iterable[int] = DEFAULT_CONTEXT_TOKENS,
) -> tuple[int, ...]:
    cap = max(1, int(max_model_len))
    usable = max(1, cap - WINDOW_RESERVE_TOKENS)
    kept = tuple(int(b) for b in bins if int(b) <= usable)
    return kept or (usable,)


def rows_for_heatmap(
    rows: Iterable[dict[str, Any]],
    max_model_len: int | None = None,
) -> list[dict[str, Any]]:
    """Drop bins that cannot fit the window, or that never completed."""
    items = [item for item in rows if not item.get("start_fail")]
    if isinstance(max_model_len, (int, float)) and int(max_model_len) > 0:
        allowed = set(context_bins_for_max_len(int(max_model_len)))
        items = [
            item
            for item in items
            if int(item.get("context_tokens") or 0) in allowed
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
        and all(
            (not item.get("response_completes", True))
            or item.get("refuse")
            or item.get("oom")
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


def first_recall_candidates(
    tic_gib: float,
    *,
    cap: float = 16.0,
) -> list[float]:
    """Double from TIC toward cap. Stops before fill. Not a fixed 4.00."""
    sizes: list[float] = []
    nxt = max(float(int(tic_gib) + 1), round(float(tic_gib) * 2.0, 2))
    while nxt < float(cap) - 0.05:
        sizes.append(round(nxt, 2))
        nxt = round(nxt * 2.0, 2)
    return sizes


def niah_discover_steps() -> list[str]:
    return [
        "TIC product serve; scrape GPU KV cache size; window follows pool tokens; full grid (1 serve)",
        "Baseline default fill; scrape GPU KV cache size; full grid (1 serve)",
        "Refuse if TIC KV is not smaller than Baseline fill (any model)",
        "Baseline same size as TIC; record start-fail if serve does not start; else full grid",
        "Baseline double KV until first recall; skip sizes that do not start; full grid on the hit serve",
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


def build_messages(
    *,
    depth_percent: float,
    approx_chars: int,
    needle: str = NEEDLE,
    question: str = QUESTION,
) -> list[dict[str, str]]:
    hay = haystack_text(approx_chars=approx_chars)
    body = insert_needle(hay, needle, depth_percent)
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
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "panel": str(panel),
        "kv_cache_gib": float(kv_cache_gib),
        "depth_percent": float(depth_percent),
        "hit": bool(hit),
        "score": 1.0 if hit else 0.0,
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
            "max_tokens": 32,
            "temperature": 0,
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
        raise RuntimeError(f"needle haystack chat HTTP {exc.code}") from exc
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("content") or "")


def _one_probe(
    *,
    url: str,
    model: str,
    panel: str,
    kv_cache_gib: float,
    depth: float,
    context_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    messages = build_messages(
        depth_percent=float(depth),
        approx_chars=chars_for_context_tokens(context_tokens),
    )
    refuse = False
    oom = False
    response_completes = True
    completion = ""
    try:
        completion = _chat_completion(
            url, model=model, messages=messages, timeout=timeout
        )
    except Exception as exc:
        response_completes = False
        text = str(exc).lower()
        refuse = "refuse" in text or "rejected" in text
        oom = "out of memory" in text or "oom" in text
    return row(
        panel=panel,
        kv_cache_gib=kv_cache_gib,
        depth_percent=float(depth),
        context_tokens=int(context_tokens),
        hit=score_hit(completion),
        response_completes=response_completes,
        refuse=refuse,
        oom=oom,
        extra={"completion_chars": len(completion)},
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tokens in context_tokens:
        for depth in depths:
            rows.append(
                _one_probe(
                    url=url,
                    model=model,
                    panel=panel,
                    kv_cache_gib=kv_cache_gib,
                    depth=float(depth),
                    context_tokens=int(tokens),
                    timeout=timeout,
                )
            )
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
) -> list[dict[str, Any]]:
    return probe_grid(
        url=url,
        model=model,
        panel=panel,
        kv_cache_gib=kv_cache_gib,
        depths=depths,
        context_tokens=(int(story_tokens),),
        timeout=timeout,
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
        return "TIC"
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
    return pivot.sort_index(ascending=False)


def _context_tick(value: float) -> str:
    tokens = int(value)
    if tokens >= 1000 and tokens % 1000 == 0:
        return f"{tokens // 1000}k"
    return str(tokens)


def _niah_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    # Same stops as EvalScope / Kamradt NIAH figures.
    return LinearSegmentedColormap.from_list(
        "niah", ["#F0496E", "#EBB839", "#0CD79F"]
    )


def render_heatmap(
    rows: list[dict[str, Any]],
    dest: Path,
    max_model_len: int | None = None,
) -> Path:
    """JSONL pivot to seaborn heatmap. Product path is PNG."""
    unstarted_gibs = _start_fail_gibs(rows)
    rows = rows_for_heatmap(rows, max_model_len)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
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
    fig = plt.figure(figsize=(3.6 * n + 1.4, 5.0))
    spec = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.07], wspace=0.12)
    axes = [fig.add_subplot(spec[0, i]) for i in range(n)]
    cax = fig.add_subplot(spec[0, -1])
    for i, (ax, (panel, size, _), pivot) in enumerate(zip(axes, setups, pivots)):
        mesh = sns.heatmap(
            pivot,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6.5},
            cbar=False,
            linewidths=0,
            xticklabels=[_context_tick(col) for col in pivot.columns],
            yticklabels=[f"{idx:g}" for idx in pivot.index],
            square=False,
        )
        for artist in mesh.collections:
            artist.set_edgecolor("face")
            artist.set_linewidth(0)
        ax.set_title(f"{_panel_label(panel)}  KV {size:.2f} GiB")
        ax.set_xlabel("Context (tokens)")
        if i == 0:
            ax.set_ylabel("Needle depth (%)")
        else:
            ax.set_ylabel("")
            ax.tick_params(left=False, labelleft=False)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap),
        cax=cax,
        label="Score",
    )
    colorbar.outline.set_visible(False)
    cax.set_frame_on(False)
    for spine in cax.spines.values():
        spine.set_visible(False)
    fig.suptitle(FIGURE_TITLE, y=0.98, fontsize=11, va="top")
    if unstarted_gibs:
        fig.text(
            0.5,
            0.935,
            figure_unstarted_note(unstarted_gibs),
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )
    fig.subplots_adjust(left=0.06, right=0.94, top=0.84, bottom=0.18, wspace=0.16)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, bbox_inches="tight", pad_inches=0.26)
    plt.close(fig)
    return dest


def mock_rows() -> list[dict[str, Any]]:
    """Qwen-class story: product TIC lock, Baseline start-fail, then sizes that start."""
    tic_gib = 1.09
    fill_gib = 13.48
    first_start = first_recall_candidates(tic_gib, cap=fill_gib)[0]
    rows: list[dict[str, Any]] = start_fail_rows(
        panel="vanilla", kv_cache_gib=tic_gib
    )
    for panel, kv_cache_gib in (
        ("tic", tic_gib),
        ("vanilla", first_start),
        ("vanilla", fill_gib),
    ):
        for depth in DEFAULT_DEPTHS:
            for context_tokens in DEFAULT_CONTEXT_TOKENS:
                rows.append(
                    row(
                        panel=panel,
                        kv_cache_gib=kv_cache_gib,
                        depth_percent=depth,
                        context_tokens=context_tokens,
                        hit=True,
                    )
                )
    return rows


def kv_sweep_table_lines(
    rows: list[dict[str, Any]],
    *,
    reserved_gib: float | None = None,
) -> list[str]:
    """One row per panel and KV size. Do not invent sizes."""
    tic_sizes = sorted(
        {
            round(_kv_cache_gib(item), 4)
            for item in rows
            if str(item.get("panel") or "").lower() in ("tic", "isiro")
            and _has_kv_cache_gib(item)
        }
    )
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for item in rows:
        if not _has_kv_cache_gib(item):
            continue
        name = str(item.get("panel") or "").lower()
        if name in ("tic", "isiro"):
            panel = "tic"
        elif name in ("vanilla", "baseline"):
            panel = "baseline"
        else:
            continue
        key = (panel, round(_kv_cache_gib(item), 4))
        groups.setdefault(key, []).append(item)
    if not groups:
        return []
    del reserved_gib
    tic_fixed = tic_sizes[0] if tic_sizes else None
    intro = (
        "Same long prompt, needle, and question on every setup that starts."
    )
    fail_gibs = sorted(
        {
            size
            for (panel, size), items in groups.items()
            if panel == "baseline" and any(item.get("start_fail") for item in items)
        }
    )
    if fail_gibs:
        intro = (
            f"{intro} The Baseline serve would not start at "
            f"{format_gib_list(fail_gibs)}."
        )
    lines = [
        "## B. Needle-in-a-haystack (context tokens × depth)",
        "",
        intro,
        "",
        "| Setup | KV cache | Size (GiB) | Response completes | Fact retrieved (needle) | Started without OOM |",
        "|-------|----------|------------|--------------------|-------------------------|---------------------|",
    ]
    baseline_sizes = sorted(size for panel, size in groups if panel == "baseline")
    fill_size = baseline_sizes[-1] if baseline_sizes else None
    order = sorted(groups, key=lambda key: (0 if key[0] == "tic" else 1, key[1]))
    for panel, size in order:
        items = groups[(panel, size)]
        story = [
            item
            for item in items
            if int(item.get("context_tokens", STORY_CONTEXT_TOKENS))
            == STORY_CONTEXT_TOKENS
        ]
        fold = story or items
        start_fail = any(item.get("start_fail") for item in items)
        if start_fail:
            completes, recalled, refused = False, False, True
        else:
            completes = any(item.get("response_completes", True) for item in fold)
            recalled = any(item.get("hit") for item in fold)
            refused = (not completes) and any(
                item.get("refuse") or item.get("oom") for item in fold
            )
        if panel == "tic":
            setup, policy = "**TIC**", "Fixed"
            size_cell = format_gib_cell(size)
        elif tic_fixed is not None and abs(size - tic_fixed) < 1e-6:
            setup, policy = "**Baseline**", "Same size as TIC"
            size_cell = format_gib_cell(size)
        elif fill_size is not None and abs(size - fill_size) < 1e-6:
            setup, policy = "**Baseline**", "Default fill"
            size_cell = format_gib_cell(size)
        else:
            setup, policy = "**Baseline**", "Larger KV"
            size_cell = format_gib_cell(size)
        flag = lambda ok: "yes" if ok else "no"
        lines.append(
            f"| {setup} | {policy} | {size_cell} | {flag(completes)} | "
            f"{flag(recalled)} | {flag(not refused)} |"
        )
    lines.append("")
    return lines


def kv_overflow_footnote_lines() -> list[str]:
    return [
        (
            "† Reserved pool on the GPU. Tokens that do not fit overflow "
            "to host memory."
        ),
        "",
    ]


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
                "Color is recall."
            ),
            "",
            "Figure. Fact retrieval by KV cache (needle-in-a-haystack).",
            "",
            f"![Fact retrieval by KV cache (needle-in-a-haystack)]({image_rel})",
            "",
        ]
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Needle-in-a-haystack grid probe and plot"
    )
    parser.add_argument("--jsonl", type=Path, help="Read or write JSONL rows")
    parser.add_argument("--out", type=Path, help="Heatmap PNG (seaborn)")
    parser.add_argument("--mock", action="store_true", help="Render mock Qwen-class rows")
    parser.add_argument("--probe-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="")
    parser.add_argument("--panel", default="tic")
    parser.add_argument("--kv-cache-gib", type=float, default=0.0)
    parser.add_argument(
        "--context-tokens",
        default="",
        help="Comma-separated context bins (default: 1k-32k)",
    )
    parser.add_argument("--story-only", action="store_true")
    parser.add_argument("--story-tokens", type=int, default=STORY_CONTEXT_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--plan", action="store_true", help="Print discover steps")
    parser.add_argument("--parse-log", type=Path, help="Print scraped KV GiB from a serve log")
    parser.add_argument(
        "--parse-log-tokens",
        type=Path,
        help="Print scraped GPU KV token count from a serve log",
    )
    parser.add_argument(
        "--record-start-fail",
        action="store_true",
        help="Append a Baseline start-fail story row (serve did not start)",
    )
    args = parser.parse_args(argv)

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
        tokens = DEFAULT_CONTEXT_TOKENS
        if args.context_tokens.strip():
            tokens = tuple(
                int(part) for part in args.context_tokens.split(",") if part.strip()
            )
        if args.max_model_len > 0:
            tokens = context_bins_for_max_len(args.max_model_len, tokens)
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
            )
        else:
            rows = probe_grid(
                url=args.probe_url,
                model=args.model,
                panel=args.panel,
                kv_cache_gib=args.kv_cache_gib,
                context_tokens=tokens,
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
    if args.out:
        render_heatmap(rows, args.out)
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
