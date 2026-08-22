#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-a-haystack grid: JSONL probe + heatmap (TIC vs baseline).

Protocol is single-needle (Kamradt / RULER style).
Each panel is one KV cache size. X is context tokens. Y is needle depth %.
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

    Prefer vLLM ``Available KV cache memory``. Product TIC logs may only
    print ``vendor_pool_bytes`` when DEBUG is off.
    """
    import re

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


def gib_to_bytes(kv_cache_gib: float) -> int:
    return int(round(float(kv_cache_gib) * (1024**3)))


def context_bins_for_max_len(
    max_model_len: int,
    bins: Iterable[int] = DEFAULT_CONTEXT_TOKENS,
) -> tuple[int, ...]:
    cap = max(1, int(max_model_len))
    kept = tuple(int(b) for b in bins if int(b) <= cap)
    return kept or (cap,)


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
        "TIC product serve; scrape GPU KV cache size; full grid (1 serve)",
        "Baseline default fill; scrape GPU KV cache size; full grid (1 serve)",
        "Refuse if TIC KV is not smaller than Baseline fill (any model)",
        "Baseline same size as TIC; full grid (1 serve)",
        "Baseline double KV until first recall; full grid only on the hit serve",
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
        if alias is None or not _has_kv_cache_gib(item) or "depth_percent" not in item:
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


def render_heatmap(rows: list[dict[str, Any]], dest: Path) -> Path:
    """JSONL pivot to seaborn heatmap. Product path is PNG."""
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
    fig.suptitle(
        "Fact retrieval by KV cache and context length (needle-in-a-haystack)",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.94, top=0.84, bottom=0.18, wspace=0.16)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    return dest


def _mock_hit(*, panel: str, kv_cache_gib: float, context_tokens: int) -> bool:
    if panel == "tic":
        return True
    if abs(kv_cache_gib - 1.09) < 1e-6:
        return context_tokens <= 2000
    if abs(kv_cache_gib - 4.00) < 1e-6:
        return context_tokens <= 16000
    return True


def mock_rows() -> list[dict[str, Any]]:
    """Qwen-class KV cache story for the report mock figure."""
    setups = (("tic", 1.09), ("vanilla", 1.09), ("vanilla", 4.00), ("vanilla", 13.65))
    rows: list[dict[str, Any]] = []
    for panel, kv_cache_gib in setups:
        for depth in DEFAULT_DEPTHS:
            for context_tokens in DEFAULT_CONTEXT_TOKENS:
                rows.append(
                    row(
                        panel=panel,
                        kv_cache_gib=kv_cache_gib,
                        depth_percent=depth,
                        context_tokens=context_tokens,
                        hit=_mock_hit(
                            panel=panel,
                            kv_cache_gib=kv_cache_gib,
                            context_tokens=context_tokens,
                        ),
                    )
                )
    return rows


def kv_sweep_table_lines(rows: list[dict[str, Any]]) -> list[str]:
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
    tic_fixed = tic_sizes[0] if tic_sizes else None
    lines = [
        "### KV cache",
        "",
        "TIC is one measured KV size. Baseline rows are the same lock, then "
        "any larger locks we ran, including default fill.",
        "",
        "Same long prompt, needle, and question on every setup.",
        "",
        "| Setup | KV cache | Size (GiB) | Response completes | Fact retrieved (needle) | Rejected or OOM |",
        "|-------|----------|------------|--------------------|-------------------------|-----------------|",
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
        completes = any(item.get("response_completes", True) for item in fold)
        recalled = any(item.get("hit") for item in fold)
        refused = any(item.get("refuse") or item.get("oom") for item in items)
        if panel == "tic":
            setup, policy = "**TIC**", "Product"
            size_cell = f"{size:g}†"
        elif tic_fixed is not None and abs(size - tic_fixed) < 1e-6:
            setup, policy = "**Baseline**", "Same size as TIC"
            size_cell = f"{size:g}"
        elif fill_size is not None and abs(size - fill_size) < 1e-6:
            setup, policy = "**Baseline**", "Default fill"
            size_cell = f"{size:g}"
        else:
            setup, policy = "**Baseline**", "Larger KV"
            size_cell = f"{size:g}"
        flag = lambda ok: "yes" if ok else "no"
        lines.append(
            f"| {setup} | {policy} | {size_cell} | {flag(completes)} | "
            f"**{flag(recalled)}** | {flag(refused)} |"
        )
    lines.append("")
    lines.extend(kv_overflow_footnote_lines())
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
    heading: str = "## B. Needle-in-a-haystack (context tokens × depth)",
) -> list[str]:
    return [
        heading,
        "",
        (
            "Each panel is one KV cache size. X is context tokens. "
            "Y is needle depth % (0 = start of prompt, 100 = end). "
            "Color is recall."
        ),
        "",
        f"![Fact retrieval by KV cache (needle-in-a-haystack)]({image_rel})",
        "",
    ]


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
    args = parser.parse_args(argv)

    if args.plan:
        for line in niah_discover_steps():
            print(line)
        return 0
    if args.parse_log:
        print(f"{parse_kv_cache_gib(args.parse_log.read_text(encoding='utf-8', errors='replace')):.4f}")
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
