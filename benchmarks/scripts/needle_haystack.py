#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-a-haystack grid: JSONL probe + paired heatmap (TIC vs baseline).

Protocol is single-needle (Kamradt / RULER style).
X is KV cache size (GiB), not advertised context length. Y is needle depth %.
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

DEFAULT_DEPTHS = (10, 50, 90)


def score_hit(completion: str, needle: str = NEEDLE) -> bool:
    """Exact needle, or the magic number if the needle contains one."""
    text = (completion or "").strip()
    if not text:
        return False
    if needle in text:
        return True
    digits = "".join(ch for ch in needle if ch.isdigit())
    return bool(digits) and digits in text


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
    if "kv_cache_gib" in item:
        return float(item["kv_cache_gib"])
    return float(item["dest_gib"])


def _has_kv_cache_gib(item: dict[str, Any]) -> bool:
    return "kv_cache_gib" in item or "dest_gib" in item


def row(
    *,
    panel: str,
    kv_cache_gib: float,
    depth_percent: float,
    hit: bool,
    continue_ok: bool = True,
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
        "continue_ok": bool(continue_ok),
        "refuse": bool(refuse),
        "oom": bool(oom),
    }
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


def probe_grid(
    *,
    url: str,
    model: str,
    panel: str,
    kv_cache_gib: float,
    depths: Iterable[float] = DEFAULT_DEPTHS,
    approx_chars: int = 8000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth in depths:
        messages = build_messages(
            depth_percent=float(depth), approx_chars=approx_chars
        )
        refuse = False
        oom = False
        continue_ok = True
        completion = ""
        try:
            completion = _chat_completion(url, model=model, messages=messages)
        except Exception as exc:
            continue_ok = False
            text = str(exc).lower()
            refuse = "http" in text or "refuse" in text
            oom = "out of memory" in text or "oom" in text
        rows.append(
            row(
                panel=panel,
                kv_cache_gib=kv_cache_gib,
                depth_percent=float(depth),
                hit=score_hit(completion),
                continue_ok=continue_ok,
                refuse=refuse,
                oom=oom,
                extra={"completion_chars": len(completion)},
            )
        )
    return rows


def _panel_label(panel: str) -> str:
    name = str(panel or "").lower()
    if name in ("tic", "isiro"):
        return "TIC"
    if name in ("vanilla", "baseline"):
        return "Baseline"
    return str(panel)


def _cell_color(score: float) -> str:
    if score >= 0.85:
        return "#2ca02c"
    if score <= 0.15:
        return "#d62728"
    return "#ffbf00"


def render_svg(rows: list[dict[str, Any]], dest: Path) -> Path:
    """Paired heatmap SVG. No extra dependency."""
    panels = ("tic", "vanilla")
    by_panel: dict[str, list[dict[str, Any]]] = {name: [] for name in panels}
    for item in rows:
        name = str(item.get("panel") or "").lower()
        if name in ("isiro", "tic"):
            by_panel["tic"].append(item)
        elif name in ("vanilla", "baseline"):
            by_panel["vanilla"].append(item)
    sizes = sorted(
        {
            round(_kv_cache_gib(item), 4)
            for item in rows
            if _has_kv_cache_gib(item)
        }
    )
    depths = sorted(
        {
            round(float(item["depth_percent"]), 4)
            for item in rows
            if "depth_percent" in item
        }
    )
    if not sizes or not depths:
        raise ValueError("rows need kv_cache_gib and depth_percent")
    lookup: dict[tuple[str, float, float], float] = {}
    for name, items in by_panel.items():
        for item in items:
            key = (
                name,
                round(_kv_cache_gib(item), 4),
                round(float(item["depth_percent"]), 4),
            )
            lookup[key] = float(item.get("score", 1.0 if item.get("hit") else 0.0))

    cell_w, cell_h = 56, 28
    left, top = 72, 36
    gap = 36
    grid_w = len(sizes) * cell_w
    grid_h = len(depths) * cell_h
    width = left + grid_w * 2 + gap + 24
    height = top + grid_h + 48
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="72" y="20" font-size="12" font-family="sans-serif">'
        "Needle-in-a-haystack: KV cache GiB x needle depth</text>",
    ]
    for p_i, panel in enumerate(panels):
        ox = left + p_i * (grid_w + gap)
        parts.append(
            f'<text x="{ox}" y="{top - 8}" font-size="11" '
            f'font-family="sans-serif">{_panel_label(panel)}</text>'
        )
        for x_i, kv_gib in enumerate(sizes):
            parts.append(
                f'<text x="{ox + x_i * cell_w + 4}" y="{top + grid_h + 14}" '
                f'font-size="9" font-family="sans-serif">{kv_gib:g}</text>'
            )
        for y_i, depth in enumerate(reversed(depths)):
            parts.append(
                f'<text x="{ox - 28}" y="{top + y_i * cell_h + 18}" '
                f'font-size="9" font-family="sans-serif">{depth:g}%</text>'
            )
            for x_i, kv_gib in enumerate(sizes):
                score = lookup.get((panel, kv_gib, depth))
                fill = "#cccccc" if score is None else _cell_color(score)
                x = ox + x_i * cell_w
                y = top + y_i * cell_h
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 2}" '
                    f'height="{cell_h - 2}" fill="{fill}" stroke="#333333"/>'
                )
    parts.append("</svg>")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return dest


def render_heatmap(rows: list[dict[str, Any]], dest: Path) -> Path:
    dest = Path(dest)
    if dest.suffix.lower() in {".png", ".pdf"}:
        try:
            return _render_matplotlib(rows, dest)
        except ImportError:
            svg = dest.with_suffix(".svg")
            return render_svg(rows, svg)
    return render_svg(rows, dest)


def _render_matplotlib(rows: list[dict[str, Any]], dest: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = sorted({round(_kv_cache_gib(r), 4) for r in rows})
    depths = sorted({round(float(r["depth_percent"]), 4) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for ax, panel in zip(axes, ("tic", "vanilla")):
        grid = np.full((len(depths), len(sizes)), float("nan"))
        aliases = ("tic", "isiro") if panel == "tic" else ("vanilla", "baseline")
        for item in rows:
            name = str(item.get("panel") or "").lower()
            if name not in aliases:
                continue
            yi = depths.index(round(float(item["depth_percent"]), 4))
            xi = sizes.index(round(_kv_cache_gib(item), 4))
            grid[yi, xi] = float(item.get("score", 1.0 if item.get("hit") else 0.0))
        im = ax.imshow(
            grid[::-1],
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlGn",
            aspect="auto",
        )
        ax.set_title(_panel_label(panel))
        ax.set_xticks(range(len(sizes)), [f"{d:g}" for d in sizes])
        ax.set_yticks(range(len(depths)), [f"{d:g}" for d in reversed(depths)])
        ax.set_xlabel("KV cache GiB")
        ax.set_ylabel("needle depth %")
        fig.colorbar(im, ax=ax, fraction=0.046)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def mock_rows() -> list[dict[str, Any]]:
    """Qwen-class KV cache story for the report mock figure."""
    baseline_sizes = (1.09, 4.00, 13.65)
    depths = DEFAULT_DEPTHS
    rows: list[dict[str, Any]] = []
    for depth in depths:
        rows.append(
            row(panel="tic", kv_cache_gib=1.09, depth_percent=depth, hit=True)
        )
        for kv_cache_gib in baseline_sizes:
            hit = kv_cache_gib > 1.09
            rows.append(
                row(
                    panel="vanilla",
                    kv_cache_gib=kv_cache_gib,
                    depth_percent=depth,
                    hit=hit,
                )
            )
    return rows


def report_lines(
    image_rel: str,
    *,
    heading: str = "## Needle-in-a-haystack (KV cache GiB × depth)",
) -> list[str]:
    return [
        heading,
        "",
        (
            "X is KV cache size (GiB). Y is needle depth % "
            "(0 = start of prompt, 100 = end). "
            "Color is recall score. TIC KV cache is fixed. Baseline KV cache grows."
        ),
        "",
        f"![Needle-in-a-haystack]({image_rel})",
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Needle-in-a-haystack grid probe and plot"
    )
    parser.add_argument("--jsonl", type=Path, help="Read or write JSONL rows")
    parser.add_argument("--out", type=Path, help="Heatmap SVG or PNG")
    parser.add_argument("--mock", action="store_true", help="Render mock Qwen-class rows")
    parser.add_argument("--probe-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="")
    parser.add_argument("--panel", default="tic")
    parser.add_argument("--kv-cache-gib", type=float, default=0.0)
    parser.add_argument("--approx-chars", type=int, default=8000)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    if args.mock:
        rows = mock_rows()
    elif args.probe_url:
        if not args.model or args.kv_cache_gib <= 0:
            raise SystemExit("--probe-url needs --model and --kv-cache-gib")
        rows = probe_grid(
            url=args.probe_url,
            model=args.model,
            panel=args.panel,
            kv_cache_gib=args.kv_cache_gib,
            approx_chars=args.approx_chars,
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
