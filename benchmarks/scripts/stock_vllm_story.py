#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stock vLLM A/B story: VRAM vs context, tok/s vs concurrency, long-ctx ITL.

Always pair vanilla ``vllm serve`` with ``isiro serve --target vllm``.
Solo ISIRO numbers are not a publishable story. 4B vs ~56 KiB is archived
token-id vs inactive KV bytes, not live nvidia-smi KV shrinking 14,000x.
TCO points are derived from measured tok/s and a documented GPU-hour price.
"""

from __future__ import annotations

import argparse
from typing import Any

STORY_SCHEMA = "isiro-benchmark-stock-vllm-story-v1"


def empty_story() -> dict[str, Any]:
    return {
        "schema": STORY_SCHEMA,
        "notes": (
            "Fill points from matched A/B runs. Do not paste 70B/8xH100 "
            "template cells. Weight ~29% is a separate measured footprint."
        ),
        "graph1_vram_vs_context": {"points": []},
        "graph2_toks_vs_concurrency": {"points": []},
        "long_ctx_itl": {"points": []},
        "tco": {"gpu_hour_usd": None, "points": []},
    }


def validate_story(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if doc.get("schema") != STORY_SCHEMA:
        errors.append(f"schema must be {STORY_SCHEMA}")
    for key in (
        "graph1_vram_vs_context",
        "graph2_toks_vs_concurrency",
        "long_ctx_itl",
        "tco",
    ):
        block = doc.get(key)
        if not isinstance(block, dict) or not isinstance(block.get("points"), list):
            errors.append(f"{key}.points must be a list")
    return errors


def derive_tco_point(
    *,
    concurrency: int,
    output_tok_s: float,
    gpu_hour_usd: float,
) -> dict[str, Any]:
    """Cost per 1M output tokens from measured tok/s. DERIVED, not a meter."""
    usd = float(gpu_hour_usd)
    toks = max(0.0, float(output_tok_s))
    cost_per_1m = None
    if toks > 0.0 and usd > 0.0:
        cost_per_1m = usd / (toks * 3600.0) * 1.0e6
    return {
        "concurrency": int(concurrency),
        "output_tok_s": toks,
        "gpu_hour_usd": usd,
        "cost_usd_per_1m_output_tokens": cost_per_1m,
        "kind": "derived",
    }


def render_story_markdown(doc: dict[str, Any]) -> list[str]:
    """Report section. Empty points: say the sweep is not in this run."""
    errors = validate_story(doc)
    lines = [
        "## Stock vLLM A/B (memory, concurrency, long context)",
        "",
        "Vanilla vLLM vs `isiro serve --target vllm` on the same host. "
        "Archived token ids are 4 B/token vs ~56 KiB BF16 KV per token "
        "(~14,336x bytes for **inactive** KV). That is not a 14,000x cut "
        "of the live vendor KV pool.",
        "",
    ]
    if errors:
        lines.append(f"Story JSON invalid: {'; '.join(errors)}")
        lines.append("")
        return lines
    g1 = list((doc.get("graph1_vram_vs_context") or {}).get("points") or [])
    g2 = list((doc.get("graph2_toks_vs_concurrency") or {}).get("points") or [])
    g3 = list((doc.get("long_ctx_itl") or {}).get("points") or [])
    tco = list((doc.get("tco") or {}).get("points") or [])
    if not g1:
        lines.append(
            "Graph 1 (VRAM / KV vs context) has no measured points in this run."
        )
        lines.append("")
    else:
        lines.extend(
            [
                "### Graph 1. KV and GPU memory vs context",
                "",
                "| Context | Baseline KV B | TIC KV B | "
                "Baseline nvidia-smi B | TIC nvidia-smi B | OOM |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for p in g1:
            oom_b = "yes" if p.get("oom_baseline") else "no"
            oom_t = "yes" if p.get("oom_tic") else "no"
            lines.append(
                f"| {p.get('context_tokens')} | {p.get('baseline_kv_bytes')} | "
                f"{p.get('tic_kv_bytes')} | {p.get('baseline_smi_bytes')} | "
                f"{p.get('tic_smi_bytes')} | B {oom_b} / T {oom_t} |"
            )
        lines.append("")
    if not g2:
        lines.append(
            "Graph 2 (output tok/s vs concurrency) has no measured points "
            "in this run."
        )
        lines.append("")
    else:
        lines.extend(
            [
                "### Graph 2. Output tok/s vs concurrency",
                "",
                "| Concurrency | Baseline tok/s | TIC tok/s |",
                "|---:|---:|---:|",
            ]
        )
        for p in g2:
            lines.append(
                f"| {p.get('concurrency')} | {p.get('baseline_tok_s')} | "
                f"{p.get('tic_tok_s')} |"
            )
        lines.append("")
    if not g3:
        lines.append(
            "Long-context ITL sweep (concurrency=1) has no measured points "
            "in this run. GPU TokenIDLog removes ID streaming; in-pool ITL "
            "still tracks the weight stack plus vendor FA on last-W."
        )
        lines.append("")
    else:
        lines.extend(
            [
                "### Long-context ITL (concurrency 1 unless noted)",
                "",
                "| Input tokens | Conc | Baseline ITL p50 ms | TIC ITL p50 ms |",
                "|---:|---:|---:|---:|",
            ]
        )
        for p in g3:
            lines.append(
                f"| {p.get('input_tokens')} | {p.get('concurrency', 1)} | "
                f"{p.get('baseline_itl_p50_ms')} | {p.get('tic_itl_p50_ms')} |"
            )
        lines.append("")
    if tco:
        lines.extend(
            [
                "### TCO scatter (derived from Graph 2 tok/s)",
                "",
                "| Conc | tok/s | GPU-hour USD | USD / 1M output tokens |",
                "|---:|---:|---:|---:|",
            ]
        )
        for p in tco:
            lines.append(
                f"| {p.get('concurrency')} | {p.get('output_tok_s')} | "
                f"{p.get('gpu_hour_usd')} | {p.get('cost_usd_per_1m_output_tokens')} |"
            )
        lines.append("")
        lines.append("TCO rows are derived, not nvidia-smi.")
        lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-empty",
        action="store_true",
        help="Print an empty story JSON template",
    )
    args = parser.parse_args()
    if args.print_empty:
        import json

        print(json.dumps(empty_story(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
