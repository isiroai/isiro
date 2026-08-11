#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the categorized public Markdown report from compact JSON."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_lib import load_json, write_json

_UUID_RE = re.compile(
    r"\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"
)
_PCI_RE = re.compile(r"\b0000:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]\b")
# Stop at ':' so Docker -v host:container mounts are not swallowed as one path.
_ABS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_<>])/(?:[^/\s\"':]+/)+[^/\s\"':]+")
_API_ROUTE_RE = re.compile(r"^/v\d+/")
_ARM_DIRS = ("baseline", "isiro")


def _human_recorded_at(value: Any) -> str:
    """Format summary recorded_at for the report header (UTC, no micros)."""
    if value is None:
        return "unknown time"
    text = str(value).strip()
    if not text:
        return "unknown time"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def number(value: Any, digits: int = 2) -> str:
    return "null" if value is None else f"{float(value):.{digits}f}"


def bytes_gb(value: Any) -> str:
    return "null" if value is None else f"{int(value) / 1e9:.2f} GB"


def bold_win(text: str, *, win: bool) -> str:
    return f"**{text}**" if win else text


def _tic_in_baseline_scale(
    tic: Any,
    *,
    baseline_total: Any,
    tic_total: Any,
) -> float | None:
    vals = (tic, baseline_total, tic_total)
    if not all(isinstance(v, (int, float)) for v in vals):
        return None
    b_tot = float(baseline_total)
    t_tot = float(tic_total)
    if b_tot <= 0 or t_tot <= 0:
        return None
    return float(tic) * (b_tot / t_tot)


def normalized_gpu_savings_pct(
    baseline: Any,
    tic: Any,
    *,
    baseline_total: Any,
    tic_total: Any,
) -> float | None:
    """``100 * (1 - TIC_in_baseline_scale / Baseline)`` (less is better)."""
    if not isinstance(baseline, (int, float)) or float(baseline) <= 0:
        return None
    tic_scaled = _tic_in_baseline_scale(
        tic, baseline_total=baseline_total, tic_total=tic_total
    )
    if tic_scaled is None:
        return None
    return (1.0 - tic_scaled / float(baseline)) * 100.0


def normalized_gpu_gain_pct(
    baseline: Any,
    tic: Any,
    *,
    baseline_total: Any,
    tic_total: Any,
) -> float | None:
    """``100 * (TIC_in_baseline_scale / Baseline - 1)`` (more is better)."""
    if not isinstance(baseline, (int, float)) or float(baseline) <= 0:
        return None
    tic_scaled = _tic_in_baseline_scale(
        tic, baseline_total=baseline_total, tic_total=tic_total
    )
    if tic_scaled is None:
        return None
    return (tic_scaled / float(baseline) - 1.0) * 100.0


def _norm_pct_cell(
    pct: float | None,
    *,
    win: bool,
    more_is_better: bool = False,
) -> str:
    if pct is None:
        return "null"
    if more_is_better:
        ratio = 1.0 + float(pct) / 100.0
        text = f"{number(pct)}% ({number(ratio)}x)"
    else:
        text = f"{number(pct)}%"
    return bold_win(text, win=win)


def command_block(commands: dict[str, Any], variant: str) -> list[str]:
    values = commands.get(variant) or []
    if values and isinstance(values[0], list):
        values = values[0]
    return [str(item) for item in values]


def _bit_exact_cell(ok: Any) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return str(ok).lower()


def _graph_mode(commands: dict[str, Any]) -> str:
    enforce_eager = commands.get("enforce_eager")
    baseline_command = command_block(commands, "baseline_serve")
    if enforce_eager is None and baseline_command:
        enforce_eager = "--enforce-eager" in baseline_command
    if enforce_eager is None:
        enforce_eager = True
    return "eager" if enforce_eager else "graphs"


def _decode_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in summary.get("profiles") or [] if int(row["output_tokens"]) > 64
    ]


def _latency_row(
    label: str,
    baseline: dict[str, Any],
    tic: dict[str, Any],
    key: str,
    *,
    digits: int = 2,
) -> str:
    b = baseline.get(key)
    t = tic.get(key)
    return f"| {label} | {number(b, digits)} | {number(t, digits)} |"


def _section_intro(
    summary: dict[str, Any],
    graph_mode: str,
    *,
    has_companion_graphs: bool,
    has_equal_batch: bool = False,
) -> list[str]:
    del graph_mode, has_companion_graphs, has_equal_batch
    model = str(summary.get("model") or "")
    return [
        f"# ISIRO Benchmark Report: `{model}`",
        "",
        (
            f"`{summary['precision']}` | `{summary['system_id']}` | "
            f"`{summary['isiro_format']}` | {_human_recorded_at(summary.get('recorded_at'))}"
        ),
        "",
        "Tooling: **`vllm bench serve`**.",
        "",
        (
            "- **Baseline:** vLLM (`vllm serve` / Docker "
            "`vllm/vllm-openai`)"
        ),
        "- **TIC:** `isiro serve … --target vllm` (same vLLM version)",
        "",
    ]


def _serve_output_match_row(correctness: dict[str, Any]) -> str:
    ok = correctness.get("serve_output_match_ok")
    if ok is None:
        return "| Serve output match | - | not run |"
    status = _bit_exact_cell(ok)
    matched = correctness.get("serve_output_match_matched")
    total = correctness.get("serve_output_match_prompt_count")
    status_cell = bold_win(status, win=status == "PASS")
    if isinstance(matched, int) and isinstance(total, int) and total > 0:
        if ok is True:
            detail = f"{matched}/{total} prompts, temp=0, token IDs equal"
        else:
            detail = f"{matched}/{total} prompts matched (temp=0 token IDs)"
        status_cell = f"{status_cell} ({detail})"
    return f"| Serve output match | - | {status_cell} |"


def _section_a(
    footprint: dict[str, Any],
    correctness: dict[str, Any],
) -> list[str]:
    baseline_fp = footprint["baseline"]
    tic_fp = footprint["tic"]
    verify_mode = str(correctness.get("verify_mode") or "integrity")
    if verify_mode == "reference":
        verify_label = "Weight bit-exactness"
        verify_cmd = "`isiro verify -r`"
        verify_status = _bit_exact_cell(correctness.get("bit_exact_ok"))
    else:
        verify_label = "Integrity"
        verify_cmd = "`isiro verify`"
        verify_status = _bit_exact_cell(correctness.get("integrity_ok"))
    savings = footprint.get("on_disk_savings_pct_derived")
    footprint_tic = bold_win(
        bytes_gb(tic_fp.get("on_disk_bytes_measured")), win=True
    )
    if savings is not None:
        footprint_tic = (
            f"{footprint_tic} "
            f"({bold_win(f'{number(savings)}% smaller', win=True)})"
        )
    status_cell = bold_win(verify_status, win=verify_status == "PASS")
    return [
        "## Correctness",
        "",
        "| Metric | Baseline | TIC |",
        "|--------|----------|-----|",
        (
            "| On-disk model size | "
            f"{bytes_gb(baseline_fp.get('on_disk_bytes_measured'))} | "
            f"{footprint_tic} |"
        ),
        f"| {verify_label} | baseline weights | {status_cell} ({verify_cmd}) |",
        _serve_output_match_row(correctness),
        "",
    ]


def _section_b(
    footprint: dict[str, Any],
    kv: dict[str, Any],
    capacity: dict[str, Any] | None = None,
    *,
    prefix: str = "1",
) -> list[str]:
    del capacity  # reserved for callers; B no longer branches on probe source
    baseline_fp = footprint["baseline"]
    tic_fp = footprint["tic"]
    b_non = baseline_fp.get("weights_and_non_kv_gpu_memory_bytes_derived")
    t_non = tic_fp.get("weights_and_non_kv_gpu_memory_bytes_derived")
    b_load = baseline_fp.get("model_loading_bytes_reported")
    t_load = tic_fp.get("model_loading_bytes_reported")
    b_kv = kv.get("baseline_memory_bytes_measured")
    t_kv = kv.get("tic_memory_bytes_measured")
    b_proc = baseline_fp.get("gpu_process_memory_after_load_bytes_measured")
    t_proc = tic_fp.get("gpu_process_memory_after_load_bytes_measured")
    non_kv_win = (
        isinstance(b_non, (int, float))
        and isinstance(t_non, (int, float))
        and float(t_non) < float(b_non)
    )
    load_win = (
        isinstance(b_load, (int, float))
        and isinstance(t_load, (int, float))
        and float(t_load) < float(b_load)
    )
    kv_win = (
        isinstance(b_kv, (int, float))
        and isinstance(t_kv, (int, float))
        and float(t_kv) > float(b_kv)
    )
    load_norm = normalized_gpu_savings_pct(
        b_load, t_load, baseline_total=b_proc, tic_total=t_proc
    )
    non_norm = normalized_gpu_savings_pct(
        b_non, t_non, baseline_total=b_proc, tic_total=t_proc
    )
    kv_norm = normalized_gpu_gain_pct(
        b_kv, t_kv, baseline_total=b_proc, tic_total=t_proc
    )
    return [
        f"## {prefix}A. GPU memory",
        "",
        (
            "Non-KV and KV cache are vLLM-reported. Total GPU memory is "
            "nvidia-smi process usage after load."
        ),
        "",
        (
            "Norm savings % scales TIC to the Baseline total GPU, then uses "
            "`1 - TIC/Baseline` for Loaded model size and Non-KV. For KV it "
            "uses `TIC/Baseline - 1` as a % gain (and the matching `x` ratio)."
        ),
        "",
        "| Metric | Baseline | TIC | Norm savings % |",
        "|--------|----------|-----|----------------|",
        (
            "| Loaded model size | "
            f"{bytes_gb(b_load)} | "
            f"{bold_win(bytes_gb(t_load), win=load_win)} | "
            f"{_norm_pct_cell(load_norm, win=load_win)} |"
        ),
        (
            "| Non-KV GPU memory | "
            f"{bytes_gb(b_non)} | "
            f"{bold_win(bytes_gb(t_non), win=non_kv_win)} | "
            f"{_norm_pct_cell(non_norm, win=non_kv_win)} |"
        ),
        (
            "| KV cache | "
            f"{bytes_gb(b_kv)} | "
            f"{bold_win(bytes_gb(t_kv), win=kv_win)} | "
            f"{_norm_pct_cell(kv_norm, win=kv_win, more_is_better=True)} |"
        ),
        (
            "| Total GPU memory | "
            f"{bytes_gb(b_proc)} | {bytes_gb(t_proc)} | |"
        ),
        "",
    ]


def _section_c(
    capacity: dict[str, Any],
    kv: dict[str, Any],
    *,
    prefix: str = "1",
) -> list[str]:
    b_seqs = (capacity.get("baseline") or {}).get("max_num_seqs")
    t_seqs = (capacity.get("tic") or {}).get("max_num_seqs")
    scale_mode = str(capacity.get("seqs_scale_mode") or "")
    if scale_mode == "kv_measured":
        scale_blurb = (
            "TIC `max_num_seqs` = round(baseline × TIC/Baseline KV token "
            "capacity) from a matched vLLM KV measurement."
        )
    elif scale_mode == "kv_estimate":
        scale_blurb = (
            "TIC `max_num_seqs` = round(baseline × estimated KV ratio), "
            "where estimated KV adds on-disk weight savings to Baseline KV."
        )
    else:
        scale_blurb = "TIC `max_num_seqs` matches the configured baseline batch."
    batch_ratio = f"{number(capacity.get('batch_ratio_derived'))}x"
    kv_token_ratio_val = capacity.get("kv_token_capacity_ratio_measured")
    kv_token_ratio = f"{number(kv_token_ratio_val)}x"
    kv_token_win = (
        isinstance(kv_token_ratio_val, (int, float))
        and float(kv_token_ratio_val) > 1.0
    )
    seqs_tic = (
        f"{bold_win(number(t_seqs, 0), win=True)} "
        f"({bold_win(batch_ratio, win=True)})"
    )
    tokens_tic = (
        f"{bold_win(number(kv.get('tic_tokens_measured'), 0), win=kv_token_win)} "
        f"({bold_win(kv_token_ratio, win=kv_token_win)})"
    )
    if str(capacity.get("kv_measured_source") or "") == "timed_asymmetric":
        tokens_tic = f"{tokens_tic}; asymmetric timed seqs"
    return [
        f"## {prefix}B. Capacity",
        "",
        scale_blurb,
        "",
        "KV token capacity is vLLM-reported (`GPU KV cache size` in serve logs).",
        "",
        "| Metric | Baseline | TIC |",
        "|--------|----------|-----|",
        f"| `max_num_seqs` | {number(b_seqs, 0)} | {seqs_tic} |",
        (
            "| KV token capacity | "
            f"{number(kv.get('baseline_tokens_measured'), 0)} | "
            f"{tokens_tic} |"
        ),
        "",
    ]


def _tok_per_non_kv_gb(tps: Any, non_kv_bytes: Any) -> float | None:
    if not isinstance(tps, (int, float)) or not isinstance(non_kv_bytes, (int, float)):
        return None
    if float(tps) <= 0 or float(non_kv_bytes) <= 0:
        return None
    return float(tps) / (float(non_kv_bytes) / 1e9)


def _section_d(
    capacity: dict[str, Any] | None,
    decode_rows: list[dict[str, Any]],
    footprint: dict[str, Any],
    *,
    prefix: str = "1",
) -> list[str]:
    lines = [
        f"## {prefix}C. Generation (input 32 / output 256)",
        "",
        "| Metric | Baseline | TIC |",
        "|--------|----------|-----|",
    ]
    tps = (capacity or {}).get("decode_output_tokens_per_sec") or {}
    b_non = footprint["baseline"].get("weights_and_non_kv_gpu_memory_bytes_derived")
    t_non = footprint["tic"].get("weights_and_non_kv_gpu_memory_bytes_derived")
    if capacity is not None and tps:
        b_tps = tps.get("baseline")
        t_tps = tps.get("tic")
        lines.append(
            "| Output tok/s | "
            f"{number(b_tps)} | "
            f"{bold_win(number(t_tps), win=True)} |"
        )
        b_eff = _tok_per_non_kv_gb(b_tps, b_non)
        t_eff = _tok_per_non_kv_gb(t_tps, t_non)
        if b_eff is not None and t_eff is not None:
            lines.append(
                "| † tok/s per Non-KV GB | "
                f"{number(b_eff)} | {number(t_eff)} |"
            )
    elif decode_rows:
        row = decode_rows[0]
        b_tps = row["baseline"]["throughput"]["output_tokens_per_sec"]
        t_tps = row["tic"]["throughput"]["output_tokens_per_sec"]
        lines.append(
            f"| Output tok/s | {number(b_tps)} | {number(t_tps)} |"
        )
        b_eff = _tok_per_non_kv_gb(b_tps, b_non)
        t_eff = _tok_per_non_kv_gb(t_tps, t_non)
        if b_eff is not None and t_eff is not None:
            lines.append(
                "| † tok/s per Non-KV GB | "
                f"{number(b_eff)} | {number(t_eff)} |"
            )
    for row in decode_rows:
        b_lat = row["baseline"]["latency_ms"]
        t_lat = row["tic"]["latency_ms"]
        for label, key in (
            ("ITL p50 (ms)", "itl_p50"),
            ("ITL p95 (ms)", "itl_p95"),
            ("ITL p99 (ms)", "itl_p99"),
            ("TPOT p50 (ms)", "tpot_p50"),
            ("TPOT p95 (ms)", "tpot_p95"),
            ("TPOT p99 (ms)", "tpot_p99"),
        ):
            lines.append(_latency_row(label, b_lat, t_lat, key))
    lines.extend(
        [
            "",
            (
                "† Derived: output tok/s ÷ Non-KV GPU memory (GB). "
                "Physical meaning: output tokens per second per GB of "
                "Non-KV GPU memory; a smaller Non-KV slice that still "
                "delivers high tok/s scores higher."
            ),
            "",
        ]
    )
    return lines


def _section_e(decode_rows: list[dict[str, Any]], *, prefix: str = "1") -> list[str]:
    lines = [
        f"## {prefix}D. TTFT",
        "",
        (
            f"TTFT is separated from generation ({prefix}C) because it is more "
            f"sensitive to the higher concurrency in {prefix}B. With more "
            "sequences in flight, new requests wait longer for the first token. "
            "That is an expected cost of the capacity setting, not a "
            "single-request prefill claim."
        ),
        "",
        "| Metric | Baseline | TIC |",
        "|--------|----------|-----|",
    ]
    for row in decode_rows:
        b_lat = row["baseline"]["latency_ms"]
        t_lat = row["tic"]["latency_ms"]
        for label, key in (
            ("TTFT p50 (ms)", "ttft_p50"),
            ("TTFT p95 (ms)", "ttft_p95"),
            ("TTFT p99 (ms)", "ttft_p99"),
        ):
            b = b_lat.get(key)
            t = t_lat.get(key)
            lines.append(
                f"| {label} | {number(b, 2)} | {number(t, 2)} |"
            )
    lines.append("")
    return lines


def _section_f(equal_batch: dict[str, Any], *, prefix: str = "1") -> list[str]:
    tps = equal_batch.get("decode_output_tokens_per_sec") or {}
    latency = equal_batch.get("latency_ms") or {}
    b_lat = latency.get("baseline") or {}
    t_lat = latency.get("tic") or {}
    lines = [
        f"## {prefix}E. Equal batch",
        "",
        "Same `max_num_seqs` and concurrency on both sides.",
        "",
        "| Metric | Baseline | TIC |",
        "|--------|----------|-----|",
        (
            f"| `max_num_seqs` / concurrency | "
            f"{number(equal_batch.get('max_num_seqs'), 0)} / "
            f"{number(equal_batch.get('max_concurrency'), 0)} | "
            f"{number(equal_batch.get('max_num_seqs'), 0)} / "
            f"{number(equal_batch.get('max_concurrency'), 0)} |"
        ),
        (
            "| Output tok/s | "
            f"{number(tps.get('baseline'))} | "
            f"{number(tps.get('tic'))} |"
        ),
    ]
    if b_lat and t_lat:
        for label, key in (
            ("ITL p50 (ms)", "itl_p50"),
            ("ITL p95 (ms)", "itl_p95"),
            ("ITL p99 (ms)", "itl_p99"),
            ("TPOT p50 (ms)", "tpot_p50"),
            ("TPOT p95 (ms)", "tpot_p95"),
            ("TPOT p99 (ms)", "tpot_p99"),
            ("TTFT p50 (ms)", "ttft_p50"),
            ("TTFT p95 (ms)", "ttft_p95"),
            ("TTFT p99 (ms)", "ttft_p99"),
        ):
            lines.append(_latency_row(label, b_lat, t_lat, key))
    lines.append("")
    return lines


def _mode_sections(summary: dict[str, Any], *, prefix: str = "1") -> list[str]:
    """Sections A-E for one graph mode (SSOT for Graph ON/OFF bodies)."""
    footprint = summary.get("footprint") or {}
    kv = summary.get("kv_cache") or {}
    capacity = (
        summary.get("capacity")
        if isinstance(summary.get("capacity"), dict)
        else None
    )
    equal_batch = (
        summary.get("equal_batch")
        if isinstance(summary.get("equal_batch"), dict)
        else None
    )
    decode_rows = _decode_rows(summary)
    experiment_kind = str(summary.get("experiment_kind") or "equal-batch")
    lines: list[str] = []
    lines.extend(_section_b(footprint, kv, capacity, prefix=prefix))
    if capacity is not None and experiment_kind == "capacity":
        lines.extend(_section_c(capacity, kv, prefix=prefix))
        lines.extend(_section_d(capacity, decode_rows, footprint, prefix=prefix))
        lines.extend(_section_e(decode_rows, prefix=prefix))
    else:
        lines.extend(_section_d(None, decode_rows, footprint, prefix=prefix))
        lines.extend(_section_e(decode_rows, prefix=prefix))
    if equal_batch is not None:
        lines.extend(_section_f(equal_batch, prefix=prefix))
    return lines


def _section_g(
    summary: dict[str, Any],
    environment: dict[str, Any],
    commands: dict[str, Any],
    graph_mode: str,
    capacity: dict[str, Any] | None,
    *,
    has_companion_graphs: bool,
) -> list[str]:
    reused = commands.get("baseline_reused_from")
    if has_companion_graphs:
        graph_cell = "ON (CUDA graphs); OFF (eager)"
    else:
        graph_cell = (
            "ON (CUDA graphs)" if graph_mode == "graphs" else "OFF (eager)"
        )
    # capacity scaling is explained in the Capacity sections, not Config.
    _ = capacity
    lines = [
        "## Config",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| System | `{summary.get('system_id', '')}` |",
        f"| Graph modes | {graph_cell} |",
        f"| vLLM | {environment.get('vllm_version', '')} |",
        (
            f"| GPU | {environment.get('gpu_model', '')} x "
            f"{environment.get('gpu_count', 0)} |"
        ),
        (
            f"| Driver / CUDA | {environment.get('driver_version', '')} / "
            f"{environment.get('cuda_version', '')} |"
        ),
    ]
    if reused:
        lines.append(f"| Baseline reused from | `{reused}` |")
    lines.extend(
        [
            "",
            (
                "Fairness check: **PASS**."
                if summary.get("fairness", {}).get("passed")
                else "Fairness check: **FAIL**."
            ),
            "",
        ]
    )
    return lines


def _order_graph_dirs(
    run_dir: Path, companion_run_dir: Path | None
) -> tuple[Path, Path | None]:
    """Prefer Graph ON first; companion is Graph OFF when present."""
    if companion_run_dir is None:
        return run_dir, None
    primary_mode = _graph_mode(load_json(run_dir / "commands.json"))
    companion_mode = _graph_mode(load_json(companion_run_dir / "commands.json"))
    if primary_mode == "graphs" and companion_mode == "eager":
        return run_dir, companion_run_dir
    if primary_mode == "eager" and companion_mode == "graphs":
        return companion_run_dir, run_dir
    # Same mode or unknown: keep caller order; no dual-mode body.
    return run_dir, None


def render(run_dir: Path, companion_run_dir: Path | None = None) -> str:
    open_dir, collapsed_dir = _order_graph_dirs(run_dir, companion_run_dir)
    summary = load_json(open_dir / "summary.json")
    environment = load_json(open_dir / "environment.json")
    commands = load_json(open_dir / "commands.json")
    footprint = summary["footprint"]
    correctness = summary["correctness"]
    graph_mode = _graph_mode(commands)
    has_companion = collapsed_dir is not None
    capacity = (
        summary.get("capacity")
        if isinstance(summary.get("capacity"), dict)
        else None
    )
    equal_batch = (
        summary.get("equal_batch")
        if isinstance(summary.get("equal_batch"), dict)
        else None
    )

    lines = _section_intro(
        summary,
        graph_mode,
        has_companion_graphs=has_companion,
        has_equal_batch=equal_batch is not None,
    )
    lines.extend(_section_a(footprint, correctness))
    if has_companion:
        lines.extend(
            [
                "## Graph ON (CUDA graphs)",
                "",
                "CUDA graphs on (product default; `--graph-on`).",
                "",
            ]
        )
        lines.extend(_mode_sections(summary, prefix="1"))
        collapsed_summary = load_json(collapsed_dir / "summary.json")
        lines.extend(
            [
                "## Graph OFF (eager)",
                "",
                "CUDA graphs off (`--graph-off`).",
                "",
            ]
        )
        lines.extend(_mode_sections(collapsed_summary, prefix="2"))
    else:
        if graph_mode == "eager":
            lines.extend(
                [
                    "## Graph OFF (eager)",
                    "",
                    "CUDA graphs off (`--graph-off`).",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## Graph ON (CUDA graphs)",
                    "",
                    "CUDA graphs on (product default; `--graph-on`).",
                    "",
                ]
            )
        lines.extend(_mode_sections(summary, prefix="1"))
    lines.extend(
        _section_g(
            summary,
            environment,
            commands,
            graph_mode,
            capacity,
            has_companion_graphs=has_companion,
        )
    )
    return "\n".join(lines)


def sanitize(value: Any, scratch: Path) -> Any:
    """Redact host paths and GPU identity before writing a public report."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"uuid", "pci_bus_id"}:
                cleaned[key] = "<REDACTED>"
                continue
            cleaned[key] = sanitize(item, scratch)
        return cleaned
    if isinstance(value, list):
        return [sanitize(item, scratch) for item in value]
    if not isinstance(value, str):
        return value
    text = value.replace(str(scratch), "<SCRATCH_RUN>")
    text = _UUID_RE.sub("<GPU_UUID>", text)
    text = _PCI_RE.sub("<PCI_BUS_ID>", text)
    home = str(Path.home())
    if home in text:
        text = text.replace(home, "<OPERATOR_PATH>")

    def _path_repl(match: re.Match[str]) -> str:
        path = match.group(0)
        if path.startswith("<") or "OPERATOR_PATH" in path:
            return path
        if _API_ROUTE_RE.match(path):
            return path
        name = Path(path.rstrip("/")).name
        return f"<OPERATOR_PATH>/{name}" if name else "<OPERATOR_PATH>"

    text = _ABS_PATH_RE.sub(_path_repl, text)
    text = re.sub(
        r"<OPERATOR_PATH>(?:/[^/\s\"':]+)+/([^/\s\"':]+)",
        r"<OPERATOR_PATH>/\1",
        text,
    )
    text = text.replace("<OPERATOR_PATH><OPERATOR_PATH>", "<OPERATOR_PATH>")
    return text


def _copy_sanitized_json(source: Path, dest: Path, scratch: Path) -> None:
    write_json(dest, sanitize(load_json(source), scratch))


def _stage_sanitized_tree(
    run_dir: Path,
    temp: Path,
    companion_run_dir: Path | None = None,
) -> Path | None:
    """Sanitize run JSON into temp for render; return staged companion or None."""
    for name in ("environment.json", "commands.json", "summary.json", "verify.json"):
        src = run_dir / name
        if src.is_file():
            _copy_sanitized_json(src, temp / name, run_dir)
    for variant in _ARM_DIRS:
        for source in sorted((run_dir / variant).glob("*.json")):
            _copy_sanitized_json(
                source, temp / variant / source.name, run_dir
            )
    equal_batch = run_dir / "equal_batch"
    if equal_batch.is_dir():
        for variant in _ARM_DIRS:
            for source in sorted((equal_batch / variant).glob("*.json")):
                _copy_sanitized_json(
                    source,
                    temp / "equal_batch" / variant / source.name,
                    run_dir,
                )
    if companion_run_dir is None or not companion_run_dir.is_dir():
        return None
    staged = temp / "companion"
    staged.mkdir(parents=True, exist_ok=True)
    for name in (
        "environment.json",
        "commands.json",
        "summary.json",
        "verify.json",
    ):
        src = companion_run_dir / name
        if src.is_file():
            _copy_sanitized_json(src, staged / name, companion_run_dir)
    return staged


def write_report(
    run_dir: Path,
    output: Path,
    *,
    companion_run_dir: Path | None = None,
) -> Path:
    """Render a sanitized report.md to ``output`` (scratch JSON stays raw)."""
    temp = Path(tempfile.mkdtemp(prefix=".isiro-bench-report-"))
    try:
        staged_companion = _stage_sanitized_tree(
            run_dir, temp, companion_run_dir=companion_run_dir
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render(temp, companion_run_dir=staged_companion),
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--companion-run-dir",
        type=Path,
        default=None,
        help="Optional second graph-mode run (eager+graphs merge into one report).",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.run_dir / "report.md"
    write_report(
        args.run_dir,
        output,
        companion_run_dir=args.companion_run_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
