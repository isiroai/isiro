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


def bytes_gib(value: Any) -> str:
    return "null" if value is None else f"{int(value) / float(1024**3):.2f} GiB"


def pool_gib(
    memory_bytes: Any,
    tokens: Any = None,
    *,
    peer_bytes: Any = None,
    peer_tokens: Any = None,
) -> float | None:
    """GiB for this serve's reserved GPU KV pool (1024**3)."""
    if isinstance(memory_bytes, (int, float)) and float(memory_bytes) > 0:
        return float(memory_bytes) / float(1024**3)
    if (
        isinstance(tokens, (int, float))
        and float(tokens) > 0
        and isinstance(peer_bytes, (int, float))
        and isinstance(peer_tokens, (int, float))
        and float(peer_bytes) > 0
        and float(peer_tokens) > 0
    ):
        return (float(tokens) * float(peer_bytes) / float(peer_tokens)) / float(
            1024**3
        )
    return None


def pct_of_baseline(ratio: Any) -> str:
    """Isiro / Baseline as a percent of Baseline (not an Nx multiplier)."""
    if not isinstance(ratio, (int, float)):
        return ""
    return f"{round(float(ratio) * 100):.0f}% of Baseline"


def _arm_col() -> str:
    """Customer product arm column label (TIC weights + optional GDX)."""
    return "Isiro"


def tokens_with_pool(
    tokens: Any,
    gib: float | None,
    *,
    extra: str = "",
    dagger: bool = False,
) -> str:
    parts = [number(tokens, 0)]
    bits: list[str] = []
    if gib is not None:
        bits.append(f"{gib:.2f} GiB KV")
    if extra:
        bits.append(extra)
    if bits:
        parts.append(f"({', '.join(bits)})")
    text = " ".join(parts)
    if dagger:
        text += "†"
    return text


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


def vllm_visible_budget(non_kv: Any, kv: Any) -> float | None:
    if not isinstance(non_kv, (int, float)) or not isinstance(kv, (int, float)):
        return None
    return float(non_kv) + float(kv)


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


def _compiler_runtime_versions(summary: dict[str, Any]) -> tuple[str, str]:
    compiler = str(
        summary.get("isiro_compiler")
        or summary.get("compiler")
        or summary.get("isiro_format")
        or ""
    )
    runtime = str(
        summary.get("isiro_runtime") or summary.get("runtime") or compiler
    )
    return compiler, runtime


def _section_intro(
    summary: dict[str, Any],
    commands: dict[str, Any] | None = None,
) -> list[str]:
    from needle_haystack import NEEDLE_COUNT

    model = str(summary.get("model") or "")
    compiler, runtime = _compiler_runtime_versions(summary)
    cmds = commands if isinstance(commands, dict) else {}
    gdx_kv = cmds.get("isiro_gdx_kv")
    kv_dtype = cmds.get("kv_cache_dtype")
    compared = [
        "Compared:",
        "",
        (
            "- **Baseline:** vLLM (`vllm serve` / Docker "
            "`vllm/vllm-openai`)"
            + (f" with `--kv-cache-dtype {kv_dtype}`" if kv_dtype else "")
        ),
        "- **Isiro:** `isiro serve … --target vllm` (TIC weights; same vLLM version)",
    ]
    if gdx_kv:
        compared[-1] = (
            f"- **Isiro:** `isiro serve … --target vllm` with TIC weights and "
            f"`ISIRO_GDX={gdx_kv}` (same vLLM version; cold KV compress on "
            "vendor attention pages)"
        )
    lines = [
        f"# ISIRO Benchmark Report: `{model}`",
        "",
        (
            f"`{summary['precision']}` | `{summary['system_id']}` | "
            f"compiler `{compiler}` | "
            f"runtime `{runtime}` | "
            f"{_human_recorded_at(summary.get('recorded_at'))}"
        ),
        "",
        "Tooling:",
        "",
        "- `vllm bench serve` (tok/s, ITL)",
        *(
            [
                "- Needle-in-a-haystack "
                f"({NEEDLE_COUNT}-needle; Kamradt / RULER / EvalScope-style) "
                "for long-context recall",
            ]
            if (summary.get("needle_haystack") or {}).get("rows")
            else []
        ),
        "",
        *compared,
        "",
    ]
    if gdx_kv:
        lines.extend(
            [
                "GDX cold coding-rate vs naive vendor KV bits (estimate): "
                "bf16 ~29-31% vs 16 bpv; fp8 ~19-25% vs 8 bpv. "
                "Section A GPU memory is the reserved vendor KV pool, not "
                "coded cold bytes.",
                "",
            ]
        )
    return lines


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


def _kv_bit_exact_row(correctness: dict[str, Any]) -> str | None:
    ok = correctness.get("kv_bit_exact_ok")
    if ok is None:
        return None
    status = _bit_exact_cell(ok)
    status_cell = bold_win(status, win=status == "PASS")
    alphabet = correctness.get("kv_bit_exact_alphabet")
    detail = "GDX cold roundtrip"
    if alphabet:
        detail = f"GDX `{alphabet}` cold roundtrip"
    return f"| KV bit-exactness | vendor KV pages | {status_cell} ({detail}) |"


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
    arm = _arm_col()
    rows = [
        "## Correctness",
        "",
        f"| Metric | Baseline | {arm} |",
        "|--------|----------|-------|",
        (
            "| On-disk model size | "
            f"{bytes_gb(baseline_fp.get('on_disk_bytes_measured'))} | "
            f"{footprint_tic} |"
        ),
        f"| {verify_label} | baseline weights | {status_cell} ({verify_cmd}) |",
    ]
    kv_row = _kv_bit_exact_row(correctness)
    if kv_row is not None:
        rows.append(kv_row)
    rows.extend([_serve_output_match_row(correctness), ""])
    return rows


def _simple_savings_pct(baseline: Any, tic: Any) -> float | None:
    """``100 * (1 - tic/baseline)`` (less is better)."""
    if not isinstance(baseline, (int, float)) or not isinstance(tic, (int, float)):
        return None
    if float(baseline) <= 0:
        return None
    return (1.0 - float(tic) / float(baseline)) * 100.0


def _simple_gain_pct(baseline: Any, tic: Any) -> float | None:
    """``100 * (tic/baseline - 1)`` (more is better; KV headroom)."""
    if not isinstance(baseline, (int, float)) or not isinstance(tic, (int, float)):
        return None
    if float(baseline) <= 0:
        return None
    return (float(tic) / float(baseline) - 1.0) * 100.0


def _section_gpu_excluding_kv(footprint: dict[str, Any]) -> list[str]:
    baseline_fp = footprint["baseline"]
    tic_fp = footprint["tic"]
    b_load = baseline_fp.get("model_loading_bytes_reported")
    t_load = tic_fp.get("model_loading_bytes_reported")
    load_win = (
        isinstance(b_load, (int, float))
        and isinstance(t_load, (int, float))
        and float(t_load) < float(b_load)
    )
    load_pct = _simple_savings_pct(b_load, t_load)
    if (
        load_pct is not None
        and isinstance(b_load, (int, float))
        and isinstance(t_load, (int, float))
        and float(b_load) > 0
    ):
        load_ratio = float(t_load) / float(b_load)
        load_cell = f"{number(load_pct)}% ({number(load_ratio)}x)"
    else:
        load_cell = "null"
    arm = _arm_col()
    return [
        "## A. GPU memory",
        "",
        (
            f"Savings % uses `1 - {arm}/Baseline` for loaded weights "
            f"(x is `{arm}/Baseline`). For KV it uses `{arm}/Baseline - 1` "
            "as a % gain (and the matching `x` ratio)."
        ),
        "",
        "### Weights",
        "",
        f"| Metric | Baseline | {arm} | Savings % |",
        "|--------|----------|-------|-----------|",
        (
            "| GPU memory for loaded weights | "
            f"{bytes_gib(b_load)} | "
            f"{bold_win(bytes_gib(t_load), win=load_win)} | "
            f"{bold_win(load_cell, win=load_win)} |"
        ),
        "",
    ]


def _section_kv_reservation(kv: dict[str, Any]) -> list[str]:
    """Reserved GPU KV vs Baseline default fill. Gain is ``Isiro/Baseline - 1``."""
    b_tokens = kv.get("baseline_tokens_measured")
    t_tokens = kv.get("tic_tokens_measured")
    b_bytes = kv.get("baseline_memory_bytes_measured")
    t_bytes = kv.get("tic_memory_bytes_measured")
    b_gib = pool_gib(b_bytes, b_tokens)
    t_gib = pool_gib(t_bytes, t_tokens, peer_bytes=b_bytes, peer_tokens=b_tokens)
    if b_gib is None or t_gib is None:
        return []
    pct = _simple_gain_pct(b_gib, t_gib)
    win = t_gib > b_gib
    if pct is None:
        pct_cell = "null"
    else:
        ratio = 1.0 + float(pct) / 100.0
        pct_cell = f"{number(pct)}% ({number(ratio)}x)"
    b_tok = number(b_tokens, 0) if b_tokens is not None else "null"
    t_tok = number(t_tokens, 0) if t_tokens is not None else "null"
    tic_cell = f"{bold_win(f'{t_gib:.2f} GiB', win=win)} ({t_tok} tokens)†"
    from needle_haystack import kv_overflow_footnote_lines

    arm = _arm_col()
    return [
        "### KV cache",
        "",
        f"| Metric | Baseline default fill | {arm} reserved KV | Savings % |",
        "|--------|-----------------------|-------------------|-----------|",
        (
            "| GPU memory reserved for KV | "
            f"{b_gib:.2f} GiB ({b_tok} tokens) | "
            f"{tic_cell} | "
            f"{bold_win(pct_cell, win=win)} |"
        ),
        "",
        *kv_overflow_footnote_lines(),
    ]


def _section_c(
    capacity: dict[str, Any],
    kv: dict[str, Any],
    *,
    prefix: str = "1",
    overflow_note: bool = False,
) -> list[str]:
    b_seqs = (capacity.get("baseline") or {}).get("max_num_seqs")
    t_seqs = (capacity.get("tic") or {}).get("max_num_seqs")
    scale_mode = str(capacity.get("seqs_scale_mode") or "")
    seqs_matched = (
        isinstance(b_seqs, (int, float))
        and isinstance(t_seqs, (int, float))
        and int(b_seqs) == int(t_seqs)
    )
    arm = _arm_col()
    if seqs_matched:
        scale_blurb = (
            f"Batch is {number(b_seqs, 0)} on both sides. "
            "Not scaled from KV token capacity."
        )
    elif scale_mode == "kv_measured":
        scale_blurb = (
            f"{arm} batch = round(Baseline batch × {arm}/Baseline KV token "
            "capacity) from a matched vLLM KV measurement."
        )
    elif scale_mode == "kv_estimate":
        scale_blurb = (
            f"{arm} batch = round(Baseline batch × estimated KV ratio), "
            "where estimated KV adds on-disk weight savings to Baseline KV."
        )
    else:
        scale_blurb = f"{arm} batch matches the configured Baseline batch."
    batch_ratio = f"{number(capacity.get('batch_ratio_derived'))}x"
    kv_token_ratio_val = capacity.get("kv_token_capacity_ratio_measured")
    kv_token_win = (
        isinstance(kv_token_ratio_val, (int, float))
        and float(kv_token_ratio_val) > 1.0
    )
    seqs_win = (
        isinstance(t_seqs, (int, float))
        and isinstance(b_seqs, (int, float))
        and int(t_seqs) > int(b_seqs)
    )
    seqs_tic = bold_win(number(t_seqs, 0), win=seqs_win)
    if not seqs_matched:
        seqs_tic = f"{seqs_tic} ({bold_win(batch_ratio, win=seqs_win)})"
    b_tokens = kv.get("baseline_tokens_measured")
    t_tokens = kv.get("tic_tokens_measured")
    b_gib = pool_gib(kv.get("baseline_memory_bytes_measured"), b_tokens)
    t_gib = pool_gib(
        kv.get("tic_memory_bytes_measured"),
        t_tokens,
        peer_bytes=kv.get("baseline_memory_bytes_measured"),
        peer_tokens=b_tokens,
    )
    tokens_base = tokens_with_pool(b_tokens, b_gib)
    tokens_tic = bold_win(
        tokens_with_pool(t_tokens, t_gib, dagger=True),
        win=kv_token_win,
    )
    if str(capacity.get("kv_measured_source") or "") == "timed_asymmetric":
        tokens_tic = f"{tokens_tic}; asymmetric timed seqs"
    letter = str(prefix) if prefix in ("C", "D") else "D"
    lines = [
        f"## {letter}. Capacity",
        "",
        scale_blurb,
        "",
        "KV token capacity is reserved pool slots when Isiro printed KV pool, "
        "else vLLM GPU tokens in this serve's reserved GPU pool.",
        "",
        f"| Metric | Baseline | {arm} |",
        "|--------|----------|-------|",
        f"| Batch | {number(b_seqs, 0)} | {seqs_tic} |",
        f"| KV token capacity | {tokens_base} | {tokens_tic} |",
        "",
    ]
    if overflow_note:
        from needle_haystack import kv_overflow_footnote_lines

        lines.extend(kv_overflow_footnote_lines())
    return lines


def _compact_pct(lat: dict[str, Any], stem: str) -> str:
    return " / ".join(number(lat.get(f"{stem}_{name}")) for name in ("p50", "p95", "p99"))


def _compact_latency_rows(
    b_lat: dict[str, Any], t_lat: dict[str, Any]
) -> list[str]:
    return [
        f"| ITL (ms) p50 / p95 / p99 | {_compact_pct(b_lat, 'itl')} | {_compact_pct(t_lat, 'itl')} |",
        f"| TPOT (ms) p50 / p95 / p99 | {_compact_pct(b_lat, 'tpot')} | {_compact_pct(t_lat, 'tpot')} |",
        f"| TTFT (ms) p50 / p95 / p99 | {_compact_pct(b_lat, 'ttft')} | {_compact_pct(t_lat, 'ttft')} |",
    ]


def _section_generation(
    capacity: dict[str, Any] | None,
    decode_rows: list[dict[str, Any]],
    *,
    letter: str = "D",
    mention_equal_batch: bool = True,
    as_subsection: bool = False,
) -> list[str]:
    arm = _arm_col()
    equal_note = (
        " Not equal batch (C)."
        if mention_equal_batch
        else ""
    )
    heading = (
        "### Generation (recall-matched KV, input 32 / output 256)"
        if as_subsection
        else f"## {letter}. Generation (recall-matched KV, input 32 / output 256)"
    )
    lines = [
        heading,
        "",
        "Tok/s and latency at the recall-matched KV sizes. Batch is in Capacity."
        f"{equal_note} TTFT is in this table because it moves with concurrency.",
        "",
        f"| Metric | Baseline | {arm} |",
        "|--------|----------|-------|",
    ]
    tps = (capacity or {}).get("decode_output_tokens_per_sec") or {}
    if capacity is not None and tps:
        b_tps = tps.get("baseline")
        t_tps = tps.get("tic")
        tps_win = (
            isinstance(b_tps, (int, float))
            and isinstance(t_tps, (int, float))
            and float(t_tps) > float(b_tps)
        )
        lines.append(
            "| Output tok/s | "
            f"{number(b_tps)} | "
            f"{bold_win(number(t_tps), win=tps_win)} |"
        )
    elif decode_rows:
        row = decode_rows[0]
        b_tps = row["baseline"]["throughput"]["output_tokens_per_sec"]
        t_tps = row["tic"]["throughput"]["output_tokens_per_sec"]
        lines.append(
            f"| Output tok/s | {number(b_tps)} | {number(t_tps)} |"
        )
    if decode_rows:
        row = decode_rows[0]
        lines.extend(
            _compact_latency_rows(
                row["baseline"]["latency_ms"], row["tic"]["latency_ms"]
            )
        )
    lines.append("")
    return lines


def _section_equal_batch(equal_batch: dict[str, Any]) -> list[str]:
    arm = _arm_col()
    tps = equal_batch.get("decode_output_tokens_per_sec") or {}
    latency = equal_batch.get("latency_ms") or {}
    b_lat = latency.get("baseline") or {}
    t_lat = latency.get("tic") or {}
    lines = [
        "## C. Equal batch",
        "",
        (
            "Same batch as Baseline. ITL stays near Baseline; "
            "capacity-matched generation in D raises concurrency and ITL."
        ),
        "",
        f"| Metric | Baseline | {arm} |",
        "|--------|----------|-------|",
        (
            "| Output tok/s | "
            f"{number(tps.get('baseline'))} | "
            f"{number(tps.get('tic'))} |"
        ),
    ]
    if b_lat and t_lat:
        lines.extend(_compact_latency_rows(b_lat, t_lat))
    lines.append("")
    return lines


def _equal_batch_redundant(
    capacity: dict[str, Any] | None,
    equal_batch: dict[str, Any],
    decode_rows: list[dict[str, Any]],
) -> bool:
    """True when Equal batch repeats Generation numbers (matched batch)."""
    eq_tps = equal_batch.get("decode_output_tokens_per_sec") or {}
    cap_tps = (capacity or {}).get("decode_output_tokens_per_sec") or {}
    if not eq_tps:
        return False
    if cap_tps:
        for side in ("baseline", "tic"):
            a = cap_tps.get(side)
            b = eq_tps.get(side)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                return False
            if abs(float(a) - float(b)) > 1e-6:
                return False
        return True
    if not decode_rows:
        return False
    row = decode_rows[0]
    for side in ("baseline", "tic"):
        a = row[side]["throughput"]["output_tokens_per_sec"]
        b = eq_tps.get(side)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if abs(float(a) - float(b)) > 1e-6:
            return False
    return True


def _section_gdx_kv_bytes(summary: dict[str, Any]) -> list[str]:
    """Cold GDX vs Baseline FA twin curve (entropy win figure)."""
    gdx = summary.get("gdx_kv_bytes")
    if not isinstance(gdx, dict) or not gdx.get("image"):
        return []
    alphabet = gdx.get("alphabet") or "bf16"
    w = gdx.get("live_window")
    max_t = gdx.get("max_t")
    b_gib = gdx.get("baseline_kv_gib_at_max_t")
    t_gib = gdx.get("isiro_kv_gib_at_max_t")
    save_gib = gdx.get("save_gib_at_max_t")
    save = gdx.get("save_at_max_t")
    if (
        isinstance(max_t, (int, float))
        and isinstance(b_gib, (int, float))
        and isinstance(t_gib, (int, float))
        and isinstance(save_gib, (int, float))
        and isinstance(save, (int, float))
    ):
        save_line = (
            f"At T={number(max_t, 0)}: Baseline KV {number(b_gib)} GiB, "
            f"Isiro GDX KV {number(t_gib)} GiB; save {number(save_gib)} GiB "
            f"({float(save) * 100:.1f}%)."
        )
    else:
        save_s = (
            f"{float(save) * 100:.1f}%"
            if isinstance(save, (int, float))
            else "n/a"
        )
        save_line = f"Save at max T: {save_s}."
    lines = [
        "### Cold KV bytes vs context",
        "",
        (
            f"Hot window W={w} stays vendor FA. Cold prefix `[0, T-W)` is GDX o1 "
            f"(`ISIRO_GDX={alphabet}`). Curve is packed cold + hot FA vs Baseline "
            f"FA for the same T. {save_line}"
        ),
        "",
        "Figure. KV bytes vs context.",
        "",
        f"![KV bytes vs context]({gdx['image']})",
        "",
    ]
    return lines


def _mode_sections(summary: dict[str, Any]) -> list[str]:
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
    lines.extend(_section_gpu_excluding_kv(footprint))
    lines.extend(_section_kv_reservation(kv))
    lines.extend(_section_gdx_kv_bytes(summary))
    reserved_gib = pool_gib(
        kv.get("tic_memory_bytes_measured"),
        kv.get("tic_tokens_measured"),
        peer_bytes=kv.get("baseline_memory_bytes_measured"),
        peer_tokens=kv.get("baseline_tokens_measured"),
    )
    haystack = (
        summary.get("needle_haystack")
        if isinstance(summary.get("needle_haystack"), dict)
        else None
    )
    haystack_rows = haystack.get("table_rows") if haystack else None
    if haystack_rows:
        from needle_haystack import kv_sweep_table_lines

        lines.extend(
            kv_sweep_table_lines(haystack_rows, reserved_gib=reserved_gib)
        )
    if haystack and haystack.get("image"):
        from needle_haystack import report_lines

        kwargs: dict[str, str] = {}
        if haystack_rows:
            kwargs["heading"] = ""
        lines.extend(report_lines(str(haystack["image"]), **kwargs))
    if haystack and haystack.get("vram_image"):
        from needle_haystack import vram_report_lines

        lines.extend(vram_report_lines(str(haystack["vram_image"])))
    skip_equal = equal_batch is not None and _equal_batch_redundant(
        capacity, equal_batch, decode_rows
    )
    show_equal = equal_batch is not None and not skip_equal
    if show_equal:
        lines.extend(_section_equal_batch(equal_batch))
    cap_letter = "D" if show_equal else "C"
    if capacity is not None and experiment_kind == "capacity":
        lines.extend(
            _section_c(
                capacity,
                kv,
                prefix=cap_letter,
                overflow_note=False,
            )
        )
        lines.extend(
            _section_generation(
                capacity,
                decode_rows,
                mention_equal_batch=show_equal,
                as_subsection=True,
            )
        )
    else:
        lines.extend(
            _section_generation(
                None,
                decode_rows,
                letter=cap_letter,
                mention_equal_batch=show_equal,
                as_subsection=False,
            )
        )
    return lines


def _section_g(
    summary: dict[str, Any],
    environment: dict[str, Any],
    commands: dict[str, Any],
    graph_mode: str,
) -> list[str]:
    reused = commands.get("baseline_reused_from")
    graph_cell = (
        "on (eager prefill, graph decode)"
        if graph_mode == "graphs"
        else "off (eager)"
    )
    compiler, runtime = _compiler_runtime_versions(summary)
    kv_dtype = commands.get("kv_cache_dtype")
    gdx_kv = commands.get("isiro_gdx_kv")
    kv_cell = "product KV cache size"
    if kv_dtype or gdx_kv:
        parts = []
        if kv_dtype:
            parts.append(f"dtype `{kv_dtype}`")
        if gdx_kv:
            parts.append(f"`ISIRO_GDX={gdx_kv}` on Isiro")
        kv_cell = "; ".join(parts)
    lines = [
        "## Config",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| System | `{summary.get('system_id', '')}` |",
        f"| CUDA graphs | {graph_cell} |",
        f"| KV cache | {kv_cell} |",
        f"| Compiler | `{compiler}` |",
        f"| Runtime | `{runtime}` |",
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
    if gdx_kv:
        lines.append(f"| ISIRO_GDX | `{gdx_kv}` |")
    attn = (
        summary.get("attention")
        if isinstance(summary.get("attention"), dict)
        else {}
    )
    tic_attn = attn.get("tic") if isinstance(attn.get("tic"), dict) else {}
    base_attn = (
        attn.get("baseline") if isinstance(attn.get("baseline"), dict) else {}
    )
    if tic_attn.get("attention_backend") or tic_attn.get("isiro_attn"):
        lines.append(
            "| Isiro attention | `"
            f"{tic_attn.get('attention_backend')}` "
            f"(`ISIRO_ATTN` {tic_attn.get('isiro_attn')}) |"
        )
    if base_attn.get("attention_backend"):
        lines.append(
            f"| Baseline attention | `{base_attn.get('attention_backend')}` |"
        )
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


def render(run_dir: Path) -> str:
    summary = load_json(run_dir / "summary.json")
    environment = load_json(run_dir / "environment.json")
    commands = load_json(run_dir / "commands.json")
    footprint = summary["footprint"]
    correctness = summary["correctness"]
    graph_mode = _graph_mode(commands)
    lines = _section_intro(summary, commands)
    lines.extend(_section_a(footprint, correctness))
    lines.extend(_mode_sections(summary))
    story_path = run_dir / "stock_vllm_story.json"
    if story_path.is_file():
        from stock_vllm_story import render_story_markdown

        lines.extend(render_story_markdown(load_json(story_path)))
    lines.extend(_section_g(summary, environment, commands, graph_mode))
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


def _stage_sanitized_tree(run_dir: Path, temp: Path) -> None:
    for name in (
        "environment.json",
        "commands.json",
        "summary.json",
        "verify.json",
        "stock_vllm_story.json",
    ):
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


def write_report(run_dir: Path, output: Path) -> Path:
    """Render a sanitized report.md to ``output`` (scratch JSON stays raw)."""
    temp = Path(tempfile.mkdtemp(prefix=".isiro-bench-report-"))
    try:
        _stage_sanitized_tree(run_dir, temp)
        gdx_curve = run_dir / "gdx_kv_bytes_curve.json"
        gdx_png_src = run_dir / "kv-bytes-vs-context.png"
        if gdx_curve.is_file() and gdx_png_src.is_file():
            import shutil as _shutil

            gdx_doc = load_json(gdx_curve)
            dest_png = output.parent / "kv-bytes-vs-context.png"
            _shutil.copy2(gdx_png_src, dest_png)
            pts = list(gdx_doc.get("points") or [])
            save_max = None
            max_t = None
            b_gib = None
            t_gib = None
            save_gib = None
            if pts:
                last = pts[-1]
                save_max = last.get("save_vs_fa")
                max_t = last.get("T")
                fa_b = last.get("fa_twin_nbytes")
                gdx_b = last.get("gdx_total_nbytes")
                if isinstance(fa_b, (int, float)):
                    b_gib = float(fa_b) / float(1024**3)
                if isinstance(gdx_b, (int, float)):
                    t_gib = float(gdx_b) / float(1024**3)
                if b_gib is not None and t_gib is not None:
                    save_gib = float(b_gib) - float(t_gib)
            summary_path = temp / "summary.json"
            if summary_path.is_file():
                summary = load_json(summary_path)
                summary["gdx_kv_bytes"] = {
                    "image": dest_png.name,
                    "alphabet": gdx_doc.get("alphabet"),
                    "live_window": gdx_doc.get("live_window"),
                    "max_t": max_t,
                    "baseline_kv_gib_at_max_t": b_gib,
                    "isiro_kv_gib_at_max_t": t_gib,
                    "save_gib_at_max_t": save_gib,
                    "save_at_max_t": save_max,
                    "ok": bool(gdx_doc.get("ok")),
                }
                write_json(summary_path, summary)
        haystack_jsonl = run_dir / "needle_haystack.jsonl"
        if haystack_jsonl.is_file():
            from needle_haystack import load_jsonl, render_heatmap, render_vram_vs_context

            rows = load_jsonl(haystack_jsonl)
            image = output.parent / "needle-haystack.png"
            vram = output.parent / "vram-vs-context.png"
            sizes_path = run_dir / "niah_sizes.json"
            max_len = None
            if sizes_path.is_file():
                sizes = load_json(sizes_path)
                raw_len = sizes.get("max_model_len")
                if isinstance(raw_len, (int, float)) and int(raw_len) > 0:
                    max_len = int(raw_len)
            gpu = ""
            gpu_count = None
            model = ""
            summary_path = temp / "summary.json"
            env_path = temp / "environment.json"
            if summary_path.is_file():
                summary = load_json(summary_path)
                model = str(summary.get("model") or "")
            if env_path.is_file():
                env = load_json(env_path)
                gpu = str(env.get("gpu_model") or "")
                raw_n = env.get("gpu_count")
                try:
                    if raw_n is not None and int(raw_n) > 0:
                        gpu_count = int(raw_n)
                except (TypeError, ValueError):
                    gpu_count = None
            render_heatmap(
                rows,
                image,
                max_model_len=max_len,
                gpu=gpu,
                gpu_count=gpu_count,
                model=model,
            )
            render_vram_vs_context(
                rows,
                vram,
                max_model_len=max_len,
                gpu=gpu,
                gpu_count=gpu_count,
                model=model,
            )
            if summary_path.is_file():
                summary = load_json(summary_path)
                summary["needle_haystack"] = {
                    "rows": True,
                    "table_rows": rows,
                    "image": image.name,
                    "vram_image": vram.name,
                }
                write_json(summary_path, summary)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render(temp), encoding="utf-8")
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.run_dir / "report.md"
    write_report(args.run_dir, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
