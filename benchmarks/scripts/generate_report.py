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
    """Format a numeric report cell.

    Integer cells (``digits=0``) use thousands separators (``54,418``).
    """
    if value is None:
        return "null"
    if int(digits) == 0:
        return f"{int(round(float(value))):,}"
    return f"{float(value):,.{int(digits)}f}"


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


def pct_gain_from_ratio(ratio: Any) -> str:
    """``Isiro/Baseline - 1`` as a signed percent gain (e.g. 1.12 → ``+12%``)."""
    if not isinstance(ratio, (int, float)):
        return ""
    gain = round((float(ratio) - 1.0) * 100.0)
    if gain > 0:
        return f"+{gain}%"
    return f"{gain}%"


def _arm_col() -> str:
    """Customer product arm column label (TIC weights + optional Isiro KV)."""
    return "Isiro"


def tokens_with_pool(
    tokens: Any,
    gib: float | None,
    *,
    extra: str = "",
    dagger: bool = False,
    gib_only: bool = False,
) -> str:
    """Format reserved pool for the report.

    Always show vendor pool token slots plus GiB. Under Isiro KV those tokens are
    admission slots at vendor bpv (not Isiro-KV-coded cold capacity); the †
    footnote says so. ``gib_only`` is retained for callers but no longer
    strips tokens (customer Capacity matches the bf16 report shape).
    """
    del gib_only  # tokens always shown; footnote covers Isiro KV semantics
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
    isiro_kv = cmds.get("isiro_kv")
    isiro_kv_only = bool(cmds.get("isiro_kv_only"))
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
    if isiro_kv_only and isiro_kv:
        compared[-1] = (
            f"- **Isiro KV-only:** `vllm serve` + `ISIRO_KV={isiro_kv}` "
            "(same AWQ/HF weights as baseline; no TIC weight claim)"
        )
    elif isiro_kv:
        compared[-1] = (
            f"- **Isiro:** `isiro serve … --target vllm` with TIC weights and "
            f"`ISIRO_KV={isiro_kv}` (same vLLM version)"
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
    if isiro_kv_only and isiro_kv:
        lines.extend(
            [
                "Section A: loaded weights and Available KV GiB are shared "
                "(same raw weights, same vendor util). KV tokens capacity and "
                "Capacity concurrency follow the measured Isiro/Baseline token "
                "ratio on that budget. KV bytes vs context are under KV vs "
                "context when the curve ran.",
                "",
            ]
        )
        time_line = _kv_only_time_claim(summary)
        if time_line:
            lines.extend([time_line, ""])
    elif isiro_kv:
        lines.extend(
            [
                "Section A Available KV is the vendor reserved pool (GiB). "
                "KV tokens capacity uses the measured Isiro/Baseline token "
                "ratio on that budget. KV bytes vs context are under KV vs "
                "context when the curve ran.",
                "",
            ]
        )
    return lines


def _kv_only_time_claim(summary: dict[str, Any]) -> str | None:
    """Equal-batch TIME stamp for KV-only path (a): tok/s, ITL, TTFT vs Hub."""
    eb = summary.get("equal_batch")
    if not isinstance(eb, dict):
        return None
    tps = eb.get("decode_output_tokens_per_sec") or {}
    lat = eb.get("latency_ms") or {}
    b_tps = tps.get("baseline")
    t_tps = tps.get("tic")
    b_lat = lat.get("baseline") if isinstance(lat.get("baseline"), dict) else {}
    t_lat = lat.get("tic") if isinstance(lat.get("tic"), dict) else {}
    b_itl = b_lat.get("itl_p50")
    t_itl = t_lat.get("itl_p50")
    b_ttft = b_lat.get("ttft_p50")
    t_ttft = t_lat.get("ttft_p50")
    vals = (b_tps, t_tps, b_itl, t_itl, b_ttft, t_ttft)
    if not all(isinstance(v, (int, float)) for v in vals):
        return None
    conc = eb.get("max_num_seqs") or eb.get("max_concurrency")
    conc_txt = number(conc, 0) if isinstance(conc, (int, float)) else "?"
    tok_ok = float(t_tps) >= float(b_tps) * 0.99
    itl_ok = float(t_itl) <= float(b_itl) * 1.01
    ttft_ok = float(t_ttft) <= float(b_ttft) * 1.01
    go = tok_ok and itl_ok and ttft_ok
    verdict = "GO" if go else "NO GO"
    return (
        f"TIME claim: **{verdict}** on equal-batch "
        f"(conc={conc_txt}; tok/s {number(t_tps)} vs {number(b_tps)}; "
        f"ITL {number(t_itl)} vs {number(b_itl)}; "
        f"TTFT {number(t_ttft)} vs {number(b_ttft)})."
    )


def _serve_output_match_detail(correctness: dict[str, Any]) -> str:
    ok = correctness.get("serve_output_match_ok")
    if ok is None:
        return "not run"
    status = _bit_exact_cell(ok)
    status_cell = bold_win(status, win=status == "PASS")
    matched = correctness.get("serve_output_match_matched")
    total = correctness.get("serve_output_match_prompt_count")
    if isinstance(matched, int) and isinstance(total, int) and total > 0:
        if ok is True:
            detail = f"{matched}/{total} prompts, temp=0, token IDs equal"
        else:
            detail = f"{matched}/{total} prompts matched (temp=0 token IDs)"
        status_cell = f"{status_cell} ({detail})"
    return status_cell


def _serve_output_match_row(correctness: dict[str, Any]) -> str:
    return f"| Serve output match | - | {_serve_output_match_detail(correctness)} |"


def _kv_bit_exact_detail(correctness: dict[str, Any]) -> str | None:
    ok = correctness.get("kv_bit_exact_ok")
    if ok is None:
        return None
    status = _bit_exact_cell(ok)
    status_cell = bold_win(status, win=status == "PASS")
    alphabet = correctness.get("kv_bit_exact_alphabet")
    detail = "Isiro KV roundtrip"
    if alphabet:
        detail = f"Isiro KV `{alphabet}` roundtrip"
    return f"{status_cell} ({detail})"


def _kv_bit_exact_row(correctness: dict[str, Any]) -> str | None:
    detail = _kv_bit_exact_detail(correctness)
    if detail is None:
        return None
    return f"| KV bit-exactness | vendor KV pages | {detail} |"


def _section_a(
    footprint: dict[str, Any],
    correctness: dict[str, Any],
    *,
    kv_only: bool = False,
) -> list[str]:
    if kv_only:
        # No TIC weight claim: omit Integrity SKIP and identical on-disk size.
        rows = [
            "## Correctness",
            "",
            "| Metric | Result |",
            "|--------|--------|",
        ]
        kv_detail = _kv_bit_exact_detail(correctness)
        if kv_detail is not None:
            rows.append(f"| KV bit-exactness | {kv_detail} |")
        rows.extend(
            [
                f"| Serve output match | {_serve_output_match_detail(correctness)} |",
                "",
            ]
        )
        return rows

    baseline_fp = footprint["baseline"]
    tic_fp = footprint["tic"]
    verify_mode = str(correctness.get("verify_mode") or "integrity")
    if verify_mode == "reference":
        verify_label = "Weight bit-exactness"
        verify_cmd = "`isiro verify -r`"
        verify_status = _bit_exact_cell(correctness.get("bit_exact_ok"))
    elif verify_mode in ("skipped", "skipped_kv_only"):
        verify_label = "Integrity"
        verify_cmd = "KV-only; same AWQ/HF weights"
        verify_status = "SKIP"
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


def _section_gpu_excluding_kv(
    footprint: dict[str, Any],
    *,
    kv_only: bool = False,
) -> list[str]:
    baseline_fp = footprint["baseline"]
    tic_fp = footprint["tic"]
    b_load = baseline_fp.get("model_loading_bytes_reported")
    t_load = tic_fp.get("model_loading_bytes_reported")
    if kv_only:
        # Same raw weights both arms; prefer baseline meter, else Isiro.
        shared = b_load if isinstance(b_load, (int, float)) else t_load
        return [
            "## A. GPU memory",
            "",
            "Loaded weights are identical on both arms (no TIC weight claim).",
            "",
            "### Weights",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| GPU memory for loaded weights | {bytes_gib(shared)} |",
            "",
        ]
    load_win = (
        isinstance(b_load, (int, float))
        and isinstance(t_load, (int, float))
        and float(t_load) < float(b_load)
    )
    load_pct = _simple_savings_pct(b_load, t_load)
    if load_pct is not None:
        load_cell = f"{number(load_pct)}%"
    else:
        load_cell = "null"
    arm = _arm_col()
    return [
        "## A. GPU memory",
        "",
        f"Savings % uses `1 - {arm}/Baseline` for loaded weights.",
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


def _fold_coded_kv_capacity(summary: dict[str, Any]) -> dict[str, Any]:
    """Attach hot/cold Isiro KV coded token capacity onto ``kv_cache`` in-place."""
    from capacity_plan import (
        coded_kv_token_capacity,
        isiro_kv_capacity_scale,
        isiro_kv_cold_savings_from_bpv,
    )

    kv = summary.get("kv_cache")
    if not isinstance(kv, dict):
        return summary
    b_adm = kv.get("baseline_tokens_measured")
    t_adm = kv.get("tic_tokens_measured")
    if not isinstance(b_adm, (int, float)) or not isinstance(t_adm, (int, float)):
        return summary

    s: float | None = None
    w = 512
    # Capacity savings SSOT: Gutenberg / real KV-vs-context curve only.
    # Bitexact avg_bpv is Shannon on a synthetic page (avg_bpv_meter); never
    # fold it into customer KV tokens capacity.
    isiro_kv_doc = summary.get("isiro_kv_bytes")
    if isinstance(isiro_kv_doc, dict):
        raw_w = isiro_kv_doc.get("live_window")
        if isinstance(raw_w, (int, float)) and int(raw_w) > 0:
            w = int(raw_w)
        raw_s = isiro_kv_doc.get("save_at_max_t")
        if isinstance(raw_s, (int, float)) and 0.0 < float(raw_s) < 1.0:
            s = float(raw_s)
        if s is None:
            # Payload bpv from the same real curve (not bitexact Shannon).
            payload = isiro_kv_doc.get("payload_bpv_at_max_t")
            fa_bpv = isiro_kv_doc.get("fa_bpv")
            if (
                isinstance(payload, (int, float))
                and isinstance(fa_bpv, (int, float))
                and float(fa_bpv) > 0.0
            ):
                try:
                    s = isiro_kv_cold_savings_from_bpv(
                        float(payload), float(fa_bpv)
                    )
                except ValueError:
                    s = None
    if s is None or s <= 0.0:
        # No Isiro KV meter: coded == vendor admission on both arms.
        kv["baseline_tokens_coded"] = int(round(float(b_adm)))
        kv["tic_tokens_coded"] = int(round(float(t_adm)))
        return summary

    scale = isiro_kv_capacity_scale(s)
    b_coded = coded_kv_token_capacity(
        b_adm, savings_s=s, live_window=w, apply_isiro_kv=False
    )
    t_coded = coded_kv_token_capacity(
        t_adm, savings_s=s, live_window=w, apply_isiro_kv=True
    )
    kv["baseline_tokens_coded"] = int(b_coded)
    kv["tic_tokens_coded"] = int(t_coded)
    kv["isiro_kv_capacity_s"] = float(s)
    kv["isiro_kv_capacity_scale"] = float(scale)
    kv["isiro_kv_live_window"] = int(w)
    return summary


def _section_kv_reservation(
    kv: dict[str, Any],
    *,
    gib_only: bool = False,
    kv_only: bool = False,
) -> list[str]:
    """Available KV GiB, then Isiro-KV-coded KV tokens capacity when metered."""
    b_tokens = kv.get("baseline_tokens_measured")
    t_tokens = kv.get("tic_tokens_measured")
    b_bytes = kv.get("baseline_memory_bytes_measured")
    t_bytes = kv.get("tic_memory_bytes_measured")
    b_gib = pool_gib(b_bytes, b_tokens)
    t_gib = pool_gib(t_bytes, t_tokens, peer_bytes=b_bytes, peer_tokens=b_tokens)
    if b_gib is None or t_gib is None:
        return []
    arm = _arm_col()
    if kv_only:
        shared_gib = b_gib
        lines = [
            "### Available KV",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Available KV | {shared_gib:.2f} GiB |",
            "",
            "Available KV is the vendor reserved pool on the GPU "
            "(same util target both arms; no TIC weight headroom).",
            "",
        ]
    else:
        gib_win = float(t_gib) > float(b_gib)
        lines = [
            "### Available KV",
            "",
            f"| Metric | Baseline | {arm} |",
            "|--------|----------|-------|",
            (
                f"| Available KV | {b_gib:.2f} GiB | "
                f"{bold_win(f'{t_gib:.2f} GiB', win=gib_win)} |"
            ),
            "",
        ]
        # Pilot surface: GiB-only Available KV. Optional short honesty note.
        if gib_only or kv.get("isiro_kv_capacity_s") is not None:
            lines.extend(
                [
                    "Available KV is the vendor reserved pool on the GPU "
                    "(same util target as Baseline).",
                    "",
                ]
            )

    b_coded = kv.get("baseline_tokens_coded")
    t_coded = kv.get("tic_tokens_coded")
    s = kv.get("isiro_kv_capacity_s")
    scale = kv.get("isiro_kv_capacity_scale")
    if (
        isinstance(b_coded, (int, float))
        and isinstance(t_coded, (int, float))
        and isinstance(s, (int, float))
        and float(s) > 0.0
    ):
        coded_win = float(t_coded) > float(b_coded)
        ratio = float(t_coded) / float(b_coded) if float(b_coded) > 0 else None
        tic_cell = bold_win(number(t_coded, 0), win=coded_win)
        if isinstance(ratio, float) and ratio != 1.0:
            gain = pct_gain_from_ratio(ratio)
            if gain:
                tic_cell = f"{tic_cell} ({bold_win(gain, win=coded_win)})"
        scale_txt = (
            number(scale)
            if isinstance(scale, (int, float))
            else number(1.0 / (1.0 - float(s)))
        )
        if kv_only:
            capacity_blurb = (
                "Same Available KV GiB both arms. Isiro KV fits more tokens "
                f"in that pool (measured scale ≈ {scale_txt})."
            )
        else:
            capacity_blurb = (
                "Weight compression grows Available KV. Isiro KV then fits more "
                f"tokens in that pool (measured scale ≈ {scale_txt})."
            )
        lines.extend(
            [
                "### KV tokens capacity",
                "",
                capacity_blurb,
                "",
                f"| Metric | Baseline | {arm} |",
                "|--------|----------|-------|",
                (
                    f"| KV tokens capacity | {number(b_coded, 0)} | "
                    f"{tic_cell} |"
                ),
                "",
            ]
        )
    elif not gib_only:
        # Legacy without Isiro KV: keep vendor admission tokens next to GiB for honesty.
        b_tok = number(b_tokens, 0) if b_tokens is not None else "null"
        t_tok = number(t_tokens, 0) if t_tokens is not None else "null"
        tok_win = (
            isinstance(t_tokens, (int, float))
            and isinstance(b_tokens, (int, float))
            and float(t_tokens) > float(b_tokens)
        )
        lines.extend(
            [
                "### KV tokens capacity",
                "",
                f"| Metric | Baseline | {arm} |",
                "|--------|----------|-------|",
                (
                    f"| KV tokens capacity | {b_tok} | "
                    f"{bold_win(t_tok, win=tok_win)} |"
                ),
                "",
            ]
        )
    _ = gib_only
    return lines


def _section_c(
    capacity: dict[str, Any],
    kv: dict[str, Any],
    *,
    prefix: str = "1",
    overflow_note: bool = False,
    gib_only: bool = False,
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
            f"Concurrent requests are {number(b_seqs, 0)} on both sides. "
            "Not scaled from KV token capacity."
        )
    elif scale_mode == "kv_coded":
        scale_blurb = (
            f"{arm} concurrent requests = round(Baseline × "
            f"{arm}/Baseline KV token capacity), then rounded up onto the "
            "vLLM CUDA-graph capture stride (multiples of 8) so FULL decode "
            "graphs cover concurrency."
        )
    elif scale_mode == "kv_measured":
        scale_blurb = (
            f"{arm} concurrent requests = round(Baseline × {arm}/Baseline KV "
            "token capacity) from a matched vLLM KV measurement, then rounded "
            "up onto the vLLM CUDA-graph capture stride (multiples of 8) so "
            "FULL decode graphs cover concurrency."
        )
    elif scale_mode == "kv_estimate":
        scale_blurb = (
            f"{arm} concurrent requests = round(Baseline × estimated KV "
            "ratio), where estimated KV adds on-disk weight savings to "
            "Baseline KV, then rounded up onto the vLLM CUDA-graph capture "
            "stride when needed."
        )
    else:
        scale_blurb = (
            f"{arm} concurrent requests match the configured Baseline "
            "concurrent-request ceiling."
        )
    batch_ratio_val = capacity.get("batch_ratio_derived")
    if (
        not isinstance(batch_ratio_val, (int, float))
        and isinstance(t_seqs, (int, float))
        and isinstance(b_seqs, (int, float))
        and float(b_seqs) > 0
    ):
        batch_ratio_val = float(t_seqs) / float(b_seqs)
    batch_gain = pct_gain_from_ratio(batch_ratio_val)
    seqs_win = (
        isinstance(t_seqs, (int, float))
        and isinstance(b_seqs, (int, float))
        and int(t_seqs) > int(b_seqs)
    )
    seqs_tic = bold_win(number(t_seqs, 0), win=seqs_win)
    if not seqs_matched and batch_gain:
        seqs_tic = f"{seqs_tic} ({bold_win(batch_gain, win=seqs_win)})"
    letter = str(prefix) if prefix in ("C", "D") else "D"
    # KV tokens capacity lives under Section A (coded when Isiro KV is on).
    _ = (gib_only, kv, overflow_note)
    lines = [
        f"## {letter}. Capacity",
        "",
        scale_blurb,
        "",
        "Concurrent requests are the serve max concurrent sequences "
        f"(`max_num_seqs`); the bench load matches that ceiling in this A/B.",
        "",
        f"| Metric | Baseline | {arm} |",
        "|--------|----------|-------|",
        f"| Concurrent requests | {number(b_seqs, 0)} | {seqs_tic} |",
        "",
    ]
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
        "Tok/s is aggregate system throughput across in-flight requests. "
        "Per-request decode rate is reflected by ITL/TPOT. Concurrent "
        "requests are in Capacity."
        f"{equal_note} Latency-matched wins live in Equal batch (same conc as "
        "Baseline). Higher Capacity concurrency raises aggregate tok/s and "
        "can move TTFT/ITL.",
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
        tps_cell = bold_win(number(t_tps), win=tps_win)
        if (
            isinstance(b_tps, (int, float))
            and isinstance(t_tps, (int, float))
            and float(b_tps) > 0.0
            and float(t_tps) != float(b_tps)
        ):
            gain = pct_gain_from_ratio(float(t_tps) / float(b_tps))
            if gain:
                tps_cell = f"{tps_cell} ({bold_win(gain, win=tps_win)})"
        lines.append(
            "| Output tok/s | "
            f"{number(b_tps)} | "
            f"{tps_cell} |"
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
    conc = equal_batch.get("max_num_seqs")
    if conc is None:
        conc = equal_batch.get("max_concurrency")
    lines = [
        "## C. Equal batch",
        "",
        (
            "Same concurrent-request ceiling as Baseline (latency-matched "
            "surface). Capacity-matched generation in D raises concurrency for "
            "aggregate tok/s; ITL/TTFT there are not the equal-batch claim."
        ),
        "",
        f"| Metric | Baseline | {arm} |",
        "|--------|----------|-------|",
    ]
    if conc is not None:
        lines.append(
            f"| Concurrent requests | {number(conc, 0)} | {number(conc, 0)} |"
        )
    lines.append(
        (
            "| Output tok/s | "
            f"{number(tps.get('baseline'))} | "
            f"{number(tps.get('tic'))} |"
        )
    )
    if b_lat and t_lat:
        lines.extend(_compact_latency_rows(b_lat, t_lat))
    lines.append("")
    return lines


def _equal_batch_redundant(
    capacity: dict[str, Any] | None,
    equal_batch: dict[str, Any],
    decode_rows: list[dict[str, Any]],
) -> bool:
    """Whether to omit Equal batch when it matches Capacity generation.

    Always keep Equal batch (C) in the customer report, even when Capacity
    concurrency snaps onto Baseline (matched seqs, same tok/s). Omitting it
    made AWQ matched-8 reports look like Capacity tok/s disappeared.
    """
    del capacity, equal_batch, decode_rows
    return False


def _isiro_kv_curve_omit_reason(summary: dict[str, Any], alph: str) -> str:
    """Customer omit copy when the KV-vs-context curve did not run."""
    del summary
    return (
        f"`ISIRO_KV={alph}` is on for serve. "
        "kv-bytes-vs-context.png omitted: curve did not run."
    )


def _section_isiro_kv_bytes(summary: dict[str, Any]) -> list[str]:
    """KV vs context curve (weights+KV panel when weight bytes differ)."""
    isiro_kv_doc = summary.get("isiro_kv_bytes")
    if not isinstance(isiro_kv_doc, dict) or not isiro_kv_doc.get("image"):
        # Isiro KV serve without curve (e.g. 27B bf16 twin OOMs): explicit omit.
        corr = summary.get("correctness") if isinstance(
            summary.get("correctness"), dict
        ) else {}
        alph = corr.get("kv_bit_exact_alphabet")
        if alph and corr.get("kv_bit_exact_ok"):
            return [
                "### KV vs context",
                "",
                _isiro_kv_curve_omit_reason(summary, str(alph)),
                "",
            ]
        return []
    alphabet = isiro_kv_doc.get("alphabet") or "bf16"
    max_t = isiro_kv_doc.get("max_t")
    b_gib = isiro_kv_doc.get("baseline_kv_gib_at_max_t")
    t_gib = isiro_kv_doc.get("isiro_kv_gib_at_max_t")
    save_gib = isiro_kv_doc.get("save_gib_at_max_t")
    save = isiro_kv_doc.get("save_at_max_t")
    b_w = isiro_kv_doc.get("baseline_weight_gib")
    t_w = isiro_kv_doc.get("isiro_weight_gib")
    if (
        isinstance(max_t, (int, float))
        and isinstance(b_gib, (int, float))
        and isinstance(t_gib, (int, float))
        and isinstance(save_gib, (int, float))
        and isinstance(save, (int, float))
    ):
        save_line = (
            f"At T={number(max_t, 0)}: Baseline KV {number(b_gib)} GiB, "
            f"Isiro KV {number(t_gib)} GiB; save {number(save_gib)} GiB "
            f"({float(save) * 100:.1f}%)."
        )
    else:
        save_s = (
            f"{float(save) * 100:.1f}%"
            if isinstance(save, (int, float))
            else "n/a"
        )
        save_line = f"Save at max T: {save_s}."
    weight_line = ""
    if (
        isinstance(b_w, (int, float))
        and isinstance(t_w, (int, float))
        and isinstance(b_gib, (int, float))
        and isinstance(t_gib, (int, float))
        and isinstance(max_t, (int, float))
    ):
        b_tot = float(b_w) + float(b_gib)
        t_tot = float(t_w) + float(t_gib)
        weight_line = (
            f" Bottom panel adds loaded weights (Baseline {number(b_w)} GiB, "
            f"Isiro {number(t_w)} GiB): at T={number(max_t, 0)}, "
            f"Baseline {number(b_tot)} GiB vs Isiro {number(t_tot)} GiB "
            f"(weights + KV)."
        )
    capacity_line = ""
    fig_caption = "Figure. KV vs 1-user context."
    if isiro_kv_doc.get("capacity_bars"):
        fig_caption = (
            "Figure. KV vs context (top); KV tokens capacity and "
            "Capacity concurrency (bottom)."
        )
        capacity_line = (
            " Bottom panel shows coded KV tokens capacity and Capacity "
            "concurrent requests (same numbers as sections A and D)."
        )
    elif weight_line:
        fig_caption = "Figure. Weights and KV vs 1-user context."
    lines = [
        "### KV vs context",
        "",
        (
            f"`ISIRO_KV={alphabet}` on KV from a Project Gutenberg text "
            f"forward. {save_line} That save rate meters KV tokens capacity "
            f"(and Capacity concurrency)."
            f"{weight_line}{capacity_line}"
        ),
        "",
        fig_caption,
        "",
        f"![{fig_caption.removeprefix('Figure. ').rstrip('.')}]({isiro_kv_doc['image']})",
        "",
    ]
    return lines


def _mode_sections(
    summary: dict[str, Any],
    commands: dict[str, Any] | None = None,
) -> list[str]:
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
    cmds = commands if isinstance(commands, dict) else {}
    if not cmds and isinstance(summary.get("commands"), dict):
        cmds = summary["commands"]
    isiro_kv_doc = summary.get("isiro_kv_bytes") if isinstance(summary.get("isiro_kv_bytes"), dict) else {}
    isiro_kv_on = bool(str(cmds.get("isiro_kv") or "").strip()) or bool(
        isiro_kv_doc.get("image") or isiro_kv_doc.get("alphabet")
    )
    kv_only = bool(cmds.get("isiro_kv_only"))
    lines: list[str] = []
    lines.extend(_section_gpu_excluding_kv(footprint, kv_only=kv_only))
    lines.extend(
        _section_kv_reservation(kv, gib_only=isiro_kv_on, kv_only=kv_only)
    )
    lines.extend(_section_isiro_kv_bytes(summary))
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
                gib_only=isiro_kv_on,
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
    isiro_kv = commands.get("isiro_kv")
    kv_cell = "product KV cache size"
    if kv_dtype or isiro_kv:
        parts = []
        if kv_dtype:
            parts.append(f"dtype `{kv_dtype}`")
        if isiro_kv:
            parts.append(f"`ISIRO_KV={isiro_kv}` on Isiro")
        kv_cell = "; ".join(parts)
    lines = [
        "## Config",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| System | `{summary.get('system_id', '')}` |",
        f"| CUDA graphs | {graph_cell} |",
        f"| Weights | `{summary.get('precision', '')}` |",
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
    if isiro_kv:
        lines.append(f"| ISIRO_KV | `{isiro_kv}` |")
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
    # Prefer curve save_vs_fa; else bitexact avg_bpv/fa_bpv from correctness.
    _fold_coded_kv_capacity(summary)
    environment = load_json(run_dir / "environment.json")
    commands = load_json(run_dir / "commands.json")
    footprint = summary["footprint"]
    correctness = summary["correctness"]
    graph_mode = _graph_mode(commands)
    lines = _section_intro(summary, commands)
    lines.extend(
        _section_a(
            footprint,
            correctness,
            kv_only=bool(commands.get("isiro_kv_only")),
        )
    )
    lines.extend(_mode_sections(summary, commands))
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
        isiro_kv_curve = run_dir / "isiro_kv_bytes_curve.json"
        isiro_kv_png_src = run_dir / "kv-bytes-vs-context.png"
        if isiro_kv_curve.is_file() and isiro_kv_png_src.is_file():
            import shutil as _shutil

            isiro_kv_doc = load_json(isiro_kv_curve)
            _isiro_kv_alph = str(isiro_kv_doc.get("alphabet") or "").strip().lower()
            if _isiro_kv_alph in ("fp8", "fp8_e4m3", "fp4", "nvfp4", "e2m1"):
                if str(isiro_kv_doc.get("kv_source") or "") != "vendor_fa_pages":
                    raise RuntimeError(
                        "isiro_kv_bytes_curve.json alphabet="
                        f"{_isiro_kv_alph!r} requires kv_source=vendor_fa_pages "
                        "(vendor FA pages; refuse bf16-stand-in curves at publish)"
                    )
            dest_png = output.parent / "kv-bytes-vs-context.png"
            # Re-render when summary has weight bytes so the 2x1 panel stays current.
            used_capacity_bars = False
            try:
                from isiro_kv_bytes_curve import render_png as _render_isiro_kv_png

                fp = None
                summary_probe = run_dir / "summary.json"
                if summary_probe.is_file():
                    fp = (load_json(summary_probe).get("footprint") or {})
                b_w = isiro_kv_doc.get("baseline_weight_bytes")
                t_w = isiro_kv_doc.get("isiro_weight_bytes")
                if fp:
                    b_fp = (fp.get("baseline") or {}).get(
                        "model_loading_bytes_reported"
                    )
                    t_fp = (fp.get("tic") or {}).get("model_loading_bytes_reported")
                    if isinstance(b_fp, (int, float)) and int(b_fp) > 0:
                        b_w = int(b_fp)
                    if isinstance(t_fp, (int, float)) and int(t_fp) > 0:
                        t_w = int(t_fp)
                if isinstance(b_w, (int, float)) and isinstance(t_w, (int, float)):
                    isiro_kv_doc["baseline_weight_bytes"] = int(b_w)
                    isiro_kv_doc["isiro_weight_bytes"] = int(t_w)
                env = load_json(run_dir / "environment.json") if (
                    run_dir / "environment.json"
                ).is_file() else {}
                model_tag = (
                    str(env.get("model_id") or "")
                    or str(env.get("model") or "")
                    or str(
                        (load_json(run_dir / "summary.json").get("model_id") or "")
                        if (run_dir / "summary.json").is_file()
                        else ""
                    )
                )
                if not model_tag and (run_dir / "commands.json").is_file():
                    cmds = load_json(run_dir / "commands.json")
                    # Prefer served-model-name from baseline argv when present.
                    argv = list(cmds.get("baseline_serve") or [])
                    if "--served-model-name" in argv:
                        i = argv.index("--served-model-name")
                        if i + 1 < len(argv):
                            model_tag = str(argv[i + 1])
                summary_for_prec = (
                    load_json(run_dir / "summary.json")
                    if (run_dir / "summary.json").is_file()
                    else {}
                )
                cmds_for_prec = (
                    load_json(run_dir / "commands.json")
                    if (run_dir / "commands.json").is_file()
                    else {}
                )
                w_prec = str(
                    summary_for_prec.get("precision")
                    or isiro_kv_doc.get("weight_precision")
                    or ""
                ).strip()
                kv_prec = str(
                    cmds_for_prec.get("isiro_kv")
                    or cmds_for_prec.get("kv_cache_dtype")
                    or isiro_kv_doc.get("kv_precision")
                    or isiro_kv_doc.get("alphabet")
                    or ""
                ).strip()
                render_kw: dict[str, Any] = {
                    "gpu": str(env.get("gpu_model") or ""),
                    "gpu_count": env.get("gpu_count"),
                    "model": model_tag,
                    "weight_precision": w_prec,
                    "kv_precision": kv_prec,
                }
                used_capacity_bars = False
                # KV-only / identical weights: 1x1 KV vs context (capacity and
                # concurrency stay in markdown tables A/D; no bar panel).
                kv_only_run = (
                    str(
                        cmds_for_prec.get("isiro_format")
                        or summary_for_prec.get("compiler")
                        or summary_for_prec.get("runtime")
                        or ""
                    )
                    .strip()
                    .lower()
                    == "kv-only"
                )
                if kv_only_run or (
                    isinstance(isiro_kv_doc.get("baseline_weight_bytes"), (int, float))
                    and isinstance(isiro_kv_doc.get("isiro_weight_bytes"), (int, float))
                    and int(isiro_kv_doc["baseline_weight_bytes"])
                    == int(isiro_kv_doc["isiro_weight_bytes"])
                ):
                    isiro_kv_doc.pop("baseline_weight_bytes", None)
                    isiro_kv_doc.pop("isiro_weight_bytes", None)
                _render_isiro_kv_png(isiro_kv_doc, dest_png, **render_kw)
            except Exception:
                _shutil.copy2(isiro_kv_png_src, dest_png)
                used_capacity_bars = False
            pts = list(isiro_kv_doc.get("points") or [])
            save_max = None
            max_t = None
            b_gib = None
            t_gib = None
            save_gib = None
            b_w_gib = None
            t_w_gib = None
            if pts:
                last = pts[-1]
                save_max = last.get("save_vs_fa")
                max_t = last.get("T")
                fa_b = last.get("fa_twin_nbytes")
                isiro_kv_b = last.get("isiro_kv_total_nbytes")
                if isinstance(fa_b, (int, float)):
                    b_gib = float(fa_b) / float(1024**3)
                if isinstance(isiro_kv_b, (int, float)):
                    t_gib = float(isiro_kv_b) / float(1024**3)
                if b_gib is not None and t_gib is not None:
                    save_gib = float(b_gib) - float(t_gib)
            bw = isiro_kv_doc.get("baseline_weight_bytes")
            tw = isiro_kv_doc.get("isiro_weight_bytes")
            if isinstance(bw, (int, float)):
                b_w_gib = float(bw) / float(1024**3)
            if isinstance(tw, (int, float)):
                t_w_gib = float(tw) / float(1024**3)
            summary_path = temp / "summary.json"
            if summary_path.is_file():
                summary = load_json(summary_path)
                payload_bpv = last.get("payload_bpv") if pts else None
                summary["isiro_kv_bytes"] = {
                    "image": dest_png.name,
                    "alphabet": isiro_kv_doc.get("alphabet"),
                    "live_window": isiro_kv_doc.get("live_window"),
                    "fa_bpv": isiro_kv_doc.get("fa_bpv"),
                    "max_t": max_t,
                    "baseline_kv_gib_at_max_t": b_gib,
                    "isiro_kv_gib_at_max_t": t_gib,
                    "save_gib_at_max_t": save_gib,
                    "save_at_max_t": save_max,
                    "payload_bpv_at_max_t": (
                        float(payload_bpv)
                        if isinstance(payload_bpv, (int, float))
                        else None
                    ),
                    "baseline_weight_gib": b_w_gib,
                    "isiro_weight_gib": t_w_gib,
                    "capacity_bars": bool(used_capacity_bars),
                    "ok": bool(isiro_kv_doc.get("ok")),
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
