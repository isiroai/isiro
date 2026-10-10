#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capacity-search helpers for the public serve A/B harness (v0.1.0)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

CAPACITY_SCHEMA = "isiro-benchmark-capacity-v1"
EXPERIMENT_CAPACITY = "capacity"
EXPERIMENT_EQUAL_BATCH = "equal-batch"


def on_disk_weight_bytes(path: Path | str) -> int:
    """Weight bytes for a dense dir (``*.safetensors``) or a ``.tic`` file/dir."""
    target = Path(path)
    if target.is_file():
        return target.stat().st_size
    tic = target / "model.tic"
    if tic.is_file():
        return tic.stat().st_size
    weight_files = sorted(target.rglob("*.safetensors"))
    if not weight_files:
        raise ValueError(
            f"no .safetensors or model.tic weights found under {target}"
        )
    return sum(item.stat().st_size for item in weight_files)


def default_scale_hi(
    baseline_seqs: int,
    weight_baseline_bytes: int | float,
    weight_tic_bytes: int | float,
) -> int:
    """Upper clamp from weight headroom: round(baseline_seqs * W_b / W_t)."""
    if baseline_seqs < 1:
        raise ValueError("baseline_seqs must be >= 1")
    w_b = float(weight_baseline_bytes)
    w_t = float(weight_tic_bytes)
    if w_b <= 0 or w_t <= 0:
        raise ValueError("weight bytes must be positive")
    headroom = w_b / w_t
    return max(baseline_seqs, int(round(baseline_seqs * headroom)))


# vLLM default capture grid after 4: multiples of 8. Off-stride capacity
# (e.g. 39) truncates FULL decode graphs to 32 and taxes ITL/tok/s.
_CUDAGRAPH_CAPTURE_STRIDE = 8

# vLLM default max_num_batched_tokens. Past this, fair-share chunked prefill
# needs a second wave (Capacity TTFT cliff once conc*prompt_tokens > 2048).
_VENDOR_DEFAULT_BATCHED_TOKENS = 2048
# Chat-template / special-token pad on top of random input_len. Metered
# generation-32-256 on Qwen chat is ~61 input tokens/prompt (pad≈29). Use 30
# so conc=33 still fits vendor 2048; pad=32 falsely forced one-wave at 33/34.
_PROMPT_TOKEN_PAD = 30


def align_seqs_to_cudagraph_stride(seqs: int, scale_hi: int) -> int:
    """Round ``seqs`` up onto the vLLM capture stride when it still fits ``hi``."""
    if seqs < 1:
        raise ValueError("seqs must be >= 1")
    if scale_hi < seqs:
        raise ValueError("scale_hi must be >= seqs")
    stride = int(_CUDAGRAPH_CAPTURE_STRIDE)
    aligned = ((int(seqs) + stride - 1) // stride) * stride
    if aligned > scale_hi:
        return int(seqs)
    return aligned


def one_wave_batched_tokens(
    concurrency: int,
    input_tokens: int,
    *,
    vendor_default: int = _VENDOR_DEFAULT_BATCHED_TOKENS,
    prompt_pad: int = _PROMPT_TOKEN_PAD,
) -> int | None:
    """Budget so one prefill wave fits ``concurrency`` prompts, or None.

    Returns None when the vendor default already fits (no harness override).
    Used only by the capacity A/B harness; product serve must not setdefault
    ``max_num_batched_tokens``.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if input_tokens < 0:
        raise ValueError("input_tokens must be non-negative")
    if vendor_default < 1:
        raise ValueError("vendor_default must be >= 1")
    prompt = int(input_tokens) + max(0, int(prompt_pad))
    need = int(concurrency) * prompt
    if need <= int(vendor_default):
        return None
    # Round up to 256 for a stable vendor knob; never below need.
    # Keep near need (e.g. 2560 at conc=40): a much larger compile range
    # (4096) lengthened Capacity TTFT on the 5090 A/B.
    step = 256
    return max(int(vendor_default), ((need + step - 1) // step) * step)


def scale_seqs_from_kv_estimate(
    baseline_seqs: int,
    kv_baseline_bytes: int | float,
    weight_baseline_bytes: int | float,
    weight_tic_bytes: int | float,
    scale_hi: int,
) -> int:
    """Scale TIC max_num_seqs from baseline KV plus freed on-disk weight bytes.

    ``kv_tic_est = kv_baseline + max(0, W_baseline - W_tic)``;
    ``seqs = clamp(round(baseline_seqs * kv_tic_est / kv_baseline), baseline, hi)``,
    then round up onto the vLLM CUDA-graph capture stride when it still fits.
    """
    if baseline_seqs < 1:
        raise ValueError("baseline_seqs must be >= 1")
    if scale_hi < baseline_seqs:
        raise ValueError("scale_hi must be >= baseline_seqs")
    kv_b = float(kv_baseline_bytes)
    if kv_b <= 0:
        raise ValueError("kv_baseline_bytes must be positive")
    w_b = float(weight_baseline_bytes)
    w_t = float(weight_tic_bytes)
    if w_b <= 0 or w_t <= 0:
        raise ValueError("weight bytes must be positive")
    freed = max(0.0, w_b - w_t)
    ratio = (kv_b + freed) / kv_b
    scaled = int(round(baseline_seqs * ratio))
    clamped = max(baseline_seqs, min(scale_hi, scaled))
    return align_seqs_to_cudagraph_stride(clamped, scale_hi)


def kv_scale_ratio_estimated(
    kv_baseline_bytes: int | float,
    weight_baseline_bytes: int | float,
    weight_tic_bytes: int | float,
) -> float:
    """Estimated TIC/baseline KV bytes ratio used for seq scaling."""
    kv_b = float(kv_baseline_bytes)
    if kv_b <= 0:
        raise ValueError("kv_baseline_bytes must be positive")
    w_b = float(weight_baseline_bytes)
    w_t = float(weight_tic_bytes)
    if w_b <= 0 or w_t <= 0:
        raise ValueError("weight bytes must be positive")
    freed = max(0.0, w_b - w_t)
    return (kv_b + freed) / kv_b


def scale_seqs_from_kv_measured(
    baseline_seqs: int,
    kv_baseline_tokens: int | float,
    kv_tic_tokens: int | float,
    scale_hi: int,
) -> int:
    """Scale TIC max_num_seqs from a measured vLLM KV token ratio.

    ``seqs = clamp(round(baseline_seqs * kv_tic / kv_baseline), baseline, hi)``,
    then round up onto the vLLM CUDA-graph capture stride when it still fits
    (39→40) so FULL decode graphs cover capacity concurrency.
    """
    if baseline_seqs < 1:
        raise ValueError("baseline_seqs must be >= 1")
    if scale_hi < baseline_seqs:
        raise ValueError("scale_hi must be >= baseline_seqs")
    kv_b = float(kv_baseline_tokens)
    kv_t = float(kv_tic_tokens)
    if kv_b <= 0:
        raise ValueError("kv_baseline_tokens must be positive")
    if kv_t < 0:
        raise ValueError("kv_tic_tokens must be non-negative")
    scaled = int(round(baseline_seqs * (kv_t / kv_b)))
    clamped = max(baseline_seqs, min(scale_hi, scaled))
    return align_seqs_to_cudagraph_stride(clamped, scale_hi)


def kv_scale_ratio_measured(
    kv_baseline_tokens: int | float,
    kv_tic_tokens: int | float,
) -> float:
    """Measured TIC/baseline KV token ratio (vLLM startup logs)."""
    kv_b = float(kv_baseline_tokens)
    kv_t = float(kv_tic_tokens)
    if kv_b <= 0:
        raise ValueError("kv_baseline_tokens must be positive")
    if kv_t < 0:
        raise ValueError("kv_tic_tokens must be non-negative")
    return kv_t / kv_b


def next_probe(lo: int, hi: int) -> int:
    """Midpoint for binary search (inclusive lo/hi)."""
    if lo > hi:
        raise ValueError(f"empty search range lo={lo} hi={hi}")
    return (lo + hi) // 2


def update_bounds_after_probe(
    lo: int, hi: int, probe: int, success: bool
) -> tuple[int, int, int | None]:
    """Return (new_lo, new_hi, best_if_success_else_None_update).

    Caller tracks `best` across iterations. On success, search higher;
    on failure, search lower.
    """
    if probe < lo or probe > hi:
        raise ValueError(f"probe {probe} outside [{lo}, {hi}]")
    if success:
        return probe + 1, hi, probe
    return lo, probe - 1, None


def search_max(
    lo: int,
    hi: int,
    probe_fn,
) -> int:
    """Binary-search the largest integer in [lo, hi] for which probe_fn is true."""
    if lo < 1:
        raise ValueError("lo must be >= 1")
    if hi < lo:
        raise ValueError("hi must be >= lo")
    best = lo if probe_fn(lo) else 0
    if best == 0:
        return 0
    cur_lo, cur_hi = lo + 1, hi
    while cur_lo <= cur_hi:
        mid = next_probe(cur_lo, cur_hi)
        if probe_fn(mid):
            best = mid
            cur_lo = mid + 1
        else:
            cur_hi = mid - 1
    return best


def search_max_hi_first(
    lo: int,
    hi: int,
    probe_fn,
) -> int:
    """Like search_max, but probe ``hi`` first (1 probe when the budget fits).

    Cold serve starts dominate wall time. When ``SERVE_MAX_NUM_SEQS`` fits in
    HBM, probing hi first avoids ~log2(hi) full engine launches.
    """
    if lo < 1:
        raise ValueError("lo must be >= 1")
    if hi < lo:
        raise ValueError("hi must be >= lo")
    if probe_fn(hi):
        return hi
    if hi == lo:
        return 0
    return search_max(lo, hi - 1, probe_fn)


def concurrency_for_seqs(max_num_seqs: int, bench_cap: int) -> int:
    """Bench concurrency cannot exceed serve max_num_seqs."""
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be >= 1")
    if bench_cap < 1:
        raise ValueError("bench_cap must be >= 1")
    return min(max_num_seqs, bench_cap)


def build_capacity_doc(
    *,
    gpu_memory_utilization: float,
    max_model_len: int,
    baseline_max_num_seqs: int,
    tic_max_num_seqs: int,
    baseline_max_concurrency: int,
    tic_max_concurrency: int,
    search_lo: int,
    search_hi: int,
    graph_mode: str,
    seqs_scale_mode: str | None = None,
    kv_scale_ratio_estimated: float | None = None,
    seqs_implied_weight_estimate: int | None = None,
    kv_scale_ratio_measured: float | None = None,
    seqs_implied_kv_measured: int | None = None,
    kv_measured_source: str | None = None,
) -> dict[str, Any]:
    batch_ratio = None
    if baseline_max_num_seqs > 0:
        batch_ratio = round(tic_max_num_seqs / baseline_max_num_seqs, 4)
    doc: dict[str, Any] = {
        "schema": CAPACITY_SCHEMA,
        "experiment_kind": EXPERIMENT_CAPACITY,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "graph_mode": graph_mode,
        "search_lo": search_lo,
        "search_hi": search_hi,
        "baseline": {
            "max_num_seqs": baseline_max_num_seqs,
            "max_concurrency": baseline_max_concurrency,
        },
        "tic": {
            "max_num_seqs": tic_max_num_seqs,
            "max_concurrency": tic_max_concurrency,
        },
        "batch_ratio_derived": batch_ratio,
    }
    if seqs_scale_mode is not None:
        doc["seqs_scale_mode"] = seqs_scale_mode
    if kv_scale_ratio_estimated is not None:
        doc["kv_scale_ratio_estimated"] = round(float(kv_scale_ratio_estimated), 4)
    if seqs_implied_weight_estimate is not None:
        doc["seqs_implied_weight_estimate"] = int(seqs_implied_weight_estimate)
    if kv_scale_ratio_measured is not None:
        doc["kv_scale_ratio_measured"] = round(float(kv_scale_ratio_measured), 4)
    if seqs_implied_kv_measured is not None:
        doc["seqs_implied_kv_measured"] = int(seqs_implied_kv_measured)
    if kv_measured_source is not None:
        doc["kv_measured_source"] = str(kv_measured_source)
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("concurrency", help="min(max_num_seqs, bench_cap)")
    p.add_argument("--max-num-seqs", type=int, required=True)
    p.add_argument("--bench-cap", type=int, required=True)
    w = sub.add_parser("weight-bytes", help="on-disk *.safetensors bytes")
    w.add_argument("--path", required=True)
    d = sub.add_parser(
        "default-scale-hi",
        help="round(baseline_seqs * weight_baseline / weight_tic)",
    )
    d.add_argument("--baseline-seqs", type=int, required=True)
    d.add_argument("--weight-baseline-bytes", type=int, required=True)
    d.add_argument("--weight-tic-bytes", type=int, required=True)
    s = sub.add_parser(
        "scale-seqs",
        help="KV-estimate TIC max_num_seqs from baseline KV + freed weights",
    )
    s.add_argument("--baseline-seqs", type=int, required=True)
    s.add_argument("--kv-baseline-bytes", type=int, required=True)
    s.add_argument("--weight-baseline-bytes", type=int, required=True)
    s.add_argument("--weight-tic-bytes", type=int, required=True)
    s.add_argument("--scale-hi", type=int, required=True)
    r = sub.add_parser(
        "kv-scale-ratio",
        help="estimated TIC/baseline KV bytes ratio for seq scaling",
    )
    r.add_argument("--kv-baseline-bytes", type=int, required=True)
    r.add_argument("--weight-baseline-bytes", type=int, required=True)
    r.add_argument("--weight-tic-bytes", type=int, required=True)
    sm = sub.add_parser(
        "scale-seqs-measured",
        help="TIC max_num_seqs from measured vLLM KV token ratio",
    )
    sm.add_argument("--baseline-seqs", type=int, required=True)
    sm.add_argument("--kv-baseline-tokens", type=int, required=True)
    sm.add_argument("--kv-tic-tokens", type=int, required=True)
    sm.add_argument("--scale-hi", type=int, required=True)
    rm = sub.add_parser(
        "kv-scale-ratio-measured",
        help="measured TIC/baseline KV token ratio",
    )
    rm.add_argument("--kv-baseline-tokens", type=int, required=True)
    rm.add_argument("--kv-tic-tokens", type=int, required=True)
    bt = sub.add_parser(
        "one-wave-batched-tokens",
        help="capacity harness max_num_batched_tokens (or empty if default fits)",
    )
    bt.add_argument("--concurrency", type=int, required=True)
    bt.add_argument("--input-tokens", type=int, required=True)
    bt.add_argument(
        "--vendor-default",
        type=int,
        default=_VENDOR_DEFAULT_BATCHED_TOKENS,
    )
    bt.add_argument("--prompt-pad", type=int, default=_PROMPT_TOKEN_PAD)
    args = parser.parse_args(argv)
    if args.cmd == "concurrency":
        print(concurrency_for_seqs(args.max_num_seqs, args.bench_cap))
        return 0
    if args.cmd == "weight-bytes":
        print(on_disk_weight_bytes(args.path))
        return 0
    if args.cmd == "default-scale-hi":
        print(
            default_scale_hi(
                args.baseline_seqs,
                args.weight_baseline_bytes,
                args.weight_tic_bytes,
            )
        )
        return 0
    if args.cmd == "scale-seqs":
        print(
            scale_seqs_from_kv_estimate(
                args.baseline_seqs,
                args.kv_baseline_bytes,
                args.weight_baseline_bytes,
                args.weight_tic_bytes,
                args.scale_hi,
            )
        )
        return 0
    if args.cmd == "kv-scale-ratio":
        ratio = kv_scale_ratio_estimated(
            args.kv_baseline_bytes,
            args.weight_baseline_bytes,
            args.weight_tic_bytes,
        )
        print(f"{ratio:.6f}")
        return 0
    if args.cmd == "scale-seqs-measured":
        print(
            scale_seqs_from_kv_measured(
                args.baseline_seqs,
                args.kv_baseline_tokens,
                args.kv_tic_tokens,
                args.scale_hi,
            )
        )
        return 0
    if args.cmd == "kv-scale-ratio-measured":
        ratio = kv_scale_ratio_measured(
            args.kv_baseline_tokens,
            args.kv_tic_tokens,
        )
        print(f"{ratio:.6f}")
        return 0
    if args.cmd == "one-wave-batched-tokens":
        budget = one_wave_batched_tokens(
            args.concurrency,
            args.input_tokens,
            vendor_default=args.vendor_default,
            prompt_pad=args.prompt_pad,
        )
        if budget is None:
            print("")
        else:
            print(int(budget))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
