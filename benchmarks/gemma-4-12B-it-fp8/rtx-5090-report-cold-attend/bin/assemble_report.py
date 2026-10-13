#!/usr/bin/env python3
"""Assemble cold-attend report.md from A/B run + residency summary.

Latency GO (fail-closed vs published baseline):
  - Isiro output tok/s > baseline tok/s
  - Isiro ITL p50 < baseline ITL p50
  - Isiro ITL p95/p99 and TTFT p95/p99 not worse than baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Published rtx-5090-report baseline bar (report.md Generation table).
PUBLISHED_BASELINE = {
    "tok_s": 2650.30,
    "itl": (11.53, 12.96, 13.42),
    "tpot": (11.64, 11.73, 11.82),
    "ttft": (115.66, 143.84, 144.06),
}


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _bench_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    # Harness profile JSON (isiro-benchmark-profile-v1).
    if doc.get("schema") == "isiro-benchmark-profile-v1" or "throughput" in doc:
        lat = doc.get("latency_ms") or (doc.get("aggregate_summary") or {}).get(
            "latency_ms"
        ) or {}
        thr = doc.get("throughput") or (doc.get("aggregate_summary") or {}).get(
            "throughput"
        ) or {}
        return {
            "tok_s": thr.get("output_tokens_per_sec"),
            "itl_p50": lat.get("itl_p50"),
            "itl_p95": lat.get("itl_p95"),
            "itl_p99": lat.get("itl_p99"),
            "tpot_p50": lat.get("tpot_p50"),
            "tpot_p95": lat.get("tpot_p95"),
            "tpot_p99": lat.get("tpot_p99"),
            "ttft_p50": lat.get("ttft_p50"),
            "ttft_p95": lat.get("ttft_p95"),
            "ttft_p99": lat.get("ttft_p99"),
        }
    tok = doc.get("output_throughput") or doc.get("mean_output_throughput")
    itls = doc.get("inter_token_latencies_ms") or doc.get("itls") or []
    tpots = doc.get("tpot_ms") or doc.get("tpots") or []
    ttfts = doc.get("ttfts_ms") or doc.get("ttfts") or []
    out = {
        "tok_s": float(tok) if tok is not None else None,
        "itl_p50": _pct(list(itls), 50) if itls else doc.get("median_itl_ms"),
        "itl_p95": _pct(list(itls), 95) if itls else doc.get("p95_itl_ms"),
        "itl_p99": _pct(list(itls), 99) if itls else doc.get("p99_itl_ms"),
        "tpot_p50": _pct(list(tpots), 50) if tpots else doc.get("median_tpot_ms"),
        "tpot_p95": _pct(list(tpots), 95) if tpots else doc.get("p95_tpot_ms"),
        "tpot_p99": _pct(list(tpots), 99) if tpots else doc.get("p99_tpot_ms"),
        "ttft_p50": _pct(list(ttfts), 50) if ttfts else doc.get("median_ttft_ms"),
        "ttft_p95": _pct(list(ttfts), 95) if ttfts else doc.get("p95_ttft_ms"),
        "ttft_p99": _pct(list(ttfts), 99) if ttfts else doc.get("p99_ttft_ms"),
    }
    for src, dst in (
        ("mean_itl_ms", "itl_p50"),
        ("median_itl_ms", "itl_p50"),
        ("p95_itl_ms", "itl_p95"),
        ("p99_itl_ms", "itl_p99"),
        ("mean_tpot_ms", "tpot_p50"),
        ("median_tpot_ms", "tpot_p50"),
        ("p95_tpot_ms", "tpot_p95"),
        ("p99_tpot_ms", "tpot_p99"),
        ("mean_ttft_ms", "ttft_p50"),
        ("median_ttft_ms", "ttft_p50"),
        ("p95_ttft_ms", "ttft_p95"),
        ("p99_ttft_ms", "ttft_p99"),
        ("output_throughput", "tok_s"),
    ):
        if out.get(dst) is None and doc.get(src) is not None:
            out[dst] = float(doc[src])
    return out


def latency_go(isiro: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    bt, it = baseline.get("tok_s"), isiro.get("tok_s")
    if bt is None or it is None or not (it > bt):
        ok = False
        reasons.append(f"tok/s lose or missing (isiro={it} baseline={bt})")
    bi, ii = baseline.get("itl_p50"), isiro.get("itl_p50")
    if bi is None or ii is None or not (ii < bi):
        ok = False
        reasons.append(f"ITL p50 lose or missing (isiro={ii} baseline={bi})")
    for key in ("itl_p95", "itl_p99", "ttft_p95", "ttft_p99"):
        b, i = baseline.get(key), isiro.get(key)
        if b is None or i is None:
            ok = False
            reasons.append(f"{key} missing")
        elif i > b:
            ok = False
            reasons.append(f"{key} regress (isiro={i} baseline={b})")
    if ok:
        reasons.append("latency GO vs fresh baseline")
    return ok, reasons


def fmt_triple(p50: Any, p95: Any, p99: Any) -> str:
    def f(x: Any) -> str:
        return "-" if x is None else f"{float(x):.2f}"

    return f"{f(p50)} / {f(p95)} / {f(p99)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--residency", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compiler", default="v0.1.0")
    ap.add_argument("--runtime", default="v0.1.1")
    args = ap.parse_args()

    residency = json.loads(args.residency.read_text(encoding="utf-8"))
    # Prefer harness-normalized bench JSON if present
    isiro_bench = None
    base_bench = None
    for cand in (
        args.run_dir / "isiro" / "generation-32-256.json",
        args.run_dir / "equal_batch" / "isiro" / "generation-32-256.json",
    ):
        if cand.is_file():
            isiro_bench = _bench_metrics(cand)
            break
    for cand in (
        args.run_dir / "baseline" / "generation-32-256.json",
        args.run_dir / "equal_batch" / "baseline" / "generation-32-256.json",
    ):
        if cand.is_file():
            base_bench = _bench_metrics(cand)
            break

    summary_path = args.run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if isiro_bench is None or isiro_bench.get("tok_s") is None:
        raise SystemExit(f"missing Isiro generation profile under {args.run_dir}")
    if base_bench is None or base_bench.get("tok_s") is None:
        raise SystemExit(f"missing Baseline generation profile under {args.run_dir}")

    lat_ok, lat_reasons = latency_go(isiro_bench, base_bench)
    # Also compare Isiro to published baseline bar (secondary callout)
    pub_base = {
        "tok_s": PUBLISHED_BASELINE["tok_s"],
        "itl_p50": PUBLISHED_BASELINE["itl"][0],
        "itl_p95": PUBLISHED_BASELINE["itl"][1],
        "itl_p99": PUBLISHED_BASELINE["itl"][2],
        "ttft_p95": PUBLISHED_BASELINE["ttft"][1],
        "ttft_p99": PUBLISHED_BASELINE["ttft"][2],
    }
    pub_ok, pub_reasons = latency_go(isiro_bench, pub_base)

    res_go = bool(residency.get("residency_go"))
    stamp = "GO" if (res_go and lat_ok) else "NO GO"

    corr = summary.get("correctness") or {}
    foot = summary.get("footprint") or {}
    kv = summary.get("kv_cache") or {}
    disk_sav = foot.get("on_disk_savings_pct_derived")
    b_disk = (foot.get("baseline") or {}).get("on_disk_bytes_measured")
    t_disk = (foot.get("tic") or {}).get("on_disk_bytes_measured")
    b_w = (foot.get("baseline") or {}).get(
        "weights_and_non_kv_gpu_memory_bytes_derived"
    ) or (foot.get("baseline") or {}).get("non_kv_cache_memory_bytes_reported")
    t_w = (foot.get("tic") or {}).get(
        "weights_and_non_kv_gpu_memory_bytes_derived"
    ) or (foot.get("tic") or {}).get("non_kv_cache_memory_bytes_reported")

    def gib(x: Any) -> str:
        if x is None:
            return "-"
        return f"{float(x) / (1024**3):.2f}"

    def gb(x: Any) -> str:
        if x is None:
            return "-"
        return f"{float(x) / 1e9:.2f} GB"

    w_sav = None
    if b_w and t_w and float(b_w) > 0:
        w_sav = 100.0 * (1.0 - float(t_w) / float(b_w))

    lines: list[str] = []
    lines.append("# ISIRO Benchmark Report: `gemma-4-12B-it-fp8` (cold-attend)")
    lines.append("")
    lines.append(
        f"`fp8` | `rtx-5090` | cold-attend FA `path_a` | compiler `{args.compiler}` | "
        f"runtime `{args.runtime}` | stamp **{stamp}**"
    )
    lines.append("")
    lines.append("Tooling:")
    lines.append("")
    lines.append("- `vllm bench serve` (tok/s, ITL)")
    lines.append("- Path A residency scrape (`remeter_path_a_vs_vanilla.py`) for cold deposit / twin free")
    lines.append("")
    lines.append("Compared:")
    lines.append("")
    lines.append("- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype bfloat16`")
    lines.append(
        "- **Isiro:** `isiro serve … --target vllm` with TIC weights and `ISIRO_GDX=bf16` "
        "(cold KV compress on vendor FA pages; `ISIRO_GDX_COLD_READY=1`, `ISIRO_GDX_FUSED_ATTEND=0`)"
    )
    lines.append("")
    lines.append(
        "**Scope:** FA `path_a` / `fa-kv` only. Not a GLM `path_a_mla` / `o1-mla-latent` proxy "
        "(see `SCOPE.md`). Dual GO: residency reclaim meters + latency beat."
    )
    lines.append("")
    lines.append("## Correctness")
    lines.append("")
    lines.append("| Metric | Baseline | Isiro |")
    lines.append("|--------|----------|-------|")
    size_cell = gb(t_disk)
    if disk_sav is not None and b_disk is not None:
        size_cell = f"**{gb(t_disk)}** (**{float(disk_sav):.2f}% smaller**)"
    lines.append(f"| On-disk model size | {gb(b_disk)} | {size_cell} |")
    lines.append(
        f"| Integrity | baseline weights | "
        f"**{'PASS' if corr.get('integrity_ok') or corr.get('bit_exact_ok') else 'see verify'}** "
        f"(`isiro verify`) |"
    )
    lines.append(
        f"| KV bit-exactness | vendor KV pages | "
        f"**{'PASS' if corr.get('kv_bit_exact_ok') else 'see meters'}** "
        f"(GDX `bf16` cold roundtrip) |"
    )
    lines.append(
        f"| Cold deposit / twin free | - | "
        f"**{'PASS' if res_go else 'FAIL'}** "
        f"(`dual_hold={residency.get('dual_hold')}`, "
        f"`f_cold={residency.get('f_cold')}`, "
        f"`physical_reclaim_armed={residency.get('physical_reclaim_armed')}`) |"
    )
    lines.append(
        f"| NIAH (residency scrape) | - | "
        f"**{'PASS' if residency.get('niah_all_pass') else 'FAIL'}** |"
    )
    lines.append("")
    lines.append("## A. GPU memory")
    lines.append("")
    lines.append("### Weights")
    lines.append("")
    lines.append("| Metric | Baseline | Isiro | Savings % |")
    lines.append("|--------|----------|-------|-----------|")
    lines.append(
        f"| GPU memory for loaded weights | {gib(b_w)} GiB | **{gib(t_w)} GiB** | "
        f"**{w_sav:.2f}%** |" if w_sav is not None else
        f"| GPU memory for loaded weights | {gib(b_w)} GiB | **{gib(t_w)} GiB** | - |"
    )
    lines.append("")
    lines.append("### KV cache (reserved pool)")
    lines.append("")
    lines.append("| Metric | Baseline | Isiro |")
    lines.append("|--------|----------|-------|")
    lines.append(
        f"| GPU KV cache tokens (log) | "
        f"{kv.get('baseline_tokens_measured', '-')} | "
        f"{residency.get('gpu_kv_tokens', kv.get('tic_tokens_measured', '-'))} |"
    )
    lines.append("")
    lines.append(
        "Reserved pool is vendor admission slots, not GDX-coded cold capacity. "
        "On FA path_a, process FA twin bytes drop when vendor Sliding or pool reclaim "
        "arms persist; `free_fa_twin` clears the live twin ledger (`live_fa_nbytes=0`)."
    )
    lines.append("")
    lines.append("### Live GDX residency (Isiro, after long prompt)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k in (
        "dual_hold",
        "f_cold",
        "save_vs_all_fa",
        "live_fa_nbytes",
        "freed_fa_nbytes",
        "n_live_persist",
        "n_deposit",
        "attend_class",
        "ledger_twin_free",
        "physical_reclaim_armed",
        "residency_go",
    ):
        lines.append(f"| `{k}` | {residency.get(k)} |")
    lines.append("")
    lines.append(f"_{residency.get('note', '')}_")
    lines.append("")
    lines.append("## C. Capacity")
    lines.append("")
    lines.append("| Metric | Baseline | Isiro |")
    lines.append("|--------|----------|-------|")
    lines.append("| Concurrent requests | 32 | 32 |")
    lines.append("")
    lines.append("### Generation (recall-matched KV, input 32 / output 256)")
    lines.append("")
    lines.append("| Metric | Baseline | Isiro |")
    lines.append("|--------|----------|-------|")
    lines.append(
        f"| Output tok/s | {base_bench.get('tok_s')} | "
        f"**{isiro_bench.get('tok_s')}** |"
    )
    lines.append(
        f"| ITL (ms) p50 / p95 / p99 | "
        f"{fmt_triple(base_bench.get('itl_p50'), base_bench.get('itl_p95'), base_bench.get('itl_p99'))} | "
        f"{fmt_triple(isiro_bench.get('itl_p50'), isiro_bench.get('itl_p95'), isiro_bench.get('itl_p99'))} |"
    )
    lines.append(
        f"| TPOT (ms) p50 / p95 / p99 | "
        f"{fmt_triple(base_bench.get('tpot_p50'), base_bench.get('tpot_p95'), base_bench.get('tpot_p99'))} | "
        f"{fmt_triple(isiro_bench.get('tpot_p50'), isiro_bench.get('tpot_p95'), isiro_bench.get('tpot_p99'))} |"
    )
    lines.append(
        f"| TTFT (ms) p50 / p95 / p99 | "
        f"{fmt_triple(base_bench.get('ttft_p50'), base_bench.get('ttft_p95'), base_bench.get('ttft_p99'))} | "
        f"{fmt_triple(isiro_bench.get('ttft_p50'), isiro_bench.get('ttft_p95'), isiro_bench.get('ttft_p99'))} |"
    )
    lines.append("")
    lines.append(f"**Latency vs fresh baseline:** {'PASS' if lat_ok else 'FAIL'} ({'; '.join(lat_reasons)})")
    lines.append("")
    lines.append(
        f"**Latency vs published `rtx-5090-report` baseline:** "
        f"{'PASS' if pub_ok else 'FAIL'} ({'; '.join(pub_reasons)})"
    )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append("| System | `rtx-5090` |")
    lines.append("| CUDA graphs | on |")
    lines.append("| Weights | `fp8` |")
    lines.append("| KV cache | dtype `bfloat16`; `ISIRO_GDX=bf16` on Isiro |")
    lines.append("| Live window | `ISIRO_GDX_LIVE_WINDOW=512` |")
    lines.append("| Cold ready | `ISIRO_GDX_COLD_READY=1` |")
    lines.append("| Fused attend | `0` (attend=`a`) |")
    lines.append(f"| Compiler | `{args.compiler}` |")
    lines.append(f"| Runtime | `{args.runtime}` |")
    lines.append("| vLLM | 0.31.0 |")
    lines.append("| GPU | NVIDIA GeForce RTX 5090 x 1 |")
    lines.append("")
    lines.append(f"Stamp: **{stamp}** (residency_go={res_go}, latency_go={lat_ok}).")
    lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    verdict = {
        "stamp": stamp,
        "residency_go": res_go,
        "latency_go": lat_ok,
        "latency_vs_published": pub_ok,
        "latency_reasons": lat_reasons,
        "published_reasons": pub_reasons,
        "isiro": isiro_bench,
        "baseline": base_bench,
    }
    args.out.with_suffix(".verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
