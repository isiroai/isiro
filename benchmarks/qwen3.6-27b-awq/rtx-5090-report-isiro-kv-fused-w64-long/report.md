# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `kv-only` | runtime `kv-only` | 2026-11-11 19:48 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype fp8`
- **Isiro KV-only:** `vllm serve` + `ISIRO_KV=fp8` + fused dens attend/decode (same AWQ/HF weights as baseline; no TIC weight claim)

Section A: loaded weights and Available KV GiB are shared (same raw weights, same vendor util). KV tokens capacity follows the measured Isiro/Baseline token ratio on that budget. Section C concurrency uses the same N both arms at each load point.

Cold attend (a+) on the W=64 long-prefix geometry (`ISIRO_KV_FUSED_ATTEND=1`, `ISIRO_KV_FUSED_DECODE=1`, `LIVE_WINDOW=64`, FA clamp, input=2048, N=6). Stamp `20261111T194214Z`. Same shape as path-a-8k-w64 so the cold region arms; isolates densify-for-vendor-FA vs fused dens attend.

TIME claim: **NO GO** on concurrency (N=6; tok/s 166.24 vs 224.12; ITL 22.41 vs 14.63; TTFT 2,307.40 vs 2,563.38). Publish refused equal_batch ITL p50 collapse (tic > 1.5× baseline). Worse than path-a densify on this shape (path-a: tok/s 211.8 / ITL 16.20). Do not product-on FUSED; do not flip LIVE_WINDOW default from 512.

## Correctness

| Metric | Result |
|--------|--------|
| KV bit-exactness | **PASS** (Isiro KV `fp8` roundtrip) |
| Serve output match | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## C. Concurrency

Same concurrent-request ceiling both arms at each N (no mismatched Baseline/Isiro N).

### Concurrent requests = 6

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 6 | 6 |
| Output tok/s | 224.12 | 166.24 |
| ITL (ms) p50 / p95 / p99 | 14.63 / 15.03 / 16.05 | 22.41 / 25.53 / 272.02 |
| TPOT (ms) p50 / p95 / p99 | 16.68 / 21.15 / 21.56 | 26.87 / 30.77 / 31.00 |
| TTFT (ms) p50 / p95 / p99 | 2,563.38 / 3,118.98 / 3,122.68 | 2,307.40 / 3,372.27 / 3,433.63 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| CUDA graphs | on (eager prefill, graph decode) |
| Weights | `awq` |
| KV cache | dtype `fp8`; `ISIRO_KV=fp8` on Isiro |
| Compiler | `kv-only` |
| Runtime | `kv-only` |
| vLLM | 0.31.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| ISIRO_KV | `fp8` |
| Isiro attention | fused dens (`ISIRO_KV_FUSED_ATTEND=1`, `ISIRO_KV_FUSED_DECODE=1`) |
| Baseline attention | `FLASH_ATTN` |
| LIVE_WINDOW | 64 |
| PAGE_BLOCK | 16 |

Fairness check: **PASS**.
