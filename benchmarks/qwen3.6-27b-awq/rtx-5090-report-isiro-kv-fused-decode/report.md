# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `kv-only` | runtime `kv-only` | 2026-11-06 15:20 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype fp8`
- **Isiro KV-only:** `vllm serve` + `ISIRO_KV=fp8` + fused dens (`FUSED_ATTEND=1` `FUSED_DECODE=1`; same AWQ/HF weights; no TIC weight claim)

Section A: loaded weights and Available KV GiB are shared (same raw weights, same vendor util). This dens scrape used W=16 so short equalize deposits cold; FA EngineArgs clamp lands in a follow-on remeter.

TIME claim: **NO GO** on equal-batch (conc=1; tok/s 69.24 vs 72.83; ITL 13.42 vs 13.57; TTFT 95.97 vs 41.15). Side-stream codes stamp improved ITL to beat baseline; TTFT still ~2.3×. Stamp stays `attend=(a)`.

## Correctness

| Metric | Result |
|--------|--------|
| KV bit-exactness | **PASS** (Isiro KV `fp8` roundtrip) |
| Serve output match | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## A. GPU memory

Loaded weights are identical on both arms (no TIC weight claim).

### Weights

| Metric | Value |
|--------|-------|
| GPU memory for loaded weights | 19.92 GiB |

### Available KV

| Metric | Value |
|--------|-------|
| Available KV | 4.09 GiB |

Available KV is the vendor reserved pool on the GPU (same util target both arms; no TIC weight headroom).

### KV vs context

`ISIRO_KV=fp8` is on for serve. kv-bytes-vs-context.png omitted: curve did not run.

## C. Equal batch

Same concurrent-request ceiling as Baseline (latency-matched surface). Dens scrape forces cold deposit (W=16, in=64).

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 1 | 1 |
| Output tok/s | 72.83 | 69.24 |
| ITL (ms) p50 / p95 / p99 | 13.57 / 14.39 / 14.56 | 13.42 / 16.80 / 27.20 |
| TPOT (ms) p50 / p95 / p99 | 13.62 / 13.62 / 13.62 | 14.12 / 14.12 / 14.12 |
| TTFT (ms) p50 / p95 / p99 | 41.15 / 41.15 / 41.15 | 95.97 / 95.97 / 95.97 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| CUDA graphs | on (PIECEWISE dens) |
| Weights | `awq` |
| KV cache | dtype `fp8`; `ISIRO_KV=fp8` on Isiro |
| Compiler | `kv-only` |
| Runtime | `kv-only` |
| vLLM | 0.31.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| ISIRO_KV | `fp8` |
| Isiro attention | fused dens a+ (lab; product TIME stays path a) |
| Baseline attention | `FLASH_ATTN` |
| Remeter | `20261106T151543Z` |

Fairness check: **PASS**.
