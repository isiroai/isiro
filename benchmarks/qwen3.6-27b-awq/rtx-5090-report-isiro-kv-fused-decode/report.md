# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `kv-only` | runtime `kv-only` | 2026-11-06 13:03 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype fp8`
- **Isiro KV-only:** `vllm serve` + `ISIRO_KV=fp8` (same AWQ/HF weights as baseline; no TIC weight claim)

Section A: loaded weights and Available KV GiB are shared (same raw weights, same vendor util). KV tokens capacity and Capacity concurrency follow the measured Isiro/Baseline token ratio on that budget. KV bytes vs context are under KV vs context when the curve ran.

TIME claim: **NO GO** on equal-batch (conc=1; tok/s 68.54 vs 72.92; ITL 14.20 vs 13.53; TTFT 90.73 vs 35.31).

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
| Available KV | 4.11 GiB |

Available KV is the vendor reserved pool on the GPU (same util target both arms; no TIC weight headroom).

### KV vs context

`ISIRO_KV=fp8` is on for serve. kv-bytes-vs-context.png omitted: curve did not run.

## C. Equal batch

Same concurrent-request ceiling as Baseline (latency-matched surface). Capacity-matched generation in D raises concurrency for aggregate tok/s; ITL/TTFT there are not the equal-batch claim.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 1 | 1 |
| Output tok/s | 72.92 | 68.54 |
| ITL (ms) p50 / p95 / p99 | 13.53 / 14.34 / 14.50 | 14.20 / 16.13 / 16.52 |
| TPOT (ms) p50 / p95 / p99 | 13.63 / 13.63 / 13.63 | 14.29 / 14.29 / 14.29 |
| TTFT (ms) p50 / p95 / p99 | 35.31 / 35.31 / 35.31 | 90.73 / 90.73 / 90.73 |

## D. Generation (recall-matched KV, input 32 / output 256)

Tok/s is aggregate system throughput across in-flight requests. Per-request decode rate is reflected by ITL/TPOT. Concurrent requests are in Capacity. Not equal batch (C). Latency-matched wins live in Equal batch (same conc as Baseline). Higher Capacity concurrency raises aggregate tok/s and can move TTFT/ITL.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 72.92 | 68.54 |
| ITL (ms) p50 / p95 / p99 | 13.53 / 14.34 / 14.50 | 14.20 / 16.13 / 16.52 |
| TPOT (ms) p50 / p95 / p99 | 13.63 / 13.63 / 13.63 | 14.29 / 14.29 / 14.29 |
| TTFT (ms) p50 / p95 / p99 | 35.31 / 35.31 / 35.31 | 90.73 / 90.73 / 90.73 |

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
| Driver / CUDA | 590.44.01 / 13.1 |
| ISIRO_KV | `fp8` |
| Isiro attention | `FLASH_ATTN` (`ISIRO_ATTN` unset) |
| Baseline attention | `FLASH_ATTN` |

Fairness check: **PASS**.
