# ISIRO Benchmark Report: `qwen2.5-7b-instruct`

`bf16` | `rtx-5090` | compiler `v0.1.1` | runtime `v0.1.1` | 2026-11-20 04:41 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype bfloat16`
- **Isiro:** `isiro serve … --target vllm` (TIC weights; same vLLM version)

## Correctness

| Metric | Baseline | Isiro |
|--------|----------|-------|
| On-disk model size | 15.23 GB | **11.46 GB** (**24.77% smaller**) |
| Integrity | baseline weights | **PASS** (`isiro verify`) |
| Serve output match | - | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## A. GPU memory

Savings % uses `1 - Isiro/Baseline` for loaded weights.

### Weights

| Metric | Baseline | Isiro | Savings % |
|--------|----------|-------|-----------|
| GPU memory for loaded weights | 14.29 GiB | **10.82 GiB** | **24.28%** |

### Available KV

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Available KV | 6.39 GiB | **9.13 GiB** |

### KV tokens capacity

| Metric | Baseline | Isiro |
|--------|----------|-------|
| KV tokens capacity | 119,616 | **170,928** |

## C. Equal-N (same concurrency)

Same concurrent-request ceiling both arms at each N (no mismatched Baseline/Isiro N). Section A KV tokens capacity is why higher N is in scope.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Input tokens | 32 | 32 |
| Output max tokens | 256 | 256 |
| Concurrent requests | 1 | 1 |
| Output tok/s | 101.32 | **113.39** (1.12x) |
| ITL (ms) p50 / p95 / p99 | 9.74 / 10.73 / 11.02 | 8.40 / 10.48 / 23.19 |
| TPOT (ms) p50 / p95 / p99 | 9.84 / 9.84 / 9.84 | 8.71 / 8.71 / 8.71 |
| TTFT (ms) p50 / p95 / p99 | 16.10 / 16.10 / 16.10 | 35.40 / 35.40 / 35.40 |

## D. Capacity @1

Fixed concurrency 1 (same ceiling both arms). ITL/TTFT at that conc.

Tok/s is aggregate system throughput across in-flight requests. Per-request decode rate is reflected by ITL/TPOT. Equal-N latency lives in section C when present.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Input tokens | 32 | 32 |
| Output max tokens | 256 | 256 |
| Concurrent requests | 1 | 1 |
| Output tok/s | 101.32 | **113.39** (1.12x) |
| ITL (ms) p50 / p95 / p99 | 9.74 / 10.73 / 11.02 | 8.40 / 10.48 / 23.19 |
| TPOT (ms) p50 / p95 / p99 | 9.84 / 9.84 / 9.84 | 8.71 / 8.71 / 8.71 |
| TTFT (ms) p50 / p95 / p99 | 16.10 / 16.10 / 16.10 | 35.40 / 35.40 / 35.40 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| CUDA graphs | on (eager prefill, graph decode) |
| Weights | `bf16` |
| KV cache | dtype `bfloat16` |
| Compiler | `v0.1.1` |
| Runtime | `v0.1.1` |
| vLLM | 0.31.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| Driver / CUDA | 590.44.01 / 13.1 |
| Isiro attention | `FLASH_ATTN` (`ISIRO_ATTN` 0) |
| Baseline attention | `FLASH_ATTN` |

Fairness check: **PASS**.
