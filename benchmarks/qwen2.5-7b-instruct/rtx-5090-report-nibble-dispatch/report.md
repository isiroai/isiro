# ISIRO Benchmark Report: `qwen2.5-7b-instruct`

`bf16` | `rtx-5090` | compiler `v0.1.1` | runtime `v0.1.1` | 2026-11-19 18:39 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`)
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
| GPU memory for loaded weights | 14.29 GiB | **10.94 GiB** | **23.44%** |

### Available KV

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Available KV | 12.56 GiB | **14.92 GiB** |

### KV tokens capacity

| Metric | Baseline | Isiro |
|--------|----------|-------|
| KV tokens capacity | 235,120 | **279,408** |

## C. Capacity @32

Fixed concurrency 32 (same ceiling both arms). ITL/TTFT at that conc.

Tok/s is aggregate system throughput across in-flight requests. Per-request decode rate is reflected by ITL/TPOT. Equal-N latency lives in section C when present.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Input tokens | 32 | 32 |
| Output max tokens | 256 | 256 |
| Concurrent requests | 32 | 32 |
| Output tok/s | 2,836.14 | 2,231.72 (0.79x) |
| ITL (ms) p50 / p95 / p99 | 10.75 / 11.28 / 11.51 | 13.61 / 14.18 / 27.60 |
| TPOT (ms) p50 / p95 / p99 | 10.72 / 11.04 / 11.08 | 13.56 / 13.56 / 13.59 |
| TTFT (ms) p50 / p95 / p99 | 151.21 / 152.06 / 152.07 | 210.39 / 211.06 / 211.28 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| CUDA graphs | on (eager prefill, graph decode) |
| Weights | `bf16` |
| KV cache | product KV cache size |
| Compiler | `v0.1.1` |
| Runtime | `v0.1.1` |
| vLLM | 0.31.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| Driver / CUDA | 590.44.01 / 13.1 |
| Isiro attention | `CUSTOM` (`ISIRO_ATTN` unset) |
| Baseline attention | `FLASH_ATTN` |

Fairness check: **PASS**.
