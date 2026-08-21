# ISIRO Benchmark Report: `gemma-4-12B-it`

`bf16` | `rtx-5090` | `v0.1.0` | 2026-08-11 06:21 UTC

Tooling: **`vllm bench serve`**.

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`)
- **TIC:** `isiro serve … --target vllm` (same vLLM version)

## Correctness

| Metric | Baseline | TIC |
|--------|----------|-----|
| On-disk model size | 23.92 GB | **17.05 GB** (**28.70% smaller**) |
| Weight bit-exactness | baseline weights | **PASS** (`isiro verify -r`) |
| Serve output match | - | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## Graph ON (CUDA graphs)

CUDA graphs on (product default; `--graph-on`).

## 1A. GPU memory

Non-KV and KV cache are vLLM-reported. Total GPU memory is nvidia-smi process usage after load.

Norm savings % scales TIC to the Baseline total GPU, then uses `1 - TIC/Baseline` for Loaded model size and Non-KV. For KV it uses `TIC/Baseline - 1` as a % gain (and the matching `x` ratio).

| Metric | Baseline | TIC | Norm savings % |
|--------|----------|-----|----------------|
| Loaded model size | 24.51 GB | **18.45 GB** | **25.77%** |
| Non-KV GPU memory | 25.21 GB | **19.49 GB** | **23.75%** |
| KV cache | 5.10 GB | **10.81 GB** | **109.13% (2.09x)** |
| Total GPU memory | 31.04 GB | 31.47 GB | |

## 1B. Capacity

TIC `max_num_seqs` = round(baseline × TIC/Baseline KV token capacity) from a matched vLLM KV measurement.

KV token capacity is vLLM-reported (`GPU KV cache size` in serve logs).

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` | 32 | **45** (**1.41x**) |
| KV token capacity | 14756 | **31304** (**2.12x**) |

## 1C. Generation (input 32 / output 256)

| Metric | Baseline | TIC |
|--------|----------|-----|
| Output tok/s | 1040.55 | **1434.19** |
| † tok/s per Non-KV GB | 41.27 | 73.59 |
| ITL p50 (ms) | 18.96 | 26.91 |
| ITL p95 (ms) | 20.29 | 28.26 |
| ITL p99 (ms) | 20.76 | 28.71 |
| TPOT p50 (ms) | 19.22 | 26.84 |
| TPOT p95 (ms) | 19.22 | 26.84 |
| TPOT p99 (ms) | 19.23 | 26.90 |

† Derived: output tok/s ÷ Non-KV GPU memory (GB). Physical meaning: output tokens per second per GB of Non-KV GPU memory; a smaller Non-KV slice that still delivers high tok/s scores higher.

## 1D. TTFT

TTFT is separated from generation (1C) because it is more sensitive to the higher concurrency in 1B. With more sequences in flight, new requests wait longer for the first token. That is an expected cost of the capacity setting, not a single-request prefill claim.

| Metric | Baseline | TIC |
|--------|----------|-----|
| TTFT p50 (ms) | 207.31 | 292.51 |
| TTFT p95 (ms) | 208.68 | 294.04 |
| TTFT p99 (ms) | 218.12 | 294.16 |

## 1E. Equal batch

Same `max_num_seqs` and concurrency on both sides.

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` / concurrency | 32 / 32 | 32 / 32 |
| Output tok/s | 1030.03 | 889.39 |
| ITL p50 (ms) | 19.08 | 23.28 |
| ITL p95 (ms) | 20.34 | 24.80 |
| ITL p99 (ms) | 20.74 | 25.23 |
| TPOT p50 (ms) | 19.18 | 23.49 |
| TPOT p95 (ms) | 19.25 | 23.50 |
| TPOT p99 (ms) | 19.25 | 23.54 |
| TTFT p50 (ms) | 207.54 | 259.81 |
| TTFT p95 (ms) | 209.03 | 260.46 |
| TTFT p99 (ms) | 209.15 | 260.53 |

## Graph OFF (eager)

CUDA graphs off (`--graph-off`).

## 2A. GPU memory

Non-KV and KV cache are vLLM-reported. Total GPU memory is nvidia-smi process usage after load.

Norm savings % scales TIC to the Baseline total GPU, then uses `1 - TIC/Baseline` for Loaded model size and Non-KV. For KV it uses `TIC/Baseline - 1` as a % gain (and the matching `x` ratio).

| Metric | Baseline | TIC | Norm savings % |
|--------|----------|-----|----------------|
| Loaded model size | 24.51 GB | **18.45 GB** | **25.16%** |
| Non-KV GPU memory | 25.02 GB | **19.49 GB** | **22.53%** |
| KV cache | 5.28 GB | **10.81 GB** | **103.55% (2.04x)** |
| Total GPU memory | 31.05 GB | 31.22 GB | |

## 2B. Capacity

TIC `max_num_seqs` = round(baseline × TIC/Baseline KV token capacity) from a matched vLLM KV measurement.

KV token capacity is vLLM-reported (`GPU KV cache size` in serve logs).

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` | 32 | **45** (**1.41x**) |
| KV token capacity | 15296 | **31316** (**2.05x**) |

## 2C. Generation (input 32 / output 256)

| Metric | Baseline | TIC |
|--------|----------|-----|
| Output tok/s | 921.94 | **1053.05** |
| † tok/s per Non-KV GB | 36.85 | 54.03 |
| ITL p50 (ms) | 21.42 | 33.28 |
| ITL p95 (ms) | 22.57 | 54.99 |
| ITL p99 (ms) | 23.00 | 59.36 |
| TPOT p50 (ms) | 21.61 | 36.93 |
| TPOT p95 (ms) | 21.61 | 36.93 |
| TPOT p99 (ms) | 21.68 | 37.01 |

† Derived: output tok/s ÷ Non-KV GPU memory (GB). Physical meaning: output tokens per second per GB of Non-KV GPU memory; a smaller Non-KV slice that still delivers high tok/s scores higher.

## 2D. TTFT

TTFT is separated from generation (2C) because it is more sensitive to the higher concurrency in 2B. With more sequences in flight, new requests wait longer for the first token. That is an expected cost of the capacity setting, not a single-request prefill claim.

| Metric | Baseline | TIC |
|--------|----------|-----|
| TTFT p50 (ms) | 201.71 | 304.07 |
| TTFT p95 (ms) | 221.44 | 304.62 |
| TTFT p99 (ms) | 221.50 | 304.70 |

## 2E. Equal batch

Same `max_num_seqs` and concurrency on both sides.

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` / concurrency | 32 / 32 | 32 / 32 |
| Output tok/s | 924.07 | 533.58 |
| ITL p50 (ms) | 21.43 | 32.79 |
| ITL p95 (ms) | 22.56 | 52.29 |
| ITL p99 (ms) | 23.60 | 55.59 |
| TPOT p50 (ms) | 21.56 | 39.77 |
| TPOT p95 (ms) | 21.57 | 39.77 |
| TPOT p99 (ms) | 21.64 | 39.84 |
| TTFT p50 (ms) | 210.53 | 276.30 |
| TTFT p95 (ms) | 211.99 | 277.01 |
| TTFT p99 (ms) | 212.25 | 277.11 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| Graph modes | ON (CUDA graphs); OFF (eager) |
| vLLM | 0.26.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| Driver / CUDA | 590.44.01 / 13.1 |

Fairness check: **PASS**.
