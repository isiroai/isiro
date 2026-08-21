# ISIRO Benchmark Report: `qwen2.5-7b-instruct`

`bf16` | `rtx-5090` | `v0.1.0` | 2026-08-11 05:12 UTC

Tooling: **`vllm bench serve`**.

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`)
- **TIC:** `isiro serve … --target vllm` (same vLLM version)

## Correctness

| Metric | Baseline | TIC |
|--------|----------|-----|
| On-disk model size | 15.23 GB | **10.86 GB** (**28.68% smaller**) |
| Weight bit-exactness | baseline weights | **PASS** (`isiro verify -r`) |
| Serve output match | - | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## Graph ON (CUDA graphs)

CUDA graphs on (product default; `--graph-on`).

## 1A. GPU memory

Non-KV and KV cache are vLLM-reported. Total GPU memory is nvidia-smi process usage after load.

Norm savings % scales TIC to the Baseline total GPU, then uses `1 - TIC/Baseline` for Loaded model size and Non-KV. For KV it uses `TIC/Baseline - 1` as a % gain (and the matching `x` ratio).

| Metric | Baseline | TIC | Norm savings % |
|--------|----------|-----|----------------|
| Loaded model size | 15.34 GB | **11.38 GB** | **27.73%** |
| Non-KV GPU memory | 16.65 GB | **12.36 GB** | **27.70%** |
| KV cache | 13.65 GB | **17.94 GB** | **28.09% (1.28x)** |
| Total GPU memory | 30.33 GB | 31.13 GB | |

## 1B. Capacity

TIC `max_num_seqs` = round(baseline × TIC/Baseline KV token capacity) from a matched vLLM KV measurement.

KV token capacity is vLLM-reported (`GPU KV cache size` in serve logs).

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` | 32 | **41** (**1.28x**) |
| KV token capacity | 238064 | **312896** (**1.31x**) |

## 1C. Generation (input 32 / output 256)

| Metric | Baseline | TIC |
|--------|----------|-----|
| Output tok/s | 1875.56 | **2876.51** |
| † tok/s per Non-KV GB | 112.62 | 232.75 |
| ITL p50 (ms) | 10.65 | 13.07 |
| ITL p95 (ms) | 11.23 | 13.89 |
| ITL p99 (ms) | 11.74 | 37.67 |
| TPOT p50 (ms) | 10.74 | 13.17 |
| TPOT p95 (ms) | 11.04 | 13.19 |
| TPOT p99 (ms) | 11.10 | 13.64 |

† Derived: output tok/s ÷ Non-KV GPU memory (GB). Physical meaning: output tokens per second per GB of Non-KV GPU memory; a smaller Non-KV slice that still delivers high tok/s scores higher.

## 1D. TTFT

TTFT is separated from generation (1C) because it is more sensitive to the higher concurrency in 1B. With more sequences in flight, new requests wait longer for the first token. That is an expected cost of the capacity setting, not a single-request prefill claim.

| Metric | Baseline | TIC |
|--------|----------|-----|
| TTFT p50 (ms) | 158.65 | 181.53 |
| TTFT p95 (ms) | 160.07 | 218.70 |
| TTFT p99 (ms) | 160.28 | 218.85 |

## 1E. Equal batch

Same `max_num_seqs` and concurrency on both sides.

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` / concurrency | 32 / 32 | 32 / 32 |
| Output tok/s | 1880.11 | 1845.21 |
| ITL p50 (ms) | 10.61 | 11.04 |
| ITL p95 (ms) | 11.26 | 11.72 |
| ITL p99 (ms) | 11.65 | 22.80 |
| TPOT p50 (ms) | 10.73 | 11.24 |
| TPOT p95 (ms) | 10.91 | 11.34 |
| TPOT p99 (ms) | 11.10 | 11.50 |
| TTFT p50 (ms) | 89.35 | 124.34 |
| TTFT p95 (ms) | 155.83 | 186.91 |
| TTFT p99 (ms) | 155.96 | 187.10 |

## Graph OFF (eager)

CUDA graphs off (`--graph-off`).

## 2A. GPU memory

Non-KV and KV cache are vLLM-reported. Total GPU memory is nvidia-smi process usage after load.

Norm savings % scales TIC to the Baseline total GPU, then uses `1 - TIC/Baseline` for Loaded model size and Non-KV. For KV it uses `TIC/Baseline - 1` as a % gain (and the matching `x` ratio).

| Metric | Baseline | TIC | Norm savings % |
|--------|----------|-----|----------------|
| Loaded model size | 15.34 GB | **11.38 GB** | **25.92%** |
| Non-KV GPU memory | 15.84 GB | **12.38 GB** | **21.93%** |
| KV cache | 14.47 GB | **17.93 GB** | **23.73% (1.24x)** |
| Total GPU memory | 30.96 GB | 31.00 GB | |

## 2B. Capacity

TIC `max_num_seqs` = round(baseline × TIC/Baseline KV token capacity) from a matched vLLM KV measurement.

KV token capacity is vLLM-reported (`GPU KV cache size` in serve logs).

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` | 32 | **40** (**1.25x**) |
| KV token capacity | 252352 | **312640** (**1.24x**) |

## 2C. Generation (input 32 / output 256)

| Metric | Baseline | TIC |
|--------|----------|-----|
| Output tok/s | 1761.70 | **2557.72** |
| † tok/s per Non-KV GB | 111.23 | 206.60 |
| ITL p50 (ms) | 11.36 | 14.85 |
| ITL p95 (ms) | 11.97 | 15.80 |
| ITL p99 (ms) | 12.34 | 30.51 |
| TPOT p50 (ms) | 11.51 | 15.03 |
| TPOT p95 (ms) | 11.67 | 15.03 |
| TPOT p99 (ms) | 11.87 | 15.25 |

† Derived: output tok/s ÷ Non-KV GPU memory (GB). Physical meaning: output tokens per second per GB of Non-KV GPU memory; a smaller Non-KV slice that still delivers high tok/s scores higher.

## 2D. TTFT

TTFT is separated from generation (2C) because it is more sensitive to the higher concurrency in 2B. With more sequences in flight, new requests wait longer for the first token. That is an expected cost of the capacity setting, not a single-request prefill claim.

| Metric | Baseline | TIC |
|--------|----------|-----|
| TTFT p50 (ms) | 93.09 | 153.41 |
| TTFT p95 (ms) | 156.63 | 216.48 |
| TTFT p99 (ms) | 156.73 | 217.15 |

## 2E. Equal batch

Same `max_num_seqs` and concurrency on both sides.

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` / concurrency | 32 / 32 | 32 / 32 |
| Output tok/s | 1763.30 | 1399.15 |
| ITL p50 (ms) | 11.34 | 13.84 |
| ITL p95 (ms) | 12.21 | 15.14 |
| ITL p99 (ms) | 12.76 | 39.15 |
| TPOT p50 (ms) | 11.53 | 13.88 |
| TPOT p95 (ms) | 11.83 | 13.95 |
| TPOT p99 (ms) | 11.83 | 14.37 |
| TTFT p50 (ms) | 81.31 | 170.85 |
| TTFT p95 (ms) | 146.20 | 171.54 |
| TTFT p99 (ms) | 146.83 | 171.60 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| Graph modes | ON (CUDA graphs); OFF (eager) |
| vLLM | 0.26.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| Driver / CUDA | 590.44.01 / 13.1 |

Fairness check: **PASS**.
