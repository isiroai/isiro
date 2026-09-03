# ISIRO Benchmark Report: `qwen3.8-27b`

`bf16` | `rtx-pro-6000-blackwell` | `v0.1.0` | 2026-09-03 20:22 UTC

Tooling: **`vllm bench serve`**.

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`)
- **TIC:** `isiro serve … --target vllm` (same vLLM version)

## Correctness

| Metric | Baseline | TIC |
|--------|----------|-----|
| On-disk model size | 55.56 GB | **39.57 GB** (**28.78% smaller**) |
| Integrity | baseline weights | **PASS** (`isiro verify`) |
| Serve output match | - | FAIL (3/4 prompts matched (temp=0 token IDs)) |

## Graph ON (CUDA graphs)

CUDA graphs on (product default; `--graph-on`).

## 1A. GPU memory

Non-KV and KV cache are vLLM-reported. Total GPU memory is nvidia-smi process usage after load.

Norm savings % scales TIC to the Baseline total GPU, then uses `1 - TIC/Baseline` for Loaded model size and Non-KV. For KV it uses `TIC/Baseline - 1` as a % gain (and the matching `x` ratio).

| Metric | Baseline | TIC | Norm savings % |
|--------|----------|-----|----------------|
| Loaded model size | 54.87 GB | **40.61 GB** | **25.84%** |
| Non-KV GPU memory | 57.14 GB | **43.79 GB** | **23.22%** |
| KV cache | 34.23 GB | **47.65 GB** | **39.49% (1.39x)** |
| Total GPU memory | 91.43 GB | 91.25 GB | |

## 1B. Capacity

TIC `max_num_seqs` = round(baseline × TIC/Baseline KV token capacity) from a matched vLLM KV measurement.

KV token capacity is vLLM-reported (`GPU KV cache size` in serve logs).

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` | 64 | **89** (**1.39x**) |
| KV token capacity | 303104 | **421888** (**1.39x**) |

## 1C. Generation (input 32 / output 256)

| Metric | Baseline | TIC |
|--------|----------|-----|
| Output tok/s | 707.11 | **560.70** |
| † tok/s per Non-KV GB | 12.37 | 12.81 |
| ITL p50 (ms) | 53.67 | 67.40 |
| ITL p95 (ms) | 57.61 | 69.89 |
| ITL p99 (ms) | 60.41 | 74.24 |
| TPOT p50 (ms) | 53.65 | 67.17 |
| TPOT p95 (ms) | 53.66 | 67.18 |
| TPOT p99 (ms) | 53.83 | 67.38 |

† Derived: output tok/s ÷ Non-KV GPU memory (GB). Physical meaning: output tokens per second per GB of Non-KV GPU memory; a smaller Non-KV slice that still delivers high tok/s scores higher.

## 1D. TTFT

TTFT is separated from generation (1C) because it is more sensitive to the higher concurrency in 1B. With more sequences in flight, new requests wait longer for the first token. That is an expected cost of the capacity setting, not a single-request prefill claim.

| Metric | Baseline | TIC |
|--------|----------|-----|
| TTFT p50 (ms) | 773.34 | 1111.88 |
| TTFT p95 (ms) | 795.45 | 1126.26 |
| TTFT p99 (ms) | 797.76 | 1127.79 |

## 1E. Equal batch

Same `max_num_seqs` and concurrency on both sides.

| Metric | Baseline | TIC |
|--------|----------|-----|
| `max_num_seqs` / concurrency | 64 / 64 | 64 / 64 |
| Output tok/s | 707.11 | 552.71 |
| ITL p50 (ms) | 53.67 | 67.71 |
| ITL p95 (ms) | 57.61 | 70.72 |
| ITL p99 (ms) | 60.41 | 76.01 |
| TPOT p50 (ms) | 53.65 | 67.75 |
| TPOT p95 (ms) | 53.66 | 67.77 |
| TPOT p99 (ms) | 53.83 | 68.05 |
| TTFT p50 (ms) | 773.34 | 1223.81 |
| TTFT p95 (ms) | 795.45 | 1240.58 |
| TTFT p99 (ms) | 797.76 | 1243.46 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-pro-6000-blackwell` |
| Graph modes | ON (CUDA graphs) |
| vLLM | 0.26.0 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition x 1 |
| Driver / CUDA | 595.71.05 / 13.2 |

Fairness check: **PASS**.
