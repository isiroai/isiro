# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `v0.1.1` | runtime `v0.1.1` | 2026-11-05 04:44 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype bfloat16`
- **Isiro:** `isiro serve … --target vllm` with TIC weights and `ISIRO_GDX=bf16` (same vLLM version; cold KV compress on vendor attention pages)

Section A Available KV is the vendor reserved pool (GiB). KV tokens capacity applies GDX cold packing on that budget. Measured cold vs FA twin bytes (when the curve ran) are under KV vs context.

## Correctness

| Metric | Baseline | Isiro |
|--------|----------|-------|
| On-disk model size | 21.85 GB | **18.50 GB** (**15.35% smaller**) |
| Integrity | baseline weights | **PASS** (`isiro verify`) |
| KV bit-exactness | vendor KV pages | **PASS** (GDX `bf16` cold roundtrip) |
| Serve output match | - | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## A. GPU memory

Savings % uses `1 - Isiro/Baseline` for loaded weights.

### Weights

| Metric | Baseline | Isiro | Savings % |
|--------|----------|-------|-----------|
| GPU memory for loaded weights | 19.92 GiB | **15.83 GiB** | **20.53%** |

### Available KV

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Available KV | 4.47 GiB | **4.86 GiB** |

Available KV is the vendor reserved pool on the GPU (same util target as Baseline).

### KV tokens capacity

Weight compression grows Available KV. GDX then stores more tokens in that memory (hot window stays dense; cold prefix is packed). With measured s=0.33 → scale 1/(1-s) ≈ 1.49 (hot-corrected).

| Metric | Baseline | Isiro |
|--------|----------|-------|
| KV tokens capacity | 54,418 | **88,036** (**+62%**) |

### KV vs context

`ISIRO_GDX=bf16` on KV from a Project Gutenberg text forward. At T=262,144: Baseline KV 16.00 GiB, Isiro KV 10.71 GiB; save 5.29 GiB (33.1%). Bottom panel adds loaded weights (Baseline 19.92 GiB, Isiro 15.83 GiB): at T=262,144, Baseline 35.92 GiB vs Isiro 26.54 GiB (weights + KV).

Figure. Weights and KV vs 1-user context.

![Weights and KV vs 1-user context](kv-bytes-vs-context.png)

## C. Equal batch

Same concurrent-request ceiling as Baseline (latency-matched surface). Capacity-matched generation in D raises concurrency for aggregate tok/s; ITL/TTFT there are not the equal-batch claim.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 8 | 8 |
| Output tok/s | 522.69 | 442.27 |
| ITL (ms) p50 / p95 / p99 | 14.74 / 16.09 / 16.44 | 17.48 / 18.61 / 34.89 |
| TPOT (ms) p50 / p95 / p99 | 14.80 / 14.89 / 14.93 | 17.44 / 17.46 / 17.47 |
| TTFT (ms) p50 / p95 / p99 | 141.79 / 142.11 / 142.12 | 181.45 / 181.89 / 181.89 |

## D. Capacity

Concurrent requests are 8 on both sides. Not scaled from KV token capacity.

Concurrent requests are the serve max concurrent sequences (`max_num_seqs`); the bench load matches that ceiling in this A/B.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 8 | 8 |

### Generation (recall-matched KV, input 32 / output 256)

Tok/s is aggregate system throughput across in-flight requests. Per-request decode rate is reflected by ITL/TPOT. Concurrent requests are in Capacity. Not equal batch (C). Latency-matched wins live in Equal batch (same conc as Baseline). Higher Capacity concurrency raises aggregate tok/s and can move TTFT/ITL.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 522.69 | 442.27 |
| ITL (ms) p50 / p95 / p99 | 14.74 / 16.09 / 16.44 | 17.48 / 18.61 / 34.89 |
| TPOT (ms) p50 / p95 / p99 | 14.80 / 14.89 / 14.93 | 17.44 / 17.46 / 17.47 |
| TTFT (ms) p50 / p95 / p99 | 141.79 / 142.11 / 142.12 | 181.45 / 181.89 / 181.89 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| CUDA graphs | on (eager prefill, graph decode) |
| Weights | `awq` |
| KV cache | dtype `bfloat16`; `ISIRO_GDX=bf16` on Isiro |
| Compiler | `v0.1.1` |
| Runtime | `v0.1.1` |
| vLLM | 0.31.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| Driver / CUDA | 590.44.01 / 13.1 |
| ISIRO_GDX | `bf16` |
| Isiro attention | `FLASH_ATTN` (`ISIRO_ATTN` unset) |
| Baseline attention | `FLASH_ATTN` |

Fairness check: **PASS**.
