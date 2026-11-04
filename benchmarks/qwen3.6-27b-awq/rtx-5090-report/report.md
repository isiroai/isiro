# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `auto` | runtime `auto` | 2026-11-04 20:59 UTC

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

Weight compression grows Available KV. GDX then stores more tokens in that memory (hot window stays dense; cold prefix is packed). With measured s=0.59 → scale 1/(1-s) ≈ 2.42 (hot-corrected).

| Metric | Baseline | Isiro |
|--------|----------|-------|
| KV tokens capacity | 54,418 | **142,474** (**+162%**) |

### KV vs context

`ISIRO_GDX=bf16` is on for serve. kv-bytes-vs-context.png omitted: curve did not run (unset ISIRO_BENCH_SKIP_GDX_CURVE; for AWQ/quant baselines ensure gptqmodel in the GDX meter venv, or set ISIRO_BENCH_GDX_MODEL_DIR to a fitting HF twin).

## C. Equal batch

Same concurrent-request ceiling as Baseline (latency-matched surface). Capacity-matched generation in D raises concurrency for aggregate tok/s; ITL/TTFT there are not the equal-batch claim.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 8 | 8 |
| Output tok/s | 523.32 | 442.03 |
| ITL (ms) p50 / p95 / p99 | 14.79 / 15.43 / 15.85 | 17.54 / 18.42 / 35.15 |
| TPOT (ms) p50 / p95 / p99 | 14.80 / 14.89 / 14.92 | 17.50 / 17.53 / 17.55 |
| TTFT (ms) p50 / p95 / p99 | 138.87 / 139.09 / 139.09 | 169.08 / 169.52 / 169.55 |

## D. Capacity

Isiro concurrent requests = round(Baseline × Isiro/Baseline KV token capacity) from a matched vLLM KV measurement, then rounded up onto the vLLM CUDA-graph capture stride (multiples of 8) so FULL decode graphs cover concurrency.

Concurrent requests are the serve max concurrent sequences (`max_num_seqs`); the bench load matches that ceiling in this A/B.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 8 | **9** (**+12%**) |

### Generation (recall-matched KV, input 32 / output 256)

Tok/s is aggregate system throughput across in-flight requests. Per-request decode rate is reflected by ITL/TPOT. Concurrent requests are in Capacity. Not equal batch (C). Latency-matched wins live in Equal batch (same conc as Baseline). Higher Capacity concurrency raises aggregate tok/s and can move TTFT/ITL.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 523.32 | **541.85** |
| ITL (ms) p50 / p95 / p99 | 14.79 / 15.43 / 15.85 | 15.92 / 16.84 / 31.97 |
| TPOT (ms) p50 / p95 / p99 | 14.80 / 14.89 / 14.92 | 15.94 / 15.98 / 16.00 |
| TTFT (ms) p50 / p95 / p99 | 138.87 / 139.09 / 139.09 | 187.46 / 187.76 / 187.77 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| CUDA graphs | on (eager prefill, graph decode) |
| Weights | `awq` |
| KV cache | dtype `bfloat16`; `ISIRO_GDX=bf16` on Isiro |
| Compiler | `auto` |
| Runtime | `auto` |
| vLLM | 0.31.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| Driver / CUDA | 590.44.01 / 13.1 |
| ISIRO_GDX | `bf16` |
| Isiro attention | `FLASH_ATTN` (`ISIRO_ATTN` unset) |
| Baseline attention | `FLASH_ATTN` |

Fairness check: **PASS**.
