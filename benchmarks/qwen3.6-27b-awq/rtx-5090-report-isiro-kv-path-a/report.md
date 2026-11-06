# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `kv-only` | runtime `kv-only` | 2026-11-05 23:16 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype fp8`
- **Isiro KV-only:** `vllm serve` + `ISIRO_KV=fp8` (same AWQ/HF weights as baseline; no TIC weight claim; cold KV compress on vendor attention pages)

Section A Available KV is the vendor reserved pool (GiB). KV tokens capacity applies Isiro KV cold packing on that budget. Measured cold vs FA twin bytes (when the curve ran) are under KV vs context.

TIME claim: **GO** on path **(a)** remeter `20261105T230832Z` (FUSED=0, fp8 KV,
`LIVE_WINDOW=512`). Equal-batch TTFT/ITL/tok match or beat baseline (TTFT 41.61
vs 43.02; ITL 13.17 vs 13.21; tok/s 75.27 vs 75.08). Match PASS; KV bitexact
PASS. Dens invent stays off this stamp (`attend=(a)`).

## Correctness

| Metric | Baseline | Isiro |
|--------|----------|-------|
| On-disk model size | 21.85 GB | **21.85 GB** (**0.00% smaller**) |
| Integrity | baseline weights | SKIP (KV-only; same AWQ/HF weights) |
| KV bit-exactness | vendor KV pages | **PASS** (Isiro KV `fp8` cold roundtrip) |
| Serve output match | - | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## A. GPU memory

Savings % uses `1 - Isiro/Baseline` for loaded weights.

### Weights

| Metric | Baseline | Isiro | Savings % |
|--------|----------|-------|-----------|
| GPU memory for loaded weights | 19.92 GiB | 19.92 GiB | 0.00% |

### Available KV

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Available KV | 4.11 GiB | 4.11 GiB |

Available KV is the vendor reserved pool on the GPU (same util target as Baseline).

### KV tokens capacity

Weight compression grows Available KV. Isiro KV then stores more tokens in that memory (hot window stays dense; cold prefix is packed). With measured s=0.50 → scale 1/(1-s) ≈ 2.01 (hot-corrected).

| Metric | Baseline | Isiro |
|--------|----------|-------|
| KV tokens capacity | 77,368 | **154,635** (**+100%**) |

### KV vs context

`ISIRO_KV=fp8` is on for serve. kv-bytes-vs-context.png omitted: curve did not run (unset ISIRO_BENCH_SKIP_ISIRO_KV_CURVE; for AWQ/quant baselines ensure gptqmodel in the Isiro KV meter venv, or set ISIRO_BENCH_KV_MODEL_DIR to a fitting HF twin).

## C. Equal batch

Same concurrent-request ceiling as Baseline (latency-matched surface). Capacity-matched generation in D raises concurrency for aggregate tok/s; ITL/TTFT there are not the equal-batch claim.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 1 | 1 |
| Output tok/s | 75.08 | 75.27 |
| ITL (ms) p50 / p95 / p99 | 13.21 / 13.50 / 13.69 | 13.17 / 13.49 / 13.63 |
| TPOT (ms) p50 / p95 / p99 | 13.20 / 13.20 / 13.20 | 13.17 / 13.17 / 13.17 |
| TTFT (ms) p50 / p95 / p99 | 43.02 / 43.02 / 43.02 | 41.61 / 41.61 / 41.61 |

## D. Capacity

Concurrent requests are 1 on both sides. Not scaled from KV token capacity.

Concurrent requests are the serve max concurrent sequences (`max_num_seqs`); the bench load matches that ceiling in this A/B.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 1 | 1 |

### Generation (recall-matched KV, input 32 / output 256)

Tok/s is aggregate system throughput across in-flight requests. Per-request decode rate is reflected by ITL/TPOT. Concurrent requests are in Capacity. Not equal batch (C). Latency-matched wins live in Equal batch (same conc as Baseline). Higher Capacity concurrency raises aggregate tok/s and can move TTFT/ITL.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 75.08 | **75.27** |
| ITL (ms) p50 / p95 / p99 | 13.21 / 13.50 / 13.69 | 13.17 / 13.49 / 13.63 |
| TPOT (ms) p50 / p95 / p99 | 13.20 / 13.20 / 13.20 | 13.17 / 13.17 / 13.17 |
| TTFT (ms) p50 / p95 / p99 | 43.02 / 43.02 / 43.02 | 41.61 / 41.61 / 41.61 |

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
