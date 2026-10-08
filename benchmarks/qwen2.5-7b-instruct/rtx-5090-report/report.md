# ISIRO Benchmark Report: `qwen2.5-7b-instruct`

`bf16` | `rtx-5090` | compiler `v0.1.1` | runtime `v0.1.1` | 2026-10-08 23:29 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype bfloat16`
- **Isiro:** `isiro serve … --target vllm` with TIC weights and `ISIRO_GDX=bf16` (same vLLM version; cold KV compress on vendor attention pages)

GDX cold coding-rate vs naive vendor KV bits (estimate): bf16 ~29-31% vs 16 bpv; fp8 ~19-25% vs 8 bpv. Section A GPU memory is the reserved vendor KV pool, not coded cold bytes.

## Correctness

| Metric | Baseline | Isiro |
|--------|----------|-------|
| On-disk model size | 15.23 GB | **10.78 GB** (**29.22% smaller**) |
| Weight bit-exactness | baseline weights | **PASS** (`isiro verify -r`) |
| KV bit-exactness | vendor KV pages | **PASS** (GDX `bf16` cold roundtrip) |
| Serve output match | - | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## A. GPU memory

### Weights

| Metric | Baseline | Isiro | Savings % |
|--------|----------|-------|-----------|
| GPU memory for loaded weights | 14.29 GiB | **10.25 GiB** | **28.27%** |

### KV cache

| Metric | Baseline default fill | Isiro reserved KV | Savings % |
|--------|-----------------------|-------------------|-----------|
| GPU memory reserved for KV | 12.56 GiB (235120 tokens) | 15.85 GiB (296800 tokens)† | -26.19% |

† Reserved pool on the GPU. Longer context uses persist after vendor reclaim of prefix blocks (TokenIDs plus archived K,V). Not a second vendor KV cache on the host.

### Cold KV bytes vs context

Hot window W=512 stays vendor FA. Cold prefix `[0, T-W)` is GDX o1 (`ISIRO_GDX=bf16`). Curve is packed cold + hot FA vs naive FA for the same T. Save at max T: 93.3%.

Figure. KV bytes vs context tokens (GDX cold compress).

![KV bytes vs context tokens](kv-bytes-vs-context.png)

## C. Capacity

Isiro batch = round(Baseline batch × Isiro/Baseline KV token capacity) from a matched vLLM KV measurement.

KV token capacity is reserved pool slots when Isiro printed KV pool, else vLLM GPU tokens in this serve's reserved GPU pool.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Batch | 32 | **40** (**1.25x**) |
| KV token capacity | 235120 (12.56 GiB KV) | **296800 (15.85 GiB KV)†** |

## D. Generation (recall-matched KV, input 32 / output 256)

Tok/s and latency at the recall-matched KV sizes. Batch is in Capacity. Not equal batch (E). TTFT is in this table because it moves with concurrency.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 1870.46 | **2499.52** |
| ITL (ms) p50 / p95 / p99 | 10.53 / 11.70 / 12.08 | 13.97 / 16.09 / 29.06 |
| TPOT (ms) p50 / p95 / p99 | 10.82 / 10.90 / 11.05 | 14.02 / 14.27 / 15.09 |
| TTFT (ms) p50 / p95 / p99 | 102.45 / 160.52 / 160.53 | 517.67 / 517.97 / 518.01 |

## E. Equal batch

Same batch as Capacity.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 1870.46 | 1780.33 |
| ITL (ms) p50 / p95 / p99 | 10.53 / 11.70 / 12.08 | 10.71 / 12.24 / 21.97 |
| TPOT (ms) p50 / p95 / p99 | 10.82 / 10.90 / 11.05 | 10.91 / 10.91 / 11.05 |
| TTFT (ms) p50 / p95 / p99 | 102.45 / 160.52 / 160.53 | 418.83 / 420.10 / 420.11 |

## Config

| Item | Value |
|------|-------|
| System | `rtx-5090` |
| CUDA graphs | on (eager prefill, graph decode) |
| KV cache | dtype `bfloat16`; `ISIRO_GDX=bf16` on Isiro |
| Compiler | `v0.1.1` |
| Runtime | `v0.1.1` |
| vLLM | 0.31.0 |
| GPU | NVIDIA GeForce RTX 5090 x 1 |
| Driver / CUDA | 590.44.01 / 13.1 |
| ISIRO_GDX | `bf16` |
| Isiro attention | `CUSTOM` (`ISIRO_ATTN` unset) |
| Baseline attention | `FLASH_ATTN` |

Fairness check: **PASS**.
