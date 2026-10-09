# ISIRO Benchmark Report: `qwen2.5-7b-instruct`

`bf16` | `rtx-5090` | compiler `v0.1.1` | runtime `v0.1.1` | 2026-10-09 15:23 UTC

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
| Integrity | baseline weights | **PASS** (`isiro verify`) |
| KV bit-exactness | vendor KV pages | **PASS** (GDX `bf16` cold roundtrip) |
| Serve output match | - | not run |

## A. GPU memory

Savings % uses `1 - Isiro/Baseline` for loaded weights. For KV it uses `Isiro/Baseline - 1` as a % gain (and the matching `x` ratio).

### Weights

| Metric | Baseline | Isiro | Savings % |
|--------|----------|-------|-----------|
| GPU memory for loaded weights | 14.29 GiB | **10.25 GiB** | **28.27%** |

### KV cache

| Metric | Baseline default fill | Isiro reserved KV | Savings % |
|--------|-----------------------|-------------------|-----------|
| GPU memory reserved for KV | 12.56 GiB (235120 tokens) | **15.19 GiB** (284464 tokens)† | **20.94% (1.21x)** |

† Reserved pool on the GPU at vendor KV bytes/token. Token counts are pool admission slots (same bpv as the vendor dtype), not GDX-coded cold capacity. Longer context uses persist after vendor reclaim of prefix blocks (TokenIDs plus archived K,V). Not a second vendor KV cache on the host.

### KV vs context

`ISIRO_GDX=bf16` on KV from a Project Gutenberg text forward. At T=8192: Baseline KV 0.44 GiB, Isiro KV 0.30 GiB; save 0.13 GiB (30.5%). Bottom panel adds loaded weights (Baseline 14.29 GiB, Isiro 10.25 GiB): at T=8192, Baseline 14.73 GiB vs Isiro 10.55 GiB (weights + KV).

Figure. Weights and KV vs 1-user context.

![Weights and KV vs 1-user context](kv-bytes-vs-context.png)

## C. Generation (equal batch, input 32 / output 256)

Tok/s and latency at matched batch (`max_concurrency=32` / `max_num_seqs=32`). TTFT is in this table because it moves with concurrency. Isiro remetered after reopening hybrid densify under GDX (`_PREFILL_HYBRID_ALLOWED`; invent-skip had left prefill on fused).

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 1871.72 | 1872.38 |
| ITL (ms) p50 / p95 / p99 | 10.55 / 11.72 / 11.94 | 10.77 / 12.03 / 22.41 |
| TPOT (ms) p50 / p95 / p99 | 10.76 / 11.06 / 11.06 | 11.02 / 11.11 / 11.29 |
| TTFT (ms) p50 / p95 / p99 | 159.40 / 160.27 / 160.49 | 125.98 / 187.95 / 187.99 |

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
