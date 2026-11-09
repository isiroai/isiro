# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `kv-only` | runtime `kv-only` | 2026-11-09 17:29 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype fp8`
- **Isiro KV-only:** `vllm serve` + `ISIRO_KV=fp8` (same AWQ/HF weights as baseline; no TIC weight claim)

Section A: loaded weights and Available KV GiB are shared (same raw weights, same vendor util). KV tokens capacity follows the measured Isiro/Baseline token ratio on that budget. KV bytes vs context are under KV vs context when the curve ran. Section C concurrency uses the same N both arms at each load point.

TIME claim: **NO GO** on concurrency (N=8; tok/s 94.72 vs 95.51; ITL 16.89 vs 16.00; TTFT 10,497.90 vs 10,540.26).

## Correctness

| Metric | Result |
|--------|--------|
| KV bit-exactness | **PASS** (Isiro KV `fp8` roundtrip) |
| Serve output match | **PASS** (4/4 prompts, temp=0, token IDs equal) |

## A. GPU memory

Loaded weights are identical on both arms (no TIC weight claim).

### Weights

| Metric | Value |
|--------|-------|
| GPU memory for loaded weights | 19.92 GiB |

### Available KV

| Metric | Value |
|--------|-------|
| Available KV | 4.00 GiB |

Available KV is the vendor reserved pool on the GPU (same util target both arms; no TIC weight headroom).

### KV vs context

`ISIRO_KV=fp8` is on for serve. kv-bytes-vs-context.png omitted: curve did not run.

## C. Concurrency

Same concurrent-request ceiling both arms at each N (no mismatched Baseline/Isiro N). Section A KV tokens capacity is why higher N is in scope.

### Concurrent requests = 8

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 8 | 8 |
| Output tok/s | 95.51 | 94.72 |
| ITL (ms) p50 / p95 / p99 | 16.00 / 309.78 / 551.75 | 16.89 / 301.58 / 547.44 |
| TPOT (ms) p50 / p95 / p99 | 41.94 / 67.95 / 70.27 | 42.72 / 68.54 / 70.85 |
| TTFT (ms) p50 / p95 / p99 | 10,540.26 / 16,886.12 / 17,288.36 | 10,497.90 / 16,842.34 / 17,236.26 |

## D. Capacity

Isiro concurrent requests use last-W skip so Isiro seats more 8K sequences than vendor FA can hold at this prompt length. Baseline stays at its max legal N.

Concurrent requests are the serve max concurrent sequences (`max_num_seqs`); the bench load matches that ceiling in this A/B.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 8 | **16** (**+100%**) |

### Generation (recall-matched KV, input 8192 / output 256)

Tok/s is aggregate system throughput across in-flight requests. Per-request decode rate is reflected by ITL/TPOT. Prefer section C Concurrency for matched-N latency and tok/s.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 93.85 | **102.13** (**+9%**) |
| ITL (ms) p50 / p95 / p99 | 15.96 / 16.38 / 400.87 | 17.78 / 18.24 / 2,188.84 |
| TPOT (ms) p50 / p95 / p99 | 32.00 / 59.14 / 61.61 | 68.28 / 127.06 / 132.16 |
| TTFT (ms) p50 / p95 / p99 | 13,609.76 / 17,749.56 / 17,755.83 | 22,573.23 / 35,553.32 / 35,583.37 |

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
