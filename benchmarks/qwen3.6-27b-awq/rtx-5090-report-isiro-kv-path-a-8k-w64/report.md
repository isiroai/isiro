# ISIRO Benchmark Report: `qwen3.6-27b-awq`

`awq` | `rtx-5090` | compiler `kv-only` | runtime `kv-only` | 2026-11-11 18:11 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`) with `--kv-cache-dtype fp8`
- **Isiro KV-only:** `vllm serve` + `ISIRO_KV=fp8` (same AWQ/HF weights as baseline; no TIC weight claim)

Section A: loaded weights and Available KV GiB are shared (same raw weights, same vendor util). KV tokens capacity follows the measured Isiro/Baseline token ratio on that budget. KV bytes vs context are under KV vs context when the curve ran. Section C concurrency uses the same N both arms at each load point.

Path-a W=64 long-prefix after timing-sin cuts (`ISIRO_KV_LIVE_WINDOW=64`, FA clamp on, input=2048, N=6). Stamp `20261111T180544Z`.

TIME claim: **NO GO** on concurrency (N=6; tok/s 211.76 vs 224.20; ITL 16.20 vs 14.64; TTFT 2,560.82 vs 2,559.52). TTFT flat; ITL/tok/s still regress under armed cold+reclaim. Do not flip LIVE_WINDOW default from 512.

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
| Available KV | 4.08 GiB |

Available KV is the vendor reserved pool on the GPU (same util target both arms; no TIC weight headroom).

### KV vs context

`ISIRO_KV=fp8` is on for serve. kv-bytes-vs-context.png omitted: curve did not run.

## C. Concurrency

Same concurrent-request ceiling both arms at each N (no mismatched Baseline/Isiro N). Section A KV tokens capacity is why higher N is in scope.

### Concurrent requests = 6

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Concurrent requests | 6 | 6 |
| Output tok/s | 224.20 | 211.76 |
| ITL (ms) p50 / p95 / p99 | 14.64 / 15.00 / 15.47 | 16.20 / 16.59 / 20.97 |
| TPOT (ms) p50 / p95 / p99 | 16.69 / 21.14 / 21.55 | 18.25 / 22.71 / 23.11 |
| TTFT (ms) p50 / p95 / p99 | 2,559.52 / 3,115.53 / 3,119.07 | 2,560.82 / 3,117.39 / 3,121.44 |

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
