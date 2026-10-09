# ISIRO Benchmark Report: `qwen2.5-7b-instruct`

`bf16` | `rtx-5090` | compiler `v0.1.1` | runtime `v0.1.1` | 2026-08-22 03:01 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)
- Needle-in-a-haystack (3-needle; Kamradt / RULER / EvalScope-style) for long-context recall

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`)
- **Isiro:** `isiro serve … --target vllm` (same vLLM version)

This directory is a mock: tables and figures use invented numbers so you can see the layout. A real `run_ab.sh` write looks the same (`report.md` plus the two PNGs).

## Correctness


| Metric             | Baseline         | Isiro                                           |
| ------------------ | ---------------- | ----------------------------------------------- |
| On-disk model size | 15.23 GB         | **10.78 GB** (**29.22% smaller**)               |
| Integrity          | baseline weights | **PASS** (`isiro verify`)                       |
| Serve output match | -                | **PASS** (4/4 prompts, temp=0, token IDs equal) |


## A. GPU memory

Savings % uses `1 - Isiro/Baseline` for loaded weights. For KV it uses `Isiro/Baseline - 1` as a % gain (and the matching `x` ratio).

### Weights


| Metric                        | Baseline | Isiro         | Savings %  |
| ----------------------------- | -------- | ------------- | ---------- |
| GPU memory for loaded weights | 15.34 GiB | **11.19 GiB** | **27.08%** |


### KV cache


| Metric                     | Baseline default fill     | Isiro reserved KV                | Savings %           |
| -------------------------- | ------------------------- | -------------------------------- | ------------------- |
| GPU memory reserved for KV | 12.71 GiB (238064 tokens) | **15.19 GiB** (288064 tokens)†   | **19.51% (1.21x)**  |


† Reserved pool on the GPU at vendor KV bytes/token. Token counts are pool admission slots, not GDX-coded cold capacity. Longer context uses persist after vendor reclaim of prefix blocks (TokenIDs plus archived K,V). Not a second vendor KV cache on the host.

## B. Needle-in-a-haystack (context tokens × depth)

Same long prompt, needles, and question on every setup that starts.

X is context tokens. Y is needle depth % (0 = start of prompt, 100 = end). Color is recall. An X in a column is OOM, same as the VRAM figure.

Figure. Fact retrieval by context length (3-needle haystack).

![Fact retrieval by context length (3-needle haystack)](needle-haystack.png)

Figure. VRAM usage vs context tokens.

Solid line is in-use: non-KV (weights) plus live vendor KV (usage_perc times pool, or derived T/pool) plus TokenID/persist outside the pool. Dashed lines are reserved pools (non-KV plus the full KV lock or fill). The dotted line is usable VRAM. The light red band is above usable VRAM. The darker red band is above GPU HBM. An X on the GPU HBM line is OOM at that context. Not nvidia-smi.

![VRAM usage vs context tokens](vram-vs-context.png)

## C. Equal batch

Same batch as Baseline. ITL stays near Baseline; capacity-matched generation in D raises concurrency and ITL.


| Metric                    | Baseline                | Isiro                    |
| ------------------------- | ----------------------- | ------------------------ |
| Output tok/s              | 1876.21                 | 1860.33                  |
| ITL (ms) p50 / p95 / p99  | 10.65 / 11.35 / 11.89   | 10.97 / 11.57 / 32.48    |
| TPOT (ms) p50 / p95 / p99 | 10.83 / 10.90 / 11.11   | 11.10 / 11.42 / 11.51    |
| TTFT (ms) p50 / p95 / p99 | 94.70 / 152.51 / 152.60 | 124.30 / 201.38 / 201.68 |


## D. Capacity

Isiro batch = round(Baseline batch × Isiro/Baseline KV token capacity) from a matched vLLM KV measurement.

KV token capacity is reserved pool slots when Isiro printed KV pool, else vLLM GPU tokens in this serve's reserved GPU pool.


| Metric            | Baseline              | Isiro                        |
| ----------------- | --------------------- | ---------------------------- |
| Batch             | 32                    | **39** (**1.21x**)           |
| KV token capacity | 238064 (12.71 GiB KV) | **288064** (15.19 GiB KV)† |


### Generation (recall-matched KV, input 32 / output 256)

Tok/s and latency at the recall-matched KV sizes. Batch is in Capacity. Not equal batch (C). TTFT is in this table because it moves with concurrency.


| Metric                    | Baseline                | Isiro                    |
| ------------------------- | ----------------------- | ------------------------ |
| Output tok/s              | 1876.21                 | 1860.33                  |
| ITL (ms) p50 / p95 / p99  | 10.65 / 11.35 / 11.89   | 10.97 / 11.57 / 32.48    |
| TPOT (ms) p50 / p95 / p99 | 10.83 / 10.90 / 11.11   | 11.10 / 11.42 / 11.51    |
| TTFT (ms) p50 / p95 / p99 | 94.70 / 152.51 / 152.60 | 124.30 / 201.38 / 201.68 |


## Config


| Item          | Value                            |
| ------------- | -------------------------------- |
| System        | `rtx-5090`                       |
| CUDA graphs   | on (eager prefill, graph decode) |
| KV cache      | product KV cache size            |
| Compiler      | `v0.1.1`                         |
| Runtime       | `v0.1.1`                         |
| vLLM          | 0.26.0                           |
| GPU           | NVIDIA GeForce RTX 5090 x 1      |
| Driver / CUDA | 590.44.01 / 13.1                 |


Fairness check: **PASS**.
