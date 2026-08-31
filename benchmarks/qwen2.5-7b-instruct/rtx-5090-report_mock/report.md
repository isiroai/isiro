# ISIRO Benchmark Report: `qwen2.5-7b-instruct`

`bf16` | `rtx-5090` | compiler `v0.1.1` | runtime `v0.1.1` | 2026-08-22 03:01 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL)
- Needle-in-a-haystack (3-needle; Kamradt / RULER / EvalScope-style) for long-context recall

Compared:

- **Baseline:** vLLM (`vllm serve` / Docker `vllm/vllm-openai`)
- **TIC:** `isiro serve … --target vllm` (same vLLM version)

This directory is a mock: tables and figures use invented numbers so you can see the layout. A real `run_ab.sh` write looks the same (`report.md` plus the two PNGs).

## Correctness


| Metric             | Baseline         | TIC                                             |
| ------------------ | ---------------- | ----------------------------------------------- |
| On-disk model size | 15.23 GB         | **10.78 GB** (**29.22% smaller**)               |
| Integrity          | baseline weights | **PASS** (`isiro verify`)                       |
| Serve output match | -                | **PASS** (4/4 prompts, temp=0, token IDs equal) |


## A. GPU memory

### Weights


| Metric                        | Baseline | TIC          | Savings %  |
| ----------------------------- | -------- | ------------ | ---------- |
| GPU memory for loaded weights | 15.34 GiB | **11.19 GiB** | **27.08%** |


### KV cache


| Metric                     | Baseline default fill     | TIC reserved KV              | Savings %  |
| -------------------------- | ------------------------- | ---------------------------- | ---------- |
| GPU memory reserved for KV | 12.71 GiB (238064 tokens) | **1.09 GiB** (20480 tokens)† | **91.42%** |


† Reserved pool on the GPU. Longer context uses persist after vendor reclaim of prefix blocks (TokenIDs plus archived K,V). Not a second vendor KV cache on the host.

## B. Needle-in-a-haystack (context tokens × depth)

Same long prompt, needles, and question on every setup that starts. The Baseline serve would not start at 1.09 GiB.

X is context tokens. Y is needle depth % (0 = start of prompt, 100 = end). Color is recall. An X in a column is OOM, same as the VRAM figure.

Figure. Fact retrieval by context length (3-needle haystack).

![Fact retrieval by context length (3-needle haystack)](needle-haystack.png)

Figure. VRAM usage vs context tokens.

Solid line is reported non-KV plus vLLM pool occupancy (metrics, or derived T/pool if /metrics is missing). Dashed line is Baseline non-KV plus its reserved KV pool. The dotted line is usable VRAM. The light red band is above usable VRAM. The darker red band is above GPU HBM. An X on the GPU HBM line is OOM at that context. Reported non-KV plus pool occupancy, not a process nvidia-smi meter.

![VRAM usage vs context tokens](vram-vs-context.png)

## C. Capacity

Batch is 32 on both sides. Not scaled from KV token capacity.

KV token capacity is vLLM-reported full-attn tokens in this serve's reserved GPU pool.


| Metric            | Baseline              | TIC                  |
| ----------------- | --------------------- | -------------------- |
| Batch             | 32                    | 32                   |
| KV token capacity | 238064 (12.71 GiB KV) | 20480 (1.09 GiB KV)† |


## D. Generation (recall-matched KV, input 32 / output 256)

Tok/s and latency at the recall-matched KV sizes. Batch is in Capacity. Not equal batch (E). TTFT is in this table because it moves with concurrency.


| Metric                    | Baseline                | TIC                      |
| ------------------------- | ----------------------- | ------------------------ |
| Output tok/s              | 1876.21                 | 1860.33                  |
| ITL (ms) p50 / p95 / p99  | 10.65 / 11.35 / 11.89   | 10.97 / 11.57 / 32.48    |
| TPOT (ms) p50 / p95 / p99 | 10.83 / 10.90 / 11.11   | 11.10 / 11.42 / 11.51    |
| TTFT (ms) p50 / p95 / p99 | 94.70 / 152.51 / 152.60 | 124.30 / 201.38 / 201.68 |


## E. Equal batch

Same batch as Capacity.


| Metric                    | Baseline                | TIC                      |
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
