# ISIRO Benchmark Report: `glm-5.3-flash-nvfp4`

`nvfp4` weights | `fp8` KV | `rtx-pro-6000-blackwell` | path_a_mla / o1-mla-latent | 2026-10-13 UTC

Tooling:

- `vllm bench serve` (tok/s, ITL / TPOT / TTFT percentiles)
- Ambient twin scrape (`remeter_path_a_vs_vanilla.py`) for NIAH / Isiro KV ledger
- Isiro KV KV-bytes curve from vendor MLA latent capture (`mla_latent_real_t100k.npz`; T through 1,000,000 via tiled remeter)

Compared:

- **Baseline:** stock `vllm serve` on Hub NVFP4 (`GLM-5.3-Flash-NVFP4`)
- **Isiro:** `isiro-serve-native` on `GLM-5.3-Flash-NVFP4-TIC` with `ISIRO_KV=fp8` (route=`path_a_mla`, payload=`o1-mla-latent`, `dual_hold=0`)

Serve knobs matched to TIC `serve.yaml` remeter: TP=4, `max_model_len=8192`, `max_num_seqs=10`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.95`, `kv_cache_dtype=fp8`, prefix cache off, flashinfer autotune off, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. Isiro: `ISIRO_KV_LIVE_WINDOW=512`, graphs-on (no `--enforce-eager`).

## Correctness


| Metric             | Baseline                                      | Isiro                                                      |
| ------------------ | --------------------------------------------- | ---------------------------------------------------------- |
| On-disk model size | 190.43 GiB                                    | **181.54 GiB** (**4.67% smaller**)                         |
| Integrity          | baseline weights                              | PASS (TIC load / serve boot)                               |
| KV cold deposit    | vendor MLA pages                              | **PASS** (`n_deposit=11` / `n_mla_hook=11`, `f_cold=1.0`) |
| Dual-hold          | -                                             | **PASS** (`dual_hold=0`, `free_fa_twin` ledger)            |
| NIAH (scrape grid) | PASS through 4000; 8000 d50 HTTP 500          | PASS (256/1024/2048/4000/8000 d50 = 1.0)                 |


## A. GPU memory

Savings % uses `1 - Isiro/Baseline` for loaded weights (on-disk stamp here; serve banner weight delta **4.7%**).

### Weights


| Metric                        | Baseline     | Isiro          | Savings % |
| ----------------------------- | ------------ | -------------- | --------- |
| On-disk / loaded weight bytes | 190.43 GiB   | **181.54 GiB** | **4.67%** |


### KV cache (reserved pool)


| Metric                    | Baseline  | Isiro     |
| ------------------------- | --------- | --------- |
| GPU KV cache tokens (log) | 2,377,500 | 2,273,735 |


Reserved pool is vendor admission slots, not Isiro-KV-coded cold capacity. Physical vendor MLA pages still attend after deposit (`free_fa_twin` is ledger-only).

### Live Isiro KV ledger (Isiro, after long prompt)

Cold o1 slab vs synthetic twin (not nvidia-smi). After attach-ordinal fix, all 11 Full MLA layers deposit.


| Metric                 | Value       |
| ---------------------- | ----------- |
| `n_deposit` / hooks    | **11 / 11** |
| `save_vs_all_fa`       | **0.987**   |
| `f_cold`               | 1.0         |
| `slab_nbytes`          | 88976       |
| `mla_latent_raw_bytes` | 6869632     |


`save_vs_all_fa = 1 - slab/raw` on deposited cold (W=512 live window stays twin-dense outside this ledger).

### KV vs context (coded cold)

`ISIRO_KV=fp8` on vendor MLA latent (`kv_source=vendor_mla_latent`, W=512). Capture T=100,000; points through T=1,000,000 remeter pack on tiled capture chunks (`fa_twin_mode=mla_latent_tiled_sequence`). At T=1,000,000: Baseline KV **6.88 GiB**, Isiro KV **3.94 GiB**; save **42.8%** (coded cold pack rate). Live ledger 0.987 remains the short deposited-prefix meter, not this full-sequence curve.

Figure. Weights and KV vs 1-user context (tagline: `weights nvfp4 | KV fp8`).

![Weights and KV vs 1-user context](kv-bytes-vs-context.png)

## C. Capacity

Concurrent requests are 10 on both sides. Not scaled from KV token capacity.


| Metric              | Baseline | Isiro |
| ------------------- | -------- | ----- |
| Concurrent requests | 10       | 10    |


### Generation (recall-matched KV, input 32 / output 256)

Tok/s is aggregate system throughput across in-flight requests (`vllm bench serve`, `--temperature 0`, `--max-concurrency 10`, 40 prompts). Per-request decode rate is reflected by ITL/TPOT.


| Metric                    | Baseline                | Isiro                   |
| ------------------------- | ----------------------- | ----------------------- |
| Output tok/s              | 451.94                  | **459.79**              |
| ITL (ms) p50 / p95 / p99  | 19.61 / 23.86 / 27.15   | 21.04 / **21.63** / **22.20** |
| TPOT (ms) p50 / p95 / p99 | **19.37** / **20.50** / 26.13 | 20.94 / 21.17 / **21.33** |
| TTFT (ms) p50 / p95 / p99 | 306.44 / 1952.48 / 1952.86 | **230.50** / **325.33** / **325.94** |


Isiro wins aggregate output tok/s and TTFT (especially p95/p99 under concurrency). ITL/TPOT p50 are slightly higher than baseline; ITL p95/p99 are tighter.

## Config

- Baseline model: `/root/models/llm/fp4/GLM-5.3-Flash-NVFP4`
- TIC model: `/root/models/llm/fp4/GLM-5.3-Flash-NVFP4-TIC` (`serve.yaml` as above)
- Bench: `isiro_gdx_bench_serve.json`, `vanilla_matched_bench_serve.json`
- Curve: `isiro_kv_bytes_curve.json`, `kv-bytes-vs-context.png` (T through 1,000,000)
- Ledger / NIAH: `isiro_gdx_mla_graphs_scrape.json`, `vanilla_matched_scrape.json`
