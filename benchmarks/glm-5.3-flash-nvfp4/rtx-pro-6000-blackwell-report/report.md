# ISIRO Benchmark Report: `glm-5.3-flash-nvfp4`

`nvfp4` weights | `fp8` KV | `rtx-pro-6000-blackwell` | path_a_mla / o1-mla-latent | 2026-10-13 UTC

Tooling:

- Ambient twin scrape (`remeter_path_a_vs_vanilla.py`) on vast-glm (no Docker)
- GDX KV-bytes curve from vendor MLA latent capture (`mla_latent_real.npz`)

Compared:

- **Baseline:** stock `vllm serve` on Hub NVFP4 (`GLM-5.3-Flash-NVFP4`)
- **Isiro:** `isiro-serve-native` on `GLM-5.3-Flash-NVFP4-TIC` with `ISIRO_GDX=fp8` (route=`path_a_mla`, payload=`o1-mla-latent`)

Serve knobs matched to TIC `serve.yaml` remeter: TP=4, `max_model_len=8192`, `max_num_seqs=10`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.95`, `kv_cache_dtype=fp8`, prefix cache off, flashinfer autotune off, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. Isiro arm additionally used `--enforce-eager` and `ISIRO_GDX_LIVE_WINDOW=512`.

## Correctness


| Metric             | Baseline                                      | Isiro                                      |
| ------------------ | --------------------------------------------- | ------------------------------------------ |
| Ping / chat        | PASS (`ping_ok=1`)                            | PASS (`ping_ok=1`)                         |
| NIAH (scrape grid) | PASS through 4000; 8000 d50 HTTP 500          | PASS (256/1024/2048/4000/8000 d50 = 1.0) |
| KV cold deposit    | -                                             | PASS (`n_deposit>=1`, `save_vs_all_fa>0`)  |


## A. GPU memory

### KV cache (reserved pool)

| Metric                     | Baseline              | Isiro                   |
| -------------------------- | --------------------- | ----------------------- |
| GPU KV cache tokens (log)  | 2,377,500             | 2,273,735               |

Reserved pool is vendor admission slots, not GDX-coded cold capacity.

### Live GDX ledger (Isiro, after long prompt)

| Metric            | Value   |
| ----------------- | ------- |
| `save_vs_all_fa`  | **0.974** |
| `f_cold`          | 1.0     |
| `slab_nbytes`     | 16072   |
| `mla_latent_raw`  | 624512  |
| `n_mla_hook`      | 11      |

### KV vs context (coded cold)

`ISIRO_GDX=fp8` on vendor MLA latent pages (`kv_source=vendor_mla_latent`, W=512). At T=2048: Baseline twin 14.10 MiB, Isiro total 10.89 MiB; save **22.7%** (hot window stays twin; cold o1-packed).

Figure. Weights and KV vs 1-user context.

![Weights and KV vs 1-user context](kv-bytes-vs-context.png)

## C. Capacity


| Metric              | Baseline | Isiro |
| ------------------- | -------- | ----- |
| Concurrent requests | 10       | 10    |


### Generation (ambient stream scrape; not full `vllm bench serve`)


| Metric                   | Baseline | Isiro   |
| ------------------------ | -------- | ------- |
| Stream tok/s             | 60.27    | 11.31   |
| Stream ITL p50 (ms)      | 11.62    | 86.78   |
| Stream TTFT (ms)         | 130.55   | 125.01  |


Isiro arm was enforce-eager path_a_mla (deposit side-stream). Tok/s is not a graphs-on capacity claim; use the KV curve / `save_vs_all_fa` for the memory headline.

## Config

- Baseline model: `/root/models/llm/fp4/GLM-5.3-Flash-NVFP4`
- TIC model: `/root/models/llm/fp4/GLM-5.3-Flash-NVFP4-TIC` (`serve.yaml` as above)
- `ISIRO_BENCH_MLA_LATENT_NPZ=/workspace/logs/mla_latent_real.npz`
- Artifacts: `isiro_gdx_mla_scrape.json`, `vanilla_matched_scrape.json`, `gdx_kv_bytes_curve.json`
