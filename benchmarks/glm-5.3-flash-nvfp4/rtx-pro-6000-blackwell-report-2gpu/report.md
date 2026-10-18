# ISIRO Benchmark Report: `glm-5.3-flash-nvfp4` (2 GPU)

`nvfp4` weights | `fp8` KV | `rtx-pro-6000-blackwell` | 2 cards | in progress

This note is the two-card twin of `rtx-pro-6000-blackwell-report`. TTFT, ITL, and NIAH are not measured yet. Fill them from the SSH run. Do not treat the fit arithmetic below as a meter reading.

Compared:

- **Baseline:** stock `vllm serve` on the Hub weights (`GLM-5.3-Flash-NVFP4`). Keep that tree after the TIC compile. The A/B needs both trees on disk.
- **Isiro:** `GLM-5.3-Flash-NVFP4-TIC`.

Vanilla is not expected to fit on these two cards. Attempt the boot anyway and record it here. Do not delete the Hub weights to make room.

## Fit

Each card is 94.97 GiB. `gpu_memory_utilization: 0.95` is a 90.22 GiB budget. That number stays 0.95.

| Checkpoint | Per card | Against 90.22 GiB |
| --- | --- | --- |
| Vanilla weights, 190.43 GiB | 95.2 GiB | over budget |
| `GLM-5.3-Flash-NVFP4-TIC`, 204.4 GB raw to 181.8 GB packed (1.12x, 11.1%) | about 84.7 GiB | inside the budget if the packed scales stay packed |

Vanilla at this utilization does not fit on two cards. Raising utilization does not put 95.2 GiB of weights into a 94.97 GiB card.

## What the two-card serve stores

Weights: the 11.1% TIC bundle. A token expands 8 of 288 expert scales into the GEMM workspace. The packed streams for the other experts stay packed. Guard: `tests/test_joint_scale_workspace_ssot.py`.

KV: one GDX order-1 latent, split on the coordinate strips. Two cards together hold one compressed copy. Each card holds 328 of the 656 latent bytes per token, and the join is the original latent. On a 16-token crop of the GLM latent fixture, a frequency bank fit on each half was smaller than one shared bank (7,388 stream bytes versus 7,881). Guard: `tests/test_gdx_mla_latent_shard_ssot.py`.

## Serve

Config: `serve.yaml` in this directory. `tensor_parallel_size: 2`, `max_model_len: 8192`, `max_num_seqs: 10`, `kv_cache_dtype: fp8`, `gpu_memory_utilization: 0.95`.

Model: `GLM-5.3-Flash-NVFP4-TIC`.

## Correctness

| Metric | Result |
| --- | --- |
| Vanilla boot on two cards | Failed, 2026-10-17. Weights loaded at 89.54 GiB per card in 626 seconds (4-card run: 45.54 GiB in about 100 seconds). Then `CUDA_ERROR_OUT_OF_MEMORY` in the memory profile (`mhc_fused_tilelang_kernel`). Engine did not start. Hub weights stay on disk. |
| NIAH | not measured |
| Weight integrity | not measured on these two cards |

## Generation

Same shape as the 4-card report once the boxes are up: concurrency 10, input 32, output 256.

| Metric | 4-card isiro (2026-10-13) | 2-card isiro |
| --- | --- | --- |
| Output tok/s | 459.79 | not measured |
| ITL (ms) p50 / p95 / p99 | 21.04 / 21.63 / 22.20 | not measured |
| TTFT (ms) p50 / p95 / p99 | 230.50 / 325.33 / 325.94 | not measured |

Stamp a win only if needles hold and these latencies are not worse than the 4-card isiro numbers.
