# Isiro KV last-W reclaim remeter (N=20, 4096-in)

Scratch: `20261107T034711Z-rtx-5090` (no skip attach) and `20261107T035642Z-rtx-5090` (factory attach). Equal-batch, graphs on, W=512, FUSED=1, EngineArgs clamp 1.09 GiB / 20,024 FA tokens. Aggregate input `20 * 4096 = 81,920` is past the leftover FA cliff (~76-77k).

## Gate

Capacity seats: **NO GO**. Do not rewrite path-a Capacity from this remeter.

| check | result |
| --- | --- |
| Serve output match (first scrape) | PASS (4/4) |
| Baseline leftover FA tokens | 76,458 |
| Isiro clamped FA tokens | 20,024 |
| First scrape Isiro N=20 complete | 20/20 queued on the cube (`n_skip_unarmed`, `attend_class=a`, no deposit) |
| Reclaim factory attach | GO (`isiro_kv_reclaim scheduler factory on`, `attach via scheduler n=10`) |
| Deposit at seq>W | GO (`n_deposit=32`, `n_live_persist=32`, `f_cold=1.0`) |
| Fused a+ on skipped prefix | NO GO (`RuntimeError: skipped prefix without cold pages`; engine reset) |
| dual_hold | 0 (no dual_hold meter) |

## First scrape (skip wraps missed)

Scheduler factory was not hooked (retry timed out before `get_scheduler_cls`). Isiro completed 20 prompts by admitting ~5 concurrent 4k sequences on the 20k cube. Baseline completed 20 on leftover 76k (about 18 concurrent). Both `load_errors=0`. That is queueing, not last-W seats.

| meter | baseline | Isiro |
| --- | --- | --- |
| FA pool tokens | 76,458 | 20,024 |
| output tok/s | 29.10 | 26.13 |
| ITL p50 (ms) | 49.13 | 15.88 |
| TTFT p50 (ms) | 11902 | 13288 |
| output tokens | 640 | 640 |

## Second scrape (skip+deposit)

Factory hooked at model load. Prefill deposited cold. Decode fused miss fail-closed and the engine dropped the match capture (`Connection reset`). Capacity not stamped.
