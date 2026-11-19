# Isiro KV fused W=512 TIME (Phase 2)

Scratch: `20261106T171935Z-rtx-5090`. Equal-batch, graphs on, W=512, 4096-in / 256-out, conc=1. EngineArgs FA clamp skipped on this scrape so leftover fill matches baseline. Persist refuses deposit under the hot cube, so this is vendor FA TIME with FUSED=1 loaded but idle (`attend_class=a`, `n_skip_unarmed`).

## Gate

TTFT <= baseline x1.10: **PASS**. Serve match: **PASS**. ITL p50 / tok/s within ~1-2%. ITL p99 is a regression (not the TIME dens claim).

| meter | baseline | Isiro | ratio |
| --- | --- | --- | --- |
| TTFT p50 (ms) | 1057.4 | 1065.8 | 1.008 |
| ITL p50 (ms) | 13.58 | 13.77 | 1.014 |
| ITL p99 (ms) | 13.95 | 21.01 | 1.51 |
| decode tok/s | 56.66 | 55.98 | 0.988 |
| output match |  | 4/4 | PASS |
| GPU KV tokens | 77368 | 77368 | same (clamp skipped) |

Capacity benefit not active on this remeter. Clamp + piecewise ping is **GO** after the Hub AWQ plugin skips the TIC weight path (`n_attach=16`, FUSED=0 and FUSED=1, 20,024-token cube). vLLM still admits only those FA pages, so one-stream ~96k and aggregate>77k serving remain **NO GO** until prefix-page reclaim after deposit.
