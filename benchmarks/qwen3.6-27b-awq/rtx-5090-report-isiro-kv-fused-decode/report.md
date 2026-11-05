# Qwen3.6-27B-AWQ / RTX 5090: ISIRO_KV fused decode TIME

Status: **TIME NO GO**. Dens fused attend is on the serve hot path
(`attend_class=a+`, `n_fused_attend` on decode) after codes-workspace sins
were fixed, but equal-batch ITL and tok/s still lose vs matched fp8 baseline.
Product stamp stays `attend=(a)`.

Scrape: `20261105T174807Z-rtx-5090` (equal-batch, conc=1).
Env: `common.env.awq.isiro-kv-fused-decode`.

## Twin

Both arms: AWQ weights, `--kv-cache-dtype fp8`, `max_model_len=8192`,
`compilation_config` mode=3 + `PIECEWISE` (fairness match).

Isiro only: `ISIRO_KV=fp8` + `ISIRO_KV_FUSED_ATTEND=1` +
`ISIRO_KV_FUSED_DECODE=1` + `ISIRO_KV_LIVE_WINDOW=16` +
`ISIRO_KV_FP8_SCALES_OK=1`.

Dens recipe this scrape: deposit stamps attend-native uint8 codes workspace;
attend loads workspace → fused e4m3 MMA (no per-attend `read_page`); float
`_fa_k_scale_f`; AOT=JIT fused cold∥live body; live Flash+LSE merge.

## Gates

| Gate | Need | Got |
|------|------|-----|
| KV bit-exact | PASS | **PASS** (fp8, bpv~3.99) |
| Serve output match | PASS | **PASS** (4/4) |
| Equal-batch ITL / tok/s | win | **NO GO** |
| NIAH | not worse | skipped (no model `serve.yaml`) |
| dens_kv_graph D=128 Tk<=8192 | >=1 | **GO** (codes workspace remeter) |

## Equal-batch (generation-32-256, conc=1)

| Metric | Baseline fp8 | Isiro fused dens | Ratio |
|--------|--------------|------------------|-------|
| ITL p50 | 13.53 ms | 14.36 ms | 0.94× |
| TTFT p50 | 36.4 ms | 241.9 ms | 0.15× |
| tok/s | 73.5 | 65.2 | 0.89× |

Prior dens TIME (read_page expand + host sync + eager): ITL ~66.6 ms (0.20×).
Codes workspace remeter recovered ITL near parity; still not a win.

Serve meter: `attend_class=a+`, `n_fused_attend` on decode (q_tok=1).

## Sandbox dens (codes workspace)

D=128 GQA 28×4: dens_kv_graph GO at Tk 1024/4096/8192.
D=256 GQA 24×4: GO at Tk=1024; invent at 4096+.

## Posture

- Do not stamp `attend=(a+)` or TIME GO from this scrape.
- Path_a bf16 capacity GO remains the published capacity claim
  (`rtx-5090-report-isiro-kv/`).
- Next dens work: cut remaining fused wall (TTFT deposit + live merge share)
  while keeping a+ bitexact / output match; then remeter conc=8 once batched
  `q_tok>1` fused exists.
