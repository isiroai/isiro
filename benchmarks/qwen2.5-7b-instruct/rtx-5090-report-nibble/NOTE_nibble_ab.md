# Nibble weights-only A/B (2026-11-19)

## What 2.8x is

Northstar / micro dens is **M=1 GEMV** (down ~2.8x vs `F.linear` on a hot
CUDA graph). Integrate like PF3 means: attach nibble planes, post-KV stamp
of capture dests, opaque `nibble_fused_linear` on the weight path. No KV
codec work.

That path is wired. N=1 e2e transfers some of it (1.14x tok/s), diluted by
attention / norms / sample. Do not expect 2.8x aggregate tok/s from weight
dens alone.

## Capacity@32 is a different shape

Decode at N=32 is **batched M≈32**, not M=1 GEMV. Wall dens vs dense:

| layer | M=1 (micro) | M=32 (batched) |
|-------|-------------|----------------|
| q/o   | ~0.6-1.7x   | ~0.6x          |
| gate  | ~2.5x       | ~1.0x          |
| down  | **~2.8x**   | **~1.0x**      |

cuBLAS at M=32 reuses W across rows and stays ~flat in time; nibble's M>1
16-row GEMM tile is ~1x, not 2.8x. dual_m / batched-GEMV invent did not
move that cliff. Reverted; keep the product M=1 GEMV dens path simple.

## Earlier sin (fixed, keep)

Product forced `splits=1` for all M>1 → down ~0.25x at M=32. Batched decode
Ms now split-K again (fat-N splits=1; fat-K span~24). That lifted e2e from
~0.47x to ~0.70x. SSOT: `test_batched_decode_m32_down_not_quarter_dense`.

## Stamps

### Equal-N / Capacity @1

Scratch: `20261119T045813Z-rtx-5090`. Graphs on. Output match PASS.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 101.4 | **115.2 (1.14x)** |
| ITL p50 | 9.69 ms | **8.39 ms** |

### Equal-N / Capacity @32

Scratch: `20261119T050915Z-rtx-5090`. Graphs on. Output match PASS.
`report.md` is this stamp.

| Metric | Baseline | Isiro |
|--------|----------|-------|
| Output tok/s | 2816 | 1959 (**0.70x**) |

## TIME

**Partial GO** on PF3-style weight integrate + N=1 transfer + splits=1 sin
fix. **NO GO** claiming micro 2.8x as Capacity@32 e2e: that needs a real
M>1 dens kernel (separate invent), not more serve/KV wiring.
