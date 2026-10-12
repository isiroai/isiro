# ISIRO Benchmarks

Matched A/B with `vllm bench serve`: vLLM **baseline** vs **Isiro**
(`isiro serve … --target vllm`; TIC weights, optional GDX KV compress).

~29% footprint savings apply across BF16 models, see [model cards](../model-cards/README.md) and [Hugging Face](https://huggingface.co/isiroai).

- Benchmarked:
  - [Qwen2.5-7B-Instruct](qwen2.5-7b-instruct/) ([Isiro report](qwen2.5-7b-instruct/rtx-5090-report/))
  - [Gemma 4 12B IT](gemma-4-12B-it/)
  - [Gemma 4 12B IT FP8](gemma-4-12B-it-fp8/) ([Isiro report](gemma-4-12B-it-fp8/rtx-5090-report/))
  - [Qwen3.8-27B](qwen3.8-27b/)

## Prerequisites

- NVIDIA GPU and Docker with GPU support
- [ISIRO install](../README.md#quick-start)
- A baseline Hugging Face model and its compiled `.tic` bundle
- Bit-exactness check with `isiro verify -r` needs the ISIRO compiler.
Get [compiler access](https://isiro.ai/compiler).



## Configure

Create a model directory under `benchmarks/`. Copy the example env into a
local `common.env` in that directory, then set `BASELINE_MODEL_DIR`,
`TIC_MODEL_DIR`, `MODEL_ID`, and any other parameters in the file.
Leave `ISIRO_FORMAT` and `ISIRO_RUNTIME` as `auto` (the example default)
to read compiler from the `.tic` header and runtime from `isiro --help`.
Set either to a semver only when you want to pin.

```bash
cp benchmarks/common.env.example benchmarks/{model}/common.env
# Optional: bf16 KV compress A/B (ISIRO_GDX=bf16; lean publish skips NIAH)
# cp benchmarks/common.env.bf16.gdx-kv.example \
#   benchmarks/{model}/common.env.bf16.gdx-kv
# Optional: Hub FP8 weights + bf16 GDX KV (basename must end in .fp8)
# cp benchmarks/common.env.fp8.example benchmarks/{model}/common.env.fp8
```



## Run

`{model}` is the directory name under `benchmarks/`. Each launch writes a
timestamped report. Default runs needle-in-a-haystack, then the tok/s A/B.
Lean GDX publish configs set `ISIRO_BENCH_SKIP_NEEDLE=1` (Correctness uses
weight + KV bit-exactness + serve output match instead).

```bash
# e.g. benchmarks/run_ab.sh qwen2.5-7b-instruct
# matches the report (Graph ON default)
benchmarks/run_ab.sh {model}
```

Other modes:

```bash
# Graph OFF / full eager (--graph-off is an alias)
benchmarks/run_ab.sh {model} --enforce-eager
```

`SYSTEM_ID` comes from that model's `common.env` (or GPU auto-detect).

Single-request latency: `SERVE_MAX_NUM_SEQS=1` and `BENCH_MAX_CONCURRENCY=1` (or a second `common.env`). Do not change Hub `serve.yaml` defaults for that; overlay locally.

## Output

The report includes needle-in-a-haystack (fact retrieval by context length) and
`vllm bench serve` (tok/s, ITL). Plot deps: `benchmarks/requirements-plot.txt`.
GDX cold meters (host CUDA, not the vLLM image): pinned in
`benchmarks/requirements-gdx-meters.txt`; the harness runs
`scripts/ensure_gdx_meter_env.py` at start (managed venv
`~/.isiro/bench-venv`, override with `ISIRO_BENCH_PYTHON` /
`ISIRO_BENCH_VENV` / `ISIRO_BENCH_TORCH_INDEX`).

Harness writes a timestamped report directory (gitignored):

`benchmarks/{model}/{system_id}-report-<UTC>/`

That folder holds `report.md` and the PNGs below. The published snapshot (tracked) is:

`benchmarks/{model}/{system_id}-report/`

A made-up illustration of that layout is
`benchmarks/qwen2.5-7b-instruct/rtx-5090-report_mock/`.

To publish, copy the timestamped folder to `{system_id}-report/` (no stamp in
the directory name) and commit that.

Logs and other run artifacts are under gitignored `benchmarks/scratch/`.

### Figures this command writes

When needle is enabled, `benchmarks/run_ab.sh {model}` (Graph ON default) writes **two PNGs**:

1. `needle-haystack.png`: 3-needle recall heatmap. X is context tokens (EvalScope/Kamradt: 20 linear bins from 1k to the advertised model max). Y is needle depth % (EvalScope 11.11% steps). Color is recall fraction. An X in a column is OOM. Two panels: Isiro, Baseline default fill.
2. `vram-vs-context.png`: VRAM in-use vs context T (non-KV plus live pool blocks plus TokenID/persist). Dashed is reserved pool on each side. Dotted usable VRAM. Light red above usable VRAM; darker red above GPU HBM. An X on the GPU HBM line is OOM at that context. Not nvidia-smi reserved.

Lean GDX configs (`ISIRO_BENCH_SKIP_NEEDLE=1`) skip those PNGs and publish Correctness + GPU memory + capacity + generation tables only.

Scratch copies live under `benchmarks/scratch/{UTC}-{system_id}/`.

Tok/s, ITL, and GPU memory are markdown tables, not extra PNGs. CUDA graphs ON is the timed serve mode for those tables. Needle probes use eager serve so long-context starts.

`--enforce-eager` is a second launch: same figures if needle already ran in that scratch dir, plus eager tok/s tables. It does not draw extra PNGs.

## Methodology

Each run compares vLLM (**baseline**) to **Isiro** (`isiro serve` with TIC weights, optional `ISIRO_GDX`) on the same host with matched vLLM version, BF16, memory util, max len, TP, graph mode, seed, warmups, prompts, and I/O lengths. Fresh server per side.

Isiro uses the product `isiro serve` default, including isiro-fa (CUSTOM vendor-family drop-in when `ISIRO_ATTN` is unset). The harness clears an inherited `ISIRO_ATTN=0` so a parent shell cannot skip that default. Pin only with `ISIRO_BENCH_ISIRO_ATTN` if you need to override. Attention backend is recorded in serve config JSON; it is not a fairness key (baseline FLASH_ATTN vs Isiro CUSTOM is expected).

The report records correctness (weight bit-exactness; KV bit-exactness when GDX is on; serve output match), GPU memory, capacity, generation throughput and latency under that matched setup.

## Optional story JSON

Default `run_ab.sh` already draws VRAM in-use vs context from needle
serves (non-KV plus live pool plus persist/ids). Extra plots (stacked composition, tok/s vs concurrency, long-context
ITL, TCO) are optional. Put measured points in `stock_vllm_story.json` in the
run dir (schema `isiro-benchmark-stock-vllm-story-v1`). `generate_report.py`
renders that file when present. Template:

```bash
python3 benchmarks/scripts/stock_vllm_story.py --print-empty
```

TCO is derived from tok/s and a GPU-hour price you supply. It is not a
meter. Do not fill 70B / 8xH100 template cells without that SKU. Do not
treat 4 B/token vs ~56 KiB KV as live pool VRAM savings.
