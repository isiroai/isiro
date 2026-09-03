# ISIRO Benchmarks

Matched A/B with `vllm bench serve`: vLLM **baseline** vs ISIRO
(`isiro serve … --target vllm`).

~29% footprint savings apply across BF16 models, see [model cards](../model-cards/) [Hugging Face](https://huggingface.co/isiroai).

- Benchmarked:
  - [Qwen2.5-7B-Instruct](qwen2.5-7b-instruct/)
  - [Gemma 4 12B IT](gemma-4-12B-it/)
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
```



## Run

`{model}` is the directory name under `benchmarks/`. Each launch writes a
timestamped report.

```bash
# e.g. benchmarks/run_ab.sh qwen2.5-7b-instruct
# matches Graph ON section of the report
benchmarks/run_ab.sh {model}
```

Other modes:

```bash
# matches the report Graph OFF / full eager section
benchmarks/run_ab.sh {model} --enforce-eager
# matches the report (Graph ON & Graph OFF)
benchmarks/run_ab.sh {model} --both-graph-modes
```

`SYSTEM_ID` comes from that model's `common.env` (or GPU auto-detect).

Single-request latency: `SERVE_MAX_NUM_SEQS=1` and `BENCH_MAX_CONCURRENCY=1` (or a second `common.env`). Do not change Hub `serve.yaml` defaults for that; overlay locally.

## Output

After a launch finishes, open the timestamped report:

`benchmarks/{model}/{system_id}-report-<UTC>.md`

Those filenames stay local (gitignored) so a customer run does not look like
a published result. The run UTC stays in the report body. To publish, copy
to `{system_id}-report.md` (no stamp in the filename) and commit that.

Logs and other run artifacts are under gitignored `benchmarks/scratch/`.

## Methodology

Each run compares vLLM (**baseline**) to ISIRO (**TIC** using vLLM target) on the same host with matched vLLM version, BF16, memory util, max len, TP, graph mode, seed, warmups, prompts, and I/O lengths. Fresh server per side.

The report records correctness, GPU memory, capacity, generation throughput
and latency under that matched setup.