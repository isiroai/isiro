# ISIRO Benchmarks

Matched A/B with `vllm bench serve`: vLLM **baseline** vs ISIRO
(`isiro serve … --target vllm`).

Current setup: NVIDIA RTX 5090 (SM120, Blackwell, 32GB VRAM), BF16. ~29%
footprint savings apply across GPUs ([model cards](../model-cards/);
[Hugging Face](https://huggingface.co/isiroai)).

- Benchmarked:
  - [Qwen2.5-7B-Instruct](qwen2.5-7b-instruct/)
  - [Gemma 4 12B IT](gemma-4-12B-it/) (multimodal)
- In progress: production GPU benches (Qwen3.5-27B, Qwen3.5-35B-A3B MoE, etc);
HBM-mature kernels (A100, H100)

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

```bash
cp benchmarks/common.env.example benchmarks/{model}/common.env
```



## Run

`{model}` is the directory name under `benchmarks/`. Each launch writes a
timestamped report. One command runs needle-in-a-haystack, then the tok/s A/B.

```bash
# e.g. benchmarks/run_ab.sh qwen2.5-7b-instruct
# matches the report (Graph ON default)
benchmarks/run_ab.sh {model}
```

Other modes:

```bash
# optional Graph OFF
benchmarks/run_ab.sh {model} --graph-off
```

`SYSTEM_ID` comes from that model's `common.env` (or GPU auto-detect).

## Output

The report includes needle-in-a-haystack (fact retrieval by KV cache) and
`vllm bench serve` (tok/s, ITL). Plot deps: `benchmarks/requirements-plot.txt`.

Harness writes a timestamped copy (gitignored):

`benchmarks/{model}/{system_id}-report-<UTC>.md`

The published snapshot (tracked) is:

`benchmarks/{model}/{system_id}-report.md`

Logs and other run artifacts are under gitignored `benchmarks/scratch/`.

## Methodology

Each run compares vLLM (**baseline**) to ISIRO (**TIC** using vLLM target) on the same host with matched vLLM version, BF16, memory util, max len, TP, graph mode, seed, warmups, prompts, and I/O lengths. Fresh server per side.

The report records correctness, GPU memory, capacity, generation throughput
and latency under that matched setup.
