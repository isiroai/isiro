---
title: Compile to .tic
description: Compress model weights into lossless .tic files with the ISIRO compiler.
group: getting-started
order: 2
anchorPrefixes:
  - compiler-access
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

<Callout>
Compiler is free. [Request access](/compiler).
</Callout>

Weights stay bit-for-bit. Model output is unchanged: no quantization, no approximation. [What is TIC?](/docs/tic)

### Usage

```sh
isiro compile <input> [--model-id org/model] [-o path/] [-j N]
```

Supported inputs: a model directory or a single weight file (`.safetensors`, `.onnx`, `.pth`, `.pt`).

`--model-id`: Optional. Auto-filled from input metadata when available. Pass `org/model` (or the id your serve target expects) when compile asks for a model id.

`-j` / `--jobs`: Parallel packing workers (default `1`). If you have enough memory, try `-j 8` so multiple layers pack concurrently.

The input model must already be on the machine.

An example model can be downloaded from Hugging Face e.g.

```sh
# Install Hugging Face Hub
python -m venv .venv
source .venv/bin/activate
pip install --upgrade huggingface_hub

# Download
hf download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
```

(Linux / macOS sample. More install options: [Hugging Face installation](https://huggingface.co/docs/huggingface_hub/en/installation))

For gated models, run `hf auth login` first ([create a token](https://huggingface.co/settings/tokens) if needed).

```sh
isiro compile Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-7B-Instruct -j 8
```

By default, compile writes a sibling `-TIC` folder (e.g. `Qwen2.5-7B-Instruct-TIC/`). Use `-o` / `--output` to set a custom bundle path.

### Bundle layout

```text
Qwen2.5-7B-Instruct-TIC/
  model.tic                  # packed weights
  model.tic.manifest.json    # hash manifest
  serve.yaml                 # defaults for isiro serve
  (sidecars)                 # config, tokenizer, and other files copied from input
```

### Verify

Bit-exact checks. Hash manifest: `model.tic.manifest.json`.

```sh
# Confirm .tic integrity via the hash manifest
isiro verify Qwen2.5-7B-Instruct-TIC

# Optional: confirm .tic weights match the raw weights (requires compiler)
isiro verify Qwen2.5-7B-Instruct-TIC -r Qwen2.5-7B-Instruct
```

Run `isiro compile --help` for more options.

To serve, see [Serve a .tic](/docs/getting-started/run).
