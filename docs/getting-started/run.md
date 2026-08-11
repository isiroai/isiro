---
title: Serve a .tic
description: Load a compressed .tic bundle and run OpenAI-compatible inference.
group: getting-started
order: 3
anchorPrefixes:
  - serve-gpu
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

<Callout>
Download [sample models](/models) on Hugging Face, or [compile your own](/docs/getting-started/compile).
</Callout>

Port 8000 by default. Model output is unchanged: no quantization, no approximation.

### Usage

```sh
isiro serve <bundle> --target vllm [--host HOST] [--port PORT]
```

`<bundle>`: a compiled `-TIC` folder or path to `model.tic` inside it. Config and tokenizer files sit beside the `.tic`. The same path form works for `isiro verify` and `isiro info`.

`--target`: required; `vllm` in v0.1.0.

`--host` / `--port`: override `serve.yaml` when passed. Prometheus `/metrics` is on the same port.

### Serve settings

The TIC bundle includes `serve.yaml`. You can edit it for defaults such as host, port, `max_model_len`, `gpu_memory_utilization`, and the target image tag under `targets.vllm`. More target options can go there too.

CLI `--host` and `--port` override the file when you pass them. For tested image tags, run `isiro status --compat`. For flag details, run `isiro serve --help`.

The bundle must already be on the machine.

An example TIC bundle can be downloaded from Hugging Face e.g.

```sh
# Install Hugging Face Hub
python -m venv .venv
source .venv/bin/activate
pip install --upgrade huggingface_hub

# Download
hf download isiroai/Qwen2.5-7B-Instruct-TIC --local-dir Qwen2.5-7B-Instruct-TIC
```

(Linux / macOS sample. More install options: [Hugging Face installation](https://huggingface.co/docs/huggingface_hub/en/installation))

```sh
isiro serve Qwen2.5-7B-Instruct-TIC --target vllm
```

### Stop

```sh
docker stop isiro-serve-8000
```

Use the port you passed (default 8000). Container name is `isiro-serve-<port>`.

Ctrl+C in the serve terminal may leave the container holding the GPU or port; if that happens, run the same `docker stop` command.

NVIDIA GPU supported today.

Run `isiro serve --help` for host, port, metrics, and other options.

See [API](/docs/reference/api) for OpenAI-compatible endpoints. To chat from a browser UI, see [Chat with your model](/docs/getting-started/chat).
