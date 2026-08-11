---
tags:
  - tic
  - isiro
  - lossless
  - bit-exact
  - inference-optimized
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
---

# Qwen2.5-7B-Instruct TIC

Bit-exact lossless `.tic` compression of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

Same weights. **No quantization**.

[What is TIC?](https://isiro.ai/docs/tic)

| | |
|---|---|
| Precision | BF16 |
| Raw weights | 15.23 GB |
| `.tic` weights | 10.86 GB (28.7% savings) |
## Benchmarks

See [Benchmarks](https://github.com/isiroai/isiro/tree/main/benchmarks).

## Run

```sh
# Install
curl -fsSL https://isiro.ai/install.sh | sh

# Download .tic bundle
pip install -U huggingface_hub
hf download isiroai/Qwen2.5-7B-Instruct-TIC --local-dir Qwen2.5-7B-Instruct-TIC

# Serve (vLLM supported today)
isiro serve Qwen2.5-7B-Instruct-TIC --target vllm
```

[Install options](https://isiro.ai/docs/getting-started/install) · [Serve docs](https://isiro.ai/docs/getting-started/run) · [Compile your own](https://isiro.ai/docs/getting-started/compile)

## Verify

Bit-exact checks. Hash manifest: [`model.tic.manifest.json`](./model.tic.manifest.json).

```sh
# Confirm .tic integrity via the hash manifest
isiro verify Qwen2.5-7B-Instruct-TIC

# Optional: confirm .tic weights match the raw weights (requires compiler)
hf download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
isiro verify Qwen2.5-7B-Instruct-TIC -r Qwen2.5-7B-Instruct
```

## License

**Weights:** same license as [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) ([Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE)).

**`.tic`, ISIRO Runtime, and Compiler:** [ISIRO EULA](https://isiro.ai/eula).
