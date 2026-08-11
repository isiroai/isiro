---
tags:
  - tic
  - isiro
  - lossless
  - bit-exact
  - inference-optimized
  - multimodal
  - moe
license: apache-2.0
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3.5-35B-A3B TIC

Bit-exact lossless `.tic` compression of [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).

Same weights. **No quantization**.

[What is TIC?](https://isiro.ai/docs/tic)

| | |
|---|---|
| Precision | BF16 |
| Type | multimodal, MoE |
| Raw weights | 71.90 GB |
| `.tic` weights | 51.27 GB (28.7% savings) |
## Benchmarks

See [Benchmarks](https://github.com/isiroai/isiro/tree/main/benchmarks).

## Run

```sh
# Install
curl -fsSL https://isiro.ai/install.sh | sh

# Download .tic bundle
pip install -U huggingface_hub
hf download isiroai/Qwen3.5-35B-A3B-TIC --local-dir Qwen3.5-35B-A3B-TIC

# Serve (vLLM supported today)
isiro serve Qwen3.5-35B-A3B-TIC --target vllm
```

[Install options](https://isiro.ai/docs/getting-started/install) · [Serve docs](https://isiro.ai/docs/getting-started/run) · [Compile your own](https://isiro.ai/docs/getting-started/compile)

## Verify

Bit-exact checks. Hash manifest: [`model.tic.manifest.json`](./model.tic.manifest.json).

```sh
# Confirm .tic integrity via the hash manifest
isiro verify Qwen3.5-35B-A3B-TIC

# Optional: confirm .tic weights match the raw weights (requires compiler)
hf download Qwen/Qwen3.5-35B-A3B --local-dir Qwen3.5-35B-A3B
isiro verify Qwen3.5-35B-A3B-TIC -r Qwen3.5-35B-A3B
```

## License

**Weights:** same license as [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE)).

**`.tic`, ISIRO Runtime, and Compiler:** [ISIRO EULA](https://isiro.ai/eula).
