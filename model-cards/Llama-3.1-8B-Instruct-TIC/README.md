---
tags:
  - tic
  - isiro
  - lossless
  - bit-exact
  - inference-optimized
license: llama3.1
language:
  - en
pipeline_tag: text-generation
---

# Llama-3.1-8B-Instruct TIC

Bit-exact lossless `.tic` compression of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

Same weights. **No quantization**.

[What is TIC?](https://isiro.ai/docs/tic)

| | |
|---|---|
| Precision | BF16 |
| Raw weights | 16.06 GB |
| `.tic` weights | 11.44 GB (28.8% savings) |
## Benchmarks

See [Benchmarks](https://github.com/isiroai/isiro/tree/main/benchmarks).

## Run

```sh
# Install
curl -fsSL https://isiro.ai/install.sh | sh

# Download .tic bundle
pip install -U huggingface_hub
hf download isiroai/Llama-3.1-8B-Instruct-TIC --local-dir Llama-3.1-8B-Instruct-TIC

# Serve (vLLM supported today)
isiro serve Llama-3.1-8B-Instruct-TIC --target vllm
```

[Install options](https://isiro.ai/docs/getting-started/install) · [Serve docs](https://isiro.ai/docs/getting-started/run) · [Compile your own](https://isiro.ai/docs/getting-started/compile)

## Verify

Bit-exact checks. Hash manifest: [`model.tic.manifest.json`](./model.tic.manifest.json).

```sh
# Confirm .tic integrity via the hash manifest
isiro verify Llama-3.1-8B-Instruct-TIC

# Optional: confirm .tic weights match the raw weights (requires compiler)
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct
isiro verify Llama-3.1-8B-Instruct-TIC -r Llama-3.1-8B-Instruct
```

## License

**Weights:** same license as [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) ([Llama 3.1 Community License](./LICENSE)). See also [`USE_POLICY.md`](./USE_POLICY.md).

**`.tic`, ISIRO Runtime, and Compiler:** [ISIRO EULA](https://isiro.ai/eula).
