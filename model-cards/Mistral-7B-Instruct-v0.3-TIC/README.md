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

# Mistral-7B-Instruct-v0.3 TIC

Bit-exact lossless `.tic` compression of [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

Same weights. **No quantization**.

[What is TIC?](https://isiro.ai/docs/tic)

| | |
|---|---|
| Precision | BF16 |
| Raw weights | 14.50 GB |
| `.tic` weights | 10.33 GB (28.8% savings) |
## Benchmarks

See [Benchmarks](https://github.com/isiroai/isiro/tree/main/benchmarks).

## Run

```sh
# Install
curl -fsSL https://isiro.ai/install.sh | sh

# Download .tic bundle
pip install -U huggingface_hub
hf download isiroai/Mistral-7B-Instruct-v0.3-TIC --local-dir Mistral-7B-Instruct-v0.3-TIC

# Serve (vLLM supported today)
isiro serve Mistral-7B-Instruct-v0.3-TIC --target vllm
```

[Install options](https://isiro.ai/docs/getting-started/install) · [Serve docs](https://isiro.ai/docs/getting-started/run) · [Compile your own](https://isiro.ai/docs/getting-started/compile)

## Verify

Bit-exact checks. Hash manifest: [`model.tic.manifest.json`](./model.tic.manifest.json).

```sh
# Confirm .tic integrity via the hash manifest
isiro verify Mistral-7B-Instruct-v0.3-TIC

# Optional: confirm .tic weights match the raw weights (requires compiler)
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir Mistral-7B-Instruct-v0.3
isiro verify Mistral-7B-Instruct-v0.3-TIC -r Mistral-7B-Instruct-v0.3
```

## License

**Weights:** same license as [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) ([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)).

**`.tic`, ISIRO Runtime, and Compiler:** [ISIRO EULA](https://isiro.ai/eula).
