---
tags:
  - tic
  - isiro
  - lossless
  - bit-exact
  - inference-optimized
  - multimodal
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
language:
  - en
pipeline_tag: any-to-any
---

# gemma-4-12B-it TIC

Bit-exact lossless `.tic` compression of [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it).

Same weights. **No quantization**.

[What is TIC?](https://isiro.ai/docs/tic)

| | |
|---|---|
| Precision | BF16 |
| Raw weights | 23.92 GB |
| `.tic` weights | 17.05 GB (28.7% savings) |
## Benchmarks

See [Benchmarks](https://github.com/isiroai/isiro/tree/main/benchmarks).

## Run

```sh
# Install
curl -fsSL https://isiro.ai/install.sh | sh

# Download .tic bundle
pip install -U huggingface_hub
hf download isiroai/gemma-4-12B-it-TIC --local-dir gemma-4-12B-it-TIC

# Serve (vLLM supported today)
isiro serve gemma-4-12B-it-TIC --target vllm
```

[Install options](https://isiro.ai/docs/getting-started/install) · [Serve docs](https://isiro.ai/docs/getting-started/run) · [Compile your own](https://isiro.ai/docs/getting-started/compile)

## Verify

Bit-exact checks. Hash manifest: [`model.tic.manifest.json`](./model.tic.manifest.json).

```sh
# Confirm .tic integrity via the hash manifest
isiro verify gemma-4-12B-it-TIC

# Optional: confirm .tic weights match the raw weights (requires compiler)
hf download google/gemma-4-12B-it --local-dir gemma-4-12B-it
isiro verify gemma-4-12B-it-TIC -r gemma-4-12B-it
```

## License

**Weights:** same license as [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) ([Gemma Terms / Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license)).

**`.tic`, ISIRO Runtime, and Compiler:** [ISIRO EULA](https://isiro.ai/eula).
