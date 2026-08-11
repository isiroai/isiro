---
title: TIC
description: The lossless inference compression format
group: reference
order: 1
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

`.tic` is a file format for lossless AI model compression. It reduces the storage and memory footprint of a model without changing a single weight value.

A model served from a `.tic` file behaves identically to the same model served from its original source. The weights are bit-exact: not quantized, approximated, or altered. Integrity is checked with a cryptographic hash manifest (`isiro verify`). Source bit-exactness against the raw model is confirmed with `isiro verify -r`.

TIC stands for **T**ensor **I**nference **C**ore. The name reflects what the format encodes: the lossless informational *core* of a model's *tensors*, preserved for *inference*. The `.tic` format is designed to keep weights compact without changing a single value.

The format and its name were introduced by Kunle Olutomilayo, founder of ISIRO, in March 2026.

## Why a new format?

Model file formats are built around the problem they were designed to solve:

| Format | Primary goal |
|---|---|
| `.pt` / `.pth` (Pickle) | Python serialization |
| `.safetensors` | Safe loading (no arbitrary code execution) |
| `.gguf` | Quantized inference via llama.cpp |
| **`.tic`** | **Lossless inference compression with execution-native decoding** |

Each format reflects its authors' constraints. `.tic` starts from a different premise:

<PullQuote>
  Weights must be bit-exact, compressed, and executed at compute speed.
</PullQuote>

## The problem `.tic` solves

AI model inference is largely a memory traffic problem. Every forward pass requires reading model weights from memory into compute units. This is the binding cost at scale and on constrained hardware.

Lossless compression reduces the bytes that must move on every forward pass without changing what the model computes. The model behaves identically. Only its memory footprint changes.

Existing formats store raw weights at rest and load them fully into memory before inference. Storage and execution are separate steps. `.tic` collapses them by storing compressed weights while the runtime uses fused decode to process weights directly on the chip.

## Why `.tic` is not an extension of existing formats

### Safetensors

Safetensors stores tensors as contiguous raw bytes at fixed offsets recorded in a JSON header: "tensor X starts at byte N, length L." This works because raw tensors have predictable, fixed sizes: shape multiplied by the dtype size.

Lossless compression can produce variable-length output: the same tensor shape may compress to a different size depending on the weight values. The safetensors header contract expects fixed, predictable tensor sizes. Variable-length compressed tensors break that contract. Storing them without changing the format would require padding each tensor to its uncompressed size, which eliminates the compression benefit entirely.

A lossless compression format designed to maximize savings requires a different byte-layout contract from the ground up.

### GGUF

GGUF is optimized for quantized inference via llama.cpp. Its primary goal is making models runnable on hardware where they would not otherwise fit, which requires reducing weight values to fewer bits. `.tic` preserves weights exactly and targets the serving cost of models that already fit. These are different problems with different design requirements, and the formats reflect that.

A lossless format that stays compact while decoding during inference needs a layout built for that path. Retrofitting the same idea into a format designed for fixed-size raw tensors would require changes well beyond a small extension.

## What `.tic` contains

`.tic` stores weight data, metadata, and an integrity manifest. It cannot execute code on load. No pickle, no arbitrary Python objects, no serialization of program state. This is a deliberate design property, not a side effect of the compression scheme.

A `.tic` file has three logical sections:

- **Header.** Format metadata describing the file's contents and structure.
- **Compressed weight payloads.** Compressed tensors stored for efficient fused decode during inference.
- **Integrity manifest.** A hash manifest, embedded in the `.tic` file and also written as a sidecar, for verifying on-disk integrity (`isiro verify`). Compare to the raw model with `isiro verify -r`.

## Format properties

| Property | Value |
|---|---|
| Compression | Lossless. Weights are not quantized or approximated. |
| Accuracy preserved | Exact. Weights are stored and decoded bit-for-bit as compiled. |
| Executable code on load | No |
| Integrity verification | Hash manifest for on-disk integrity, embedded in the `.tic` file and generated as a sidecar (`isiro verify`). Source bit-exactness: `isiro verify -r`. |

## Related

- Sample `.tic` models: [huggingface.co/isiroai](https://huggingface.co/isiroai)
- Benchmarks: [github.com/isiroai/isiro](https://github.com/isiroai/isiro)
- Compiler and runtime: [ISIRO Runtime](/product/runtime)
