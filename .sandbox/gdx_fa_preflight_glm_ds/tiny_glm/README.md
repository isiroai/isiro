---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
library_name: transformers
---

# GLM-5.3-Flash-0.1B-A0.1B

This is a tiny version of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) created for testing and development.

## Model Details

- **Base Model**: zai-org/GLM-5.3-Flash
- **Architecture**: glm5_next (`Glm5NextForConditionalGeneration`)
- **Total Parameters**: 0.084B
- **Activated Parameters**: 0.084B (MoE: 4 of 8 routed experts + 1 shared expert per sparse layer)

This tiny model preserves the full architecture of the base model:
- Hybrid attention: **KDA linear attention** (`linear_attention`) layers **and** **DeepSeek sparse attention / MLA** (`deepseek_sparse_attention`) layers with the token indexer.
- Mixed FFN schedule: **dense** MLP layers (first 3) **and** **sparse MoE** layers (routed experts + shared expert).
- Manifold-Constrained **Hyper-Connections** (mHC) at every attention/FFN site.
- The **vision tower** (`Glm5NextVisionModel`) and multimodal projector.

The model is a bf16 dense checkpoint (the base model's fp8 `quantization_config` was removed so the tiny model can be randomly initialized and fine-tuned).

## Configuration Changes

The following parameters were reduced from the original model:

| Parameter | Original | Tiny |
|---|---|---|
| text `hidden_size` | 4096 | 256 |
| text `num_hidden_layers` | 45 | 5 |
| text `intermediate_size` (dense) | 12288 | 256 |
| text `moe_intermediate_size` | 2048 | 128 |
| `n_routed_experts` | 288 | 8 |
| `num_experts_per_tok` | 8 | 4 |
| `n_shared_experts` | 1 | 1 |
| `num_attention_heads` / `num_key_value_heads` | 64 | 4 |
| `q_lora_rank` | 1536 | 128 |
| `kv_lora_rank` | 512 | 64 |
| `qk_nope_head_dim` / `v_head_dim` | 256 | 64 |
| `index_n_heads` | 32 | 4 |
| `index_head_dim` | 128 | 64 |
| `index_topk` | 2048 | 64 |
| `index_kpool` | 4 | 4 |
| linear attn `num_heads` | 64 | 4 |
| vision `depth` | 24 | 2 |
| vision `hidden_size` | 1024 | 128 |
| vision `out_hidden_size` | 4096 | 256 |
| `vocab_size` | 154880 | 154880 (unchanged) |
| quantization | fp8 (block 128×128) | none (bf16) |

Per-layer schedules were regenerated for the reduced depth:
- `layer_types`: `[linear, linear, linear, deepseek_sparse_attention, linear]`
- `mlp_layer_types`: `[dense, dense, dense, sparse, sparse]`
- `indexer_types`: `[full, full, full, full, full]`

## Checkpoint Structure

Single-file `model.safetensors` (223 tensors). The tensor naming is analogous to the
original sharded checkpoint (`model.language_model.layers.*`, `model.visual.*`,
`lm_head.weight`, hyper-connection params `hc_attn_*` / `hc_ffn_*`, MLA params
`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`, indexer params, KDA linear-attention
params, and packed MoE `mlp.experts.*`).

Two intentional differences vs. the original:
- **No `weight_scale_inv` tensors** — the tiny model is bf16, not fp8.
- **No MTP layer** (original layer index 45 with `eh_proj`/`enorm`/`hnorm`/`shared_head.norm`)
  — the transformers `Glm5Next` model does not build the multi-token-prediction layer
  (`_keys_to_ignore_on_load_unexpected` skips `layers.45.*`), so no converter is required.

## Usage

```python
from transformers import Glm5NextForConditionalGeneration, AutoTokenizer

model = Glm5NextForConditionalGeneration.from_pretrained(
    "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("inference-optimization/GLM-5.3-Flash-0.1B-A0.1B")

input_ids = tokenizer("According to all known laws", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

## Creation Process

This model was created using the llm-compressor `create-tiny-model` claude skill.

1. Built a reduced `Glm5NextConfig` from the base config (removed `quantization_config`; shrank hidden/layer/expert/MLA/indexer/vision dims; regenerated per-layer schedules).
2. Randomly initialized weights (`init_weights` + non-finite/extreme-value fixup) with transformers 5.16.1.
3. Fine-tuned text-only on a small copypasta dataset until the training perplexity converged well below 3.0.
4. Verified the saved checkpoint structure matches the original naming convention (minus fp8 scales and the MTP layer).

## Validation

```
perplexity = 1.05 (target <= 10)   PASS
GEN: According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small
total params: 84,361,950
```

## Notes

- Requires `transformers >= 5.16.0` (which registers the `glm5_next` model type).
- This is a **randomly-initialized, fine-tuned-on-toy-data** model. It is intended solely for
  testing/development of tooling (quantization, serving, CI) and has **no** real language or
  vision capability.
- Fine-tuning was **text-only**; the vision tower is randomly initialized.
