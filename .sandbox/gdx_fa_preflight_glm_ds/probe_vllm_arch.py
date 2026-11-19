"""Locate vLLM support + summarize HF config FA-page suitability."""
from __future__ import annotations
import json, sys
from pathlib import Path

def summarize(cfg_path: Path) -> dict:
    cfg=json.loads(cfg_path.read_text())
    tc=cfg.get('text_config') if isinstance(cfg.get('text_config'), dict) else cfg
    lt=tc.get('layer_types') or []
    from collections import Counter
    return {
      'path': str(cfg_path),
      'model_type': tc.get('model_type') or cfg.get('model_type'),
      'architectures': cfg.get('architectures'),
      'num_hidden_layers': tc.get('num_hidden_layers'),
      'layer_types': dict(Counter(lt)) if lt else None,
      'kv_lora_rank': tc.get('kv_lora_rank'),
      'kv_source_layer_ids': tc.get('kv_source_layer_ids'),
      'index_source_layer_ids': tc.get('index_source_layer_ids'),
      'sliding_window': tc.get('sliding_window'),
      'num_key_value_heads': tc.get('num_key_value_heads'),
      'head_dim': tc.get('head_dim'),
      'qk_rope_head_dim': tc.get('qk_rope_head_dim'),
      'linear_attn_config_keys': sorted((tc.get('linear_attn_config') or {}).keys()) if isinstance(tc.get('linear_attn_config'), dict) else None,
    }

print('=== host config summarize ===')
for p in sys.argv[1:]:
  print(json.dumps(summarize(Path(p)), indent=2))

print('=== vllm inventory ===')
import vllm
root=Path(vllm.__file__).parent
keys=('Glm5Next','DeepseekV41','glm5_next','deepseek_v41','MLA','mla_kv','flashmla')
for p in sorted(root.rglob('*.py')):
  try:
    t=p.read_text(errors='ignore')
  except Exception:
    continue
  if any(k in t for k in ('Glm5Next','DeepseekV41','glm5_next','deepseek_v41')):
    print('HIT', p.relative_to(root))
