"""Print KV cache shape contracts for GLM5Next + DeepseekV41 from vLLM sources."""
from pathlib import Path
import vllm
root = Path(vllm.__file__).parent

def grep_file(rel, needles, ctx=2):
    p = root / rel
    if not p.is_file():
        print('MISSING', rel); return
    lines = p.read_text(errors='ignore').splitlines()
    print('\n####', rel)
    for i,l in enumerate(lines):
      if any(n.lower() in l.lower() for n in needles):
        a=max(0,i-ctx); b=min(len(lines), i+ctx+1)
        for j in range(a,b):
          print(f'{j+1}:{lines[j]}')
        print('---')

# GLM attention / KDA
for rel in [
  'models/glm5next/common/attention.py',
  'models/glm5next/common/kda.py',
  'models/deepseek_v41/attention.py',
  'models/deepseek_v41/sparse_mla.py',
  'v1/attention/backends/mla/sparse_swa.py',
  'v1/core/kv_cache_utils.py',
]:
  grep_file(rel, [
    'kv_cache', 'get_kv_cache', 'kv_lora', 'page', 'block_size',
    'MLA', 'shape', 'dtype', 'FlashAttention', 'uint8', 'fp8',
    'linear_attention', 'kda', 'sparse',
  ], ctx=1)

# Also try importing model classes and reading AttentionSpec helpers
print('\n==== runtime imports ====')
try:
  from vllm.model_executor.models.registry import ModelRegistry
  for name in ('Glm5NextForConditionalGeneration','DeepseekV41ForCausalLM'):
    try:
      cls = ModelRegistry._try_load_model_cls(name) if hasattr(ModelRegistry,'_try_load_model_cls') else None
      print(name, 'cls', cls)
    except Exception as e:
      print(name, 'err', type(e).__name__, e)
except Exception as e:
  print('registry err', e)

# Inspect Attention backend registry for mla/sparse
try:
  from vllm.v1.attention.backends import registry as reg
  print('backend registry attrs', [a for a in dir(reg) if 'mla' in a.lower() or 'sparse' in a.lower() or 'FA' in a or 'flash' in a.lower()][:40])
except Exception as e:
  print('backend', e)
