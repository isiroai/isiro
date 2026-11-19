import os, json
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
model='/probe/tiny_glm'
print('loading', model)
try:
  llm=LLM(model=model, max_model_len=256, enforce_eager=True,
          gpu_memory_utilization=0.5, trust_remote_code=True,
          dtype='auto', enable_prefix_caching=False, kv_cache_dtype='fp8')
except Exception as e:
  print('LOAD_FAIL', type(e).__name__, e)
  raise SystemExit(2)
worker=llm.llm_engine.model_executor.driver_worker
worker=getattr(worker,'worker',worker)
runner=worker.model_runner
caches=list(getattr(runner,'kv_caches',[]) or [])
print('n_caches', len(caches))
for i,kv in enumerate(caches[:20]):
  import torch
  if torch.is_tensor(kv):
    print(i, 'tensor', kv.dtype, tuple(kv.shape))
  else:
    print(i, type(kv), getattr(kv,'shape',None), getattr(kv,'dtype',None))
print('DONE')
