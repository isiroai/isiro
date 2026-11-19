from pathlib import Path
import vllm
root=Path(vllm.__file__).parent
# find block_tables usage in gpu model runner
for rel in [
 'v1/worker/gpu_model_runner.py',
 'v1/worker/gpu/block_table.py',
 'v1/core/kv_cache_manager.py',
]:
  p=root/rel
  if not p.exists():
    # search
    continue
  print('FILE', rel, 'exists', p.stat().st_size)
hits=[]
for p in root.rglob('*block_table*'):
  hits.append(str(p.relative_to(root)))
print('block_table files', hits[:30])
# grep hybrid page size
for p in root.rglob('*.py'):
  try:t=p.read_text(errors='ignore')
  except: continue
  if 'attention block size' in t or 'mamba page size' in t:
    print('HIT', p.relative_to(root))
    for i,l in enumerate(t.splitlines()):
      if 'mamba page' in l.lower() or 'attention block size' in l.lower() or 'page size' in l and 'attention' in l.lower():
        print(f'  {i+1}:{l[:160]}')
