#!/usr/bin/env bash
# Vanilla twin matched to TIC serve.yaml remeter knobs (path_a_mla A/B).
set -euo pipefail
MODEL="${MODEL:-/root/models/llm/fp4/GLM-5.3-Flash-NVFP4}"
LOG="${LOG:-/workspace/logs/vanilla_serve_matched.log}"
mkdir -p "$(dirname "$LOG")"
unset ISIRO_ATTN ISIRO_GDX ISIRO_GDX_KV ISIRO_GDX_C ISIRO_HISTORY_OWNER ISIRO_GDX_COLD_READY || true
unset ISIRO_HOME PYTHONPATH || true
export PATH="/usr/local/bin:/usr/bin:${PATH}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export PYTHONFAULTHANDLER=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
echo "[$(date -u +%FT%TZ)] VANILLA matched serve.yaml knobs tp=4 seqs=10 util=0.95 -> $LOG" | tee "$LOG"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 10 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --no-enable-flashinfer-autotune \
  --no-enable-prefix-caching \
  >>"$LOG" 2>&1
