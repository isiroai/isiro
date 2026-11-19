#!/usr/bin/env bash
# TIC + ISIRO_GDX=fp8 (path_a_mla / o1-mla-latent) graphs-on (no enforce-eager)
set -euo pipefail
MODEL="${MODEL:-/root/models/llm/fp4/GLM-5.3-Flash-NVFP4-TIC}"
LOG="${LOG:-/workspace/logs/isiro_serve_gdx_fp8_graphs.log}"
mkdir -p "$(dirname "$LOG")"
unset ISIRO_ATTN ISIRO_GDX_C ISIRO_HISTORY_OWNER ISIRO_SERVE_ENFORCE_EAGER || true
export ISIRO_HOME="${ISIRO_HOME:-/root/.isiro}"
export PATH="/root/bin:${ISIRO_HOME}/bin:${PATH}"
export ISIRO_GDX=fp8
export ISIRO_GDX_FP8_KV_SCALES_OK=1
export ISIRO_GDX_METER=1
export ISIRO_GDX_LIVE_WINDOW="${ISIRO_GDX_LIVE_WINDOW:-512}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export PYTHONFAULTHANDLER=1
echo "[$(date -u +%FT%TZ)] ISIRO TIC GDX=fp8 path_a_mla graphs-on -> $LOG" | tee "$LOG"
exec /root/bin/isiro-serve-native "$MODEL" --target vllm >>"$LOG" 2>&1
