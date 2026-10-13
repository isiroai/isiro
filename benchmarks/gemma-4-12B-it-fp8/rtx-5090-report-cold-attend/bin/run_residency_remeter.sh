#!/usr/bin/env bash
# FA path_a cold-attend residency scrape on live Isiro serve (port 8000).
# Writes logs/residency_scrape.json under this report dir.
#
# Requires REMETER_PATH_A (path to remeter_path_a_vs_vanilla.py) and a Python
# with the runtime + deps (ISIRO_BENCH_PYTHON).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HERE}/logs/residency_scrape.json"
LOG="${HERE}/logs/isiro_serve_residency.log"
REMETER="${REMETER_PATH_A:?set REMETER_PATH_A to remeter_path_a_vs_vanilla.py}"
PY="${ISIRO_BENCH_PYTHON:?set ISIRO_BENCH_PYTHON}"

mkdir -p "${HERE}/logs"
DOCKER_NAME="$(docker ps --format '{{.Names}}' | grep -E 'isiro-serve|vllm' | head -1 || true)"
EXTRA=()
if [[ -n "${DOCKER_NAME}" ]]; then
  EXTRA+=(--docker-name "${DOCKER_NAME}")
fi
if [[ -f "${LOG}" ]]; then
  EXTRA+=(--log "${LOG}")
fi

exec "${PY}" "${REMETER}" \
  --url "${ISIRO_BENCH_URL:-http://127.0.0.1:8000}" \
  --arm gdx \
  --out "${OUT}" \
  --gdx-kv bf16 \
  --fused 0 \
  --live-window "${ISIRO_GDX_LIVE_WINDOW:-512}" \
  --kv-cache-dtype bfloat16 \
  --max-model-len 8192 \
  "${EXTRA[@]}"
