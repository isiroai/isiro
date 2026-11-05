#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Discover measured KV sizes, probe the needle grid, write JSONL + heatmap.
# Does not start the public A/B harness.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${HERE}/.." && pwd)"
NIAH_PY="${HERE}/needle_haystack.py"
PLOT_PY="${PLOT_PY:-python3}"

DRY_RUN=0
MODEL_SLUG=""
POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      echo "usage: needle phase via benchmarks/run_ab.sh {model} (or this script)" >&2
      echo "  Discovers TIC KV GiB, Baseline same-size start-fail, and default fill." >&2
      echo "  Writes JSONL + heatmap + VRAM figure under the shared run directory." >&2
      exit 0
      ;;
    *) POSITIONAL+=("${arg}") ;;
  esac
done
if [[ "${#POSITIONAL[@]}" -ne 1 ]]; then
  echo "usage: scripts/run_needle_haystack.sh {model} [--dry-run]" >&2
  exit 2
fi
MODEL_SLUG="${POSITIONAL[0]}"
MODEL_DIR="${MODEL_DIR:-${BENCH_ROOT}/${MODEL_SLUG}}"
CONFIG="${ISIRO_BENCH_CONFIG:-${MODEL_DIR}/common.env}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "model directory not found: ${MODEL_DIR}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "missing ${CONFIG}; copy benchmarks/common.env.example to ${MODEL_DIR}/common.env" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONFIG}"

# Same pin as the tok/s harness: product isiro-fa unless ISIRO_BENCH_ISIRO_ATTN.
if [[ -n "${ISIRO_BENCH_ISIRO_ATTN:-}" ]]; then
  export ISIRO_ATTN="${ISIRO_BENCH_ISIRO_ATTN}"
else
  unset ISIRO_ATTN || true
fi

: "${BASELINE_MODEL_DIR:?set BASELINE_MODEL_DIR in common.env}"
: "${TIC_MODEL_DIR:?set TIC_MODEL_DIR in common.env}"
: "${MODEL_ID:?set MODEL_ID in common.env}"
: "${VLLM_IMAGE:?set VLLM_IMAGE in common.env}"
PRECISION="${PRECISION:-bf16}"
API_PORT="${API_PORT:-8000}"
API_URL="http://127.0.0.1:${API_PORT}"
SERVE_GPU_MEMORY_UTILIZATION="${SERVE_GPU_MEMORY_UTILIZATION:-0.90}"
SERVE_TENSOR_PARALLEL_SIZE="${SERVE_TENSOR_PARALLEL_SIZE:-1}"
SERVE_KV_CACHE_DTYPE="${SERVE_KV_CACHE_DTYPE:-}"
ISIRO_BENCH_KV="${ISIRO_BENCH_KV:-${ISIRO_BENCH_KV:-}}"
ISIRO_BENCH_KV="${ISIRO_BENCH_KV}"
case "${ISIRO_BENCH_KV}" in
  ""|fp8|fp4|bf16) ;;
  *)
    echo "ISIRO_BENCH_KV must be empty, fp8, fp4, or bf16 (got ${ISIRO_BENCH_KV})" >&2
    exit 2
    ;;
esac
if [[ -n "${ISIRO_BENCH_KV}" ]]; then
  if [[ -n "${ISIRO_KV_C:-}" && "${ISIRO_KV_C}" != "0" && "${ISIRO_KV_C}" != "off" && "${ISIRO_KV_C}" != "false" ]]; then
    echo "ISIRO_BENCH_KV=${ISIRO_BENCH_KV}: unsetting inherited ISIRO_KV_C=${ISIRO_KV_C}" >&2
    unset ISIRO_KV_C || true
  fi
  if [[ -z "${SERVE_KV_CACHE_DTYPE}" ]]; then
    case "${ISIRO_BENCH_KV}" in
      fp8) SERVE_KV_CACHE_DTYPE="fp8" ;;
      fp4) SERVE_KV_CACHE_DTYPE="nvfp4" ;;
      bf16) SERVE_KV_CACHE_DTYPE="bfloat16" ;;
    esac
  fi
fi
# Probe traffic is one request. Do not overlay max_num_seqs=1 on TIC.
BASELINE_PROBE_SEQS=1
SYSTEM_ID="${SYSTEM_ID:-rtx-5090}"
# Serve window for needle. Do not rewrite this to KV pool tokens.
NIAH_MAX_MODEL_LEN="${NIAH_MAX_MODEL_LEN:-}"
if [[ -z "${NIAH_MAX_MODEL_LEN}" && -n "${BASELINE_MODEL_DIR:-}" && -d "${BASELINE_MODEL_DIR}" ]]; then
  NIAH_MAX_MODEL_LEN="$(
    python3 "${NIAH_PY}" --advertised-max "${BASELINE_MODEL_DIR}"
  )"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Config: ${CONFIG}"
  echo "max_num_seqs=product (shipped serve.yaml; do not overlay 1)"
  echo "max_model_len follows advertised HF config.json (NIAH_MAX_MODEL_LEN overrides), not KV pool tokens"
  python3 "${NIAH_PY}" --plan
  echo "TIC KV is scraped from product serve (model-shaped pool, not a fixed GiB)."
  echo "Needle bins follow NIAH_MAX_MODEL_LEN. Do not crop them to the pool. Do not grow TIC to 1.75."
  echo "Baseline same-size uses the same lock and the same window. Record start-fail only if the serve does not start (not a fixed 4.00)."
  echo "Default fill scrapes GPU KV cache size at gpu_memory_utilization=${SERVE_GPU_MEMORY_UTILIZATION}."
  echo "Serve order: TIC scrape, fill scrape and compare, fill grid, TIC grid, same-size start or start-fail (no extra KV-size grids)."
  echo "TIC ISIRO_ATTN: ${ISIRO_ATTN:-unset (product isiro_fa)}"
  echo "Figures: needle-haystack.png and vram-vs-context.png (TIC product KV + Baseline default fill)"
  exit 0
fi

if ! "${PLOT_PY}" -c "import matplotlib, seaborn, pandas" >/dev/null 2>&1; then
  if [[ -x /tmp/isiro-plot-venv/bin/python ]]; then
    PLOT_PY=/tmp/isiro-plot-venv/bin/python
  else
    echo "heatmap needs matplotlib seaborn pandas (or /tmp/isiro-plot-venv)" >&2
    echo "  python3 -m venv /tmp/isiro-plot-venv && /tmp/isiro-plot-venv/bin/pip install -r ${BENCH_ROOT}/requirements-plot.txt" >&2
    exit 2
  fi
fi

if [[ "${ALLOW_BUSY_GPU:-}" != "1" ]]; then
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)"
  if [[ -n "${apps}" ]]; then
    echo "GPU already has compute processes; refuse NIAH start" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    exit 2
  fi
fi

[[ -d "${BASELINE_MODEL_DIR}" ]] || { echo "baseline model directory not found" >&2; exit 2; }
[[ -d "${TIC_MODEL_DIR}" ]] || { echo "TIC model directory not found" >&2; exit 2; }
if [[ -z "${NIAH_MAX_MODEL_LEN}" ]]; then
  NIAH_MAX_MODEL_LEN="$(
    python3 "${NIAH_PY}" --advertised-max "${BASELINE_MODEL_DIR}"
  )"
fi
if [[ -z "${NIAH_MAX_MODEL_LEN}" ]]; then
  echo "could not read advertised max from ${BASELINE_MODEL_DIR}/config.json; set NIAH_MAX_MODEL_LEN" >&2
  exit 2
fi
STORY_TOKENS="$(
  python3 -c "
import sys
sys.path.insert(0, '${HERE}')
from needle_haystack import STORY_CONTEXT_TOKENS, story_tokens_for_max_len
cap = '${NIAH_MAX_MODEL_LEN}'.strip()
print(story_tokens_for_max_len(int(cap)) if cap else STORY_CONTEXT_TOKENS)
"
)"

export PATH="${HOME}/.isiro/bin:${PATH}"
ISIRO_SERVE_BIN="${ISIRO_SERVE_BIN:-${HOME}/.isiro/bin/isiro}"
if [[ ! -x "${ISIRO_SERVE_BIN}" ]]; then
  ISIRO_SERVE_BIN="$(command -v isiro || true)"
fi
if [[ -z "${ISIRO_SERVE_BIN}" || ! -x "${ISIRO_SERVE_BIN}" ]]; then
  echo "isiro serve binary not found; install customer isiro to ~/.isiro/bin" >&2
  exit 2
fi

if [[ -n "${ISIRO_BENCH_RUN_DIR:-}" ]]; then
  RUN_DIR="${ISIRO_BENCH_RUN_DIR}"
else
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="${BENCH_ROOT}/scratch/${STAMP}-${SYSTEM_ID}-niah"
fi
LOG_DIR="${RUN_DIR}/logs"
TIC_BENCH_BUNDLE="${RUN_DIR}/tic-bundle"
JSONL="${RUN_DIR}/needle_haystack.jsonl"
SCOUT="${RUN_DIR}/scout.jsonl"
PNG="${RUN_DIR}/needle_haystack.png"
VRAM_PNG="${RUN_DIR}/vram-vs-context.png"
BASELINE_CONTAINER="isiro-niah-baseline-${API_PORT}"
TIC_PID=""
mkdir -p "${LOG_DIR}" "${TIC_BENCH_BUNDLE}"
if [[ "${NIAH_SKIP_VANILLA_FILL:-}" == "1" ]]; then
  if [[ ! -s "${JSONL}" ]]; then
    echo "NIAH_SKIP_VANILLA_FILL=1 needs vanilla rows in ${JSONL}" >&2
    exit 2
  fi
else
  : >"${JSONL}"
fi

cleanup() {
  if [[ -n "${TIC_PID}" ]] && kill -0 "${TIC_PID}" 2>/dev/null; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
  fi
  docker rm -f "${BASELINE_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

api_port_listening() {
  ss -ltn "( sport = :${API_PORT} )" 2>/dev/null | grep -qE ":${API_PORT}([^0-9]|$)"
}

ensure_api_port_free() {
  if api_port_listening; then
    docker rm -f "${BASELINE_CONTAINER}" >/dev/null 2>&1 || true
    docker ps -aq --filter "publish=${API_PORT}" | xargs -r docker rm -f >/dev/null 2>&1 || true
    fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  fi
  for _ in $(seq 1 40); do
    if ! api_port_listening; then
      return 0
    fi
    sleep 0.25
  done
  echo "API port ${API_PORT} still busy" >&2
  exit 2
}

wait_ready() {
  local serve_log="${1:-}"
  for _ in $(seq 1 600); do
    if curl -fsS "${API_URL}/version" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${serve_log}" && -f "${serve_log}" ]] &&
       grep -qE 'Engine failed to start|OutOfMemoryError|SERV-0301' "${serve_log}" 2>/dev/null; then
      echo "serve failed during start; see ${serve_log}" >&2
      return 1
    fi
    sleep 2
  done
  echo "server did not become ready at ${API_URL}" >&2
  return 1
}

stop_all() {
  if [[ -n "${TIC_PID}" ]] && kill -0 "${TIC_PID}" 2>/dev/null; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    TIC_PID=""
  fi
  docker rm -f "${BASELINE_CONTAINER}" >/dev/null 2>&1 || true
  ensure_api_port_free
}

scrape_kv_gib() {
  python3 "${NIAH_PY}" --parse-log "$1"
}

scrape_kv_tokens() {
  python3 "${NIAH_PY}" --parse-log-tokens "$1"
}

scrape_non_kv_bytes() {
  python3 "${NIAH_PY}" --parse-log-non-kv "$1"
}

# TIC product serve prints "estimated maximum model length" on a successful
# 32k boot with a small KV pool. Do not treat that line as a start fail.

prepare_tic_bundle() {
  mkdir -p "${TIC_BENCH_BUNDLE}"
  for source in "${TIC_MODEL_DIR}"/*; do
    name="$(basename "${source}")"
    if [[ "${name}" != "serve.yaml" ]]; then
      if [[ -e "${TIC_BENCH_BUNDLE}/${name}" || -L "${TIC_BENCH_BUNDLE}/${name}" ]]; then
        rm -f "${TIC_BENCH_BUNDLE}/${name}"
      fi
      if ! ln "${source}" "${TIC_BENCH_BUNDLE}/${name}" 2>/dev/null; then
        cp -a "${source}" "${TIC_BENCH_BUNDLE}/${name}"
      fi
    fi
  done
  python3 - \
    "${TIC_MODEL_DIR}/serve.yaml" \
    "${TIC_BENCH_BUNDLE}/serve.yaml" \
    "${NIAH_MAX_MODEL_LEN}" \
    "${SERVE_GPU_MEMORY_UTILIZATION}" \
    "${SERVE_TENSOR_PARALLEL_SIZE}" <<'PY'
import re, sys
source, target, max_model_len, gpu_util, tp = sys.argv[1:6]
text = open(source, encoding="utf-8").read()

def upsert(text: str, key: str, value: str) -> str:
    pattern = rf"^(\s*){re.escape(key)}:\s*.*$"
    replacement = rf"\1{key}: {value}"
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count:
        return updated
    anchor = r"^(\s*)gpu_memory_utilization:([^\n]*)$"
    updated, count = re.subn(
        anchor,
        rf"\1gpu_memory_utilization:\2\n\1{key}: {value}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"could not set {key} in serve.yaml overlay")
    return updated

if max_model_len.strip():
    text = upsert(text, "max_model_len", max_model_len)
text = upsert(text, "gpu_memory_utilization", gpu_util)
text = upsert(text, "tensor_parallel_size", tp)
text = upsert(text, "no_enable_prefix_caching", "true")
open(target, "w", encoding="utf-8").write(text)
PY
}

start_tic() {
  local log="$1"
  prepare_tic_bundle
  stop_all
  export ISIRO_SERVE_CUDA_GRAPHS=0
  : "${VLLM_LOGGING_LEVEL:=DEBUG}"
  export VLLM_LOGGING_LEVEL
  unset ISIRO_KV_C || true
  unset ISIRO_KV_PAGE || true
  if [[ -n "${ISIRO_BENCH_KV}" ]]; then
    export ISIRO_KV="${ISIRO_BENCH_KV}"
    export ISIRO_KV_COLD_READY=1
  else
    unset ISIRO_KV || true
  fi
  : >"${log}"
  local args=(
    "${ISIRO_SERVE_BIN}" serve "${TIC_BENCH_BUNDLE}" --target vllm
    --host 0.0.0.0 --port "${API_PORT}"
    --enforce-eager
  )
  if [[ -n "${NIAH_MAX_MODEL_LEN}" ]]; then
    args+=(--max-model-len "${NIAH_MAX_MODEL_LEN}")
  fi
  if [[ -n "${SERVE_KV_CACHE_DTYPE}" && -z "${ISIRO_BENCH_KV}" ]]; then
    args+=(--kv-cache-dtype "${SERVE_KV_CACHE_DTYPE}")
  fi
  "${args[@]}" >"${log}" 2>&1 &
  TIC_PID=$!
  if wait_ready "${log}"; then
    return 0
  fi
  return 1
}

start_baseline() {
  local log="$1"
  local lock_bytes="${2:-}"
  stop_all
  : "${VLLM_LOGGING_LEVEL:=DEBUG}"
  local args=(
    docker run --rm --name "${BASELINE_CONTAINER}" --gpus all
    -p "127.0.0.1:${API_PORT}:8000"
    -v "${BASELINE_MODEL_DIR}:/model:ro"
    -e VLLM_USE_FLASHINFER_SAMPLER=0
    -e "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}"
    "${VLLM_IMAGE}"
    --model /model
    --served-model-name "${MODEL_ID}"
    --dtype bfloat16
    --max-model-len "${NIAH_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${SERVE_GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${SERVE_TENSOR_PARALLEL_SIZE}"
    --max-num-seqs "${BASELINE_PROBE_SEQS}"
    --no-enable-prefix-caching
    --trust-remote-code
    --no-enable-flashinfer-autotune
    --enforce-eager
  )
  if [[ -n "${SERVE_KV_CACHE_DTYPE}" ]]; then
    args+=(--kv-cache-dtype "${SERVE_KV_CACHE_DTYPE}")
  fi
  if [[ -n "${lock_bytes}" ]]; then
    args+=(--kv-cache-memory "${lock_bytes}")
  fi
  : >"${log}"
  "${args[@]}" >"${log}" 2>&1 &
  if wait_ready "${log}"; then
    return 0
  fi
  return 1
}

record_start_fail() {
  local gib="$1"
  python3 "${NIAH_PY}" \
    --record-start-fail \
    --panel vanilla \
    --kv-cache-gib "${gib}" \
    --story-tokens "${STORY_TOKENS}" \
    --jsonl "${JSONL}"
}

probe_full() {
  local panel="$1"
  local gib="$2"
  local log="${3:-}"
  local tokens="0"
  local non_kv="0"
  if [[ -n "${log}" && -f "${log}" ]]; then
    tokens="$(scrape_kv_tokens "${log}")"
    non_kv="$(scrape_non_kv_bytes "${log}")"
  fi
  python3 "${NIAH_PY}" \
    --probe-url "${API_URL}" \
    --model "${MODEL_ID}" \
    --panel "${panel}" \
    --kv-cache-gib "${gib}" \
    --max-model-len "${NIAH_MAX_MODEL_LEN}" \
    --pool-tokens "${tokens:-0}" \
    --non-kv-bytes "${non_kv:-0}" \
    --gpu-memory-utilization "${SERVE_GPU_MEMORY_UTILIZATION}" \
    --jsonl "${JSONL}"
}

probe_story() {
  local panel="$1"
  local gib="$2"
  rm -f "${SCOUT}"
  python3 "${NIAH_PY}" \
    --probe-url "${API_URL}" \
    --model "${MODEL_ID}" \
    --panel "${panel}" \
    --kv-cache-gib "${gib}" \
    --story-only \
    --story-tokens "${STORY_TOKENS}" \
    --max-model-len "${NIAH_MAX_MODEL_LEN}" \
    --jsonl "${SCOUT}"
}

story_hit() {
  python3 - "${SCOUT}" "$1" "${STORY_TOKENS}" <<'PY'
import json, sys
path, size, story = sys.argv[1], round(float(sys.argv[2]), 4), int(sys.argv[3])
hit = False
for line in open(path, encoding="utf-8"):
    if not line.strip():
        continue
    item = json.loads(line)
    if str(item.get("panel") or "").lower() not in ("vanilla", "baseline"):
        continue
    if round(float(item.get("kv_cache_gib") or 0), 4) != size:
        continue
    if int(item.get("context_tokens") or 0) != story:
        continue
    if item.get("hit"):
        hit = True
        break
raise SystemExit(0 if hit else 1)
PY
}

jsonl_story_hit() {
  python3 - "${JSONL}" "$1" "${STORY_TOKENS}" <<'PY'
import json, sys
path, size, story = sys.argv[1], round(float(sys.argv[2]), 4), int(sys.argv[3])
hit = False
for line in open(path, encoding="utf-8"):
    if not line.strip():
        continue
    item = json.loads(line)
    if str(item.get("panel") or "").lower() not in ("vanilla", "baseline"):
        continue
    if round(float(item.get("kv_cache_gib") or 0), 4) != size:
        continue
    if int(item.get("context_tokens") or 0) != story:
        continue
    if item.get("hit"):
        hit = True
        break
raise SystemExit(0 if hit else 1)
PY
}

require_tic_smaller_than_fill() {
  # ISIRO_KV compresses cold page bytes; the reserved vendor pool can grow
  # when TIC weights free HBM. Do not refuse that case.
  if [[ -n "${ISIRO_BENCH_KV}" ]]; then
    echo "KV compress A/B (ISIRO_KV=${ISIRO_BENCH_KV}): skip TIC-pool-smaller-than-fill gate (TIC=${1} GiB fill=${2} GiB)" >&2
    return 0
  fi
  python3 -c "
import sys
sys.path.insert(0, '${HERE}')
from needle_haystack import tic_pool_smaller_than_fill
tic, fill = float('${1}'), float('${2}')
if not tic_pool_smaller_than_fill(tic, fill):
    raise SystemExit(
        f'TIC KV cache {tic:.2f} GiB is not smaller than Baseline fill {fill:.2f} GiB. Stopped.'
    )
"
}

echo "NIAH run: ${RUN_DIR}" >&2
echo "max_num_seqs=product max_model_len=${NIAH_MAX_MODEL_LEN:-serve.yaml} (not KV pool tokens)" >&2

TIC_LOG="${LOG_DIR}/tic-serve.log"
if ! start_tic "${TIC_LOG}"; then
  echo "TIC did not start at max_model_len=${NIAH_MAX_MODEL_LEN:-serve.yaml}; see ${TIC_LOG}" >&2
  exit 2
fi
TIC_GIB="$(scrape_kv_gib "${TIC_LOG}")"
if [[ -z "${NIAH_MAX_MODEL_LEN}" ]]; then
  echo "could not read advertised max from ${BASELINE_MODEL_DIR}/config.json; set NIAH_MAX_MODEL_LEN" >&2
  exit 2
fi
echo "TIC KV cache: ${TIC_GIB} GiB (scraped) window=${NIAH_MAX_MODEL_LEN}" >&2
stop_all

FILL_LOG="${LOG_DIR}/baseline-fill-serve.log"
if [[ "${NIAH_SKIP_VANILLA_FILL:-}" == "1" ]]; then
  FILL_GIB="$(
    python3 - "${JSONL}" <<'PY'
import json, sys
gibs = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    if str(row.get("panel") or "").lower() in ("vanilla", "baseline") and not row.get("start_fail"):
        if "kv_cache_gib" in row:
            gibs.append(float(row["kv_cache_gib"]))
if not gibs:
    raise SystemExit("NIAH_SKIP_VANILLA_FILL=1 needs vanilla rows in the JSONL")
print(f"{max(gibs):.4f}")
PY
  )"
  echo "Baseline default fill: ${FILL_GIB} GiB (reused JSONL; skip fill serve and grid)" >&2
  require_tic_smaller_than_fill "${TIC_GIB}" "${FILL_GIB}"
else
  if ! start_baseline "${FILL_LOG}"; then
    echo "Baseline default fill did not start; see ${FILL_LOG}" >&2
    exit 2
  fi
  FILL_GIB="$(scrape_kv_gib "${FILL_LOG}")"
  echo "Baseline default fill: ${FILL_GIB} GiB (scraped)" >&2
  require_tic_smaller_than_fill "${TIC_GIB}" "${FILL_GIB}"
  probe_full vanilla "${FILL_GIB}" "${FILL_LOG}"
  stop_all
fi

start_tic "${TIC_LOG}"
probe_full tic "${TIC_GIB}" "${TIC_LOG}"
stop_all

TIC_BYTES="$(python3 -c "import sys; sys.path.insert(0, '${HERE}'); from needle_haystack import gib_to_bytes; print(gib_to_bytes(float('${TIC_GIB}')))")"
SAME_LOG="${LOG_DIR}/baseline-same-serve.log"
if start_baseline "${SAME_LOG}" "${TIC_BYTES}"; then
  SAME_GIB="$(scrape_kv_gib "${SAME_LOG}")"
  echo "Baseline same size as TIC started at ${SAME_GIB} GiB; no extra grid" >&2
  stop_all
else
  SAME_GIB="${TIC_GIB}"
  echo "Baseline would not start at ${TIC_GIB} GiB (same size as TIC)" >&2
  record_start_fail "${TIC_GIB}"
  stop_all
fi

"${PLOT_PY}" "${NIAH_PY}" --jsonl "${JSONL}" --out "${PNG}" --vram-out "${VRAM_PNG}"
python3 - "${RUN_DIR}/niah_sizes.json" "${TIC_GIB}" "${SAME_GIB}" "${FILL_GIB}" "${NIAH_MAX_MODEL_LEN}" <<'PY'
import json, sys
path, tic, same, fill, max_len = sys.argv[1:6]
json.dump(
    {
        "tic_gib": float(tic),
        "baseline_same_gib": float(same),
        "baseline_fill_gib": float(fill),
        "max_num_seqs": "product",
        "max_model_len": int(max_len),
    },
    open(path, "w", encoding="utf-8"),
    indent=2,
    sort_keys=True,
)
print(path)
PY
echo "JSONL ${JSONL}" >&2
echo "heatmap ${PNG}" >&2
echo "vram ${VRAM_PNG}" >&2
echo "sizes ${RUN_DIR}/niah_sizes.json" >&2
echo "needle phase done" >&2
