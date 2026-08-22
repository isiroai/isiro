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
      echo "  Discovers TIC KV GiB, Baseline same size, first recall, and default fill." >&2
      echo "  Writes JSONL + heatmap under the shared run directory." >&2
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

: "${BASELINE_MODEL_DIR:?set BASELINE_MODEL_DIR in common.env}"
: "${TIC_MODEL_DIR:?set TIC_MODEL_DIR in common.env}"
: "${MODEL_ID:?set MODEL_ID in common.env}"
: "${VLLM_IMAGE:?set VLLM_IMAGE in common.env}"
PRECISION="${PRECISION:-bf16}"
API_PORT="${API_PORT:-8000}"
API_URL="http://127.0.0.1:${API_PORT}"
SERVE_GPU_MEMORY_UTILIZATION="${SERVE_GPU_MEMORY_UTILIZATION:-0.90}"
SERVE_TENSOR_PARALLEL_SIZE="${SERVE_TENSOR_PARALLEL_SIZE:-1}"
# NIAH owns concurrency. Length follows the bundle unless overridden.
NIAH_MAX_NUM_SEQS=1
SYSTEM_ID="${SYSTEM_ID:-rtx-5090}"
if [[ -z "${NIAH_MAX_MODEL_LEN:-}" && -f "${TIC_MODEL_DIR}/serve.yaml" ]]; then
  NIAH_MAX_MODEL_LEN="$(
    python3 - "${TIC_MODEL_DIR}/serve.yaml" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"^max_model_len:\s*([0-9]+)", text, re.M)
print(m.group(1) if m else "")
PY
  )"
fi
NIAH_MAX_MODEL_LEN="${NIAH_MAX_MODEL_LEN:-32768}"
STORY_TOKENS="$(
  python3 -c "
import sys
sys.path.insert(0, '${HERE}')
from needle_haystack import story_tokens_for_max_len
print(story_tokens_for_max_len(int('${NIAH_MAX_MODEL_LEN}')))
"
)"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Config: ${CONFIG}"
  echo "max_num_seqs=${NIAH_MAX_NUM_SEQS}"
  echo "max_model_len=${NIAH_MAX_MODEL_LEN}"
  python3 "${NIAH_PY}" --plan
  echo "TIC KV is scraped from product serve (model-shaped pool, not a fixed GiB)."
  echo "Baseline same-size and first-recall locks use that scrape (not a fixed 4.00)."
  echo "Default fill scrapes GPU KV cache size at gpu_memory_utilization=${SERVE_GPU_MEMORY_UTILIZATION}."
  echo "Serve order: TIC scrape, fill scrape and compare, fill grid, TIC grid, same-size grid, double-until-recall (full grid on hit only)."
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
BASELINE_CONTAINER="isiro-niah-baseline-${API_PORT}"
TIC_PID=""
mkdir -p "${LOG_DIR}" "${TIC_BENCH_BUNDLE}"
: >"${JSONL}"

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
    "${SERVE_TENSOR_PARALLEL_SIZE}" \
    "${NIAH_MAX_NUM_SEQS}" <<'PY'
import re, sys
source, target, max_model_len, gpu_util, tp, max_seqs = sys.argv[1:7]
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

text = upsert(text, "max_model_len", max_model_len)
text = upsert(text, "gpu_memory_utilization", gpu_util)
text = upsert(text, "tensor_parallel_size", tp)
text = upsert(text, "max_num_seqs", max_seqs)
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
  : >"${log}"
  "${ISIRO_SERVE_BIN}" serve "${TIC_BENCH_BUNDLE}" --target vllm \
    --host 0.0.0.0 --port "${API_PORT}" \
    --max-model-len "${NIAH_MAX_MODEL_LEN}" \
    >"${log}" 2>&1 &
  TIC_PID=$!
  wait_ready "${log}"
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
    --max-num-seqs "${NIAH_MAX_NUM_SEQS}"
    --no-enable-prefix-caching
    --trust-remote-code
    --no-enable-flashinfer-autotune
    --enforce-eager
  )
  if [[ -n "${lock_bytes}" ]]; then
    args+=(--kv-cache-memory "${lock_bytes}")
  fi
  : >"${log}"
  "${args[@]}" >"${log}" 2>&1 &
  wait_ready "${log}"
}

probe_full() {
  local panel="$1"
  local gib="$2"
  python3 "${NIAH_PY}" \
    --probe-url "${API_URL}" \
    --model "${MODEL_ID}" \
    --panel "${panel}" \
    --kv-cache-gib "${gib}" \
    --max-model-len "${NIAH_MAX_MODEL_LEN}" \
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
echo "max_num_seqs=${NIAH_MAX_NUM_SEQS} max_model_len=${NIAH_MAX_MODEL_LEN}" >&2

TIC_LOG="${LOG_DIR}/tic-serve.log"
start_tic "${TIC_LOG}"
TIC_GIB="$(scrape_kv_gib "${TIC_LOG}")"
echo "TIC KV cache: ${TIC_GIB} GiB (scraped)" >&2
stop_all

FILL_LOG="${LOG_DIR}/baseline-fill-serve.log"
start_baseline "${FILL_LOG}"
FILL_GIB="$(scrape_kv_gib "${FILL_LOG}")"
echo "Baseline default fill: ${FILL_GIB} GiB (scraped)" >&2
require_tic_smaller_than_fill "${TIC_GIB}" "${FILL_GIB}"
probe_full vanilla "${FILL_GIB}"
stop_all

start_tic "${TIC_LOG}"
probe_full tic "${TIC_GIB}"
stop_all

TIC_BYTES="$(python3 -c "import sys; sys.path.insert(0, '${HERE}'); from needle_haystack import gib_to_bytes; print(gib_to_bytes(float('${TIC_GIB}')))")"
SAME_LOG="${LOG_DIR}/baseline-same-serve.log"
start_baseline "${SAME_LOG}" "${TIC_BYTES}"
SAME_GIB="$(scrape_kv_gib "${SAME_LOG}")"
echo "Baseline same size as TIC: ${SAME_GIB} GiB (scraped)" >&2
probe_full vanilla "${SAME_GIB}"
if jsonl_story_hit "${SAME_GIB}"; then
  FIRST_RECALL_GIB="${SAME_GIB}"
  echo "First recall is same size as TIC ${SAME_GIB} GiB" >&2
  stop_all
else
  stop_all
  FIRST_RECALL_GIB=""
  mapfile -t CANDIDATES < <(
    python3 -c "
import sys
sys.path.insert(0, '${HERE}')
from needle_haystack import first_recall_candidates
for x in first_recall_candidates(float('${TIC_GIB}'), cap=float('${FILL_GIB}')):
    print(f'{x:.2f}')
"
  )
  for cand in "${CANDIDATES[@]}"; do
    CAND_BYTES="$(python3 -c "import sys; sys.path.insert(0, '${HERE}'); from needle_haystack import gib_to_bytes; print(gib_to_bytes(float('${cand}')))")"
    CAND_LOG="${LOG_DIR}/baseline-sweep-${cand}.log"
    echo "Baseline sweep lock ${cand} GiB" >&2
    start_baseline "${CAND_LOG}" "${CAND_BYTES}"
    MEAS="$(scrape_kv_gib "${CAND_LOG}")"
    probe_story vanilla "${MEAS}"
    if story_hit "${MEAS}"; then
      FIRST_RECALL_GIB="${MEAS}"
      echo "First recall: ${FIRST_RECALL_GIB} GiB (scraped)" >&2
      probe_full vanilla "${MEAS}"
      stop_all
      break
    fi
    stop_all
  done
  if [[ -z "${FIRST_RECALL_GIB}" ]]; then
    FIRST_RECALL_GIB="${FILL_GIB}"
    echo "Sweep did not hit; first recall is default fill ${FILL_GIB} GiB" >&2
  fi
fi

"${PLOT_PY}" "${NIAH_PY}" --jsonl "${JSONL}" --out "${PNG}"
python3 - "${RUN_DIR}/niah_sizes.json" "${TIC_GIB}" "${SAME_GIB}" "${FIRST_RECALL_GIB}" "${FILL_GIB}" "${NIAH_MAX_MODEL_LEN}" <<'PY'
import json, sys
path, tic, same, first, fill, max_len = sys.argv[1:7]
json.dump(
    {
        "tic_gib": float(tic),
        "baseline_same_gib": float(same),
        "baseline_first_recall_gib": float(first),
        "baseline_fill_gib": float(fill),
        "max_num_seqs": 1,
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
echo "sizes ${RUN_DIR}/niah_sizes.json" >&2
echo "needle phase done" >&2
