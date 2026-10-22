#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Expect MODEL_DIR (benchmarks/{model}) and MODEL_SLUG from launch_ab / run_harness.
if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "MODEL_DIR must be set before sourcing harness_lib.sh" >&2
  exit 2
fi
HERE="${MODEL_DIR}"
BENCH_ROOT="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${BENCH_ROOT}/.." && pwd)"
CONFIG="${ISIRO_BENCH_CONFIG:-${HERE}/common.env}"
SCRATCH_ROOT="${BENCH_ROOT}/scratch"

DRY_RUN=0
REUSE_BASELINE=0
REUSE_BASELINE_EXPLICIT=0
NO_REUSE_BASELINE=0
# Public default: outer CUDA graphs ON (lean FULL_DECODE_ONLY on both baseline
# and TIC). The lean TIC pool is ~free on KV/HBM. --eager opts out (matched
# enforce-eager A/B).
GRAPHS_ON=1
EAGER_EXPLICIT=0
PROFILES_CSV=""
# Public default: capacity-primary at a fixed operating point.
EXPERIMENT_KIND="capacity"
POSITIONAL=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --reuse-baseline)
      REUSE_BASELINE=1
      REUSE_BASELINE_EXPLICIT=1
      ;;
    --no-reuse-baseline) NO_REUSE_BASELINE=1 ;;
    --graphs|--graph-on) GRAPHS_ON=1 ;;
    --enforce-eager|--eager|--graph-off)
      GRAPHS_ON=0
      EAGER_EXPLICIT=1
      ;;
    --mode=capacity) EXPERIMENT_KIND="capacity" ;;
    --mode=equal-batch) EXPERIMENT_KIND="equal-batch" ;;
    --mode=*)
      echo "unknown --mode (use capacity or equal-batch): ${arg}" >&2
      exit 2
      ;;
    --profiles=*)
      PROFILES_CSV="${arg#--profiles=}"
      ;;
    --profiles)
      echo "usage: --profiles=name[,name...] (for example --profiles=generation-32-256)" >&2
      exit 2
      ;;
    -h|--help)
      echo "usage: benchmarks/run_ab.sh {model} [--dry-run] [--enforce-eager] [--no-reuse-baseline] [--profiles=...]" >&2
      echo "  ISIRO_BENCH_CONFIG=path/to/common.env  (default: ${HERE}/common.env)" >&2
      echo "  Public path: capacity-primary A/B; Graph ON by default." >&2
      echo "  --enforce-eager: full eager A/B (same meaning as vLLM). --graph-off is an alias." >&2
      echo "  --profiles=generation-32-256: default." >&2
      exit 0
      ;;
    *)
      POSITIONAL+=("${arg}")
      ;;
  esac
done
if [[ "${#POSITIONAL[@]}" -gt 0 ]]; then
  echo "usage: benchmarks/run_ab.sh {model} [--dry-run] [--no-reuse-baseline] [--profiles=...]" >&2
  exit 2
fi
if [[ "${ISIRO_BENCH_BASELINE_CACHE:-0}" == "1" ]]; then
  REUSE_BASELINE=1
  REUSE_BASELINE_EXPLICIT=1
fi
# Capacity cold-starts unless explicitly requested
# (--reuse-baseline / ISIRO_BENCH_BASELINE_CACHE=1).
if [[ "${EXPERIMENT_KIND}" == "capacity" && "${REUSE_BASELINE_EXPLICIT}" -eq 0 ]]; then
  REUSE_BASELINE=0
fi
if [[ "${NO_REUSE_BASELINE}" -eq 1 ]]; then
  if [[ "${REUSE_BASELINE_EXPLICIT}" -eq 1 ]]; then
    echo "pass only one of --reuse-baseline and --no-reuse-baseline" >&2
    exit 2
  fi
  REUSE_BASELINE=0
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing ${CONFIG}; copy benchmarks/common.env.example to ${HERE}/common.env and set model paths" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG}"

# Public TIC vs vLLM must not inherit a parent ISIRO_ATTN=0 (that skips the
# product CUSTOM drop-in). Pin only via ISIRO_BENCH_ISIRO_ATTN.
if [[ -n "${ISIRO_BENCH_ISIRO_ATTN:-}" ]]; then
  export ISIRO_ATTN="${ISIRO_BENCH_ISIRO_ATTN}"
else
  unset ISIRO_ATTN || true
fi

: "${MODEL_SLUG:?set MODEL_SLUG (launch_ab.py sets it to the model directory name)}"
: "${BASELINE_MODEL_DIR:?set BASELINE_MODEL_DIR in common.env}"
: "${TIC_MODEL_DIR:?set TIC_MODEL_DIR in common.env}"
: "${MODEL_ID:?set MODEL_ID in common.env}"
: "${VLLM_IMAGE:?set VLLM_IMAGE in common.env}"
# Local tokenizer for vllm bench (gated Hub ids like meta-llama/*).
: "${BENCH_TOKENIZER:=${BASELINE_MODEL_DIR}}"
BENCH_TOKENIZER_ARGS=()
if [[ -d "${BENCH_TOKENIZER}" ]]; then
  BENCH_TOKENIZER_ARGS=(--tokenizer "${BENCH_TOKENIZER}")
fi
# Serve knobs (defaults match benchmarks/common.env.example).
: "${SERVE_MAX_MODEL_LEN:=4096}"
: "${SERVE_GPU_MEMORY_UTILIZATION:=0.90}"
: "${SERVE_TENSOR_PARALLEL_SIZE:=1}"
SERVE_MAX_NUM_SEQS="${SERVE_MAX_NUM_SEQS:-32}"
: "${SERVE_PREFIX_CACHING:=false}"
: "${SERVE_ENFORCE_EAGER:=false}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
# Stale capacity exports (e.g. BENCH_MAX_CONCURRENCY=40 with seqs=32) overload
# equal-batch (load_errors, bogus TTFT). Never admit more than max_num_seqs.
if [[ "${BENCH_MAX_CONCURRENCY}" -gt "${SERVE_MAX_NUM_SEQS}" ]]; then
  echo "clamp BENCH_MAX_CONCURRENCY ${BENCH_MAX_CONCURRENCY} -> ${SERVE_MAX_NUM_SEQS} (cannot exceed max_num_seqs)" >&2
  BENCH_MAX_CONCURRENCY="${SERVE_MAX_NUM_SEQS}"
fi
PRECISION="${PRECISION:-bf16}"
case "${PRECISION}" in
  bf16|fp8|awq) ;;
  *)
    echo "PRECISION must be bf16, fp8, or awq (got ${PRECISION})" >&2
    exit 2
    ;;
esac
# Opt-in KV alphabet for matched A/B (empty = leave vendor default / ISIRO stamp).
SERVE_KV_CACHE_DTYPE="${SERVE_KV_CACHE_DTYPE:-}"
# Opt-in ISIRO KV compress on the TIC arm only (fp8|fp4|bf16). Empty = off.
# Prefer ISIRO_BENCH_GDX; legacy ISIRO_BENCH_GDX_KV is an alphabet alias.
ISIRO_BENCH_GDX="${ISIRO_BENCH_GDX:-${ISIRO_BENCH_GDX_KV:-}}"
ISIRO_BENCH_GDX_KV="${ISIRO_BENCH_GDX}"
case "${ISIRO_BENCH_GDX}" in
  ""|fp8|fp4|bf16) ;;
  *)
    echo "ISIRO_BENCH_GDX must be empty, fp8, fp4, or bf16 (got ${ISIRO_BENCH_GDX})" >&2
    exit 2
    ;;
esac
if [[ -n "${ISIRO_BENCH_GDX}" ]]; then
  # Product GDX auto-routes; drop stale unique-C / path-A parent envs.
  if [[ -n "${ISIRO_GDX_C:-}" && "${ISIRO_GDX_C}" != "0" && "${ISIRO_GDX_C}" != "off" && "${ISIRO_GDX_C}" != "false" ]]; then
    echo "ISIRO_BENCH_GDX=${ISIRO_BENCH_GDX}: unsetting inherited ISIRO_GDX_C=${ISIRO_GDX_C}" >&2
    unset ISIRO_GDX_C || true
  fi
  if [[ -z "${SERVE_KV_CACHE_DTYPE}" ]]; then
    case "${ISIRO_BENCH_GDX}" in
      fp8) SERVE_KV_CACHE_DTYPE="fp8" ;;
      fp4) SERVE_KV_CACHE_DTYPE="nvfp4" ;;
      bf16) SERVE_KV_CACHE_DTYPE="bfloat16" ;;
    esac
  fi
fi

# Compiler from the .tic header; runtime from `isiro --help`.
# Set ISIRO_FORMAT / ISIRO_RUNTIME to a semver to pin; auto (default) probes.
_TIC_FOR_VERSION="${TIC_MODEL_DIR}/model.tic"
if [[ ! -f "${_TIC_FOR_VERSION}" ]]; then
  _TIC_FOR_VERSION="$(find "${TIC_MODEL_DIR}" -maxdepth 1 -name '*.tic' -print -quit 2>/dev/null || true)"
fi
eval "$(
  python3 "${BENCH_ROOT}/scripts/probe_isiro_versions.py" \
    --tic "${_TIC_FOR_VERSION}" \
    --format-override "${ISIRO_FORMAT:-auto}" \
    --runtime-override "${ISIRO_RUNTIME:-auto}"
)"
echo "versions: compiler=${ISIRO_FORMAT} runtime=${ISIRO_RUNTIME}" >&2

# Precision confusion guard: the config file basename encodes the intended
# precision (common.env => bf16, common.env.fp8 => fp8). A stale
# ISIRO_BENCH_CONFIG in the shell has silently routed "bf16" runs to the FP8
# config before; refuse to proceed when the resolved config and PRECISION
# disagree so reports can never be mislabeled.
CONFIG_BASENAME="$(basename "${CONFIG}")"
case "${CONFIG_BASENAME}" in
  *.fp8) CONFIG_PRECISION="fp8" ;;
  *.awq) CONFIG_PRECISION="awq" ;;
  *)     CONFIG_PRECISION="bf16" ;;
esac
if [[ "${CONFIG_PRECISION}" != "${PRECISION}" ]]; then
  echo "precision mismatch: config '${CONFIG_BASENAME}' implies ${CONFIG_PRECISION} but PRECISION=${PRECISION}" >&2
  echo "  resolved CONFIG=${CONFIG}" >&2
  echo "  set ISIRO_BENCH_CONFIG to the matching file (or unset it for bf16), or set PRECISION accordingly" >&2
  exit 2
fi

# Outer CUDA-graph mode. Public default is graphs (GRAPHS_ON=1). --eager opts
# out to matched enforce-eager on both baseline and TIC (matched A/B).
if [[ "${GRAPHS_ON}" -eq 1 ]]; then
  SERVE_ENFORCE_EAGER="false"
else
  SERVE_ENFORCE_EAGER="true"
fi
case "${SERVE_ENFORCE_EAGER}" in
  true|false) ;;
  *)
    echo "SERVE_ENFORCE_EAGER must be true or false (got ${SERVE_ENFORCE_EAGER})" >&2
    exit 2
    ;;
esac
if [[ "${SERVE_ENFORCE_EAGER}" == "true" ]]; then
  GRAPH_MODE="eager"
else
  GRAPH_MODE="graphs"
fi

# Loud, unambiguous echo of the resolved A/B identity so a mis-set config is
# obvious at a glance in logs and terminals.
echo "=== isiro bench config ===" >&2
echo "  CONFIG            = ${CONFIG}" >&2
echo "  PRECISION         = ${PRECISION}" >&2
echo "  BASELINE_MODEL_DIR= ${BASELINE_MODEL_DIR}" >&2
echo "  TIC_MODEL_DIR     = ${TIC_MODEL_DIR}" >&2
echo "  MODEL_ID          = ${MODEL_ID}" >&2
echo "  GRAPH_MODE        = ${GRAPH_MODE} (enforce_eager=${SERVE_ENFORCE_EAGER})" >&2
echo "  EXPERIMENT_KIND   = ${EXPERIMENT_KIND}" >&2
echo "  SERVE_KV_CACHE_DTYPE = ${SERVE_KV_CACHE_DTYPE:-unset}" >&2
echo "  ISIRO_BENCH_GDX      = ${ISIRO_BENCH_GDX:-off}" >&2
# Isiro-first: timed Isiro arm before cold baseline (GDX publish default on).
ISIRO_FIRST=0
_ISIRO_FIRST_RAW="${ISIRO_BENCH_ISIRO_FIRST-}"
if [[ -z "${_ISIRO_FIRST_RAW}" && -n "${ISIRO_BENCH_GDX}" ]]; then
  _ISIRO_FIRST_RAW=1
fi
case "${_ISIRO_FIRST_RAW}" in
  1|true|yes|on) ISIRO_FIRST=1 ;;
esac
echo "  ISIRO_FIRST         = ${ISIRO_FIRST}" >&2
# Eng remeter: serve+bench only (skip verify / GDX cold meters / baseline).
_bench_env_truthy() {
  case "${1:-}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac
}
SKIP_VERIFY=0
SKIP_GDX_METERS=0
SKIP_GDX_CURVE=0
SKIP_BASELINE=0
if _bench_env_truthy "${ISIRO_BENCH_SKIP_VERIFY:-}"; then SKIP_VERIFY=1; fi
if _bench_env_truthy "${ISIRO_BENCH_SKIP_GDX_METERS:-}"; then SKIP_GDX_METERS=1; fi
if _bench_env_truthy "${ISIRO_BENCH_SKIP_GDX_CURVE:-}"; then SKIP_GDX_CURVE=1; fi
if _bench_env_truthy "${ISIRO_BENCH_SKIP_BASELINE:-}"; then SKIP_BASELINE=1; fi

# GDX cold meters use host Python (serve stays in pinned VLLM_IMAGE). Remotes
# drift on transformers/kernels; bootstrap a managed venv + fail-closed
# preflight before verify/serve so a missing kernels pin cannot burn a run.
if [[ -n "${ISIRO_BENCH_GDX}" && "${SKIP_GDX_METERS}" -eq 0 ]]; then
  echo "gdx_meter_env: ensuring pinned host meter python" >&2
  _gdx_ensure_args=()
  if [[ -n "${ISIRO_BENCH_PYTHON:-}" && -x "${ISIRO_BENCH_PYTHON}" ]]; then
    _gdx_ensure_args+=(--python "${ISIRO_BENCH_PYTHON}")
  fi
  if [[ -n "${ISIRO_BENCH_VENV:-}" ]]; then
    _gdx_ensure_args+=(--venv "${ISIRO_BENCH_VENV}")
  fi
  if [[ -n "${ISIRO_BENCH_TORCH_INDEX:-}" ]]; then
    _gdx_ensure_args+=(--torch-index-url "${ISIRO_BENCH_TORCH_INDEX}")
  fi
  ISIRO_BENCH_PYTHON="$(
    python3 "${BENCH_ROOT}/scripts/ensure_gdx_meter_env.py" \
      --requirements "${BENCH_ROOT}/requirements-gdx-meters.txt" \
      --print-python \
      "${_gdx_ensure_args[@]}"
  )" || {
    echo "gdx_meter_env: FAIL (fix host meter deps before remeter)" >&2
    echo "  pin file: ${BENCH_ROOT}/requirements-gdx-meters.txt" >&2
    echo "  managed venv: \${ISIRO_BENCH_VENV:-~/.isiro/bench-venv}" >&2
    exit 2
  }
  export ISIRO_BENCH_PYTHON
  echo "  ISIRO_BENCH_PYTHON = ${ISIRO_BENCH_PYTHON}" >&2
  unset _gdx_ensure_args
fi
# PROFILES / BASELINE_REUSE are printed after counts resolve.

# Matched A/B requires these knobs; refuse silent drift.
if [[ "${SERVE_PREFIX_CACHING}" != "false" ]]; then
  echo "SERVE_PREFIX_CACHING must be false for matched TTFT sweeps" >&2
  exit 2
fi

export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -n "${ISIRO_BENCH_RUN_DIR:-}" ]]; then
  RUN_DIR="${ISIRO_BENCH_RUN_DIR}"
  RUN_ID="$(basename "${RUN_DIR}")"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SYSTEM_ID}"
  RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
fi
RAW_DIR="${RUN_DIR}/raw"
LOG_DIR="${RUN_DIR}/logs"
API_URL="http://127.0.0.1:${API_PORT}"
TIC_BENCH_BUNDLE="${RUN_DIR}/tic-bundle"
BASELINE_CONTAINER="isiro-bench-${RUN_ID,,}-baseline"
TIC_PID=""
BASELINE_REUSED_FROM=""

WARMUPS="${PUBLISH_WARMUPS}"
PROMPTS="${PUBLISH_PROMPTS}"
REPLICATES="${PUBLISH_REPLICATES}"
PUBLISH_QUALITY=1
# Equalize must pack the same concurrency × input as timed generation so
# hybrid densify + inductor see tall prefill M (not 16×128 while timed is 32×61).
# Output stays short: warm prefill/TTFT path, not a full 256-token decode.
EQUALIZE_PROMPTS="${EQUALIZE_PROMPTS:-0}"
EQUALIZE_INPUT_TOKENS="${EQUALIZE_INPUT_TOKENS:-32}"
EQUALIZE_OUTPUT_TOKENS="${EQUALIZE_OUTPUT_TOKENS:-8}"
# Timed prompts default to concurrency. 40@32 leaves a second admit wave of
# singles (M~61) on fused pf (~189ms) while the packed wave is hybrid (~97ms):
# that is the equal-batch TTFT p95/p99 cliff, not a true latency regress.
PROMPTS_MATCH_CONCURRENCY="${PROMPTS_MATCH_CONCURRENCY:-1}"

timed_num_prompts() {
  if [[ "${PROMPTS_MATCH_CONCURRENCY}" == "1" ]]; then
    echo "${BENCH_MAX_CONCURRENCY}"
  else
    echo "${PROMPTS}"
  fi
}

# Profile selection. Public default is decode-only. TTFT names remain allowed
# for eng iteration via --profiles=... or ISIRO_BENCH_FULL_PROFILES=1.
ALL_PROFILES=(ttft-128 ttft-512 ttft-2048 generation-32-256 generation-16384-256)
DEFAULT_PROFILES=(generation-32-256)
SELECTED_PROFILES=("${DEFAULT_PROFILES[@]}")
if [[ "${ISIRO_BENCH_FULL_PROFILES:-0}" == "1" ]]; then
  SELECTED_PROFILES=("${ALL_PROFILES[@]}")
fi
if [[ -n "${PROFILES_CSV}" ]]; then
  SELECTED_PROFILES=()
  declare -A _wanted=()
  IFS=',' read -r -a _raw_profiles <<<"${PROFILES_CSV}"
  for name in "${_raw_profiles[@]}"; do
    name="$(echo "${name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${name}" ]] || continue
    _wanted["${name}"]=1
  done
  for name in "${ALL_PROFILES[@]}"; do
    if [[ -n "${_wanted[${name}]:-}" ]]; then
      SELECTED_PROFILES+=("${name}")
      unset "_wanted[${name}]"
    fi
  done
  if [[ "${#SELECTED_PROFILES[@]}" -eq 0 ]]; then
    echo "--profiles must name at least one of: ${ALL_PROFILES[*]}" >&2
    exit 2
  fi
  if [[ "${#_wanted[@]}" -gt 0 ]]; then
    echo "unknown --profiles entries: ${!_wanted[*]}" >&2
    echo "  allowed: ${ALL_PROFILES[*]}" >&2
    exit 2
  fi
fi

echo "  PROFILES          = ${SELECTED_PROFILES[*]}" >&2
echo "  BASELINE_REUSE    = ${REUSE_BASELINE}" >&2
echo "  SKIP_VERIFY       = ${SKIP_VERIFY}" >&2
echo "  SKIP_GDX_METERS   = ${SKIP_GDX_METERS}" >&2
echo "  SKIP_GDX_CURVE    = ${SKIP_GDX_CURVE}" >&2
echo "  SKIP_BASELINE     = ${SKIP_BASELINE}" >&2
echo "==========================" >&2

for command_name in python3 docker curl nvidia-smi ss; do
  command -v "${command_name}" >/dev/null || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done
# Bench client: host `vllm` or the same Docker image used for baseline serve.
if ! command -v vllm >/dev/null 2>&1; then
  if ! docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    echo "vllm CLI not on PATH and Docker image missing: ${VLLM_IMAGE}" >&2
    echo "  pull that image (baseline serve uses it) or install a host vllm CLI" >&2
    exit 2
  fi
fi
[[ -d "${BASELINE_MODEL_DIR}" ]] || { echo "baseline model directory not found" >&2; exit 2; }
[[ -d "${TIC_MODEL_DIR}" ]] || { echo "TIC model directory not found" >&2; exit 2; }

# Precision/path guard: refuse mismatched Hub quant configs.
# FP8: modelopt, compressed-tensors, or native Hub quant_method=fp8.
# AWQ: Hub quant_method=awq / awq_marlin.
FP8_BASELINE_QUANT=""
if [[ "${PRECISION}" == "fp8" ]]; then
  FP8_BASELINE_QUANT="$(
    PYTHONPATH="${BENCH_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
      "${BASELINE_MODEL_DIR}" <<'PY'
import sys
from pathlib import Path
from benchmark_lib import resolve_fp8_baseline_quantization

try:
    print(resolve_fp8_baseline_quantization(Path(sys.argv[1])))
except ValueError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
  )" || {
    echo "PRECISION=fp8 baseline quant resolve failed" >&2
    exit 2
  }
elif [[ "${PRECISION}" == "awq" ]]; then
  if ! PYTHONPATH="${BENCH_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
    "${BASELINE_MODEL_DIR}/config.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
method = str((doc.get("quantization_config") or {}).get("quant_method") or "").lower()
raise SystemExit(0 if method in ("awq", "awq_marlin") else 1)
PY
  then
    echo "PRECISION=awq but baseline is not Hub AWQ (quant_method awq/awq_marlin)" >&2
    exit 2
  fi
elif [[ -f "${BASELINE_MODEL_DIR}/hf_quant_config.json" ]]; then
  echo "PRECISION=bf16 but baseline has hf_quant_config.json; use PRECISION=fp8 / common.env.fp8" >&2
  exit 2
elif [[ -f "${BASELINE_MODEL_DIR}/config.json" ]]; then
  if PYTHONPATH="${BENCH_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
    "${BASELINE_MODEL_DIR}/config.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
method = str((doc.get("quantization_config") or {}).get("quant_method") or "").lower()
raise SystemExit(
    0
    if method in ("compressed-tensors", "compressed_tensors", "fp8")
    else 1
)
PY
  then
    echo "PRECISION=bf16 but baseline is Hub FP8; use PRECISION=fp8 / common.env.fp8" >&2
    exit 2
  fi
  if PYTHONPATH="${BENCH_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
    "${BASELINE_MODEL_DIR}/config.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
method = str((doc.get("quantization_config") or {}).get("quant_method") or "").lower()
raise SystemExit(0 if method in ("awq", "awq_marlin") else 1)
PY
  then
    echo "PRECISION=bf16 but baseline is Hub AWQ; use PRECISION=awq / common.env.awq" >&2
    exit 2
  fi
fi

build_baseline_serve() {
  # DEBUG surfaces MemoryProfilingResult ("Total non KV cache memory") for §B.
  : "${VLLM_LOGGING_LEVEL:=DEBUG}"
  export VLLM_LOGGING_LEVEL
  BASELINE_SERVE=(
    docker run --rm --name "${BASELINE_CONTAINER}" --gpus all
    -p "127.0.0.1:${API_PORT}:8000"
    -v "${BASELINE_MODEL_DIR}:/model:ro"
    -e VLLM_USE_FLASHINFER_SAMPLER=0
    -e "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}"
    "${VLLM_IMAGE}"
    --model /model
    --served-model-name "${MODEL_ID}"
  )
  if [[ "${PRECISION}" == "fp8" ]]; then
    # Hub FP8 (modelopt or compressed-tensors): do not force --dtype bfloat16.
    BASELINE_SERVE+=(--quantization "${FP8_BASELINE_QUANT}")
  elif [[ "${PRECISION}" == "awq" ]]; then
    # Hub AWQ: Marlin W4A16 path under vLLM; match TIC bf16 activations.
    BASELINE_SERVE+=(--quantization awq --dtype bfloat16)
  else
    BASELINE_SERVE+=(--dtype bfloat16)
  fi
  BASELINE_SERVE+=(
    --max-model-len "${SERVE_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${SERVE_GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${SERVE_TENSOR_PARALLEL_SIZE}"
    --max-num-seqs "${SERVE_MAX_NUM_SEQS}"
    --no-enable-prefix-caching
    --trust-remote-code
    --no-enable-flashinfer-autotune
  )
  # Capacity harness only: one-wave prefill budget (not a product setdefault).
  if [[ -n "${SERVE_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    BASELINE_SERVE+=(--max-num-batched-tokens "${SERVE_MAX_NUM_BATCHED_TOKENS}")
  fi
  if [[ "${SERVE_ENFORCE_EAGER}" == "true" ]]; then
    BASELINE_SERVE+=(--enforce-eager)
  fi
  if [[ -n "${SERVE_KV_CACHE_DTYPE}" ]]; then
    BASELINE_SERVE+=(--kv-cache-dtype "${SERVE_KV_CACHE_DTYPE}")
  fi
}
# Initial baseline command for dry-run / early logs; rebuilt after capacity search.
build_baseline_serve
# Customer launcher for serve (vendor Docker + staged AOT). Prefer ~/.isiro/bin.
# Refuse a private checkout .venv on PATH so publish never meters an eng tree.
export PATH="${HOME}/.isiro/bin:${PATH}"
ISIRO_SERVE_BIN="${ISIRO_SERVE_BIN:-${HOME}/.isiro/bin/isiro}"
if [[ ! -x "${ISIRO_SERVE_BIN}" ]]; then
  ISIRO_SERVE_BIN="$(command -v isiro || true)"
fi
if [[ -z "${ISIRO_SERVE_BIN}" || ! -x "${ISIRO_SERVE_BIN}" ]]; then
  echo "isiro serve binary not found; install customer isiro to ~/.isiro/bin" >&2
  exit 2
fi
if [[ "${ISIRO_SERVE_BIN}" == *".venv"* ]]; then
  echo "refusing isiro from a .venv (${ISIRO_SERVE_BIN}); use ~/.isiro/bin/isiro" >&2
  exit 2
fi
ISIRO_ON_PATH="$(command -v isiro || true)"
if [[ "${ISIRO_ON_PATH}" == *".venv"* ]]; then
  echo "isiro on PATH resolves to a .venv (${ISIRO_ON_PATH}); put ~/.isiro/bin first" >&2
  exit 2
fi
# Default verify is integrity-only (cli-slim). Set ISIRO_VERIFY_REFERENCE=1 for
# bit-exact -r via the customer launcher (compiler-verify image / activate).
VERIFY_REFERENCE=0
if [[ "${ISIRO_VERIFY_REFERENCE:-0}" == "1" ]]; then
  VERIFY_REFERENCE=1
fi
build_tic_serve() {
  TIC_SERVE=(
    # Bind 0.0.0.0 inside the serve container so Docker -p ${API_PORT}:${API_PORT}
    # is reachable from the host. The bench client still uses 127.0.0.1:${API_PORT}.
    # Binding only 127.0.0.1 in-container makes host curl get connection reset.
    "${ISIRO_SERVE_BIN}" serve "${TIC_BENCH_BUNDLE}" --target vllm
    --host 0.0.0.0 --port "${API_PORT}"
    --max-model-len "${SERVE_MAX_MODEL_LEN}"
  )
  if [[ -n "${SERVE_KV_CACHE_DTYPE}" && -z "${ISIRO_BENCH_GDX}" ]]; then
    # When GDX is on, the runtime stamps the matching vendor dtype.
    TIC_SERVE+=(--kv-cache-dtype "${SERVE_KV_CACHE_DTYPE}")
  fi
}
build_tic_serve

# Sets TIC_PID. Appends --enforce-eager when SERVE_ENFORCE_EAGER=true.
start_tic_serve() {
  local log="$1"
  unset ISIRO_SERVE_CUDA_GRAPHS
  unset ISIRO_GDX_C || true
  unset ISIRO_GDX_KV || true
  if [[ -n "${ISIRO_BENCH_GDX}" ]]; then
    export ISIRO_GDX="${ISIRO_BENCH_GDX}"
    export ISIRO_GDX_COLD_READY=1
    # common.env may set ISIRO_GDX_* without export; Docker forward uses compgen -e.
    if [[ -n "${ISIRO_GDX_FP8_KV_SCALES_OK:-}" ]]; then
      export ISIRO_GDX_FP8_KV_SCALES_OK
    fi
  else
    unset ISIRO_GDX || true
  fi
  build_tic_serve
  if [[ "${SERVE_ENFORCE_EAGER}" == "true" ]]; then
    "${TIC_SERVE[@]}" --enforce-eager >"${log}" 2>&1 &
  else
    "${TIC_SERVE[@]}" >"${log}" 2>&1 &
  fi
  TIC_PID=$!
}
VERIFY=(
  "${ISIRO_SERVE_BIN}" verify "${TIC_MODEL_DIR}/model.tic"
)
if [[ "${VERIFY_REFERENCE}" -eq 1 ]]; then
  VERIFY+=(--reference "${BASELINE_MODEL_DIR}")
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

write_commands_json() {
  PYTHONPATH="${BENCH_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
    "${RUN_DIR}/commands.json" \
    "${BASELINE_MODEL_DIR}" "${TIC_MODEL_DIR}" "${TIC_BENCH_BUNDLE}" \
    "${VLLM_IMAGE}" "${MODEL_ID}" "${API_PORT}" \
    "${SERVE_MAX_MODEL_LEN}" "${SERVE_GPU_MEMORY_UTILIZATION}" \
    "${SERVE_TENSOR_PARALLEL_SIZE}" "${SERVE_MAX_NUM_SEQS}" \
    "${BASELINE_REUSED_FROM}" "${PRECISION}" "${SERVE_ENFORCE_EAGER}" \
    "${VERIFY_REFERENCE}" \
    "${SERVE_KV_CACHE_DTYPE:-}" "${ISIRO_BENCH_GDX_KV:-}" \
    "${FP8_BASELINE_QUANT:-}" <<'PY'
import json, os, sys
from benchmark_lib import resolve_fp8_baseline_quantization

(
    out,
    baseline_model,
    tic_model,
    tic_bundle,
    image,
    model_id,
    port,
    max_model_len,
    gpu_util,
    tp,
    max_seqs,
    reused_from,
    precision,
    enforce_eager,
    verify_reference,
    kv_cache_dtype,
    gdx_kv,
    fp8_quant,
) = sys.argv[1:]
baseline = [
    "docker", "run", "--rm", "--gpus", "all",
    "-p", f"127.0.0.1:{port}:8000",
    "-v", f"{baseline_model}:/model:ro",
    "-e", "VLLM_USE_FLASHINFER_SAMPLER=0",
    image, "--model", "/model",
    "--served-model-name", model_id,
]
if precision == "fp8":
    quant = fp8_quant or resolve_fp8_baseline_quantization(baseline_model)
    baseline.extend(["--quantization", quant])
elif precision == "awq":
    baseline.extend(["--quantization", "awq", "--dtype", "bfloat16"])
else:
    baseline.extend(["--dtype", "bfloat16"])
baseline.extend([
    "--max-model-len", max_model_len,
    "--gpu-memory-utilization", gpu_util,
    "--tensor-parallel-size", tp,
    "--max-num-seqs", max_seqs,
    "--no-enable-prefix-caching",
    "--trust-remote-code",
    "--no-enable-flashinfer-autotune",
])
if kv_cache_dtype:
    baseline.extend(["--kv-cache-dtype", kv_cache_dtype])
if enforce_eager == "true":
    baseline.insert(baseline.index("--no-enable-prefix-caching") + 1, "--enforce-eager")
tic_verify = ["isiro", "verify", f"{tic_model}/model.tic"]
if verify_reference == "1":
    tic_verify.extend(["--reference", baseline_model])
tic_serve = [
    "isiro", "serve", tic_bundle, "--target", "vllm",
    "--host", "127.0.0.1", "--port", port,
    "--max-model-len", max_model_len,
]
if kv_cache_dtype and not gdx_kv:
    tic_serve.extend(["--kv-cache-dtype", kv_cache_dtype])
doc = {
    "baseline_serve": baseline,
    "precision": precision,
    "enforce_eager": enforce_eager == "true",
    "tic_verify": tic_verify,
    "verify_mode": "reference" if verify_reference == "1" else "integrity",
    "tic_serve": tic_serve,
    "kv_cache_dtype": kv_cache_dtype or None,
    "isiro_gdx_kv": gdx_kv or None,
}
if precision == "fp8":
    doc["fp8_baseline_quantization"] = (
        fp8_quant or resolve_fp8_baseline_quantization(baseline_model)
    )
if precision == "awq":
    doc["awq_baseline_quantization"] = "awq"
if gdx_kv:
    doc["tic_serve_env"] = {
        "ISIRO_GDX": gdx_kv,
        "ISIRO_GDX_COLD_READY": "1",
    }
    scales_ok = os.environ.get("ISIRO_GDX_FP8_KV_SCALES_OK", "").strip()
    if scales_ok:
        doc["tic_serve_env"]["ISIRO_GDX_FP8_KV_SCALES_OK"] = scales_ok
if reused_from:
    doc["baseline_reused_from"] = reused_from
json.dump(doc, open(out, "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
}

write_baseline_fingerprint() {
  publish_flag=()
  if [[ "${PUBLISH_QUALITY}" -eq 1 ]]; then
    publish_flag=(--publish-quality)
  fi
  python3 "${BENCH_ROOT}/scripts/baseline_cache.py" write \
    --out "${RUN_DIR}/baseline_fingerprint.json" \
    --vllm-image "${VLLM_IMAGE}" \
    --model-id "${MODEL_ID}" \
    --precision "${PRECISION}" \
    --baseline-model "${BASELINE_MODEL_DIR}" \
    --max-model-len "${SERVE_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${SERVE_GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${SERVE_TENSOR_PARALLEL_SIZE}" \
    --max-num-seqs "${SERVE_MAX_NUM_SEQS}" \
    --enforce-eager "${SERVE_ENFORCE_EAGER}" \
    --seed "${BENCH_SEED}" \
    --warmups "${WARMUPS}" \
    --prompts "${PROMPTS}" \
    --replicates "${REPLICATES}" \
    --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    "${publish_flag[@]}"
}

try_reuse_baseline() {
  find_flags=()
  if [[ "${PUBLISH_QUALITY}" -eq 1 ]]; then
    find_flags=(--require-publish-quality)
  fi
  write_baseline_fingerprint >/dev/null
  if hit="$(
    python3 "${BENCH_ROOT}/scripts/baseline_cache.py" find \
      --scratch-root "${SCRATCH_ROOT}" \
      --fingerprint-file "${RUN_DIR}/baseline_fingerprint.json" \
      "${find_flags[@]}"
  )"; then
    # Do not reuse the run directory we just created.
    if [[ "${hit}" == "${RUN_DIR}" ]]; then
      return 1
    fi
    python3 "${BENCH_ROOT}/scripts/baseline_cache.py" copy \
      --from-run "${hit}" \
      --to-run "${RUN_DIR}" >/dev/null
    BASELINE_REUSED_FROM="$(basename "${hit}")"
    echo "reusing baseline from ${BASELINE_REUSED_FROM}"
    return 0
  fi
  return 1
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Config: ${CONFIG}"
  echo "Precision: ${PRECISION}"
  echo "Graph mode: ${GRAPH_MODE} (enforce_eager=${SERVE_ENFORCE_EAGER})"
  echo "Experiment kind: ${EXPERIMENT_KIND}"
  if [[ "${EXPERIMENT_KIND}" == "capacity" ]]; then
    if [[ "${ISIRO_BENCH_CAPACITY_SEARCH:-0}" == "1" ]]; then
      echo "Capacity search: lo=${CAPACITY_SEARCH_LO:-1} hi=${CAPACITY_SEARCH_HI:-${SERVE_MAX_NUM_SEQS}} (eng)"
    else
      echo "Capacity operating point: baseline max_num_seqs=${SERVE_MAX_NUM_SEQS}; TIC KV-scaled (no search)"
    fi
    echo "  equal-batch transparency: off by default (ISIRO_BENCH_EQUAL_BATCH=1 to enable)"
  else
    echo "Equal-batch: matched max_num_seqs=${SERVE_MAX_NUM_SEQS} concurrency=${BENCH_MAX_CONCURRENCY}"
    echo "  (eng transparency; capacity benefit not active)"
  fi
  echo "TIC ISIRO_ATTN: ${ISIRO_ATTN:-unset (product isiro_fa)}"
  echo "Baseline server:"
  print_command "${BASELINE_SERVE[@]}"
  echo "TIC verification:"
  print_command "${VERIFY[@]}"
  echo "TIC server:"
  echo "  (runner creates an ignored bundle overlay with prefix caching disabled)"
  print_command isiro serve '<scratch>/tic-bundle' --target vllm \
    --host 0.0.0.0 --port "${API_PORT}" --max-model-len "${SERVE_MAX_MODEL_LEN}"
  echo "Profiles: ${SELECTED_PROFILES[*]}"
  echo "Counts: warmups=${WARMUPS} prompts=${PROMPTS} replicates=${REPLICATES}"
  if [[ "${REUSE_BASELINE}" -eq 1 ]]; then
    echo "Baseline reuse: requested (will copy matching scratch baseline if present)"
  else
    echo "Baseline reuse: off"
  fi
  exit 0
fi

mkdir -p "${RUN_DIR}/baseline" "${RUN_DIR}/isiro" "${RAW_DIR}" "${LOG_DIR}"

# Fail early on verify before long GPU work (capacity search / Docker serve).
if [[ "${SKIP_VERIFY}" -eq 1 ]]; then
  echo "verify: SKIP (ISIRO_BENCH_SKIP_VERIFY=1)" >&2
  python3 - "${RUN_DIR}/verify.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "integrity_ok": True,
            "verify_mode": "skipped",
            "command": [],
            "exit_code": 0,
            "bit_exact_ok": True,
            "skipped": True,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
else
if [[ "${VERIFY_REFERENCE}" -eq 1 ]]; then
  # Host-RAM hazard: verify -r decodes every TIC tensor vs Hub (27B AWQ
  # ~21+18 GiB on disk; peak host RSS can OOM a desktop session). Refuse
  # large AWQ / large on-disk pairs unless explicitly forced.
  _FORCE_VR="$(_bench_env_truthy "${ISIRO_BENCH_FORCE_VERIFY_R:-}" && echo 1 || echo 0)"
  _ONDISK_GIB="$(
    python3 - "${BASELINE_MODEL_DIR}" "${TIC_MODEL_DIR}" <<'PY'
import sys
from pathlib import Path

def gib(p: Path) -> float:
    total = 0
    if p.is_file():
        return p.stat().st_size / (1024**3)
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024**3)

print(f"{gib(Path(sys.argv[1])) + gib(Path(sys.argv[2])):.1f}")
PY
  )"
  if [[ "${_FORCE_VR}" != "1" ]]; then
    if [[ "${PRECISION}" == "awq" ]] || python3 -c "import sys; sys.exit(0 if float('${_ONDISK_GIB}') >= 24.0 else 1)"; then
      echo "verify: refusing ISIRO_VERIFY_REFERENCE=1 (bit-exact -r) for this A/B." >&2
      echo "  on-disk baseline+tic ≈ ${_ONDISK_GIB} GiB; PRECISION=${PRECISION}" >&2
      echo "  verify -r can peak host RAM enough to OOM / log out a desktop session." >&2
      echo "  Use integrity-only (unset ISIRO_VERIFY_REFERENCE) for publish A/B." >&2
      echo "  Lab override only: ISIRO_BENCH_FORCE_VERIFY_R=1" >&2
      exit 2
    fi
  else
    echo "verify: ISIRO_BENCH_FORCE_VERIFY_R=1 (bit-exact -r allowed; watch host RAM)" >&2
  fi
  echo "verify: bit-exact isiro verify -r" >&2
  echo "verify: checking compiler access..." >&2
  # Capture status text first. Do not pipe into grep -q under pipefail:
  # early grep exit can SIGPIPE isiro and falsely look like "not activated".
  STATUS_OUT="$("${ISIRO_SERVE_BIN}" status 2>&1 || true)"
  if ! printf '%s\n' "${STATUS_OUT}" | grep -Eqi 'Compiler:[[:space:]]+activated'; then
    echo "bit-exact verify (-r) needs the ISIRO compiler activated." >&2
    echo "Get access: https://isiro.ai/product/runtime#compiler-access" >&2
    printf '%s\n' "${STATUS_OUT}" >&2
    exit 2
  fi
  echo "verify: compiler ready; running isiro verify -r (may take a few minutes)..." >&2
else
  echo "verify: integrity-only (ISIRO_VERIFY_REFERENCE!=1)" >&2
  echo "verify: running isiro verify..." >&2
fi
echo "verify: log ${LOG_DIR}/verify.log" >&2
# Optional compiler checkout for hop1 remat on verify -r (ISIRO_CORE_SRC).
# Do not leak into isiro serve. Auto-detect invent checkout next to TIC model.
_VERIFY_REPO_SRC=""
if [[ -z "${ISIRO_CORE_SRC:-}" ]]; then
  for _cand in \
    "$(cd "$(dirname "${TIC_MODEL_DIR}")/../.." 2>/dev/null && pwd)/src" \
    "${HOME}/repos/isiroai/isiro"'-core'"/src"; do
    if [[ -n "${_cand}" && -d "${_cand}/isiro_encoder" && -d "${_cand}/isiro_format" ]]; then
      ISIRO_CORE_SRC="${_cand}"
      break
    fi
  done
fi
if [[ -n "${ISIRO_CORE_SRC:-}" && -d "${ISIRO_CORE_SRC}/isiro_encoder" ]]; then
  _VERIFY_REPO_SRC="$(cd "${ISIRO_CORE_SRC}" && pwd)"
  echo "verify: ISIRO_REPO_SRC=${_VERIFY_REPO_SRC}" >&2
fi
set +e
if [[ -n "${_VERIFY_REPO_SRC}" ]]; then
  ISIRO_REPO_SRC="${_VERIFY_REPO_SRC}" "${VERIFY[@]}" >"${LOG_DIR}/verify.log" 2>&1
else
  "${VERIFY[@]}" >"${LOG_DIR}/verify.log" 2>&1
fi
VERIFY_EXIT=$?
set -e
python3 - "${RUN_DIR}/verify.json" "${VERIFY_EXIT}" "${BASELINE_MODEL_DIR}" \
  "${TIC_MODEL_DIR}" "${VERIFY_REFERENCE}" <<'PY'
import json, sys
ok = int(sys.argv[2]) == 0
verify_reference = sys.argv[5] == "1"
command = ["isiro", "verify", f"{sys.argv[4]}/model.tic"]
if verify_reference:
    command.extend(["--reference", sys.argv[3]])
doc = {
    "integrity_ok": ok,
    "verify_mode": "reference" if verify_reference else "integrity",
    "command": command,
    "exit_code": int(sys.argv[2]),
}
if verify_reference:
    doc["bit_exact_ok"] = ok
json.dump(doc, open(sys.argv[1], "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
[[ "${VERIFY_EXIT}" -eq 0 ]] || { echo "isiro verify failed; see ${LOG_DIR}/verify.log" >&2; exit 2; }
echo "verify: PASS" >&2
fi

if [[ -n "${ISIRO_BENCH_GDX}" && "${SKIP_GDX_METERS}" -eq 0 ]]; then
  # Host meter python was preflighted at harness start (ensure_gdx_meter_env).
  : "${ISIRO_BENCH_PYTHON:?ISIRO_BENCH_PYTHON unset after gdx_meter_env ensure}"
  _gdx_host_python="${ISIRO_BENCH_PYTHON}"
  echo "gdx_kv_bitexact: measuring cold roundtrip (alphabet=${ISIRO_BENCH_GDX}; python=${_gdx_host_python})" >&2
  "${_gdx_host_python}" "${BENCH_ROOT}/scripts/gdx_kv_bitexact.py" \
    --alphabet "${ISIRO_BENCH_GDX}" \
    --out "${RUN_DIR}/gdx_kv_bitexact.json"
  echo "gdx_kv_bitexact: PASS" >&2
  if [[ "${SKIP_GDX_CURVE}" -eq 1 ]]; then
    echo "gdx_kv_bytes_curve: SKIP (ISIRO_BENCH_SKIP_GDX_CURVE=1; no kv-bytes-vs-context.png until a fitting twin or vendor-page curve)" >&2
  else
  echo "gdx_kv_bytes_curve: measuring cold vs FA twin (alphabet=${ISIRO_BENCH_GDX}; python=${_gdx_host_python})" >&2
  _gdx_gpu_model="${ISIRO_BENCH_GPU_MODEL:-}"
  _gdx_gpu_count="${ISIRO_BENCH_GPU_COUNT:-0}"
  if [[ -z "${_gdx_gpu_model}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    _gdx_gpu_model="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    _gdx_gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  fi
  _gdx_curve_extra=()
  if [[ -f "${RUN_DIR}/summary.json" ]]; then
    _gdx_curve_extra+=(--summary-json "${RUN_DIR}/summary.json")
  fi
  if [[ -n "${ISIRO_BENCH_GDX_TEXT_FILE:-}" && -f "${ISIRO_BENCH_GDX_TEXT_FILE}" ]]; then
    _gdx_curve_extra+=(--text-file "${ISIRO_BENCH_GDX_TEXT_FILE}")
  fi
  # fp8/fp4: vendor FA pages via VLLM_IMAGE (same kv_cache_dtype as serve).
  # bf16: HF forward; optional ISIRO_BENCH_GDX_MODEL_DIR twin when baseline is Hub FP8.
  _gdx_curve_model_dir="${BASELINE_MODEL_DIR}"
  _gdx_vendor_pages=""
  case "${ISIRO_BENCH_GDX}" in
    fp8|fp4)
      _gdx_curve_max_t="$(
        printf '%s\n' "${SERVE_MAX_MODEL_LEN:-8192}" \
          | awk -F, '{print $1+0}'
      )"
      [[ "${_gdx_curve_max_t}" -ge 512 ]] || _gdx_curve_max_t=8192
      _gdx_vendor_pages="${RUN_DIR}/gdx_vendor_fa_pages.npz"
      # Fail closed before docker when HF ABI is MLA/sparse/CSA2 (GLM/DeepSeek Flash).
      # path_a_mla: use ISIRO_BENCH_MLA_LATENT_NPZ ([L,T,D] uint8) instead of FA dump.
      _gdx_use_mla_latent=0
      _gdx_preflight_err="$(
        "${_gdx_host_python}" - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "${BENCH_ROOT}/scripts")
from gdx_kv_curve_vendor_pages import vendor_fa_pages_unsupported_reason
r = vendor_fa_pages_unsupported_reason(Path("${BASELINE_MODEL_DIR}"))
if r:
    print(r)
    raise SystemExit(2)
PY
      )" || {
        _gdx_mla_npz="${ISIRO_BENCH_MLA_LATENT_NPZ:-}"
        if [[ -n "${_gdx_mla_npz}" && -f "${_gdx_mla_npz}" ]]; then
          echo "gdx_kv_curve_mla_latent: FA pages unsupported (${_gdx_preflight_err}); using MLA latent ${_gdx_mla_npz}" >&2
          _gdx_curve_extra+=(--mla-latent "${_gdx_mla_npz}")
          _gdx_curve_extra+=(--kv-cache-dtype "${SERVE_KV_CACHE_DTYPE:-fp8_ds_mla}")
          _gdx_use_mla_latent=1
        else
          echo "gdx_kv_curve_vendor_pages: FAIL ${_gdx_preflight_err}" >&2
          echo "gdx_kv_curve_vendor_pages: MLA/sparse HF; set ISIRO_BENCH_MLA_LATENT_NPZ to a [L,T,D] uint8 capture (path_a_mla), or ISIRO_BENCH_SKIP_GDX_METERS=1." >&2
          exit 2
        fi
      }
      if [[ "${_gdx_use_mla_latent}" -eq 0 ]]; then
        echo "gdx_kv_curve_vendor_pages: capturing FA pages (kv_cache_dtype=${SERVE_KV_CACHE_DTYPE}; image=${VLLM_IMAGE}; model=${BASELINE_MODEL_DIR})" >&2
        _gdx_vendor_mounts=(-v "${BASELINE_MODEL_DIR}:${BASELINE_MODEL_DIR}:ro")
        _gdx_vendor_mounts+=(-v "${BENCH_ROOT}/scripts:${BENCH_ROOT}/scripts:ro")
        _gdx_vendor_mounts+=(-v "${RUN_DIR}:${RUN_DIR}")
        if [[ -n "${ISIRO_BENCH_GDX_TEXT_FILE:-}" && -f "${ISIRO_BENCH_GDX_TEXT_FILE}" ]]; then
          _gdx_vendor_mounts+=(-v "${ISIRO_BENCH_GDX_TEXT_FILE}:${ISIRO_BENCH_GDX_TEXT_FILE}:ro")
        fi
        _gdx_fixture="${BENCH_ROOT}/fixtures/pride_and_prejudice.txt"
        if [[ -f "${_gdx_fixture}" ]]; then
          _gdx_vendor_mounts+=(-v "${_gdx_fixture}:${_gdx_fixture}:ro")
        fi
        _gdx_vendor_args=(
          "${BENCH_ROOT}/scripts/gdx_kv_curve_vendor_pages.py"
          --model-dir "${BASELINE_MODEL_DIR}"
          --kv-cache-dtype "${SERVE_KV_CACHE_DTYPE}"
          --max-t "${_gdx_curve_max_t}"
          --out "${_gdx_vendor_pages}"
        )
        if [[ -n "${ISIRO_BENCH_GDX_TEXT_FILE:-}" && -f "${ISIRO_BENCH_GDX_TEXT_FILE}" ]]; then
          _gdx_vendor_args+=(--text-file "${ISIRO_BENCH_GDX_TEXT_FILE}")
        fi
        docker run --rm --gpus all \
          -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
          "${_gdx_vendor_mounts[@]}" \
          --entrypoint python3 \
          "${VLLM_IMAGE}" \
          "${_gdx_vendor_args[@]}"
        _gdx_curve_extra+=(--vendor-pages "${_gdx_vendor_pages}")
        _gdx_curve_extra+=(--kv-cache-dtype "${SERVE_KV_CACHE_DTYPE}")
      fi
      ;;
    bf16)
      _gdx_curve_model_dir="${ISIRO_BENCH_GDX_MODEL_DIR:-${BASELINE_MODEL_DIR}}"
      ;;
  esac
  _gdx_live_window="${ISIRO_GDX_LIVE_WINDOW:-512}"
  "${_gdx_host_python}" "${BENCH_ROOT}/scripts/gdx_kv_bytes_curve.py" \
    --alphabet "${ISIRO_BENCH_GDX}" \
    --out-json "${RUN_DIR}/gdx_kv_bytes_curve.json" \
    --out-png "${RUN_DIR}/kv-bytes-vs-context.png" \
    --model-dir "${_gdx_curve_model_dir}" \
    --model "${MODEL_ID}" \
    --gpu-model "${_gdx_gpu_model}" \
    --gpu-count "${_gdx_gpu_count}" \
    --weight-precision "${PRECISION}" \
    --kv-precision "${ISIRO_BENCH_GDX}" \
    --live-window "${_gdx_live_window}" \
    "${_gdx_curve_extra[@]}"
  echo "gdx_kv_bytes_curve: PASS" >&2
  fi  # SKIP_GDX_CURVE
elif [[ -n "${ISIRO_BENCH_GDX}" && "${SKIP_GDX_METERS}" -eq 1 ]]; then
  echo "gdx_kv_meters: SKIP (ISIRO_BENCH_SKIP_GDX_METERS=1)" >&2
  # compare_ab requires gdx_kv_bitexact.json when ISIRO_GDX is set; reuse ref.
  _gdx_meter_src="${ISIRO_BENCH_BASELINE_REF:-}"
  if [[ -z "${_gdx_meter_src}" || ! -e "${_gdx_meter_src}/gdx_kv_bitexact.json" ]]; then
    # Fall back to newest prior scratch for this system that already metered.
    _gdx_meter_src="$(
      ls -1dt "${SCRATCH_ROOT}"/*-"${SYSTEM_ID}" 2>/dev/null \
        | while read -r _d; do
            [[ -e "${_d}/gdx_kv_bitexact.json" && "${_d}" != "${RUN_DIR}" ]] && echo "${_d}" && break
          done
    )"
  fi
  if [[ -n "${_gdx_meter_src}" ]]; then
    for _gdx_ref in gdx_kv_bitexact.json gdx_kv_bytes_curve.json kv-bytes-vs-context.png; do
      if [[ -e "${_gdx_meter_src}/${_gdx_ref}" && ! -e "${RUN_DIR}/${_gdx_ref}" ]]; then
        cp -a "${_gdx_meter_src}/${_gdx_ref}" "${RUN_DIR}/${_gdx_ref}"
        echo "gdx_kv_meters: reused ${_gdx_ref} from ${_gdx_meter_src}" >&2
      fi
    done
  fi
  unset _gdx_meter_src _gdx_ref
fi

cleanup() {
  if [[ -n "${TIC_PID}" ]] && kill -0 "${TIC_PID}" 2>/dev/null; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
  fi
  docker rm -f "${BASELINE_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Capacity harness: raise vendor max_num_batched_tokens so one prefill wave
# fits concurrent short prompts (avoids TTFT cliff past ~33 seqs @ 2048).
# Empty = leave vendor default. Never set by product isiro serve.
: "${SERVE_MAX_NUM_BATCHED_TOKENS:=}"
: "${CAPACITY_BENCH_INPUT_TOKENS:=32}"

apply_capacity_one_wave_batched_tokens() {
  # Sets SERVE_MAX_NUM_BATCHED_TOKENS from current BENCH_MAX_CONCURRENCY, or clears it.
  if [[ "${EXPERIMENT_KIND}" != "capacity" ]]; then
    SERVE_MAX_NUM_BATCHED_TOKENS=""
    return 0
  fi
  local conc="${BENCH_MAX_CONCURRENCY:-${SERVE_MAX_NUM_SEQS}}"
  local budget
  budget="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" one-wave-batched-tokens \
      --concurrency "${conc}" \
      --input-tokens "${CAPACITY_BENCH_INPUT_TOKENS}"
  )"
  if [[ -n "${budget}" ]]; then
    SERVE_MAX_NUM_BATCHED_TOKENS="${budget}"
    echo "capacity one-wave: max_num_batched_tokens=${budget} for conc=${conc} (harness overlay; not product setdefault)" >&2
  else
    SERVE_MAX_NUM_BATCHED_TOKENS=""
    echo "capacity one-wave: vendor default batched tokens fit conc=${conc}" >&2
  fi
}

prepare_tic_bundle() {
  mkdir -p "${TIC_BENCH_BUNDLE}"
  for source in "${TIC_MODEL_DIR}"/*; do
    name="$(basename "${source}")"
    if [[ "${name}" != "serve.yaml" ]]; then
      # Hard link so Docker bind-mounts of tic-bundle see file inodes.
      # Symlinks to absolute host paths are invisible inside the serve container.
      if [[ -e "${TIC_BENCH_BUNDLE}/${name}" || -L "${TIC_BENCH_BUNDLE}/${name}" ]]; then
        rm -f "${TIC_BENCH_BUNDLE}/${name}"
      fi
      if ! ln "${source}" "${TIC_BENCH_BUNDLE}/${name}" 2>/dev/null; then
        cp -a "${source}" "${TIC_BENCH_BUNDLE}/${name}"
      fi
    fi
  done
  # Match TIC serve Docker tag to VLLM_IMAGE so image-ABI .so (glibc) matches
  # the Baseline arm (bundle serve.yaml often pins an older tag).
  local _vllm_tag="${VLLM_IMAGE##*:}"
  python3 - \
    "${TIC_MODEL_DIR}/serve.yaml" \
    "${TIC_BENCH_BUNDLE}/serve.yaml" \
    "${SERVE_MAX_MODEL_LEN}" \
    "${SERVE_GPU_MEMORY_UTILIZATION}" \
    "${SERVE_TENSOR_PARALLEL_SIZE}" \
    "${SERVE_MAX_NUM_SEQS}" \
    "${SERVE_ENFORCE_EAGER}" \
    "${_vllm_tag}" \
    "${SERVE_MAX_NUM_BATCHED_TOKENS:-}" <<'PY'
import re, sys

(
    source,
    target,
    max_model_len,
    gpu_util,
    tp,
    max_seqs,
    enforce_eager,
    docker_tag,
    max_batched,
) = sys.argv[1:10]
text = open(source, encoding="utf-8").read()

def upsert(text: str, key: str, value: str) -> str:
    pattern = rf"^(\s*){re.escape(key)}:\s*.*$"
    replacement = rf"\1{key}: {value}"
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count:
        return updated
    if key.startswith("no_enable_"):
        enabled = key.removeprefix("no_")
        updated, _ = re.subn(
            rf"^(\s*){re.escape(enabled)}:\s*.*$",
            "",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        text = updated
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

def drop_key(text: str, key: str) -> str:
    updated, _ = re.subn(
        rf"^(\s*){re.escape(key)}:\s*.*\n?",
        "",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return updated

text = upsert(text, "max_model_len", max_model_len)
text = upsert(text, "gpu_memory_utilization", gpu_util)
text = upsert(text, "tensor_parallel_size", tp)
text = upsert(text, "max_num_seqs", max_seqs)
text = upsert(text, "no_enable_prefix_caching", "true")
# Baseline Hub graphs-on uses VLLM_COMPILE + FULL_AND_PIECEWISE. Lean TIC
# default (DYNAMO_TRACE_ONCE + FULL_DECODE_ONLY) loses TTFT vs Hub. Pin
# path_a parity whenever graphs are on so A/B is fair (same as
# _SERVE_CUDAGRAPH_COMPILATION_CONFIG_PATH_A). Quote as a YAML string:
# unquoted {...} becomes a mapping and breaks CLI.
_path_a_cc = (
    "'{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\","
    "\"inductor_compile_config\":{\"combo_kernels\":true,"
    "\"benchmark_combo_kernel\":true}}'"
)
if str(max_batched).strip():
    text = upsert(text, "max_num_batched_tokens", str(max_batched).strip())
    # Domino: equal-batch hybrid win sits on vendor compile [(1,2048)]
    # (conc=32, no batched overlay). Capacity conc=40 needs batched=2560;
    # omitting compilation_config fattens to [(1,N)] (TTFT ~224). Pinning
    # endpoints to [2048] alone still grows a second range from batched
    # tokens (225810). Dual-range [2048,N] is the least-bad capacity
    # compile shape (~141 best). Keep path_a mode/cudagraph fields.
    try:
        mb = int(str(max_batched).strip())
    except ValueError:
        mb = 0
    if mb > 2048:
        cc = (
            "'{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\","
            "\"compile_ranges_endpoints\":[2048," + str(mb) + "],"
            "\"inductor_compile_config\":{\"combo_kernels\":true,"
            "\"benchmark_combo_kernel\":true}}'"
        )
        text = upsert(text, "compilation_config", cc)
        text = upsert(text, "max_cudagraph_capture_size", "64")
    elif enforce_eager == "true":
        text = drop_key(text, "compilation_config")
        text = drop_key(text, "max_cudagraph_capture_size")
    else:
        text = upsert(text, "compilation_config", _path_a_cc)
        # Path_a compile without ISIRO_GDX still needs stock capture 64;
        # lean admit-align(max_num_seqs=8) → 8 collapses TTFT vs Hub.
        text = upsert(text, "max_cudagraph_capture_size", "64")
else:
    # Equal-batch / non-capacity: drop batched overlay; keep compile parity.
    text = drop_key(text, "max_num_batched_tokens")
    if enforce_eager == "true":
        text = drop_key(text, "compilation_config")
        text = drop_key(text, "max_cudagraph_capture_size")
    else:
        text = upsert(text, "compilation_config", _path_a_cc)
        text = upsert(text, "max_cudagraph_capture_size", "64")
if docker_tag:
    # targets.vllm.docker.tag - keep indent under docker:
    text = upsert(text, "tag", docker_tag)
# enforce_eager is a forbidden serve.yaml key (isiro serve controls it).
# Product default is outer graphs ON; eager A/B sets ISIRO_SERVE_ENFORCE_EAGER=1
# on the serve process env (see the TIC serve launch), so it is not written here.
open(target, "w", encoding="utf-8").write(text)
PY
}

require_quiet_gpu() {
  if [[ "${ALLOW_BUSY_GPU:-}" == "1" ]]; then
    echo "ALLOW_BUSY_GPU=1: skipping quiet-host GPU occupancy check" >&2
    return 0
  fi
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' || true)"
  if [[ -n "${apps}" ]]; then
    echo "GPU already has compute processes; refuse start for quiet-host capture" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    exit 2
  fi
}

api_port_listening() {
  ss -ltn "( sport = :${API_PORT} )" 2>/dev/null | grep -qE ":${API_PORT}([^0-9]|$)"
}

# Drop stale listeners before baseline Docker publish or TIC host serve.
ensure_api_port_free() {
  if ! api_port_listening; then
    return 0
  fi
  echo "API port ${API_PORT} still in use; clearing before serve" >&2
  # Orphaned baseline Docker publishes keep 127.0.0.1:PORT after a killed
  # harness; fuser alone does not remove the container.
  if command -v docker >/dev/null 2>&1; then
    docker ps -aq --filter "name=isiro-bench-" | xargs -r docker rm -f >/dev/null 2>&1 || true
    docker ps -aq --filter "name=isiro-serve-" | xargs -r docker rm -f >/dev/null 2>&1 || true
    # Host TIC serve sometimes leaves a named container on the API port.
    docker ps -aq --filter "publish=${API_PORT}" | xargs -r docker rm -f >/dev/null 2>&1 || true
  fi
  # Prefer fuser only for host listeners. Avoid `pkill -f 'isiro serve'` here:
  # it can match the harness shell cmdline and kill the bench mid-flight.
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do
    if ! api_port_listening; then
      return 0
    fi
    sleep 0.25
  done
  echo "API port ${API_PORT} still busy after cleanup" >&2
  ss -ltnp "( sport = :${API_PORT} )" >&2 || true
  docker ps -a --format '{{.Names}} {{.Status}} {{.Ports}}' 2>/dev/null | grep -E 'isiro|8000' >&2 || true
  exit 2
}

wait_ready() {
  # Large TIC + CUDA graphs can exceed 6 minutes on first load.
  # Do not key off the launcher PID: `isiro serve` may re-exec / replace the
  # process (parent exits while the engine child still owns the port).
  local serve_log="${1:-}"
  local serve_pid="${2:-}"
  for _ in $(seq 1 600); do
    if curl -fsS "${API_URL}/version" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${serve_log}" && -f "${serve_log}" ]] &&
       grep -qE 'Engine failed to start|OutOfMemoryError|SERV-0301' "${serve_log}" 2>/dev/null; then
      echo "serve failed during start; see ${serve_log}" >&2
      return 1
    fi
    # SERV-0101 can race with replace/cleanup. Only fail-fast when the launcher
    # has already exited (port conflict left nothing running).
    if [[ -n "${serve_log}" && -f "${serve_log}" ]] &&
       grep -qE 'SERV-0101|Port .* is already in use' "${serve_log}" 2>/dev/null; then
      if [[ -n "${serve_pid}" ]] && ! kill -0 "${serve_pid}" 2>/dev/null; then
        echo "serve failed during start; see ${serve_log}" >&2
        return 1
      fi
    fi
    sleep 2
  done
  echo "server did not become ready at ${API_URL}" >&2
  return 1
}

# Fail closed before equalize / scrape / profiles when Model loading took already
# shows TIC weights heavier than baseline (e.g. leftover densify-at-attach).
assert_tic_weight_hbm_win() {
  local tic_log="${1:?tic serve log}"
  local baseline_env="${RUN_DIR}/baseline_environment.json"
  if [[ ! -f "${baseline_env}" ]]; then
    echo "assert_tic_weight_hbm_win: missing ${baseline_env}; skip" >&2
    return 0
  fi
  if [[ ! -f "${tic_log}" ]]; then
    echo "assert_tic_weight_hbm_win: missing ${tic_log}" >&2
    return 1
  fi
  python3 - "${baseline_env}" "${tic_log}" <<'PY'
import json, re, sys

baseline_path, tic_log = sys.argv[1], sys.argv[2]
doc = json.load(open(baseline_path, encoding="utf-8"))
base = doc.get("model_loading_bytes_reported")
if base is None:
    print(
        "assert_tic_weight_hbm_win: baseline model_loading_bytes_reported "
        "missing; skip",
        file=sys.stderr,
    )
    raise SystemExit(0)
text = open(tic_log, encoding="utf-8", errors="replace").read()
m = re.search(r"Model loading took\s*([0-9.]+)\s*GiB", text)
if not m:
    print(
        "assert_tic_weight_hbm_win: TIC Model loading took missing in "
        f"{tic_log}; refuse to continue",
        file=sys.stderr,
    )
    raise SystemExit(2)
gib = 1024 ** 3
tic = round(float(m.group(1)) * gib)
base_gib = base / gib
tic_gib = tic / gib
savings = (1.0 - (tic / base)) * 100.0
print(
    f"weight HBM check: baseline={base_gib:.2f} GiB "
    f"tic={tic_gib:.2f} GiB savings={savings:.2f}%",
    file=sys.stderr,
)
if tic >= base:
    print(
        "NO GO: TIC Model loading took is not a weight win "
        f"({tic_gib:.2f} GiB >= baseline {base_gib:.2f} GiB). "
        "Abort before equalize/profiles. Fix resident compressed leftovers "
        "(see NOTE_awq_leftover_dense / NOTE_raw_f8_fuse).",
        file=sys.stderr,
    )
    raise SystemExit(3)
PY
}

equalize_warmup() {
  variant="$1"
  local eq_prompts="${EQUALIZE_PROMPTS}"
  if [[ "${eq_prompts}" -le 0 ]]; then
    eq_prompts="${BENCH_MAX_CONCURRENCY}"
  fi
  curl -fsS "${API_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"temperature\":0}" \
    >/dev/null
  echo "equalize: prompts=${eq_prompts} conc=${BENCH_MAX_CONCURRENCY} in=${EQUALIZE_INPUT_TOKENS} out=${EQUALIZE_OUTPUT_TOKENS}" >&2
  python3 "${BENCH_ROOT}/scripts/run_bench.py" \
    --variant "${variant}" \
    --profile "warmup-equalize" \
    --run-id "${RUN_ID}" \
    --model "${MODEL_ID}" \
    "${BENCH_TOKENIZER_ARGS[@]}" \
    --api-url "${API_URL}" \
    --input-tokens "${EQUALIZE_INPUT_TOKENS}" \
    --output-tokens "${EQUALIZE_OUTPUT_TOKENS}" \
    --num-warmups 0 \
    --num-prompts "${eq_prompts}" \
    --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --seed "${BENCH_SEED}" \
    --replicates 1 \
    --serve-config "${RUN_DIR}/${variant}_serve_config.json" \
    --raw-dir "${LOG_DIR}/warmup-equalize/${variant}" \
    --out "${LOG_DIR}/warmup-equalize-${variant}.json" \
    --vllm-image "${VLLM_IMAGE}"
}

# Greedy serve output capture (before timed vllm bench). compare_ab folds A/B.
# Thinking off is in serve_output_match.py. Closed-form prompts fit in 32 tokens.
capture_serve_output_match() {
  variant="$1"
  python3 "${BENCH_ROOT}/scripts/serve_output_match.py" capture \
    --api-url "${API_URL}" \
    --model "${MODEL_ID}" \
    --variant "${variant}" \
    --seed "${BENCH_SEED}" \
    --max-tokens 32 \
    --out "${RUN_DIR}/output_match_${variant}.json"
}

write_effective_config() {
  variant="$1"
  yaml_path="${2:-}"
  python3 - \
    "${LOG_DIR}/${variant}-serve.log" \
    "${RUN_DIR}/${variant}_serve_config.json" \
    "${yaml_path}" \
    "${SERVE_ENFORCE_EAGER}" \
    "${BENCH_ROOT}/scripts" <<'PY'
import json, re, sys
from pathlib import Path

sys.path.insert(0, sys.argv[5])
from attention_ab import parse_serve_attention, requested_isiro_attn

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
yaml_path = Path(sys.argv[3]) if sys.argv[3] else None
expect_enforce_eager = sys.argv[4] == "true"

def flag(pattern, cast=str):
    match = re.search(pattern, text)
    return cast(match.group(1)) if match else None

def as_bool(value):
    return str(value) in ("True", "true", "1")

def yaml_value(key, cast=str):
    if yaml_path is None or not yaml_path.is_file():
        return None
    match = re.search(
        rf"^\s*{re.escape(key)}:\s*([^\s#]+)",
        yaml_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return cast(match.group(1)) if match else None

doc = {
    "max_model_len": flag(r"max_seq_len=([0-9]+)", int)
    or flag(r"max_model_len['\"]?:\s*([0-9]+)", int)
    or yaml_value("max_model_len", int),
    "gpu_memory_utilization": flag(r"gpu_memory_utilization['\"]?:\s*([0-9.]+)", float)
    or yaml_value("gpu_memory_utilization", float),
    "tensor_parallel_size": flag(r"tensor_parallel_size=([0-9]+)", int)
    or yaml_value("tensor_parallel_size", int),
    "max_num_seqs": flag(r"max_num_seqs['\"]?:\s*([0-9]+)", int)
    or yaml_value("max_num_seqs", int),
    "prefix_caching": None,
    "enforce_eager": None,
    "trust_remote_code": None,
    "flashinfer_autotune": None,
    "sources": {},
}
doc.update(parse_serve_attention(text))
doc["isiro_attn"] = requested_isiro_attn()
if doc.get("attention_backend") is not None:
    doc["sources"]["attention_backend"] = "server_log"
doc["sources"]["isiro_attn"] = "env"
pc_flag = flag(r"enable_prefix_caching=(True|False)")
if pc_flag is not None:
    doc["prefix_caching"] = as_bool(pc_flag)
elif yaml_value("no_enable_prefix_caching") == "true":
    doc["prefix_caching"] = False
    doc["sources"]["prefix_caching"] = "serve_yaml_overlay"

eager_flag = flag(r"enforce_eager=(True|False)")
if eager_flag is not None:
    doc["enforce_eager"] = as_bool(eager_flag)
elif "Enforce eager set" in text or "Cudagraph is disabled under eager mode" in text:
    # Quiet TIC may hide the V1 config dump; eager warnings still print.
    doc["enforce_eager"] = True
    doc["sources"]["enforce_eager"] = "server_log_eager_warning"
elif re.search(r"CUDAGraphs? (is |are )?enabled", text):
    doc["enforce_eager"] = False
    doc["sources"]["enforce_eager"] = "server_log_graph_hint"

# Compile / cudagraph mode (fairness vs Hub FULL_AND_PIECEWISE).
# Logs look like: cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>
# or cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>
cg_mode = flag(
    r"cudagraph_mode['\"]?\s*[:=]\s*<CUDAGraphMode\.([A-Z_]+)"
) or flag(r"cudagraph_mode=<CUDAGraphMode\.([A-Z_]+)")
if cg_mode is not None:
    doc["cudagraph_mode"] = str(cg_mode)
    doc["sources"]["cudagraph_mode"] = "server_log"
# CompilationMode.VLLM_COMPILE: 3  or  mode': <CompilationMode.DYNAMO_TRACE_ONCE: 2>
comp_mode = flag(
    r"CompilationMode\.([A-Z_]+)\s*:\s*[0-9]+"
) or flag(r"'mode':\s*<CompilationMode\.([A-Z_]+)")
if comp_mode is not None:
    doc["compilation_mode"] = str(comp_mode)
    doc["sources"]["compilation_mode"] = "server_log"

tr_flag = flag(r"trust_remote_code=(True|False)")
if tr_flag is not None:
    doc["trust_remote_code"] = as_bool(tr_flag)
elif yaml_value("trust_remote_code") is not None:
    doc["trust_remote_code"] = as_bool(yaml_value("trust_remote_code"))
    doc["sources"]["trust_remote_code"] = "serve_yaml_overlay"

autotune = flag(r"enable_flashinfer_autotune=(True|False)")
if autotune is not None:
    doc["flashinfer_autotune"] = as_bool(autotune)
elif "Skipping FlashInfer autotune because it is disabled" in text:
    doc["flashinfer_autotune"] = False
elif "Using FlashInfer autotune cache file" in text:
    doc["flashinfer_autotune"] = True

for key, patterns in (
    ("max_model_len", (r"max_seq_len=([0-9]+)", r"max_model_len['\"]?:\s*([0-9]+)")),
    ("gpu_memory_utilization", (r"gpu_memory_utilization['\"]?:\s*([0-9.]+)",)),
    ("tensor_parallel_size", (r"tensor_parallel_size=([0-9]+)",)),
    ("max_num_seqs", (r"max_num_seqs['\"]?:\s*([0-9]+)",)),
):
    if any(re.search(pattern, text) for pattern in patterns):
        doc["sources"][key] = "server_log"
    elif yaml_path is not None and doc.get(key) is not None:
        doc["sources"][key] = "serve_yaml_overlay"
for key in (
    "prefix_caching",
    "enforce_eager",
    "trust_remote_code",
    "flashinfer_autotune",
    "cudagraph_mode",
    "compilation_mode",
):
    if doc.get(key) is not None and key not in doc["sources"]:
        doc["sources"][key] = "server_log"

required = (
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "max_num_seqs",
    "prefix_caching",
    "enforce_eager",
    "trust_remote_code",
    "flashinfer_autotune",
)
missing = [key for key in required if doc.get(key) is None]
if missing:
    raise SystemExit(f"could not read effective serve settings: {missing}")
if doc["prefix_caching"] is not False:
    raise SystemExit("prefix caching must be disabled for matched A/B")
if doc["enforce_eager"] is not expect_enforce_eager:
    raise SystemExit(
        f"enforce_eager mismatch: server reports {doc['enforce_eager']} "
        f"but matched A/B expects {expect_enforce_eager}"
    )
if doc["flashinfer_autotune"] is not False:
    raise SystemExit("flashinfer autotune must be disabled for matched A/B")
json.dump(doc, open(sys.argv[2], "w", encoding="utf-8"), indent=2, sort_keys=True)
print(sys.argv[2])
PY
}

capture_environment() {
  variant="$1"
  substrate="$2"
  image_arg=()
  identity_arg=()
  if [[ "${substrate}" == "docker" ]]; then
    image="$(docker inspect -f '{{index .Config.Image}}' "${BASELINE_CONTAINER}" 2>/dev/null || true)"
    [[ -n "${image}" ]] || image="${VLLM_IMAGE}"
    image_arg=(--image "${image}")
  fi
  # Scratch audit identity for TIC: runtime SHA, .tic digest, fused extension.
  # Optional ISIRO_RUNTIME_REPO points at an installed runtime checkout for SHA.
  if [[ "${variant}" == "isiro" ]]; then
    identity_arg=(
      --tic-model "${TIC_MODEL_DIR}"
      --probe-fused-extension
    )
    if [[ -n "${ISIRO_RUNTIME_REPO:-}" && -d "${ISIRO_RUNTIME_REPO}" ]]; then
      identity_arg+=(--runtime-repo "${ISIRO_RUNTIME_REPO}")
    fi
  fi
  python3 "${BENCH_ROOT}/scripts/collect_env.py" \
    --out "${RUN_DIR}/${variant}_environment.json" \
    --system-id "${SYSTEM_ID}" \
    --substrate "${substrate}" \
    --repo "${REPO_ROOT}" \
    --version-url "${API_URL}/version" \
    --isiro-version "${ISIRO_RUNTIME:-${ISIRO_FORMAT}}" \
    --quiet-host \
    "${image_arg[@]}" \
    "${identity_arg[@]}"
  python3 - "${RUN_DIR}/${variant}_environment.json" "$(gpu_process_bytes)" "${LOG_DIR}/${variant}-serve.log" "${variant}" "${GPU_PROCESS_BYTES_AFTER_READY:-0}" "${SERVE_GPU_MEMORY_UTILIZATION}" "${BENCH_ROOT}/scripts" <<'PY'
import json, math, re, sys
path, value, log_path, variant, ready_value, util_s, scripts = (
    sys.argv[1],
    int(sys.argv[2] or 0),
    sys.argv[3],
    sys.argv[4],
    int(sys.argv[5] or 0),
    sys.argv[6],
    sys.argv[7],
)
doc = json.load(open(path, encoding="utf-8"))
doc["gpu_process_memory_after_load_bytes_measured"] = value or None
if variant == "isiro" and ready_value:
    doc["gpu_process_memory_after_ready_bytes_measured"] = ready_value
text = open(log_path, encoding="utf-8", errors="replace").read()
gib = 1024 ** 3
sys.path.insert(0, scripts)
from needle_haystack import parse_kv_cache_gib, parse_kv_cache_tokens
pool_gib = pool_tokens = None
if "KV pool cap:" in text or "KV pool:" in text:
    try:
        pool_gib = parse_kv_cache_gib(text)
        pool_tokens = parse_kv_cache_tokens(text)
    except ValueError:
        pool_gib = pool_tokens = None
kv_gib = re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text)
kv_tokens = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
load_gib = re.search(r"Model loading took\s*([0-9.]+)\s*GiB", text)
# vLLM MemoryProfilingResult (DEBUG): Total non KV cache memory / requested.
# Note: vLLM format_gib often prints "11.76GiB" with no space before GiB.
non_kv_gib = re.search(r"Total non KV cache memory:\s*([0-9.]+)\s*GiB", text)
req_gib = re.search(
    r"worker requested memory:\s*([0-9.]+)\s*GiB",
    text,
) or re.search(
    r"Requested memory:\s*[0-9.]+ \(util\),\s*([0-9.]+)\s*GiB",
    text,
)
if pool_gib is not None:
    doc["kv_cache_memory_bytes_measured"] = round(float(pool_gib) * gib)
else:
    doc["kv_cache_memory_bytes_measured"] = (
        round(float(kv_gib.group(1)) * gib) if kv_gib else None
    )
if pool_tokens is not None:
    doc["kv_cache_tokens_measured"] = int(pool_tokens)
else:
    doc["kv_cache_tokens_measured"] = (
        int(kv_tokens.group(1).replace(",", "")) if kv_tokens else None
    )
doc["model_loading_bytes_reported"] = (
    round(float(load_gib.group(1)) * gib) if load_gib else None
)
if non_kv_gib:
    doc["non_kv_cache_memory_bytes_reported"] = round(
        float(non_kv_gib.group(1)) * gib
    )
if req_gib:
    doc["gpu_requested_memory_bytes_reported"] = round(
        float(req_gib.group(1)) * gib
    )
util = float(util_s) if util_s else None
doc["gpu_memory_utilization"] = util
# Prefer vLLM-logged requested (torch total × util). nvidia-smi total × util
# can disagree by hundreds of MiB on this host.
if (
    doc.get("non_kv_cache_memory_bytes_reported") is None
    and doc.get("kv_cache_memory_bytes_measured") is not None
    and doc.get("gpu_requested_memory_bytes_reported") is not None
):
    doc["non_kv_cache_memory_bytes_derived"] = int(
        doc["gpu_requested_memory_bytes_reported"]
    ) - int(doc["kv_cache_memory_bytes_measured"])
elif (
    doc.get("non_kv_cache_memory_bytes_reported") is None
    and doc.get("kv_cache_memory_bytes_measured") is not None
):
    gpus = doc.get("gpus") or []
    total_mib = gpus[0].get("memory_total_mib") if gpus else None
    if total_mib is not None and util is not None:
        # Audit fallback only; §B prefers DEBUG scrape / requested scrape.
        doc["gpu_requested_memory_bytes_from_smi_derived"] = math.ceil(
            int(total_mib) * 1024 * 1024 * util
        )
# Last TIC HBM probe line (post-ready / post-equalize): torch meters at probe.
probe_matches = re.findall(
    r"TIC HBM probe \(([^)]+)\): memory_allocated=([0-9.]+) GiB, "
    r"memory_reserved=([0-9.]+) GiB",
    text,
)
if probe_matches:
    reason, alloc_s, reserved_s = probe_matches[-1]
    doc["torch_memory_allocated_at_probe_bytes"] = round(
        float(alloc_s) * 1024 ** 3
    )
    doc["torch_memory_reserved_at_probe_bytes"] = round(
        float(reserved_s) * 1024 ** 3
    )
    doc["torch_memory_probe_reason"] = reason
# End-of-load account line (alloc/reserved at TIC attach).
acct = re.search(
    r"TIC HBM account \(once\):.*memory_allocated=([0-9.]+) GiB, "
    r"memory_reserved=([0-9.]+) GiB",
    text,
)
if acct:
    doc["torch_memory_allocated_at_load_bytes"] = round(
        float(acct.group(1)) * 1024 ** 3
    )
    doc["torch_memory_reserved_at_load_bytes"] = round(
        float(acct.group(2)) * 1024 ** 3
    )
# KV cache: serve uses vendor FlashAttention by default (ring KV is opt-in and
# byte-identical to bf16 today), so KV dtype carries no capacity signal. The
# report measures KV tokens/bytes directly (kv_cache object), not the dtype.
json.dump(doc, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
}

gpu_process_bytes() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits |
    awk '{sum += $1} END {printf "%.0f", sum * 1024 * 1024}'
}

run_profiles() {
  variant="$1"
  profile=""
  input_tokens=""
  output_tokens=""
  local nprompts
  nprompts="$(timed_num_prompts)"
  echo "profiles: prompts=${nprompts} conc=${BENCH_MAX_CONCURRENCY} (match_concurrency=${PROMPTS_MATCH_CONCURRENCY})" >&2
  declare -A _selected=()
  for name in "${SELECTED_PROFILES[@]}"; do
    _selected["${name}"]=1
  done
  while read -r profile input_tokens output_tokens; do
    [[ -n "${_selected[${profile}]:-}" ]] || continue
    python3 "${BENCH_ROOT}/scripts/run_bench.py" \
      --variant "${variant}" \
      --profile "${profile}" \
      --run-id "${RUN_ID}" \
      --model "${MODEL_ID}" \
      "${BENCH_TOKENIZER_ARGS[@]}" \
      --api-url "${API_URL}" \
      --input-tokens "${input_tokens}" \
      --output-tokens "${output_tokens}" \
      --num-warmups "${WARMUPS}" \
      --num-prompts "${nprompts}" \
      --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
      --seed "${BENCH_SEED}" \
      --replicates "${REPLICATES}" \
      --serve-config "${RUN_DIR}/${variant}_serve_config.json" \
      --raw-dir "${RAW_DIR}/${variant}" \
      --out "${RUN_DIR}/${variant}/${profile}.json" \
      --vllm-image "${VLLM_IMAGE}"
  done <<'PROFILES'
ttft-128 128 64
ttft-512 512 64
ttft-2048 2048 64
generation-32-256 32 256
generation-16384-256 16384 256
PROFILES
}

# After a full baseline cache copy, drop profiles that this run did not select so
# compare_ab sees matching baseline/TIC sets.
prune_unselected_baseline_profiles() {
  if [[ "${#SELECTED_PROFILES[@]}" -eq "${#ALL_PROFILES[@]}" ]]; then
    return 0
  fi
  declare -A _keep=()
  for name in "${SELECTED_PROFILES[@]}"; do
    _keep["${name}"]=1
  done
  for name in "${ALL_PROFILES[@]}"; do
    if [[ -z "${_keep[${name}]:-}" ]]; then
      rm -f "${RUN_DIR}/baseline/${name}.json"
    fi
  done
}

# Capacity binary search is eng-only (many cold starts). Public lean path keeps
# baseline at SERVE_MAX_NUM_SEQS and scales TIC max_num_seqs from matched-seqs
# vLLM KV tokens (§B probe; +1 TIC probe boot). Opt out: ISIRO_BENCH_KV_MEASURED=0
# (eng weight-estimate scale only). Mutual exclusion with capacity search.
CAPACITY_SEARCH_ENABLED=0
if [[ "${ISIRO_BENCH_CAPACITY_SEARCH:-0}" == "1" ]]; then
  CAPACITY_SEARCH_ENABLED=1
fi
CAPACITY_KV_MEASURED=1
if [[ "${ISIRO_BENCH_KV_MEASURED:-1}" == "0" ]]; then
  CAPACITY_KV_MEASURED=0
fi
if [[ "${CAPACITY_SEARCH_ENABLED}" -eq 1 && "${CAPACITY_KV_MEASURED}" -eq 1 ]]; then
  echo "ISIRO_BENCH_KV_MEASURED=1 cannot combine with ISIRO_BENCH_CAPACITY_SEARCH=1" >&2
  exit 2
fi
CAPACITY_KV_SCALE=0
_CAPACITY_HI_USER="${CAPACITY_SEARCH_HI-}"
CAPACITY_SEARCH_LO="${CAPACITY_SEARCH_LO:-1}"
if [[ -n "${_CAPACITY_HI_USER}" ]]; then
  CAPACITY_SEARCH_HI="${_CAPACITY_HI_USER}"
else
  CAPACITY_SEARCH_HI="${SERVE_MAX_NUM_SEQS}"
fi
# Equal-batch transparency (+cold starts only when operating point differs).
# Off by default in harness; launch_ab sets ISIRO_BENCH_EQUAL_BATCH=1.
CAPACITY_EQUAL_BATCH=0
if [[ "${ISIRO_BENCH_EQUAL_BATCH:-}" == "1" || "${ISIRO_BENCH_CAPACITY_EQUAL_BATCH:-}" == "1" ]]; then
  CAPACITY_EQUAL_BATCH=1
fi
if [[ "${ISIRO_BENCH_EQUAL_BATCH:-}" == "0" || "${ISIRO_BENCH_CAPACITY_EQUAL_BATCH:-}" == "0" ]]; then
  CAPACITY_EQUAL_BATCH=0
fi
BENCH_MAX_CONCURRENCY_CAP="${BENCH_MAX_CONCURRENCY}"
BASELINE_CAPACITY_SEQS="${SERVE_MAX_NUM_SEQS}"
TIC_CAPACITY_SEQS="${SERVE_MAX_NUM_SEQS}"
CAPACITY_SEQS_SCALE_MODE="fixed"
CAPACITY_KV_SCALE_RATIO_EST=""
CAPACITY_SEQS_IMPLIED_WEIGHT=""
CAPACITY_KV_SCALE_RATIO_MEAS=""
CAPACITY_SEQS_IMPLIED_KV_MEAS=""
CAPACITY_KV_MEASURED_SOURCE=""
# Timed A/B graph mode. Eng capacity *probes* force eager when search is on.
CAPACITY_TIMED_ENFORCE_EAGER="${SERVE_ENFORCE_EAGER}"
CAPACITY_TIMED_GRAPH_MODE="${GRAPH_MODE}"

read_env_kv_bytes() {
  local env_json="$1"
  python3 - "${env_json}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
value = doc.get("kv_cache_memory_bytes_measured")
if not isinstance(value, (int, float)) or float(value) <= 0:
    raise SystemExit(f"{sys.argv[1]}: kv_cache_memory_bytes_measured missing or non-positive")
print(int(value))
PY
}

read_env_kv_tokens() {
  local env_json="$1"
  python3 - "${env_json}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
value = doc.get("kv_cache_tokens_measured")
if not isinstance(value, (int, float)) or float(value) <= 0:
    raise SystemExit(f"{sys.argv[1]}: kv_cache_tokens_measured missing or non-positive")
print(int(value))
PY
}

read_serve_log_kv_tokens() {
  local serve_log="$1"
  python3 - "${BENCH_ROOT}/scripts" "${serve_log}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from needle_haystack import parse_kv_cache_tokens
text = open(sys.argv[2], encoding="utf-8", errors="replace").read()
print(parse_kv_cache_tokens(text))
PY
}

capacity_scale_hi_and_weights() {
  # Sets weight_b, weight_t, scale_hi (caller declares locals or uses globals).
  weight_b="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" weight-bytes \
      --path "${BASELINE_MODEL_DIR}"
  )"
  weight_t="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" weight-bytes \
      --path "${TIC_MODEL_DIR}/model.tic"
  )"
  if [[ -n "${CAPACITY_SCALE_HI:-}" ]]; then
    scale_hi="${CAPACITY_SCALE_HI}"
  else
    scale_hi="$(
      python3 "${BENCH_ROOT}/scripts/capacity_plan.py" default-scale-hi \
        --baseline-seqs "${BASELINE_CAPACITY_SEQS}" \
        --weight-baseline-bytes "${weight_b}" \
        --weight-tic-bytes "${weight_t}"
    )"
  fi
  if [[ "${scale_hi}" -lt "${BASELINE_CAPACITY_SEQS}" ]]; then
    echo "CAPACITY_SCALE_HI=${scale_hi} must be >= baseline max_num_seqs=${BASELINE_CAPACITY_SEQS}" >&2
    exit 2
  fi
}

stop_baseline_container() {
  docker rm -f "${BASELINE_CONTAINER}" >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do
    if ! api_port_listening; then
      return 0
    fi
    sleep 0.25
  done
}

probe_baseline_max_seqs() {
  local seqs="$1"
  local log="${LOG_DIR}/capacity-probe-baseline-${seqs}.log"
  SERVE_MAX_NUM_SEQS="${seqs}"
  # Eager probe: capacity is an HBM fit check, not a graphs timing claim.
  SERVE_ENFORCE_EAGER="true"
  build_baseline_serve
  stop_baseline_container
  ensure_api_port_free
  "${BASELINE_SERVE[@]}" >"${log}" 2>&1 &
  local pid=$!
  if wait_ready "${log}"; then
    stop_baseline_container
    wait "${pid}" 2>/dev/null || true
    return 0
  fi
  stop_baseline_container
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  return 1
}

probe_tic_max_seqs() {
  local seqs="$1"
  local log="${LOG_DIR}/capacity-probe-tic-${seqs}.log"
  SERVE_MAX_NUM_SEQS="${seqs}"
  prepare_tic_bundle
  # Capacity probes stay eager for fast boot; timed serve uses GRAPH_MODE below.
  export ISIRO_SERVE_ENFORCE_EAGER=1
  SERVE_ENFORCE_EAGER="true"
  if [[ -n "${TIC_PID}" ]] && kill -0 "${TIC_PID}" 2>/dev/null; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    TIC_PID=""
  fi
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  ensure_api_port_free
  : >"${log}"
  start_tic_serve "${log}"
  if wait_ready "${log}"; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    TIC_PID=""
    return 0
  fi
  kill "${TIC_PID}" 2>/dev/null || true
  wait "${TIC_PID}" 2>/dev/null || true
  TIC_PID=""
  return 1
}

binary_search_max_seqs() {
  local side="$1"
  local lo="${CAPACITY_SEARCH_LO}"
  local hi="${CAPACITY_SEARCH_HI}"
  local best=0
  echo "capacity search (${side}): lo=${lo} hi=${hi} (eager probes; hi-first)" >&2
  # Fast path: one probe when the configured ceiling fits.
  echo "capacity probe (${side}): max_num_seqs=${hi}" >&2
  if [[ "${side}" == "baseline" ]]; then
    if probe_baseline_max_seqs "${hi}"; then
      echo "capacity search (${side}): best max_num_seqs=${hi}" >&2
      printf '%s' "${hi}"
      return 0
    fi
  else
    if probe_tic_max_seqs "${hi}"; then
      echo "capacity search (${side}): best max_num_seqs=${hi}" >&2
      printf '%s' "${hi}"
      return 0
    fi
  fi
  if [[ "${hi}" -le "${lo}" ]]; then
    echo "capacity search (${side}) found no sustainable max_num_seqs in [${CAPACITY_SEARCH_LO}, ${CAPACITY_SEARCH_HI}]" >&2
    exit 2
  fi
  hi=$(( hi - 1 ))
  while [[ "${lo}" -le "${hi}" ]]; do
    local mid=$(( (lo + hi) / 2 ))
    echo "capacity probe (${side}): max_num_seqs=${mid}" >&2
    if [[ "${side}" == "baseline" ]]; then
      if probe_baseline_max_seqs "${mid}"; then
        best="${mid}"
        lo=$(( mid + 1 ))
      else
        hi=$(( mid - 1 ))
      fi
    else
      if probe_tic_max_seqs "${mid}"; then
        best="${mid}"
        lo=$(( mid + 1 ))
      else
        hi=$(( mid - 1 ))
      fi
    fi
  done
  if [[ "${best}" -lt 1 ]]; then
    echo "capacity search (${side}) found no sustainable max_num_seqs in [${CAPACITY_SEARCH_LO}, ${CAPACITY_SEARCH_HI}]" >&2
    exit 2
  fi
  echo "capacity search (${side}): best max_num_seqs=${best}" >&2
  printf '%s' "${best}"
}

restore_timed_graph_mode() {
  SERVE_ENFORCE_EAGER="${CAPACITY_TIMED_ENFORCE_EAGER}"
  GRAPH_MODE="${CAPACITY_TIMED_GRAPH_MODE}"
}

write_capacity_json() {
  local baseline_seqs="$1"
  local tic_seqs="$2"
  local baseline_conc="$3"
  local tic_conc="$4"
  python3 - \
    "${BENCH_ROOT}/scripts" \
    "${RUN_DIR}/capacity.json" \
    "${SERVE_GPU_MEMORY_UTILIZATION}" \
    "${SERVE_MAX_MODEL_LEN}" \
    "${baseline_seqs}" \
    "${tic_seqs}" \
    "${baseline_conc}" \
    "${tic_conc}" \
    "${CAPACITY_SEARCH_LO}" \
    "${CAPACITY_SEARCH_HI}" \
    "${GRAPH_MODE}" \
    "${CAPACITY_SEQS_SCALE_MODE}" \
    "${CAPACITY_KV_SCALE_RATIO_EST}" \
    "${CAPACITY_SEQS_IMPLIED_WEIGHT}" \
    "${CAPACITY_KV_SCALE_RATIO_MEAS}" \
    "${CAPACITY_SEQS_IMPLIED_KV_MEAS}" \
    "${CAPACITY_KV_MEASURED_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

(
    scripts,
    out,
    util,
    max_len,
    b_seqs,
    t_seqs,
    b_conc,
    t_conc,
    lo,
    hi,
    graph,
    scale_mode,
    kv_ratio_est,
    seqs_weight,
    kv_ratio_meas,
    seqs_kv,
    kv_src,
) = sys.argv[1:18]
sys.path.insert(0, scripts)
from capacity_plan import build_capacity_doc

kwargs = {}
if scale_mode:
    kwargs["seqs_scale_mode"] = scale_mode
if kv_ratio_est:
    kwargs["kv_scale_ratio_estimated"] = float(kv_ratio_est)
if seqs_weight:
    kwargs["seqs_implied_weight_estimate"] = int(seqs_weight)
if kv_ratio_meas:
    kwargs["kv_scale_ratio_measured"] = float(kv_ratio_meas)
if seqs_kv:
    kwargs["seqs_implied_kv_measured"] = int(seqs_kv)
if kv_src:
    kwargs["kv_measured_source"] = kv_src
doc = build_capacity_doc(
    gpu_memory_utilization=float(util),
    max_model_len=int(max_len),
    baseline_max_num_seqs=int(b_seqs),
    tic_max_num_seqs=int(t_seqs),
    baseline_max_concurrency=int(b_conc),
    tic_max_concurrency=int(t_conc),
    search_lo=int(lo),
    search_hi=int(hi),
    graph_mode=graph,
    **kwargs,
)
Path(out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY
}

_apply_tic_conc_from_seqs() {
  local effective_cap
  if [[ "${TIC_CAPACITY_SEQS}" -gt "${BENCH_MAX_CONCURRENCY_CAP}" ]]; then
    effective_cap="${TIC_CAPACITY_SEQS}"
  else
    effective_cap="${BENCH_MAX_CONCURRENCY_CAP}"
  fi
  TIC_CONC="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" concurrency \
      --max-num-seqs "${TIC_CAPACITY_SEQS}" \
      --bench-cap "${effective_cap}"
  )"
}

scale_tic_seqs_from_baseline_kv() {
  # Raise TIC max_num_seqs from baseline KV + on-disk weight bytes freed.
  local kv_bytes weight_b weight_t scale_hi ratio
  kv_bytes="$(read_env_kv_bytes "${RUN_DIR}/baseline_environment.json")"
  capacity_scale_hi_and_weights
  TIC_CAPACITY_SEQS="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" scale-seqs \
      --baseline-seqs "${BASELINE_CAPACITY_SEQS}" \
      --kv-baseline-bytes "${kv_bytes}" \
      --weight-baseline-bytes "${weight_b}" \
      --weight-tic-bytes "${weight_t}" \
      --scale-hi "${scale_hi}"
  )"
  ratio="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" kv-scale-ratio \
      --kv-baseline-bytes "${kv_bytes}" \
      --weight-baseline-bytes "${weight_b}" \
      --weight-tic-bytes "${weight_t}"
  )"
  CAPACITY_SEQS_SCALE_MODE="kv_estimate"
  CAPACITY_KV_SCALE_RATIO_EST="${ratio}"
  CAPACITY_SEQS_IMPLIED_WEIGHT="${TIC_CAPACITY_SEQS}"
  CAPACITY_KV_SCALE_RATIO_MEAS=""
  CAPACITY_SEQS_IMPLIED_KV_MEAS=""
  CAPACITY_KV_MEASURED_SOURCE=""
  CAPACITY_SEARCH_HI="${scale_hi}"
  _apply_tic_conc_from_seqs
  write_capacity_json "${BASELINE_CAPACITY_SEQS}" "${TIC_CAPACITY_SEQS}" \
    "${BASELINE_CONC}" "${TIC_CONC}"
  echo "capacity KV-scale (weight-estimate): baseline_seqs=${BASELINE_CAPACITY_SEQS} tic_seqs=${TIC_CAPACITY_SEQS} conc=${BASELINE_CONC}/${TIC_CONC} est_kv_ratio=${ratio} scale_hi=${scale_hi}" >&2
}

scale_tic_seqs_from_measured_kv() {
  # Opt-in: matched-seqs TIC probe, then scale from vLLM KV token ratio.
  local kv_bytes weight_b weight_t scale_hi ratio_est kv_base_tokens
  local probe_log probe_pid kv_tic_tokens ratio_meas saved_seqs
  kv_bytes="$(read_env_kv_bytes "${RUN_DIR}/baseline_environment.json")"
  kv_base_tokens="$(read_env_kv_tokens "${RUN_DIR}/baseline_environment.json")"
  capacity_scale_hi_and_weights
  ratio_est="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" kv-scale-ratio \
      --kv-baseline-bytes "${kv_bytes}" \
      --weight-baseline-bytes "${weight_b}" \
      --weight-tic-bytes "${weight_t}"
  )"
  CAPACITY_SEQS_IMPLIED_WEIGHT="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" scale-seqs \
      --baseline-seqs "${BASELINE_CAPACITY_SEQS}" \
      --kv-baseline-bytes "${kv_bytes}" \
      --weight-baseline-bytes "${weight_b}" \
      --weight-tic-bytes "${weight_t}" \
      --scale-hi "${scale_hi}"
  )"
  CAPACITY_KV_SCALE_RATIO_EST="${ratio_est}"
  CAPACITY_SEARCH_HI="${scale_hi}"

  saved_seqs="${SERVE_MAX_NUM_SEQS}"
  SERVE_MAX_NUM_SEQS="${BASELINE_CAPACITY_SEQS}"
  # Matched KV probe stays at vendor default batched tokens.
  SERVE_MAX_NUM_BATCHED_TOKENS=""
  prepare_tic_bundle
  # Matched probe uses the timed graph mode (not eager capacity-search probes).
  if [[ "${SERVE_ENFORCE_EAGER}" == "false" ]]; then
    export ISIRO_SERVE_ENFORCE_EAGER=0
  else
    export ISIRO_SERVE_ENFORCE_EAGER=1
  fi
  probe_log="${LOG_DIR}/capacity-kv-measured-probe.log"
  if [[ -n "${TIC_PID}" ]] && kill -0 "${TIC_PID}" 2>/dev/null; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    TIC_PID=""
  fi
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  ensure_api_port_free
  : >"${probe_log}"
  echo "capacity KV-measured probe: max_num_seqs=${BASELINE_CAPACITY_SEQS} (matched)" >&2
  start_tic_serve "${probe_log}"
  probe_pid="${TIC_PID}"
  if ! wait_ready "${probe_log}" "${probe_pid}"; then
    echo "TIC KV-measured probe failed; see ${probe_log}" >&2
    kill "${probe_pid}" 2>/dev/null || true
    wait "${probe_pid}" 2>/dev/null || true
    TIC_PID=""
    SERVE_MAX_NUM_SEQS="${saved_seqs}"
    exit 2
  fi
  kv_tic_tokens="$(read_serve_log_kv_tokens "${probe_log}")"
  kill "${probe_pid}" 2>/dev/null || true
  wait "${probe_pid}" 2>/dev/null || true
  TIC_PID=""
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  SERVE_MAX_NUM_SEQS="${saved_seqs}"

  TIC_CAPACITY_SEQS="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" scale-seqs-measured \
      --baseline-seqs "${BASELINE_CAPACITY_SEQS}" \
      --kv-baseline-tokens "${kv_base_tokens}" \
      --kv-tic-tokens "${kv_tic_tokens}" \
      --scale-hi "${scale_hi}"
  )"
  ratio_meas="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" kv-scale-ratio-measured \
      --kv-baseline-tokens "${kv_base_tokens}" \
      --kv-tic-tokens "${kv_tic_tokens}"
  )"
  CAPACITY_SEQS_SCALE_MODE="kv_measured"
  CAPACITY_KV_SCALE_RATIO_MEAS="${ratio_meas}"
  CAPACITY_SEQS_IMPLIED_KV_MEAS="${TIC_CAPACITY_SEQS}"
  CAPACITY_KV_MEASURED_SOURCE="matched_probe"
  _apply_tic_conc_from_seqs
  write_capacity_json "${BASELINE_CAPACITY_SEQS}" "${TIC_CAPACITY_SEQS}" \
    "${BASELINE_CONC}" "${TIC_CONC}"
  echo "capacity KV-scale (measured): baseline_seqs=${BASELINE_CAPACITY_SEQS} tic_seqs=${TIC_CAPACITY_SEQS} conc=${BASELINE_CONC}/${TIC_CONC} meas_kv_ratio=${ratio_meas} weight_implied=${CAPACITY_SEQS_IMPLIED_WEIGHT} scale_hi=${scale_hi}" >&2
}

if [[ "${EXPERIMENT_KIND}" == "capacity" ]]; then
  if [[ "${CAPACITY_SEARCH_ENABLED}" -eq 1 ]]; then
    require_quiet_gpu
    BASELINE_CAPACITY_SEQS="$(binary_search_max_seqs baseline)"
    CAPACITY_SEQS_SCALE_MODE="search"
  else
    if [[ "${CAPACITY_KV_MEASURED}" -eq 1 ]]; then
      echo "capacity operating point: baseline max_num_seqs=${SERVE_MAX_NUM_SEQS}; TIC measured-KV scaled (opt-in)" >&2
      CAPACITY_SEQS_SCALE_MODE="kv_measured"
    else
      echo "capacity operating point: baseline max_num_seqs=${SERVE_MAX_NUM_SEQS}; TIC KV-scaled from weight estimate (default)" >&2
      CAPACITY_SEQS_SCALE_MODE="kv_estimate"
    fi
    BASELINE_CAPACITY_SEQS="${SERVE_MAX_NUM_SEQS}"
    TIC_CAPACITY_SEQS="${SERVE_MAX_NUM_SEQS}"
    CAPACITY_SEARCH_LO="${SERVE_MAX_NUM_SEQS}"
    CAPACITY_KV_SCALE=1
  fi
  if [[ "${CAPACITY_SEARCH_ENABLED}" -eq 1 ]]; then
    require_quiet_gpu
    TIC_CAPACITY_SEQS="$(binary_search_max_seqs tic)"
    restore_timed_graph_mode
  fi
  BASELINE_CONC="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" concurrency \
      --max-num-seqs "${BASELINE_CAPACITY_SEQS}" \
      --bench-cap "${BENCH_MAX_CONCURRENCY_CAP}"
  )"
  if [[ "${CAPACITY_KV_SCALE}" -eq 0 ]]; then
    TIC_CONC="$(
      python3 "${BENCH_ROOT}/scripts/capacity_plan.py" concurrency \
        --max-num-seqs "${TIC_CAPACITY_SEQS}" \
        --bench-cap "${BENCH_MAX_CONCURRENCY_CAP}"
    )"
    write_capacity_json "${BASELINE_CAPACITY_SEQS}" "${TIC_CAPACITY_SEQS}" \
      "${BASELINE_CONC}" "${TIC_CONC}"
  else
    # Provisional until baseline KV is known.
    TIC_CONC="${BASELINE_CONC}"
    write_capacity_json "${BASELINE_CAPACITY_SEQS}" "${TIC_CAPACITY_SEQS}" \
      "${BASELINE_CONC}" "${TIC_CONC}"
  fi
fi

# Timed arms. Default: baseline then Isiro. ISIRO_FIRST=1: Isiro then cold baseline.
# Capacity KV-scale still needs baseline_environment.json first (env scrape only).
run_timed_baseline_arm() {
  SERVE_MAX_NUM_SEQS="${BASELINE_CAPACITY_SEQS}"
  BENCH_MAX_CONCURRENCY="${BASELINE_CONC:-${BENCH_MAX_CONCURRENCY_CAP}}"
  if [[ "${EXPERIMENT_KIND}" == "equal-batch" ]]; then
    BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY_CAP}"
    SERVE_MAX_NUM_BATCHED_TOKENS=""
  else
    apply_capacity_one_wave_batched_tokens
  fi
  build_baseline_serve
  prepare_tic_bundle

  BASELINE_SKIPPED=0
  if [[ "${REUSE_BASELINE}" -eq 1 ]] && try_reuse_baseline; then
    BASELINE_SKIPPED=1
    prune_unselected_baseline_profiles
  else
    write_baseline_fingerprint >/dev/null
    require_quiet_gpu
    ensure_api_port_free
    "${BASELINE_SERVE[@]}" >"${LOG_DIR}/baseline-serve.log" 2>&1 &
    BASELINE_PID=$!
    wait_ready "${LOG_DIR}/baseline-serve.log" || {
      echo "baseline serve failed; see ${LOG_DIR}/baseline-serve.log" >&2
      tail -n 80 "${LOG_DIR}/baseline-serve.log" >&2 || true
      kill "${BASELINE_PID}" 2>/dev/null || true
      exit 2
    }
    write_effective_config baseline
    equalize_warmup baseline
    capture_serve_output_match baseline
    capture_environment baseline docker
    BENCH_MAX_CONCURRENCY="${BASELINE_CONC:-${BENCH_MAX_CONCURRENCY}}"
    run_profiles baseline
    stop_baseline_container
    write_baseline_fingerprint >/dev/null
  fi
}

scrape_baseline_environment_for_scale() {
  # Boot baseline long enough to scrape KV env for TIC capacity scale, then stop.
  # Used only when ISIRO_FIRST=1 so timed Isiro can still KV-scale.
  # Log name must be baseline-serve.log: capture_environment reads that path.
  # When SKIP_BASELINE + BASELINE_REF already have env JSON, reuse (no boot).
  if [[ -n "${ISIRO_BENCH_BASELINE_REF:-}" && -f "${ISIRO_BENCH_BASELINE_REF}/baseline_environment.json" ]]; then
    cp -a "${ISIRO_BENCH_BASELINE_REF}/baseline_environment.json" \
      "${RUN_DIR}/baseline_environment.json"
    echo "isiro-first: reused baseline_environment.json from ${ISIRO_BENCH_BASELINE_REF} (no scrape)" >&2
    return 0
  fi
  SERVE_MAX_NUM_SEQS="${BASELINE_CAPACITY_SEQS}"
  build_baseline_serve
  require_quiet_gpu
  ensure_api_port_free
  echo "isiro-first: baseline env scrape for KV scale (no timed profiles)" >&2
  "${BASELINE_SERVE[@]}" >"${LOG_DIR}/baseline-serve.log" 2>&1 &
  BASELINE_PID=$!
  wait_ready "${LOG_DIR}/baseline-serve.log" || {
    echo "baseline env scrape failed; see ${LOG_DIR}/baseline-serve.log" >&2
    tail -n 80 "${LOG_DIR}/baseline-serve.log" >&2 || true
    kill "${BASELINE_PID}" 2>/dev/null || true
    exit 2
  }
  capture_environment baseline docker
  stop_baseline_container
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  sleep 2
}

run_timed_isiro_arm() {
  SERVE_MAX_NUM_SEQS="${TIC_CAPACITY_SEQS}"
  if [[ "${EXPERIMENT_KIND}" == "capacity" ]]; then
    BENCH_MAX_CONCURRENCY="${TIC_CONC}"
    apply_capacity_one_wave_batched_tokens
  else
    BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY_CAP}"
    SERVE_MAX_NUM_BATCHED_TOKENS=""
  fi
  prepare_tic_bundle

  # Select the TIC CUDA-graph mode to match the baseline A/B.
  # Product serve defaults to outer graphs ON; --enforce-eager sets ISIRO_SERVE_ENFORCE_EAGER=1.
  if [[ "${SERVE_ENFORCE_EAGER}" == "false" ]]; then
    export ISIRO_SERVE_ENFORCE_EAGER=0
  else
    export ISIRO_SERVE_ENFORCE_EAGER=1
  fi
  # Drop inherited fused-kernel override env from parent shells so serve A/B stays clean.
  while IFS= read -r _knob; do
    unset "${_knob}" 2>/dev/null || true
  done < <(compgen -v | grep -E '^ISIRO_PF_' || true)
  # DEBUG: scrape vLLM non_kv_cache_memory into environment.json (§B).
  : "${VLLM_LOGGING_LEVEL:=DEBUG}"
  export VLLM_LOGGING_LEVEL
  # Eng HBM meters (account/probe/reclaim) are opt-in; needed for torch scrape.
  : "${ISIRO_HBM_TRACE:=1}"
  export ISIRO_HBM_TRACE
  # Docker-proxy / prior engine can linger after baseline; settle then start once.
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  sleep 8
  ensure_api_port_free
  # Bind probe: ss can report free while docker-proxy still owns the port briefly.
  for _ in $(seq 1 40); do
    if python3 - "${API_PORT}" <<'PY'
import errno, socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("", port))
except OSError as exc:
    sys.exit(0 if exc.errno != errno.EADDRINUSE else 1)
finally:
    s.close()
sys.exit(0)
PY
    then
      break
    fi
    fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
    if command -v docker >/dev/null 2>&1; then
      docker ps -aq --filter "name=isiro-bench-" | xargs -r docker rm -f >/dev/null 2>&1 || true
      docker ps -aq --filter "publish=${API_PORT}" | xargs -r docker rm -f >/dev/null 2>&1 || true
    fi
    sleep 0.5
  done
  # Path for post-equalize HBM reclaim (touched after equalize).
  ISIRO_HBM_RECLAIM_FLAG="${LOG_DIR}/isiro-hbm-reclaim.flag"
  rm -f "${ISIRO_HBM_RECLAIM_FLAG}"
  export ISIRO_HBM_RECLAIM_FLAG
  : >"${LOG_DIR}/isiro-serve.log"
  start_tic_serve "${LOG_DIR}/isiro-serve.log"
  # SERV-0101 often means a still-dying prior holder; one delayed retry only.
  if ! wait_ready "${LOG_DIR}/isiro-serve.log" "${TIC_PID}"; then
    echo "TIC serve not ready; delayed retry after port cleanup" >&2
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
    sleep 8
    ensure_api_port_free
    : >"${LOG_DIR}/isiro-serve.log"
    start_tic_serve "${LOG_DIR}/isiro-serve.log"
    wait_ready "${LOG_DIR}/isiro-serve.log" "${TIC_PID}" || {
      echo "TIC serve failed; see ${LOG_DIR}/isiro-serve.log" >&2
      if [[ "${CAPACITY_KV_SCALE}" -eq 1 ]]; then
        echo "KV-scaled max_num_seqs=${TIC_CAPACITY_SEQS} did not fit. Lower SERVE_MAX_NUM_SEQS or CAPACITY_SCALE_HI (fail closed; no matched fallback)." >&2
      fi
      tail -n 80 "${LOG_DIR}/isiro-serve.log" >&2 || true
      exit 2
    }
  fi
  # Same-instant audit: process RSS at ready (before equalize growth).
  # Serve now reclaims warm scratch before READY; this should track TIC load.
  GPU_PROCESS_BYTES_AFTER_READY="$(gpu_process_bytes)"
  export GPU_PROCESS_BYTES_AFTER_READY
  assert_tic_weight_hbm_win "${LOG_DIR}/isiro-serve.log" || {
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    exit 3
  }
  write_effective_config isiro "${TIC_BENCH_BUNDLE}/serve.yaml"
  equalize_warmup isiro
  # Arm CUDA-thread post-equalize reclaim on the next TIC forward.
  : >"${ISIRO_HBM_RECLAIM_FLAG}"
  capture_serve_output_match isiro
  # §B non-KV uses vLLM non_kv_cache_memory (scraped/derived in capture).
  # process-at-ready stays as audit; post-traffic process is also audit-only.
  capture_environment isiro host_isiro
  BENCH_MAX_CONCURRENCY="${TIC_CONC:-${BENCH_MAX_CONCURRENCY}}"
  run_profiles isiro
  # Keep capacity serve log before equal-batch restart overwrites isiro-serve.log.
  if [[ "${EXPERIMENT_KIND}" == "capacity" && -f "${LOG_DIR}/isiro-serve.log" ]]; then
    cp -f "${LOG_DIR}/isiro-serve.log" "${LOG_DIR}/capacity-isiro-serve.log"
  fi
  kill "${TIC_PID}" 2>/dev/null || true
  wait "${TIC_PID}" 2>/dev/null || true
  TIC_PID=""
}

apply_capacity_kv_scale() {
  if [[ "${EXPERIMENT_KIND}" == "capacity" && "${CAPACITY_KV_SCALE}" -eq 1 ]]; then
    if [[ "${CAPACITY_KV_MEASURED}" -eq 1 ]]; then
      scale_tic_seqs_from_measured_kv
    else
      scale_tic_seqs_from_baseline_kv
    fi
  fi
}

build_baseline_serve
prepare_tic_bundle
BASELINE_SKIPPED=0

if [[ "${ISIRO_FIRST}" -eq 1 ]]; then
  echo "timed arm order: Isiro then cold baseline (ISIRO_FIRST=1)" >&2
  if [[ "${EXPERIMENT_KIND}" == "capacity" && "${CAPACITY_KV_SCALE}" -eq 1 ]]; then
    scrape_baseline_environment_for_scale
    apply_capacity_kv_scale
  fi
  write_commands_json
  run_timed_isiro_arm
  if [[ "${SKIP_BASELINE}" -eq 1 ]]; then
    echo "baseline: SKIP (ISIRO_BENCH_SKIP_BASELINE=1)" >&2
    BASELINE_SKIPPED=1
    # compare_ab needs baseline_* JSON; reuse a prior equal-batch arm when set.
    if [[ -n "${ISIRO_BENCH_BASELINE_REF:-}" && -d "${ISIRO_BENCH_BASELINE_REF}" ]]; then
      for _ref in \
        baseline_environment.json \
        baseline/generation-32-256.json \
        baseline_serve_config.json \
        baseline_fingerprint.json
      do
        if [[ -e "${ISIRO_BENCH_BASELINE_REF}/${_ref}" ]]; then
          mkdir -p "$(dirname "${RUN_DIR}/${_ref}")"
          cp -a "${ISIRO_BENCH_BASELINE_REF}/${_ref}" "${RUN_DIR}/${_ref}"
        fi
      done
      echo "baseline: reused artifacts from ${ISIRO_BENCH_BASELINE_REF}" >&2
    fi
  else
    # Cold baseline on the product stack (no reuse after Isiro).
    REUSE_BASELINE=0
    run_timed_baseline_arm
  fi
else
  echo "timed arm order: baseline then Isiro" >&2
  if [[ "${SKIP_BASELINE}" -eq 1 ]]; then
    echo "baseline: SKIP (ISIRO_BENCH_SKIP_BASELINE=1); running Isiro only" >&2
    BASELINE_SKIPPED=1
    write_commands_json
    run_timed_isiro_arm
  else
    run_timed_baseline_arm
    apply_capacity_kv_scale
    write_commands_json
    run_timed_isiro_arm
  fi
fi

# Capacity publish: matched equal-batch decode (transparency; capacity benefit not active).
# Same order for every graph mode. Reuse capacity timed profiles when seqs+concurrency
# already match (no mode-specific branching).
copy_equal_batch_profile_from_capacity() {
  local variant="$1"
  local src_profile="${RUN_DIR}/${variant}/generation-32-256.json"
  local dst_profile="${RUN_DIR}/equal_batch/${variant}/generation-32-256.json"
  local src_cfg="${RUN_DIR}/${variant}_serve_config.json"
  local dst_cfg="${RUN_DIR}/equal_batch/${variant}_serve_config.json"
  if [[ ! -f "${src_profile}" ]]; then
    echo "equal-batch ${variant}: missing capacity profile ${src_profile}" >&2
    return 1
  fi
  mkdir -p "${RUN_DIR}/equal_batch/${variant}" "${RAW_DIR}/equal_batch/${variant}"
  cp -f "${src_profile}" "${dst_profile}"
  if [[ -f "${src_cfg}" ]]; then
    cp -f "${src_cfg}" \
      "${RUN_DIR}/equal_batch/${variant}_serve_config.capacity.json"
    cp -f "${src_cfg}" "${dst_cfg}"
  fi
  return 0
}

run_equal_batch_transparency() {
  local matched_seqs="$1"
  local matched_conc="$2"
  local capacity_base_seqs="${BASELINE_CAPACITY_SEQS}"
  local capacity_base_conc="${BASELINE_CONC:-}"
  local capacity_tic_seqs="${TIC_CAPACITY_SEQS}"
  local capacity_tic_conc="${TIC_CONC:-}"
  local reuse_baseline=0
  local reuse_tic=0

  mkdir -p \
    "${RUN_DIR}/equal_batch/baseline" \
    "${RUN_DIR}/equal_batch/isiro" \
    "${RAW_DIR}/equal_batch/baseline" \
    "${RAW_DIR}/equal_batch/isiro"
  python3 - \
    "${RUN_DIR}/equal_batch.json" \
    "${matched_seqs}" \
    "${matched_conc}" \
    "${SERVE_GPU_MEMORY_UTILIZATION}" \
    "${SERVE_MAX_MODEL_LEN}" \
    "${GRAPH_MODE}" <<'PY'
import json, sys
from pathlib import Path
out, seqs, conc, util, max_len, graph = sys.argv[1:7]
doc = {
    "schema": "isiro-benchmark-equal-batch-v1",
    "label": "capacity benefit not active",
    "max_num_seqs": int(seqs),
    "max_concurrency": int(conc),
    "gpu_memory_utilization": float(util),
    "max_model_len": int(max_len),
    "graph_mode": graph,
}
Path(out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY

  if [[ "${matched_seqs}" == "${capacity_base_seqs}" && \
        -n "${capacity_base_conc}" && \
        "${matched_conc}" == "${capacity_base_conc}" ]]; then
    # Reuse only when capacity timed prompts already match concurrency
    # (PROMPTS_MATCH_CONCURRENCY). Else fairness fails on num_prompts.
    _cap_prompts="$(
      python3 -c "import json;print(json.load(open('${RUN_DIR}/baseline/generation-32-256.json')).get('bench_config',{}).get('num_prompts',''))" 2>/dev/null || true
    )"
    if [[ "${_cap_prompts}" == "${matched_conc}" ]] && \
       copy_equal_batch_profile_from_capacity baseline; then
      reuse_baseline=1
      echo "equal-batch baseline: reused capacity timed profiles" >&2
    fi
  fi
  if [[ "${reuse_baseline}" -eq 0 ]]; then
    if [[ "${SKIP_BASELINE}" -eq 1 ]]; then
      # Never cold-start baseline when the operator skipped it; reuse REF.
      _eq_ref_base=""
      if [[ -n "${ISIRO_BENCH_BASELINE_REF:-}" ]]; then
        for _cand in \
          "${ISIRO_BENCH_BASELINE_REF}/equal_batch/baseline/generation-32-256.json" \
          "${ISIRO_BENCH_BASELINE_REF}/baseline/generation-32-256.json"
        do
          if [[ -f "${_cand}" ]]; then
            _eq_ref_base="${_cand}"
            break
          fi
        done
      fi
      if [[ -n "${_eq_ref_base}" ]]; then
        mkdir -p "${RUN_DIR}/equal_batch/baseline"
        cp -f "${_eq_ref_base}" \
          "${RUN_DIR}/equal_batch/baseline/generation-32-256.json"
        echo "equal-batch baseline: SKIP cold start; reused ${_eq_ref_base}" >&2
        reuse_baseline=1
      else
        echo "equal-batch baseline: SKIP (ISIRO_BENCH_SKIP_BASELINE=1; no REF profile)" >&2
      fi
    else
      echo "equal-batch baseline: cold start" >&2
      SERVE_MAX_NUM_SEQS="${matched_seqs}"
      BENCH_MAX_CONCURRENCY="${matched_conc}"
      SERVE_MAX_NUM_BATCHED_TOKENS=""
      build_baseline_serve
      require_quiet_gpu
      ensure_api_port_free
      "${BASELINE_SERVE[@]}" >"${LOG_DIR}/equal-batch-baseline-serve.log" 2>&1 &
      BASELINE_PID=$!
      wait_ready "${LOG_DIR}/equal-batch-baseline-serve.log" || {
        echo "equal-batch baseline serve failed; see ${LOG_DIR}/equal-batch-baseline-serve.log" >&2
        tail -n 80 "${LOG_DIR}/equal-batch-baseline-serve.log" >&2 || true
        kill "${BASELINE_PID}" 2>/dev/null || true
        exit 2
      }
      if [[ -f "${RUN_DIR}/baseline_serve_config.json" ]]; then
        cp -f "${RUN_DIR}/baseline_serve_config.json" \
          "${RUN_DIR}/equal_batch/baseline_serve_config.capacity.json"
      fi
      cp -f "${LOG_DIR}/equal-batch-baseline-serve.log" "${LOG_DIR}/baseline-serve.log"
      write_effective_config baseline
      cp -f "${RUN_DIR}/baseline_serve_config.json" \
        "${RUN_DIR}/equal_batch/baseline_serve_config.json"
      equalize_warmup baseline
      python3 "${BENCH_ROOT}/scripts/run_bench.py" \
        --variant baseline \
        --profile generation-32-256 \
        --run-id "${RUN_ID}" \
        --model "${MODEL_ID}" \
        "${BENCH_TOKENIZER_ARGS[@]}" \
        --api-url "${API_URL}" \
        --input-tokens 32 \
        --output-tokens 256 \
        --num-warmups "${WARMUPS}" \
        --num-prompts "$(timed_num_prompts)" \
        --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
        --seed "${BENCH_SEED}" \
        --replicates "${REPLICATES}" \
        --serve-config "${RUN_DIR}/equal_batch/baseline_serve_config.json" \
        --raw-dir "${RAW_DIR}/equal_batch/baseline" \
        --out "${RUN_DIR}/equal_batch/baseline/generation-32-256.json" \
        --vllm-image "${VLLM_IMAGE}"
      stop_baseline_container
    fi
  fi

  if [[ "${matched_seqs}" == "${capacity_tic_seqs}" && \
        -n "${capacity_tic_conc}" && \
        "${matched_conc}" == "${capacity_tic_conc}" ]]; then
    _cap_prompts="$(
      python3 -c "import json;print(json.load(open('${RUN_DIR}/isiro/generation-32-256.json')).get('bench_config',{}).get('num_prompts',''))" 2>/dev/null || true
    )"
    if [[ "${_cap_prompts}" == "${matched_conc}" ]] && \
       copy_equal_batch_profile_from_capacity isiro; then
      reuse_tic=1
      echo "equal-batch TIC: reused capacity timed profiles" >&2
    fi
  fi
  if [[ "${reuse_tic}" -eq 0 ]]; then
    echo "equal-batch TIC: cold start" >&2
    SERVE_MAX_NUM_SEQS="${matched_seqs}"
    BENCH_MAX_CONCURRENCY="${matched_conc}"
    # Equal-batch keeps vendor default batched tokens (no capacity one-wave overlay).
    SERVE_MAX_NUM_BATCHED_TOKENS=""
    prepare_tic_bundle
    if [[ "${SERVE_ENFORCE_EAGER}" == "false" ]]; then
      export ISIRO_SERVE_ENFORCE_EAGER=0
    else
      export ISIRO_SERVE_ENFORCE_EAGER=1
    fi
    fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
    sleep 3
    ensure_api_port_free
    : >"${LOG_DIR}/equal-batch-isiro-serve.log"
    start_tic_serve "${LOG_DIR}/equal-batch-isiro-serve.log"
    if ! wait_ready "${LOG_DIR}/equal-batch-isiro-serve.log" "${TIC_PID}"; then
      echo "equal-batch TIC serve not ready; delayed retry" >&2
      kill "${TIC_PID}" 2>/dev/null || true
      wait "${TIC_PID}" 2>/dev/null || true
      fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
      sleep 5
      ensure_api_port_free
      : >"${LOG_DIR}/equal-batch-isiro-serve.log"
      start_tic_serve "${LOG_DIR}/equal-batch-isiro-serve.log"
      wait_ready "${LOG_DIR}/equal-batch-isiro-serve.log" "${TIC_PID}" || {
        echo "equal-batch TIC serve failed; see ${LOG_DIR}/equal-batch-isiro-serve.log" >&2
        tail -n 80 "${LOG_DIR}/equal-batch-isiro-serve.log" >&2 || true
        exit 2
      }
    fi
    if [[ -f "${RUN_DIR}/isiro_serve_config.json" ]]; then
      cp -f "${RUN_DIR}/isiro_serve_config.json" \
        "${RUN_DIR}/equal_batch/isiro_serve_config.capacity.json"
    fi
    cp -f "${LOG_DIR}/equal-batch-isiro-serve.log" "${LOG_DIR}/isiro-serve.log"
    assert_tic_weight_hbm_win "${LOG_DIR}/equal-batch-isiro-serve.log" || {
      kill "${TIC_PID}" 2>/dev/null || true
      wait "${TIC_PID}" 2>/dev/null || true
      TIC_PID=""
      exit 3
    }
    write_effective_config isiro "${TIC_BENCH_BUNDLE}/serve.yaml"
    cp -f "${RUN_DIR}/isiro_serve_config.json" \
      "${RUN_DIR}/equal_batch/isiro_serve_config.json"
    equalize_warmup isiro
    python3 "${BENCH_ROOT}/scripts/run_bench.py" \
      --variant isiro \
      --profile generation-32-256 \
      --run-id "${RUN_ID}" \
      --model "${MODEL_ID}" \
      "${BENCH_TOKENIZER_ARGS[@]}" \
      --api-url "${API_URL}" \
      --input-tokens 32 \
      --output-tokens 256 \
      --num-warmups "${WARMUPS}" \
      --num-prompts "$(timed_num_prompts)" \
      --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
      --seed "${BENCH_SEED}" \
      --replicates "${REPLICATES}" \
      --serve-config "${RUN_DIR}/equal_batch/isiro_serve_config.json" \
      --raw-dir "${RAW_DIR}/equal_batch/isiro" \
      --out "${RUN_DIR}/equal_batch/isiro/generation-32-256.json" \
      --vllm-image "${VLLM_IMAGE}"
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    TIC_PID=""
  fi
}

if [[ "${EXPERIMENT_KIND}" == "capacity" && "${CAPACITY_EQUAL_BATCH}" -eq 1 ]]; then
  MATCHED_EQ_SEQS="$(
    python3 -c "print(min(int('${BASELINE_CAPACITY_SEQS}'), int('${TIC_CAPACITY_SEQS}')))"
  )"
  MATCHED_EQ_CONC="$(
    python3 "${BENCH_ROOT}/scripts/capacity_plan.py" concurrency \
      --max-num-seqs "${MATCHED_EQ_SEQS}" \
      --bench-cap "${BENCH_MAX_CONCURRENCY_CAP}"
  )"
  echo "equal-batch transparency: max_num_seqs=${MATCHED_EQ_SEQS} concurrency=${MATCHED_EQ_CONC}" >&2
  restore_timed_graph_mode
  run_equal_batch_transparency "${MATCHED_EQ_SEQS}" "${MATCHED_EQ_CONC}"
elif [[ "${EXPERIMENT_KIND}" == "capacity" ]]; then
  echo "equal-batch transparency: skipped (set ISIRO_BENCH_EQUAL_BATCH=1 to enable)" >&2
fi

python3 "${BENCH_ROOT}/scripts/compare_ab.py" \
  --run-dir "${RUN_DIR}" \
  --baseline-model "${BASELINE_MODEL_DIR}" \
  --tic-artifact "${TIC_MODEL_DIR}/model.tic" \
  --model "${MODEL_SLUG}" \
  --precision "${PRECISION}" \
  --system-id "${SYSTEM_ID}" \
  --isiro-format "${ISIRO_FORMAT}" \
  --isiro-runtime "${ISIRO_RUNTIME:-${ISIRO_FORMAT}}" \
  --experiment-kind "${EXPERIMENT_KIND}"

# Model report is written by launch_ab after the full launch finishes.
if [[ "${BASELINE_SKIPPED}" -eq 1 ]]; then
  echo "baseline_reused_from=${BASELINE_REUSED_FROM}"
fi
echo "ISIRO_BENCH_RUN_DIR=${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" >"${SCRATCH_ROOT}/.last_run_${GRAPH_MODE}"
echo "${RUN_DIR}"
