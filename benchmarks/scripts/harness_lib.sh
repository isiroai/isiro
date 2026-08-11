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
    --eager|--graph-off)
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
      echo "usage: benchmarks/run_ab.sh {model} [--dry-run] [--graph-on|--graph-off] [--no-reuse-baseline] [--profiles=...]" >&2
      echo "  ISIRO_BENCH_CONFIG=path/to/common.env  (default: ${HERE}/common.env)" >&2
      echo "  Public path: capacity-primary A/B; Graph ON by default." >&2
      echo "  --graph-on: product default (matched outer CUDA graphs A/B)." >&2
      echo "  --graph-off: matched graphs-OFF A/B (--eager is an alias)." >&2
      echo "  Prefer: benchmarks/run_ab.sh <model> --both-graph-modes to record both." >&2
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

: "${MODEL_SLUG:?set MODEL_SLUG (launch_ab.py sets it to the model directory name)}"
: "${BASELINE_MODEL_DIR:?set BASELINE_MODEL_DIR in common.env}"
: "${TIC_MODEL_DIR:?set TIC_MODEL_DIR in common.env}"
: "${MODEL_ID:?set MODEL_ID in common.env}"
: "${VLLM_IMAGE:?set VLLM_IMAGE in common.env}"
# Serve knobs (defaults match benchmarks/common.env.example).
: "${SERVE_MAX_MODEL_LEN:=4096}"
: "${SERVE_GPU_MEMORY_UTILIZATION:=0.90}"
: "${SERVE_TENSOR_PARALLEL_SIZE:=1}"
SERVE_MAX_NUM_SEQS="${SERVE_MAX_NUM_SEQS:-32}"
: "${SERVE_PREFIX_CACHING:=false}"
: "${SERVE_ENFORCE_EAGER:=false}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
PRECISION="${PRECISION:-bf16}"
case "${PRECISION}" in
  bf16|fp8) ;;
  *)
    echo "PRECISION must be bf16 or fp8 (got ${PRECISION})" >&2
    exit 2
    ;;
esac

# Precision confusion guard: the config file basename encodes the intended
# precision (common.env => bf16, common.env.fp8 => fp8). A stale
# ISIRO_BENCH_CONFIG in the shell has silently routed "bf16" runs to the FP8
# config before; refuse to proceed when the resolved config and PRECISION
# disagree so reports can never be mislabeled.
CONFIG_BASENAME="$(basename "${CONFIG}")"
case "${CONFIG_BASENAME}" in
  *.fp8) CONFIG_PRECISION="fp8" ;;
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
# PROFILES / BASELINE_REUSE are printed after counts resolve.

# Matched A/B requires these knobs; refuse silent drift.
if [[ "${SERVE_PREFIX_CACHING}" != "false" ]]; then
  echo "SERVE_PREFIX_CACHING must be false for matched TTFT sweeps" >&2
  exit 2
fi

export VLLM_USE_FLASHINFER_SAMPLER=0
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SYSTEM_ID}"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
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
EQUALIZE_PROMPTS="${EQUALIZE_PROMPTS:-16}"
EQUALIZE_INPUT_TOKENS="${EQUALIZE_INPUT_TOKENS:-128}"
EQUALIZE_OUTPUT_TOKENS="${EQUALIZE_OUTPUT_TOKENS:-64}"

# Profile selection. Public default is decode-only. TTFT names remain allowed
# for eng iteration via --profiles=... or ISIRO_BENCH_FULL_PROFILES=1.
ALL_PROFILES=(ttft-128 ttft-512 ttft-2048 generation-32-256)
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
if [[ "${PRECISION}" == "fp8" ]]; then
  if [[ ! -f "${BASELINE_MODEL_DIR}/hf_quant_config.json" ]]; then
    echo "PRECISION=fp8 requires Hub FP8 baseline with hf_quant_config.json" >&2
    exit 2
  fi
  if ! python3 - "${BASELINE_MODEL_DIR}/hf_quant_config.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
algo = str((doc.get("quantization") or {}).get("quant_algo") or "").upper()
raise SystemExit(0 if algo == "FP8" else 1)
PY
  then
    echo "PRECISION=fp8 baseline hf_quant_config.json must set quant_algo FP8" >&2
    exit 2
  fi
elif [[ -f "${BASELINE_MODEL_DIR}/hf_quant_config.json" ]]; then
  echo "PRECISION=bf16 but baseline has hf_quant_config.json; use PRECISION=fp8 / common.env.fp8" >&2
  exit 2
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
    # Hub modelopt FP8: do not force --dtype bfloat16.
    BASELINE_SERVE+=(--quantization modelopt)
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
  if [[ "${SERVE_ENFORCE_EAGER}" == "true" ]]; then
    BASELINE_SERVE+=(--enforce-eager)
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
TIC_SERVE=(
  # Bind 0.0.0.0 inside the serve container so Docker -p ${API_PORT}:${API_PORT}
  # is reachable from the host. The bench client still uses 127.0.0.1:${API_PORT}.
  # Binding only 127.0.0.1 in-container makes host curl get connection reset.
  "${ISIRO_SERVE_BIN}" serve "${TIC_BENCH_BUNDLE}" --target vllm
  --host 0.0.0.0 --port "${API_PORT}"
  --max-model-len "${SERVE_MAX_MODEL_LEN}"
)
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
  python3 - "${RUN_DIR}/commands.json" \
    "${BASELINE_MODEL_DIR}" "${TIC_MODEL_DIR}" "${TIC_BENCH_BUNDLE}" \
    "${VLLM_IMAGE}" "${MODEL_ID}" "${API_PORT}" \
    "${SERVE_MAX_MODEL_LEN}" "${SERVE_GPU_MEMORY_UTILIZATION}" \
    "${SERVE_TENSOR_PARALLEL_SIZE}" "${SERVE_MAX_NUM_SEQS}" \
    "${BASELINE_REUSED_FROM}" "${PRECISION}" "${SERVE_ENFORCE_EAGER}" \
    "${VERIFY_REFERENCE}" <<'PY'
import json, sys
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
    baseline.extend(["--quantization", "modelopt"])
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
if enforce_eager == "true":
    baseline.insert(baseline.index("--no-enable-prefix-caching") + 1, "--enforce-eager")
tic_verify = ["isiro", "verify", f"{tic_model}/model.tic"]
if verify_reference == "1":
    tic_verify.extend(["--reference", baseline_model])
doc = {
    "baseline_serve": baseline,
    "precision": precision,
    "enforce_eager": enforce_eager == "true",
    "tic_verify": tic_verify,
    "verify_mode": "reference" if verify_reference == "1" else "integrity",
    "tic_serve": [
        "isiro", "serve", tic_bundle, "--target", "vllm",
        "--host", "127.0.0.1", "--port", port,
        "--max-model-len", max_model_len,
    ],
}
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
if [[ "${VERIFY_REFERENCE}" -eq 1 ]]; then
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
set +e
"${VERIFY[@]}" >"${LOG_DIR}/verify.log" 2>&1
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

cleanup() {
  if [[ -n "${TIC_PID}" ]] && kill -0 "${TIC_PID}" 2>/dev/null; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
  fi
  docker rm -f "${BASELINE_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

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
  python3 - \
    "${TIC_MODEL_DIR}/serve.yaml" \
    "${TIC_BENCH_BUNDLE}/serve.yaml" \
    "${SERVE_MAX_MODEL_LEN}" \
    "${SERVE_GPU_MEMORY_UTILIZATION}" \
    "${SERVE_TENSOR_PARALLEL_SIZE}" \
    "${SERVE_MAX_NUM_SEQS}" \
    "${SERVE_ENFORCE_EAGER}" <<'PY'
import re, sys

source, target, max_model_len, gpu_util, tp, max_seqs, enforce_eager = sys.argv[1:8]
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

text = upsert(text, "max_model_len", max_model_len)
text = upsert(text, "gpu_memory_utilization", gpu_util)
text = upsert(text, "tensor_parallel_size", tp)
text = upsert(text, "max_num_seqs", max_seqs)
text = upsert(text, "no_enable_prefix_caching", "true")
# enforce_eager is a forbidden serve.yaml key (isiro serve controls it).
# Product default is outer graphs ON; eager A/B sets ISIRO_SERVE_CUDA_GRAPHS=0
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

# Drop leftover listeners before baseline Docker publish or TIC host serve.
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

equalize_warmup() {
  variant="$1"
  curl -fsS "${API_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"temperature\":0}" \
    >/dev/null
  python3 "${BENCH_ROOT}/scripts/run_bench.py" \
    --variant "${variant}" \
    --profile "warmup-equalize" \
    --run-id "${RUN_ID}" \
    --model "${MODEL_ID}" \
    --api-url "${API_URL}" \
    --input-tokens "${EQUALIZE_INPUT_TOKENS}" \
    --output-tokens "${EQUALIZE_OUTPUT_TOKENS}" \
    --num-warmups 0 \
    --num-prompts "${EQUALIZE_PROMPTS}" \
    --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --seed "${BENCH_SEED}" \
    --replicates 1 \
    --serve-config "${RUN_DIR}/${variant}_serve_config.json" \
    --raw-dir "${LOG_DIR}/warmup-equalize/${variant}" \
    --out "${LOG_DIR}/warmup-equalize-${variant}.json" \
    --vllm-image "${VLLM_IMAGE}"
}

# Greedy serve output capture (before timed vllm bench). compare_ab folds A/B.
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
    "${SERVE_ENFORCE_EAGER}" <<'PY'
import json, re, sys
from pathlib import Path

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
    --isiro-version "${ISIRO_FORMAT}" \
    --quiet-host \
    "${image_arg[@]}" \
    "${identity_arg[@]}"
  python3 - "${RUN_DIR}/${variant}_environment.json" "$(gpu_process_bytes)" "${LOG_DIR}/${variant}-serve.log" "${variant}" "${GPU_PROCESS_BYTES_AFTER_READY:-0}" "${SERVE_GPU_MEMORY_UTILIZATION}" <<'PY'
import json, math, re, sys
path, value, log_path, variant, ready_value, util_s = (
    sys.argv[1],
    int(sys.argv[2] or 0),
    sys.argv[3],
    sys.argv[4],
    int(sys.argv[5] or 0),
    sys.argv[6],
)
doc = json.load(open(path, encoding="utf-8"))
doc["gpu_process_memory_after_load_bytes_measured"] = value or None
if variant == "isiro" and ready_value:
    doc["gpu_process_memory_after_ready_bytes_measured"] = ready_value
text = open(log_path, encoding="utf-8", errors="replace").read()
gib = 1024 ** 3
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
doc["kv_cache_memory_bytes_measured"] = (
    round(float(kv_gib.group(1)) * gib) if kv_gib else None
)
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
      --api-url "${API_URL}" \
      --input-tokens "${input_tokens}" \
      --output-tokens "${output_tokens}" \
      --num-warmups "${WARMUPS}" \
      --num-prompts "${PROMPTS}" \
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
  python3 - "${serve_log}" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
if not m:
    raise SystemExit(f"{sys.argv[1]}: GPU KV cache size not found")
print(int(m.group(1).replace(",", "")))
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
  export ISIRO_SERVE_CUDA_GRAPHS=0
  SERVE_ENFORCE_EAGER="true"
  if [[ -n "${TIC_PID}" ]] && kill -0 "${TIC_PID}" 2>/dev/null; then
    kill "${TIC_PID}" 2>/dev/null || true
    wait "${TIC_PID}" 2>/dev/null || true
    TIC_PID=""
  fi
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  ensure_api_port_free
  : >"${log}"
  "${TIC_SERVE[@]}" >"${log}" 2>&1 &
  TIC_PID=$!
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
  prepare_tic_bundle
  # Matched probe uses the timed graph mode (not eager capacity-search probes).
  if [[ "${SERVE_ENFORCE_EAGER}" == "false" ]]; then
    export ISIRO_SERVE_CUDA_GRAPHS=1
  else
    export ISIRO_SERVE_CUDA_GRAPHS=0
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
  "${TIC_SERVE[@]}" >"${probe_log}" 2>&1 &
  probe_pid=$!
  TIC_PID="${probe_pid}"
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

# Timed baseline at resolved max_num_seqs (capacity) or env value (equal-batch).
SERVE_MAX_NUM_SEQS="${BASELINE_CAPACITY_SEQS}"
BENCH_MAX_CONCURRENCY="${BASELINE_CONC:-${BENCH_MAX_CONCURRENCY_CAP}}"
if [[ "${EXPERIMENT_KIND}" == "equal-batch" ]]; then
  BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY_CAP}"
fi
build_baseline_serve
prepare_tic_bundle

BASELINE_SKIPPED=0
# Reuse when requested for equal-batch or capacity (explicit).
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

if [[ "${EXPERIMENT_KIND}" == "capacity" && "${CAPACITY_KV_SCALE}" -eq 1 ]]; then
  if [[ "${CAPACITY_KV_MEASURED}" -eq 1 ]]; then
    scale_tic_seqs_from_measured_kv
  else
    scale_tic_seqs_from_baseline_kv
  fi
fi

write_commands_json

# Timed TIC at its capacity max (or matched equal-batch seqs).
SERVE_MAX_NUM_SEQS="${TIC_CAPACITY_SEQS}"
if [[ "${EXPERIMENT_KIND}" == "capacity" ]]; then
  BENCH_MAX_CONCURRENCY="${TIC_CONC}"
else
  BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY_CAP}"
fi
prepare_tic_bundle

# Select the TIC CUDA-graph mode to match the baseline A/B.
# Product serve defaults to outer graphs ON; --eager sets ISIRO_SERVE_CUDA_GRAPHS=0.
if [[ "${SERVE_ENFORCE_EAGER}" == "false" ]]; then
  export ISIRO_SERVE_CUDA_GRAPHS=1
else
  export ISIRO_SERVE_CUDA_GRAPHS=0
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
"${TIC_SERVE[@]}" >"${LOG_DIR}/isiro-serve.log" 2>&1 &
TIC_PID=$!
# SERV-0101 often means a still-dying prior holder; one delayed retry only.
if ! wait_ready "${LOG_DIR}/isiro-serve.log" "${TIC_PID}"; then
  echo "TIC serve not ready; delayed retry after port cleanup" >&2
  kill "${TIC_PID}" 2>/dev/null || true
  wait "${TIC_PID}" 2>/dev/null || true
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
  sleep 8
  ensure_api_port_free
  : >"${LOG_DIR}/isiro-serve.log"
  "${TIC_SERVE[@]}" >"${LOG_DIR}/isiro-serve.log" 2>&1 &
  TIC_PID=$!
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
kill "${TIC_PID}" 2>/dev/null || true
wait "${TIC_PID}" 2>/dev/null || true
TIC_PID=""

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
    if copy_equal_batch_profile_from_capacity baseline; then
      reuse_baseline=1
      echo "equal-batch baseline: reused capacity timed profiles" >&2
    fi
  fi
  if [[ "${reuse_baseline}" -eq 0 ]]; then
    echo "equal-batch baseline: cold start" >&2
    SERVE_MAX_NUM_SEQS="${matched_seqs}"
    BENCH_MAX_CONCURRENCY="${matched_conc}"
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
      --api-url "${API_URL}" \
      --input-tokens 32 \
      --output-tokens 256 \
      --num-warmups "${WARMUPS}" \
      --num-prompts "${PROMPTS}" \
      --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
      --seed "${BENCH_SEED}" \
      --replicates "${REPLICATES}" \
      --serve-config "${RUN_DIR}/equal_batch/baseline_serve_config.json" \
      --raw-dir "${RAW_DIR}/equal_batch/baseline" \
      --out "${RUN_DIR}/equal_batch/baseline/generation-32-256.json" \
      --vllm-image "${VLLM_IMAGE}"
    stop_baseline_container
  fi

  if [[ "${matched_seqs}" == "${capacity_tic_seqs}" && \
        -n "${capacity_tic_conc}" && \
        "${matched_conc}" == "${capacity_tic_conc}" ]]; then
    if copy_equal_batch_profile_from_capacity isiro; then
      reuse_tic=1
      echo "equal-batch TIC: reused capacity timed profiles" >&2
    fi
  fi
  if [[ "${reuse_tic}" -eq 0 ]]; then
    echo "equal-batch TIC: cold start" >&2
    SERVE_MAX_NUM_SEQS="${matched_seqs}"
    BENCH_MAX_CONCURRENCY="${matched_conc}"
    prepare_tic_bundle
    if [[ "${SERVE_ENFORCE_EAGER}" == "false" ]]; then
      export ISIRO_SERVE_CUDA_GRAPHS=1
    else
      export ISIRO_SERVE_CUDA_GRAPHS=0
    fi
    fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
    sleep 3
    ensure_api_port_free
    : >"${LOG_DIR}/equal-batch-isiro-serve.log"
    "${TIC_SERVE[@]}" >"${LOG_DIR}/equal-batch-isiro-serve.log" 2>&1 &
    TIC_PID=$!
    if ! wait_ready "${LOG_DIR}/equal-batch-isiro-serve.log" "${TIC_PID}"; then
      echo "equal-batch TIC serve not ready; delayed retry" >&2
      kill "${TIC_PID}" 2>/dev/null || true
      wait "${TIC_PID}" 2>/dev/null || true
      fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
      sleep 5
      ensure_api_port_free
      : >"${LOG_DIR}/equal-batch-isiro-serve.log"
      "${TIC_SERVE[@]}" >"${LOG_DIR}/equal-batch-isiro-serve.log" 2>&1 &
      TIC_PID=$!
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
    write_effective_config isiro "${TIC_BENCH_BUNDLE}/serve.yaml"
    cp -f "${RUN_DIR}/isiro_serve_config.json" \
      "${RUN_DIR}/equal_batch/isiro_serve_config.json"
    equalize_warmup isiro
    python3 "${BENCH_ROOT}/scripts/run_bench.py" \
      --variant isiro \
      --profile generation-32-256 \
      --run-id "${RUN_ID}" \
      --model "${MODEL_ID}" \
      --api-url "${API_URL}" \
      --input-tokens 32 \
      --output-tokens 256 \
      --num-warmups "${WARMUPS}" \
      --num-prompts "${PROMPTS}" \
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
  --experiment-kind "${EXPERIMENT_KIND}"

# Model report is written by launch_ab after the full launch finishes.
if [[ "${BASELINE_SKIPPED}" -eq 1 ]]; then
  echo "baseline_reused_from=${BASELINE_REUSED_FROM}"
fi
echo "ISIRO_BENCH_RUN_DIR=${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" >"${SCRATCH_ROOT}/.last_run_${GRAPH_MODE}"
echo "${RUN_DIR}"
