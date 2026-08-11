#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Expect MODEL_DIR and MODEL_SLUG from launch_ab.py, then run the shared harness.
set -euo pipefail
if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "MODEL_DIR must be set (launch_ab.py sets it to benchmarks/{model})" >&2
  exit 2
fi
if [[ -z "${MODEL_SLUG:-}" ]]; then
  echo "MODEL_SLUG must be set (launch_ab.py sets it to the model directory name)" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/harness_lib.sh" "$@"
