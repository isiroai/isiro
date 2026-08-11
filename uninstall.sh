#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=dash
# ISIRO uninstall; works even when ~/.isiro/bin/isiro is missing.
#
#   curl -fsSL https://isiro.ai/uninstall.sh | sh
#   curl -fsSL https://isiro.ai/uninstall.sh | sh -s -- --purge -y
#
# Partial-execution safe: logic in functions; main invoked only on final line.

set -eu

ISIRO_HOME="${ISIRO_HOME:-$HOME/.isiro}"
UNINSTALL_SCRIPT_URL="${ISIRO_UNINSTALL_SCRIPT_URL:-https://isiro.ai/uninstall.sh}"

status() { echo ">>> $*" >&2; }

resolve_rc_files() {
  local receipt="${ISIRO_HOME}/install-receipt.json"
  if [ -f "$receipt" ] && command -v python3 >/dev/null 2>&1; then
    python3 - "$receipt" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(0)
d = json.loads(p.read_text(encoding="utf-8"))
rc = (d.get("rc_file") or "").strip()
if rc:
    print(rc)
PY
    return 0
  fi
  local rc
  for rc in "${HOME}/.bashrc" "${HOME}/.bash_profile" "${HOME}/.zshrc" "${HOME}/.profile"; do
    if [ -f "$rc" ] && { grep -Fq "ISIRO" "$rc" 2>/dev/null || grep -Fq '.isiro/bin' "$rc" 2>/dev/null; }; then
      echo "$rc"
    fi
  done
}

strip_rc_file() {
  local rc="$1"
  [ -f "$rc" ] || return 0
  if ! grep -Fq "ISIRO" "$rc" 2>/dev/null && ! grep -Fq '.isiro/bin' "$rc" 2>/dev/null; then
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 required to strip shell rc blocks" >&2
    exit 1
  fi
  python3 - "$rc" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
out = []
skip = False
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.rstrip("\n")
    if stripped.startswith("# BEGIN ISIRO bash completion") or stripped.startswith(
        "# BEGIN ISIRO runtime launcher"
    ):
        skip = True
        i += 1
        continue
    if stripped.startswith("# END ISIRO bash completion") or stripped.startswith(
        "# END ISIRO runtime launcher"
    ):
        skip = False
        i += 1
        continue
    if stripped == "# ISIRO runtime launcher (added by install.sh)":
        i += 1
        if i < len(lines) and ".isiro/bin" in lines[i]:
            i += 1
        continue
    if not skip:
        out.append(line)
    i += 1
text = "".join(out).rstrip()
if text:
    text += "\n"
path.write_text(text, encoding="utf-8")
PY
}

collect_image_refs() {
  local manifest="${ISIRO_HOME}/release-manifest.json"
  if [ ! -f "$manifest" ] || ! command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  python3 - "$manifest" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("runtime", "compiler"):
    item = data.get(key) or {}
    reg = (item.get("registry") or "").strip()
    if not reg:
        continue
    digest = (item.get("digest") or "").strip()
    tag = (item.get("tag") or "").strip()
    if digest and "REPLACE" not in digest:
        print(f"{reg}@{digest}")
    elif tag:
        print(f"{reg}:{tag}")
PY
}

# Remove a path that may contain root-owned files from prior root-in-container serves.
# Prefer host rm; on EACCES use Docker (no host sudo).
rm_rf_maybe_docker() {
  local target="$1"
  local parent base err img cand
  [ -e "$target" ] || return 0
  if rm -rf "$target" 2>/dev/null; then
    [ ! -e "$target" ] && return 0
  fi
  if [ ! -e "$target" ]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: cannot remove ${target} (permission denied) and docker is unavailable." >&2
    echo "  Stop any isiro serve containers, then re-run uninstall." >&2
    exit 1
  fi
  status "Clearing root-owned files under ${target} via Docker..."
  parent="$(dirname "$target")"
  base="$(basename "$target")"
  img=""
  for cand in alpine:3.20 python:3.12-slim-bookworm busybox:1.36; do
    if docker image inspect "$cand" >/dev/null 2>&1; then
      img="$cand"
      break
    fi
  done
  if [ -z "$img" ]; then
    img="alpine:3.20"
  fi
  err="$(mktemp)"
  if ! docker run --rm \
      -v "${parent}:/parent" \
      --entrypoint rm \
      "$img" \
      -rf "/parent/${base}" 2>"$err"; then
    echo "ERROR: failed to remove ${target} via Docker." >&2
    cat "$err" >&2 || true
    rm -f "$err"
    exit 1
  fi
  rm -f "$err"
  if [ -e "$target" ]; then
    echo "ERROR: ${target} still present after Docker remove." >&2
    exit 1
  fi
}

remove_launcher_tree() {
  local path
  for path in bin lib wheels cli-slim serve-site cache; do
    if [ -e "${ISIRO_HOME}/${path}" ]; then
      rm_rf_maybe_docker "${ISIRO_HOME}/${path}"
    fi
  done
  rm -f "${ISIRO_HOME}/release-manifest.json" "${ISIRO_HOME}/install-receipt.json"
}

isiro_data_present() {
  [ -d "$ISIRO_HOME" ] || return 1
  local path
  for path in bin lib wheels cli-slim serve-site cache release-manifest.json install-receipt.json \
    compiler.env credential.json eula_acceptance.json compiler_eula_acceptance.json; do
    if [ -e "${ISIRO_HOME}/${path}" ]; then
      return 0
    fi
  done
  return 1
}

shell_hooks_present() {
  local rc
  for rc in $(resolve_rc_files); do
    [ -n "$rc" ] && return 0
  done
  return 1
}

completion_file_path() {
  local completion_file="${HOME}/.bash_completion.d/isiro"
  if [ -f "${ISIRO_HOME}/install-receipt.json" ] && command -v python3 >/dev/null 2>&1; then
    completion_file="$(python3 - "${ISIRO_HOME}/install-receipt.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print((d.get("completion_file") or "").strip())
PY
)"
    [ -n "$completion_file" ] || completion_file="${HOME}/.bash_completion.d/isiro"
  fi
  echo "$completion_file"
}

usage() {
  cat <<EOF
Usage: uninstall.sh [-y|--yes] [--images] [--purge] [--dry-run]

Remove ISIRO launcher, runtime cache, shell rc hooks, and tab completion.

If the isiro command is missing (stale shell hash), use:
  curl -fsSL ${UNINSTALL_SCRIPT_URL} | sh -s -- [options]

Options:
  -y, --yes      Skip confirmation prompt
  --images       Remove runtime/compiler Docker images (also implied by --purge)
  --purge        Remove all install data including entitlements, and Docker images
  --dry-run      Show what would be removed, without removing anything
  -h, --help     Show this help
EOF
}

main() {
  local yes=0 images=0 dry_run=0 purge=0
  while [ $# -gt 0 ]; do
    case "$1" in
      -y|--yes) yes=1; shift ;;
      --images) images=1; shift ;;
      --dry-run) dry_run=1; shift ;;
      --purge) purge=1; images=1; shift ;;
      -h|--help) usage; return 0 ;;
      *)
        echo "ERROR: unknown option '$1' (try uninstall.sh --help)" >&2
        exit 1
        ;;
    esac
  done

  if ! isiro_data_present && ! shell_hooks_present; then
    echo ""
    echo "ISIRO is already uninstalled."
    echo "Restart your shell or run: hash -r"
    return 0
  fi

  local completion_file rc seen="" unique_rcs=""
  completion_file="$(completion_file_path)"

  for rc in $(resolve_rc_files); do
    [ -n "$rc" ] || continue
    case "$seen" in *"|${rc}|"*) continue ;; esac
    seen="${seen}|${rc}|"
    unique_rcs="${unique_rcs}${unique_rcs:+ }${rc}"
  done

  echo "ISIRO uninstall will remove:"
  if [ "$purge" -eq 1 ]; then
    echo "  ${ISIRO_HOME}/ (all data, including entitlements)"
  else
    echo "  ${ISIRO_HOME}/bin, lib, wheels, cli-slim, serve-site, cache, release-manifest.json"
    echo "  (entitlements kept: compiler.env, credential.json, EULA markers)"
  fi
  if [ -n "$unique_rcs" ]; then
    for rc in $unique_rcs; do
      echo "  ISIRO blocks in ${rc}"
    done
  else
    echo "  (no shell rc files with ISIRO hooks found)"
  fi
  if [ -f "$completion_file" ]; then
    echo "  ${completion_file}"
  fi
  if [ "$images" -eq 1 ]; then
    collect_image_refs | while IFS= read -r ref; do
      [ -n "$ref" ] && echo "  docker image ${ref}"
    done
  else
    echo "  (Docker images kept; pass --images to remove)"
  fi

  if [ "$dry_run" -eq 1 ]; then
    echo ""
    echo "Dry run; no changes made."
    return 0
  fi

  if [ "$yes" -eq 0 ]; then
    printf '%s' "Continue? [y/N] "
    read -r reply
    case "${reply:-}" in
      y|Y|yes|YES) ;;
      *) echo "Uninstall cancelled."; return 1 ;;
    esac
  fi

  for rc in $unique_rcs; do
    strip_rc_file "$rc"
  done

  if [ -f "$completion_file" ]; then
    rm -f "$completion_file"
  fi

  if [ "$images" -eq 1 ] && command -v docker >/dev/null 2>&1; then
    collect_image_refs | while IFS= read -r ref; do
      [ -z "$ref" ] && continue
      docker rmi "$ref" >/dev/null 2>&1 || true
    done
  fi

  if [ "$purge" -eq 1 ]; then
    rm_rf_maybe_docker "$ISIRO_HOME"
  else
    remove_launcher_tree
  fi

  echo ""
  echo "ISIRO uninstalled."
  echo "Restart your shell or run: hash -r"
}

{ main "$@"; }
