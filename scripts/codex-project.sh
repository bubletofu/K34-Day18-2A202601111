#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_CODEX_DIR="$PROJECT_ROOT/.codex-local"
PROJECT_HOME_DIR="$PROJECT_ROOT/.codex-home"
PROJECT_CONFIG_DIR="$PROJECT_ROOT/.config-local"
PROJECT_CACHE_DIR="$PROJECT_ROOT/.cache-local"
PROJECT_TMP_DIR="$PROJECT_ROOT/.tmp-local"

mkdir -p "$PROJECT_CODEX_DIR" "$PROJECT_HOME_DIR" "$PROJECT_CONFIG_DIR" "$PROJECT_CACHE_DIR" "$PROJECT_TMP_DIR"

# Scope every mutable Codex/config path to this repository.
export CODEX_HOME="$PROJECT_CODEX_DIR"
export HOME="$PROJECT_HOME_DIR"
export XDG_CONFIG_HOME="$PROJECT_CONFIG_DIR"
export XDG_CACHE_HOME="$PROJECT_CACHE_DIR"
export TMPDIR="$PROJECT_TMP_DIR"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

if [[ "${1:-}" == "--show-isolation" ]]; then
  printf "PROJECT_ROOT=%s\n" "$PROJECT_ROOT"
  printf "HOME=%s\n" "$HOME"
  printf "CODEX_HOME=%s\n" "$CODEX_HOME"
  printf "XDG_CONFIG_HOME=%s\n" "$XDG_CONFIG_HOME"
  printf "XDG_CACHE_HOME=%s\n" "$XDG_CACHE_HOME"
  printf "TMPDIR=%s\n" "$TMPDIR"
  exit 0
fi

exec codex "$@"
