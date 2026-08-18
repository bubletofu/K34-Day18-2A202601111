#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_CODEX_DIR="$PROJECT_ROOT/.codex-local"
PROJECT_HOME_DIR="$PROJECT_ROOT/.codex-home"
PROJECT_CONFIG_DIR="$PROJECT_ROOT/.config-local"
PROJECT_CACHE_DIR="$PROJECT_ROOT/.cache-local"
PROJECT_TMP_DIR="$PROJECT_ROOT/.tmp-local"
INSTALLER_PATH="$PROJECT_CODEX_DIR/ckey-installer.sh"
INSTALLER_URL='https://ckey.vn/install/apiai-install?tool=codex&model=gpt-oss-120b'

mkdir -p "$PROJECT_CODEX_DIR" "$PROJECT_HOME_DIR" "$PROJECT_CONFIG_DIR" "$PROJECT_CACHE_DIR" "$PROJECT_TMP_DIR"

curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 "$INSTALLER_URL" --output "$INSTALLER_PATH"
chmod 700 "$INSTALLER_PATH"

printf 'Installer: %s\nSHA256: ' "$INSTALLER_PATH"
shasum -a 256 "$INSTALLER_PATH"
printf '\nFirst 120 lines (not executed):\n'
sed -n '1,120p' "$INSTALLER_PATH"

if [[ "${1:-}" != "--run" ]]; then
  printf '\nNot executed. Review it, then rerun this command with --run.\n'
  exit 0
fi

CODEX_BIN="$(command -v codex)"
CODEX_BIN_DIR="$(dirname "$CODEX_BIN")"
PROJECT_PATH="$PROJECT_ROOT/.venv/bin:$CODEX_BIN_DIR:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

for isolated_dir in "$PROJECT_CODEX_DIR" "$PROJECT_HOME_DIR" "$PROJECT_CONFIG_DIR" "$PROJECT_CACHE_DIR" "$PROJECT_TMP_DIR"; do
  case "$isolated_dir" in
    "$PROJECT_ROOT"/*) ;;
    *)
      printf 'Refusing to run: %s escaped the project root.\n' "$isolated_dir" >&2
      exit 1
      ;;
  esac
done

printf '\nRunning with project-local HOME, CODEX_HOME, and XDG paths.\n'
env -i HOME="$PROJECT_HOME_DIR" CODEX_HOME="$PROJECT_CODEX_DIR" XDG_CONFIG_HOME="$PROJECT_CONFIG_DIR" XDG_CACHE_HOME="$PROJECT_CACHE_DIR" TMPDIR="$PROJECT_TMP_DIR" PATH="$PROJECT_PATH" CODEX_BIN="$CODEX_BIN" /bin/bash "$INSTALLER_PATH"
