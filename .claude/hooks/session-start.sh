#!/bin/bash
# SessionStart hook for Claude Code cloud sessions.
# Installs the cellauto package with dev extras once pyproject.toml exists (created in plan phase P01).
# Before P01 it is a no-op. It never installs torch: download.pytorch.org is blocked by the sandbox
# proxy, and torch tests skip without it (CI's `torch` job covers them).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

if [ ! -f pyproject.toml ]; then
  echo "session-start: no pyproject.toml yet (plan phase P01 not done); skipping install." >&2
  exit 0
fi

python3 -m pip install --quiet --disable-pip-version-check -e ".[dev]"

# The container ships uv-tool shims (pytest, mypy, ruff) in /root/.local/bin with their own
# interpreters; they cannot import cellauto. Put this interpreter's scripts dir first on PATH.
SCRIPTS_DIR="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$SCRIPTS_DIR:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
echo "session-start: installed cellauto[dev] in editable mode." >&2
