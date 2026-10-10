#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DocPilot] backend -> http://127.0.0.1:8008"
echo "[DocPilot] frontend -> http://127.0.0.1:5173"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

cd "${ROOT_DIR}/backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8008 &
BACKEND_PID=$!

cd "${ROOT_DIR}/frontend"
npm run dev -- --host 0.0.0.0 --port 5173
