#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
  source "venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# Load .env nếu tồn tại
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

DISPLAY_HOST=$HOST
if [ "$DISPLAY_HOST" = "0.0.0.0" ]; then
    DISPLAY_HOST="localhost"
fi

echo "Khởi động Trung Tâm Tri Thức tại http://${DISPLAY_HOST}:${PORT}/"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
