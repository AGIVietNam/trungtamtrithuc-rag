#!/usr/bin/env bash
set -euo pipefail

APP_NAME="trungtamtrithuc-rag"

main() {
  DEPLOY_BRANCH="${DEPLOY_BRANCH:-main}"

  echo "[INFO] Deploy branch: ${DEPLOY_BRANCH}"
  git fetch origin "${DEPLOY_BRANCH}"
  git checkout "${DEPLOY_BRANCH}"
  git reset --hard "origin/${DEPLOY_BRANCH}"

  # Source runtime env before install/start commands.
  # shellcheck disable=SC1091
  source bin/local-env.sh
  if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
  elif [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi

  if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt || true
  fi

  export NVM_DIR="$HOME/.nvm"
  if [ -s "$NVM_DIR/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"
  fi
  if command -v npm >/dev/null 2>&1; then
    NPM_GLOBAL_BIN="$(npm bin -g 2>/dev/null || true)"
    if [ -n "${NPM_GLOBAL_BIN:-}" ]; then
      export PATH="$NPM_GLOBAL_BIN:$PATH"
    fi
  fi

  if ! command -v pm2 >/dev/null 2>&1; then
    echo "[INFO] PM2 not found. Installing..."

    if ! command -v npm >/dev/null 2>&1; then
      echo "[INFO] npm not found. Installing Node.js..."
      curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
      sudo apt install -y nodejs
    fi

    sudo npm install -g pm2
  fi

  echo "[INFO] Restarting service with PM2..."
  START_CMD="uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"
  if pm2 describe "$APP_NAME" >/dev/null 2>&1; then
    pm2 restart "$APP_NAME" --update-env
  else
    pm2 start "$START_CMD" --name "$APP_NAME"
  fi

  pm2 save
  echo "[INFO] Deploy finished."
}

main
