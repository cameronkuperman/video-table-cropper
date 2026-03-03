#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKGROUND=0
STOP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --background)
      BACKGROUND=1
      ;;
    --stop)
      STOP_ONLY=1
      ;;
  esac
done

if [[ -d "venv" ]]; then
  VENV_DIR="venv"
elif [[ -d ".venv" ]]; then
  VENV_DIR=".venv"
else
  VENV_DIR=".venv"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$PYTHON_VERSION" == "3.14" || "$PYTHON_VERSION" == "3.15" || "$PYTHON_VERSION" == "4.0" ]]; then
  echo "Local web app preflight failed: Python $PYTHON_VERSION is not a supported local runtime for this project."
  echo "Use Python 3.12 for local app testing. The production web image also uses Python 3.12."
  echo "Set ALLOW_UNSUPPORTED_PYTHON=1 if you want to bypass this guard temporarily."
  if [[ "${ALLOW_UNSUPPORTED_PYTHON:-0}" != "1" ]]; then
    exit 1
  fi
fi

REQ_HASH="$(python3 - <<'PY'
from hashlib import sha256
from pathlib import Path
payload = Path("requirements.base.txt").read_bytes() + b"\n" + Path("requirements.web.txt").read_bytes()
print(sha256(payload).hexdigest())
PY
)"
REQ_HASH_FILE="$VENV_DIR/.requirements.web.sha256"

if [[ "${BOOTSTRAP_DEPS:-0}" == "1" || ! -f "$REQ_HASH_FILE" || "$(cat "$REQ_HASH_FILE")" != "$REQ_HASH" ]]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.web.txt
  printf '%s' "$REQ_HASH" > "$REQ_HASH_FILE"
fi

echo "Starting app with .env from: $ROOT_DIR/.env"
export PYTHONFAULTHANDLER=1
PORT="${PORT:-8080}"
PID_FILE="$ROOT_DIR/.local_app.pid"
LOG_FILE="$ROOT_DIR/.local_app.log"

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

wait_for_pid_exit() {
  local pid="$1"
  local timeout_seconds="${2:-30}"
  local waited=0
  while is_pid_running "$pid"; do
    if [[ "$waited" -ge "$timeout_seconds" ]]; then
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 0
}

listener_pids_for_port() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

if [[ "$STOP_ONLY" == "1" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if is_pid_running "$PID"; then
      kill "$PID"
      if wait_for_pid_exit "$PID" 30; then
        echo "Stopped local app (pid $PID)"
      else
        echo "Local app is still shutting down (pid $PID)"
        exit 1
      fi
    else
      echo "Local app pid file exists but process $PID is not running"
    fi
    rm -f "$PID_FILE"
  else
    echo "Local app is not running"
  fi
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE")"
  if is_pid_running "$EXISTING_PID"; then
    echo "Local app is already running (pid $EXISTING_PID)"
    echo "Open http://localhost:${PORT}/video-review"
    echo "To stop it: bash run_local_app.sh --stop"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

PORT_LISTENERS="$(listener_pids_for_port)"
if [[ -n "$PORT_LISTENERS" ]]; then
  echo "Local app start failed: port ${PORT} is already in use by pid(s): ${PORT_LISTENERS//$'\n'/, }"
  echo "If this is a stale local app, stop it with: bash run_local_app.sh --stop"
  echo "Otherwise free the port first, then retry."
  exit 1
fi

if [[ "$BACKGROUND" == "1" ]]; then
  gunicorn app:app \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --pid "$PID_FILE" \
    --access-logfile "$LOG_FILE" \
    --error-logfile "$LOG_FILE" \
    --daemon
  echo "Local app started in background"
  echo "URL: http://localhost:${PORT}/video-review"
  echo "PID file: $PID_FILE"
  echo "Log file: $LOG_FILE"
  echo "Stop with: bash run_local_app.sh --stop"
  exit 0
fi

gunicorn app:app \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --threads 4 \
  --timeout 600 \
  --pid "$PID_FILE" \
  --access-logfile - \
  --error-logfile -
