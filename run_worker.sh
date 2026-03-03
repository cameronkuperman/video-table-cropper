#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -d "venv" ]]; then
  VENV_DIR="venv"
elif [[ -d ".venv" ]]; then
  VENV_DIR=".venv"
else
  VENV_DIR=".venv"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
export TOKENIZERS_PARALLELISM=false

LIST_SOURCE_FOLDERS_ONLY=0
for arg in "$@"; do
  if [[ "$arg" == "--list-source-folders" ]]; then
    LIST_SOURCE_FOLDERS_ONLY=1
    break
  fi
done

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$PYTHON_VERSION" == "3.14" || "$PYTHON_VERSION" == "3.15" || "$PYTHON_VERSION" == "4.0" ]]; then
  echo "Worker warning: Python $PYTHON_VERSION is not the recommended local runtime for this project."
  echo "Use Python 3.12 for local testing when possible. The production worker image does not use this runtime."
fi

REQ_HASH="$(python3 - <<'PY'
from hashlib import sha256
from pathlib import Path
payload = Path("requirements.base.txt").read_bytes() + b"\n" + Path("requirements.worker.txt").read_bytes()
print(sha256(payload).hexdigest())
PY
)"
REQ_HASH_FILE="$VENV_DIR/.requirements.worker.sha256"

if [[ "${BOOTSTRAP_DEPS:-0}" == "1" || ! -f "$REQ_HASH_FILE" || "$(cat "$REQ_HASH_FILE")" != "$REQ_HASH" ]]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.worker.txt
  printf '%s' "$REQ_HASH" > "$REQ_HASH_FILE"
fi

if [[ "$LIST_SOURCE_FOLDERS_ONLY" != "1" ]] && ! python3 - <<'PY'
try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
then
  echo
  echo "Worker preflight failed: torch and torchvision must both be installed in $VENV_DIR."
  echo "The video worker needs both before the SAM3 transformers backend can import."
  echo "For real processing, use a Linux CUDA box. For a local smoke test, install CPU wheels first."
  exit 1
fi

echo "Starting video worker with .env from: $ROOT_DIR/.env"
python3 video_dataset_worker.py "$@"
