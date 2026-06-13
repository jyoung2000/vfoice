#!/usr/bin/env bash
# Launch Inflect Studio (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv in $VENV ..."
    "$PYTHON" -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "Tip: for GPU, install CUDA torch first, e.g.:"
    echo "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

exec python -m inflect "$@"
