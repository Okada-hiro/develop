#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[environment.sh] using Python: $("$PYTHON_BIN" --version 2>&1)"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# Install this Whisper checkout in editable mode.
"$PYTHON_BIN" -m pip install -e ./whisper

# Speaker diarization / VAD stack used by develop/*.py
"$PYTHON_BIN" -m pip install "pyannote.audio==3.4.0"

cat <<'EOF'

[environment.sh] pip dependencies installed.

Notes:
- Whisper audio loading still requires ffmpeg to be available on PATH.
- pyannote models require a Hugging Face token and accepted model terms.
- On Colab, run: !bash environment.sh

EOF
