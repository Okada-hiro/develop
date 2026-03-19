#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

echo "[environment.sh] using Python: $("$PYTHON_BIN" --version 2>&1)"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# Colab often comes with preinstalled torch packages. Remove them first so the
# stack below is reinstalled as a clean, matching set.
"$PYTHON_BIN" -m pip uninstall -y \
  pyannote.audio \
  lightning \
  pytorch-lightning \
  torchmetrics \
  torch \
  torchaudio \
  torchvision || true

if [[ -z "${TORCH_INDEX_URL}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
  else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  fi
fi

# pyannote.audio 3.4.0 pulls in torchmetrics/pytorch-lightning, so keep the
# full torch stack aligned. PyTorch's compatibility table lists:
# torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0
"$PYTHON_BIN" -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

# Install this Whisper checkout in editable mode.
"$PYTHON_BIN" -m pip install -e ./whisper

# Speaker diarization / VAD stack used by develop/*.py
"$PYTHON_BIN" -m pip install "pyannote.audio==3.4.0"

cat <<'EOF'

[environment.sh] pip dependencies installed.

Notes:
- Torch wheel index: ${TORCH_INDEX_URL}
- Whisper audio loading still requires ffmpeg to be available on PATH.
- pyannote models require a Hugging Face token and accepted model terms.
- In Colab, always restart the runtime once after this script finishes.
- If torchvision was preinstalled in Colab, this script replaces it with a matching version.
- On Colab, run: !bash develop/environment.sh

EOF
