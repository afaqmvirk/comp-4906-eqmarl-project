#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
REFERENCE_VENV="${REFERENCE_VENV:-${HOME}/.venvs/eqmarl-paper-2024}"

if [[ ! -x "${UV_BIN}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

"${UV_BIN}" python install 3.9.19
if [[ ! -x "${REFERENCE_VENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.9.19 "${REFERENCE_VENV}"
fi
"${UV_BIN}" pip install \
  --python "${REFERENCE_VENV}/bin/python" \
  --require-hashes \
  -r "${PROJECT_ROOT}/requirements-paper-2024.lock"
"${UV_BIN}" pip check --python "${REFERENCE_VENV}/bin/python"

"${REFERENCE_VENV}/bin/python" - <<'PY'
import cirq
import gymnasium
import numpy
import tensorflow
import tensorflow_quantum

print("TensorFlow", tensorflow.__version__)
print("TensorFlow Quantum", tensorflow_quantum.__version__)
print("Cirq", cirq.__version__)
print("Gymnasium", gymnasium.__version__)
print("NumPy", numpy.__version__)
PY
