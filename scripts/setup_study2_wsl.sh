#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${HOME}/.local/bin/uv"
VENV_DIR="${HOME}/.venvs/eqmarl-study2"
cd "${PROJECT_ROOT}"

if [[ ! -x "${UV_BIN}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

"${UV_BIN}" python install 3.9
mkdir -p "$(dirname "${VENV_DIR}")"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.9 "${VENV_DIR}"
fi
"${UV_BIN}" pip install \
  --python "${VENV_DIR}/bin/python" \
  -r "${PROJECT_ROOT}/requirements-study2.lock"

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/upstream/eqmarl" \
  TF_CPP_MIN_LOG_LEVEL=2 \
  "${VENV_DIR}/bin/python" -c \
  "import tensorflow as tf, tensorflow_quantum as tfq, cirq, eqmarl; print(tf.__version__, tfq.__version__, cirq.__version__)"

printf 'Study 2 environment ready: %s\n' "${VENV_DIR}"
