#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_env.sh -- Set up Python virtual environment and install dependencies
# Usage: ./setup_env.sh [ENV_DIR] [PYTHON_BIN]
#   ENV_DIR     Name/path of the virtual environment directory (default: .venv)
#   PYTHON_BIN  Python executable to use (default: python3)
#
# This script:
#   1. Creates a virtualenv
#   2. Activates it
#   3. Upgrades pip
#   4. Installs all libraries from requirements.txt
#   5. Re-installs PyTorch (torch, torchvision, torchaudio) compatible with the detected CUDA version (or CPU build if CUDA is not found)
# ---------------------------------------------------------------------------

set -euo pipefail

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${2:-python3}"

printf "\033[36m[INFO] Creating virtual environment in '%s' using %s...\033[0m\n" "$ENV_DIR" "$PYTHON_BIN"

if [ ! -d "$ENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
else
  printf "\033[33m[WARN] Virtual environment already exists, skipping creation.\033[0m\n"
fi

# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

printf "\033[36m[INFO] Upgrading pip...\033[0m\n"
pip install --upgrade pip

if [ ! -f requirements.txt ]; then
  printf "\033[31m[ERROR] requirements.txt not found in current directory.\033[0m\n" >&2
  exit 1
fi

printf "\033[36m[INFO] Installing dependencies from requirements.txt...\033[0m\n"
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# (Re)install PyTorch according to detected CUDA version
# ---------------------------------------------------------------------------

printf "\033[36m[INFO] Removing any existing PyTorch packages (torch, torchvision, torchaudio)...\033[0m\n"
# Ignore errors if they are not installed yet
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

# Detect CUDA version (try nvcc first, fallback to nvidia-smi)
printf "\033[36m[INFO] Detecting CUDA version...\033[0m\n"
CUDA_VERSION=""

if command -v nvcc >/dev/null 2>&1; then
  CUDA_VERSION=$(nvcc --version | grep -o -E "release [0-9]+\\.[0-9]+" | awk '{print $2}')
elif command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VERSION=$(nvidia-smi | grep -o -E "CUDA Version: [0-9]+\\.[0-9]+" | awk '{print $3}')
fi

# Decide index-url based on detection result
if [ -z "$CUDA_VERSION" ]; then
  printf "\033[33m[WARN] CUDA not detected. Installing CPU-only PyTorch...\033[0m\n"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
  CUDA_DIGITS=$(echo "$CUDA_VERSION" | tr -d '.')
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_DIGITS}"
  printf "\033[36m[INFO] CUDA %s detected. Using PyTorch wheels from %s\033[0m\n" "$CUDA_VERSION" "$TORCH_INDEX_URL"
fi

printf "\033[36m[INFO] Installing PyTorch (torch, torchvision, torchaudio)...\033[0m\n"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

printf "\n\033[32m[SUCCESS] Environment setup completed.\033[0m\n"
printf "\nTo activate the environment later run:\n  source %s/bin/activate\n" "$ENV_DIR" 