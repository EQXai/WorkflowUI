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
#   5. Installs PyTorch, torchvision y torchaudio con soporte CUDA 12.8
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

printf "\033[36m[INFO] Installing PyTorch (CUDA 12.8) ...\033[0m\n"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

printf "\n\033[32m[SUCCESS] Environment setup completed.\033[0m\n"
printf "\nTo activate the environment later run:\n  source %s/bin/activate\n" "$ENV_DIR" 