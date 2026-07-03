#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${HOME}/miniforge3/bin/activate"
conda activate llm

cd "${PROJECT_ROOT}"

export MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
export GPU="${GPU:-0}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"

CUDA_VISIBLE_DEVICES="${GPU}" uvicorn server:app --host "${HOST}" --port "${PORT}"
