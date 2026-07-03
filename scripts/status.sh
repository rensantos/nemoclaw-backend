#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${HOME}/miniforge3/bin/activate"
conda activate llm

CONFIG_OUTPUT="$(cd "${PROJECT_ROOT}" && python -m config)"

HOST="$(printf "%s\n" "${CONFIG_OUTPUT}" | awk -F': ' '/^Host:/ {print $2}')"
PORT="$(printf "%s\n" "${CONFIG_OUTPUT}" | awk -F': ' '/^Port:/ {print $2}')"

echo "Configuration:"
printf "%s\n" "${CONFIG_OUTPUT}"
echo
echo "Port ${PORT}:"
if command -v ss >/dev/null 2>&1; then
  ss -ltn | awk '{print $4}' | grep -E "(:|\\.)${PORT}$" || true
elif command -v netstat >/dev/null 2>&1; then
  netstat -ltn | awk '{print $4}' | grep -E "(:|\\.)${PORT}$" || true
else
  echo "ss/netstat not available"
fi

echo
echo "Health:"
curl -sS "http://${HOST}:${PORT}/health" || true

echo
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits || true
