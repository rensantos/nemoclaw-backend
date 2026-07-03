#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

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
