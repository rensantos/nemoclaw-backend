#!/usr/bin/env bash
set -euo pipefail

pkill -f "uvicorn server:app --host .* --port .*" || true
pkill -f "python server.py" || true
