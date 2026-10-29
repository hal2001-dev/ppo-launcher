#!/usr/bin/env bash
# TensorBoard for runs/ — port 3005 is registered in ../PORTS.md (Web band).
# Usage: scripts/tensorboard.sh            → http://localhost:3005  (LAN: http://macstudio:3005)
#        PORT=3105 HOST=127.0.0.1 scripts/tensorboard.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/tensorboard --logdir "${LOGDIR:-runs}" --port "${PORT:-3005}" --host "${HOST:-0.0.0.0}"
