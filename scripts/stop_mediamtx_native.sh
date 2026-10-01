#!/usr/bin/env bash
# stop_mediamtx_native.sh — 停止原生 mediamtx
set -euo pipefail

PIDFILE="/tmp/mediamtx.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "[stop] 停止 mediamtx (pid $PID)"
    kill "$PID" 2>/dev/null || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi
echo "[stop] 完成"
