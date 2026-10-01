#!/usr/bin/env bash
# stop_rtsp_test.sh — 停止 RTSP 测试服务器 (FFmpeg 推流 + MediaMTX)
set -euo pipefail

PIDFILE=".rtsp_ffmpeg.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "[stop] 停止 FFmpeg (pid $PID)"
    kill "$PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

echo "[stop] 停止 MediaMTX 容器"
docker rm -f mediamtx >/dev/null 2>&1 || true

echo "[stop] 完成"
