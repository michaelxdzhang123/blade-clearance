#!/usr/bin/env bash
# stop_rtsp_stream_daemon.sh — 停止 FFmpeg 推流
set -euo pipefail

PIDFILE="/tmp/rtsp_ffmpeg.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "[stop] 停止 FFmpeg 推流 (pid $PID)"
    kill "$PID" 2>/dev/null || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi
echo "[stop] 完成"
