#!/usr/bin/env bash
# start_rtsp_stream_daemon.sh — 后台循环推流 MP4 到 MediaMTX (setsid 脱离会话)
#
# 数据链: MP4 -> FFmpeg(-re, -stream_loop -1, -c copy) -> RTSP -> MediaMTX
# 用法:
#   scripts/start_rtsp_stream_daemon.sh [video.mp4] [rtsp_url]
#   scripts/stop_rtsp_stream_daemon.sh
#
set -euo pipefail

VIDEO="${1:-data/training-data/test01-video-2026-08-28_134655_607.mp4}"
RTSP_URL="${2:-rtsp://127.0.0.1:8554/blade}"
PIDFILE="/tmp/rtsp_ffmpeg.pid"
LOGFILE="/tmp/rtsp_ffmpeg.log"

if [ ! -f "$VIDEO" ]; then
  echo "[stream] ERROR: 找不到视频 $VIDEO" >&2
  exit 1
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[stream] 已在运行 (pid $(cat "$PIDFILE"))"
  exit 0
fi

echo "[stream] 视频: $VIDEO"
echo "[stream] RTSP: $RTSP_URL"

setsid ffmpeg \
  -re \
  -stream_loop -1 \
  -i "$VIDEO" \
  -c copy \
  -f rtsp \
  -rtsp_transport tcp \
  "$RTSP_URL" > "$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 2

if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[stream] 已启动 (pid $(cat "$PIDFILE")), 日志 $LOGFILE"
else
  echo "[stream] 启动失败, 日志:" >&2
  tail -20 "$LOGFILE" >&2
  exit 1
fi
