#!/usr/bin/env bash
# start_rtsp_test.sh — 一键启动本地 RTSP 测试服务器 (后台模式)
#
# 拉起的进程:
#   [1] MediaMTX RTSP Server   (Docker, 8554)
#   [2] FFmpeg 循环推流          (MP4 -> RTSP)
#
# 结果 RTSP 地址: rtsp://127.0.0.1:8554/blade
#
# 用法:
#   scripts/start_rtsp_test.sh [video.mp4]
#   scripts/stop_rtsp_test.sh        # 停止
#
set -euo pipefail

VIDEO="${1:-data/training-data/test01-video-2026-08-28_134655_607.mp4}"
RTSP_URL="rtsp://127.0.0.1:8554/blade"
PORT=8554

log() { echo "[rtsp-test] $*"; }

if [ ! -f "$VIDEO" ]; then
  log "ERROR: 找不到视频文件 $VIDEO" >&2
  exit 1
fi

# ---- 1. MediaMTX (后台容器) ----
log "启动 MediaMTX ..."
if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx mediamtx; then
  docker rm -f mediamtx >/dev/null 2>&1 || true
  docker run -d --rm \
    --name mediamtx \
    -e MTX_RTSPTRANSPORTS=tcp \
    -p "${PORT}:8554" \
    bluenviron/mediamtx:1 >/dev/null
  log "MediaMTX 容器已启动 (端口 $PORT)"
else
  log "MediaMTX 已在运行"
fi

# ---- 2. FFmpeg 推流 (后台进程, PID 写入 pid 文件) ----
PIDFILE=".rtsp_ffmpeg.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  log "FFmpeg 已在运行 (pid $(cat "$PIDFILE"))"
else
  log "启动 FFmpeg 推流 ..."
  nohup ffmpeg \
    -re \
    -stream_loop -1 \
    -i "$VIDEO" \
    -c copy \
    -f rtsp \
    -rtsp_transport tcp \
    "$RTSP_URL" > .rtsp_ffmpeg.log 2>&1 &
  echo $! > "$PIDFILE"
  log "FFmpeg 已启动 (pid $(cat "$PIDFILE"))"
fi

# ---- 3. 等待并探测 (非交互) ----
sleep 4
log "探测 RTSP 流 (ffprobe):"
if ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 "$RTSP_URL"; then
  log "RTSP 链路就绪: $RTSP_URL"
else
  log "WARNING: ffprobe 暂未探测到流 (推流启动较慢属正常), 稍后可用 scripts/verify_rtsp.py 复验"
fi

log "停止: scripts/stop_rtsp_test.sh"
