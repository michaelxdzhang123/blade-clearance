#!/usr/bin/env bash
# start_rtsp_stream_h264.sh — H.264 重新编码推流方案 (兼容性更好)
#
# 当 -c copy 失败 (原视频编码不适合直接 RTSP Copy) 时使用本脚本。
# 数据链: MP4 -> Decode -> libx264 Encode -> RTSP
#
# 用法:
#   scripts/start_rtsp_stream_h264.sh [video.mp4] [rtsp_url]
#
set -euo pipefail

VIDEO="${1:-data/training-data/test01-video-2026-08-28_134655_607.mp4}"
RTSP_URL="${2:-rtsp://127.0.0.1:8554/blade}"

if [ ! -f "$VIDEO" ]; then
  echo "[stream-h264] ERROR: 找不到视频文件 $VIDEO" >&2
  exit 1
fi

echo "[stream-h264] 视频: $VIDEO"
echo "[stream-h264] RTSP: $RTSP_URL"

exec ffmpeg \
  -re \
  -stream_loop -1 \
  -i "$VIDEO" \
  -an \
  -c:v libx264 \
  -preset veryfast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  "$RTSP_URL"
