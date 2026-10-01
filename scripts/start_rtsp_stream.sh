#!/usr/bin/env bash
# start_rtsp_stream.sh — 将本地 MP4 按实时速度循环推送到 MediaMTX
#
# 模拟一台"虚拟网络摄像机"，供 TASK-1 YOLO26 实时推理测试。
# 数据链: MP4 -> FFmpeg(-re, -stream_loop -1) -> RTSP -> MediaMTX
#
# 用法:
#   scripts/start_rtsp_stream.sh [video.mp4] [rtsp_url]
#
# 默认 RTSP 地址: rtsp://127.0.0.1:8554/blade
#
set -euo pipefail

VIDEO="${1:-data/training-data/test01-video-2026-08-28_134655_607.mp4}"
RTSP_URL="${2:-rtsp://127.0.0.1:8554/blade}"

if [ ! -f "$VIDEO" ]; then
  echo "[stream] ERROR: 找不到视频文件 $VIDEO" >&2
  exit 1
fi

echo "[stream] 视频: $VIDEO"
echo "[stream] RTSP: $RTSP_URL"
echo "[stream] 编码检测:"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 "$VIDEO" || true

# 优先 -c copy (低 CPU、无损)；失败则由用户改用 start_rtsp_stream_h264.sh
echo "[stream] 启动推流 (Ctrl+C 停止)..."
exec ffmpeg \
  -re \
  -stream_loop -1 \
  -i "$VIDEO" \
  -c copy \
  -f rtsp \
  -rtsp_transport tcp \
  "$RTSP_URL"
