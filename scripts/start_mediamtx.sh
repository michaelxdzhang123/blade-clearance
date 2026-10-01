#!/usr/bin/env bash
# start_mediamtx.sh — 启动 MediaMTX RTSP Server (监听 8554)
#
# 优先使用 Docker；若 Docker 不可用则尝试本机 mediamtx 二进制。
# RTSP 地址: rtsp://127.0.0.1:8554/<path>
#
# 用法:
#   scripts/start_mediamtx.sh [port]
#
set -euo pipefail

PORT="${1:-8554}"
CONTAINER="mediamtx"
IMAGE="bluenviron/mediamtx:1"

log() { echo "[mediamtx] $*"; }

# 1) 已存在同名容器 -> 直接复用
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    log "restarting existing container $CONTAINER"
    docker start "$CONTAINER" >/dev/null
  else
    log "container $CONTAINER already running"
  fi
  exit 0
fi

# 2) Docker 可用 -> 拉起容器
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  log "starting MediaMTX via Docker on port $PORT"
  exec docker run --rm -it \
    --name "$CONTAINER" \
    -e MTX_RTSPTRANSPORTS=tcp \
    -p "${PORT}:8554" \
    "$IMAGE"
fi

# 3) 回退到本机二进制
if command -v mediamtx >/dev/null 2>&1; then
  log "starting native mediamtx on port $PORT"
  exec mediamtx
fi

log "ERROR: 既无可用 Docker 也无 mediamtx 二进制。"
log "请安装 Docker，或从 https://github.com/bluenviron/mediamtx/releases 下载二进制。"
exit 1
