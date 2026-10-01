#!/usr/bin/env bash
# start_mediamtx_native.sh — 用原生 mediamtx 二进制后台启动 RTSP Server (8554)
#
# 与 start_mediamtx.sh (Docker 优先) 不同, 本脚本用已下载的二进制 ~/.local/bin/mediamtx,
# 不依赖 Docker。用 setsid 完全脱离当前会话, 写 pid 文件, 日志到 /tmp/mediamtx.log。
#
# 用法:
#   scripts/start_mediamtx_native.sh
#   scripts/stop_mediamtx_native.sh   # 停止
#
set -euo pipefail

BIN="${MEDIAMTX_BIN:-$HOME/.local/bin/mediamtx}"
CFG="${MEDIAMTX_CFG:-$(dirname "$0")/mediamtx.yml}"
PIDFILE="/tmp/mediamtx.pid"
LOGFILE="/tmp/mediamtx.log"

if [ ! -x "$BIN" ]; then
  echo "[mediamtx-native] ERROR: 找不到 mediamtx 二进制 $BIN" >&2
  exit 1
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[mediamtx-native] 已在运行 (pid $(cat "$PIDFILE"))"
  exit 0
fi

echo "[mediamtx-native] 启动 $BIN (配置 $CFG)"
setsid "$BIN" "$CFG" > "$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 1

if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[mediamtx-native] 已启动 (pid $(cat "$PIDFILE")), 日志 $LOGFILE"
else
  echo "[mediamtx-native] 启动失败, 日志:" >&2
  tail -20 "$LOGFILE" >&2
  exit 1
fi
