#!/usr/bin/env bash
# =============================================================================
# 启动 blade_rtsp_system/mp4_to_rtsp_daemon.py (uv 虚拟环境)
# =============================================================================
# 功能: 读本地 MP4 → 推流到 RTSP 服务器 (加 --with-server 可自包含拉起 MediaMTX)。
#       是 start_rtsp_daemon.sh (消费侧) 的对称推流侧。
#
# 用法:
#   ./start_mp4_to_rtsp.sh            # 用下方默认参数运行
#   或直接改脚本顶部的「输入参数」区
#
# 说明:
#   - 用 `uv run` 在项目 uv 环境里运行 (按 pyproject.toml 解析依赖)
#   - 用 `exec` 把 shell 替换为 python 进程, Ctrl+C / SIGTERM 直达守护进程优雅退出
# =============================================================================
set -euo pipefail

# 项目根目录 = 本脚本所在目录 (从任意 cwd 运行都能正确定位)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ── 输入参数 (按需修改) ────────────────────────────────────────────────
INPUT="data/training-data/test01-video-2026-08-28_134655_607.mp4"       # 本地 MP4 文件路径
RTSP_URL="rtsp://127.0.0.1:8554/blade"      # 推流目标 RTSP 地址
WITH_SERVER=1                               # 1=同时拉起 MediaMTX 服务器 (自包含模式)
RE_ENCODE=0                                 # 1=强制 libx264 转码 (原编码不兼容 RTSP 时)
NO_LOOP=0                                   # 1=不循环 (播完一次即停)
MAX_RESTARTS=0                              # 推流崩溃后重启次数上限 (0 = 无限)
RESTART_BACKOFF=2.0                         # 重启前等待秒数
MAX_DURATION=0.0                            # 最大运行秒数 (0 = 不限)
FFMPEG=""                                   # ffmpeg 二进制路径 (留空 = 自动探测)
MEDIAMTX=""                                 # mediamtx 二进制路径 (留空 = 自动探测)
FFMPEG_LOG="./log/mp4_to_rtsp_ffmpeg.log"    # ffmpeg 日志路径
MEDIAMTX_LOG="./log/mp4_to_rtsp_mediamtx.log"  # mediamtx 日志路径

# 前置检查: uv 必须可用
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 未找到 uv, 请先安装 (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

# 组装命令 (必传参数全部显式给出)
CMD=(
  uv run python blade_rtsp_system/mp4_to_rtsp_daemon.py
  --input "$INPUT"
  --rtsp-url "$RTSP_URL"
  --max-restarts "$MAX_RESTARTS"
  --restart-backoff "$RESTART_BACKOFF"
  --max-duration "$MAX_DURATION"
  --ffmpeg-log "$FFMPEG_LOG"
  --mediamtx-log "$MEDIAMTX_LOG"
)
# 布尔开关: 置 1 才附加
[[ "$WITH_SERVER" == "1" ]] && CMD+=(--with-server)
[[ "$RE_ENCODE" == "1" ]] && CMD+=(--re-encode)
[[ "$NO_LOOP" == "1" ]] && CMD+=(--no-loop)
# 可选二进制路径: 非空才附加
[[ -n "$FFMPEG" ]] && CMD+=(--ffmpeg "$FFMPEG")
[[ -n "$MEDIAMTX" ]] && CMD+=(--mediamtx "$MEDIAMTX")

echo "推流: $INPUT → $RTSP_URL"
exec "${CMD[@]}"
