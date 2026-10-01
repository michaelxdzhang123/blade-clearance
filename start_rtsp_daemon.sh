#!/usr/bin/env bash
# =============================================================================
# 启动 blade_rtsp_system/rtsp_daemon.py (RTSP 消费守护, uv 环境)
# =============================================================================
# 功能: 连接 RTSP 视频源 → 持续读帧 → 断流自动重连 → 供外部取最新帧。
#       是 start_mp4_to_rtsp.sh (推流侧) 的对称消费侧。
#
# 用法:
#   ./start_rtsp_daemon.sh            # 用下方默认参数运行
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
RTSP_URL="rtsp://127.0.0.1:8554/blade"  # RTSP 视频源地址
RECONNECT_INTERVAL=2.0                   # 断流重连等待间隔 (秒)
MAX_RECONNECT_ATTEMPTS=0                 # 重连次数上限 (0 = 无限重连)
STATS_INTERVAL=5.0                       # 打印统计的间隔 (秒)
MAX_DURATION=0.0                         # 最大运行秒数 (0 = 不限)
SAVE_DIR="./data/rtsp-daemon"                              # 可选: 每帧存 jpg 到该目录 (留空 = 不存帧)

# 前置检查: uv 必须可用
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 未找到 uv, 请先安装 (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

# 组装命令 (必传参数全部显式给出)
CMD=(
  uv run python blade_rtsp_system/rtsp_daemon.py
  --url "$RTSP_URL"
  --reconnect-interval "$RECONNECT_INTERVAL"
  --max-reconnect-attempts "$MAX_RECONNECT_ATTEMPTS"
  --stats-interval "$STATS_INTERVAL"
  --max-duration "$MAX_DURATION"
)
# --save-dir 仅在非空时附加 (空则不存帧, 保持默认行为)
if [[ -n "$SAVE_DIR" ]]; then
  CMD+=(--save-dir "$SAVE_DIR")
fi

echo "启动 RTSP 消费守护: $RTSP_URL"
exec "${CMD[@]}"
