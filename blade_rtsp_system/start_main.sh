#!/usr/bin/env bash
set -euo pipefail
# 用法: ./start_main.sh yolo [main.py 参数]
#       ./start_main.sh classic [main.py 参数]
# 别名: --with-yolo / --without-yolo；默认 yolo。
# # 使用 YOLO
./blade_rtsp_system/start_main.sh yolo 

# 无 YOLO，使用 Classic CV
./blade_rtsp_system/start_main.sh classic

# 限时运行 60 秒
./start_main.sh classic --max-duration 60
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
MODE="${BLADE_MODE:-yolo}"
case "${1:-}" in
  yolo|--with-yolo) MODE=yolo; shift ;;
  classic|--without-yolo|--no-yolo) MODE=classic; shift ;;
  -h|--help)
    printf '%s\n' 'Usage: start_main.sh [yolo|classic] [main.py options]' \
      '  yolo / --with-yolo       使用 YOLO 模型 (默认)' \
      '  classic / --without-yolo 使用 Classic CV，不加载模型' \
      '  可追加 --max-duration 60、--resize 640 360、--config /path/config.yaml'
    exit 0 ;;
  ''|--*) ;;
  *) printf 'Unknown mode: %s\n' "$1" >&2; exit 2 ;;
esac
PYTHON_BIN="${BLADE_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/main.py" --config "$SCRIPT_DIR/config/config.yaml" --mode "$MODE" "$@"
