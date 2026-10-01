#!/usr/bin/env bash
set -euo pipefail


# 当前入口仅用于生成 segmentation labeling 数据集。
# 暂不启动训练；训练入口由 yolo26n-seg/train_blade_yolo26_semisupervised.py 单独负责。
CONFIG="./yolo26/train-0915/train_config.yaml"
OUT_DIR_RAW="$(uv run python -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["dataset"]["output"])' \
    "$CONFIG")"
IMG_SIZE="$(uv run python -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["train"]["imgsz"])' \
    "$CONFIG")"
if [[ -z "$IMG_SIZE" ]]; then
    echo "ERROR: train.imgsz is missing in $CONFIG" >&2
    exit 1
fi
OUT_DIR="$(realpath -m -- "$OUT_DIR_RAW")"

# 防止配置错误导致 rm -rf 删除根目录、当前项目目录或空路径。
PROJECT_ROOT="$(realpath -m -- "$(pwd)")"
if [[ -z "$OUT_DIR_RAW" || "$OUT_DIR" == "/" || "$OUT_DIR" == "$PROJECT_ROOT" ]]; then
    echo "ERROR: unsafe dataset.output in $CONFIG: '$OUT_DIR_RAW' -> '$OUT_DIR'" >&2
    exit 1
fi

echo "清理旧 labeling 输出: $OUT_DIR"
rm -rf -- "$OUT_DIR"
mkdir -p -- "$OUT_DIR"

echo "Using image size from config: $IMG_SIZE"

uv run ./yolo26/train_from_videos_seg.py \
  --config "$CONFIG" \
  --out "$OUT_DIR" \
  --imgsz "$IMG_SIZE" \
  --skip-train


echo "Labeling verified: dataset generated under $OUT_DIR"
