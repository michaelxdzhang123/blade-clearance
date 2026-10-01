#!/usr/bin/env bash
set -euo pipefail

# 只生成自动标注数据集，不启动 YOLO 训练。
# train_from_videos.py 按每帧实际 frame_w/frame_h 归一化，支持混合分辨率 MP4。
uv run ./yolo26/train_from_videos.py \
  --skip-train \
  --config ./yolo26/train-0914/train_config.yaml
