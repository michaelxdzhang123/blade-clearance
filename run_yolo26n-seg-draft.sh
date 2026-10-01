#!/usr/bin/env bash
set -euo pipefail

# ── Experiment parameters ────────────────────────────────────────────────
LABELED_ROOT="train-0915-640"       # run_train_seg.sh 自动生成的 auto-label 数据集
UNLABELED_DIR="unlabeled-0915-640-e1"  # create_unlabeled_dataset.py 生成(640, every=1)
WORK_DIR="train-seg-draft-0916"
IMG_SIZE=640                         # 允许: 640 / 1024 / 1280
BATCH=80                              # 640 下 batch 80
WORKERS=0                            # CPU 调试阶段禁用 DataLoader 子进程
EPOCHS_STAGE1=200
EPOCHS_STAGE2=150
PSEUDO_CONF=0.30
MIN_AREA_RATIO=0.00075              # 伪标签 mask 面积下限(=307px), 滤雨滴/雾斑/噪点
DEVICE=1

case "$IMG_SIZE" in
  640|1024|1280) ;;
  *) echo "ERROR: IMG_SIZE must be 640, 1024, or 1280" >&2; exit 1 ;;
esac

if [[ ! -d "$LABELED_ROOT/images/train" || ! -d "$LABELED_ROOT/labels/train" ]]; then
  echo "ERROR: auto-labeled dataset is incomplete: $LABELED_ROOT" >&2
  exit 1
fi
if [[ ! -d "$UNLABELED_DIR" ]]; then
  echo "ERROR: unlabeled directory does not exist: $UNLABELED_DIR" >&2
  echo "Create it with yolo26n-seg/create_unlabeled_dataset.py first." >&2
  exit 1
fi

# Remove only the previous semi-supervised run output.
rm -rf -- "$WORK_DIR"

echo "Auto-labeled root: $LABELED_ROOT"
echo "Unlabeled input:   $UNLABELED_DIR"
echo "Image size:        $IMG_SIZE"
echo "Batch/workers:     $BATCH/$WORKERS"

uv run python ./yolo26n-seg/train_blade_yolo26_semisupervised.py \
  --labeled-root "$LABELED_ROOT" \
  --unlabeled-dir "$UNLABELED_DIR" \
  --work-dir "$WORK_DIR" \
  --names blade \
  --model-yaml yolo26n-seg/26/yolo26-seg.yaml \
  --imgsz "$IMG_SIZE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --epochs-stage1 "$EPOCHS_STAGE1" \
  --epochs-stage2 "$EPOCHS_STAGE2" \
  --pseudo-conf "$PSEUDO_CONF" \
  --min-area-ratio "$MIN_AREA_RATIO" \
  --device "$DEVICE"
