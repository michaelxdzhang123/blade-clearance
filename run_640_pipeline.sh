#!/usr/bin/env bash
# =============================================================================
# 640 半监督三步流水线(从 run_1024_pipeline.sh 派生, 作为 1024 OOM 时的兜底)
# =============================================================================
# 三步: 生成 640 labeled → 生成 640 unlabeled → 训练(640, conf=0.30, batch=80)
# 640 已验证能跑通且伪标签生效(Stage2 mask mAP50=0.730), 是最稳的兜底方案。
#
# 注意: 本脚本会临时把 train_from_videos_seg.py 的 WRITE_IMG_SIZE 和
#       run_yolo26n-seg-draft.sh 的 IMG_SIZE/BATCH/UNLABELED_DIR 切到 640。
#       切换后不回改(保持 640 状态), 若要回到 1024 需手动改回或重跑 1024 流水线。
# =============================================================================
set -euo pipefail
cd /home/mich/project/blade-clearance
LOG="$PWD/run_640_pipeline.log"

run_step() {
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] $name ==="
  "$@"
}

{
  # ── 切换到 640 参数 ──────────────────────────────────────
  echo "--- 切换标注程序 WRITE_IMG_SIZE 到 640 ---"
  sed -i 's/^WRITE_IMG_SIZE = .*/WRITE_IMG_SIZE = 640  # 数据集写入图片边长/' yolo26/train_from_videos_seg.py
  grep -n "WRITE_IMG_SIZE =" yolo26/train_from_videos_seg.py

  echo "--- 切换训练脚本到 640 / batch 80 / unlabeled-640 ---"
  sed -i 's/^IMG_SIZE=.*/IMG_SIZE=640                         # 允许: 640 \/ 1024 \/ 1280/' run_yolo26n-seg-draft.sh
  sed -i 's/^BATCH=.*/BATCH=80                              # 640 下 batch 80/' run_yolo26n-seg-draft.sh
  sed -i 's/^UNLABELED_DIR=.*/UNLABELED_DIR="unlabeled-0915-640-e1"  # create_unlabeled_dataset.py 生成(640, every=1)/' run_yolo26n-seg-draft.sh
  grep -E "IMG_SIZE=|BATCH=|UNLABELED_DIR=" run_yolo26n-seg-draft.sh

  # ── STEP1: 生成 640 labeled ─────────────────────────────
  run_step "STEP1: 生成 640 labeled (run_train_seg.sh)" bash run_train_seg.sh || { echo "STEP1 FAILED"; exit 1; }

  # ── STEP2: 生成 640 unlabeled ───────────────────────────
  if [ -d "unlabeled-0915-640-e1" ] && [ -f "unlabeled-0915-640-e1/manifest.json" ]; then
    echo "--- unlabeled-0915-640-e1 已存在, 跳过 STEP2 ---"
  else
    run_step "STEP2: 生成 640 unlabeled (every=8)" \
      uv run python yolo26n-seg/create_unlabeled_dataset.py \
        --config yolo26/train-0915/train_config.yaml \
        --labeled-root train-0915-640 \
        --out unlabeled-0915-640-e1 \
        --imgsz 640 \
        --every 8 || { echo "STEP2 FAILED"; exit 1; }
  fi

  # ── STEP3: 训练 ─────────────────────────────────────────
  run_step "STEP3: 半监督训练 (640, conf=0.30, batch=80)" bash run_yolo26n-seg-draft.sh || { echo "STEP3 FAILED"; exit 1; }

  echo "=== [$(date '+%F %T')] ALL DONE ==="
} > "$LOG" 2>&1

RC=$?
if [ "$RC" -eq 0 ]; then
  echo "OK 640 半监督流水线全部完成: $(date '+%F %T')"
  if [ -f train-seg-draft-0916/pseudo_label_audit.csv ]; then
    ACCEPTED=$(uv run python -c "import csv; print(sum(1 for r in csv.DictReader(open('train-seg-draft-0916/pseudo_label_audit.csv')) if r['status']=='accepted'))" 2>/dev/null || echo "?")
    echo "伪标签 accepted 图数: $ACCEPTED"
  fi
else
  echo "FAIL 640 半监督流水线失败(第 $RC 步): $(date '+%F %T')"
fi
echo "日志: $LOG"
echo "关键产物: train-seg-draft-0916/training/stage2_auto_labeled_plus_pseudo/weights/best.pt"
