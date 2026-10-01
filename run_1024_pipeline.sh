#!/usr/bin/env bash
# 1024 半监督三步流水线: 生成 1024 labeled → 生成 1024 unlabeled → 训练
# 详细输出写入 run_1024_pipeline.log; stdout 只输出简短结果供 cron 投递。
cd /home/mich/project/blade-clearance
LOG="$PWD/run_1024_pipeline.log"

run_step() {
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] $name ==="
  "$@"
}

{
  run_step "STEP1: 生成 1024 labeled (run_train_seg.sh)" bash run_train_seg.sh || { echo "STEP1 FAILED"; exit 1; }

  run_step "STEP2: 生成 1024 unlabeled (every=1)" \
    uv run python yolo26n-seg/create_unlabeled_dataset.py \
      --config yolo26/train-0915/train_config.yaml \
      --labeled-root train-0915-640 \
      --out unlabeled-0915-1024-e1 \
      --imgsz 1024 \
      --every 1 || { echo "STEP2 FAILED"; exit 1; }

  run_step "STEP3: 半监督训练 (1024, conf=0.30, batch=32)" bash run_yolo26n-seg-draft.sh || { echo "STEP3 FAILED"; exit 1; }

  echo "=== [$(date '+%F %T')] ALL DONE ==="
} > "$LOG" 2>&1

RC=$?
if [ "$RC" -eq 0 ]; then
  echo "OK 1024 半监督流水线全部完成: $(date '+%F %T')"
  if [ -f train-seg-draft-0916/pseudo_label_audit.csv ]; then
    ACCEPTED=$(uv run python -c "import csv; print(sum(1 for r in csv.DictReader(open('train-seg-draft-0916/pseudo_label_audit.csv')) if r['status']=='accepted'))" 2>/dev/null || echo "?")
    echo "伪标签 accepted 图数: $ACCEPTED"
  fi
else
  echo "FAIL 1024 半监督流水线失败(第 $RC 步): $(date '+%F %T')"
fi
echo "日志: $LOG"
echo "关键产物: train-seg-draft-0916/training/stage2_auto_labeled_plus_pseudo/weights/best.pt"
