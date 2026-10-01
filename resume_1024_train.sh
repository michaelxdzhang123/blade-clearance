#!/usr/bin/env bash
# =============================================================================
# 一键恢复 1024 半监督训练(重启 WSL 清理僵尸显存后直接跑)
# =============================================================================
# 用途: 解决 PID 167 幽灵进程 + GPU 0 的 41GB 僵尸显存导致的反复 CUDA OOM。
#       重启 WSL 后, GPU 显存全部清空, 直接跑本脚本即可恢复 1024 训练。
#
# 重启操作步骤:
#   1. 在 Windows PowerShell 执行:  wsl --shutdown
#   2. 重新打开 WSL 终端(会自动重启)
#   3. 启动 Hermes:  hermes -p thinker   (或你平时的方式)
#   4. 在 Hermes 里让我跑, 或直接终端执行:
#        cd /home/mich/project/blade-clearance && bash resume_1024_train.sh
#
# 说明: 数据(train-0915-640 + unlabeled-0915-1024-e1)已生成好, 本脚本只跑训练,
#       不重新生成数据, 省去 run_train_seg.sh + create_unlabeled_dataset.py 的时间。
# =============================================================================
set -euo pipefail
cd /home/mich/project/blade-clearance

echo "=== [$(date '+%F %T')] 一键恢复 1024 半监督训练 ==="

# ── 1. 前置检查: 数据是否就绪 ──────────────────────────────
echo ""
echo "--- 前置检查 ---"
OK=1
if [ ! -d "train-0915-640/images/train" ] || [ ! -d "train-0915-640/labels/train" ]; then
  echo "❌ train-0915-640 不完整(需先跑 run_train_seg.sh)" >&2
  OK=0
fi
if [ ! -d "unlabeled-0915-1024-e1" ]; then
  echo "❌ unlabeled-0915-1024-e1 不存在(需先生成 unlabeled)" >&2
  OK=0
fi
if [ "$OK" -eq 0 ]; then
  echo ""
  echo "数据不全, 请先跑完整流水线:  bash run_1024_pipeline.sh" >&2
  exit 1
fi
echo "✓ train-0915-640: $(ls train-0915-640/images/train/*.jpg | wc -l) train + $(ls train-0915-640/images/val/*.jpg | wc -l) val"
echo "✓ unlabeled-0915-1024-e1: $(ls unlabeled-0915-1024-e1/*.jpg | wc -l) 张"

# ── 2. GPU 状态检查(重启后应干净) ─────────────────────────
echo ""
echo "--- GPU 状态(重启后应无僵尸显存) ---"
if command -v nvidia-smi >/dev/null 2>&1 || [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
else
  echo "⚠️ 找不到 nvidia-smi, 可能驱动未加载, 稍等或重启后再试"
fi

# ── 3. 启动训练 ───────────────────────────────────────────
echo ""
echo "--- 启动训练(batch=16 @ 1024, conf=0.30) ---"
echo "日志: run_1024_pipeline.log 之外的训练输出直接打到终端"
echo ""

# 训练输出同时 tee 到日志, 便于事后追溯
bash run_yolo26n-seg-draft.sh 2>&1 | tee -a resume_1024_train.log

RC=${PIPESTATUS[0]}
echo ""
if [ "$RC" -eq 0 ]; then
  echo "=== ✅ 训练完成 $(date '+%F %T') ==="
  echo "Stage2 best.pt: train-seg-draft-0916/training/stage2_auto_labeled_plus_pseudo/weights/best.pt"
  echo "伪标签审计:     train-seg-draft-0916/pseudo_label_audit.csv"
else
  echo "=== ❌ 训练失败(exit $RC) $(date '+%F %T') ==="
  echo "若仍是 CUDA OOM, 说明 1024 显存需求超出单卡 49GB, 建议回 640:"
  echo "  改 run_yolo26n-seg-draft.sh: IMG_SIZE=640, BATCH=80, UNLABELED_DIR=unlabeled-0915-640-e1"
fi
