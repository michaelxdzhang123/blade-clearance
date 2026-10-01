# blade-clearance

风机叶片净空监测项目:识别叶尖最靠近塔筒的距离及变化。

## 子项目

| 目录 | 说明 |
|------|------|
| `blade_rtsp_system/` | TASK-1: YOLO26 + RTSP 实时叶片定位(YOLO 分割推理 → Mask/Polygon → 原始坐标) |
| `opencv/` | 经典 CV 方案(背景差分 + 几何先验抓叶片, 无 YOLO) |
| `yolo26/` | YOLO 检测/分割训练数据生成(背景差分自动标注) |
| `yolo26n-seg/` | 半监督训练(Stage1 teacher → 伪标签 → Stage2) |
| `scripts/` | RTSP 测试服务器(MediaMTX + FFmpeg, 本地 MP4 模拟摄像机) |
| `docs/` | 任务书 + 几何方法 + 训练报告 |

## 环境

- Python 3.13, 由 uv 管理(`uv sync` 重建环境)
- torch/torchvision 从 PyTorch 官方 CUDA 13.0 (cu130) index 安装
- GPU: NVIDIA RTX 4090 × 2

## 快速开始

```bash
uv sync          # 安装依赖(含 cu130 版 torch)
uv sync --dev    # 含 pytest 等开发依赖
uv run pytest    # 跑单元测试
```

## 关键数据链

```
MP4 → 背景差分自动标注 → YOLO26-seg 训练(640/1024)
                              ↓
RTSP(MediaMTX) → Frame → YOLO best.pt → Blade Mask → Polygon → 原始坐标
```

## 中文运行命令

以下命令均在项目根目录执行：

```bash
cd /home/mich/projects/blade-clearance
```

### 环境和测试

```bash
uv sync --dev
uv run pytest
```

### 经典 CV：生成背景并检测叶片

```bash
uv run python opencv/build_bg.py \
  --video data/training-data/1-rain.mp4 \
  --output output/bg_1-rain.npy \
  --n-idle 50

uv run python opencv/catch_blade.py \
  --video data/training-data/1-rain.mp4 \
  --bg output/bg_1-rain.npy \
  --output output/catch_1-rain
```

输出：`annotated.mp4` 和 `blade_boxes.csv`。

### 生成 YOLO 检测数据集

```bash
uv run python yolo26/make_yolo_dataset.py \
  --video data/training-data/1-rain.mp4 \
  --boxes output/catch_1-rain/blade_boxes.csv \
  --out output/dataset_1-rain \
  --every 4 \
  --val-ratio 0.2
```

### 生成 YOLO 实例分割数据集（不训练）

```bash
bash run_train_seg.sh
```

或者直接执行：

```bash
uv run python yolo26/train_from_videos_seg.py \
  --config yolo26/train-0915/train_config.yaml \
  --out output/seg_dataset \
  --imgsz 1024 \
  --skip-train
```

### 生成无标签图片

```bash
uv run python yolo26n-seg/create_unlabeled_dataset.py \
  --config yolo26/train-0915/train_config.yaml \
  --labeled-root train-0915-640 \
  --out output/unlabeled-smoke \
  --imgsz 640 \
  --every 8 \
  --max-images 20
```

### 半监督训练（会实际启动训练）

确认数据集和 GPU 后再执行：

```bash
uv run python yolo26n-seg/train_blade_yolo26_semisupervised.py \
  --labeled-root train-0915-640 \
  --unlabeled-dir unlabeled-0915-640-e1 \
  --work-dir train-seg-draft-0916 \
  --names blade \
  --model-yaml yolo26n-seg/26/yolo26-seg.yaml \
  --imgsz 640 \
  --batch 80 \
  --workers 0 \
  --epochs-stage1 200 \
  --epochs-stage2 150 \
  --pseudo-conf 0.30 \
  --min-area-ratio 0.00075 \
  --device 0
```

### 本地 MP4 叶片定位

```bash
cd blade_rtsp_system
uv run python main.py \
  --config config/config.yaml \
  --source ../data/training-data/1-rain.mp4
```

输出：`output/csv/blade_positions.csv`、`output/jsonl/blade_positions.jsonl` 和运行日志。

> 注意：`run_640_pipeline.sh` 和 `run_1024_pipeline.sh` 中的旧路径为
> `/home/mich/project/blade-clearance`，当前项目实际路径为
> `/home/mich/projects/blade-clearance`。这两个脚本运行前需要先修正路径。
