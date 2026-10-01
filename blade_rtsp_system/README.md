# blade_rtsp_system — TASK-1 YOLO26 + RTSP 实时叶片定位

实时叶片定位程序：用已有 `best.pt`（YOLO26 segmentation，imgsz=640）对 RTSP 视频流逐帧推理，
输出叶片 Mask / Raw Polygon / Simple Polygon / BBox / Center / Confidence，并恢复到原始帧坐标。

数据链（TASK-1 只做到 Blade Position Output，**不计算** blade-to-tower 物理距离）：

```
RTSP → Frame → ROI(可选) → YOLO26 best.pt(640) → Blade Mask
     → Raw Polygon → Simple Polygon → Original Coordinate → CSV/JSONL/可视化
```

## 目录结构

```
blade_rtsp_system/
├── config/config.yaml      # 配置(模型/RTSP/ROI/buffer/output)
├── models/best.pt          # 已复制的分割权重
├── src/
│   ├── rtsp_reader.py      # RTSP 读取 + 本地 MP4 回退 + 自动重连
│   ├── frame_buffer.py     # 最新帧缓冲(latest-frame strategy)
│   ├── yolo_detector.py    # YOLO 加载 + 单帧推理(模型只加载一次)
│   ├── blade_segmenter.py  # 从 Results 提取 blade instances
│   ├── mask_processor.py   # mask → raw/simple polygon
│   ├── coordinate_mapper.py# ROI/640 → 原始帧坐标恢复
│   ├── blade_locator.py    # 最终定位结果 + 叶尖候选点
│   ├── result_writer.py    # CSV + JSONL 输出
│   └── visualizer.py       # 实时可视化
├── tests/
│   ├── test_coordinate_mapper.py
│   └── test_mask_processor.py
├── main.py
└── requirements.txt
```

## 快速开始

### 1. 本地 MP4 调试（无需 RTSP 服务器）

```bash
cd blade_rtsp_system
../.venv/bin/python main.py \
  --config config/config.yaml \
  --source ../data/training-data/1-rain.mp4
```

### 2. RTSP 模式

先启动 RTSP 测试服务器（见 `../scripts/README.md`），然后：

```bash
cd blade_rtsp_system
../.venv/bin/python main.py --config config/config.yaml
```

默认 RTSP 地址：`rtsp://127.0.0.1:8554/blade`（在 config.yaml 里改）。

### 3. 单元测试

```bash
cd blade_rtsp_system
../.venv/bin/python tests/test_coordinate_mapper.py
../.venv/bin/python tests/test_mask_processor.py
```

## 关键设计（对应 TASK-1 任务书）

- **imgsz=640 只作 AI 推理尺寸**：RTSP 原始帧（可能是 2560×1440）不 resize，
  Ultralytics `result.masks.xy` 已恢复到输入图像坐标，再经 coordinate_mapper 恢复 ROI 偏移。
- **raw_polygon 永久保留**：`CHAIN_APPROX_NONE` 原始轮廓，`simple_polygon` 仅 approxPolyDP 简化用于显示。
- **latest-frame buffer**：推理慢于摄像头 FPS 时不积压，统计 received/processed/dropped。
- **断流重连**：RTSP 读帧失败 → release → wait → reconnect，`max_reconnect_attempts: 0` 无限重连。
- **多 blade 保留全部候选**：`primary_blade` 规则 = 最大 mask area，其他候选不丢弃。

## 输出

- `output/csv/blade_positions.csv`：扁平行（bbox/center/confidence/mask_area）
- `output/jsonl/blade_positions.jsonl`：每帧一行（含完整 polygon）
- `output/logs/runtime.log`：运行日志（FPS/inference time/dropped/reconnect）

## 已知限制

- TASK-1 不计算 blade-to-tower clearance（TASK-2 职责）。
- `blade_tip_candidate` 仅为候选（y 最大顶点），非真实物理叶尖。
