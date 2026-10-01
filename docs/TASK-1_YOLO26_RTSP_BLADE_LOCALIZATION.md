# TASK-1：YOLO26 + RTSP 实时叶片定位模块开发任务书

## 1. 任务目标

开发一个可运行的实时叶片定位程序，使用已有训练模型 `best.pt` 对 RTSP 视频流逐帧进行 YOLO26 分割推理，并输出每帧中的叶片位置、Mask、Polygon、Bounding Box、中心点、置信度和原始图像坐标。

本任务只完成：

```text
RTSP
→ Frame
→ ROI
→ YOLO26 best.pt
→ Blade Mask
→ Blade Polygon
→ Original Coordinate
→ Blade Position
```

本 TASK-1 **不进行最终 blade-to-tower clearance 物理距离计算**。

---

# 2. 已知输入条件

## 2.1 模型

已有训练完成的 YOLO26 segmentation 模型：

```text
best.pt
```

模型推理尺寸：

```text
imgsz = 640
```

要求程序从配置文件读取模型路径，不允许在核心代码中写死绝对路径。

---

## 2.2 视频源

视频通过 RTSP 输入。

示例：

```text
rtsp://username:password@camera-ip:554/Streaming/Channels/101
```

RTSP 地址必须从配置文件读取。

程序必须支持：

- RTSP 实时读取；
- 每帧处理；
- RTSP 短时中断后的自动重连；
- 保留原始视频帧分辨率；
- 记录 frame_id；
- 记录 timestamp。

---

# 3. TASK-1 核心原则

## 3.1 640 只作为 AI 推理尺寸

YOLO26 输入尺寸：

```text
640 × 640
```

不代表 RTSP 原始视频必须是 640×640。

原始 RTSP 视频可能是：

```text
1920×1080
2560×1440
3840×2160
```

必须保留原始图像。

最终输出的叶片位置必须转换到：

```text
RTSP 原始图像坐标系
```

禁止直接把 640×640 推理坐标作为最终定位结果。

---

## 3.2 YOLO26 负责“找叶片”

本阶段 YOLO26 负责：

- blade detection；
- blade segmentation；
- blade mask；
- blade polygon；
- bounding box；
- confidence；
- blade center；
- 原始坐标恢复。

不在本任务内实现：

- 塔架精确边缘；
- blade-to-tower 最短距离；
- 亚像素边缘；
- 摄像机物理标定；
- 米制 clearance；
- 告警策略。

---

## 3.3 必须逐帧处理

默认：

```text
vid_stride = 1
```

即每一帧都进入分析链。

如果实时性能不足，不允许默认修改为跳帧处理。

应首先：

1. 统计实际 FPS；
2. 统计 YOLO 推理耗时；
3. 记录是否存在掉帧；
4. 再决定是否优化。

---

# 4. 软件目录要求

建议创建以下目录结构：

```text
blade_rtsp_system/
│
├── config/
│   └── config.yaml
│
├── models/
│   └── best.pt
│
├── src/
│   ├── __init__.py
│   ├── rtsp_reader.py
│   ├── frame_buffer.py
│   ├── yolo_detector.py
│   ├── blade_segmenter.py
│   ├── mask_processor.py
│   ├── coordinate_mapper.py
│   ├── blade_locator.py
│   ├── result_writer.py
│   └── visualizer.py
│
├── output/
│   ├── csv/
│   ├── jsonl/
│   ├── video/
│   └── logs/
│
├── tests/
│   ├── test_coordinate_mapper.py
│   ├── test_mask_processor.py
│   └── test_rtsp_reader.py
│
├── requirements.txt
├── README.md
└── main.py
```

如果已有项目目录，则在现有结构下实现，不要求强制重建整个工程。

---

# 5. 配置文件

创建：

```text
config/config.yaml
```

至少包含：

```yaml
model:
  path: models/best.pt
  imgsz: 640
  device: 0
  conf: 0.50
  retina_masks: true

rtsp:
  url: "rtsp://username:password@camera-ip:554/Streaming/Channels/101"
  reconnect_interval_sec: 2
  max_reconnect_attempts: 0

processing:
  use_roi: false
  roi:
    x1: 0
    y1: 0
    x2: 0
    y2: 0

  vid_stride: 1

buffer:
  strategy: latest
  max_size: 1

output:
  save_csv: true
  save_jsonl: true
  save_video: false
  show_window: true

debug:
  verbose: false
```

说明：

```text
max_reconnect_attempts: 0
```

表示持续尝试重连。

---

# 6. 模块开发要求

## 6.1 `rtsp_reader.py`

职责：

```text
RTSP
→ Open
→ Read Frame
→ Timestamp
→ Reconnect
```

至少实现：

```python
class RTSPReader:
    def __init__(self, url, reconnect_interval_sec=2):
        ...

    def open(self):
        ...

    def read(self):
        ...

    def reconnect(self):
        ...

    def release(self):
        ...
```

`read()` 建议返回：

```python
success, frame, timestamp
```

要求：

- RTSP 首次连接失败不能导致程序崩溃；
- `cap.read()` 失败时进入重连逻辑；
- 日志记录断流时间；
- 日志记录恢复时间；
- 保留原始 frame；
- 禁止在 reader 中 resize 到 640。

---

# 7. Frame Buffer

实现：

```text
src/frame_buffer.py
```

采用：

```text
latest-frame strategy
```

默认 buffer：

```text
max_size = 1
```

设计目标：

```text
Reader Thread
    ↓
Latest Frame
    ↓
Inference Thread
```

不允许由于 YOLO 推理速度低于摄像头 FPS 而无限积压帧。

如果新帧到来而上一帧尚未被处理：

```text
旧的未处理帧允许被新的最新帧覆盖
```

但是必须统计：

```text
received_frames
processed_frames
dropped_frames
```

最终日志中输出。

---

# 8. YOLO26 推理模块

实现：

```text
src/yolo_detector.py
```

加载：

```python
from ultralytics import YOLO
```

模型：

```python
model = YOLO(model_path)
```

推理：

```python
results = model.predict(
    source=roi_or_frame,
    imgsz=640,
    conf=conf_threshold,
    device=device,
    retina_masks=True,
    verbose=False
)
```

要求：

- 模型只加载一次；
- 禁止每帧重新创建 YOLO 对象；
- 记录 inference time；
- 若 GPU 不可用，明确报错或按配置切换 CPU；
- 不允许静默切换设备而不记录日志。

---

# 9. Blade 提取

实现：

```text
src/blade_segmenter.py
```

目标类别：

```text
blade
```

需要从模型结果中得到：

```text
class_id
confidence
bbox
mask
polygon
```

如果一帧中没有 blade：

```text
blade_detected = false
```

程序继续运行，不应退出。

如果一帧中出现多个 blade instance：

必须保留全部候选：

```text
blade_instances[]
```

不要只静默选 confidence 最大者。

可以额外输出：

```text
primary_blade
```

但选择规则必须明确写在代码中。

建议默认：

```text
primary_blade = 最大 mask area 的 blade instance
```

同时保留其他 blade。

---

# 10. Mask 处理

实现：

```text
src/mask_processor.py
```

每个 blade 至少保存：

```text
mask
raw_contour
raw_polygon
simple_polygon
```

Contour 提取：

```python
cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)
```

注意：

```text
CHAIN_APPROX_NONE
```

用于保留原始轮廓点。

禁止直接只保留：

```text
approxPolyDP
```

结果。

必须同时维护：

```text
raw_polygon
simple_polygon
```

---

# 11. Polygon 简化规则

允许增加：

```python
simple_polygon = simplify_polygon(raw_polygon)
```

内部可以使用：

```python
cv2.approxPolyDP()
```

但：

```text
raw_polygon
```

必须永久保留。

用途定义：

```text
raw_polygon
→ 后续精确边缘 / clearance

simple_polygon
→ 实时显示 / tracking / 快速几何处理
```

禁止：

```python
raw_polygon = simplify_polygon(raw_polygon)
```

覆盖原始轮廓。

---

# 12. ROI 支持

如果：

```yaml
processing:
  use_roi: true
```

则：

```python
roi = frame[y1:y2, x1:x2]
```

需要保存：

```text
roi_offset_x = x1
roi_offset_y = y1
```

YOLO 输出最终必须映射回：

```text
Full Original Frame
```

ROI 仅允许提高：

- 推理速度；
- blade 像素占比；
- 减少背景误检。

ROI 不得改变最终全局坐标定义。

---

# 13. 坐标系统

系统必须明确区分以下坐标系：

```text
Coordinate System A
RTSP Original Frame

Coordinate System B
ROI Local Frame

Coordinate System C
YOLO Internal 640 Frame
```

最终标准输出必须统一为：

```text
Coordinate System A
```

即：

```text
RTSP Original Frame Coordinate
```

---

# 14. 坐标恢复模块

实现：

```text
src/coordinate_mapper.py
```

至少提供：

```python
def roi_to_global(points, roi_offset_x, roi_offset_y):
    ...
```

如果直接使用 Ultralytics 已恢复到输入图像尺寸的：

```text
result.masks.xy
```

必须首先确认这些 polygon 坐标对应：

```text
输入给 model.predict() 的 image
```

如果输入的是 ROI，则：

```text
result.masks.xy
```

依然是 ROI 坐标。

因此仍需执行：

```text
ROI coordinate
→ Original Frame coordinate
```

---

# 15. Blade 定位模块

实现：

```text
src/blade_locator.py
```

每个 blade instance 至少输出：

```text
bbox
center
raw_polygon
simple_polygon
confidence
mask_area
```

Center：

$$
x_c = \frac{x_1+x_2}{2}
$$

$$
y_c = \frac{y_1+y_2}{2}
$$

所有坐标必须为：

```text
Original Frame Coordinate
```

---

# 16. Blade Tip

TASK-1 中允许实现一个：

```text
blade_tip_candidate
```

但必须标记为：

```text
candidate
```

而不是认为已经得到真实物理叶尖。

如果实现，要求算法规则明确，例如：

```text
根据 blade polygon 几何主轴
+
距离根部候选区域最远点
```

但 TASK-1 不把 blade tip 精度作为最终验收条件。

---

# 17. 每帧标准输出

建议创建统一结果对象：

```python
{
    "frame_id": 12034,
    "timestamp": 1726886401.245,

    "frame_width": 1920,
    "frame_height": 1080,

    "blade_detected": True,
    "blade_count": 1,

    "blades": [
        {
            "instance_id": 0,
            "class_id": 0,
            "class_name": "blade",
            "confidence": 0.934,

            "bbox": [
                632,
                118,
                1054,
                902
            ],

            "center": [
                843.0,
                510.0
            ],

            "mask_area": 125832,

            "raw_polygon": [
                [701, 125],
                [702, 126]
            ],

            "simple_polygon": [
                [701, 125],
                [815, 308]
            ]
        }
    ]
}
```

---

# 18. CSV 输出

创建：

```text
output/csv/blade_positions.csv
```

至少包含：

```text
timestamp
frame_id
frame_width
frame_height
blade_detected
blade_count
instance_id
confidence
bbox_x1
bbox_y1
bbox_x2
bbox_y2
center_x
center_y
mask_area
```

Polygon 不要求完整存入主 CSV。

---

# 19. JSONL 输出

Polygon 推荐存入：

```text
output/jsonl/blade_positions.jsonl
```

每一行对应一帧：

```json
{"frame_id": 1, "timestamp": 0.0, "blades": [...]}
```

优势：

- 支持变长 polygon；
- 方便后续 clearance 模块读取；
- 方便离线 replay；
- 方便 debug。

---

# 20. 可视化模块

实现：

```text
src/visualizer.py
```

实时窗口至少显示：

```text
Frame ID
FPS
Blade Confidence
Blade Bounding Box
Blade Mask
Blade Polygon
```

建议：

```text
raw polygon 不全部绘制
```

避免影响实时性能。

可视化可以使用：

```text
simple_polygon
```

---

# 21. FPS 与性能统计

程序必须实时统计：

```text
RTSP receive FPS
YOLO inference FPS
Total processing FPS
Average inference time
Maximum inference time
Dropped frames
```

每隔一定时间，例如：

```text
5 秒
```

打印一次统计。

程序退出时打印最终 summary。

---

# 22. 多线程设计

推荐：

```text
Thread 1
RTSP Reader

Thread 2
YOLO Inference

Thread 3
Output / Visualization
```

最低要求：

```text
RTSP Reader
```

不能长期被 YOLO inference 阻塞。

如果第一版为了简化采用单线程，需要：

1. 代码结构允许后续拆成多线程；
2. 明确记录实际 RTSP FPS 与 processing FPS；
3. 不得隐藏积压问题。

---

# 23. 异常处理

必须处理：

## 23.1 RTSP 断流

行为：

```text
log
→ release
→ wait
→ reconnect
```

程序不能直接退出。

---

## 23.2 YOLO 未检测到 blade

行为：

```text
blade_detected = false
```

继续下一帧。

---

## 23.3 Model load fail

立即终止并给出明确错误：

```text
model file not found
model load failed
CUDA error
```

---

## 23.4 空 Mask

如果：

```text
box exists
mask is None
```

必须记录 warning。

不能生成虚假的 polygon。

---

# 24. 日志

创建：

```text
output/logs/runtime.log
```

至少记录：

```text
程序启动
模型路径
模型加载状态
CUDA device
RTSP 地址（密码应脱敏）
RTSP 连接状态
RTSP 断流
RTSP 重连
原始视频分辨率
原始 FPS
YOLO imgsz
运行 FPS
inference time
dropped frame 数
异常
程序退出
```

---

# 25. 安全要求

RTSP URL 中可能包含：

```text
username
password
```

日志中禁止明文输出完整 RTSP URL。

例如：

原始：

```text
rtsp://admin:password123@192.168.1.100:554/...
```

日志应写：

```text
rtsp://admin:***@192.168.1.100:554/...
```

---

# 26. 单元测试

至少实现：

```text
test_coordinate_mapper.py
test_mask_processor.py
```

---

## 26.1 Coordinate Mapper Test

构造：

```text
ROI offset = (100, 200)
ROI point = (50, 60)
```

输出必须：

```text
Global point = (150, 260)
```

测试：

```text
bbox
center
polygon
```

全部坐标恢复。

---

## 26.2 Mask Processor Test

创建简单二值 mask。

验证：

```text
contour exists
raw_polygon node count > 0
simple_polygon node count > 0
simple_polygon node count <= raw_polygon node count
```

并确认：

```text
raw_polygon
```

没有因为 simplify 操作被修改。

---

# 27. 本地视频调试模式

除了 RTSP，程序必须支持：

```text
--source local.mp4
```

用于离线调试。

例如：

```bash
python main.py \
  --config config/config.yaml \
  --source test.mp4
```

RTSP：

```bash
python main.py \
  --config config/config.yaml
```

这样即使现场摄像机不可用，仍然可以完整测试：

```text
Frame
→ YOLO
→ Mask
→ Polygon
→ Coordinate
→ Output
```

---

# 28. main.py 工作流

建议：

```python
load_config()

load_model()

open_source()

while running:

    frame = get_latest_frame()

    if frame is None:
        continue

    result = run_yolo(frame)

    blades = extract_blades(result)

    blades = map_to_original_coordinates(blades)

    write_result(blades)

    visualize(blades)

shutdown()
```

要求代码：

- 函数化；
- 模块化；
- 不允许将全部功能堆在 `main.py`；
- 不允许一个函数承担完整流程。

---

# 29. TASK-1 验收标准

## Gate 1：模型加载

必须确认：

```text
best.pt loaded successfully
device correctly detected
imgsz = 640
```

---

## Gate 2：本地 MP4 测试

必须先用本地视频证明：

```text
Video
→ Frame
→ YOLO
→ Mask
→ Polygon
```

流程正确。

输出：

```text
CSV
JSONL
annotated video/window
```

---

## Gate 3：RTSP 接入

必须证明：

```text
RTSP connected
frames received continuously
```

至少连续运行：

```text
10 min
```

过程中不能因普通读帧失败直接退出。

---

## Gate 4：Blade Detection

检查实际 RTSP 帧。

至少随机抽查：

```text
100 frames
```

确认：

```text
blade_detected
bbox
mask
polygon
```

数据结构正确。

TASK-1 暂不定义模型本身的 detection accuracy 门限。

---

## Gate 5：坐标正确性

随机选取：

```text
≥ 10 frames
```

将 Polygon 绘制回：

```text
Original RTSP Frame
```

要求：

```text
Polygon 与 blade 可视位置一致
```

如果存在 ROI：

必须验证：

```text
ROI local coordinates
→ global coordinates
```

恢复正确。

---

## Gate 6：持续运行

RTSP 模式连续运行：

```text
30 min
```

记录：

```text
received_frames
processed_frames
dropped_frames
average_fps
average_inference_ms
reconnect_count
```

不得出现：

```text
memory持续增长
线程死锁
RTSP永久假死
模型重复加载
```

---

# 30. TASK-1 最终交付物

必须交付：

```text
1. 完整 Python 源代码

2. config/config.yaml

3. requirements.txt

4. README.md

5. output/csv/blade_positions.csv 示例

6. output/jsonl/blade_positions.jsonl 示例

7. 运行日志

8. 本地 MP4 测试结果

9. RTSP 测试结果

10. TASK1_EXECUTION_REPORT.md
```

---

# 31. TASK1_EXECUTION_REPORT.md 必须包含

报告至少包含：

```text
1. 开发环境

2. Python 版本

3. Ultralytics 版本

4. OpenCV 版本

5. GPU 型号

6. CUDA 状态

7. best.pt 路径

8. RTSP 原始分辨率

9. RTSP FPS

10. imgsz

11. 平均 inference time

12. processing FPS

13. dropped frame 数

14. RTSP reconnect 次数

15. 叶片定位截图

16. Polygon 原图坐标验证

17. 已知问题

18. TASK-2 建议
```

---

# 32. 编程智能体执行要求

编程智能体必须遵守：

### Rule 1

不得擅自重新训练：

```text
best.pt
```

TASK-1 只使用已有模型。

---

### Rule 2

不得擅自修改：

```text
imgsz = 640
```

如果认为需要修改，只能写入报告建议。

---

### Rule 3

不得进入最终：

```text
blade-to-tower clearance
```

物理距离计算。

TASK-1 截止于：

```text
Blade Original-Frame Localization
```

---

### Rule 4

不得只输出 Bounding Box。

必须输出：

```text
Mask
+
Raw Polygon
+
Simple Polygon
```

---

### Rule 5

不得用：

```text
simple_polygon
```

替代：

```text
raw_polygon
```

---

### Rule 6

所有最终叶片坐标必须明确属于：

```text
Original RTSP Frame Coordinate System
```

---

### Rule 7

每完成一个 Gate：

```text
运行
→ 检查
→ 保存证据
```

不得只写代码而不运行验证。

---

### Rule 8

如果某个 Gate FAIL：

必须先：

```text
定位原因
→ 修复
→ 重新运行
```

再进入下一 Gate。

---

### Rule 9

不得因为实时 FPS 不够，未经记录直接：

```text
跳帧
降低分辨率
关闭 mask
```

性能问题必须进入：

```text
TASK1_EXECUTION_REPORT.md
```

---

### Rule 10

所有异常必须显式记录。

禁止：

```python
try:
    ...
except:
    pass
```

---

# 33. 完成判定

只有以下链条被实际验证后，TASK-1 才算完成：

```text
RTSP Camera
     ↓
Original Frame
     ↓
ROI
     ↓
YOLO26 best.pt
imgsz=640
     ↓
Blade Detection
     ↓
Blade Mask
     ↓
Raw Polygon
     ↓
Simple Polygon
     ↓
Original Coordinate Restoration
     ↓
Blade Position Output
     ↓
CSV / JSONL / Visualization
```

最终必须能够回答：

```text
对于任意一个 RTSP frame：

1. 是否检测到 blade？
2. confidence 是多少？
3. blade bbox 在哪里？
4. blade center 在哪里？
5. blade mask 是什么？
6. blade raw polygon 是什么？
7. blade polygon 在原始 RTSP 图像中的坐标是什么？
8. 当前处理 FPS 是多少？
9. 是否发生掉帧？
```

如果这些问题都可以通过程序输出和日志明确回答，则：

```text
TASK-1 PASS
```

否则：

```text
TASK-1 FAIL
```

---

# 34. TASK-1 后续接口

TASK-1 输出将作为 TASK-2 输入。

TASK-2 预期：

```text
Blade Polygon
+
Tower Region
        ↓
Original Resolution Edge
        ↓
Blade Edge / Tower Edge
        ↓
Nearest Point Search
        ↓
Cpx(t)
```

因此 TASK-1 最重要的接口资产是：

```text
Original Frame
frame_id
timestamp
raw blade polygon
blade mask
```

这些数据必须设计成可被后续模块直接读取，而不是只用于实时显示。
