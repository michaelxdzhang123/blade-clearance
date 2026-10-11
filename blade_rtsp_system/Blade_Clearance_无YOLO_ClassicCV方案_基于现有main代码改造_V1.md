# Blade Clearance 无 YOLO 方案设计报告
## —— 基于现有 `main.py` 的最小改造版：Classic CV + 时序几何 + 亚像素边缘 + Geometry

> 目标：在尽量保留当前 RTSP、多线程、ROI、CSV/JSONL、可视化和工程框架的前提下，移除 YOLO26 推理链，改为完全基于 Classic CV 的叶片检测与净空测量方案。  
> 适用场景：固定机舱摄像机；画面中主要移动目标只有叶片；塔筒基本静止；叶片轨迹连续且具有周期性；最终目标是 Blade–Tower Clearance，而不是通用目标检测。  
> 核心原则：**AI 不是测量核心。只要 Classic CV 能稳定产生正确的 Blade Safe ROI，并在原始分辨率上得到与 YOLO 路线相同的 Blade/Tower 边界，则最终 `Cpx(t)` 可以与 YOLO 方案一致。**

---

# 1. 为什么这个项目可以不用 YOLO

你的场景和普通目标检测问题差别很大。

普通 YOLO 解决的是：

```text
一张图里有很多未知类别
+
目标数量未知
+
位置未知
+
背景未知
+
相机可能移动
```

而你的 Blade Clearance 场景是：

```text
固定相机
+
固定塔筒
+
主要移动目标只有一个：Blade
+
Blade 运动连续
+
Blade Pass 周期可预测
+
Blade ROI 基本已知
+
Tower ROI 基本已知
+
最终任务是测量 Clearance
```

因此真正需要解决的问题不是：

```text
“这是什么物体？”
```

而是：

```text
“唯一的运动叶片现在在哪里？”
“它朝向塔筒的真实边界在哪里？”
“Blade 与 Tower 的最小距离是多少？”
```

这三个问题可以由：

```text
Motion
+
Temporal Tracking
+
Geometry
+
Original-resolution Edge
```

完成，不一定需要深度学习。

---

# 2. 对现有 YOLO `main.py` 的理解

当前主流程本质是：

```text
RTSP
  ↓
Frame
  ↓
ROI
  ↓
YOLO26 best.pt @ 640
  ↓
Blade Mask
  ↓
blade_segmenter
  ↓
blade_locator
  ↓
ROI → Full-frame Coordinate
  ↓
CSV / JSONL / Visualization
```

现有代码的工程骨架已经比较完整，主要包括：

```text
RTSPReader
FrameBuffer
Reader Thread
Main Processing Thread
ROI
ResultWriter
Visualizer
Buffered Logging
Reconnect
Runtime Statistics
```

这些都不需要推翻。

真正与 YOLO 强绑定的部分主要是：

```text
YOLODetector
blade_segmenter.extract_blades()
blade_locator.locate_blades()
```

无 YOLO 版本应尽量只替换这一段业务核心，而不是重新写整个实时系统。

---

# 3. 推荐的无 YOLO 总体架构

将当前：

```text
RTSP
↓
ROI
↓
YOLO26
↓
Mask
↓
Polygon
```

替换为：

```text
RTSP
↓
Full-resolution Frame
↓
Frame Registration
↓
Tracking ROI
↓
Motion Detection
↓
Connected Components
↓
Blade Candidate Scoring
↓
Temporal Tracking / Blade Pass FSM
↓
Safe Blade ROI
↓
Original-resolution Current Frame
↓
Blade Contour / Local Edge
↓
PCA + Dynamic Width
↓
Tower-facing Blade Boundary
↓
EdgesSubPix
↓
Blade Subpixel Boundary
```

Tower 分支：

```text
Tower Reference
↓
Tower Local ROI
↓
Local Edge
↓
EdgesSubPix
↓
Tower Subpixel Boundary
```

最终：

```text
Blade Boundary
+
Tower Boundary
↓
Geometry
↓
Cpx(t)
↓
Pass Aggregation
↓
Cmin_px
```

---

# 4. 无 YOLO 方案中各模块的职责

## 4.1 RTSPReader

保留现有模块。

职责不变：

```text
RTSP
↓
read()
↓
frame + timestamp + frame_id
```

保留：

```text
自动重连
URL 脱敏
frame_id
timestamp
```

---

## 4.2 FrameBuffer：这里需要重点修改

现有代码使用：

```text
max_size = 1
latest-frame
```

也就是新帧覆盖旧帧。

这种策略适合：

```text
实时预览
普通检测
YOLO 推理慢于视频时保持低延迟
```

但是对 Clearance Measurement 有风险：

> **最小净空可能恰好发生在被丢弃的那一帧。**

因此建议把系统分成两种状态：

```text
IDLE / SEARCH
→ latest-frame buffer 可以接受

MEASURING
→ 不允许主动丢帧
```

推荐增加：

```text
Measurement Ring Buffer
```

或：

```text
Bounded Queue
```

在 Blade Pass 测量窗口内要求连续处理完整帧。

---

# 5. Frame Registration：无 YOLO 方案的第一项增强

Classic CV 最大风险之一不是 Blade，而是：

```text
Camera vibration
Nacelle vibration
small image shift
```

如果直接做：

$$
|I_t-I_{t-1}|
$$

相机抖动会让：

```text
Tower
Sky boundary
Nacelle structure
```

全部变成“运动目标”。

因此推荐：

```text
Current Frame
↓
Tower Reference Band
↓
ECC / Phase Correlation / Edge Alignment
↓
Registered Frame
↓
Motion Detection
```

优先使用 Tower 附近的静态结构作为 registration anchor。

输出：

```text
dx
dy
rotation(optional)
registration_quality
```

如果配准失败：

```text
FRAME_REGISTRATION_FAIL
```

该帧不进入正式测量。

---

# 6. Motion Detection：替代 YOLO 的第一核心

推荐同时使用两种运动证据。

## A. 相邻帧差

$$
M_{fd}(t)
=
|I_t-I_{t-1}|
$$

优点：

```text
对慢速光照变化不敏感
非常适合快速移动 Blade
```

## B. 静态背景差

预先在无 Blade 或 IDLE 状态建立：

$$
B_{bg}
$$

然后：

$$
M_{bg}(t)
=
|I_t-B_{bg}|
$$

优点：

```text
Blade 整体区域更完整
```

缺点：

```text
受云层、光照变化影响较大
```

推荐组合：

```text
frame-difference
→ trigger

background-difference
→ localization
```

---

# 7. Dual Threshold

推荐使用双阈值：

```text
T_trigger < T_localize
```

低阈值用于：

```text
Blade 是否开始进入
```

高阈值用于：

```text
提取可靠运动 component
```

流程：

```text
absdiff
↓
threshold
↓
morphology open/close
↓
connected components
```

不要用一个阈值同时承担 trigger、localization 和 measurement。

---

# 8. Connected Components：因为只有一个主要移动目标

Blade Pass 窗口中，候选通常只有少数几个 motion components。

对每个 component 计算：

```text
area
bbox
centroid
aspect
PCA angle
elongation
position
distance to previous state
```

只保留 Top-K，例如：

```text
K = 3
```

而不是直接假定最大 component 就是 Blade。

---

# 9. Blade Candidate Score

推荐：

$$
S_B
=
w_mS_{motion}
+
w_aS_{area}
+
w_eS_{elongation}
+
w_pS_{position}
+
w_tS_{temporal}
+
w_gS_{geometry}
$$

其中：

- `S_motion`：运动证据；
- `S_area`：面积软约束；
- `S_elongation`：PCA 细长程度；
- `S_position`：是否在允许区域；
- `S_temporal`：与上一帧状态是否连续；
- `S_geometry`：主轴方向是否符合已知轨迹。

---

# 10. Pitch-aware：不能使用固定 Blade Width

无 YOLO 方案尤其不能写：

```text
blade_width = constant
```

Pitch 改变后，Blade apparent width 会变化。

应采用：

$$
W=W(s,t)
$$

即沿 Blade 主轴不同 station 动态计算局部宽度。

流程：

```text
Blade Contour
↓
PCA
↓
Long Axis
↓
多个 station
↓
局部法向
↓
两侧 Boundary
↓
Width Profile W(s,t)
```

保存：

```text
median_width_px
width_p10_px
width_p90_px
width_profile
```

Width 用于 QC、candidate score、dynamic edge band，而不是硬 reject。

---

# 11. Temporal Tracking：替代“每帧重新识别”

YOLO 方案是：

```text
每一帧重新回答：
“这里有没有 Blade？”
```

无 YOLO 方案应该充分利用视频时序：

```text
Frame t
↓
State
↓
Frame t+1
↓
Prediction + Correction
```

建议保存：

```text
center
velocity
angle
width
area
trajectory
quality
```

V1 可以先用简单 temporal continuity，不必马上引入复杂 Kalman Filter。

---

# 12. Blade Pass FSM

建议正式实现：

```text
IDLE
↓
WAITING_FOR_BLADE
↓
BLADE_ENTERING
↓
TRACKING
↓
MEASURING
↓
CLOSEST_APPROACH
↓
BLADE_LEAVING
↓
PASS_COMPLETE
```

一旦进入：

```text
TRACKING
```

系统已经知道当前连续运动结构属于当前 Blade Pass。

这就减少了“每帧重新语义识别”的必要性。

---

# 13. RPM / 周期性是额外强约束

三叶片机组近似：

$$
T_{pass}
=
rac{20}{RPM}
$$

因此：

```text
前一次 Blade Pass
+
当前 RPM
```

可以预测下一次 Blade 出现时间。

这能显著降低：

```text
cloud motion
shadow
rain
```

造成的误触发。

如果有 SCADA RPM，优先直接使用；否则可以从 Blade Pass 周期估计。

---

# 14. Motion Mask 不能直接作为最终 Blade Edge

相邻帧差通常包含：

```text
current blade
+
previous blade ghost
```

因此 Motion Mask 只能用于：

```text
Candidate
Safe ROI
Tracking
```

不能直接用于：

```text
Cpx
```

最终 Blade Edge 必须回到：

```text
当前帧原始图像
```

重新提取。

---

# 15. Safe Blade ROI

由运动 component 得到 coarse bbox 后：

```text
expand margin
```

形成 Safe ROI。

Safe ROI 的目标不是像 Mask 一样精确，而是：

> Tower-facing Blade measurement boundary 必须完整落入 Safe ROI。

这进一步降低了 Classic CV detector 的难度。

---

# 16. Original-resolution Blade Contour

在 Safe ROI 中必须使用：

```text
original current image
```

而不是：

```text
frame difference
background difference
```

去提取真实 Blade。

推荐结合：

```text
local contrast
gradient
motion prior
region mask
continuity
```

得到：

$$
B_{raw}(t)
$$

---

# 17. PCA 动态方向

对粗 contour 计算 PCA：

$$
	heta_{blade}(t)
=
\operatorname{atan2}(v_y,v_x)
$$

得到 Blade 长轴单位向量：

$$
\mathbf e_1=(v_x,v_y)
$$

法向：

$$
\mathbf e_2=(-v_y,v_x)
$$

支持：

```text
dynamic width
tower-facing edge
local edge search
```

---

# 18. 动态选择 Tower-facing Boundary

不能固定：

```text
left edge
```

或：

```text
right edge
```

因为 Pitch、Blade rotation 和 Camera projection 都会改变可见边。

推荐：

```text
Full Blade Contour
↓
Tower Reference
↓
计算每个 contour point 到 Tower 的距离
↓
选择 near-side points
↓
continuity filter
↓
Tower-facing Boundary
```

对：

$$
p_i\in B(t)
$$

计算：

$$
d_i=d(p_i,T)
$$

定义：

$$
d_{min}=\min_i d_i
$$

在：

$$
d_i<d_{min}+d_{band}
$$

范围内选择连续分支。

---

# 19. EdgesSubPix

粗边界只负责告诉系统真实边缘大概在哪里。

然后仅在局部 narrow band 中做亚像素精修。

推荐 band 随动态 width 调整：

$$
b(t)
=
clip(
kW_{med}(t),
b_{min},
b_{max}
)
$$

最终得到：

$$
B_{subpixel}(t)
$$

---

# 20. Tower 完全不需要 YOLO

Tower 是固定结构。

建议：

```text
Commissioning
↓
Tower Reference
```

运行时：

```text
Tower Reference
↓
Local narrow band
↓
gradient / orientation
↓
line / spline refinement
↓
EdgesSubPix
↓
T_subpixel(t)
```

Tower Reference 必须是有限边界，并具有：

```text
valid y-range
measurement side
```

避免：

```text
TOWER_EDGE_SWITCH
```

---

# 21. Geometry

最终：

$$
C_{px}(t)
=
\min_{\mathbf p\in B_{subpixel}(t),
\mathbf q\in T_{subpixel}(t)}
\|\mathbf p-\mathbf q\|_2
$$

工程上建议计算：

```text
polyline segment-to-segment distance
```

而不是只比较 contour vertices。

同时保存最近点：

```text
blade_nearest_x
blade_nearest_y
tower_nearest_x
tower_nearest_y
```

---

# 22. Pass Aggregation

单帧输出：

```text
Cpx(t)
```

整个 Blade Pass：

$$
C_{min,sampled}
=
\min_{t\in pass,\ valid(t)}
C_{px}(t)
$$

V1 建议明确称：

```text
Cmin_sampled_px
```

因为 25 fps 只得到离散采样最小值。

---

# 23. 与当前 `main.py` 的代码级映射

## 保留

现有以下部分建议继续复用：

```text
load_config()
resolve_path()

BufferedFileHandler
setup_logging()

RTSPReader
FrameBuffer
ResultWriter
Visualizer

reader_loop()
reconnect
SIGTERM
runtime statistics
```

---

# 24. 删除 / 替换的模块

当前：

```python
from src import blade_locator, blade_segmenter
from src.yolo_detector import YOLODetector
```

无 YOLO 版建议替换为：

```python
from src.frame_registrar import FrameRegistrar
from src.motion_detector import MotionDetector
from src.blade_candidate import BladeCandidateExtractor
from src.blade_tracker import BladeTracker
from src.blade_pass_fsm import BladePassFSM
from src.blade_geometry import BladeGeometryExtractor
from src.subpixel_edge import SubpixelEdgeRefiner
from src.tower_reference import TowerReference
from src.clearance_geometry import ClearanceGeometry
```

---

# 25. 删除模型加载阶段

现有：

```text
读取 model config
检查 best.pt
YOLODetector(...)
detector.load()
```

全部删除。

替换为：

```text
CV Configuration
↓
FrameRegistrar
MotionDetector
BladeTracker
BladePassFSM
TowerReference
SubpixelEdgeRefiner
ClearanceGeometry
```

这些模块：

```text
不需要权重文件
不需要 GPU
不需要训练
```

---

# 26. 主循环替换关系

当前核心：

```python
result, inf_ms = detector.predict(infer_input)

detection = blade_segmenter.extract_blades(
    result,
    class_name="blade"
)

located = blade_locator.locate_blades(...)
```

替换为概念上的：

```python
registered = registrar.align(frame)

motion = motion_detector.detect(
    registered
)

candidates = candidate_extractor.extract(
    motion,
    registered
)

tracked = tracker.update(
    candidates
)

state = pass_fsm.update(
    tracked
)

blade = blade_geometry.extract(
    registered,
    tracked,
    state
)

blade_edge = subpixel_refiner.refine_blade(
    registered,
    blade
)

tower_edge = tower_reference.refine(
    registered
)

clearance = geometry.compute(
    blade_edge,
    tower_edge
)
```

从工程结构上看，仍然和现有 `main.py` 非常接近：

```text
取帧
↓
处理
↓
得到 result / located
↓
Visualizer
↓
ResultWriter
```

只是把 YOLO inference 换成 CV measurement pipeline。

---

# 27. 推荐的新主流程

```python
while running.is_set():

    frame, ts, fid = buffer.get_latest()

    if frame is None:
        continue

    reg = registrar.align(frame)

    motion = motion_detector.detect(reg.frame)

    candidates = candidate_extractor.extract(
        reg.frame,
        motion
    )

    track = tracker.update(candidates)

    pass_state = pass_fsm.update(track)

    if not pass_state.should_measure:
        write_tracking_result(...)
        continue

    blade_geom = blade_geometry.extract(
        reg.frame,
        track
    )

    blade_edge = subpixel_refiner.refine_blade(
        reg.frame,
        blade_geom
    )

    tower_edge = tower_reference.refine(
        reg.frame
    )

    clearance = geometry.compute(
        blade_edge,
        tower_edge
    )

    writer.write(...)
```

---

# 28. build_frame_result 应升级

当前主要字段：

```text
blade_detected
blade_count
blades
loop_ms
```

无 YOLO 版建议：

```text
frame_id
timestamp

pass_id
pass_state

motion_triggered
candidate_count

blade_found
blade_tracking_quality

blade_center_x
blade_center_y
blade_angle_deg

median_width_px
width_p10_px
width_p90_px

blade_boundary_valid
tower_boundary_valid

blade_nearest_x_px
blade_nearest_y_px

tower_nearest_x_px
tower_nearest_y_px

cpx_euclidean_px

registration_quality
edge_quality

valid
failure_code

loop_ms
```

---

# 29. 与当前 Visualizer 的兼容

现有 Visualizer 可以保留。

把原来的：

```text
mask
polygon
bbox
```

逐步换成：

```text
Motion Candidate
Tracking ROI
Blade Contour
PCA Axis
Tower-facing Boundary
Tower Reference
Nearest Pair
Cpx
Pass State
Quality
```

这样的可视化更接近真正的 measurement evidence。

---

# 30. 配置文件如何改

当前：

```yaml
model:
  path: models/best.pt
  imgsz: 640
  conf: 0.5
```

无 YOLO 后删除。

增加：

```yaml
motion:
  use_background: true
  use_frame_diff: true
  trigger_threshold: 15
  localize_threshold: 30
  min_component_area: 800
  top_k: 3

registration:
  enabled: true
  method: tower_ecc
  min_quality: 0.8

tracking:
  max_center_jump_px: 120
  max_angle_jump_deg: 25

blade_geometry:
  dynamic_width: true
  width_soft_constraint: true
  pca_enabled: true

edge:
  local_band_min_px: 8
  local_band_max_px: 25
  width_band_ratio: 0.25

tower:
  reference_file: config/tower_reference.yaml
  local_band_px: 20

clearance:
  distance_mode: polyline_segment
```

所有数值都只是初始工程参数，必须由真实数据验证。

---

# 31. Background Model

建议只在：

```text
IDLE
+
no blade
+
high-confidence static
```

时更新背景：

$$
B_t
=
(1-lpha)B_{t-1}
+
lpha I_t
$$

或者在 commissioning 阶段生成：

```text
bg_idle_median.npy
```

Blade Pass 期间冻结 background update，防止 Blade 被吸收进背景。

---

# 32. Hard Negative 如何处理

YOLO 依赖训练 hard negative。

无 YOLO 则依靠：

```text
motion
+
position
+
shape
+
temporal continuity
+
periodicity
+
tower geometry
```

例如：

## 云

```text
运动慢
形状分散
PCA elongation 弱
轨迹不符合 Blade
```

## Camera vibration

```text
registration 后应大幅消失
```

## 鸟

```text
面积小
周期不对
轨迹不对
```

这就是为什么固定单目标场景并不一定需要语义模型。

---

# 33. Failure Codes

建议：

```text
FRAME_REGISTRATION_FAIL
NO_MOTION
MULTI_COMPONENT

BLADE_CANDIDATE_NOT_FOUND
BLADE_CANDIDATE_AMBIGUOUS

BLADE_TEMPORAL_JUMP
BLADE_ORIENTATION_INVALID

BLADE_WIDTH_OUTLIER
BLADE_WIDTH_JUMP
BLADE_WIDTH_PROFILE_INVALID

BLADE_CONTOUR_INVALID
BLADE_BOUNDARY_FRAGMENTED
BLADE_TOWER_SIDE_AMBIGUOUS

BLADE_EDGE_NOT_FOUND
BLADE_EDGESUBPIX_FAIL

TOWER_REFERENCE_INVALID
TOWER_EDGE_NOT_FOUND
TOWER_EDGE_SWITCH
TOWER_EDGESUBPIX_FAIL

CLEARANCE_GEOMETRY_INVALID
CLEARANCE_NO_NEAREST_PAIR

FRAME_DROPPED_IN_MEASUREMENT
FRAME_QUALITY_LOW
```

原则：

```text
不能测
→ INVALID

不能：
→ 使用上一帧 Cpx
→ extrapolate 一个 Cpx
```

---

# 34. 无 YOLO 方案是否能获得“同样结果”

可以，但必须明确“同样结果”的定义。

如果 YOLO 方案最终得到：

```text
Blade subpixel boundary = B
Tower subpixel boundary = T
```

无 YOLO 方案最终也得到相同：

```text
B
T
```

则：

$$
C_{px}
=
\min_{p\in B,q\in T}
\|p-q\|_2
$$

完全一样。

所以：

> **Detector 只决定“能否把 Blade ROI 找对”，不决定最终净空精度。**

真正决定 Clearance 精度的是：

```text
Original-resolution Blade Edge
Tower Edge
EdgesSubPix
Geometry
```

---

# 35. 与 YOLO 版本必须做 A/B

不要只比较：

```text
YOLO mAP
vs
CV detection rate
```

应该比较：

| Metric | Classic CV | YOLO Version |
|---|---:|---:|
| Blade Pass Recall | | |
| Closest-region Recall | | |
| Safe ROI Containment | | |
| Blade Boundary Valid Rate | | |
| Tower Boundary Valid Rate | | |
| Cpx Valid Rate | | |
| Cpx P95 Error | | |
| False Valid | | |
| CPU % | | |
| GPU/NPU % | | |
| Latency | | |
| Power | | |

如果 Classic CV 达到同样的：

```text
Cpx
Cmin
valid rate
```

则没有理由强制保留 YOLO。

---

# 36. 推荐验收 Gate

## G1 Blade Pass Recall

$$
R_{pass}\ge99\%
$$

## G2 Closest-region Recall

$$
R_{closest}\ge99\%
$$

## G3 Safe ROI Containment

$$
R_{contain}\ge99\%
$$

## G4 Tower Valid

$$
R_{tower}\ge99\%
$$

并要求：

```text
TOWER_EDGE_SWITCH = 0
```

## G5 Cpx Valid Rate

$$
R_{valid}\ge95\%
$$

## G6 Cpx Error

建议初始：

$$
P95(E_{Cpx})\le2px
$$

## G7 Silent Failure

$$
N_{silent}=0
$$

---

# 37. 无 YOLO 的直接收益

## 1. 不需要 Dataset

省掉：

```text
标注
train/val/test
retraining
model version
weights
```

## 2. 不需要 GPU / NPU

核心可以由 CPU 完成。

## 3. 更适合 RK3588

架构可简化为：

```text
MPP/VPU
→ Decode

CPU
→ Motion
→ OpenCV
→ EdgesSubPix
→ Geometry
```

## 4. 可解释性更高

每次失败都能对应：

```text
motion
component
trajectory
edge
geometry
```

## 5. 更容易审计

算法更：

```text
deterministic
parameterized
auditable
```

---

# 38. 无 YOLO 的主要风险

## 风险 1：Fog / White Sky

Blade 与背景 contrast 下降。

## 风险 2：Camera vibration

如果 registration 不稳，false motion 会增加。

## 风险 3：Cloud motion

需要 temporal + geometry 去除。

## 风险 4：Extreme Pitch

Blade apparent width 很窄，edge quality 下降。

## 风险 5：Camera remount

ROI / Tower reference 需要重新 commissioning。

---

# 39. 未来仍然可以保留 AI Fallback

无 YOLO V1 不等于永远禁止 AI。

长期架构可以是：

```text
Classic CV Primary
       ↓
CV confidence high?
    /          YES           NO
  ↓             ↓
Measurement   YOLO26 / FR-DETR
                ↓
          Semantic Confirmation
                ↓
            Measurement
```

即：

```text
Normal condition
→ no AI

Difficult condition
→ AI fallback
```

但 V1 可以先完全不加载 AI 模型。

---

# 40. 对当前代码最建议的改造策略

不要重写整个系统。

建议：

```text
现有 main.py
↓
复制
↓
main_cv.py
```

保持：

```text
CLI
config
RTSPReader
FrameBuffer
Reader Thread
logging
ResultWriter
Visualizer
reconnect
statistics
```

只替换：

```text
YOLODetector
blade_segmenter
blade_locator
```

为：

```text
FrameRegistrar
MotionDetector
BladeCandidateExtractor
BladeTracker
BladePassFSM
BladeGeometryExtractor
SubpixelEdgeRefiner
TowerReference
ClearanceGeometry
```

这样可以最大程度复用当前 RTSP 实时工程框架。

---

# 41. 推荐目录结构

```text
blade_rtsp_system/
│
├── main.py                  # 当前 YOLO 版
├── main_cv.py               # 新无 YOLO 版
│
├── src/
│   ├── rtsp_reader.py       # 复用
│   ├── frame_buffer.py      # 修改 measurement buffer
│   ├── result_writer.py     # 扩展字段
│   ├── visualizer.py        # 扩展 measurement overlay
│   │
│   ├── frame_registrar.py
│   ├── motion_detector.py
│   ├── blade_candidate.py
│   ├── blade_tracker.py
│   ├── blade_pass_fsm.py
│   ├── blade_geometry.py
│   ├── subpixel_edge.py
│   ├── tower_reference.py
│   └── clearance_geometry.py
│
├── config/
│   ├── config.yaml
│   └── tower_reference.yaml
│
└── output/
    ├── csv/
    ├── jsonl/
    ├── evidence/
    └── logs/
```

---

# 42. 最终推荐

针对当前 Blade Clearance，我建议把无 YOLO 版定义为：

# `Blade Clearance Classic-CV V1`

职责链：

```text
RTSP
↓
Frame Registration
↓
Motion Detection
↓
Connected Components
↓
Temporal Tracking
↓
Blade Pass FSM
↓
Safe ROI
↓
Original-resolution Blade Contour
↓
PCA + Dynamic Width
↓
Tower-facing Boundary
↓
EdgesSubPix
↓
Tower Local Edge
↓
Geometry
↓
Cpx(t)
↓
Cmin_sampled_px
```

这条路线比：

```text
RTSP
→ YOLO
→ Mask
```

更贴近当前项目的实际物理条件。

---

# 43. 最核心的设计判断

对于这个项目，YOLO 解决的是：

```text
“哪个东西是 Blade？”
```

而你的实际场景已经通过：

```text
唯一主要运动目标
+
固定 Camera
+
固定 Tower
+
周期运动
+
连续轨迹
```

几乎提前回答了这个问题。

因此真正应该投入工程精力的地方不是：

```text
把 Blade 再分类一次
```

而是：

```text
把 Blade Boundary 找准
把 Tower Boundary 找准
把 Frame 不丢掉
把 Geometry 算正确
把 INVALID 明确输出
```

最终推荐原则：

> **Classic CV 负责发现并跟踪唯一运动叶片；原始分辨率边缘与 Geometry 负责计量。YOLO 可以作为未来困难天气下的可选 fallback，但不应成为 V1 必需依赖。**
