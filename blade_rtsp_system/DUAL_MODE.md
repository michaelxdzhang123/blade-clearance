# YOLO / Classic CV 双模式

在项目根目录运行（实际目录名是 `blade_rtsp_system`，下划线）：

```bash
./blade_rtsp_system/start_main.sh yolo
./blade_rtsp_system/start_main.sh classic
# 等价开关
./blade_rtsp_system/start_main.sh --with-yolo
./blade_rtsp_system/start_main.sh --without-yolo
# 限时运行
./start_main.sh classic --max-duration 60
```

无参数默认 YOLO；脚本支持从任意工作目录启动，自动使用项目 `.venv/bin/python`。
`BLADE_PYTHON` 可以指定其他 Python，`BLADE_MODE` 可以指定脚本默认模式。
其他参数原样传给 main.py，例如 `--config`、`--resize W H`。

直接运行：

```bash
.venv/bin/python blade_rtsp_system/main.py --mode classic
.venv/bin/python blade_rtsp_system/main.py --mode yolo
```

直接运行时，CLI `--mode` 优先于配置 `detection.mode`，旧配置缺省为 YOLO。
视频源仍从 `rtsp.url` 读取；启动脚本不会自动启动 RTSP 推流服务。

## 已实现

- YOLO 分支保留原模型、实例提取、轮廓定位；只在选择 YOLO 时导入 Ultralytics。
- Classic 分支只需要 OpenCV / NumPy。启动前 `warmup_frames` 帧以中值建立背景；帧差用于触发，当前帧与背景差用于粗定位，形态学及面积筛选候选，按面积、形状和位置连续性评分保留 Top-K。
- 检测到叶片期间冻结背景；仅无前景、无运动时缓慢更新背景。
- 两种模式复用 RTSP、ROI、可视化、CSV/JSONL。CSV 新增 `detector_mode`、`detector_status`、`confidence_kind`、`frames_skipped`；JSONL 额外记录 Classic 状态。
- Classic 的 `confidence` 是候选评分，不是神经网络概率；显示的叶尖仍为候选位置。
- ROI 与输出坐标遵循现有程序的 `processing.resize` 后图像坐标。配置 Classic 面积阈值时也使用此尺度。

## 与设计方案的边界

本次实现的是可切换的检测/粗定位后端，不是文档中的整套计量系统。
尚未实现相机配准、原始分辨率测量边界、PCA 宽度剖面、EdgesSubPix、塔筒边界精修、线段净空计算和完整测量 FSM。
因此 Classic 明确输出 `measurement_valid=false`、`cpx_euclidean_px=null`、`MEASUREMENT_NOT_IMPLEMENTED`；不会用运动轮廓代替真实净空。

现有 latest-frame 缓冲仍可能跳帧，`frames_skipped` 记录主处理线程观察到的帧号缺口；它不能证明相机或传输端没有丢帧。未实现测量窗口无丢帧队列。

使用固定摄像机，尽量从无叶片场景启动。持续占据背景的物体、相机抖动、阴影、光照变化及其他运动物体仍可能造成误检；需要现场数据调参。无可靠候选时不会沿用上一帧的叶片坐标。

## 验证

```bash
.venv/bin/python -m unittest discover -s blade_rtsp_system/tests -p test_detection_modes.py -v
bash -n blade_rtsp_system/start_main.sh start_main.sh
```

测试覆盖静态场景、全局变化、当前叶片与前帧残影区分、ROI 映射，以及两个后端经主循环写出 CSV/JSONL。
