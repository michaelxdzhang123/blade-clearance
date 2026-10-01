# TASK-1 执行报告：YOLO26 + RTSP 实时叶片定位

## 1. 开发环境

| 项 | 值 |
|----|----|
| 主机 | WSL (Windows Subsystem for Linux) |
| Python | 3.13.13 (.venv, uv 管理) |
| Ultralytics | 8.4.158 |
| OpenCV | 5.0.0(cv2.VideoCapture/CAP_FFMPEG) |
| GPU | NVIDIA GeForce RTX 4090 × 2 |
| CUDA | torch 2.14.0+cu130, cuda available=True |
| best.pt 路径 | blade_rtsp_system/models/best.pt |
| RTSP 原始分辨率 | 2560×1440 @ 25fps |
| imgsz | 640 |
| 平均 inference time | ~21.7 ms(2560×1440 输入, YOLO 内部 imgsz=640) |

## 2. 全部 Gate 状态

| Gate | 状态 |
|------|------|
| Gate 1 模型加载 | ✅ PASS |
| Gate 2 本地 MP4 | ✅ PASS |
| Gate 3 RTSP 接入 | ✅ PASS |
| Gate 4 Blade 检测 | ✅ PASS(984/1233 帧检出, 80%) |
| Gate 5 坐标正确 | ✅ PASS |
| Gate 6 持续 30min | ✅ PASS(1800s, recv=43760, reconnect=0, 0 错误) |

## 3. Gate 详细验证

### Gate 1：模型加载 ✅

```
model loaded: True
device: cuda:0
imgsz: 640
task: segment
names: {0: 'blade'}
```

### Gate 2：本地 MP4 ✅

用 `data/training-data/1-rain.mp4`(640×360, 418 帧)跑完整链路：

```
处理 369 帧(49 dropped = 推理慢于视频 FPS 时旧帧被覆盖, 预期行为)
检出 157 帧(42.5%, 叶片在视频中段横扫进入画面)
首帧检出 frame_id=74, conf=0.7295, raw_nodes=588, simple_nodes=17
CSV 376 行, JSONL 369 行
```

### Gate 3：RTSP 接入 ✅

**环境**：mediamtx v1.21.1 + ffmpeg/ffprobe 7.0.2(静态二进制, 装到 `/home/mich/.local/bin/`)。

RTSP 链路(ffprobe 探测)：

```
codec_name=h264  width=2560  height=1440  r_frame_rate=25/1
MediaMTX: stream is available and online, is publishing to path 'blade'
```

TASK-1 RTSP 模式运行(60 秒)：

```
视频源已打开: rtsp://127.0.0.1:8554/blade (2560x1440 @ 25.00 fps)
统计: recv=1264 proc=1140 dropped=124 avg_inf=25.1ms
```

- RTSP 原始帧 2560×1440 完整保留(未 resize 到 640)
- polygon 坐标映射回 2560×1440 系(首个检出帧 bbox=[2195,933,2559,1440], 全部在界内)
- 检出 984/1233 帧(80% 检出率)

**断流重连验证** ✅：

```
停推流 → 断流, 第 1~11 次重连(无限模式 max_reconnect_attempts=0, 不退出)
重启推流 → 第 11 次重连成功, 恢复推理(统计恢复输出, recv 继续增长)
```

### Gate 4：Blade 检测 ✅

RTSP 帧抽查：blade_detected / bbox / mask / polygon 数据结构正确, 984/1233 帧检出。

### Gate 5：坐标正确性 ✅

本地 MP4 frame 15 数值验证：

```
raw_polygon 顶点数 = 1060
原图尺寸 640×360, 所有顶点在界内: True
polygon 在 bbox 内: True
center(385.3, 291.1) ≈ bbox 中心(384.5, 290.5), 误差 < 1px
```

RTSP 帧(2560×1440)：

```
首个检出帧 bbox=[2195,933,2559,1440]
polygon x[2196,2558] y[934,1439], 全部在 2560×1440 界内
raw_nodes=1417(高清图上轮廓更精细)
```

### Gate 6：持续运行 ✅ PASS (30 分钟长跑)

`--max-duration 1800` 连续运行 1800.1s, 全程无异常：

```
最终统计: recv=43760 proc=42686 dropped=1073 avg_inf=21.7ms avg_fps=23.7 reconnect=0 运行=1800.1s
```

- 接收 43760 帧, 处理 42686 帧, 掉帧 1073(2.5%, 推理 21.7ms vs 视频 40ms/帧, 少量旧帧覆盖)
- 平均推理 21.7ms, 平均处理 FPS 23.7(接近视频 25fps)
- reconnect=0(推流全程未断)
- 0 条 ERROR/Traceback/Exception
- 首帧 ~1792ms(CUDA warm-up) → 稳态 21.7ms, 30 分钟无明显漂移
- 无内存增长/线程死锁/模型重复加载

## 4. 单元测试 ✅

```
test_coordinate_mapper: ALL PASS  (ROI offset=(100,200) → Global=(150,260))
test_mask_processor:   ALL PASS  (raw_polygon 不被 simplify 修改)
```

## 5. 代码结构

```
blade_rtsp_system/
├── config/config.yaml        # 配置(模型路径/RTSP/ROI/buffer/output)
├── models/best.pt            # 已复制(640 segment blade)
├── src/
│   ├── rtsp_reader.py        # RTSP 读取 + 本地 MP4 回退 + 自动重连 + 密码脱敏
│   ├── frame_buffer.py       # latest-frame buffer + received/processed/dropped 统计
│   ├── yolo_detector.py      # 模型只加载一次 + inference time + 设备记录
│   ├── blade_segmenter.py    # 提取全部 blade instances + primary(最大mask面积)
│   ├── mask_processor.py     # CHAIN_APPROX_NONE raw contour + approxPolyDP simple
│   ├── coordinate_mapper.py  # ROI→原始帧 坐标恢复(bbox/center/polygon/mask)
│   ├── blade_locator.py      # 最终定位 + blade_tip_candidate
│   ├── result_writer.py      # CSV(扁平行) + JSONL(含polygon)
│   └── visualizer.py         # simple_polygon 绘制(headless 自动降级)
├── tests/
│   ├── test_coordinate_mapper.py  ✅
│   └── test_mask_processor.py     ✅
├── output/{csv,jsonl,video,logs}/
├── main.py                   # 多线程 Reader + 主循环推理
├── requirements.txt
└── README.md
```

## 6. 关键设计决策

1. **imgsz=640 只作推理尺寸**：RTSP 原始帧不 resize；`result.masks.xy` 已恢复到输入图像坐标，再经 coordinate_mapper 恢复 ROI 偏移(第 14 节)。

2. **raw_polygon 永久保留**：`CHAIN_APPROX_NONE` 原始轮廓(1060~1417 顶点)，`simple_polygon` 仅 approxPolyDP 简化(9~17 顶点)用于显示(Rule 4/5)。

3. **latest-frame buffer**：本地 MP4 模式按视频 FPS 节流(否则 Reader 瞬间读完整视频导致 416/418 dropped)，RTSP 模式天然实时不节流。

4. **headless 自动降级**：WSL 无 DISPLAY 时关闭 cv2.imshow 并记录 warning(不静默)。

5. **CUDA tensor 处理**：mask 是 CUDA tensor，`_mask_area` 和 `binarize_mask` 先 `.detach().cpu().numpy()` 再转 numpy(修复 TypeError)。

6. **密码脱敏**：RTSP URL 含 username:password 时日志输出 `user:***@host`(第 25 节)。

7. **无限重连语义**：`max_reconnect_attempts=0` = 无限重连，单次 open 失败(如推流尚未重新发布导致 404)继续重试，不退出(修复断流后一次 404 就退出的 bug)。

## 7. 修复的 bug(开发过程中)

| Bug | 修复 |
|-----|------|
| mask 是 CUDA tensor, np.asarray 报 TypeError | `.detach().cpu().numpy()` 先转 CPU |
| config 相对路径解析到 config/ 子目录 | 改为解析到 blade_rtsp_system 根目录 |
| headless 环境 cv2.imshow 崩溃 | 检测 DISPLAY 环境变量自动降级 |
| 本地 MP4 快速读取致 dropped=416 | Reader 按视频 FPS 节流 |
| 断流重连一次 404 就退出(无限模式失效) | 拆出 reconnect_exhausted(), 循环重连直到成功 |

## 8. 已知问题

1. 首帧推理 ~1.8s 是模型 warm-up(加载 CUDA kernel), 后续稳定 ~21.7ms。
2. `blade_tip_candidate`(y 最大顶点)是候选点, 非真实物理叶尖(需 TASK-2 细化)。

## 9. TASK-2 建议

1. **坐标接口资产**：JSONL 每帧含 `raw_polygon`(原始帧坐标)+ `blade_tip_candidate`, 可直接作为 TASK-2 输入。
2. **叶尖精度**：TASK-1 的 `blade_tip_candidate` 在 640×360 上 tip_y=359 稳定触底, 但叶尖真实精度需 TASK-2 用原始分辨率边缘(暗→亮阶跃)细化, 参考 opencv/refine_blade_tip.py 的梯度抛物线拟合法。
3. **性能**：2560×1440 源视频推理 avg 25ms(~40 FPS), 已经够实时；若后续加塔筒边缘检测变慢, 优先用 ROI(processing.use_roi)而非跳帧。
4. **RTSP 延迟**：建议给测试视频加动态时间戳(见 RTSP_MP4_TEST_SERVER方案.md 第 14 节)评估 RTSP+Decode+Inference 总延迟。

## 10. 结论

- **Gate 1/2/3/4/5/6 全部 ✅ PASS**, 本地 MP4 + RTSP 完整链路 + 断流重连 + 30 分钟长跑全部验证通过。
- TASK-1 全部核心代码 + 单元测试 + 双源(本地/RTSP)验证完成。
