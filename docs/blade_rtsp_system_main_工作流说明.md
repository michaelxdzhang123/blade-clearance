# blade_rtsp_system/main.py 工作流说明

> 文档对象:`blade_rtsp_system/main.py`(TASK-1 主入口,共 411 行)
> 生成日期:2026-10-01
> 配套配置:`blade_rtsp_system/config/config.yaml`

---

## 1. 为什么有这个程序(WHY)

TASK-1 的目标是:从**实时 RTSP 视频流**里,用 **YOLO26 分割模型**把风机**叶片(blade)**检测出来,把叶片 mask 转成多边形 polygon,并**恢复回原始图像坐标系**,最终输出结构化结果(CSV/JSONL)供下游做**塔筒-叶片净空距离**计算。

程序要解决的核心工程问题:
1. **实时性** — 视频源是 RTSP 流,不是本地文件,读帧和推理必须解耦(多线程),否则推理耗时会把读帧卡死。
2. **精度** — 分割 mask 要做 polygon 提取 + ROI 坐标回映射,坐标必须回到原始帧。
3. **鲁棒性** — 断流要自动重连、headless 环境要自动关窗口。
4. **性能可观测** — 每帧耗时(推理 `inf_ms`、主循环 `loop_ms`)、FPS、丢帧数都要能统计出来。

---

## 2. 整体流水线(一图概览)

```
RTSP 流
  │
  ▼
[Thread 1] RTSPReader 读帧 ──► FrameBuffer (latest, max_size=1)
  (断流自动重连)                       │
                                      │ get_latest()
                                      ▼
[Thread 2 主线程] ──► ROI 裁剪(可选) ──► YOLODetector.predict (imgsz=640)
                                      │  result + inf_ms(推理耗时)
                                      ▼
                          blade_segmenter.extract_blades
                          (筛出 class_name="blade" 的 instances)
                                      │
                                      ▼
                          blade_locator.locate_blades
                          (mask→polygon + ROI→原图坐标恢复)
                                      │
                                      ▼
                          Visualizer.draw (画标注 + 帧号 + FPS)
                                      │
                                      ▼
                          build_frame_result → ResultWriter (CSV/JSONL, 内存缓冲)
                                      │
                                      ▼
                          每 5 秒打印周期统计 / 每帧记录 loop_ms
```

---

## 3. 多线程模型

| 线程 | 职责 | 代码位置 |
|------|------|----------|
| Thread 1(Reader 后台线程) | 读 RTSP 帧 → 塞进 latest-frame buffer;断流自动重连 | `main.py:234-255` |
| Thread 2(主线程) | YOLO 推理 + 叶片提取/定位 + 可视化 + 输出 | `main.py:284-370` |

两线程通过 `FrameBuffer`(latest 策略,`max_size=1`)交接:Reader 只管塞帧,主线程只管取最新帧,**新帧会覆盖未处理的旧帧**(这就是 `dropped_frames` 的统计来源)。两者用 `threading.Event`(`running`)统一控制退出。

---

## 4. 分阶段工作流(带行号)

### 阶段 0 — 入口与模块导入(`main.py:410-411, 21-40`)
- `sys.path.insert(...)` 把 main.py 所在目录加入搜索路径,保证 `from src import ...` 可用(`:32`)。
- 导入 6 个 src 模块:RTSPReader(读流)、FrameBuffer(缓冲)、YOLODetector(模型)、blade_segmenter/blade_locator(叶片业务)、ResultWriter(写结果)、Visualizer(画图)。

### 阶段 1 — 解析参数(`main.py:133-137`)
- `--config`:配置文件路径(默认 `config/config.yaml`)。
- `--max-duration`:最大运行秒数(默认 0=不限),用于定时长跑测试。

### 阶段 2 — 加载配置 + 路径解析(`main.py:139-143`)
- `load_config()` 读 YAML → dict。
- `root_dir = main.py 所在目录` = `blade_rtsp_system/`。
- **关键约定**:config.yaml 里所有相对路径都相对 `root_dir`(不是 config/ 子目录),用 `resolve_path()` 解析(`:96-104`)。

### 阶段 3 — 日志初始化(`main.py:148-153`)
- 用自定义 `BufferedFileHandler`(`:47-75`):`emit()` 只把格式化字符串追加到内存 list(**运行时零 I/O**),退出时 `flush_to_file()` 一次性写 `output/logs/runtime.log`。
- 与 `rtsp_daemon.py` 同款,目的是避免逐条日志的磁盘 I/O 拖慢读帧。

### 阶段 4 — 加载 YOLO 模型(`main.py:156-177`)
- 检查 `models/best.pt` 存在,否则报错返回 1。
- 构造 `YOLODetector`:imgsz=640(固定)、conf=0.5、device、retina_masks=true。
- `detector.load()` 加载(失败抛异常 → 返回 1)。

### 阶段 5 — 创建输出 Writer(`main.py:180-193`)
- `save_csv` / `save_jsonl` 开关。
- CSV 默认 `output/csv/blade_positions.csv`,JSONL 默认 `output/jsonl/blade_positions.jsonl`。
- `ResultWriter` 现在也是**内存缓冲**:`write()` 只追加到内存,`close()` 时一次性落盘。

### 阶段 6 — ROI 配置(`main.py:196-203`)
- `use_roi` 开关 + `roi_offset=(x1,y1)`(用于把 ROI 局部坐标平移回原图)。

### 阶段 7 — 打开视频源(`main.py:206-224`)
- 只用 RTSP(无 MP4 模式),地址来自 `rtsp.url`(默认 `rtsp://127.0.0.1:8554/blade`)。
- 日志不打印明文 URL(用 `_masked_source()` 脱敏)。
- `reader.open()` 失败 → 返回 1。

### 阶段 8 — 启动 Reader 线程(`main.py:227-256`)
- `FrameBuffer(max_size=1)`。
- `reader_loop()`:循环读帧 → 塞 buffer;读帧失败进入**重连循环**(`reconnect_interval_sec` 间隔、`max_reconnect_attempts` 上限,0=无限)。
- 用 `threading.Thread(daemon=True)` 启动。

### 阶段 9 — 可视化 + 帧尺寸(`main.py:259-270`)
- `show_window=true` 但无 `DISPLAY`(headless)→ 自动降级关窗口。
- 取原始帧宽高(`CAP_PROP_FRAME_WIDTH/HEIGHT`)供坐标回映射用。

### 阶段 10 — SIGTERM 处理器(`main.py:278-282`)
- 把 SIGTERM 也映射成 `raise KeyboardInterrupt`,复用 `except KeyboardInterrupt → finally → 落盘`。
- **目的**:避免缓冲的结果/日志在 SIGTERM 时被直接杀掉不落盘而丢失。

### 阶段 11 — 主推理循环(`main.py:284-370`)
每帧执行:
1. `buffer.get_latest()` 取最新帧;buffer 空则 `sleep(0.01)` 继续(`:288-294`)。
2. `t_frame = perf_counter()` 计时起点(`:297`)。
3. 可选 ROI 裁剪(`:300-306`)。
4. `detector.predict()` → `(result, inf_ms)`,记录推理耗时(`:309-310`)。
5. `blade_segmenter.extract_blades(result, class_name="blade")` 筛 blade 实例(`:315`)。
6. `blade_locator.locate_blades(...)` 做 mask→polygon + ROI→原图坐标恢复,输出 bbox/center/polygon/叶尖候选(`:319-323`)。
7. 可视化:算 FPS、`vis.draw()` 画标注、`vis.show()`(`:326-333`)。
8. `loop_ms = perf_counter() - t_frame`,记录主循环本帧耗时(`:336-337`)。
9. `build_frame_result(...)` 组装结果 + `writer.write(fr)`(`:340-342`)。
10. 每 5 秒打印一次周期统计(recv/proc/dropped + avg/max 推理耗时 + avg/max 主循环耗时)(`:345-365`)。
11. 达到 `--max-duration` 则 `break` 正常退出(`:368-370`)。

### 阶段 12 — 收尾(`main.py:372-405`)
- `except KeyboardInterrupt`(Ctrl+C / SIGTERM)→ 记录停止。
- `finally` 保证一定执行:停 Reader 线程 → 释放视频源 → `writer.close()`(落盘结果)→ 关窗口 → 最终统计 → `log_handler.flush_to_file()`(落盘日志)。

---

## 5. 每帧数据流(输入 → 输出)

| 步骤 | 输入 | 输出 |
|------|------|------|
| 读帧 | RTSP 流 | `(frame, ts, frame_id)` |
| 缓冲 | frame | 最新帧(`FrameBuffer`) |
| 推理 | frame 或 ROI 子图 | `(result, inf_ms)` |
| 提取 | result | `detection["blade_instances"]` |
| 定位 | blade_instances + roi_offset + 原图尺寸 | `located`(bbox/center/polygon/叶尖) |
| 组装 | fid/ts/尺寸/detection/located/loop_ms | 结果 dict |
| 写盘 | 结果 dict | CSV 行 / JSONL 行(close 时落盘) |

**最终结果字段**(`build_frame_result`,`main.py:107-125`):
`frame_id` / `timestamp` / `frame_width` / `frame_height` / `blade_detected` / `blade_count` / `loop_ms` / `blades[]`

CSV 列(`result_writer.py CSV_FIELDS`):
`timestamp, frame_id, frame_width, frame_height, loop_ms, blade_detected, blade_count, instance_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, center_x, center_y, mask_area`

---

## 6. 关键组件职责

| 模块 | 职责 |
|------|------|
| `RTSPReader` | RTSP 读帧 + 断流重连 + URL 脱敏 |
| `FrameBuffer` | latest 帧缓冲(max_size=1)+ recv/proc/dropped 统计 |
| `YOLODetector` | 模型加载(`load()` 返回 None)+ 单帧 `predict()` 返回 `(result, inf_ms)` |
| `blade_segmenter.extract_blades` | 从 result 筛出 `class_name="blade"` 的 instances |
| `blade_locator.locate_blades` | mask→polygon + ROI→原图坐标恢复 + 叶尖候选 |
| `ResultWriter` | 结果内存缓冲 → close 时一次性写 CSV/JSONL |
| `Visualizer` | 画 mask/polygon + 帧号 + FPS(headless 自动降级) |
| `BufferedFileHandler` | 日志内存缓冲 → flush_to_file 时落盘 |

---

## 7. 退出路径总结

| 触发 | 方式 | 是否落盘 |
|------|------|----------|
| 用户 Ctrl+C | `except KeyboardInterrupt` | ✅ finally 落盘 |
| SIGTERM 信号 | SIGTERM handler → `raise KeyboardInterrupt` | ✅ finally 落盘 |
| `--max-duration` 到点 | `break` 正常走出循环 | ✅ finally 落盘 |
| 模型缺失/加载失败 | `return 1`(早退) | ✅ 手动 `flush_to_file` |
| 视频源打开失败 | `return 1`(早退) | ✅ 手动 `flush_to_file` |
| SIGKILL(`kill -9`) | 强制杀 | ❌ 缓冲丢失(无法避免) |

---

## 8. 输出文件位置

| 文件 | 路径 | 内容 |
|------|------|------|
| 结果 CSV | `output/csv/blade_positions.csv` | 每帧一行,含 `loop_ms` 列 |
| 结果 JSONL | `output/jsonl/blade_positions.jsonl` | 每帧一行,含 `loop_ms` 字段 |
| 运行日志 | `output/logs/runtime.log` | 启动/周期统计/最终统计(含 `avg_loop`/`max_loop`) |

> 注意:CSV/JSONL/日志都是**退出时才一次性落盘**(内存缓冲),运行中途看不到文件更新,这是为了消除每帧磁盘 I/O 对推理的干扰。

---

## 9. 性能指标解读

- `inf_ms`:纯 YOLO 推理耗时(单帧)。
- `loop_ms`:主循环处理单帧总耗时(取帧后 → 可视化完成),即"每帧处理时间"。
- `recv / proc / dropped`:Reader 收到帧数 / 主循环处理帧数 / 因 latest 策略被覆盖丢弃的帧数。
- `avg_fps = processed_frames / elapsed`:平均处理帧率。

`loop_ms` 就是用户关心的"每帧处理时间",分布在 CSV 的 `loop_ms` 列、JSONL 的 `loop_ms` 字段、以及 runtime.log 的 `avg_loop`/`max_loop` 统计里。
