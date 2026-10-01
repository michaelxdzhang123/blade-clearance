# scripts/ — RTSP 测试服务器脚本

用本地 MP4 模拟实时 RTSP 摄像机流（`MediaMTX` + `FFmpeg`），供 YOLO26 / OpenCV 实时推理与
断流重连测试，无需真实 IPC 相机、无需 Docker/sudo。

数据链：

```
MP4 -> FFmpeg(-re, -stream_loop -1) -> MediaMTX (RTSP :8554) -> OpenCV / TASK-1
```

默认 RTSP 地址：`rtsp://127.0.0.1:8554/blade`

---

## 快速开始（推荐：原生二进制，无需 Docker）

```bash
# 1. 启动 MediaMTX RTSP 服务器（监听 8554）
./scripts/start_mediamtx_native.sh

# 2. 循环推流本地 MP4 到 RTSP
./scripts/start_rtsp_stream_daemon.sh data/training-data/test01-video-2026-08-28_134655_607.mp4

# 3. 验证（应打印 h264 2560x1440 25fps）
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/blade

# 4. 停止
./scripts/stop_rtsp_stream_daemon.sh
./scripts/stop_mediamtx_native.sh
```

> 前提：`~/.local/bin/mediamtx` 已安装（下载见下）。`mediamtx.yml` 精简配置只开 RTSP TCP，
> 并显式 `source: publisher`（MediaMTX v1.21 否则拒绝推流）。

---

## 脚本清单

### MediaMTX 服务器

| 脚本 | 说明 |
|------|------|
| `start_mediamtx_native.sh` | **推荐**。用原生 `mediamtx` 二进制后台启动（setsid daemon），无需 Docker/sudo |
| `stop_mediamtx_native.sh` | 停止原生 mediamtx |
| `start_mediamtx.sh` | Docker 版（`bluenviron/mediamtx:1`），需 Docker daemon 运行 |
| `mediamtx.yml` | 精简配置：仅 RTSP TCP 8554 + `source: publisher` |

### FFmpeg 推流

| 脚本 | 说明 |
|------|------|
| `start_rtsp_stream_daemon.sh` | **推荐**。后台循环推流（`-c copy`，setsid daemon），写 pid 文件 |
| `stop_rtsp_stream_daemon.sh` | 停止后台推流 |
| `start_rtsp_stream.sh` | 前台推流版（Ctrl+C 停止，适合临时调试） |
| `start_rtsp_stream_h264.sh` | libx264 重编码版（`-c copy` 因编码不兼容失败时用） |

### 一键 / 验证

| 脚本 | 说明 |
|------|------|
| `start_rtsp_test.sh` | Docker 一键启动（MediaMTX + FFmpeg），需 Docker |
| `stop_rtsp_test.sh` | 停止上述 Docker 一键环境 |
| `verify_rtsp.py` | OpenCV 读帧验证（需 GUI，headless 用下面的 inline 方式） |

---

## MediaMTX 二进制安装（一次性）

```bash
# 资产名含版本号 mediamtx_vX.Y.Z_linux_amd64.tar.gz（不带版本号会 404）
curl -sL -o /tmp/mediamtx.tar.gz \
  "https://github.com/bluenviron/mediamtx/releases/download/v1.21.1/mediamtx_v1.21.1_linux_amd64.tar.gz"
tar xzf /tmp/mediamtx.tar.gz -C /tmp
cp /tmp/mediamtx ~/.local/bin/mediamtx && chmod +x ~/.local/bin/mediamtx
~/.local/bin/mediamtx --version
```

---

## 验证 RTSP 链路（从快到全）

```bash
# ① 进程层：8554 是否监听
ss -tlnp | grep ':8554'

# ② 推流层：FFmpeg 是否在推帧（frame 计数持续增长）
pgrep -af "ffmpeg.*rtsp"
tail -5 /tmp/rtsp_ffmpeg.log

# ③ 协议层：ffprobe 探测（最直接的"通不通"）
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/blade

# ④ 应用层：OpenCV 读一帧（headless 无 imshow）
.venv/bin/python -c "import cv2; c=cv2.VideoCapture('rtsp://127.0.0.1:8554/blade',cv2.CAP_FFMPEG); ok,f=c.read(); print('OK' if ok else 'FAIL', None if f is None else f.shape); c.release()"

# ⑤ 发布确认：MediaMTX 日志出现 is publishing / is reading from path 'blade'
tail -5 /tmp/mediamtx.log
```

---

## 断流重连测试

1. 后台跑消费端（如 `blade_rtsp_system/main.py`）持续读 RTSP。
2. `./scripts/stop_rtsp_stream_daemon.sh` 停推流 → 消费端检测断流、进入重连。
3. `./scripts/start_rtsp_stream_daemon.sh ...` 重启 → 消费端重连成功、恢复推理。

---

## 已知坑（详见 skill `mediamtx-rtsp-test-server`）

1. **MediaMTX v1.21 默认拒绝 publish**：`paths.all_others` 空条目导致
   `path 'xxx' is not configured`（客户端 400 Bad Request），必须 `source: publisher`。
2. **UDP 8000 端口冲突**：默认 `rtpAddress :8000`，设 `rtspTransports: [tcp]` 避开。
3. **Hermes background 的 `bash -lic` 包装**会触发 conda init 报错 + 延迟长驻进程启动，
   故本目录 daemon 脚本一律用 `setsid` 后台化。
4. **`-c copy` 报 400** 时先查 MediaMTX 日志区分「path 未配置」vs「编码不兼容」，
   多数是前者（加 `source: publisher`），非 SPS/PPS 问题。
