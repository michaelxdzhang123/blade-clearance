# 使用已有 MP4 搭建 RTSP 测试服务器方案

## 1. 目标

已有本地视频文件，例如：

```text
blade_clearance_02.mp4
```

目标是将该视频模拟成一个实时 RTSP 摄像机视频流，供后续 YOLO26 实时推理系统测试使用。

推荐架构：

```text
已有 MP4 视频
      │
      ▼
    FFmpeg
  实时速度循环播放
      │
      │ RTSP Publish
      ▼
   MediaMTX
   RTSP Server
      │
      ▼
rtsp://127.0.0.1:8554/blade
      │
      ▼
TASK-1 / OpenCV / YOLO26
```

该方案适用于：

- 本地开发；
- YOLO26 实时推理测试；
- RTSP 断流/重连测试；
- 多线程 Frame Buffer 测试；
- 后续真实 IPC 摄像机接入前的仿真验证。

---

# 2. 推荐组件

使用两个开源工具：

```text
MediaMTX
+
FFmpeg
```

其中：

- `MediaMTX`：作为 RTSP Server；
- `FFmpeg`：将本地 MP4 按实时速度循环推送到 RTSP Server。

推荐 RTSP 地址：

```text
rtsp://127.0.0.1:8554/blade
```

---

# 3. 启动 MediaMTX RTSP Server

如果系统已经安装 Docker，可以直接执行：

```bash
docker run --rm -it \
  --name mediamtx \
  -e MTX_RTSPTRANSPORTS=tcp \
  -p 8554:8554 \
  bluenviron/mediamtx:1
```

其中：

```text
8554
```

为 RTSP 服务端口。

启动成功后：

```text
MediaMTX
  ↓
RTSP Listener
  ↓
0.0.0.0:8554
```

---

# 4. 使用 MP4 模拟摄像机

假设视频：

```text
blade_clearance_02.mp4
```

执行：

```bash
ffmpeg \
  -re \
  -stream_loop -1 \
  -i blade_clearance_02.mp4 \
  -c copy \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

---

# 5. 参数说明

## 5.1 `-re`

```bash
-re
```

表示：

```text
按照原视频实际播放速度读取
```

例如原视频：

```text
25 FPS
```

则 FFmpeg 将尽量按照：

```text
25 frame/s
```

输出。

这非常重要。

如果不使用 `-re`，FFmpeg 可能会尽可能快地读取文件，而不是模拟真实实时摄像机。

---

## 5.2 `-stream_loop -1`

```bash
-stream_loop -1
```

表示：

```text
视频无限循环播放
```

即：

```text
video.mp4
   ↓
播放结束
   ↓
重新从头播放
   ↓
继续 RTSP Stream
```

适合长时间 YOLO 调试。

---

## 5.3 `-c copy`

```bash
-c copy
```

表示：

```text
不重新编码视频
```

优点：

- CPU 占用低；
- 不损失画质；
- 基本保持原始视频分辨率；
- 基本保持原始编码。

如果原 MP4 是标准 H.264/H.265，一般优先使用该模式。

---

## 5.4 `-rtsp_transport tcp`

```bash
-rtsp_transport tcp
```

表示 RTSP 使用 TCP 传输。

测试阶段推荐 TCP，原因：

```text
稳定
+
避免 UDP 丢包干扰
+
便于排查问题
```

---

# 6. RTSP 测试地址

最终：

```text
rtsp://127.0.0.1:8554/blade
```

就可以作为一台“虚拟网络摄像机”。

在 TASK-1 的配置中：

```yaml
rtsp:
  url: "rtsp://127.0.0.1:8554/blade"
```

即可直接读取。

---

# 7. 首先使用 ffplay 验证

在启动 YOLO26 前，先验证 RTSP 链路。

执行：

```bash
ffplay \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

如果可以正常看到视频，说明：

```text
MP4
 ↓
FFmpeg
 ↓
MediaMTX
 ↓
RTSP
```

已经正常工作。

---

# 8. 使用 OpenCV 验证 RTSP

可以使用最简单的 Python 脚本：

```python
import cv2

url = "rtsp://127.0.0.1:8554/blade"

cap = cv2.VideoCapture(url)

while True:

    ret, frame = cap.read()

    if not ret:
        print("RTSP read failed")
        break

    print(frame.shape)

    cv2.imshow("RTSP TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

# 9. 如果 `-c copy` 失败

先检查原始视频编码：

```bash
ffprobe blade_clearance_02.mp4
```

重点查看：

```text
Video: h264
```

或者：

```text
Video: hevc
```

如果原视频编码不适合直接 RTSP Copy，可以重新编码为 H.264。

---

# 10. H.264 重新编码推流方案

执行：

```bash
ffmpeg \
  -re \
  -stream_loop -1 \
  -i blade_clearance_02.mp4 \
  -an \
  -c:v libx264 \
  -preset veryfast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

数据链变成：

```text
MP4
 ↓
Decode
 ↓
H.264 Encode
 ↓
RTSP
```

这种方式兼容性更好。

---

# 11. 推荐保留原始视频分辨率

如果原视频是：

```text
1920 × 1080
25 FPS
```

建议 RTSP 流仍然保持：

```text
1920 × 1080
25 FPS
```

不要在 RTSP Server 侧提前转换成：

```text
640 × 640
```

因为当前项目的正确架构应该是：

```text
RTSP
1920×1080
      ↓
Original Frame
      │
      ├────────────────────┐
      │                    │
      ▼                    │
YOLO26 preprocess          │
640×640                    │
      ↓                    │
best.pt                    │
      ↓                    │
Blade Mask                 │
      ↓                    │
恢复坐标                    │
      ↓                    ▼
Original 1920×1080 Coordinates
```

其中：

```text
640
```

只是 YOLO26 的 AI 推理尺寸。

最终 Blade Position 和后续 Clearance 计算应回到：

```text
Original RTSP Frame Coordinate System
```

---

# 12. 与 TASK-1 的连接方式

TASK-1 当前流程：

```text
RTSP
→ Frame
→ ROI
→ YOLO26 best.pt
→ Blade Mask
→ Raw Polygon
→ Original Coordinate
→ Blade Position
```

测试时：

```text
blade_clearance_02.mp4
          ↓
       FFmpeg
          ↓
    MediaMTX RTSP
          ↓
rtsp://127.0.0.1:8554/blade
          ↓
      TASK-1
          ↓
     RTSP Reader
          ↓
       Frame
          ↓
 YOLO26 best.pt / 640
          ↓
     Blade Mask
          ↓
      Polygon
          ↓
 Original Coordinates
```

因此 TASK-1 本身无需因为测试视频源而修改。

后续接真实 IPC 时，只需要替换：

```yaml
rtsp:
  url: "..."
```

---

# 13. RTSP 断流测试

TASK-1 中要求实现：

```text
RTSP reconnect
```

因此可以利用当前测试环境主动制造断流。

正常运行：

```text
MediaMTX
+
FFmpeg
+
YOLO26
```

然后在 FFmpeg 终端：

```text
Ctrl + C
```

停止推流。

此时 TASK-1 应检测到：

```text
RTSP read failed
```

并进入：

```text
Reconnect
```

逻辑。

随后重新启动 FFmpeg：

```bash
ffmpeg \
  -re \
  -stream_loop -1 \
  -i blade_clearance_02.mp4 \
  -c copy \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

程序应能够：

```text
重新连接
→ 重新收到 Frame
→ 恢复 YOLO 推理
```

---

# 14. 建议测试 RTSP 延迟

在后续测试中，需要关注：

```text
Camera Time
vs
YOLO Processing Time
```

可以给测试视频增加动态时间戳。

例如：

```bash
ffmpeg \
  -re \
  -stream_loop -1 \
  -i blade_clearance_02.mp4 \
  -vf "drawtext=text='%{localtime\:%H\\:%M\\:%S}':x=20:y=20:fontsize=32:fontcolor=white" \
  -an \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

这样可以比较：

```text
视频帧时间
vs
YOLO 当前系统时间
```

从而估算：

```text
RTSP + Decode + Inference Latency
```

---

# 15. 双机测试方式

如果 RTSP Server 和 YOLO GPU Server 不在同一台机器上：

假设 RTSP Server：

```text
192.168.1.50
```

结构：

```text
PC-A

video.mp4
   ↓
FFmpeg
   ↓
MediaMTX
192.168.1.50:8554
          │
          │ LAN
          ▼
PC-B / GPU Server
          │
          ▼
YOLO26 best.pt
```

PC-B 读取：

```text
rtsp://192.168.1.50:8554/blade
```

这种方式比 localhost 测试更接近真实 IPC 部署环境。

---

# 16. 推荐测试步骤

## Step 1

启动 MediaMTX：

```bash
docker run --rm -it \
  --name mediamtx \
  -e MTX_RTSPTRANSPORTS=tcp \
  -p 8554:8554 \
  bluenviron/mediamtx:1
```

---

## Step 2

启动 FFmpeg：

```bash
ffmpeg \
  -re \
  -stream_loop -1 \
  -i blade_clearance_02.mp4 \
  -c copy \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

---

## Step 3

使用 ffplay：

```bash
ffplay \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

验证视频是否正常。

---

## Step 4

使用 OpenCV 验证：

```text
RTSP
→ cv2.VideoCapture()
→ Frame
```

---

## Step 5

启动 TASK-1：

```text
RTSP
→ Frame
→ YOLO26
→ Blade Mask
→ Polygon
→ Original Coordinate
```

---

## Step 6

测试断流和自动重连。

---

## Step 7

记录：

```text
RTSP FPS
Processing FPS
Inference Time
Dropped Frames
Reconnect Count
```

---

# 17. 三终端快速启动方式

## Terminal 1 — RTSP Server

```bash
docker run --rm -it \
  --name mediamtx \
  -e MTX_RTSPTRANSPORTS=tcp \
  -p 8554:8554 \
  bluenviron/mediamtx:1
```

---

## Terminal 2 — MP4 模拟摄像机

```bash
ffmpeg \
  -re \
  -stream_loop -1 \
  -i blade_clearance_02.mp4 \
  -c copy \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

---

## Terminal 3 — RTSP 验证

```bash
ffplay \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/blade
```

---

# 18. 最终推荐方案

当前项目建议采用：

```text
blade_clearance_02.mp4
          ↓
       FFmpeg
          ↓
    MediaMTX RTSP
          ↓
rtsp://127.0.0.1:8554/blade
          ↓
        TASK-1
          ↓
     RTSP Reader
          ↓
       Frame
          ↓
 YOLO26 best.pt
   imgsz = 640
          ↓
     Blade Mask
          ↓
      Polygon
          ↓
 Original Coordinates
```

该方案最大的优势是：

```text
测试环境
      ↓
与真实 IPC 接口形式相同
```

因此未来从：

```text
虚拟 RTSP
```

切换为：

```text
真实机舱 IPC RTSP
```

时，YOLO26 实时处理程序基本不需要修改，只需要替换 RTSP URL。
