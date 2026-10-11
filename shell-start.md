# Shell 脚本启动与使用说明

更新日期：2026-10-10。工作目录：`/home/mich/projects/blade-clearance`。

本文先介绍根目录的 `./start_*.sh`，再介绍其他 `.sh` 脚本，共覆盖 20 个仓库脚本。说明以当前文件内容为准，不把旧注释中的性能、硬件或训练完成情况视为本次验证结果。本次只阅读文件并执行 `bash -n` 语法检查，没有启动推流、训练或删除数据。

## 一、根目录的 `./start_*.sh`

三个入口的职责如下：

| 脚本 | 作用 | 是否做 YOLO 推理 | 运行方式 |
|---|---|---|---|
| `./start_mp4_to_rtsp.sh` | 本地 MP4 → FFmpeg → MediaMTX → RTSP | 否 | 前台 Python 守护进程，默认同时启动服务端 |
| `./start_rtsp_daemon.sh` | 读取 RTSP、重连、按配置保存图像 | 否 | 前台 Python 守护进程，内部线程读帧 |
| `./start_main.sh` | 读取 RTSP 并运行叶片分割、定位和结果输出 | 是 | 预期前台运行，但当前脚本有语法错误 |

`daemon` 一词不代表 Shell 自动后台运行。根目录这两个守护入口通过 `exec` 启动 Python，占用当前终端。主程序已经包含自己的读帧线程，因此运行 YOLO 时，不需要先运行独立的 `start_rtsp_daemon.sh`。

### 1. `./start_mp4_to_rtsp.sh`：启动视频发布端

**调用对象：** `blade_rtsp_system/mp4_to_rtsp_daemon.py`。

脚本先根据自身位置切换到仓库根目录，检查 `uv`，再组装参数并执行 `uv run python ...`。Python 负责启动 FFmpeg，按配置启动 MediaMTX，并在推流进程退出后尝试重启。

```bash
cd /home/mich/projects/blade-clearance
bash ./start_mp4_to_rtsp.sh
```

| 脚本顶部变量 | 默认值 | 说明 |
|---|---|---|
| `INPUT` | `data/training-data/test01-video-2026-08-28_134655_607.mp4` | 本地输入视频 |
| `RTSP_URL` | `rtsp://127.0.0.1:8554/blade` | 发布地址 |
| `WITH_SERVER` | `1` | 同时启动本机 MediaMTX |
| `RE_ENCODE` | `0` | 默认直接复制编码；设为 1 时请求重新编码 |
| `NO_LOOP` | `0` | 默认循环播放 |
| `MAX_RESTARTS` | `0` | 推流失败重启次数不限 |
| `RESTART_BACKOFF` | `2.0` | 重启等待秒数 |
| `MAX_DURATION` | `0.0` | 不限制运行时间 |
| `FFMPEG` / `MEDIAMTX` | 空字符串 | 自动查找二进制，也可填写路径 |
| `FFMPEG_LOG` | `./log/mp4_to_rtsp_ffmpeg.log` | FFmpeg 日志 |
| `MEDIAMTX_LOG` | `./log/mp4_to_rtsp_mediamtx.log` | 服务端日志 |

依赖：`uv`、Python 环境、FFmpeg、输入视频；启用 `WITH_SERVER` 时还需要 MediaMTX。`uv run` 使用项目环境，可能执行依赖同步。该脚本不转发额外命令行参数，也不采用同名环境变量覆盖其顶部赋值；要改变行为，应修改变量，或直接调用 Python 入口并传参。

停止时在终端按 Ctrl+C。Python 清理其 FFmpeg 子进程及由它启动的服务端。默认自带服务端，因此不要再同时启动另一个占用 8554 端口的 MediaMTX。

### 2. `./start_rtsp_daemon.sh`：独立读流与存帧

**调用对象：** `blade_rtsp_system/rtsp_daemon.py`。

脚本自动切换到仓库根目录，检查 `uv`，再启动 RTSP 消费程序。它用于验证读流、重连和图像保存，不加载叶片分割模型。它没有在此入口提供跨进程共享最新帧的网络服务，不能理解为 `main.py` 的必需前置服务。

```bash
cd /home/mich/projects/blade-clearance
bash ./start_rtsp_daemon.sh
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RTSP_URL` | `rtsp://127.0.0.1:8554/blade` | 已运行的 RTSP 源 |
| `RECONNECT_INTERVAL` | `2.0` | 重连间隔秒数 |
| `MAX_RECONNECT_ATTEMPTS` | `0` | 重连次数不限 |
| `STATS_INTERVAL` | `5.0` | 统计记录间隔秒数 |
| `MAX_DURATION` | `0.0` | 不限制时长 |
| `SAVE_DIR` | `./data/rtsp-daemon` | 默认开启 JPEG 存帧；设为空字符串可关闭 |

实际保存的是主循环取到的帧，不能保证每个接收帧都保存。文件名为 `frame_000001.jpg` 这类格式；重复运行可能覆盖同名图像。持续存帧会增加磁盘占用。

Python 的默认日志位置为 `/tmp/rtsp_daemon.log`，应用日志缓存在内存，退出时写入。脚本注释中的“打印统计”不等于每五秒在终端实时输出日志。Ctrl+C 或 SIGTERM 请求正常停止。此脚本同样不转发附加命令行参数。

### 3. `./start_main.sh`：叶片分割主程序入口

**预期调用对象：** `blade_rtsp_system/main.py`，配置为 `blade_rtsp_system/config/config.yaml`。

当前文件只有一行命令，末尾多了一个未配对的反引号。`bash -n start_main.sh` 检查失败，因此目前不能直接使用这个脚本启动。它也没有解释器声明或自动切换工作目录的逻辑。本说明没有修改脚本。

等效的正确启动命令为：

```bash
cd /home/mich/projects/blade-clearance
source .venv/bin/activate
.venv/bin/python ./blade_rtsp_system/main.py \
  --config ./blade_rtsp_system/config/config.yaml
```

需要提前有可读取的 RTSP 视频流、Python 依赖及模型权重。主程序执行读帧、分割、叶片定位、标注绘制和结果缓存；默认输出位于 `blade_rtsp_system/output/`。Ctrl+C 正常停止后保存 CSV、JSONL 和日志。强制杀死进程可能丢失尚未落盘的缓冲结果。

详细代码说明见 [main.py 中文报告](docs/blade_rtsp_system_main_report.md)。

### 4. 本地测试建议顺序

在两个终端分别执行，先发布视频，再运行推理：

**终端 A：**

```bash
cd /home/mich/projects/blade-clearance
bash ./start_mp4_to_rtsp.sh
```

**终端 B：**

```bash
cd /home/mich/projects/blade-clearance
.venv/bin/python ./blade_rtsp_system/main.py \
  --config ./blade_rtsp_system/config/config.yaml
```

如果只想检查读取和存帧，可在终端 B 改用 `bash ./start_rtsp_daemon.sh`。停止时先在终端 B 按 Ctrl+C，等待结果保存，再停止终端 A。

## 二、其他应用启动脚本

### 5. `blade_rtsp_system/start_main.sh`

内容是使用仓库根目录的 `.venv/bin/python` 启动主程序，并附加 `--max-duration 60`：

```bash
cd /home/mich/projects/blade-clearance
bash blade_rtsp_system/start_main.sh
```

虽然脚本放在应用子目录，但内部路径仍以仓库根目录为基准，不能先进入 `blade_rtsp_system` 再按原命令运行。文件没有解释器声明，当前也没有可执行权限，使用 `bash` 调用即可。它不转发其他参数。

60 秒检查发生在处理帧之后；主程序在断流、缓冲为空时可能不会严格按时退出。仍可使用 Ctrl+C 正常停止。

## 三、`scripts/` 下的 RTSP 服务、推流与停止脚本

这一组提供另一套可组合的启动方式。选择同一套配对脚本管理进程，避免与根目录自带服务端的推流脚本重复启动。

### 6. `scripts/start_mediamtx.sh [port]`

启动服务端，不负责发布视频。默认端口 8554，容器名固定为 `mediamtx`，镜像为 `bluenviron/mediamtx:1`。

执行顺序：发现同名容器则复用或启动它；否则优先使用 Docker，以 `docker run --rm -it` 前台启动；Docker 不可用时，回退执行 PATH 中的本机 `mediamtx`。

```bash
bash scripts/start_mediamtx.sh 8554
```

端口参数只用于新建 Docker 容器的端口映射。复用已有容器不会更新其端口；本机回退分支没有把端口参数传给 MediaMTX，因此不能依据终端提示断定本机实际端口。新建 Docker 分支的日志显示在终端，通常用 Ctrl+C 停止；复用已有容器后脚本立即退出，不会在当前终端托管容器。

### 7. `scripts/start_mediamtx_native.sh`

使用 `setsid` 在后台启动本机 MediaMTX，不依赖 Docker。启动后等待一秒，通过进程存活检查判断启动状态，这不等同于视频流已经可读。

```bash
bash scripts/start_mediamtx_native.sh
```

| 项目 | 默认值 |
|---|---|
| 二进制 | `$HOME/.local/bin/mediamtx` |
| 配置 | 脚本所在目录的 `mediamtx.yml` |
| PID 文件 | `/tmp/mediamtx.pid` |
| 日志 | `/tmp/mediamtx.log` |

支持环境变量 `MEDIAMTX_BIN`、`MEDIAMTX_CFG` 覆盖。随附配置只启用 RTSP TCP，监听 `:8554`，允许发布路径，关闭 RTMP、HLS、WebRTC、SRT。停止使用下一项脚本。

### 8. `scripts/stop_mediamtx_native.sh`

读取 `/tmp/mediamtx.pid`，进程存在时发送默认终止信号，等待一秒，再尝试 `kill -9`，最后删除 PID 文件。不管理 Docker 容器或 Python 推流守护进程自带的服务端。

```bash
bash scripts/stop_mediamtx_native.sh
```

脚本只检查 PID 存活，没有验证进程身份。PID 文件过期或 PID 被复用时，应先确认它仍对应目标进程；该限制同样适用于本组其他基于 PID 文件的停止脚本。

### 9. `scripts/start_rtsp_stream.sh [video.mp4] [rtsp_url]`

前台循环发布视频，要求服务端提前启动。默认视频和地址与根目录推流入口相同。先用 `ffprobe` 打印编码、尺寸和帧率，再执行 FFmpeg。

```bash
bash scripts/start_rtsp_stream.sh \
  data/training-data/test01-video-2026-08-28_134655_607.mp4 \
  rtsp://127.0.0.1:8554/blade
```

关键参数：`-re` 按媒体速度读取，`-stream_loop -1` 无限循环，`-c copy` 直接复制编码，`-rtsp_transport tcp` 使用 TCP。日志在终端，Ctrl+C 停止。编码不兼容时不会自动转码，需要选择下一项。

### 10. `scripts/start_rtsp_stream_h264.sh [video.mp4] [rtsp_url]`

参数与默认输入同上，前台运行，需先启动服务端。它解码后使用 `libx264` 重新编码，设置 `veryfast`、`zerolatency`、`yuv420p`，并通过 `-an` 去掉音频。

```bash
bash scripts/start_rtsp_stream_h264.sh
```

适用于直接复制编码不能正常发布的情况。会增加编码计算开销，并可能改变图像质量。日志在终端，Ctrl+C 停止。

### 11. `scripts/start_rtsp_stream_daemon.sh [video.mp4] [rtsp_url]`

使用 `setsid` 后台运行 FFmpeg，采用实时、循环、直接复制编码和 TCP 发布。要求服务端提前启动。默认视频和地址同上。

```bash
bash scripts/start_rtsp_stream_daemon.sh
```

PID 保存于 `/tmp/rtsp_ffmpeg.pid`，日志保存于 `/tmp/rtsp_ffmpeg.log`。脚本等待两秒后检查进程存活；已有活跃 PID 时直接返回，不更新视频或 URL。它只是后台 FFmpeg 包装器，没有崩溃自动重启机制；与根目录 Python 推流守护脚本不同。

### 12. `scripts/stop_rtsp_stream_daemon.sh`

读取 `/tmp/rtsp_ffmpeg.pid`，发送终止信号，等待一秒，再尝试强制停止并删除 PID 文件。

```bash
bash scripts/stop_rtsp_stream_daemon.sh
```

它只对应上一项后台推流脚本，不停止 MediaMTX，也不停止主程序。

### 13. `scripts/start_rtsp_test.sh [video.mp4]`

一键启动 Docker MediaMTX 和后台 FFmpeg。视频参数可选，默认视频同上；端口固定为 8554，发布地址固定为 `rtsp://127.0.0.1:8554/blade`。

```bash
cd /home/mich/projects/blade-clearance
bash scripts/start_rtsp_test.sh
```

执行流程：检查视频；复用运行中的 `mediamtx` 容器，或移除同名旧容器再后台创建；使用 `nohup` 启动循环推流；等待四秒，再用 `ffprobe` 探测。探测失败只打印警告，最终退出状态不能单独证明链路就绪。

依赖 Docker、FFmpeg 和 ffprobe，没有本机服务端回退。PID 文件 `.rtsp_ffmpeg.pid`、日志 `.rtsp_ffmpeg.log` 位于调用时的工作目录，**与 `/tmp/rtsp_ffmpeg.*` 不是同一套文件**。

### 14. `scripts/stop_rtsp_test.sh`

从当前目录读取 `.rtsp_ffmpeg.pid` 并终止 FFmpeg，删除 PID 文件，然后执行 `docker rm -f mediamtx` 强制移除同名容器。

```bash
cd /home/mich/projects/blade-clearance
bash scripts/stop_rtsp_test.sh
```

启动、停止必须在同一工作目录执行。这个脚本会移除名为 `mediamtx` 的容器，即使它最初由其他入口创建或被其他任务复用；它不区分容器所有权。

### 两套后台管理方式

| 方案 | 启动顺序 | 停止顺序 |
|---|---|---|
| 原生后台 | `start_mediamtx_native.sh` → `start_rtsp_stream_daemon.sh` → 推理程序 | 停推理 → `stop_rtsp_stream_daemon.sh` → `stop_mediamtx_native.sh` |
| Docker 一键测试 | `start_rtsp_test.sh` → 推理程序 | 停推理 → `stop_rtsp_test.sh` |

此表中的 RTSP 脚本均位于 `scripts/`。上述方案是根目录 `start_mp4_to_rtsp.sh` 的替代组合，不应在相同端口上重复叠加。

## 四、数据集生成、训练与训练流水线

以下脚本会生成数据或启动训练，其中部分会删除旧输出。它们不属于普通 RTSP 推理启动步骤。除特别说明外，相对路径要求从仓库根目录执行。

### 15. `run_train_only.sh`

调用 `uv run ./yolo26/train_from_videos.py --skip-train`，配置为 `yolo26/train-0914/train_config.yaml`。尽管名称包含 `train_only`，实际显式跳过训练，仅生成自动标注数据集。输出由该配置和 Python 程序决定。无位置参数转发，运行信息输出到终端。

### 16. `run_train_seg.sh`

生成分割标注数据集，不训练模型。先从 `yolo26/train-0915/train_config.yaml` 读取 `dataset.output` 和 `train.imgsz`，再删除并重建目标目录，调用 `yolo26/train_from_videos_seg.py --skip-train`。

审阅时配置为输出 `train-0915-640/`、尺寸 **1024**。目录名中的 640 不代表实际生成尺寸。

脚本使用 `rm -rf` 清除整个目标目录。它拒绝空路径、根目录及当前工作目录，但没有限制输出必须位于仓库内，也没有保护任意其他目录。确认配置目标确实是可重建的数据集后再使用。依赖 `uv`、YAML 解析库和 `realpath`。

### 17. `run_yolo26n-seg-draft.sh`

调用 `yolo26n-seg/train_blade_yolo26_semisupervised.py` 执行半监督训练。先检查标注训练图像、标签目录和未标注目录存在，然后**删除整个旧工作目录**再训练。

| 参数 | 当前值 |
|---|---|
| 标注集 `LABELED_ROOT` | `train-0915-640` |
| 未标注集 `UNLABELED_DIR` | `unlabeled-0915-640-e1` |
| 工作目录 `WORK_DIR` | `train-seg-draft-0916` |
| 图像尺寸 / 批量 | 640 / 80 |
| 数据加载进程数 | 0 |
| 第一阶段 / 第二阶段轮数 | 200 / 150 |
| 伪标签置信度 | 0.30 |
| 最小掩膜面积比例 | 0.00075 |
| 设备 | `1`，即请求编号 1 的设备 |
| 模型配置 | `yolo26n-seg/26/yolo26-seg.yaml` |

没有命令行参数转发，参数由顶部变量决定。需要满足数据、训练依赖和设备条件，不能将此前 CPU 推理成功视为训练环境就绪。

外层脚本预期的第二阶段权重路径为 `train-seg-draft-0916/training/stage2_auto_labeled_plus_pseudo/weights/best.pt`，伪标签审计为工作目录下的 `pseudo_label_audit.csv`。这些是预期产物路径，不表示本次已运行或验证完成。

### 18. `run_1024_pipeline.sh`

预期依次执行：

1. 调用 `run_train_seg.sh` 生成标注数据。
2. 调用 `create_unlabeled_dataset.py`，明确指定 `--imgsz 1024 --every 1`，输出 `unlabeled-0915-1024-e1`。
3. 调用 `run_yolo26n-seg-draft.sh` 训练。

日志覆盖写入 `run_1024_pipeline.log`。该脚本硬编码 `cd /home/mich/project/blade-clearance`，与当前目录 `/home/mich/projects/blade-clearance` 不同。它没有 `set -e` 或单独检查 `cd`，目录切换失败后仍可能在原工作目录继续执行。

它也没有将下游训练脚本切换为 1024。当前下游仍为 640、batch 80、读取 640 未标注目录，因此不能凭该脚本名称认定整条流程都采用 1024。

步骤失败分支使用 `exit 1`，会直接退出脚本；下方基于 `RC` 的失败总结可能不会执行。输出中的“第 RC 步”也不能正确表达失败步骤编号。

### 19. `run_640_pipeline.sh`

用于生成数据并运行 640 训练，但包含持久修改源码的操作：

- 用 `sed -i` 将 `yolo26/train_from_videos_seg.py` 的 `WRITE_IMG_SIZE` 改为 640。
- 将 `run_yolo26n-seg-draft.sh` 的尺寸、批量、未标注目录改为 640、80、`unlabeled-0915-640-e1`。
- 调用 `run_train_seg.sh`；若未标注目录和 `manifest.json` 同时存在则跳过生成，否则以 `--imgsz 640 --every 8` 生成未标注数据。
- 调用训练脚本，日志覆盖写入 `run_640_pipeline.log`。

它同样硬编码旧的 `project` 路径；由于开启 `set -e`，切换目录失败会立即退出。修改不会自动恢复。注释或目录名称中的 `every=1` 与实际命令 `--every 8` 不一致。

更重要的是，`run_train_seg.sh` 显式传入 YAML 中的 `train.imgsz`，审阅时该值为 1024，因此仅改 Python 默认常量不能保证标注数据变成 640。重新执行 `run_1024_pipeline.sh` 也不会自动恢复该脚本修改的训练参数。失败时也可能在最终总结之前退出。

### 20. `resume_1024_train.sh`

预期在已有数据基础上重新启动训练。它检查标注集及 `unlabeled-0915-1024-e1`，显示 GPU 信息，再执行 `bash run_yolo26n-seg-draft.sh`，通过 `tee -a` 追加日志到 `resume_1024_train.log`。

当前实现有以下限制：

- 硬编码旧的 `/home/mich/project/blade-clearance`，且开启 `set -e`，路径不存在就退出。
- 名称和提示称“1024、batch 16”，实际没有给训练脚本设置这些值；当前下游仍使用 640、batch 80。
- 它检查的是 1024 未标注目录，而下游当前读取 640 未标注目录。
- 它没有传递检查点恢复参数；下游会先删除工作目录。因此这里的“恢复”应理解为重新发起流程，不能视为从上次训练检查点续训。
- 即使 PATH 中找到 `nvidia-smi`，仍固定调用 `/usr/lib/wsl/lib/nvidia-smi`，非 WSL 环境可能失败。
- 在 `set -euo pipefail` 下，训练管道失败可能直接结束脚本，导致后续读取 `PIPESTATUS` 和失败总结没有机会执行。

脚本顶部涉及特定历史显存故障、WSL 重启和硬件容量的注释，只是历史操作背景，不是当前机器状态。本说明没有执行其中的重启操作。

## 五、环境变量辅助脚本

### 21. `configs/export-env.sh`

设置并导出 `CAMERA_USER`、`CAMERA_PASSWORD`，供读取这些变量的程序使用。应在当前终端加载：

```bash
cd /home/mich/projects/blade-clearance
source configs/export-env.sh
```

如果使用 `bash configs/export-env.sh`，变量仅存在于子进程，不会保留到父终端。该文件包含明文凭据，此处不重复列出值。它本身不启动相机或 RTSP 服务；上述使用固定本地 URL 的入口也不会自动将这些变量拼接到地址中。

## 六、检查结果与使用前核对

本次对 20 个脚本执行了 `bash -n`：19 个通过，根目录 `start_main.sh` 因末尾反引号未配对而失败。语法通过不代表依赖、路径、端口、数据或设备满足运行条件。

| 核对项 | 当前结论 |
|---|---|
| 普通本地推理 | 先运行根目录推流入口，再使用本文正确 Python 命令 |
| 根目录 `start_main.sh` | 需要去掉多余反引号后才能作为入口使用 |
| 旧训练绝对路径 | 三个流水线／恢复脚本使用 `project`，需与实际 `projects` 路径一致 |
| 640 / 1024 一致性 | YAML、生成脚本、训练脚本的参数和目录名称存在差异，需逐项核对 |
| 停止脚本配对 | `/tmp` PID、本地隐藏 PID、Docker 容器三类管理方式不能混用 |
| 数据保留 | 数据集生成和训练入口可能删除旧输出；640 流水线还会修改源码 |
| 运行验证范围 | 本次仅静态阅读和 Shell 语法检查，未实际运行这些脚本 |

编号中第 4 项是启动顺序说明，不是脚本，因此共介绍 20 个脚本。应用主程序的逐行与流程说明另见 [main.py 中文报告](docs/blade_rtsp_system_main_report.md)。
