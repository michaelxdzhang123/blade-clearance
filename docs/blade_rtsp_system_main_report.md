# Blade RTSP main.py：逐行说明与工作流程报告

日期：2026-10-10  
源文件：`blade_rtsp_system/main.py`，共 426 行。实际目录名称使用下划线。  
范围：说明原报告所审阅的工作区源代码及其本地依赖。本次仅翻译报告，不修改应用代码或重新启动程序。“逐行说明”按相关语句、续行、注释和空行分组，文末保留完整的带行号源代码快照。

## 1. 程序用途与功能边界

该入口程序读取视频，运行 YOLO 分割模型，筛选类别名称为 `blade` 的目标，计算图像中的几何信息，并将结构化结果缓存在内存中，在退出时保存。它负责协调各模块，神经网络和轮廓算法由其他模块实现。

输出是**以像素为单位的二维图像测量结果**。程序不计算叶片与塔筒之间的物理净空距离，不进行相机标定、深度估计或跨帧身份跟踪。源代码将模型称为 YOLO26；本报告没有独立确认权重的模型结构，也没有评估预测准确率。

## 2. 整体工作流程

```mermaid
flowchart TD
    A[解析命令行并读取 YAML] --> B[配置内存缓冲日志]
    B --> C[解析权重路径并一次性加载检测器]
    C --> D[准备结果缓冲及处理参数]
    D --> E{打开视频源}
    E -->|失败| F[关闭结果写入器、保存日志、返回 1]
    E -->|成功| G[启动读帧线程]
    G --> H[读取图像帧和时间戳]
    H -->|读帧失败| I[重连直到成功或达到上限]
    I --> H
    H -->|成功| J[按配置缩放图像]
    J --> K[单槽最新帧缓冲区]
    K --> L[主线程取出最新帧]
    L --> M[按配置裁剪 ROI]
    M --> N[YOLO 分割推理]
    N --> O[提取全部叶片实例]
    O --> P[计算轮廓、多边形、包围框、中心及叶尖候选点]
    P --> Q[绘制标注并按配置显示窗口]
    Q --> R[缓存帧结果并更新统计]
    R -->|继续| L
    R -->|达到时限或收到中断| S[停止读帧并释放视频源]
    S --> T[保存 CSV 和 JSONL、关闭窗口、保存日志]
```

启动后，读帧与处理并发运行。图中的 ROI 阶段表示预期的处理分支；当前实现使用浮点数作为切片起始索引，启用后会失败，详见第 7 节。

### 启动阶段

1. 解析 `--config`、`--max-duration` 和 `--resize`，对应第 133–140 行。
2. 在配置日志之前读取 YAML，第 142 行。模型和输出路径以 `main.py` 所在目录为基准解析，第 146–155 行。
3. 检查权重是否存在，构造检测器并只加载一次模型，第 159–180 行。
4. 准备输出缓冲区、ROI 和缩放参数，第 183–214 行。
5. 打开配置的视频源。首次打开失败会立即退出，不进入重连循环，第 217–235 行。
6. 创建帧缓冲区，启动读帧线程，准备可视化，确定坐标尺寸并注册 SIGTERM 处理函数，第 238–297 行。

### 读帧线程

`RTSPReader.read()` 返回 BGR 图像、成功标志和解码后记录的 `time.time()` 时间戳。这个时间戳表示主机接收、解码完成后的时间，并非相机曝光时间或媒体显示时间。每成功读取一帧，读取器递增 `frame_id`。线程按配置缩放图像后，将其放入带锁保护的单槽缓冲区。

新帧到达时，尚未被取走的旧帧会被覆盖。这样可以避免推理队列不断积压，但当处理速度低于接收速度时，会主动丢弃中间帧。`processed_frames` 统计从缓冲区取出的帧数，不等同于成功保存的结果数。

读帧失败后，线程尝试重连：释放视频采集对象，等待配置的间隔，再重新打开。重连上限为零表示不限次数。重连计数在整个运行期间累计，不会在每次成功重连后清零。

### 主线程处理

主线程轮询缓冲区；缓冲区为空时休眠 10 毫秒。每取到一帧，就启动处理计时，按配置裁剪 ROI，调用检测器，提取全部 `blade` 实例，计算定位几何信息，绘制标注，记录处理耗时并缓存结果。

未检测到叶片也是有效结果：程序继续运行，输出叶片数量为零的帧记录。检测到多个叶片时全部保留。分割提取模块会计算主要实例编号，但入口程序没有将这一选择写入最终帧记录。

### 退出阶段

Ctrl+C 被捕获为 `KeyboardInterrupt`。安装信号处理函数后，SIGTERM 也会转换成同一异常。`finally` 清除运行标志，最多等待读帧线程五秒，释放视频源，保存结果，关闭可视化，计算最终统计并保存日志。主循环正常结束时返回零。受保护的处理循环内若出现未捕获异常，会先执行清理，再向外传播异常。

## 3. 原报告审阅时的配置与启动方法

| 配置项 | 配置值 | 含义 |
|---|---|---|
| 模型 | `models/best.pt` | 相对于 `blade_rtsp_system` 目录 |
| 设备 | `cpu` | 明确使用 CPU 推理 |
| `imgsz` | 640 | 传给模型的推理尺寸，与输入图像缩放不同 |
| 置信度 | 0.50 | 检测器筛选阈值 |
| 高分辨率掩膜 | `retina_masks: true` | 向检测器请求该选项 |
| 视频流 | `rtsp://127.0.0.1:8554/blade` | 服务端和推流端需要提前运行 |
| 重连 | 间隔 2 秒，不限次数 | 已建立的连接中断后生效 |
| 输入缩放 | 640 × 640 | 读帧线程在推理之前缩放图像 |
| ROI | 关闭 | 未进入当前存在错误的裁剪分支 |
| CSV / JSONL | 均开启 | 缓存至关闭时保存 |
| 显示窗口 | 关闭 | 仍然绘制标注，但不显示窗口 |

在仓库根目录执行：

```bash
cd /home/mich/projects/blade-clearance
source .venv/bin/activate
.venv/bin/python ./blade_rtsp_system/main.py \
  --config ./blade_rtsp_system/config/config.yaml
```

需要限定运行时长时，可追加 `--max-duration 60`。该限制在处理帧后检查，**不能保证断流期间严格超时退出**。命令行 `--resize W H` 优先于 YAML 缩放配置。若要禁用缩放，应在所选配置文件中设置 `processing.resize: null`；命令行没有专门的禁用选项。

YAML 文件本身的路径以终端工作目录为基准。YAML 内部的模型和输出路径则以应用目录为基准。此前出现的配置文件找不到错误，正是这两种路径基准不同造成的。

本文件不启动 MediaMTX，也不负责将 MP4 推送为视频流。这些功能由独立进程提供，例如仓库中的 `mp4_to_rtsp_daemon.py`。

## 4. 数据与坐标传递

| 数据对象 | 生产方 → 使用方 | 内容与含义 |
|---|---|---|
| `frame, ts, fid` | 读取器 → 缓冲区 → 主线程 | BGR 像素、主机时间戳、读取帧序号 |
| `result, inf_ms` | 检测器 → 主线程 | 首个模型结果，以及 `model.predict()` 调用耗时 |
| `detection` | 分割提取模块 → 主线程 | 检测标志、数量和全部匹配叶片实例 |
| `located` | 定位模块 → 主线程 | 包围框、框中心、多边形、面积及叶尖候选点 |
| `fr` | `build_frame_result` → 写入器 | 帧元数据与定位后的叶片列表 |

对于宽 `W0`、高 `H0` 的源图像，可选缩放生成尺寸为 `Wr × Hr` 的工作图像。按审阅时的配置，输出尺寸和坐标对应 **640 × 640**，而非原始视频流分辨率。流程中没有反向缩放操作。

ROI 的预期映射是在工作图像坐标系中进行平移：

```text
x_working = x_roi + roi_x1
y_working = y_roi + roi_y1
```

如果需要在缩放后恢复源图像像素坐标，还需要执行 `x_source = x_working * W0/Wr` 和 `y_source = y_working * H0/Hr`。这两式描述的是当前缺少的操作。源图像不是正方形时，非等比例缩放还会改变图像的宽高比。

定位模块对原始张量掩膜进行阈值化，选取最大的外轮廓，再分别生成未简化多边形和简化多边形。因此，“原始多边形”指所选轮廓未简化的顶点，并不保留所有不连通区域或孔洞。简化容差为轮廓周长的 1%。掩膜面积统计有效掩膜像素数，不一定等于所选轮廓的几何面积。

中心点是包围框的中点，不是掩膜的形心。`blade_tip_candidate` 选择图像中最上方的轮廓点，即 y 最小的点，并对其 x 坐标求平均。这是依赖叶片朝向的图像启发式方法，并非经过验证的物理叶尖。`instance_id` 是单帧结果中的索引，不是跨帧跟踪编号。

定位模块只对多边形进行平移，不会将原始掩膜缩放到预测输入尺寸；传入的帧宽、高参数在其内部也未使用。因此，坐标是否对齐取决于掩膜分辨率，修改 `retina_masks` 时需要检查这一点。

### 保存的文件

| 文件类型 | 在 `blade_rtsp_system/` 下的默认路径 | 数据组织 |
|---|---|---|
| CSV | `output/csv/blade_positions.csv` | 每个叶片一行；无叶片帧也写一行，实例字段为空 |
| JSONL | `output/jsonl/blade_positions.jsonl` | 每个保存的帧一个 JSON 对象，包含多边形列表 |
| 日志 | `output/logs/runtime.log` | 应用日志先缓存在内存，退出时写入 |

帧级 JSON 字段为：`frame_id`、`timestamp`、`frame_width`、`frame_height`、`blade_detected`、`blade_count`、`loop_ms`、`blades`。

每个叶片包含：`instance_id`、`class_id`、`class_name`、`confidence`、`bbox`、`center`、`mask_area`、`raw_polygon`、`simple_polygon`、`raw_node_count`、`simple_node_count`、`blade_tip_candidate`。

CSV 不包含多边形和叶尖候选点。结果和日志均以覆盖写入模式保存，因此刷新或关闭时会覆盖上一次文件。新任务运行期间，已有文件可能仍显示旧数据。缓存占用随运行增长；强制终止或清理前失败可能导致数据丢失。

## 5. 逐行说明

行号对应文末标识的源代码快照。连续行按一个逻辑操作分组，空行与说明性注释并入相邻代码块。

| 行号 | 说明 |
|---|---|
| 1–18 | 解释器与编码声明，以及模块说明文档，概述预期流程、线程和日志策略。 |
| 19 | 推迟类型注解的求值；该语句本身并不实现运行时联合类型支持。 |
| 20–28 | 导入命令行解析、日志、路径、信号、进程退出、线程和计时模块。 |
| 29–31 | 导入 OpenCV 和 YAML 解析模块。 |
| 32–33 | 将应用目录插入 Python 模块搜索路径的开头。 |
| 34 | 导入叶片筛选与定位模块。 |
| 35 | 导入 `map_mask`，但本文件未使用，当前定位模块也没有调用它。 |
| 36–40 | 导入缓冲区、结果写入器、读取器、可视化和检测器类。 |
| 41–46 | 创建模块日志对象，定义带时间戳的公共日志格式。 |
| 47–57 | 定义内存日志处理器，并说明预期的锁保护行为。 |
| 58–62 | 初始化父类处理器、格式化器及用于保存格式化日志的空列表。 |
| 63–66 | 格式化每条日志记录并追加到内存列表。 |
| 67–70 | 定义落盘方法并创建目标目录。 |
| 71–75 | 将缓存日志覆盖写入文件，补充末尾换行，然后清空内存列表。 |
| 76–83 | 定义日志配置函数，说明输出目录约定。 |
| 84–87 | 创建日志目录和处理器，配置 INFO 级别根日志并返回处理器。`basicConfig` 默认不会替换已有的根日志处理器。 |
| 88–93 | 定义 YAML 加载函数，打开传入路径并调用 `safe_load`，没有执行配置结构校验。 |
| 94–101 | 定义路径解析函数，说明应用目录相对路径规则。 |
| 102–104 | 绝对路径直接返回；相对路径与给定基准目录拼接。 |
| 105–115 | 定义帧结果构造函数，说明参数及处理耗时含义。 |
| 116–120 | 创建结果字典，填入帧编号、时间戳和图像宽高。 |
| 121–125 | 加入检测标志、叶片数量、处理耗时和已定位叶片列表。 |
| 126–132 | 定义主函数，并说明常规退出码。 |
| 133–134 | 创建参数解析器及配置文件选项，默认路径相对于工作目录。 |
| 135–136 | 添加浮点数类型的最大运行时长参数，零表示不限制。 |
| 137–139 | 添加两个整数缩放参数 `(W,H)`，未检查尺寸是否为正。 |
| 140–142 | 解析命令行参数，在日志配置之前读取配置文件。 |
| 143–146 | 根据 `__file__` 确定应用目录的绝对路径。 |
| 147–150 | 读取输出配置，缺省为空字典。 |
| 151–155 | 解析日志目录，确定 `runtime.log` 路径并配置内存缓冲日志。 |
| 156–158 | 记录程序启动，进入一次性模型初始化阶段。 |
| 159–160 | 读取必需的模型配置并解析权重路径。 |
| 161–164 | 权重不存在时记录错误，保存日志并返回 1。 |
| 165–167 | 记录权重路径。注释写固定 640，但后续代码实际读取配置值。 |
| 168–174 | 使用权重路径、图像尺寸、置信度、设备和掩膜参数构造检测器。 |
| 175–176 | 在异常保护块中加载模型。 |
| 177–180 | 捕获模型加载异常，保存错误日志并返回 1。 |
| 181–187 | 读取输出开关，选用配置或默认的 CSV、JSONL 路径。 |
| 188–191 | 初始不创建写入器，任一输出开关为真时才启用。 |
| 192–196 | 将两种路径都传给写入器，准备目录和缓冲区，此时尚未打开结果文件。 |
| 197–201 | 读取处理配置、ROI 开关和 ROI 参数字典。 |
| 202–206 | 将 ROI 起点坐标转换为浮点数并组成偏移元组；后续 NumPy 切片会因此出错。 |
| 207–209 | 说明缩放用途及命令行优先于 YAML 的规则。 |
| 210–214 | 优先使用命令行尺寸，否则转换配置中的尺寸，未配置时禁用缩放。 |
| 215–219 | 读取视频配置并选择地址或默认地址。主程序面向 RTSP，但未校验地址协议。 |
| 220–223 | 临时构造读取器，对日志中的视频地址凭据进行脱敏。 |
| 224–229 | 使用视频地址和重连参数构造实际读取器。 |
| 230–235 | 首次打开失败时关闭写入器、保存日志并返回 1；关闭可能以空结果覆盖旧文件。 |
| 236–239 | 根据最大尺寸参数创建帧缓冲区，但辅助类实际只保存一帧。 |
| 240–243 | 创建并设置线程共享的运行事件。 |
| 244–247 | 定义读帧循环，在运行事件保持设置时重复执行。 |
| 248–249 | 读取一帧，并在失败时进入异常分支。 |
| 250–255 | 运行期间持续尝试重连；次数耗尽时记录错误，清除事件并返回。 |
| 256–258 | 尝试重连；成功后跳出重试循环，恢复正常读帧。 |
| 259–263 | 按需缩放成功读取的图像，并连同时间戳、编号写入缓冲区。 |
| 264–265 | 正常退出读帧循环后清除运行事件；未捕获的线程异常会跳过这一步。 |
| 266–269 | 创建并启动后台读帧线程，记录线程启动日志。 |
| 270–275 | 读取显示选项；不存在 `DISPLAY` 环境变量时关闭显示。 |
| 276 | 创建可视化对象；关闭窗口后仍会执行标注绘制。 |
| 277–283 | 从视频采集对象读取源图像尺寸；采集对象不存在时使用零。 |
| 284–285 | 若启用缩放，则以目标尺寸替换宽高，使输出坐标基于缩放图像。 |
| 286–291 | 创建耗时样本列表，记录运行起点及上次统计的系统时间。 |
| 292–297 | 安装 SIGTERM 处理函数，通过抛出 `KeyboardInterrupt` 实现处理循环的正常清理退出。 |
| 298–301 | 进入受异常保护的主循环，运行标志有效时继续处理。 |
| 302–306 | 取出最新帧；缓冲区为空时休眠 10 毫秒，再重新尝试。 |
| 307–309 | 断言已取出的帧具有帧编号。 |
| 310–312 | 取帧后启动单帧高精度处理计时。 |
| 313–317 | 启用 ROI 时，以整数读取裁剪区域的右、下边界。 |
| 318–319 | 执行数组裁剪并将其作为推理输入；浮点起点索引会触发 `TypeError`。 |
| 320–321 | 未启用 ROI 时使用完整工作图像。 |
| 322–325 | 执行预测，并将检测器调用耗时加入当前统计周期。 |
| 326–330 | 提取类别名称严格等于 `blade` 的模型实例。 |
| 331–333 | 选择 ROI 偏移量，或使用零偏移。 |
| 334–338 | 计算全部所选实例的几何信息；虽然传入宽高，当前辅助函数内部未使用。 |
| 339–346 | 帧编号大于一时，以总运行时长的倒数作为显示 FPS；该吞吐率公式不正确。 |
| 347–348 | 在图像副本上绘制标注，并按配置显示。 |
| 349–352 | 记录截至可视化完成的处理耗时，不包含结果写入器的工作。 |
| 353–357 | 构造帧结果字典；启用输出时，将序列化数据及表格数据追加到内存。 |
| 358–361 | 有帧处理时，每隔约五秒读取一次累计缓冲区计数。 |
| 362–367 | 计算当前统计周期的平均、最大推理耗时；没有样本时记零。 |
| 368–373 | 计算同一周期的平均、最大主循环耗时。 |
| 374–377 | 追加统计日志；该处理器只缓存在内存，不实时打印。 |
| 378–380 | 清空耗时样本并更新上次统计时间。 |
| 381–385 | 处理完一帧后检查正数时长限制，达到时限则退出循环。 |
| 386–388 | 捕获 Ctrl+C 或转换后的 SIGTERM，记录中断。 |
| 389–393 | 进入清理：清除运行标志，最多等待读帧线程五秒，释放视频采集对象。 |
| 394–396 | 保存已启用的缓冲输出，并关闭显示资源。 |
| 397–402 | 获取最终累计计数，仅对剩余推理样本求平均。 |
| 403–406 | 仅对剩余主循环耗时样本求平均；没有样本时记零。 |
| 407–412 | 计算总运行时长，以累计取出帧数除以时长得到最终平均 FPS。 |
| 413–418 | 记录最终计数、剩余周期耗时、平均 FPS、重连次数和运行时长。 |
| 419–420 | 记录退出，并将全部缓冲日志写入文件。 |
| 421–422 | 主循环正常结束后返回零，重连次数耗尽也包含在内。 |
| 423–426 | 直接执行本脚本时调用 main，并将返回值作为进程退出码。 |

## 6. 10 月 9 日已保存运行记录的证据

原报告审阅的 `output/logs/runtime.log` 以正常中断及结果保存记录结束。以下数据来自该日志，不是本次新增的性能测试：

| 指标 | 记录值 |
|---|---|
| 进入缓冲区的帧数 | 12,743 |
| 取出待处理的帧数 | 1,886 |
| 取出前被覆盖的帧数 | 10,856 |
| 保存到 JSONL 的帧数 | 1,885 |
| CSV 实例或无检测结果行数 | 1,935 |
| 最终平均 FPS | 3.7 |
| 运行时长 | 509.8 秒 |
| 重连次数 | 0 |
| 最后报告的平均推理／循环耗时 | 500.5／504.2 毫秒 |

保存帧数比取出帧数少一帧，这与“取帧后、写入前收到中断”的情况一致，但日志不能证明具体中断位置。另有 `received - retrieved - dropped = 1`，与退出时仍有一帧未取出的情况一致。部分帧包含多个叶片，因此 CSV 行数大于 JSONL 行数。

最终耗时均值只涵盖上一次周期清零后剩余的样本，不代表整个运行过程；FPS 则使用累计取帧数。因此，不应将这些数值都理解为同一统计范围下的全程性能指标。

## 7. 实现问题及实际影响

| 问题 | 代码依据 | 影响及建议修正方式 |
|---|---|---|
| ROI 索引为浮点数 | main 第 204–205、318 行 | 启用 ROI 会出现切片错误。应使用经过校验的整数边界，并从实际边界计算偏移量。当前右、下边界为零的配置也需改为有效的非空区域。 |
| 缩放坐标被称为原图坐标 | main 第 261–262、284–285、334–338 行 | 当前输出使用缩放后的像素。若需要源图像坐标，应保留源尺寸并明确执行反向缩放，同时统一 ROI 的坐标基准。 |
| 输出开关不能独立选择格式 | main 第 183–196 行；写入器 `write/close` | 任一开关为真时都会缓存并保存两种文件。应将独立开关或可选路径传入写入器。 |
| 显示 FPS 公式错误 | main 第 343–344 行 | `1 / 总运行时长` 会不断趋近零。应采用处理帧数除以时长，或明确定义的滑动窗口 FPS。 |
| 空缓冲等待期间不检查时限 | main 第 304–306、383 行 | 断流并无限重连时，运行可能无限超出设定时长。应在等待之前基于单调时钟检查截止时间。 |
| 最终均值只覆盖剩余周期 | main 第 378–379、399–406 行 | 若需要全程均值，应另行维护累计耗时总和与样本数。 |
| 内存增长与数据持久化风险 | 主程序日志处理器；结果写入器 | 全部结果和日志在关闭之前驻留内存。周期落盘及每次运行独立目录可减少丢失和误覆盖。 |
| 读帧异常与清理存在缺口 | main 第 245–265、292–299、392–393 行 | 未捕获的读帧异常可能使运行标志保持有效，主循环一直等待。限时 join 不能证明线程已经退出，释放采集对象可能与阻塞中的读帧重叠。进入受保护循环之前的启动阶段也未完整覆盖清理。 |
| 重连次数耗尽仍返回成功 | main 第 251–255、422 行 | 视频流失败可能产生零退出码。应记录终止原因，必要时返回明确的失败码。 |
| 设备回退记录与实际调用不一致 | 检测器 `_resolve_device` 与 `predict` | 辅助函数记录回退至 CPU，但预测仍接收原始 `self.device`。当前显式配置 CPU，未进入该分支。 |
| 多个配置项未生效 | 主程序及辅助模块实现 | 此处未使用 `processing.vid_stride`、`buffer.strategy`、`output.save_video`、`debug.verbose`；最大缓冲尺寸参数也不改变单槽存储。不能据此认为对应功能已启用。 |
| 空多边形映射缺少保护 | 定位模块 → 映射模块 `map_polygon` | 空多边形转换成 `(1,0)` 数组后与二维偏移向量相加，可能产生广播错误。应在平移之前直接返回空列表。 |
| 配置校验与异常覆盖有限 | main 第 90–93、142–159、210–214 行 | 安全解析 YAML 不等于校验配置结构。缺失字段、类型错误、非法尺寸及初始化错误，可能在缓冲清理之前直接抛出。 |

这些结论来自代码检查，本报告未实施修复，也没有新增精度、模型质量、ROI、重连或 GPU 验证。

## 8. 查阅的配套文件

以下路径均相对于仓库根目录：

- `blade_rtsp_system/main.py`
- `blade_rtsp_system/config/config.yaml`
- `blade_rtsp_system/src/rtsp_reader.py`
- `blade_rtsp_system/src/frame_buffer.py`
- `blade_rtsp_system/src/yolo_detector.py`
- `blade_rtsp_system/src/blade_segmenter.py`
- `blade_rtsp_system/src/blade_locator.py`
- `blade_rtsp_system/src/mask_processor.py`
- `blade_rtsp_system/src/coordinate_mapper.py`
- `blade_rtsp_system/src/result_writer.py`
- `blade_rtsp_system/src/visualizer.py`
- `blade_rtsp_system/output/logs/runtime.log`

旧工作流文档描述的是 411 行版本。本报告依据原审阅时的 426 行工作区文件，包括其中尚未提交的修改。

## 附录：完整带行号源代码快照

源代码 SHA-256：`28d71d6203c9a887991f32b06cad8ab8509e9db471e0272d3d5dda156f185fea`

以下数字前缀是报告中的行号标签，不属于可执行 Python 代码。

```text
001 | #!/usr/bin/env python
002 | # -*- coding: utf-8 -*-
003 | """
004 | main.py — TASK-1 主入口 (TASK-1 第 28 节工作流)
005 | 
006 |   RTSP → Frame → ROI → YOLO26 best.pt(640) → Blade Mask
007 |        → Raw/Simple Polygon → Original Coordinate → CSV/JSONL/Visualization
008 | 
009 | 用法:
010 |   python main.py --config config/config.yaml
011 | 
012 | 多线程设计 (TASK-1 第 22 节):
013 |   Thread 1: RTSP Reader (读帧 → latest-frame buffer)
014 |   Thread 2 (主线程): YOLO Inference + 定位 + 输出
015 | 
016 | 日志策略 (与 rtsp_daemon.py 一致):
017 |   日志运行时缓冲到内存 list (零 I/O), 进程退出时一次性写 output/logs/runtime.log。
018 | """
019 | from __future__ import annotations  # 启用「推迟求值注解」, 支持 `str | int` 写法
020 | 
021 | import argparse                        # 解析命令行参数 (--config / --max-duration)
022 | import logging                         # 日志 (内存缓冲, 退出时落盘)
023 | import os                              # 路径拼接 / 文件是否存在 / 环境变量
024 | import signal                          # 信号处理 (SIGTERM → 优雅退出)
025 | import sys                             # sys.path / sys.exit
026 | import threading                       # 多线程 (Reader 线程 + Event 控制)
027 | import time                            # 计时 (推理耗时 / FPS / 定时长跑)
028 | 
029 | import cv2                             # OpenCV (取视频分辨率 CAP_PROP_*)
030 | import yaml                            # 解析 config.yaml
031 | 
032 | sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 把本目录加入模块搜索路径, 保证 `from src import` 可用
033 | 
034 | from src import blade_locator, blade_segmenter  # noqa: E402  # 叶片提取 + 定位 (业务核心)
035 | from src.coordinate_mapper import map_mask  # noqa: E402     # ROI→原图坐标映射 (本文件未直接调用, 由 locator 内部使用)
036 | from src.frame_buffer import FrameBuffer  # noqa: E402       # 最新帧缓冲 (Reader↔主线程交接)
037 | from src.result_writer import ResultWriter  # noqa: E402     # 结果写出 CSV/JSONL
038 | from src.rtsp_reader import RTSPReader  # noqa: E402         # RTSP 读帧 + 断流重连
039 | from src.visualizer import Visualizer  # noqa: E402          # 实时窗口可视化
040 | from src.yolo_detector import YOLODetector  # noqa: E402     # YOLO 模型加载 + 单帧推理
041 | 
042 | logger = logging.getLogger("blade_rtsp.main")  # 本模块的 logger 实例
043 | 
044 | LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"  # 与 rtsp_daemon.py 一致
045 | 
046 | 
047 | class BufferedFileHandler(logging.Handler):
048 |     """把日志记录缓冲到内存 list, 运行时零 I/O, 结束时一次性写文件。
049 | 
050 |     与 blade_rtsp_system/rtsp_daemon.py 的 BufferedFileHandler 同款:
051 |       - emit() 仅追加格式化字符串到 list, 不碰磁盘/终端
052 |       - flush_to_file() 退出时一次性落盘
053 | 
054 |     线程安全: logging.Handler.handle() 自带 RLock, emit() 在锁内执行,
055 |     因此 Reader 线程与主线程并发 logger 调用安全。
056 |     """
057 | 
058 |     def __init__(self, fmt: str) -> None:
059 |         super().__init__()
060 |         self.setFormatter(logging.Formatter(fmt))  # 复用与 rtsp_daemon.py 一致的格式
061 |         self._records: list[str] = []              # 格式化后的日志行缓冲
062 | 
063 |     def emit(self, record: logging.LogRecord) -> None:
064 |         """仅把格式化后的字符串追加到内存 list, 不做任何磁盘/终端 I/O。"""
065 |         self._records.append(self.format(record))
066 | 
067 |     def flush_to_file(self, path: str) -> None:
068 |         """把全部缓冲日志一次性写入文件 (UTF-8), 写后清空缓冲。"""
069 |         directory = os.path.dirname(path) or "."    # 日志文件所在目录
070 |         os.makedirs(directory, exist_ok=True)       # 目录不存在则创建
071 |         with open(path, "w", encoding="utf-8") as f:
072 |             f.write("\n".join(self._records))       # 每行一条日志
073 |             if self._records:                       # 非空时补结尾换行
074 |                 f.write("\n")
075 |         self._records.clear()                       # 落盘后清空, 避免重复写
076 | 
077 | 
078 | def setup_logging(log_dir: str) -> BufferedFileHandler:
079 |     """配置日志为内存缓冲模式 (运行时零 I/O), 返回 handler 供结束时 flush 落盘。
080 | 
081 |     log_dir 在调用前已按根目录解析为绝对路径, 不存在则自动创建。
082 |     日志文件为 log_dir/runtime.log, 由调用方在退出时 flush_to_file 写入。
083 |     """
084 |     os.makedirs(log_dir, exist_ok=True)                     # 确保日志目录存在 (已存在不报错)
085 |     handler = BufferedFileHandler(LOG_FORMAT)               # 缓冲 handler (零 I/O)
086 |     logging.basicConfig(level=logging.INFO, handlers=[handler])  # 全局日志只配这一个 handler
087 |     return handler                                          # 返回, 供调用方退出时 flush
088 | 
089 | 
090 | def load_config(path: str) -> dict:
091 |     """读取 YAML 配置, 返回 dict (见 config/config.yaml 的键结构)。"""
092 |     with open(path, encoding="utf-8") as f:  # 打开配置文件
093 |         return yaml.safe_load(f)             # 解析 YAML → dict
094 | 
095 | 
096 | def resolve_path(base_dir: str, p: str) -> str:
097 |     """把相对路径解析到 blade_rtsp_system 根目录 (main.py 所在目录)。
098 | 
099 |     config.yaml 里的相对路径 (如 models/best.pt, output/csv/...) 都是
100 |     相对于 blade_rtsp_system 根目录, 而不是 config/ 子目录。
101 |     """
102 |     if os.path.isabs(p):                    # 已是绝对路径则原样返回
103 |         return p
104 |     return os.path.join(base_dir, p)        # 否则拼到根目录下
105 | 
106 | 
107 | def build_frame_result(frame_id, timestamp, frame_width, frame_height,
108 |                        detection, located, loop_ms) -> dict:
109 |     """组装单帧的最终结果 dict, 供 ResultWriter 写入 CSV/JSONL。
110 | 
111 |     返回结构:
112 |       frame_id / timestamp / frame_width / frame_height /
113 |       blade_detected / blade_count / loop_ms / blades (=located 定位结果列表)
114 |     loop_ms = 主循环处理本帧的耗时(取帧后 → 可视化完成), 单位毫秒。
115 |     """
116 |     return {                                    # 返回单帧结果字典
117 |         "frame_id": frame_id,                   # 帧序号
118 |         "timestamp": timestamp,                 # 帧时间戳
119 |         "frame_width": frame_width,             # 原始帧宽
120 |         "frame_height": frame_height,           # 原始帧高
121 |         "blade_detected": detection["blade_detected"],  # 本帧是否检测到叶片
122 |         "blade_count": detection["blade_count"],        # 本帧叶片实例数
123 |         "loop_ms": loop_ms,                     # 主循环本帧耗时 (ms)
124 |         "blades": located,                      # 定位结果列表 (bbox/polygon/叶尖候选等)
125 |     }
126 | 
127 | 
128 | def main() -> int:
129 |     """TASK-1 主流程: 配置→模型→输出→视频源→Reader 线程→推理主循环→收尾。
130 | 
131 |     返回 0=正常退出, 1=初始化失败(模型缺失/加载失败/视频源打开失败)。
132 |     """
133 |     parser = argparse.ArgumentParser(description="TASK-1 YOLO26 RTSP 叶片定位")  # 参数解析器
134 |     parser.add_argument("--config", default="config/config.yaml")                # --config: 配置文件路径
135 |     parser.add_argument("--max-duration", type=float, default=0.0,               # --max-duration: 定时长跑
136 |                         help="最大运行秒数 (0=不限, 用于 Gate 6 定时长跑)")
137 |     parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),       # --resize: 输入缩放
138 |                         default=None,
139 |                         help="输入帧缩放 W H (覆盖 config processing.resize; 省略=用 config)")
140 |     args = parser.parse_args()                  # 解析命令行参数
141 | 
142 |     cfg = load_config(args.config)              # 读取配置文件为 dict
143 | 
144 |     # 相对路径基准 = blade_rtsp_system 根目录 (main.py 所在目录), 而非 config/ 子目录
145 |     # (注意: 不要用 config/ 子目录作基准, 否则 models/best.pt 会解析错位置)
146 |     root_dir = os.path.dirname(os.path.abspath(__file__))  # main.py 所在目录 = 根目录
147 | 
148 |     # 输出配置块 (日志 / CSV / JSONL / 窗口 都从这里取, 提前取出复用)
149 |     out_cfg = cfg.get("output", {})             # 取 output 配置段 (缺失给空 dict)
150 | 
151 |     # ── 日志 (缓冲到内存, 退出时一次性写 runtime.log) ──
152 |     logs_rel = out_cfg.get("logs") or "output/logs"   # 日志目录相对路径, 未配置时用默认
153 |     log_dir = resolve_path(root_dir, logs_rel)  # 解析为绝对路径
154 |     log_path = os.path.join(log_dir, "runtime.log")   # 日志落盘路径 (退出时写入)
155 |     log_handler = setup_logging(log_dir)        # 初始化缓冲日志, 拿到 handler
156 |     logger.info("=== blade_rtsp_system 启动 ===")  # 启动日志
157 | 
158 |     # ── 模型 (只加载一次, 禁止每帧重建) ──
159 |     model_cfg = cfg["model"]                    # 取 model 配置段
160 |     model_path = resolve_path(root_dir, model_cfg["path"])  # 权重文件绝对路径
161 |     if not os.path.exists(model_path):          # 权重不存在则直接报错退出
162 |         logger.error("模型文件不存在: %s", model_path)
163 |         log_handler.flush_to_file(log_path)     # 早退也要落盘
164 |         return 1                                # 返回 1 = 初始化失败
165 |     logger.info("模型路径: %s", model_path)     # 记录权重路径
166 | 
167 |     # YOLODetector: imgsz=640(推理尺寸, Rule 2 固定) / conf=0.5 / device / retina_masks
168 |     detector = YOLODetector(                    # 构造检测器 (此时只存参数, 未加载)
169 |         model_path=model_path,                  # 权重路径
170 |         imgsz=int(model_cfg.get("imgsz", 640)), # 推理尺寸
171 |         conf=float(model_cfg.get("conf", 0.5)), # 置信度阈值
172 |         device=model_cfg.get("device", 0),      # 设备 (0/1=GPU, "cpu"=CPU)
173 |         retina_masks=bool(model_cfg.get("retina_masks", True)),  # 高分辨率 mask
174 |     )
175 |     try:                                        # 加载模型 (失败会抛异常)
176 |         detector.load()
177 |     except Exception as exc:  # noqa: BLE001    # 捕获加载异常
178 |         logger.error("模型加载失败: %s", exc)
179 |         log_handler.flush_to_file(log_path)     # 早退也要落盘
180 |         return 1                                # 返回 1 = 初始化失败
181 | 
182 |     # ── 输出 (CSV / JSONL 二选一或都开) ──
183 |     save_csv = bool(out_cfg.get("save_csv", True))     # 是否写 CSV
184 |     save_jsonl = bool(out_cfg.get("save_jsonl", True)) # 是否写 JSONL
185 | 
186 |     csv_rel = out_cfg.get("csv") or "output/csv/blade_positions.csv"    # CSV 相对路径
187 |     jsonl_rel = out_cfg.get("jsonl") or "output/jsonl/blade_positions.jsonl"  # JSONL 相对路径
188 | 
189 |     # 仅当至少一种输出开启时才创建 writer; 否则保持 None (主循环里靠 `if writer` 判空)
190 |     writer = None                               # 默认无 writer
191 |     if save_csv or save_jsonl:                  # 至少一种输出开启才创建
192 |         writer = ResultWriter(                  # 构造结果写出器
193 |             csv_path=resolve_path(root_dir, csv_rel),       # CSV 绝对路径
194 |             jsonl_path=resolve_path(root_dir, jsonl_rel),   # JSONL 绝对路径
195 |         )
196 |         writer.open()                           # 打开文件准备写入
197 | 
198 |     # ── ROI (可选: 裁剪后再推理, 减少无关区域的计算量) ──
199 |     proc_cfg = cfg.get("processing", {})        # 取 processing 配置段
200 |     use_roi = bool(proc_cfg.get("use_roi", False))  # 是否启用 ROI
201 |     roi = proc_cfg.get("roi", {})               # ROI 配置 (x1/y1/x2/y2)
202 | 
203 |     # roi_offset = ROI 左上角 (x1,y1): mask/bbox 是 ROI 局部坐标时, 用此偏移平移回原图
204 |     roi_x1 = float(roi.get("x1", 0))            # ROI 左上角 x
205 |     roi_y1 = float(roi.get("y1", 0))            # ROI 左上角 y
206 |     roi_offset = (roi_x1, roi_y1)               # 打包成偏移元组
207 | 
208 |     # 输入缩放 (可选): 把原始帧缩到 resize 目标分辨率后再推理, 降低 mask 上采样等开销
209 |     # 优先级: 命令行 --resize W H > config processing.resize > None(不缩放)
210 |     if args.resize is not None:                 # 命令行显式给了 --resize W H
211 |         resize_target = tuple(args.resize)      # 用命令行值 (W, H)
212 |     else:                                       # 命令行没给, 回退到 config
213 |         resize_cfg = proc_cfg.get("resize")     # resize 配置 [w,h] 或 None
214 |         resize_target = tuple(int(x) for x in resize_cfg) if resize_cfg else None  # 或 None
215 | 
216 |     # ── 视频源 (RTSP, 唯一模式) ──
217 |     rtsp_cfg = cfg.get("rtsp", {})              # 取 rtsp 配置段
218 |     default_rtsp_url = "rtsp://127.0.0.1:8554/blade"  # 默认 RTSP 地址
219 |     source = rtsp_cfg.get("url", default_rtsp_url)  # 只用配置里的 RTSP 地址
220 | 
221 |     # 日志不输出明文 RTSP URL (含密码): 用 _masked_source() 脱敏成 rtsp://user:***@host
222 |     source_label = RTSPReader(source)._masked_source()  # 临时构造 Reader 只为了脱敏 URL
223 |     logger.info("视频源: %s", source_label)     # 记录视频源 (已脱敏)
224 | 
225 |     reader = RTSPReader(                        # 构造读帧器
226 |         source=source,                          # 视频源地址
227 |         reconnect_interval_sec=float(rtsp_cfg.get("reconnect_interval_sec", 2)),  # 重连间隔
228 |         max_reconnect_attempts=int(rtsp_cfg.get("max_reconnect_attempts", 0)),    # 重连上限 (0=无限)
229 |     )
230 |     if not reader.open():                       # 打开视频源失败
231 |         logger.error("视频源打开失败, 退出")
232 |         if writer:                              # 若已创建 writer, 先关闭
233 |             writer.close()
234 |         log_handler.flush_to_file(log_path)     # 早退也要落盘
235 |         return 1                                # 返回 1 = 初始化失败
236 | 
237 |     # ── Frame buffer (latest-frame 策略, max_size=1: 新帧覆盖未处理的旧帧) ──
238 |     buf_cfg = cfg.get("buffer", {})             # 取 buffer 配置段
239 |     buffer = FrameBuffer(max_size=int(buf_cfg.get("max_size", 1)))  # 构造最新帧缓冲
240 | 
241 |     # ── Reader 线程 (Thread 1) ──
242 |     running = threading.Event()                 # 运行标志 (控制 Reader 线程与主循环退出)
243 |     running.set()                               # 置为运行中
244 | 
245 |     def reader_loop():
246 |         # RTSP 实时流: 读帧失败即断流, 循环重连直到成功或达到上限 (无限模式永不放弃)
247 |         while running.is_set():                    # 只要还在运行就继续读帧
248 |             ok, frame, ts = reader.read()          # 读一帧 → (成功?, 帧, 时间戳)
249 |             if not ok:                             # 读帧失败 = 断流
250 |                 while running.is_set():            # 重连循环
251 |                     if reader.reconnect_exhausted():  # 达到重连上限
252 |                         logger.error("达到最大重连次数 %d, 退出",
253 |                                      reader.max_reconnect_attempts)
254 |                         running.clear()            # 通知主循环停止
255 |                         return                     # 退出 reader_loop
256 |                     if reader.reconnect():         # 尝试重连
257 |                         break                      # 重连成功
258 |                 continue                           # 重连成功后回到外层继续读帧
259 | 
260 |             # 帧写入 latest-frame buffer (附带时间戳 ts 与帧序号 frame_id)
261 |             if resize_target is not None:        # 启用输入缩放
262 |                 frame = cv2.resize(frame, resize_target)  # 缩放到目标分辨率 (降低后续开销)
263 |             buffer.put(frame, ts, reader.frame_id) # 把帧塞进缓冲
264 | 
265 |         running.clear()                            # 读帧循环结束, 通知主循环停止
266 | 
267 |     t_reader = threading.Thread(target=reader_loop, daemon=True)  # 创建 Reader 后台线程
268 |     t_reader.start()                               # 启动 Reader 线程
269 |     logger.info("Reader 线程已启动")               # 记录线程已启动
270 | 
271 |     # ── 可视化 (headless 环境自动降级, 记录日志不静默) ──
272 |     show_window = bool(out_cfg.get("show_window", True))  # 是否显示实时窗口
273 |     if show_window and not os.environ.get("DISPLAY"):     # headless 无显示环境
274 |         logger.warning("headless 环境(无 DISPLAY), 关闭实时窗口(show_window=false)")
275 |         show_window = False                       # 自动关闭窗口
276 |     vis = Visualizer(show_window=show_window)     # 构造可视化器
277 | 
278 |     # 原始帧尺寸 (用于把 ROI 局部坐标映射回原图坐标)
279 |     if reader.cap is not None:                    # cap 存在才取分辨率
280 |         frame_w = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 原始帧宽
281 |         frame_h = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 原始帧高
282 |     else:                                         # cap 缺失
283 |         frame_w = frame_h = 0                     # 尺寸记 0
284 |     if resize_target is not None:                 # 输入缩放后, 坐标基准改为缩放后尺寸
285 |         frame_w, frame_h = resize_target          # 覆盖为缩放后分辨率
286 | 
287 |     # 推理耗时统计 (周期日志 + 最终统计用)
288 |     inference_times = []                          # 推理耗时列表 (ms)
289 |     loop_times = []                               # 主循环本帧耗时列表 (ms)
290 |     t0 = time.time()                              # 启动时刻 (用于 FPS/定时长跑)
291 |     last_report = t0                              # 上次周期统计时刻
292 | 
293 |     # SIGTERM 也走优雅退出 (复用 except KeyboardInterrupt → finally → 落盘),
294 |     # 否则 buffered 结果/日志在 SIGTERM 时会被直接杀、不落盘而丢失。
295 |     def _sigterm_to_keyboard_interrupt(signum, frame):
296 |         raise KeyboardInterrupt                   # 触发 except KeyboardInterrupt → finally 落盘
297 |     signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
298 | 
299 |     try:
300 |         # ── 主线程推理循环 (Thread 2) ──
301 |         while running.is_set():                   # 只要 Reader 还在运行
302 |             # 取最新帧 (旧帧可能已被覆盖/丢弃, 见 frame_buffer 的 dropped 统计)
303 |             frame, ts, fid = buffer.get_latest()  # 取最新帧 → (帧, 时间戳, 帧序号)
304 |             if frame is None:                     # 暂无可用帧 (buffer 空)
305 |                 time.sleep(0.01)                  # 短暂等待
306 |                 continue                          # 回到循环开头再取
307 | 
308 |             # get_latest 空时返回 (None,None,None); 上面已 continue, 故 frame 非空 → fid 必为 int
309 |             assert fid is not None                # 窄化类型: 明确告知类型检查器 fid 非 None
310 | 
311 |             # 计时起点: 主循环处理本帧的耗时 (取帧后 → 可视化完成)
312 |             t_frame = time.perf_counter()         # 高精度计时起点
313 | 
314 |             # ROI 裁剪 (可选): 只对 ROI 子图推理, 结果坐标是 ROI 局部坐标
315 |             if use_roi:                           # 启用 ROI
316 |                 roi_x2 = int(roi.get("x2", frame.shape[1]))  # ROI 右下角 x (默认到帧右边界)
317 |                 roi_y2 = int(roi.get("y2", frame.shape[0]))  # ROI 右下角 y (默认到帧下边界)
318 |                 roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]  # 裁出 ROI 子图
319 |                 infer_input = roi_frame           # 用 ROI 子图推理
320 |             else:                                 # 不用 ROI
321 |                 infer_input = frame               # 直接用整帧推理
322 | 
323 |             # YOLO 推理: 返回 (result, 推理耗时 ms)
324 |             result, inf_ms = detector.predict(infer_input)  # 单帧分割推理
325 |             inference_times.append(inf_ms)        # 记录本次推理耗时
326 | 
327 |             # blade 提取 + 定位 (坐标恢复到原始帧)
328 |             #   extract_blades: 从 result 筛出 class_name="blade" 的 instances
329 |             #   locate_blades:   mask→polygon + ROI→原图坐标, 输出 bbox/center/polygon/叶尖候选
330 |             detection = blade_segmenter.extract_blades(result, class_name="blade")  # 提取 blade 实例
331 | 
332 |             # 传给 locator 的坐标偏移: 启用 ROI 时用 roi_offset, 否则零偏移
333 |             offset = roi_offset if use_roi else (0.0, 0.0)  # 决定坐标恢复用的平移量
334 |             located = blade_locator.locate_blades(  # 对每个实例做 mask 处理 + 坐标恢复
335 |                 detection["blade_instances"],       # 输入: 提取出的 blade 实例列表
336 |                 roi_offset=offset,                  # 坐标偏移
337 |                 frame_width=frame_w, frame_height=frame_h,  # 原始帧尺寸
338 |             )
339 | 
340 |             # 可视化 (画 mask/polygon + 帧号 + FPS)
341 |             now = time.time()                     # 当前时刻
342 |             # FPS = 1 / 运行耗时; 第一帧(fid<=1)尚无时间差, 记 0 避免除零
343 |             if fid > 1:                           # (fid 已在上面 assert 非 None)
344 |                 fps = 1.0 / max(now - t0, 1e-6)   # FPS = 1 / 总运行耗时 (防止除零)
345 |             else:                                 # 首帧或 fid 缺失
346 |                 fps = 0.0                         # FPS 记 0
347 |             annotated = vis.draw(frame, fid, located, fps)  # 在帧上画标注
348 |             vis.show(annotated)                   # 显示 (headless 时为空操作)
349 | 
350 |             # 主循环本帧耗时 (取帧后 → 可视化完成), 单位 ms
351 |             loop_ms = (time.perf_counter() - t_frame) * 1000.0  # 高精度计时差
352 |             loop_times.append(loop_ms)            # 累计 (周期统计 + 最终统计用)
353 | 
354 |             # 组装单帧结果并写入 CSV/JSONL (含 loop_ms)
355 |             fr = build_frame_result(fid, ts, frame_w, frame_h, detection, located, loop_ms)  # 组装结果 dict
356 |             if writer:                            # 有 writer 才写
357 |                 writer.write(fr)                  # 写出本帧结果
358 | 
359 |             # 周期统计 (每 5 秒打印一次 recv/proc/dropped + 推理/主循环耗时)
360 |             if now - last_report >= 5.0:          # 距上次统计 ≥5 秒
361 |                 st = buffer.stats()               # 取 buffer 统计 (recv/proc/dropped)
362 |                 if inference_times:               # 有推理耗时样本
363 |                     avg_inf = sum(inference_times) / len(inference_times)  # 平均推理耗时
364 |                     max_inf = max(inference_times)  # 最大推理耗时
365 |                 else:                             # 无样本
366 |                     avg_inf = 0.0                 # 平均推理耗时记 0
367 |                     max_inf = 0.0                 # 最大推理耗时记 0
368 |                 if loop_times:                    # 有主循环耗时样本
369 |                     avg_loop = sum(loop_times) / len(loop_times)  # 平均主循环耗时
370 |                     max_loop = max(loop_times)    # 最大主循环耗时
371 |                 else:                             # 无样本
372 |                     avg_loop = 0.0                # 平均主循环耗时记 0
373 |                     max_loop = 0.0                # 最大主循环耗时记 0
374 |                 logger.info(                      # 打印周期统计
375 |                     "统计: recv=%d proc=%d dropped=%d  avg_inf=%.1fms max_inf=%.1fms avg_loop=%.1fms max_loop=%.1fms",
376 |                     st["received_frames"], st["processed_frames"],
377 |                     st["dropped_frames"], avg_inf, max_inf, avg_loop, max_loop)
378 |                 inference_times.clear()           # 清空推理耗时样本, 进入下一统计周期
379 |                 loop_times.clear()                # 清空主循环耗时样本, 进入下一统计周期
380 |                 last_report = now                 # 更新统计时刻
381 | 
382 |             # 定时长跑 (Gate 6): 用 break 正常走出循环, 让 finally 完成收尾统计
383 |             if args.max_duration > 0 and (now - t0) >= args.max_duration:  # 达到设定时长
384 |                 logger.info("达到最大运行时长 %.0fs, 正常停止", args.max_duration)
385 |                 break                             # 正常退出主循环
386 | 
387 |     except KeyboardInterrupt:                     # 用户按 Ctrl+C
388 |         logger.info("收到 Ctrl+C, 停止")          # 记录手动停止
389 |     finally:
390 |         # ── 收尾: 停 Reader 线程 → 释放视频源 → 关输出 → 关窗口 → 最终统计 ──
391 |         running.clear()                           # 通知 Reader 线程停止
392 |         t_reader.join(timeout=5)                  # 等 Reader 线程退出 (最多 5 秒)
393 |         reader.release()                          # 释放视频源
394 |         if writer:                                # 若创建了 writer
395 |             writer.close()                        # 关闭输出文件
396 |         vis.close()                               # 关闭可视化窗口
397 | 
398 |         st = buffer.stats()                       # 取最终 buffer 统计
399 |         if inference_times:                       # 有耗时样本
400 |             avg_inf = sum(inference_times) / len(inference_times)  # 最终平均推理耗时
401 |         else:                                     # 无样本
402 |             avg_inf = 0.0                         # 记 0
403 |         if loop_times:                            # 有主循环耗时样本
404 |             avg_loop = sum(loop_times) / len(loop_times)  # 最终平均主循环耗时
405 |         else:                                     # 无样本
406 |             avg_loop = 0.0                        # 记 0
407 | 
408 |         elapsed = time.time() - t0                # 总运行时长
409 |         if elapsed > 0:                           # 运行过 (避免除零)
410 |             avg_fps = st["processed_frames"] / elapsed  # 平均处理 FPS = 处理帧数/时长
411 |         else:                                     # 几乎没运行
412 |             avg_fps = 0.0                         # FPS 记 0
413 | 
414 |         logger.info("=== 最终统计: recv=%d proc=%d dropped=%d avg_inf=%.1fms "
415 |                     "avg_loop=%.1fms avg_fps=%.1f reconnect=%d 运行=%.1fs ===",   # 最终统计日志 (续行)
416 |                     st["received_frames"], st["processed_frames"],
417 |                     st["dropped_frames"], avg_inf, avg_loop, avg_fps,
418 |                     reader.reconnect_count, elapsed)
419 |         logger.info("=== blade_rtsp_system 退出 ===")  # 退出日志
420 |         log_handler.flush_to_file(log_path)     # 全部缓冲日志一次性落盘
421 | 
422 |     return 0                                     # 返回 0 = 正常退出
423 | 
424 | 
425 | if __name__ == "__main__":                       # 直接运行本文件时
426 |     sys.exit(main())                             # 执行 main 并以返回码退出
```
