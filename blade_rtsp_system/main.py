#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py — TASK-1 主入口 (TASK-1 第 28 节工作流)

  RTSP → Frame → ROI → YOLO26 best.pt(640) → Blade Mask
       → Raw/Simple Polygon → Original Coordinate → CSV/JSONL/Visualization

用法:
  python main.py --config config/config.yaml

多线程设计 (TASK-1 第 22 节):
  Thread 1: RTSP Reader (读帧 → latest-frame buffer)
  Thread 2 (主线程): YOLO Inference + 定位 + 输出

日志策略 (与 rtsp_daemon.py 一致):
  日志运行时缓冲到内存 list (零 I/O), 进程退出时一次性写 output/logs/runtime.log。
"""
from __future__ import annotations  # 启用「推迟求值注解」, 支持 `str | int` 写法

import argparse                        # 解析命令行参数 (--config / --max-duration)
import logging                         # 日志 (内存缓冲, 退出时落盘)
import os                              # 路径拼接 / 文件是否存在 / 环境变量
import signal                          # 信号处理 (SIGTERM → 优雅退出)
import sys                             # sys.path / sys.exit
import threading                       # 多线程 (Reader 线程 + Event 控制)
import time                            # 计时 (推理耗时 / FPS / 定时长跑)

import cv2                             # OpenCV (取视频分辨率 CAP_PROP_*)
import yaml                            # 解析 config.yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 把本目录加入模块搜索路径, 保证 `from src import` 可用

from src import blade_locator, blade_segmenter  # noqa: E402  # 叶片提取 + 定位 (业务核心)
from src.coordinate_mapper import map_mask  # noqa: E402     # ROI→原图坐标映射 (本文件未直接调用, 由 locator 内部使用)
from src.frame_buffer import FrameBuffer  # noqa: E402       # 最新帧缓冲 (Reader↔主线程交接)
from src.result_writer import ResultWriter  # noqa: E402     # 结果写出 CSV/JSONL
from src.rtsp_reader import RTSPReader  # noqa: E402         # RTSP 读帧 + 断流重连
from src.visualizer import Visualizer  # noqa: E402          # 实时窗口可视化
from src.yolo_detector import YOLODetector  # noqa: E402     # YOLO 模型加载 + 单帧推理

logger = logging.getLogger("blade_rtsp.main")  # 本模块的 logger 实例

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"  # 与 rtsp_daemon.py 一致


class BufferedFileHandler(logging.Handler):
    """把日志记录缓冲到内存 list, 运行时零 I/O, 结束时一次性写文件。

    与 blade_rtsp_system/rtsp_daemon.py 的 BufferedFileHandler 同款:
      - emit() 仅追加格式化字符串到 list, 不碰磁盘/终端
      - flush_to_file() 退出时一次性落盘

    线程安全: logging.Handler.handle() 自带 RLock, emit() 在锁内执行,
    因此 Reader 线程与主线程并发 logger 调用安全。
    """

    def __init__(self, fmt: str) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(fmt))  # 复用与 rtsp_daemon.py 一致的格式
        self._records: list[str] = []              # 格式化后的日志行缓冲

    def emit(self, record: logging.LogRecord) -> None:
        """仅把格式化后的字符串追加到内存 list, 不做任何磁盘/终端 I/O。"""
        self._records.append(self.format(record))

    def flush_to_file(self, path: str) -> None:
        """把全部缓冲日志一次性写入文件 (UTF-8), 写后清空缓冲。"""
        directory = os.path.dirname(path) or "."    # 日志文件所在目录
        os.makedirs(directory, exist_ok=True)       # 目录不存在则创建
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._records))       # 每行一条日志
            if self._records:                       # 非空时补结尾换行
                f.write("\n")
        self._records.clear()                       # 落盘后清空, 避免重复写


def setup_logging(log_dir: str) -> BufferedFileHandler:
    """配置日志为内存缓冲模式 (运行时零 I/O), 返回 handler 供结束时 flush 落盘。

    log_dir 在调用前已按根目录解析为绝对路径, 不存在则自动创建。
    日志文件为 log_dir/runtime.log, 由调用方在退出时 flush_to_file 写入。
    """
    os.makedirs(log_dir, exist_ok=True)                     # 确保日志目录存在 (已存在不报错)
    handler = BufferedFileHandler(LOG_FORMAT)               # 缓冲 handler (零 I/O)
    logging.basicConfig(level=logging.INFO, handlers=[handler])  # 全局日志只配这一个 handler
    return handler                                          # 返回, 供调用方退出时 flush


def load_config(path: str) -> dict:
    """读取 YAML 配置, 返回 dict (见 config/config.yaml 的键结构)。"""
    with open(path, encoding="utf-8") as f:  # 打开配置文件
        return yaml.safe_load(f)             # 解析 YAML → dict


def resolve_path(base_dir: str, p: str) -> str:
    """把相对路径解析到 blade_rtsp_system 根目录 (main.py 所在目录)。

    config.yaml 里的相对路径 (如 models/best.pt, output/csv/...) 都是
    相对于 blade_rtsp_system 根目录, 而不是 config/ 子目录。
    """
    if os.path.isabs(p):                    # 已是绝对路径则原样返回
        return p
    return os.path.join(base_dir, p)        # 否则拼到根目录下


def build_frame_result(frame_id, timestamp, frame_width, frame_height,
                       detection, located, loop_ms) -> dict:
    """组装单帧的最终结果 dict, 供 ResultWriter 写入 CSV/JSONL。

    返回结构:
      frame_id / timestamp / frame_width / frame_height /
      blade_detected / blade_count / loop_ms / blades (=located 定位结果列表)
    loop_ms = 主循环处理本帧的耗时(取帧后 → 可视化完成), 单位毫秒。
    """
    return {                                    # 返回单帧结果字典
        "frame_id": frame_id,                   # 帧序号
        "timestamp": timestamp,                 # 帧时间戳
        "frame_width": frame_width,             # 原始帧宽
        "frame_height": frame_height,           # 原始帧高
        "blade_detected": detection["blade_detected"],  # 本帧是否检测到叶片
        "blade_count": detection["blade_count"],        # 本帧叶片实例数
        "loop_ms": loop_ms,                     # 主循环本帧耗时 (ms)
        "blades": located,                      # 定位结果列表 (bbox/polygon/叶尖候选等)
    }


def main() -> int:
    """TASK-1 主流程: 配置→模型→输出→视频源→Reader 线程→推理主循环→收尾。

    返回 0=正常退出, 1=初始化失败(模型缺失/加载失败/视频源打开失败)。
    """
    parser = argparse.ArgumentParser(description="TASK-1 YOLO26 RTSP 叶片定位")  # 参数解析器
    parser.add_argument("--config", default="config/config.yaml")                # --config: 配置文件路径
    parser.add_argument("--max-duration", type=float, default=0.0,               # --max-duration: 定时长跑
                        help="最大运行秒数 (0=不限, 用于 Gate 6 定时长跑)")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),       # --resize: 输入缩放
                        default=None,
                        help="输入帧缩放 W H (覆盖 config processing.resize; 省略=用 config)")
    args = parser.parse_args()                  # 解析命令行参数

    cfg = load_config(args.config)              # 读取配置文件为 dict

    # 相对路径基准 = blade_rtsp_system 根目录 (main.py 所在目录), 而非 config/ 子目录
    # (注意: 不要用 config/ 子目录作基准, 否则 models/best.pt 会解析错位置)
    root_dir = os.path.dirname(os.path.abspath(__file__))  # main.py 所在目录 = 根目录

    # 输出配置块 (日志 / CSV / JSONL / 窗口 都从这里取, 提前取出复用)
    out_cfg = cfg.get("output", {})             # 取 output 配置段 (缺失给空 dict)

    # ── 日志 (缓冲到内存, 退出时一次性写 runtime.log) ──
    logs_rel = out_cfg.get("logs") or "output/logs"   # 日志目录相对路径, 未配置时用默认
    log_dir = resolve_path(root_dir, logs_rel)  # 解析为绝对路径
    log_path = os.path.join(log_dir, "runtime.log")   # 日志落盘路径 (退出时写入)
    log_handler = setup_logging(log_dir)        # 初始化缓冲日志, 拿到 handler
    logger.info("=== blade_rtsp_system 启动 ===")  # 启动日志

    # ── 模型 (只加载一次, 禁止每帧重建) ──
    model_cfg = cfg["model"]                    # 取 model 配置段
    model_path = resolve_path(root_dir, model_cfg["path"])  # 权重文件绝对路径
    if not os.path.exists(model_path):          # 权重不存在则直接报错退出
        logger.error("模型文件不存在: %s", model_path)
        log_handler.flush_to_file(log_path)     # 早退也要落盘
        return 1                                # 返回 1 = 初始化失败
    logger.info("模型路径: %s", model_path)     # 记录权重路径

    # YOLODetector: imgsz=640(推理尺寸, Rule 2 固定) / conf=0.5 / device / retina_masks
    detector = YOLODetector(                    # 构造检测器 (此时只存参数, 未加载)
        model_path=model_path,                  # 权重路径
        imgsz=int(model_cfg.get("imgsz", 640)), # 推理尺寸
        conf=float(model_cfg.get("conf", 0.5)), # 置信度阈值
        device=model_cfg.get("device", 0),      # 设备 (0/1=GPU, "cpu"=CPU)
        retina_masks=bool(model_cfg.get("retina_masks", True)),  # 高分辨率 mask
    )
    try:                                        # 加载模型 (失败会抛异常)
        detector.load()
    except Exception as exc:  # noqa: BLE001    # 捕获加载异常
        logger.error("模型加载失败: %s", exc)
        log_handler.flush_to_file(log_path)     # 早退也要落盘
        return 1                                # 返回 1 = 初始化失败

    # ── 输出 (CSV / JSONL 二选一或都开) ──
    save_csv = bool(out_cfg.get("save_csv", True))     # 是否写 CSV
    save_jsonl = bool(out_cfg.get("save_jsonl", True)) # 是否写 JSONL

    csv_rel = out_cfg.get("csv") or "output/csv/blade_positions.csv"    # CSV 相对路径
    jsonl_rel = out_cfg.get("jsonl") or "output/jsonl/blade_positions.jsonl"  # JSONL 相对路径

    # 仅当至少一种输出开启时才创建 writer; 否则保持 None (主循环里靠 `if writer` 判空)
    writer = None                               # 默认无 writer
    if save_csv or save_jsonl:                  # 至少一种输出开启才创建
        writer = ResultWriter(                  # 构造结果写出器
            csv_path=resolve_path(root_dir, csv_rel),       # CSV 绝对路径
            jsonl_path=resolve_path(root_dir, jsonl_rel),   # JSONL 绝对路径
        )
        writer.open()                           # 打开文件准备写入

    # ── ROI (可选: 裁剪后再推理, 减少无关区域的计算量) ──
    proc_cfg = cfg.get("processing", {})        # 取 processing 配置段
    use_roi = bool(proc_cfg.get("use_roi", False))  # 是否启用 ROI
    roi = proc_cfg.get("roi", {})               # ROI 配置 (x1/y1/x2/y2)

    # roi_offset = ROI 左上角 (x1,y1): mask/bbox 是 ROI 局部坐标时, 用此偏移平移回原图
    roi_x1 = float(roi.get("x1", 0))            # ROI 左上角 x
    roi_y1 = float(roi.get("y1", 0))            # ROI 左上角 y
    roi_offset = (roi_x1, roi_y1)               # 打包成偏移元组

    # 输入缩放 (可选): 把原始帧缩到 resize 目标分辨率后再推理, 降低 mask 上采样等开销
    # 优先级: 命令行 --resize W H > config processing.resize > None(不缩放)
    if args.resize is not None:                 # 命令行显式给了 --resize W H
        resize_target = tuple(args.resize)      # 用命令行值 (W, H)
    else:                                       # 命令行没给, 回退到 config
        resize_cfg = proc_cfg.get("resize")     # resize 配置 [w,h] 或 None
        resize_target = tuple(int(x) for x in resize_cfg) if resize_cfg else None  # 或 None

    # ── 视频源 (RTSP, 唯一模式) ──
    rtsp_cfg = cfg.get("rtsp", {})              # 取 rtsp 配置段
    default_rtsp_url = "rtsp://127.0.0.1:8554/blade"  # 默认 RTSP 地址
    source = rtsp_cfg.get("url", default_rtsp_url)  # 只用配置里的 RTSP 地址

    # 日志不输出明文 RTSP URL (含密码): 用 _masked_source() 脱敏成 rtsp://user:***@host
    source_label = RTSPReader(source)._masked_source()  # 临时构造 Reader 只为了脱敏 URL
    logger.info("视频源: %s", source_label)     # 记录视频源 (已脱敏)

    reader = RTSPReader(                        # 构造读帧器
        source=source,                          # 视频源地址
        reconnect_interval_sec=float(rtsp_cfg.get("reconnect_interval_sec", 2)),  # 重连间隔
        max_reconnect_attempts=int(rtsp_cfg.get("max_reconnect_attempts", 0)),    # 重连上限 (0=无限)
    )
    if not reader.open():                       # 打开视频源失败
        logger.error("视频源打开失败, 退出")
        if writer:                              # 若已创建 writer, 先关闭
            writer.close()
        log_handler.flush_to_file(log_path)     # 早退也要落盘
        return 1                                # 返回 1 = 初始化失败

    # ── Frame buffer (latest-frame 策略, max_size=1: 新帧覆盖未处理的旧帧) ──
    buf_cfg = cfg.get("buffer", {})             # 取 buffer 配置段
    buffer = FrameBuffer(max_size=int(buf_cfg.get("max_size", 1)))  # 构造最新帧缓冲

    # ── Reader 线程 (Thread 1) ──
    running = threading.Event()                 # 运行标志 (控制 Reader 线程与主循环退出)
    running.set()                               # 置为运行中

    def reader_loop():
        # RTSP 实时流: 读帧失败即断流, 循环重连直到成功或达到上限 (无限模式永不放弃)
        while running.is_set():                    # 只要还在运行就继续读帧
            ok, frame, ts = reader.read()          # 读一帧 → (成功?, 帧, 时间戳)
            if not ok:                             # 读帧失败 = 断流
                while running.is_set():            # 重连循环
                    if reader.reconnect_exhausted():  # 达到重连上限
                        logger.error("达到最大重连次数 %d, 退出",
                                     reader.max_reconnect_attempts)
                        running.clear()            # 通知主循环停止
                        return                     # 退出 reader_loop
                    if reader.reconnect():         # 尝试重连
                        break                      # 重连成功
                continue                           # 重连成功后回到外层继续读帧

            # 帧写入 latest-frame buffer (附带时间戳 ts 与帧序号 frame_id)
            if resize_target is not None:        # 启用输入缩放
                frame = cv2.resize(frame, resize_target)  # 缩放到目标分辨率 (降低后续开销)
            buffer.put(frame, ts, reader.frame_id) # 把帧塞进缓冲

        running.clear()                            # 读帧循环结束, 通知主循环停止

    t_reader = threading.Thread(target=reader_loop, daemon=True)  # 创建 Reader 后台线程
    t_reader.start()                               # 启动 Reader 线程
    logger.info("Reader 线程已启动")               # 记录线程已启动

    # ── 可视化 (headless 环境自动降级, 记录日志不静默) ──
    show_window = bool(out_cfg.get("show_window", True))  # 是否显示实时窗口
    if show_window and not os.environ.get("DISPLAY"):     # headless 无显示环境
        logger.warning("headless 环境(无 DISPLAY), 关闭实时窗口(show_window=false)")
        show_window = False                       # 自动关闭窗口
    vis = Visualizer(show_window=show_window)     # 构造可视化器

    # 原始帧尺寸 (用于把 ROI 局部坐标映射回原图坐标)
    if reader.cap is not None:                    # cap 存在才取分辨率
        frame_w = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 原始帧宽
        frame_h = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 原始帧高
    else:                                         # cap 缺失
        frame_w = frame_h = 0                     # 尺寸记 0
    if resize_target is not None:                 # 输入缩放后, 坐标基准改为缩放后尺寸
        frame_w, frame_h = resize_target          # 覆盖为缩放后分辨率

    # 推理耗时统计 (周期日志 + 最终统计用)
    inference_times = []                          # 推理耗时列表 (ms)
    loop_times = []                               # 主循环本帧耗时列表 (ms)
    t0 = time.time()                              # 启动时刻 (用于 FPS/定时长跑)
    last_report = t0                              # 上次周期统计时刻

    # SIGTERM 也走优雅退出 (复用 except KeyboardInterrupt → finally → 落盘),
    # 否则 buffered 结果/日志在 SIGTERM 时会被直接杀、不落盘而丢失。
    def _sigterm_to_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt                   # 触发 except KeyboardInterrupt → finally 落盘
    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

    try:
        # ── 主线程推理循环 (Thread 2) ──
        while running.is_set():                   # 只要 Reader 还在运行
            # 取最新帧 (旧帧可能已被覆盖/丢弃, 见 frame_buffer 的 dropped 统计)
            frame, ts, fid = buffer.get_latest()  # 取最新帧 → (帧, 时间戳, 帧序号)
            if frame is None:                     # 暂无可用帧 (buffer 空)
                time.sleep(0.01)                  # 短暂等待
                continue                          # 回到循环开头再取

            # get_latest 空时返回 (None,None,None); 上面已 continue, 故 frame 非空 → fid 必为 int
            assert fid is not None                # 窄化类型: 明确告知类型检查器 fid 非 None

            # 计时起点: 主循环处理本帧的耗时 (取帧后 → 可视化完成)
            t_frame = time.perf_counter()         # 高精度计时起点

            # ROI 裁剪 (可选): 只对 ROI 子图推理, 结果坐标是 ROI 局部坐标
            if use_roi:                           # 启用 ROI
                roi_x2 = int(roi.get("x2", frame.shape[1]))  # ROI 右下角 x (默认到帧右边界)
                roi_y2 = int(roi.get("y2", frame.shape[0]))  # ROI 右下角 y (默认到帧下边界)
                roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]  # 裁出 ROI 子图
                infer_input = roi_frame           # 用 ROI 子图推理
            else:                                 # 不用 ROI
                infer_input = frame               # 直接用整帧推理

            # YOLO 推理: 返回 (result, 推理耗时 ms)
            result, inf_ms = detector.predict(infer_input)  # 单帧分割推理
            inference_times.append(inf_ms)        # 记录本次推理耗时

            # blade 提取 + 定位 (坐标恢复到原始帧)
            #   extract_blades: 从 result 筛出 class_name="blade" 的 instances
            #   locate_blades:   mask→polygon + ROI→原图坐标, 输出 bbox/center/polygon/叶尖候选
            detection = blade_segmenter.extract_blades(result, class_name="blade")  # 提取 blade 实例

            # 传给 locator 的坐标偏移: 启用 ROI 时用 roi_offset, 否则零偏移
            offset = roi_offset if use_roi else (0.0, 0.0)  # 决定坐标恢复用的平移量
            located = blade_locator.locate_blades(  # 对每个实例做 mask 处理 + 坐标恢复
                detection["blade_instances"],       # 输入: 提取出的 blade 实例列表
                roi_offset=offset,                  # 坐标偏移
                frame_width=frame_w, frame_height=frame_h,  # 原始帧尺寸
            )

            # 可视化 (画 mask/polygon + 帧号 + FPS)
            now = time.time()                     # 当前时刻
            # FPS = 1 / 运行耗时; 第一帧(fid<=1)尚无时间差, 记 0 避免除零
            if fid > 1:                           # (fid 已在上面 assert 非 None)
                fps = 1.0 / max(now - t0, 1e-6)   # FPS = 1 / 总运行耗时 (防止除零)
            else:                                 # 首帧或 fid 缺失
                fps = 0.0                         # FPS 记 0
            annotated = vis.draw(frame, fid, located, fps)  # 在帧上画标注
            vis.show(annotated)                   # 显示 (headless 时为空操作)

            # 主循环本帧耗时 (取帧后 → 可视化完成), 单位 ms
            loop_ms = (time.perf_counter() - t_frame) * 1000.0  # 高精度计时差
            loop_times.append(loop_ms)            # 累计 (周期统计 + 最终统计用)

            # 组装单帧结果并写入 CSV/JSONL (含 loop_ms)
            fr = build_frame_result(fid, ts, frame_w, frame_h, detection, located, loop_ms)  # 组装结果 dict
            if writer:                            # 有 writer 才写
                writer.write(fr)                  # 写出本帧结果

            # 周期统计 (每 5 秒打印一次 recv/proc/dropped + 推理/主循环耗时)
            if now - last_report >= 5.0:          # 距上次统计 ≥5 秒
                st = buffer.stats()               # 取 buffer 统计 (recv/proc/dropped)
                if inference_times:               # 有推理耗时样本
                    avg_inf = sum(inference_times) / len(inference_times)  # 平均推理耗时
                    max_inf = max(inference_times)  # 最大推理耗时
                else:                             # 无样本
                    avg_inf = 0.0                 # 平均推理耗时记 0
                    max_inf = 0.0                 # 最大推理耗时记 0
                if loop_times:                    # 有主循环耗时样本
                    avg_loop = sum(loop_times) / len(loop_times)  # 平均主循环耗时
                    max_loop = max(loop_times)    # 最大主循环耗时
                else:                             # 无样本
                    avg_loop = 0.0                # 平均主循环耗时记 0
                    max_loop = 0.0                # 最大主循环耗时记 0
                logger.info(                      # 打印周期统计
                    "统计: recv=%d proc=%d dropped=%d  avg_inf=%.1fms max_inf=%.1fms avg_loop=%.1fms max_loop=%.1fms",
                    st["received_frames"], st["processed_frames"],
                    st["dropped_frames"], avg_inf, max_inf, avg_loop, max_loop)
                inference_times.clear()           # 清空推理耗时样本, 进入下一统计周期
                loop_times.clear()                # 清空主循环耗时样本, 进入下一统计周期
                last_report = now                 # 更新统计时刻

            # 定时长跑 (Gate 6): 用 break 正常走出循环, 让 finally 完成收尾统计
            if args.max_duration > 0 and (now - t0) >= args.max_duration:  # 达到设定时长
                logger.info("达到最大运行时长 %.0fs, 正常停止", args.max_duration)
                break                             # 正常退出主循环

    except KeyboardInterrupt:                     # 用户按 Ctrl+C
        logger.info("收到 Ctrl+C, 停止")          # 记录手动停止
    finally:
        # ── 收尾: 停 Reader 线程 → 释放视频源 → 关输出 → 关窗口 → 最终统计 ──
        running.clear()                           # 通知 Reader 线程停止
        t_reader.join(timeout=5)                  # 等 Reader 线程退出 (最多 5 秒)
        reader.release()                          # 释放视频源
        if writer:                                # 若创建了 writer
            writer.close()                        # 关闭输出文件
        vis.close()                               # 关闭可视化窗口

        st = buffer.stats()                       # 取最终 buffer 统计
        if inference_times:                       # 有耗时样本
            avg_inf = sum(inference_times) / len(inference_times)  # 最终平均推理耗时
        else:                                     # 无样本
            avg_inf = 0.0                         # 记 0
        if loop_times:                            # 有主循环耗时样本
            avg_loop = sum(loop_times) / len(loop_times)  # 最终平均主循环耗时
        else:                                     # 无样本
            avg_loop = 0.0                        # 记 0

        elapsed = time.time() - t0                # 总运行时长
        if elapsed > 0:                           # 运行过 (避免除零)
            avg_fps = st["processed_frames"] / elapsed  # 平均处理 FPS = 处理帧数/时长
        else:                                     # 几乎没运行
            avg_fps = 0.0                         # FPS 记 0

        logger.info("=== 最终统计: recv=%d proc=%d dropped=%d avg_inf=%.1fms "
                    "avg_loop=%.1fms avg_fps=%.1f reconnect=%d 运行=%.1fs ===",   # 最终统计日志 (续行)
                    st["received_frames"], st["processed_frames"],
                    st["dropped_frames"], avg_inf, avg_loop, avg_fps,
                    reader.reconnect_count, elapsed)
        logger.info("=== blade_rtsp_system 退出 ===")  # 退出日志
        log_handler.flush_to_file(log_path)     # 全部缓冲日志一次性落盘

    return 0                                     # 返回 0 = 正常退出


if __name__ == "__main__":                       # 直接运行本文件时
    sys.exit(main())                             # 执行 main 并以返回码退出
