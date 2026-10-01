#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rtsp_daemon.py — 独立 RTSP 视频源守护进程 (仅视频源, 与 blade 推理解耦)

职责: 只负责「连接 RTSP → 持续读帧 → 断流自动重连 → 供外部取最新帧」。
不做任何 YOLO / 叶片定位 / 坐标计算, 可单独运行, 也可被其他进程 import 复用。

自包含设计: 不依赖 blade_rtsp_system/src 任何模块, 单文件即可运行。

两种用法:
  1. 独立运行 (守护读流, 周期统计, 可 --save-dir 存帧验证):
       python rtsp_daemon.py --url rtsp://127.0.0.1:8554/blade
  2. 作为库复用 (你的推理进程 import 它取最新帧):
       from rtsp_daemon import RTSPDaemon
       d = RTSPDaemon(url).start()
       frame, ts, fid = d.get_latest()   # 或 d.peek() 只读不取走
       d.stop()

线程模型 (与 main.py 的 Reader 线程一致):
  后台 daemon 线程: 读帧 → 写入 latest-frame buffer (max_size=1, 新帧覆盖旧帧)
  主线程/消费者:   从 buffer 取最新帧 (get_latest), 推理慢时自动丢旧帧

日志策略 (关键):
  日志运行时只缓冲到内存 list (零 I/O, 不写 stderr/磁盘), 进程退出时一次性
  写 --log-file。避免周期统计/读帧告警等逐条 I/O 拖慢读流; 代价是硬 kill
  (SIGKILL) 会丢缓冲日志, 正常退出 (Ctrl+C/SIGTERM/--max-duration) 都会落盘。

关于「守护进程」:
  本实现是「后台 daemon 线程 + 前台主循环 + 信号优雅退出」。
  若要脱离终端真正后台运行, 用 nohup/systemd 包裹即可 (日志走 --log-file)。
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time

import cv2

logger = logging.getLogger("rtsp_daemon")

DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/blade"  # 默认测试地址 (与 config.yaml 一致)
DEFAULT_LOG_FILE = "/tmp/rtsp_daemon.log"          # 默认日志落盘路径 (退出时一次性写入)
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class BufferedFileHandler(logging.Handler):
    """把日志记录缓冲到内存 list, 运行时零 I/O, 结束时一次性写文件。

    用法:
      h = BufferedFileHandler(LOG_FORMAT)
      logging.basicConfig(level=logging.INFO, handlers=[h])
      ... 运行期间 logger.info() 只追加到 list, 不碰磁盘/终端 ...
      h.flush_to_file(path)   # 结束时一次性落盘

    线程安全: logging.Handler.handle() 自带 RLock, emit() 在锁内执行,
    因此 Reader 线程与主线程并发 logger 调用安全。
    """

    def __init__(self, fmt: str) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(fmt))  # 复用与原 stderr 一致的格式
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


class LatestFrameBuffer:
    """线程安全的最新帧缓冲 (latest-frame 策略, max_size=1)。

    新帧到来时覆盖尚未被取走的旧帧, 并计为 dropped, 避免推理慢于流 FPS 时无限积压。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()       # 保护以下所有状态的互斥锁
        self._frame = None                  # 最新帧 (BGR ndarray, 原始分辨率)
        self._timestamp = 0.0               # 该帧时间戳
        self._frame_id = 0                  # 该帧序号

        self.received = 0                   # 累计写入帧数
        self.processed = 0                  # 累计被取走帧数
        self.dropped = 0                    # 累计被覆盖(丢弃)帧数

    def put(self, frame, timestamp: float, frame_id: int) -> None:
        """写入最新帧。若上一帧还没被取走, 计为 dropped。"""
        with self._lock:
            if self._frame is not None:     # 上一帧未被消费, 即将被覆盖
                self.dropped += 1           # 丢弃计数 +1
            self._frame = frame             # 覆盖为最新帧
            self._timestamp = timestamp     # 记录时间戳
            self._frame_id = frame_id       # 记录帧序号
            self.received += 1              # 收到计数 +1

    def get_latest(self):
        """取走最新帧。返回 (frame, timestamp, frame_id) 或 None(空)。"""
        with self._lock:
            if self._frame is None:         # buffer 为空
                return None
            frame = self._frame             # 取出帧
            ts = self._timestamp            # 取出时间戳
            fid = self._frame_id            # 取出帧序号
            self._frame = None              # 取走后清空, 允许下一帧写入
            self.processed += 1             # 处理计数 +1
            return frame, ts, fid

    def peek(self):
        """只读最新帧但不取走 (供多消费者观察, 不影响 dropped/processed 语义)。

        返回 (frame, timestamp, frame_id) 或 None(空)。
        """
        with self._lock:
            if self._frame is None:         # buffer 为空
                return None
            return self._frame, self._timestamp, self._frame_id

    def stats(self) -> dict:
        """返回累计统计: received / processed / dropped。"""
        with self._lock:
            return {
                "received": self.received,
                "processed": self.processed,
                "dropped": self.dropped,
            }


class RTSPReader:
    """最小 RTSP 读取器, 带自动重连。

    断流重连语义 (血泪教训, 与 blade_rtsp_system/src/rtsp_reader.py 保持一致):
      - reconnect()            = 单次尝试 (release → wait → open), 返回本次是否连上
      - reconnect_exhausted()  = 上限判断 (max_reconnect_attempts>0 且 count>=attempts)
      - max_reconnect_attempts=0 → 永远 False (无限重连, 推流暂时 404 也不退出)
    调用方循环: while running: if exhausted(): break; if reconnect(): break
    """

    def __init__(self, source: str, reconnect_interval_sec: float = 2.0,
                 max_reconnect_attempts: int = 0) -> None:
        self.source = source                        # RTSP 地址
        self.reconnect_interval_sec = reconnect_interval_sec  # 重连等待间隔
        self.max_reconnect_attempts = max_reconnect_attempts  # 重连上限 (0=无限)
        self.cap: cv2.VideoCapture | None = None    # OpenCV 视频捕获对象
        self.is_rtsp = source.lower().startswith("rtsp://")  # 是否 RTSP (本模块恒 True)
        self.reconnect_count = 0                    # 累计重连次数
        self.frame_id = 0                           # 累计读取帧序号
        self.fps = 0.0                              # 视频源 FPS

    def open(self) -> bool:
        """打开视频源。返回是否成功 (失败不抛异常)。"""
        backend = cv2.CAP_FFMPEG                    # RTSP 用 FFMPEG 后端
        self.cap = cv2.VideoCapture(self.source, backend)
        if not self.cap.isOpened():                 # 打开失败
            logger.error("无法打开视频源: %s", self._masked_source())
            self.cap = None
            return False
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # 读取帧宽
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # 读取帧高
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)  # 读取 FPS
        logger.info("视频源已打开: %s (%dx%d @ %.2f fps)",
                    self._masked_source(), w, h, self.fps)
        return True

    def read(self):
        """读一帧。返回 (success, frame, timestamp)。

        frame 为 BGR ndarray (原始分辨率, 未 resize); 失败时 success=False。
        """
        if self.cap is None:                        # 未打开
            return False, None, time.time()
        ret, frame = self.cap.read()                # 读一帧
        if not ret or frame is None:                # 读帧失败
            logger.warning("读帧失败 (frame_id=%d)", self.frame_id)
            return False, None, time.time()
        self.frame_id += 1                          # 帧序号 +1
        return True, frame, time.time()             # 成功返回 (帧, 接收时刻)

    def reconnect(self) -> bool:
        """单次重连尝试: release → wait → open。返回本次是否成功连上。"""
        self.reconnect_count += 1                   # 重连计数 +1
        logger.warning("断流, 第 %d 次重连 (等待 %.1fs) ...",
                       self.reconnect_count, self.reconnect_interval_sec)
        self.release()                              # 释放旧连接
        time.sleep(self.reconnect_interval_sec)     # 等待间隔
        return self.open()                          # 重新打开

    def reconnect_exhausted(self) -> bool:
        """是否已达重连上限。max_reconnect_attempts=0 → 永远 False(无限重连)。"""
        return (self.max_reconnect_attempts > 0
                and self.reconnect_count >= self.max_reconnect_attempts)

    def release(self) -> None:
        """释放资源。"""
        if self.cap is not None:
            self.cap.release()                      # 释放 VideoCapture
            self.cap = None                         # 置空

    def _masked_source(self) -> str:
        """脱敏输出: rtsp://user:pass@host → rtsp://user:***@host。"""
        s = self.source
        if "://" in s and "@" in s:                 # 含凭据才需要脱敏
            scheme, rest = s.split("://", 1)
            cred, host = rest.split("@", 1)
            if ":" in cred:                         # 用户名:密码 形式
                user = cred.split(":", 1)[0]        # 只保留用户名
                return f"{scheme}://{user}:***@{host}"
        return s


class RTSPDaemon:
    """RTSP 视频源守护进程: 后台线程持续读流 → 最新帧缓冲, 带断流重连。

    用法:
      d = RTSPDaemon(url).start()
      frame, ts, fid = d.get_latest()   # 取最新帧 (消费)
      d.peek()                          # 只读不取走
      d.stats()                         # received/processed/dropped/reconnect
      d.stop()                          # 优雅停止
    """

    def __init__(self, url: str, reconnect_interval_sec: float = 2.0,
                 max_reconnect_attempts: int = 0) -> None:
        self.url = url                                    # RTSP 地址
        self._reader = RTSPReader(                        # 内部读取器
            source=url,
            reconnect_interval_sec=reconnect_interval_sec,
            max_reconnect_attempts=max_reconnect_attempts,
        )
        self._buffer = LatestFrameBuffer()                # 最新帧缓冲
        self._running = threading.Event()                 # 运行标志
        self._thread: threading.Thread | None = None      # 后台读流线程

    def start(self) -> "RTSPDaemon":
        """启动守护: 打开视频源并起后台读流线程。失败抛 RuntimeError。"""
        if not self._reader.open():                       # 首次连接失败
            raise RuntimeError(f"无法打开视频源: {self._reader._masked_source()}")
        self._running.set()                               # 置运行中
        self._thread = threading.Thread(                  # 创建 daemon 线程
            target=self._reader_loop, name="rtsp-reader", daemon=True)
        self._thread.start()                              # 启动线程
        logger.info("RTSP 守护已启动: %s", self._reader._masked_source())
        return self

    def _reader_loop(self) -> None:
        """后台读流主循环 (与 main.py reader_loop 相同的重连语义)。"""
        while self._running.is_set():                     # 运行期间持续读
            ok, frame, ts = self._reader.read()           # 读一帧
            if not ok:                                    # 读失败 = 断流
                # 断流: 循环重连直到成功或达到上限 (无限模式永不放弃)
                while self._running.is_set():
                    if self._reader.reconnect_exhausted():  # 达上限
                        logger.error("达到最大重连次数 %d, 退出",
                                     self._reader.max_reconnect_attempts)
                        self._running.clear()             # 通知停止
                        return
                    if self._reader.reconnect():          # 尝试重连
                        break                             # 重连成功
                continue                                  # 继续读帧
            # 写入最新帧缓冲 (新帧覆盖未消费旧帧)
            self._buffer.put(frame, ts, self._reader.frame_id)

    def get_latest(self):
        """取走最新帧 (消费)。返回 (frame, timestamp, frame_id) 或 None。"""
        return self._buffer.get_latest()

    def peek(self):
        """只读最新帧但不取走。返回 (frame, timestamp, frame_id) 或 None。"""
        return self._buffer.peek()

    def stats(self) -> dict:
        """返回统计: received/processed/dropped + reconnect_count。"""
        s = self._buffer.stats()                          # 取缓冲统计
        s["reconnect_count"] = self._reader.reconnect_count  # 附加重连次数
        return s

    def stop(self) -> None:
        """优雅停止: 通知线程退出 → 等待 → 释放视频源。"""
        self._running.clear()                             # 通知读流线程退出
        if self._thread is not None:                      # 线程存在才 join
            self._thread.join(timeout=5)                  # 最多等 5 秒
        self._reader.release()                            # 释放视频源

    def __enter__(self) -> "RTSPDaemon":
        """上下文管理器入口: 启动守护。"""
        return self.start()

    def __exit__(self, *exc) -> None:
        """上下文管理器出口: 停止守护。"""
        self.stop()


def _install_signal_handlers(daemon: RTSPDaemon) -> None:
    """安装 SIGINT/SIGTERM 处理器, 保证 Ctrl+C / kill 时优雅退出。"""
    def _handle(signum, _frame):                          # 收到信号
        logger.info("收到信号 %d, 停止守护...", signum)
        daemon.stop()                                     # 停止守护
        raise SystemExit(0)                               # 触发主循环退出 (走 finally)
    signal.signal(signal.SIGINT, _handle)                 # Ctrl+C
    signal.signal(signal.SIGTERM, _handle)                # kill / systemd stop


def main() -> int:
    """独立运行入口: 起守护 → 周期 drain 帧 + 统计 → 信号/时长优雅退出。"""
    parser = argparse.ArgumentParser(description="独立 RTSP 视频源守护进程")
    parser.add_argument("--url", default=DEFAULT_RTSP_URL,
                        help="RTSP 地址 (默认 rtsp://127.0.0.1:8554/blade)")
    parser.add_argument("--reconnect-interval", type=float, default=2.0,
                        help="断流重连等待间隔秒 (默认 2)")
    parser.add_argument("--max-reconnect-attempts", type=int, default=0,
                        help="重连次数上限 (0=无限, 默认 0)")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="打印统计的间隔秒 (默认 5)")
    parser.add_argument("--max-duration", type=float, default=0.0,
                        help="最大运行秒数 (0=不限)")
    parser.add_argument("--save-dir", default=None,
                        help="可选: 每帧存为 jpg 到该目录 (验证流是否有帧)")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE,
                        help="日志落盘路径 (运行时缓冲到内存, 退出时一次性写入)")
    args = parser.parse_args()

    # 日志缓冲到内存 list (运行时零 I/O), 退出时一次性写 --log-file。
    # 不再写 stderr, 避免周期统计/读帧告警的逐条 I/O 拖慢读流。
    log_handler = BufferedFileHandler(LOG_FORMAT)
    logging.basicConfig(level=logging.INFO, handlers=[log_handler])

    daemon = RTSPDaemon(                                  # 构造守护 (不启动)
        url=args.url,
        reconnect_interval_sec=args.reconnect_interval,
        max_reconnect_attempts=args.max_reconnect_attempts,
    )

    try:
        daemon.start()                                    # 启动 (首次连接失败抛异常)
    except RuntimeError as exc:                           # 首次连接失败
        logger.error("%s", exc)
        log_handler.flush_to_file(args.log_file)          # 早退也要落盘
        return 1

    _install_signal_handlers(daemon)                      # 装信号处理器
    if args.save_dir:                                     # 需要存帧验证
        os.makedirs(args.save_dir, exist_ok=True)         # 创建目录

    t0 = time.time()                                      # 启动时刻
    last_report = t0                                      # 上次统计时刻
    saved = 0                                             # 已存帧数

    logger.info("=== RTSP 守护运行中 (Ctrl+C 停止) ===")
    try:
        while True:                                       # 主循环: 消费帧 + 统计
            item = daemon.get_latest()                    # 取最新帧 (消费)
            if item is None:                              # 暂无帧
                time.sleep(0.01)                          # 短暂等待
            else:                                         # 有帧
                frame, ts, fid = item                     # 解包
                if args.save_dir:                         # 存帧验证
                    path = os.path.join(args.save_dir, f"frame_{fid:06d}.jpg")
                    cv2.imwrite(path, frame)              # 写 jpg
                    saved += 1

            now = time.time()                             # 当前时刻
            if now - last_report >= args.stats_interval:  # 到统计周期
                s = daemon.stats()                        # 取统计
                logger.info(
                    "统计: received=%d processed=%d dropped=%d reconnect=%d",
                    s["received"], s["processed"], s["dropped"], s["reconnect_count"])
                last_report = now                         # 更新统计时刻

            if args.max_duration > 0 and (now - t0) >= args.max_duration:  # 到时长
                logger.info("达到最大运行时长 %.0fs, 正常停止", args.max_duration)
                break
    finally:
        daemon.stop()                                     # 无论如何都优雅停止
        s = daemon.stats()                                # 最终统计
        elapsed = time.time() - t0                        # 总运行时长
        avg_fps = s["processed"] / elapsed if elapsed > 0 else 0.0  # 平均处理 FPS
        logger.info("=== 最终统计: received=%d processed=%d dropped=%d "
                    "reconnect=%d avg_fps=%.1f 运行=%.1fs 存帧=%d ===",
                    s["received"], s["processed"], s["dropped"],
                    s["reconnect_count"], avg_fps, elapsed, saved)
        log_handler.flush_to_file(args.log_file)          # 全部缓冲日志一次性落盘

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
