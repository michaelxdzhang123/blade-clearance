#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mp4_to_rtsp_daemon.py — 独立「本地 MP4 → RTSP 推流」守护进程 (视频源侧)

职责: 读取本地 .mp4 文件, 以实时速率无限循环推送到一个 RTSP 服务器,
      模拟一台「永远在线的实时摄像机」, 供 rtsp_daemon.py / main.py 消费。
不做任何读取/推理, 是 rtsp_daemon.py(消费侧)的对称相反端。

数据链:
  MP4 → FFmpeg(-re -stream_loop -1) → MediaMTX(RTSP :8554) → 消费端

两种运行模式:
  1. 推流器 (默认): 假设 RTSP 服务器已在运行, 只负责推流。
       python mp4_to_rtsp_daemon.py --input video.mp4 --rtsp-url rtsp://127.0.0.1:8554/blade
  2. 自包含 (--with-server): 自动拉起 MediaMTX RTSP 服务器 + 推流, 一条命令跑通。
       python mp4_to_rtsp_daemon.py --input video.mp4 --with-server

依赖:
  - ffmpeg (系统 /usr/bin/ffmpeg, 或静态二进制)
  - mediamtx (仅 --with-server 时需要; 生成 source:publisher 配置)

关键参数 (与 rtsp-test-server.md 一致):
  - `-re`            按原视频实时速率读 (不加会瞬间读完)
  - `-stream_loop -1` 无限循环
  - `-c copy`        低 CPU 无损复用原编码; 编码不兼容 RTSP 时用 --re-encode 转 libx264

关于「守护进程」:
  本实现是「后台 FFmpeg 子进程 + 前台监控循环 + 信号优雅退出」。
  若要脱离终端真正后台运行, 用 nohup/systemd 包裹:
      nohup python mp4_to_rtsp_daemon.py --input v.mp4 --with-server >> pub.log 2>&1 &
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import time
from urllib.parse import urlparse

logger = logging.getLogger("mp4_to_rtsp")

DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/blade"  # 默认推流目标 (与 config.yaml 一致)

# MediaMTX 最小配置 (v1.21 两个必配项, 不配会推流 400):
#   - source: publisher        → 允许 FFmpeg 以 publisher 身份推流
#   - rtspTransports: [tcp]    → 强制 TCP, 避开 UDP 8000 端口冲突
MEDIAMTX_CONF_TMPL = """\
logLevel: info
rtsp: true
rtspTransports: [tcp]
rtspAddress: :{port}
rtmp: false
hls: false
webrtc: false
srt: false
paths:
  all_others:
    source: publisher
"""


def find_binary(names: tuple[str, ...]) -> str | None:
    """在 PATH 与常见 .local/bin 位置里找可执行文件, 返回完整路径或 None。

    候选顺序: PATH → 用户真实 home → Hermes profile home。
    (mediamtx/静态 ffmpeg 常被装在 ~/.local/bin, 而 Hermes 的 $HOME 被重定向,
     所以这里显式搜多个 home, 避免「用户脚本找不到二进制」的坑。)
    """
    for name in names:                          # 先看 PATH
        path = shutil.which(name)
        if path:
            return path
    bases = (                                   # 再看常见 .local/bin
        os.path.expanduser("~/.local/bin"),
        "/home/mich/.local/bin",
        "/home/mich/.hermes/profiles/thinker/home/.local/bin",
    )
    for base in bases:
        for name in names:
            cand = os.path.join(base, name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def build_ffmpeg_cmd(ffmpeg: str, input_path: str, rtsp_url: str,
                     loop: bool = True, realtime: bool = True,
                     re_encode: bool = False) -> list[str]:
    """拼装推流命令: 读 MP4 → RTSP。

    - realtime: -re 按原视频速率读 (不加会瞬间读完)
    - loop:     -stream_loop -1 无限循环
    - re_encode: False 用 `-c copy`(无损低 CPU); True 转 libx264 (编码不兼容时)
    """
    cmd = [ffmpeg]
    if realtime:                                # 实时速率读
        cmd.append("-re")
    if loop:                                    # 无限循环
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", input_path]                   # 输入 MP4
    if re_encode:                               # 转码 (兼容性优先)
        cmd += ["-an",                          # 丢弃音频
                "-c:v", "libx264",              # H.264 视频编码
                "-preset", "veryfast",          # 低延迟预设
                "-tune", "zerolatency",         # 零延迟调优 (实时流)
                "-pix_fmt", "yuv420p"]          # 通用像素格式
    else:                                       # 无损复用原编码
        cmd += ["-c", "copy"]
    cmd += ["-f", "rtsp",                       # 输出格式 RTSP
            "-rtsp_transport", "tcp",           # TCP 传输
            rtsp_url]                           # 推流目标
    return cmd


def rtsp_port(rtsp_url: str) -> int:
    """从 rtsp://host:port/path 提取端口, 缺省 8554。"""
    try:
        return urlparse(rtsp_url).port or 8554  # 解析端口
    except ValueError:                          # 端口非法
        return 8554


class Subprocess:
    """一个后台子进程的最小封装: start / poll / stop。"""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None

    def start(self, cmd: list[str], log_path: str | None = None) -> None:
        """启动子进程; 若给 log_path, stdout/stderr 重定向到该文件。"""
        if log_path:                                # 输出到日志文件
            # Popen 会 dup 文件描述符给子进程, with 退出后父进程关句柄不影响子进程写日志
            with open(log_path, "ab") as logf:      # 追加写
                self.proc = subprocess.Popen(
                    cmd, stdout=logf, stderr=subprocess.STDOUT)
        else:                                       # 否则丢弃输出
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def poll(self) -> int | None:
        """返回退出码 (None=仍在运行)。"""
        return self.proc.poll() if self.proc else None

    def stop(self, timeout: float = 5.0) -> None:
        """优雅终止: SIGTERM → 等待 → SIGKILL。"""
        if self.proc is None or self.proc.poll() is not None:  # 已退出
            return
        self.proc.terminate()                   # 发 SIGTERM
        try:
            self.proc.wait(timeout=timeout)     # 等待退出
        except subprocess.TimeoutExpired:       # 超时未退
            self.proc.kill()                    # 强制 SIGKILL
            self.proc.wait()


class MediaMTXServer:
    """可选: 启动 MediaMTX RTSP 服务器 (source:publisher 配置)。"""

    def __init__(self, binary: str, port: int, conf_path: str, log_path: str) -> None:
        self.binary = binary                    # mediamtx 二进制路径
        self.port = port                        # RTSP 监听端口
        self.conf_path = conf_path              # 生成的配置路径
        self.log_path = log_path                # 服务器日志路径
        self._proc = Subprocess()               # 子进程封装

    def start(self) -> bool:
        """写配置并启动服务器。返回是否成功。"""
        with open(self.conf_path, "w", encoding="utf-8") as f:  # 写配置
            f.write(MEDIAMTX_CONF_TMPL.format(port=self.port))
        self._proc.start([self.binary, self.conf_path], self.log_path)  # 启动
        time.sleep(0.5)                         # 给服务器一点启动时间
        if self._proc.poll() is not None:       # 立即退出 = 启动失败
            logger.error("MediaMTX 启动失败, 详见 %s", self.log_path)
            return False
        logger.info("MediaMTX RTSP 服务器已启动 (端口 %d)", self.port)
        return True

    def stop(self) -> None:
        """停止服务器。"""
        self._proc.stop()


def main() -> int:
    """入口: 解析参数 → (可选)拉起服务器 → 推流循环(带重启) → 优雅退出。"""
    parser = argparse.ArgumentParser(description="独立 MP4 → RTSP 推流守护进程")
    parser.add_argument("--input", required=True,
                        help="本地 .mp4 文件路径")
    parser.add_argument("--rtsp-url", default=DEFAULT_RTSP_URL,
                        help="推流目标 RTSP 地址 (默认 rtsp://127.0.0.1:8554/blade)")
    parser.add_argument("--with-server", action="store_true",
                        help="同时拉起 MediaMTX RTSP 服务器 (自包含模式)")
    parser.add_argument("--re-encode", action="store_true",
                        help="强制 libx264 转码 (原编码不兼容 RTSP copy 时用)")
    parser.add_argument("--no-loop", action="store_true",
                        help="不循环 (播完一次即停)")
    parser.add_argument("--max-restarts", type=int, default=0,
                        help="推流崩溃后重启次数上限 (0=无限, 默认 0)")
    parser.add_argument("--restart-backoff", type=float, default=2.0,
                        help="重启前等待秒数 (默认 2)")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg 二进制路径 (默认自动探测)")
    parser.add_argument("--mediamtx", default=None, help="mediamtx 二进制路径 (默认自动探测)")
    parser.add_argument("--max-duration", type=float, default=0.0,
                        help="最大运行秒数 (0=不限)")
    parser.add_argument("--ffmpeg-log", default="/tmp/mp4_to_rtsp_ffmpeg.log",
                        help="ffmpeg 日志路径")
    parser.add_argument("--mediamtx-log", default="/tmp/mp4_to_rtsp_mediamtx.log",
                        help="mediamtx 日志路径")
    args = parser.parse_args()

    logging.basicConfig(                          # 日志到 stderr
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.isfile(args.input):            # 输入文件必须存在
        logger.error("输入文件不存在: %s", args.input)
        return 1

    ffmpeg = args.ffmpeg or find_binary(("ffmpeg",))  # 定位 ffmpeg
    if not ffmpeg:                                # 找不到
        logger.error("未找到 ffmpeg, 请用 --ffmpeg 指定路径")
        return 1
    logger.info("使用 ffmpeg: %s", ffmpeg)

    server = None                                 # MediaMTX 服务器 (可选)
    if args.with_server:                          # 自包含模式: 先起服务器
        mediamtx = args.mediamtx or find_binary(("mediamtx",))
        if not mediamtx:                          # 找不到 mediamtx
            logger.error("未找到 mediamtx, 请用 --mediamtx 指定路径")
            return 1
        port = rtsp_port(args.rtsp_url)           # 从目标 URL 提取端口
        server = MediaMTXServer(                  # 构造服务器
            binary=mediamtx, port=port,
            conf_path="/tmp/mp4_to_rtsp_mediamtx.yml",
            log_path=args.mediamtx_log,
        )
        if not server.start():                    # 启动服务器失败
            return 1

    # 拼装推流命令
    cmd = build_ffmpeg_cmd(                       # 读 MP4 → RTSP
        ffmpeg=ffmpeg, input_path=args.input, rtsp_url=args.rtsp_url,
        loop=not args.no_loop, realtime=True, re_encode=args.re_encode,
    )
    logger.info("推流命令: %s", " ".join(cmd))
    logger.info("推流目标: %s (loop=%s re_encode=%s)",
                args.rtsp_url, not args.no_loop, args.re_encode)

    publisher = Subprocess()                      # ffmpeg 推流子进程
    publisher.start(cmd, args.ffmpeg_log)         # 首次启动推流

    restarts = 0                                  # 重启计数
    t0 = time.time()                              # 启动时刻
    running = True                                # 运行标志

    def _shutdown(signum=None, _frame=None):      # 信号处理器
        nonlocal running
        if signum is not None:                    # 有信号号
            logger.info("收到信号 %d, 停止推流...", signum)
        running = False

    signal.signal(signal.SIGINT, _shutdown)       # Ctrl+C
    signal.signal(signal.SIGTERM, _shutdown)      # kill / systemd stop

    logger.info("=== 推流运行中 (Ctrl+C 停止) ===")
    try:
        while running:                            # 监控循环
            # 定时长跑检查
            if args.max_duration > 0 and (time.time() - t0) >= args.max_duration:
                logger.info("达到最大运行时长 %.0fs, 正常停止", args.max_duration)
                break

            ret = publisher.poll()                # 查 ffmpeg 是否退出
            if ret is None:                       # 仍在运行
                time.sleep(0.2)                   # 短暂轮询
                continue

            # ffmpeg 已退出 (崩溃 / 推流失败 / 播完未循环)
            if not running:                       # 是用户要求停止
                break
            logger.warning("推流进程退出 (code=%s), 详见 %s", ret, args.ffmpeg_log)
            restarts += 1                         # 重启计数 +1
            if args.max_restarts > 0 and restarts >= args.max_restarts:  # 达上限
                logger.error("达到重启上限 %d, 退出", args.max_restarts)
                break
            logger.warning("第 %d 次重启 (等待 %.1fs)...",
                           restarts, args.restart_backoff)
            time.sleep(args.restart_backoff)      # 退避等待
            publisher.start(cmd, args.ffmpeg_log)  # 重启推流
    finally:
        publisher.stop()                          # 停 ffmpeg (SIGTERM→SIGKILL)
        if server:                                # 停 MediaMTX (若启动)
            server.stop()
        elapsed = time.time() - t0                # 总运行时长
        logger.info("=== 退出: 重启=%d 运行=%.1fs ===", restarts, elapsed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
