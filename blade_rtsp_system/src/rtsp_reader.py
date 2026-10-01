#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rtsp_reader.py — RTSP 读取模块 (TASK-1 第 6 节)

职责: RTSP → Open → Read Frame → Timestamp → Reconnect

要求:
  - 支持 RTSP 实时读取 + 本地 MP4 调试 (TASK-1 第 27 节 --source local.mp4)
  - 短时中断后自动重连 (max_reconnect_attempts=0 表示无限重连)
  - 保留原始帧分辨率 (禁止在 reader 中 resize 到 640)
  - 记录 frame_id 和 timestamp
  - 首次连接失败不崩溃
"""
from __future__ import annotations

import logging
import time

import cv2

logger = logging.getLogger("blade_rtsp.rtsp_reader")


class RTSPReader:
    """RTSP / 本地视频读取器, 带自动重连。"""

    def __init__(self, source: str, reconnect_interval_sec: float = 2.0,
                 max_reconnect_attempts: int = 0):
        self.source = source
        self.reconnect_interval_sec = reconnect_interval_sec
        self.max_reconnect_attempts = max_reconnect_attempts  # 0 = 无限
        self.cap: cv2.VideoCapture | None = None
        self.is_rtsp = source.lower().startswith("rtsp://")
        self.reconnect_count = 0
        self.frame_id = 0
        self.fps = 0.0

    def open(self) -> bool:
        """打开视频源。返回是否成功。失败不抛异常。"""
        backend = cv2.CAP_FFMPEG if self.is_rtsp else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.source, backend)
        if not self.cap.isOpened():
            logger.error("无法打开视频源: %s", self._masked_source())
            self.cap = None
            return False
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        logger.info("视频源已打开: %s (%dx%d @ %.2f fps)",
                    self._masked_source(), w, h, self.fps)
        return True

    def read(self):
        """读一帧。返回 (success, frame, timestamp)。

        frame 为 BGR ndarray (原始分辨率, 未 resize)。失败时 success=False。
        """
        if self.cap is None:
            return False, None, time.time()
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("读帧失败 (frame_id=%d)", self.frame_id)
            return False, None, time.time()
        self.frame_id += 1
        return True, frame, time.time()

    def reconnect(self) -> bool:
        """单次重连尝试: release → wait → open。返回本次是否成功连上。

        注意: 不在这里做"达到上限就停止"的判断——上限判断由调用方
        (main.py reader_loop) 根据 reconnect_exhausted() 决定是否继续。
        max_reconnect_attempts=0 表示无限重连, 即使本次 open 失败(如推流
        尚未重新发布导致 404)也应继续重试, 而不是退出。
        """
        self.reconnect_count += 1
        logger.warning("断流, 第 %d 次重连 (等待 %.1fs) ...",
                       self.reconnect_count, self.reconnect_interval_sec)
        self.release()
        time.sleep(self.reconnect_interval_sec)
        return self.open()

    def reconnect_exhausted(self) -> bool:
        """是否已达重连上限 (max_reconnect_attempts>0 且 count>=attempts)。

        max_reconnect_attempts=0 → 永远 False (无限重连)。
        """
        return self.max_reconnect_attempts > 0 and \
            self.reconnect_count >= self.max_reconnect_attempts

    def release(self):
        """释放资源。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _masked_source(self) -> str:
        """脱敏输出: rtsp://user:pass@host → rtsp://user:***@host。"""
        s = self.source
        if "://" in s and "@" in s:
            scheme, rest = s.split("://", 1)
            cred, host = rest.split("@", 1)
            if ":" in cred:
                user = cred.split(":", 1)[0]
                return f"{scheme}://{user}:***@{host}"
        return s
