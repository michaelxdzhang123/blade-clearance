#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
frame_buffer.py — 最新帧缓冲模块 (TASK-1 第 7 节)

采用 latest-frame strategy, 默认 max_size=1:
  新帧到来时覆盖尚未被处理的旧帧 (不允许无限积压)。

统计: received_frames / processed_frames / dropped_frames。
"""
from __future__ import annotations

import threading


class FrameBuffer:
    """线程安全的最新帧缓冲。"""

    def __init__(self, max_size: int = 1):
        self.max_size = max(1, max_size)
        self._lock = threading.Lock()
        self._frame = None          # 最新帧 (BGR ndarray)
        self._timestamp = 0.0
        self._frame_id = 0
        self._seq = 0               # 单调递增序列号, 用于覆盖检测

        self.received_frames = 0    # 累计收到帧数
        self.processed_frames = 0   # 累计处理帧数
        self.dropped_frames = 0     # 累计被覆盖(丢弃)帧数

    def put(self, frame, timestamp: float, frame_id: int):
        """写入最新帧。若上一帧未被取走, 计为 dropped。"""
        with self._lock:
            if self._frame is not None:
                self.dropped_frames += 1
            self._frame = frame
            self._timestamp = timestamp
            self._frame_id = frame_id
            self._seq += 1
            self.received_frames += 1

    def get_latest(self):
        """取最新帧。返回 (frame, timestamp, frame_id) 或 (None, None, None)。"""
        with self._lock:
            if self._frame is None:
                return None, None, None
            frame = self._frame
            ts = self._timestamp
            fid = self._frame_id
            self._frame = None      # 取走后清空, 允许下一帧覆盖
            self.processed_frames += 1
            return frame, ts, fid

    def stats(self) -> dict:
        with self._lock:
            return {
                "received_frames": self.received_frames,
                "processed_frames": self.processed_frames,
                "dropped_frames": self.dropped_frames,
            }
