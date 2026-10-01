#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_rtsp.py — 用 OpenCV 验证 RTSP 链路 (rtsp://127.0.0.1:8554/blade)

数据链: MP4 -> FFmpeg -> MediaMTX -> RTSP -> cv2.VideoCapture -> Frame

用法:
    python scripts/verify_rtsp.py [rtsp_url]

默认地址: rtsp://127.0.0.1:8554/blade
按 'q' 退出。
"""
import sys
import time

import cv2

DEFAULT_URL = "rtsp://127.0.0.1:8554/blade"


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"[verify] RTSP url: {url}")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[verify] ERROR: 无法打开 RTSP 流 (MediaMTX/FFmpeg 是否已启动?)")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[verify] 已连接: {width}x{height} @ {fps:.2f} FPS")

    frames = 0
    failures = 0
    t0 = time.time()
    last_report = t0

    while True:
        ret, frame = cap.read()
        if not ret:
            failures += 1
            print(f"[verify] RTSP read failed (#{failures})")
            if failures >= 10:
                print("[verify] 连续读帧失败，退出")
                break
            time.sleep(0.5)
            continue

        failures = 0
        frames += 1

        now = time.time()
        if now - last_report >= 5.0:
            dt = now - t0
            print(f"[verify] frames={frames}  recv_fps={frames/dt:.1f}  "
                  f"shape={frame.shape}")
            last_report = now

        cv2.imshow("RTSP TEST", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"[verify] 结束: 共 {frames} 帧, 平均 {frames/dt:.1f} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
