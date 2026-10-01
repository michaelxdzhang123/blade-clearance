#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
opencv/catch_blade.py — 按任务描述先验逐帧抓叶片, 标绿色框 (经典 CV, 无 YOLO)

任务描述 (docs/task-description.md) 的核心先验:
  1. 叶片 = 视频中最大的运动物体, 其他都是固定的
  2. 塔筒 = 最大的固定物体, 在图像上层
  3. 叶片进入摄像头在下层
  4. 塔筒面积 > 屏幕 1/5

检测方法 (背景差分 + 几何先验, 全无监督):
  背景 = IDLE 帧 median (bg_idle_median.npy)。塔筒静止、叶片运动,
  故 |img - bg| 在叶片处大、在塔筒处≈0, 天然分离两者。
  - 下层: 只扫 y > 800 (塔筒在上层 y<564)
  - 触底: 叶尖横扫到画面最底端 y≈1436, 叶片框必触底 (y+h >= 1430)
  - 面积: 连通域面积 > 5000 (塔筒阴影等噪声 <3000)
  取触底最大连通域 = 叶片, 画绿色框 BGR(0,255,0)。

用法:
  python opencv/catch_blade.py --video data/test01-video.mp4 --output blade-box [--bg ...]

输出 (逐帧迭代全视频):
  <output>/annotated.mp4     逐帧绿色框标注视频
  <output>/blade_boxes.csv   每帧叶片框 (frame, x, y, w, h)
"""
import argparse
import csv
import os

import av
import cv2
import numpy as np

GREEN = (0, 255, 0)
THICKNESS = 4
DIFF_THR = 25
Y_MIN = 800          # 叶片在下层, 排除上层塔筒
BOTTOM_REACH = 1430  # 叶片框必须触底 (叶尖 y≈1436)
MIN_AREA = 5000      # 叶片=最大运动物体; 真叶片横扫带 2万~23万, 塔筒阴影噪声 <3000


def blade_boxes(img, bg, return_polygon=False, max_poly_verts=50):
    """背景差分找叶片, 支持两种返回模式。

    return_polygon=False (默认): 返回触底最大连通域 bbox (x,y,w,h) 或 None。
    return_polygon=True:         返回该连通域的 polygon 轮廓顶点
                                  [[x1,y1],[x2,y2],...] (像素坐标) 或 None。
    max_poly_verts=50:           polygon 顶点数上限 (approxPolyDP 简化到 ≤ 此值)。

    原理: connectedComponentsWithStats 返回 labels(像素级 mask 信息),
    bbox 模式只用 stats(矩形统计) 丢弃 labels; polygon 模式用 labels
    取最大连通域的 mask, 再 findContours 提取轮廓, 最后 approxPolyDP
    简化(背景差分边缘是锯齿状噪声, 原始轮廓可达 1000+ 顶点, 需压到 ~50)。
    """
    # Geometry thresholds are defined for the original 2560x1440 camera
    # stream. Scale them for resized training videos (for example 640x360).
    height, width = img.shape[:2]
    scale_x = width / 2560.0
    scale_y = height / 1440.0
    y_min = max(1, int(round(Y_MIN * scale_y)))
    bottom_reach = int(round(BOTTOM_REACH * scale_y))
    min_area = max(1, int(round(MIN_AREA * scale_x * scale_y)))

    diff = img - bg
    moving = np.abs(diff) > DIFF_THR
    moving[:y_min, :] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        moving.astype(np.uint8), 8)
    best = None  # (area, x, y, w, h, label_id)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if y + h >= bottom_reach and a > min_area:
            if best is None or a > best[0]:
                best = (a, x, y, w, h, i)
    if best is None:
        return None
    if return_polygon:
        # 从最大连通域的 mask 提取 polygon 轮廓
        mask = (labels == best[5]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            # approxPolyDP 自适应简化: 逐步增大 epsilon(周长比例), 直到顶点数 ≤ 上限。
            # 背景差分的叶片边缘是锯齿状噪声, 原始 findContours 可达 1000+ 顶点,
            # YOLO-seg 训练推荐 <100 顶点, 故简化到 ~50 保留叶片整体形状。
            peri = cv2.arcLength(c, True)
            approx = c
            for ratio in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
                approx = cv2.approxPolyDP(c, ratio * peri, True)
                if len(approx) <= max_poly_verts:
                    break
            return approx.reshape(-1, 2).tolist()   # [[x1,y1],[x2,y2],...]
        return None
    return best[1:5]  # (x, y, w, h)


def main():
    parser = argparse.ArgumentParser(description="逐帧抓叶片并标绿色框")
    parser.add_argument("--video", required=True, help="输入视频路径 (*.mp4)")
    parser.add_argument("--output", default="blade-box", help="输出目录 (默认 blade-box)")
    parser.add_argument("--bg", default="outputs/TASK2_GATE_CLOSURE/bg_idle_median.npy",
                        help="背景模型 .npy 路径 (默认 IDLE median 背景)")
    parser.add_argument("--polygon", action="store_true",
                        help="输出 polygon 轮廓 (默认输出 bbox 矩形框)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    bg = np.load(args.bg).astype(np.float32)

    c = av.open(args.video)
    s = c.streams.video[0]
    fps = float(s.average_rate or 25.0)
    w = s.codec_context.width
    h = s.codec_context.height
    total = s.frames or 0

    video_path = os.path.join(args.output, "annotated.mp4")
    if args.polygon:
        out_path = os.path.join(args.output, "blade_polygons.json")
    else:
        out_path = os.path.join(args.output, "blade_boxes.csv")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    # bbox 模式: 打开 CSV + 写表头; polygon 模式: 不写 CSV, 最后统一写 JSON
    csvw = None
    csvf = None
    if not args.polygon:
        csvf = open(out_path, "w", newline="")
        csvw = csv.writer(csvf)
        csvw.writerow(["frame", "x", "y", "w", "h"])

    print(f"视频: {args.video}  ({w}x{h}, {fps:.1f}fps, {total} 帧)")
    print(f"背景: {args.bg}")
    print(f"模式: {'polygon 轮廓' if args.polygon else 'bbox 矩形框'}")
    print(f"先验: 叶片=最大运动物体(下层 y>{Y_MIN}), 框触底>{BOTTOM_REACH}, 面积>{MIN_AREA}\n")

    frame_idx = 0
    n_blade = 0
    polygons = {}  # polygon 模式: {frame: [[x1,y1],...]}
    for frame in c.decode(s):
        img = frame.to_ndarray(format="gray").astype(np.float32)
        result = blade_boxes(img, bg, return_polygon=args.polygon)

        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if result is not None:
            if args.polygon:
                # polygon 模式: 画多边形轮廓 + 存 JSON
                pts = np.array(result, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(bgr, [pts], isClosed=True, color=GREEN, thickness=THICKNESS)
                polygons[frame_idx] = result
            else:
                # bbox 模式: 画矩形 + 写 CSV
                x, y, w, hh = result
                cv2.rectangle(bgr, (x, y), (x + w, y + hh), GREEN, THICKNESS)
                cv2.putText(bgr, "blade", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
                csvw.writerow([frame_idx, x, y, w, hh])
            n_blade += 1
        writer.write(bgr)

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{total} 帧  已标叶片 {n_blade} 帧", flush=True)

    c.close()
    writer.release()
    if args.polygon:
        import json
        with open(out_path, "w") as f:
            json.dump(polygons, f)
    else:
        csvf.close()

    print(f"\n完成: {frame_idx} 帧, 标出叶片 {n_blade} 帧")
    print(f"标注视频 → {video_path}")
    print(f"叶片{'polygon' if args.polygon else '框'} → {out_path}")


if __name__ == "__main__":
    main()
