#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_tip_crop.py — 叶尖 crop 放大方案的可视化验证脚本(不训练, 只 crop + 保存图)

目的:
  验证"训练时只喂叶尖附近的高清 crop"这一 cascade 方案是否可行。
  用现有 Stage1 best.pt(1024 训练的 segment 模型)在 2560x1440 高清帧上推理,
  找到叶尖最前端, 从原始高清图(不经 resize)crop 出 512x512 子图保存,
  让用户目视判断: 叶尖在 crop 里是否清晰、是否携带足够细节。

叶尖定义 (依据 opencv/refine_blade_tip.py 的诊断结论):
  叶尖 = 叶片 mask 上离轮毂(塔筒顶端)最远的顶点(叶片自由端)。
  叶尖是"暗叶片→亮天空"的一维灰度阶跃边缘, 不是角点。
  塔筒贯穿画面 x[1075,1625] 中心≈1350, 轮毂在塔筒顶端 (1350,~100);
  叶片横扫, 叶尖触底 y≈1436 时离塔筒最近(净空关键帧)。

流程:
  1. 抽高清帧 (2560x1440)
  2. best.pt 推理 → mask polygon (归一化坐标)
  3. 叶尖点 = mask 上离轮毂最远的顶点 (映射回原图像素坐标)
  4. 过滤: 叶尖 y < min-tip-y(默认1200)的实例跳过(塔筒误检/横扫中间态)
  5. 从原始高清帧 crop 512x512 (越界则 pad, 不 resize)
  6. 保存: crop 图 + 原图标注(mask 轮廓 + 叶尖点 + crop 框)

用法:
  uv run python verify_tip_crop.py --video data/training-data/test01-xxx.mp4 \
      --weights train-seg-draft-0916/training/stage1_auto_labeled_scratch/weights/best.pt \
      --frame 3000 --out crop-verify
"""
import argparse
import os

import av
import cv2
import numpy as np
from ultralytics import YOLO


def find_tip_point(polygon, H, W, hub=(1350.0, 100.0)):
    """从归一化 mask polygon 找叶尖点(像素坐标)。

    叶尖 = 叶片 mask 上离轮毂(塔筒顶端)最远的顶点(叶片自由端)。
    这比"y 最大顶点"更鲁棒: 叶片竖直向下时叶尖=最低点, 但横扫中间态
    时叶尖仍在离轮毂最远处, 而 y 最大顶点会错位到叶片侧边。

    polygon: (N, 2) 归一化 [x, y] ∈ [0,1]
    hub: 轮毂像素坐标(塔筒顶端中心), 默认 (1350, 100)
    返回 (x_px, y_px) 像素坐标。
    """
    xy = np.asarray(polygon, dtype=np.float32)
    px = xy[:, 0] * W
    py = xy[:, 1] * H
    hub_xy = np.array(hub, dtype=np.float32)
    dist = np.hypot(px - hub_xy[0], py - hub_xy[1])
    tip_idx = int(np.argmax(dist))
    return int(round(px[tip_idx])), int(round(py[tip_idx]))


def crop_square(img, cx, cy, size=512):
    """以 (cx, cy) 为中心 crop size×size 子图, 越界用灰边(114) pad, 不 resize。"""
    H, W = img.shape[:2]
    half = size // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    # 源图内可见区域
    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)
    crop = np.full((size, size, 3), 114, dtype=np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        # pad 后粘贴到目标区域
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return crop, (sx0, sy0, sx1, sy1)


def main():
    ap = argparse.ArgumentParser(description="叶尖 crop 可视化验证")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frame", type=int, default=3000, help="抽帧索引(默认 3000)")
    ap.add_argument("--out", default="crop-verify")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--imgsz", type=int, default=1024, help="推理尺寸(与 best.pt 训练一致)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--hub-x", type=float, default=1350.0, help="轮毂 x(塔筒中心)")
    ap.add_argument("--hub-y", type=float, default=100.0, help="轮毂 y(塔筒顶端)")
    ap.add_argument("--min-tip-y", type=float, default=1200.0,
                    help="叶尖最低 y 阈值: 净空关键帧的叶尖触底(y≈1436), "
                         "塔筒误检 tip y≈500、横扫中间态叶尖未触底, 都会被过滤")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)

    # ── 抽帧 ──
    c = av.open(args.video)
    s = c.streams.video[0]
    W = s.codec_context.width
    H = s.codec_context.height
    print(f"视频 {W}x{H}, 抽第 {args.frame} 帧")

    frame_bgr = None
    for i, f in enumerate(c.decode(s)):
        if i == args.frame:
            frame_bgr = f.to_ndarray(format="bgr24")
            break
    c.close()
    if frame_bgr is None:
        raise RuntimeError(f"第 {args.frame} 帧不存在")

    # ── 推理 ──
    res = model.predict(frame_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

    annot = frame_bgr.copy()
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        print("!! 该帧未检测到叶片, 换个 --frame 试试")
        return

    # ── 每个实例: 找叶尖, crop, 保存 ──
    n_saved = 0
    for idx, (poly, conf) in enumerate(zip(res.masks.xyn, res.boxes.conf)):
        poly_np = np.asarray(poly, dtype=np.float32)
        tip_x, tip_y = find_tip_point(poly_np, H, W, hub=(args.hub_x, args.hub_y))
        if tip_y < args.min_tip_y:
            print(f"  实例{idx}: conf={float(conf):.3f} 叶尖 y={tip_y} < {args.min_tip_y} → 跳过(塔筒误检/横扫中间态)")
            continue
        crop, (sx0, sy0, sx1, sy1) = crop_square(frame_bgr, tip_x, tip_y, args.crop_size)

        # 保存 crop 图
        crop_path = os.path.join(args.out, f"crop_f{args.frame}_inst{idx}_tip({tip_x},{tip_y}).png")
        cv2.imwrite(crop_path, crop)
        n_saved += 1

        # 原图标注: mask 轮廓(绿) + 叶尖点(红圆) + crop 框(蓝)
        pts = (poly_np * np.array([W, H])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annot, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.circle(annot, (tip_x, tip_y), 10, (0, 0, 255), -1)
        cv2.rectangle(annot, (sx0, sy0), (sx1, sy1), (255, 0, 0), 4)
        cv2.putText(annot, f"conf={float(conf):.2f}", (sx0, max(20, sy0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        print(f"  实例{idx}: conf={float(conf):.3f} 叶尖=({tip_x},{tip_y}) "
              f"crop 源区=[{sx0}:{sx1}, {sy0}:{sy1}] -> {crop_path}")

    # 保存原图标注(压到 1280 宽便于查看)
    h2, w2 = 720, 1280
    annot_small = cv2.resize(annot, (w2, h2))
    annot_path = os.path.join(args.out, f"annot_f{args.frame}.jpg")
    cv2.imwrite(annot_path, annot_small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"标注图(1280x720) -> {annot_path}")

    print(f"\n完成: 保存 {n_saved} 个叶尖 crop。请打开 {args.out}/ 目视检查叶尖细节是否清晰。")


if __name__ == "__main__":
    main()
