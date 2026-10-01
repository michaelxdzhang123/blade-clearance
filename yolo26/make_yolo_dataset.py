#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo26/make_yolo_dataset.py — 视频 → 自动标注 → YOLO 训练数据集

把 catch_blade.py 的叶片框(blade_boxes.csv) 转成 YOLO 监督学习数据集。
这是"无监督方法(背景差分)自动生成监督学习(YOLO)训练数据"的桥梁:
  视频 → catch_blade 背景差分检测 → blade_boxes.csv (像素框)
       → 本脚本 → 抽帧图片 + YOLO 归一化标注 → dataset/ + data.yaml

关键设计:
  - 均匀抽样: 10720 帧高度冗余(25fps 横扫周期 ~2.5s), 每 N 行取 1 帧
  - 时间顺序划分 train/val: 前 80% train, 后 20% val, 避免相邻帧泄漏
    (单视频场景用时间顺序; 多视频场景用 train_from_videos.py 的按视频划分)
  - YOLO 标注: <class_id> <x_center> <y_center> <width> <height> (归一化 0~1)

用法:
  python yolo26/make_yolo_dataset.py \
      --video data/training-data/test01-video-2026-08-28_134655_607.mp4 \
      --boxes blade-box/blade_boxes.csv \
      --out dataset_blade \
      --every 4
"""
import argparse
import csv
import os

import av
import cv2

CLASS_ID = 0          # blade 类 ID(单类检测)
CLASS_NAME = "blade"
W, H = 2560, 1440     # 视频分辨率(归一化坐标的分母)


def main():
    parser = argparse.ArgumentParser(description="视频→自动标注→YOLO 数据集")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--boxes", required=True, help="blade_boxes.csv 路径 (frame,x,y,w,h)")
    parser.add_argument("--out", default="dataset_blade", help="输出数据集目录")
    parser.add_argument("--every", type=int, default=4, help="每 N 帧取 1 帧抽样 (默认 4)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="val 比例 (默认 0.2)")
    args = parser.parse_args()

    # ── 1. 读叶片框, 均匀抽样 ──────────────────────────
    # 为什么抽样: 25fps 下叶片横扫周期 ~2.5s(约 62 帧), 相邻帧几乎一样,
    # 10720 帧高度冗余。均匀抽样 [::every] 覆盖整个横扫周期, 且控制数据集大小。
    rows = list(csv.DictReader(open(args.boxes)))
    sample = rows[::args.every]
    print(f"读入 {len(rows)} 帧叶片框, 抽样 1/{args.every} → {len(sample)} 帧")

    # ── 2. 时间顺序划分 train/val ──────────────────────
    # 前 (1-val_ratio) 做 train, 后 val_ratio 做 val。
    # 为什么时间顺序而非随机: 相邻帧几乎一样, 若随机混合, val 会"见过"
    # train 的近邻帧 → mAP 虚高。时间顺序划分 = val 是"没见过的时段",
    # 评估更真实。多视频场景应改用按视频划分(train_from_videos.py)。
    n_val = int(len(sample) * args.val_ratio)
    train = sample[:-n_val]
    val = sample[-n_val:]
    print(f"划分: train={len(train)} 帧, val={len(val)} 帧")

    # ── 3. 建 YOLO 目录结构 ────────────────────────────
    # YOLO 标准布局: images/ 与 labels/ 平级, 各自分 train/val 子目录。
    # 图片与标注靠"同名(不同后缀)"配对: frame_01617.jpg ↔ frame_01617.txt
    dirs = {
        "train_img": os.path.join(args.out, "images", "train"),
        "train_lbl": os.path.join(args.out, "labels", "train"),
        "val_img": os.path.join(args.out, "images", "val"),
        "val_lbl": os.path.join(args.out, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 目标帧 → (split, x, y, w, h) 字典, 供流式解码时 O(1) 查找
    target = {}
    for r in train:
        target[int(r["frame"])] = ("train", int(r["x"]), int(r["y"]),
                                   int(r["w"]), int(r["h"]))
    for r in val:
        target[int(r["frame"])] = ("val", int(r["x"]), int(r["y"]),
                                   int(r["w"]), int(r["h"]))

    # ── 4. 流式解码视频, 命中目标帧就抽图 + 写标注 ──────
    # 为什么流式(O(1) 内存): 全视频近万帧含叶片, 若全缓存进内存会 OOM。
    # 逐帧解码, 只在命中目标帧(idx in target)时抽图+写标注, 其余跳过。
    c = av.open(args.video)
    s = c.streams.video[0]
    n_saved = 0
    for idx, frame in enumerate(c.decode(s)):
        if idx not in target:
            continue
        split, x, y, w, h = target[idx]
        bgr = frame.to_ndarray(format="bgr24")

        name = f"frame_{idx:05d}"
        img_path = os.path.join(dirs[f"{split}_img"], name + ".jpg")
        lbl_path = os.path.join(dirs[f"{split}_lbl"], name + ".txt")
        cv2.imwrite(img_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 像素框(x,y,w,h) → YOLO 归一化中心点+宽高:
        #   (cx,cy) = 框中心 = (x+w/2, y+h/2), 除以 W,H 归一到 [0,1]
        #   (nw,nh) = 框宽高, 同样归一到 [0,1]
        # 为什么归一化: YOLO 训练时图片会被 resize/letterbox 到 imgsz,
        # 归一化坐标不依赖原始分辨率, resize 后仍正确。
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw = w / W
        nh = h / H
        with open(lbl_path, "w") as f:
            f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        n_saved += 1
        if n_saved % 500 == 0:
            print(f"  已处理 {n_saved}/{len(sample)} 帧", flush=True)

    c.close()

    # ── 5. data.yaml ──────────────────────────────────
    # 相对路径(train: images/train), 不写绝对 path, 数据集整体可搬到服务器直接用
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write(f"  {CLASS_ID}: {CLASS_NAME}\n")

    print(f"\n完成: 保存 {n_saved} 帧 (图片 + 标注)")
    print(f"数据集目录 → {args.out}/")
    print(f"  images/train ({len(train)} 帧) + labels/train")
    print(f"  images/val   ({len(val)} 帧)  + labels/val")
    print(f"  data.yaml")


if __name__ == "__main__":
    main()
