#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo26/add_fog_augmentation.py — 雾天数据增强：把雾视频的叶片框加入训练集

预处理路线的"合成雾"思想用于训练增强（非推理去雾）：
  清晰叶片视频 → 大气散射模型加雾 → 合成雾视频
  → catch_blade 检测雾视频叶片框 → 本脚本加入训练集
  → 模型在清晰+雾天混合数据上训练 → 学到雾天鲁棒

用法:
  python yolo26/add_fog_augmentation.py \
      --video data/test01-video-2026-09-09-heavy-foggy.mp4 \
      --boxes blade-box-heavy-foggy-cv/blade_boxes.csv \
      --dataset dataset_blade --every 4 --prefix fog_
"""
import argparse
import csv
import os

import av
import cv2

W, H = 2560, 1440
CLASS_ID = 0


def main():
    parser = argparse.ArgumentParser(description="雾天数据增强(雾视频叶片框加入训练集)")
    parser.add_argument("--video", required=True, help="雾视频路径")
    parser.add_argument("--boxes", required=True, help="雾视频叶片框 csv (frame,x,y,w,h)")
    parser.add_argument("--dataset", default="dataset_blade", help="现有数据集目录")
    parser.add_argument("--every", type=int, default=4, help="抽样间隔")
    parser.add_argument("--prefix", default="fog_", help="文件名前缀(避免与清晰数据冲突)")
    args = parser.parse_args()

    # 读叶片框 + 抽样
    rows = list(csv.DictReader(open(args.boxes)))
    sample = rows[::args.every]
    print(f"读入 {len(rows)} 帧雾天叶片框, 抽样 1/{args.every} → {len(sample)} 帧")

    target = {int(r["frame"]): (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
              for r in sample}

    img_dir = os.path.join(args.dataset, "images", "train")
    lbl_dir = os.path.join(args.dataset, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 从雾视频抽帧 + 写标注 (fog_ 前缀)
    c = av.open(args.video)
    s = c.streams.video[0]
    n = 0
    for idx, frame in enumerate(c.decode(s)):
        if idx not in target:
            continue
        x, y, w, h = target[idx]
        bgr = frame.to_ndarray(format="bgr24")

        name = f"{args.prefix}{idx:05d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"),
                    bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw = w / W
        nh = h / H
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        n += 1
        if n % 500 == 0:
            print(f"  已写入 {n}/{len(sample)} 帧", flush=True)
    c.close()

    print(f"\n完成: 写入 {n} 帧雾天数据 → {args.dataset}/images/train + labels/train")
    # 统计当前 train 集
    n_img = len(os.listdir(img_dir))
    n_lbl = len(os.listdir(lbl_dir))
    print(f"当前 train 集: {n_img} 图, {n_lbl} 标注 (清晰 + 雾天混合)")


if __name__ == "__main__":
    main()
