#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_blade.py — 训练 YOLO26 叶片(blade)检测模型

在 GPU 服务器(RTX4090)上运行。输入是 dataset_blade/ 数据集
(清晰 + 雾天混合自动标注的叶片框)。

流程:
  加载预训练 yolo26n.pt → 迁移训练 blade 类 → 验证 → 输出 best.pt

用法(在 GPU 服务器, 数据集已 scp 到位):
  python yolo26/train_blade.py                          # 全部用默认值
  python yolo26/train_blade.py --epochs 200 --batch 32  # 调超参数
  python yolo26/train_blade.py --resume                 # 中断后续训

关键超参数说明:
  - nms=False   YOLO26 NMS-free one-to-one head(端到端, 部署简单)
                True=one-to-many+NMS(精度略高)
  - imgsz=640   YOLO26n 标准输入, 640 是精度/速度平衡点
  - batch=16    RTX4090 24GB 可到 32~64, 显存不足再降
  - patience=30 30 epoch 不提升 mAP 就早停(默认 100 epoch 通常 50~80 就停)

YOLO26 新知识(2026-09):
  - 官方论文 arXiv:2606.03748 (非 2509.25164, 那是第三方分析)
  - MuSGD 优化器(默认): Muon(2502.16982) + SGD 混合, 收敛更快
  - 雾天数据增强: dataset_blade 已含 2596 帧 heavy-foggy 雾天数据
    (合成雾用于"训练增强"有效, 用于"推理去雾"被证伪)
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练 YOLO26 blade 检测模型")
    parser.add_argument("--data", default="dataset_blade/data.yaml",
                        help="data.yaml 路径 (默认 dataset_blade/data.yaml)")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="预训练权重路径 (默认 yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size (RTX4090 可 32~64, 默认 16)")
    parser.add_argument("--device", default="0", help="GPU 设备号 (默认 0)")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程 (默认 8)")
    parser.add_argument("--patience", type=int, default=30,
                        help="早停耐心 (默认 30 epoch)")
    parser.add_argument("--nms", action="store_true",
                        help="用 one-to-many head + NMS (默认 NMS-free one-to-one head)")
    parser.add_argument("--name", default="blade_train", help="训练运行名称")
    parser.add_argument("--resume", action="store_true", help="从上次中断续训")
    args = parser.parse_args()

    # ── 迁移学习: 加载 COCO 预训练权重 ─────────────────
    # YOLO("yolo26n.pt") 是 COCO 80 类预训练。train 时 ultralytics 自动:
    #   - 把检测头改成我们的单类(blade), 输出维度 80 → 1
    #   - 主干/颈层迁移 COCO 参数(已学会通用视觉特征), 只微调检测头+精调主干
    # 这是"迁移学习"(非从零): 几百帧叶片数据就能收敛, 无需百万级数据。
    model = YOLO(args.model)

    if args.resume:
        # 续训: 从 last.pt 恢复(权重+优化器状态+lr调度+epoch), 至少跑完 1 epoch 才能 resume
        print("续训模式: 从上次 runs/detect/blade_train 恢复")
        results = model.train(resume=True)
    else:
        results = model.train(
            data=args.data,          # 数据集 yaml(images/labels + names)
            epochs=args.epochs,      # 训练轮数(早停通常 50~80 就停)
            imgsz=args.imgsz,        # 输入分辨率(640 是 YOLO26n 标准)
            batch=args.batch,        # batch size(受显存限制, RTX4090 24GB 可 32~64)
            device=args.device,      # GPU 设备(0=第一块)
            workers=args.workers,    # 数据加载并行线程(CPU 侧)
            patience=args.patience,  # 早停: N epoch 不提升 mAP 就提前结束
            name=args.name,          # 运行名 → 输出 runs/detect/<name>/
            # YOLO26 双头: 默认 nms=False 用 one-to-one head(NMS-free 端到端),
            # --nms 切 one-to-many head(需 NMS, 精度略高)。训练时 nms 参数选择
            # "验证/早停/checkpoint 用的推理 head", 两个头都保留训练损失。
            nms=args.nms,
            # 数据增强: 默认开启 mosaic/mixup 等。叶片横扫带已含运动模糊,
            # 且已混入雾天数据, 这里不过度增强(避免模糊+增强叠加失真)。
            # 需要可显式开: mosaic=1.0, mixup=0.5, copy_paste=0.1
        )

    # ── 训练完成后, 显式验证 best.pt 并打印 mAP ────────
    print("\n" + "=" * 60)
    print("训练完成, 用 best.pt 在 val 集上验证...")
    print("=" * 60)
    # ultralytics 训练输出路径确定: runs/detect/<name>/weights/best.pt
    # (best.pt = 验证 mAP 最高的 checkpoint, last.pt = 最后一轮)
    best_path = f"runs/detect/{args.name}/weights/best.pt"

    val_model = YOLO(best_path)
    # 在 val 集上验证: mAP50(IoU=0.5) + mAP50-95(严格, IoU 0.5~0.95 平均)
    metrics = val_model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    print("\n" + "=" * 60)
    print(f"最佳权重: {best_path}")
    print(f"mAP50:    {metrics.box.map50:.4f}")   # 宽松(框大致对准就算中)
    print(f"mAP50-95: {metrics.box.map:.4f}")     # 严格(框精确对准才中)
    print("=" * 60)
    print(f"\n用 best.pt 推理视频:")
    print(f"  python yolo26/yolo26-capture.py {best_path} "
          f"--drive data/training-data/test01-video-2026-08-28_134655_607.mp4 --out blade-box-yolo")


if __name__ == "__main__":
    main()
