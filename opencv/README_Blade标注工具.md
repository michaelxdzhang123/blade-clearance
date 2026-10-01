# Blade 测量边 人工标注工具 — 安装与操作说明

> 路线 A：人工标注 Blade 测量边（暗色叶片边界折线），用于计算真实 Blade→Tower 净空 Cpx
> 生成时间：2026-08-30
> 工具：~/projects/blade-clearance/blade_annotator.py

---

## 1. 工具概述

这是一个人工标注工具，用鼠标在 closest-approach 帧上沿 Blade 暗色叶片边界画折线（polyline），保存为 gt_blade.json。这些人工标注是审核要求的人工 GT（Ground Truth），用于闭合 G2/G3/G4（true static suppression / Blade recall / Blade precision）。

**约束**：纯 OpenCV 实现，无 AI 模型，符合任务书"禁止 AI 自动生成 GT"的要求。

---

## 2. 环境依赖

| 依赖 | 要求 | 当前状态 |
|---|---|---|
| Python | 3.13（.venv） | ✅ ~/projects/blade-clearance/.venv |
| OpenCV | 5.0.0（含 highgui/imshow） | ✅ 已装 |
| PyAV | av（视频解码） | ✅ 已装 |
| GUI 显示 | WSLg（DISPLAY + WAYLAND） | ✅ DISPLAY=:0 |

**检查命令**：
```bash
cd ~/projects/blade-clearance
.venv/bin/python -c "import cv2; print(cv2.__version__, 'imshow' in dir(cv2))"
echo "DISPLAY=$DISPLAY"
```

---

## 3. 安装（无需额外 pip 安装）

工具已就绪，零安装（复用现有 cv2 5.0.0 + PyAV）。如果工具丢失，用以下内容重建（见 blade_annotator.py 源文件）。

**输入底图**（已导出，原始灰度帧，无叠加）：
```
~/projects/blade-clearance/gt_frames/frame_1617.png
~/projects/blade-clearance/gt_frames/frame_1620.png
~/projects/blade-clearance/gt_frames/frame_1625.png
~/projects/blade-clearance/gt_frames/frame_1630.png
~/projects/blade-clearance/gt_frames/frame_1635.png
```

---

## 4. 运行

```bash
cd ~/projects/blade-clearance
.venv/bin/python blade_annotator.py
```

弹出一个 **Blade Annotator** 窗口，默认加载 5 帧：1617、1620、1625、1630、1635。

**自定义帧 / 输出文件**：
```bash
.venv/bin/python blade_annotator.py --frames 1617,1620 --out ~/projects/blade-clearance/gt_blade.json
```

---

## 5. 操作说明

### 鼠标操作

| 操作 | 功能 |
|---|---|
| 左键点击 | 添加折线顶点（沿 Blade 暗色叶片边界点） |
| 右键点击 | 结束当前折线，开始下一条折线 |
| 滚轮 | 放大 / 缩小（看清叶片边缘） |
| 中键拖动 | 平移视图（放大后移动） |

### 键盘操作

| 键 | 功能 |
|---|---|
| `s` | 保存标注到 gt_blade.json |
| `n` | 下一帧 |
| `p` | 上一帧 |
| `u` | 撤销最后一个顶点 |
| `d` | 删除最后一条折线 |
| `c` | 清空当前帧全部标注 |
| `+` / `=` | 放大 |
| `-` | 缩小 |
| `r` | 重置视图（取消缩放/平移） |
| `q` / ESC | 退出（自动保存） |

---

## 6. 标注内容（重要）

标注目标是 **Blade 的暗色叶片边界线**（那条深色叶片轮廓），**不是** Tower（垂直塔筒，x≈437）。

- 用折线**沿着暗色叶片的边缘**点几段，勾勒叶片轮廓
- 每条折线至少 2 个点（右键结束）
- 一条 Blade 边界可用 1~3 段折线勾勒
- **关键帧 1617**（Blade 尖部最接近 Tower）重点标注尖部轮廓，这是 Cpx 最小、最需要精确标注的帧

---

## 7. 标注注意事项

1. 沿**暗色叶片的可见边界**画，不要画进 Tower 塔筒区域
2. 叶片边缘在放大后更清晰，建议先滚轮放大再点
3. 折线顶点密度适中（边缘弯曲处密一点，直线段疏一点）
4. 坐标自动还原为原始 2560×1440 分辨率保存（不受缩放/平移影响）

---

## 8. 输出格式（gt_blade.json）

```json
{
  "video": ".../test01-video-...mp4",
  "frames": {
    "1617": {
      "blade_polylines": [
        [[1234.0, 890.0], [1240.0, 895.0], [1245.0, 902.0]],
        [[1300.0, 870.0], [1305.0, 875.0]]
      ]
    },
    "1620": { "blade_polylines": [...] }
  }
}
```

每条折线是 [x, y] 点列表（原始分辨率坐标，单位 px）。

---

## 9. 标注完成后的流程

标注保存到 gt_blade.json 后，我会：

1. 读取人工标注（Blade 测量边折线）
2. 计算 Blade 折线到 Tower 拟合直线的**真实最短欧氏距离**（不再用 tower band 排除，避开截断 bug）
3. 重新生成准确的 Cpx 对比图
4. 输出 gt_metrics.csv，闭合 G2/G3/G4（true static suppression / Blade recall / precision）

---

## 10. 故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 窗口没弹出 / 黑屏 | WSLg 未启动 | 检查 `echo $DISPLAY`；重启 WSL；或改用备选方案 |
| cv2.imshow 报错 | highgui 缺失 | 重新装 OpenCV（conda cvbuild 环境） |
| 视频帧读不到 | 视频路径变化 | 检查 data/test01-video-...mp4 是否存在 |

**备选方案（WSLg 不可用时）**：直接用 Windows 看图工具打开 gt_frames/frame_*.png，在图上手动标记 Blade 边界折线的像素坐标，把每条折线的 [x,y] 列表发给我即可（我手动写入 gt_blade.json）。
