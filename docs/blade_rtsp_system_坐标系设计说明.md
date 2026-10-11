# blade_rtsp_system 坐标系设计说明

> 文档对象:`blade_rtsp_system/`(main.py + src/coordinate_mapper.py + src/blade_locator.py)
> 生成日期:2026-10-01
> 配套:`config/config.yaml`(processing.resize / processing.roi)

---

## 1. 为什么需要这份说明(WHY)

整个 pipeline 里坐标会经过多次变换(原始帧 → 缩放 → ROI → YOLO → 输出),而**输出的坐标到底在哪个坐标系**,直接决定下游(可视化、净空距离计算)能否正确还原到真实像素位置。最容易踩的坑是:

- YOLO 的 `masks.xy` 返回的是**输入图坐标**,不是 640 内部坐标;
- 加了 `resize` 之后,输出坐标从「原始帧系」悄悄变成了「缩放帧系」;
- `coordinate_mapper` 只做**平移**(ROI 偏移),不做**缩放**(resize)。

这份说明把四个坐标系、变换链、输出坐标系和缩放因子一次讲清。

---

## 2. 四个坐标系

图像坐标约定(OpenCV):原点在**左上角 (0,0)**,x 向右、**y 向下**。因此「y 最小 = 顶部」「y 最大 = 底部」——这直接决定了叶尖(在顶部)是取 `min-y` 而非 `max-y`。

| 符号 | 名称 | 分辨率 | 说明 |
|------|------|--------|------|
| **A** | 原始帧坐标系 | 2560×1440 | RTSP/视频源的原生分辨率(相机物理像素) |
| **A′** | 缩放帧坐标系 | resize 目标(如 640×360) | resize 后、推理前的帧 |
| **B** | ROI 局部坐标系 | 可变 | 仅 `use_roi=true` 时,A′ 里裁出的子区域 |
| **C** | YOLO 内部坐标系 | 640×640 | 模型 letterbox 后推理尺寸,**不对外输出** |

---

## 3. 坐标变换链(数据流)

```
A 原始帧 2560×1440
   │  cv2.resize(→ resize_target)           [main.py reader_loop]
   ▼
A′ 缩放帧 640×360 (或 640×640)
   │  可选 ROI 裁剪 frame[y1:y2, x1:x2]     [main.py 主循环, use_roi]
   ▼
B  ROI 局部 (可选)
   │  detector.predict() → YOLO letterbox 到 C(内部 640)
   │  retina_masks → masks.xy 返回「输入图」坐标(即 A′ 或 B, 不是 C)
   ▼
extract_blades → blade 实例 (坐标在 A′ 或 B 系)
   │  locate_blades: coordinate_mapper 平移 (B → A′)
   ▼
输出 (bbox/center/polygon/blade_tip_candidate)  —— 坐标系 = A′(缩放帧), 不是 A(原始帧)
```

关键:YOLO 的 `C(640 内部)` 只在模型内部存在。`result.masks.xy` 给出的 polygon 坐标**已经是输入给 `model.predict()` 的那张图的坐标**(即 A′ 或 B 系),不是 C 系。见 `coordinate_mapper.py:11-15`。

---

## 4. 输出坐标系(最容易错的一点)

**最终输出(CSV/JSONL 里的 bbox / center / polygon / blade_tip_candidate)在 A′(缩放帧)坐标系,不在 A(原始帧)。**

证据:
- `main.py` 里 `frame_w, frame_h` 被覆盖成 `resize_target`(缩放后尺寸),不是 cap 的原生 2560×1440(`main.py:277-278`);
- `locate_blades(..., frame_width=frame_w, frame_height=frame_h)` 用缩放尺寸做坐标基准;
- `coordinate_mapper` 只做 ROI 平移,从不做 resize 缩放。

因此 JSONL 里 `frame_width=640, frame_height=360`、`bbox y2=360` 之类,都是 640×360 缩放帧系的值。

> 注意:`coordinate_mapper.py:7` 的 docstring 写的「A. RTSP Original Frame —— 最终标准输出坐标系」是**加 resize 之前**的旧描述。加了 resize 后,实际输出在 A′ 缩放帧系。这是需要对齐的一处文档不一致。

---

## 5. 缩放因子(A′ → A 还原)

要从缩放帧坐标还原到原始帧坐标:

```
x_A = x_A′ × (W_original / W_resize)
y_A = y_A′ × (H_original / H_resize)
```

| resize 目标 | sx = 2560/W | sy = 1440/H | 是否等比 |
|-------------|-------------|-------------|----------|
| [640, 360] (16:9, 推荐) | 4.0 | 4.0 | ✅ 等比 4× |
| [640, 640] (正方形) | 4.0 | 2.25 | ❌ 非等比, 画面被压扁 |

- `[640, 360]` 保持 16:9,缩放是均匀 4×,坐标还原简单(x、y 都 ×4)。
- `[640, 640]` 是正方形,会把 2560×1440 压扁,`sy=2.25 ≠ sx=4.0`,坐标还原需要非等比缩放,且**叶片形状被压扁**——之前实测会把检出率从 80% 拉到 74%。所以推荐用 `[640, 360]`。

---

## 6. coordinate_mapper 的职责边界

`src/coordinate_mapper.py` 只做一件事:**把 ROI 局部坐标(B)平移回缩放帧坐标(A′)**。函数:

| 函数 | 作用 |
|------|------|
| `roi_to_global(points, ox, oy)` | 点集平移 `+ (ox, oy)` |
| `map_bbox(bbox, ox, oy)` | bbox `[x1,y1,x2,y2]` 整体平移 |
| `map_center(center, ox, oy)` | 中心点平移 |
| `map_polygon(polygon, ox, oy)` | 多边形逐点平移 |
| `map_mask(mask, ox, oy, full_w, full_h)` | 把 ROI mask 放到 full_w×full_h 画布的正确偏移处 |

**它只做平移,不做缩放。** resize(A → A′)的缩放没有对应的 mapper——这是当前设计的一个缺口:如果下游要 A(原始帧)坐标,需要在 `coordinate_mapper` 之外额外乘缩放因子(见 §5)。可视化脚本 `visualize_blade_tips_direct.py` 就是这么做的(乘 4×)。

---

## 7. ROI 与 resize 的交互(一处一致性问题)

`main.py` 里的实际顺序是:**先 resize(A→A′),再在 A′ 上做 ROI 裁剪**:

```python
# reader_loop: 先 resize
frame = cv2.resize(frame, resize_target)          # A → A′
# 主循环: 再 ROI 裁剪 (在 A′ 上!)
roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]   # A′ → B
```

因此 ROI 的 `x1/y1/x2/y2` 是**缩放帧(A′)坐标系下的值**,不是原始帧坐标。

但 `config.yaml:21` 的注释写的是「仅当 use_roi=true 时生效(**原始帧坐标**)」——这与实际代码(ROI 在 resize 之后、A′ 系)不一致。两种修法:
1. 改注释为「缩放帧坐标」;或
2. 把 ROI 裁剪挪到 resize 之前(先裁原始帧再 resize),这样 ROI 才是原始帧坐标,同时还能省掉 ROI 外区域的 resize 计算量。

---

## 8. 叶尖坐标

`blade_tip_candidate`(在 `src/blade_locator.py`)返回的 `[x, y]` 也是 **A′(缩放帧)坐标**。

本视频叶片斜向横扫,**叶尖在顶部(y 最小)**,根部在底部(宽)。故:

```python
ymin = pts[:, 1].min()                    # 顶部
top = pts[pts[:, 1] <= ymin + 1.0]        # 顶部点集
return [top[:, 0].mean(), top[:, 1].mean()]  # 顶部尖端中心
```

> 之前用「y 最大顶点」落在叶片底边(侧边),是错的;详见 `tower-blade-clearance` skill 的「叶尖判据」节。换视频时叶尖方向可能不同,须重新用标注图确认。

---

## 9. 下游如何正确还原

| 目标 | 做法 |
|------|------|
| 画到原始 2560×1440 帧上 | 坐标 × `(W_orig/W_resize, H_orig/H_resize)`(640×360 时即 ×4) |
| 读 JSONL 里的 bbox/叶尖 | 先看 `frame_width`/`frame_height` 字段确认是哪个 resize 尺寸,再乘对应缩放因子 |
| 净空距离计算(像素级) | 在同一坐标系下算;若要物理米,需额外相机标定(GSD),不属于本模块 |

---

## 10. 一句话总结

- 输入链:A 原始帧(2560×1440)→ **resize** → A′ 缩放帧 → (可选 ROI) B → YOLO C(内部,不输出)。
- **输出在 A′ 缩放帧系**,不在 A 原始帧系。
- `coordinate_mapper` 只做 ROI 平移(B→A′),不做 resize 缩放(A→A′)。
- 还原到原始帧:坐标乘 `(W_orig/W_resize, H_orig/H_resize)`;用 `[640, 360]` 保持等比(×4)。
