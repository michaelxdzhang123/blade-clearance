---
title: "机舱下方单摄像机 Blade Tip 到 Tower 净空距离计算公式"
date: 2026-09-15
tags: [blade-clearance, nacelle-camera, monocular-vision, tower-clearance, geometry, formula]
type: theory
language: zh
---

# 机舱下方单摄像机 Blade Tip 到 Tower 净空距离计算公式

本文基于：

- `docs/2026-0915-净空专利调研与机舱相机纯视觉新方案.md`
- NB-MGC（Nacelle-Bottom Monocular Geometric Clearance）方案

核心计算分为两级：

1. 图像平面中的像素净空 `C_px`；
2. 满足单目几何可观测条件后的物理净空 `C_m`。

不能直接用 YOLO bbox 距离代替 Blade 到 Tower 的真实边界距离。

---

## 1. 变量定义

### 1.1 图像与边界变量

| 符号 | 含义 | 单位 |
|---|---|---|
| `I_t(u,v)` | 第 `t` 帧图像 | gray level / RGB |
| `(u,v)` | 图像像素坐标 | px |
| `Γ_B(t)` | 第 `t` 帧 Blade 测量边界 | px |
| `Γ_T(t)` | 第 `t` 帧 Tower 测量边界 | px |
| `p=(u_B,v_B)^T` | Blade 边界候选点 | px |
| `q=(u_T,v_T)^T` | Tower 边界候选点 | px |
| `S_B` | YOLO26-seg 输出的 Blade 候选 mask | binary mask |
| `T` | 帧间差分阈值 | gray level |
| `A_t` | 第 `t` 帧运动面积比例 | dimensionless |

Blade 测量边界 `Γ_B(t)` 不是整个 segmentation mask 的边界，而是同时满足以下条件的边界：

- 位于 Tower 一侧；
- 属于当前 Blade-pass；
- 经过局部 EdgesSubPix refinement；
- 没有发生边界截断；
- 通过时序一致性检查。

### 1.2 Camera model 变量

| 符号 | 含义 |
|---|---|
| `K` | camera intrinsic matrix |
| `R` | camera/world rotation matrix |
| `t` | translation vector |
| `C` | camera center in world coordinates |
| `d(p)` | 图像点对应的空间射线方向 |
| `Π_B` | Blade measurement plane 或等效空间测量平面 |

Camera intrinsic matrix 为：

$$
\mathbf K=
\begin{bmatrix}
 f_x & 0 & c_x\\
 0 & f_y & c_y\\
 0 & 0 & 1
\end{bmatrix}
$$

其中：

- `f_x, f_y`：水平和垂直焦距，单位为 px；
- `c_x, c_y`：主点坐标，单位为 px。

### 1.3 Tower 几何变量

| 符号 | 含义 |
|---|---|
| `O_T` | Tower axis 上的参考点 |
| `a_T` | Tower axis unit vector |
| `R_T(z)` | Tower 在高度 `z` 处的局部半径 |
| `e_1,e_2` | 垂直于 Tower axis 的正交基 |
| `Σ_T` | Tower 外表面 |
| `z` | 沿 Tower axis 的坐标 |

局部 Tower 可近似表示为圆柱或锥台：

$$
\mathbf X_T(z,\theta)
=
\mathbf O_T
+z\mathbf a_T
+R_T(z)
\left(
\cos\theta\,\mathbf e_1
+\sin\theta\,\mathbf e_2
\right)
$$

其中：

$$
0\leq\theta<2\pi
$$

Tower surface 为：

$$
\Sigma_T
=
\left\{
\mathbf X_T(z,\theta)
\right\}
$$

---

## 2. Blade-pass 触发

首先用帧间差分判断 Blade 是否进入测量区域：

$$
D_t(u,v)
=
\left|
I_t(u,v)-I_{t-1}(u,v)
\right|
$$

运动面积比例定义为：

$$
A_t
=
\frac{1}{HW}
\sum_{u=1}^{W}
\sum_{v=1}^{H}
\mathbf 1
\left[
D_t(u,v)>T
\right]
$$

其中：

- `H,W`：图像高度和宽度；
- `T`：帧间灰度差阈值；
- `1[·]`：indicator function。

当：

$$
A_t>T_{\mathrm{trigger}}
$$

并且运动区域存在有效大连通域时，系统进入 `BLADE_PASS` 状态。

该公式只用于 Blade-pass 触发和 ROI 定位，不直接给出 Blade 边界，也不直接计算净空。

---

## 3. 像素净空公式

### 3.1 一般形式

Blade 与 Tower 的图像边界分别为：

$$
\Gamma_B(t)
$$

和：

$$
\Gamma_T(t)
$$

则第 `t` 帧的像素净空为：

$$
\boxed{
C_{px}(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\min_{\mathbf q\in\Gamma_T(t)}
\left\|
\mathbf p-\mathbf q
\right\|_2
}
$$

展开为：

$$
\boxed{
C_{px}(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\min_{\mathbf q\in\Gamma_T(t)}
\sqrt{
(u_B-u_T)^2+(v_B-v_T)^2
}
}
$$

这是真正的 Blade measurement boundary 到 Tower measurement boundary 的最近欧氏距离。

### 3.2 Tower 近似竖直时

如果 Tower 在当前局部 ROI 内近似竖直：

$$
\Gamma_T:
 u=u_T(v)
$$

并且 Tower 倾角满足小角度条件，则可以近似为横向净空：

$$
\boxed{
C_{px}(t)
\approx
\min_{(u_B,v_B)\in\Gamma_B(t)}
\left|
 u_B-u_T(v_B)
\right|
}
$$

如果 Tower measurement edge 在 ROI 内可拟合为直线：

$$
 u_T(v)=a_Tv+b_T
$$

则点到 Tower 直线的欧氏距离为：

$$
\boxed{
C_{px}(t)
=
\min_{(u_B,v_B)\in\Gamma_B(t)}
\frac{
|u_B-a_Tv_B-b_T|
}{
\sqrt{1+a_T^2}
}
}
$$

其中：

- `a_T,b_T`：Tower measurement edge 拟合参数；
- `sqrt(1+a_T^2)`：将代数距离转换为欧氏距离。

当 Tower 几乎垂直时：

$$
 a_T\approx 0
$$

因此：

$$
C_{px}(t)
\approx
\min_{(u_B,v_B)\in\Gamma_B(t)}
|u_B-b_T|
$$

---

## 4. 局部尺度近似下的物理距离

当 Blade 和 Tower 在局部 ROI 内的深度差很小，并且投影尺度近似不变时，可以定义局部尺度：

$$
 s_x
=
\frac{R_{T,\mathrm{real}}}
{R_{T,\mathrm{pixel}}}
$$

其中：

- `R_T,real`：Tower 局部真实半径，单位 m；
- `R_T,pixel`：Tower 局部图像半径，单位 px；
- `s_x`：局部横向比例尺，单位 m/px。

于是：

$$
\boxed{
C_m(t)
\approx
s_xC_{px}(t)
}
$$

即：

$$
\boxed{
C_m(t)
\approx
\frac{R_{T,\mathrm{real}}}
{R_{T,\mathrm{pixel}}}
C_{px}(t)
}
$$

如果使用 Tower 直径，则：

$$
\boxed{
C_m(t)
\approx
\frac{D_{T,\mathrm{real}}}
{D_{T,\mathrm{pixel}}}
C_{px}(t)
}
$$

该公式只有在以下条件同时成立时才可靠：

1. Blade 与 Tower 位于相近深度；
2. ROI 足够小，使局部投影尺度近似恒定；
3. 镜头畸变已经校正；
4. Tower 半径对应正确高度 `z`；
5. Camera pose 没有明显变化；
6. Blade 与 Tower 边界处于相同或近似相同的测量平面。

因此，该公式是局部投影近似，不应无条件作为全局单目测距公式。

---

## 5. 一般单目三维几何模型

### 5.1 Camera projection model

采用带畸变修正的 pinhole model：

$$
\lambda
\begin{bmatrix}
 u\\v\\1
\end{bmatrix}
=
\mathbf K[\mathbf R|\mathbf t]
\begin{bmatrix}
 X\\Y\\Z\\1
\end{bmatrix}
$$

其中：

- `K`：通过离线 camera calibration 得到；
- `R,t`：由 nacelle 安装几何和 Tower reference 约束确定；
- `lambda`：投影尺度因子。

### 5.2 图像点去畸变

原始 Blade 像素点为：

$$
\mathbf p=
\begin{bmatrix}
 u_B\\v_B\\1
\end{bmatrix}
$$

经过 lens distortion correction 后得到：

$$
\mathbf p_u=
\begin{bmatrix}
 u_B^u\\v_B^u\\1
\end{bmatrix}
$$

上标 `u` 表示 undistorted。

### 5.3 由图像点构造空间射线

在 camera coordinate system 中，射线方向为：

$$
\mathbf r_C(\mathbf p)
=
\frac{
\mathbf K^{-1}\mathbf p_u
}{
\left\|
\mathbf K^{-1}\mathbf p_u
\right\|_2
}
$$

Camera center 在 world coordinate system 中为：

$$
\mathbf C=-\mathbf R^T\mathbf t
$$

对应的世界坐标射线为：

$$
\boxed{
\mathbf X(\lambda;\mathbf p)
=
\mathbf C
+
\lambda\mathbf R^T\mathbf r_C(\mathbf p)
}
$$

其中：

$$
\lambda>0
$$

为射线深度参数。

### 5.4 Blade measurement plane 求交

如果 Blade measurement boundary 可近似位于已知平面：

$$
\Pi_B:
\mathbf n_B^T\mathbf X+d_B=0
$$

将相机射线代入平面方程：

$$
\mathbf n_B^T
\left[
\mathbf C+
\lambda\mathbf R^T\mathbf r_C
\right]
+d_B=0
$$

得到：

$$
\boxed{
\lambda_B
=
-
\frac{
\mathbf n_B^T\mathbf C+d_B
}{
\mathbf n_B^T\mathbf R^T\mathbf r_C
}
}
$$

Blade 边界点的三维位置为：

$$
\boxed{
\mathbf X_B(\mathbf p)
=
\mathbf C
+
\lambda_B
\mathbf R^T\mathbf r_C(\mathbf p)
}
$$

`Π_B` 可以是：

- Blade tip clearance plane；
- 由 rotor geometry 和 nacelle pose 定义的局部平面；
- 小范围测量中使用的等效平面。

如果 `Π_B` 未知，则仅凭单目图像无法唯一确定 `lambda_B`，也就不能唯一恢复真实三维距离。

---

## 6. Blade tip 到 Tower 表面的物理距离

对于 Blade measurement boundary 上的每个三维点：

$$
\mathbf X_B(\mathbf p)
$$

计算其到 Tower surface `Σ_T` 的最短距离：

$$
 d_T\left(\mathbf X_B(\mathbf p)\right)
=
\min_{\mathbf X_T\in\Sigma_T}
\left\|
\mathbf X_B(\mathbf p)-\mathbf X_T
\right\|_2
$$

因此，第 `t` 帧 Blade-to-Tower clearance 为：

$$
\boxed{
C_m(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\min_{\mathbf X_T\in\Sigma_T}
\left\|
\mathbf X_B(\mathbf p,t)-\mathbf X_T
\right\|_2
}
$$

将 Tower surface 参数化为：

$$
\mathbf X_T(z,\theta)
=
\mathbf O_T
+z\mathbf a_T
+R_T(z)
\left(
\cos\theta\,\mathbf e_1+
\sin\theta\,\mathbf e_2
\right)
$$

则：

$$
\boxed{
C_m(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\min_{z,\theta}
\left\|
\mathbf X_B(\mathbf p,t)
-
\mathbf X_T(z,\theta)
\right\|_2
}
$$

这就是 NB-MGC 方案的严格物理净空公式。

---

## 7. Tower 近似为无限圆柱时

若局部 Tower 可看作轴线为：

$$
\mathbf X_{\mathrm{axis}}(z)
=
\mathbf O_T+z\mathbf a_T
$$

半径为 `R_T` 的圆柱，则 Blade 点到 Tower 外表面的距离为：

$$
 d_T(\mathbf X_B)
=
\left\|
\left(
\mathbf I-\mathbf a_T\mathbf a_T^T
\right)
\left(
\mathbf X_B-\mathbf O_T
\right)
\right\|_2
-R_T
$$

其中：

$$
\mathbf I-\mathbf a_T\mathbf a_T^T
$$

是垂直于 Tower axis 的投影矩阵。

因此：

$$
\boxed{
C_m(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\left[
\left\|
\left(
\mathbf I-\mathbf a_T\mathbf a_T^T
\right)
\left(
\mathbf X_B(\mathbf p,t)-\mathbf O_T
\right)
\right\|_2
-R_T
\right]
}
$$

解释：

- 第一项是 Blade point 到 Tower axis 的径向距离；
- 减去 Tower 半径 `R_T` 后，得到 Blade boundary 到 Tower outer surface 的 clearance；
- 若 `C_m(t)<0`，表示几何上发生穿透或扫塔；
- 若 `C_m(t)=0`，表示 Blade boundary 与 Tower surface 接触。

---

## 8. 最小净空 `C_min`

只允许使用通过所有质量门禁的 measured frames：

$$
\mathcal T_{\mathrm{valid}}
=
\left\{
 t:
\begin{array}{l}
\mathrm{TOWER\_VISIBLE}\\
\land\mathrm{BLADE\_VISIBLE}\\
\land\mathrm{NO\_BOUNDARY\_CLIP}\\
\land\mathrm{EDGE\_STABILITY}\\
\land\mathrm{POSE\_STABLE}\\
\land\mathrm{METRIC\_OBSERVABLE}
\end{array}
\right\}
$$

则整个 Blade-pass 窗口的最小净空为：

$$
\boxed{
C_{\min}
=
\min_{t\in\mathcal T_{\mathrm{valid}}}
C_m(t)
}
$$

不能把预测帧加入 `T_valid`：

$$
\hat{\mathbf p}_{t+1}
=
\mathbf p_t+\mathbf v_t\Delta t
$$

因此：

$$
 t\notin\mathcal T_{\mathrm{valid}}
\quad\Rightarrow\quad
 C_m(t)\text{ 不得进入 }C_{\min}
$$

预测只能用于下一帧 ROI，不能作为实际净空测量。

---

## 9. 三个计算层级

### Level A：像素净空

$$
\boxed{
C_{px}(t)
=
\min_{\mathbf p\in\Gamma_B(t),\mathbf q\in\Gamma_T(t)}
\|\mathbf p-\mathbf q\|_2
}
$$

适用范围：

- 视频趋势分析；
- Blade-pass 检测；
- 算法调试；
- 视觉质量监测；
- 三维可观测性不满足时的降级输出。

### Level B：局部尺度近似

$$
\boxed{
C_m(t)
\approx
\frac{D_{T,\mathrm{real}}}
{D_{T,\mathrm{pixel}}}
C_{px}(t)
}
$$

适用范围：

- Blade/Tower 深度差较小；
- 局部 ROI 足够小；
- Tower diameter 或局部半径已知；
- camera calibration 和畸变校正已经完成。

### Level C：几何约束单目测距

$$
\boxed{
C_m(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\min_{z,\theta}
\left\|
\mathbf X_B(\mathbf p,t)
-
\mathbf X_T(z,\theta)
\right\|_2
}
$$

适用条件：

- 已知 `K`；
- 已知 camera pose `(R,t)`；
- 已知 Tower geometry `R_T(z)`；
- Blade measurement plane 或深度约束可确定；
- 当前帧通过 `METRIC_OBSERVABLE` gate。

---

## 10. 推荐正式定义

当前项目建议把以下公式作为正式的像素级测量定义：

$$
\boxed{
C_{px}(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\min_{\mathbf q\in\Gamma_T(t)}
\|\mathbf p-\mathbf q\|_2
}
$$

当单目几何可观测时，使用：

$$
\boxed{
C_m(t)
=
\min_{\mathbf p\in\Gamma_B(t)}
\min_{z,\theta}
\left\|
\mathbf X_B(\mathbf p,t)
-
\left[
\mathbf O_T+z\mathbf a_T+
R_T(z)
\left(
\cos\theta\,\mathbf e_1+
\sin\theta\,\mathbf e_2
\right)
\right]
\right\|_2
}
$$

整个 Blade-pass 窗口的最小净空为：

$$
\boxed{
C_{\min}
=
\min_{t\in\mathcal T_{\mathrm{valid}}}
C_m(t)
}
$$

最终原则：

$$
\boxed{
\text{AI 用于定位，EdgesSubPix 用于边界，camera geometry 用于尺度，Tower model 用于物理距离。}
}
$$

不能由：

$$
\text{YOLO bbox distance}
$$

直接替代：

$$
\text{Blade boundary-to-Tower surface clearance}
$$
