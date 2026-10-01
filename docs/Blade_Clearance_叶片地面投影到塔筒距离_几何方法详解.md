# 叶片投影到地面并计算投影点到塔筒距离的方法说明

> 项目：Blade–Tower Clearance 视频测量  
> 依据：用户手绘几何草图  
> 目的：说明如何从相机画面中的叶片点出发，恢复空间几何位置，将叶片点投影到地面，并计算该投影点到塔筒轴线/塔筒表面的距离。

---

# 1. 对手绘草图的理解

从草图中可以明确辨认出的核心元素包括：

- `画面`：摄像机成像平面；
- `Q 光心`：摄像机光心；
- 叶片/叶端附近的特征点；
- 塔筒或塔架位置；
- 若干“已知距离/参数”；
- 由光心、图像点和实际目标点形成的投影关系。

草图的核心思想可以概括为：

```text
图像中识别叶片特征点
        ↓
利用相机光心 Q 与像素点建立空间视线
        ↓
利用已知几何约束恢复叶片点 P 的三维位置
        ↓
将 P 沿重力方向投影到地面
        ↓
得到地面投影点 Pg
        ↓
计算 Pg 到塔筒中心轴/塔筒外表面的距离
```

需要特别指出：

> **“相机光线与地面的交点”不能直接等同于“叶片点的垂直地面投影”。**

这是整个方法中最重要的几何区别。

手绘图中的部分小字较难可靠辨认，因此本文只依据能够确认的几何关系展开，不对模糊字迹做强行解释。

---

# 2. 先定义我们真正想计算什么

设叶片上的测量点为：

$$
P=(X_P,Y_P,Z_P)
$$

塔筒中心轴在地面的投影为：

$$
O_T=(X_T,Y_T,0)
$$

将叶片点沿世界坐标系竖直方向投影到地面：

$$
P_g=(X_P,Y_P,0)
$$

那么叶片点到塔筒中心轴的**水平距离**为：

$$
D_{axis}
=
\sqrt{
(X_P-X_T)^2+
(Y_P-Y_T)^2
}
$$

这也等于地面投影点 $P_g$ 到塔筒中心点 $O_T$ 的平面距离。

因此，只要能恢复叶片点 $P$ 的三维坐标，地面投影距离就非常容易计算。

---

# 3. 为什么要先恢复三维点 P

摄像机在图像中检测到的只是一个二维像素：

$$
p=(u,v)
$$

一个像素并不对应唯一的三维点。

它只定义一条从摄像机光心出发的空间射线：

```text
Camera Center Q
        \
         \
          \
           P? P? P?
```

也就是说：

> **单目相机的一个像素，只能告诉我们目标在哪一条射线上，不能单独告诉我们目标离相机有多远。**

因此必须增加一个额外的物理约束。

对于风机项目，这个约束非常丰富，例如：

1. 叶片点位于已知转子平面；
2. 叶尖到轮毂中心的距离基本等于已知叶片长度；
3. 已知轮毂中心位置；
4. 已知机舱/主轴方向；
5. 已知转子方位角；
6. 已知 Pitch；
7. 可以从 SCADA 获取 RPM / Azimuth / Pitch；
8. 可以使用 HeroWind/OpenFAST 的结构形态作为附加约束。

因此，本项目比一般单目测距问题更有优势。

---

# 4. 推荐的世界坐标系

建议定义风机世界坐标系：

```text
                 Z
                 ↑
                 │
                 │
                 │
                 O────→ X
                /
               /
              Y
```

定义：

- 原点 $O$：塔筒中心轴与地面的交点；
- $Z$：竖直向上；
- $X,Y$：地面水平坐标；
- 地面平面：

$$
Z=0
$$

塔筒中心轴理想情况下：

$$
X=0,\quad Y=0
$$

若塔筒基础测量存在偏置，也可以定义：

$$
(X_T,Y_T)
$$

作为塔筒中心。

---

# 5. 摄像机坐标系

相机内参：

$$
K=
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

还需要：

```text
distortion coefficients
k1, k2, p1, p2, k3 ...
```

以及相机相对于世界坐标系的外参：

$$
R_{wc},\quad t_{wc}
$$

更推荐直接保存：

- 摄像机光心世界坐标：

$$
C=(X_C,Y_C,Z_C)
$$

- Camera → World 旋转矩阵：

$$
R_{wc}
$$

---

# 6. 从像素建立空间射线

首先对检测到的像素：

$$
p=(u,v)
$$

进行去畸变。

得到齐次像素：

$$
\tilde p=
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
$$

在相机坐标系中的射线方向：

$$
d_c
=
K^{-1}\tilde p
$$

归一化：

$$
\hat d_c
=
\frac{d_c}{\|d_c\|}
$$

转换到世界坐标：

$$
\hat d_w
=
R_{wc}\hat d_c
$$

因此该像素对应的三维射线为：

$$
L(\lambda)
=
C+\lambda\hat d_w
$$

其中：

$$
\lambda>0
$$

---

# 7. 最关键的问题：如何确定 λ

只有射线还不够。

必须利用风机几何模型确定：

$$
\lambda
$$

这一步才是真正的“单目三维恢复”。

对于 Blade Clearance 项目，推荐优先使用：

# 方法 A：射线与转子平面求交

---

# 8. 转子平面模型

设轮毂中心：

$$
H=(X_H,Y_H,Z_H)
$$

主轴方向单位向量：

$$
n_r
$$

那么理想转子平面为：

$$
n_r^T(X-H)=0
$$

叶片大部分几何中心线近似位于这个转子平面附近。

摄像机射线：

$$
X=C+\lambda\hat d_w
$$

代入转子平面：

$$
n_r^T(C+\lambda\hat d_w-H)=0
$$

得到：

$$
\lambda
=
\frac{
n_r^T(H-C)
}{
n_r^T\hat d_w
}
$$

然后：

$$
P
=
C+\lambda\hat d_w
$$

这样就从：

```text
2D pixel
```

恢复到了：

```text
3D Blade Point
```

---

# 9. 为什么这种方法特别适合风机

因为本项目不是未知物体。

已知：

```text
Tower
Hub
Rotor Axis
Rotor Plane
Blade Length
Camera Mount
```

这些都是强物理先验。

因此：

```text
Camera Ray
+
Rotor Plane
```

已经可以唯一确定一个三维交点。

再进一步还可以使用叶片长度做 QC。

如果检测的是叶尖：

$$
\|P-H\|
\approx
R_{blade}
$$

其中：

$$
R_{blade}
$$

是已知叶片长度。

如果：

$$
|\|P-H\|-R_{blade}|
$$

非常大，则说明：

```text
检测点错误
相机外参错误
转子平面错误
或者该点并非叶尖
```

---

# 10. 三维点恢复后再投影到地面

得到：

$$
P=(X_P,Y_P,Z_P)
$$

以后，所谓“投影到地面”应该采用**重力方向正交投影**：

$$
P_g=(X_P,Y_P,0)
$$

注意：

```text
P → Pg
```

的方向是世界坐标的：

```text
-Z
```

而不是：

```text
Camera → P
```

的光线方向。

---

# 11. 正确投影与错误投影的区别

## 正确方法

```text
Camera
   \
    \
     P  ← 先恢复真实三维点
     |
     |
     | gravity / vertical
     |
     Pg
──────────── Ground
```

---

## 容易出现的错误方法

```text
Camera
   \
    \
     P
      \
       \
        Pg'
──────────── Ground
```

其中 $P_g'$ 是：

```text
Camera optical ray
```

与地面的交点。

这个点一般：

$$
P_g'\neq P_g
$$

所以：

> **不能直接把像素反投影射线与地面的交点，当成真实叶片点的地面投影。**

只有目标本身位于地面平面时，这种做法才成立。

---

# 12. 地面投影点到塔筒轴线的距离

如果：

$$
O_T=(X_T,Y_T,0)
$$

那么：

$$
D_{axis}
=
\|P_g-O_T\|
$$

即：

$$
D_{axis}
=
\sqrt{
(X_P-X_T)^2+
(Y_P-Y_T)^2
}
$$

如果世界坐标原点已经定义在塔筒中心：

$$
X_T=Y_T=0
$$

则：

$$
D_{axis}
=
\sqrt{
X_P^2+Y_P^2
}
$$

这就是图中“投影点到塔筒”的核心平面距离。

---

# 13. 如果要计算到塔筒表面的距离

塔筒不是一条线，而是一个有半径的锥形/圆柱形结构。

设塔筒在高度 $Z$ 处的半径为：

$$
R_T(Z)
$$

则叶片点在高度 $Z_P$ 附近到塔筒表面的水平间距近似为：

$$
C_h
=
D_{axis}-R_T(Z_P)
$$

即：

$$
C_h
=
\sqrt{
(X_P-X_T)^2+
(Y_P-Y_T)^2
}
-
R_T(Z_P)
$$

这比简单计算：

```text
投影点 → 塔筒中心
```

更接近 Blade–Tower Clearance 的物理定义。

---

# 14. 塔筒是锥形时必须使用局部半径

不能简单使用：

```text
Tower base radius
```

因为风机塔筒通常：

```text
底部粗
顶部细
```

可以建立：

$$
R_T(Z)
$$

例如最简单采用线性模型：

$$
R_T(Z)
=
R_b
+
\frac{R_t-R_b}{H_T}Z
$$

其中：

- $R_b$：塔底半径；
- $R_t$：塔顶半径；
- $H_T$：塔高。

更精确可以使用：

```text
Tower CAD
OpenFAST Tower Geometry
HeroWind Tower Geometry
实测塔筒截面
```

建立分段函数：

$$
R_T(Z)
$$

---

# 15. 需要区分三个不同的“距离”

整个系统必须明确命名，不要都叫 clearance。

## ① 地面投影点到塔筒中心距离

$$
D_{ground-axis}
$$

表示：

```text
Pg → Tower Axis
```

---

## ② 叶片点到塔筒中心轴的水平距离

$$
D_{horizontal-axis}
$$

对于竖直塔筒：

$$
D_{horizontal-axis}
=
D_{ground-axis}
$$

---

## ③ Blade–Tower Surface Clearance

真正需要的量：

$$
C_{tower}
$$

近似：

$$
C_{tower}
=
D_{horizontal-axis}
-
R_T(Z_P)
$$

因此推荐 CSV 里分别保存：

```text
ground_axis_distance_m
tower_radius_at_z_m
horizontal_surface_clearance_m
```

不要只保存一个：

```text
distance
```

---

# 16. 更严格的三维 Clearance 定义

真实 Blade–Tower Clearance 应定义为：

$$
C_{3D}
=
\min_{\mathbf p\in B,\mathbf q\in T}
\|\mathbf p-\mathbf q\|_2
$$

其中：

- $B$：Blade 三维表面；
- $T$：Tower 三维表面。

地面投影法实际上是在利用：

```text
塔筒接近竖直
+
局部截面近似圆形
```

把三维问题简化成水平平面问题。

因此：

```text
Ground Projection Method
```

非常适合：

- 快速测量；
- 在线算法；
- 工程估计；
- 和视频测量融合。

但如果要求：

```text
厘米级绝对 Clearance
```

最后仍应通过完整三维几何验证。

---

# 17. 如果 Blade 已经发生明显挠曲怎么办

理想转子平面模型：

$$
n_r^T(P-H)=0
$$

隐含：

```text
Blade point lies in rotor plane
```

但真实风机存在：

```text
flap deformation
edge deformation
precone
shaft tilt
yaw
pitch
tower motion
nacelle motion
```

因此实际 Blade 点可能：

$$
P\notin \Pi_{rotor}
$$

这时有三个升级级别。

---

# 18. Level 1：刚性转子平面

适合第一阶段：

```text
Camera Ray
+
Rotor Plane
```

恢复 $P$。

优点：

```text
简单
稳定
容易标定
```

---

# 19. Level 2：转子平面 + Blade Length / Azimuth

如果知道：

```text
Blade length
Rotor azimuth
Pitch
```

可以对三维点增加：

$$
\|P-H\|\approx R
$$

以及叶片方向约束。

这会明显增强鲁棒性。

---

# 20. Level 3：HeroWind/OpenFAST 结构模型融合

最终高级方案可以：

```text
Video
+
SCADA
+
HeroWind structural state
```

得到动态 Blade surface：

$$
B(t)
$$

此时摄像机不再和理想转子平面求交，而是与预测 Blade surface / centerline 求交。

这可以考虑：

```text
Blade flap deformation
Pitch
Precone
Shaft tilt
Tower deflection
Platform motion
```

对于漂浮式风机尤其重要。

---

# 21. Pitch 对这个方法的影响

Pitch 改变会造成：

```text
Blade apparent width
Blade visible edge
Leading/trailing edge visibility
```

发生变化。

因此 YOLO / OpenCV 不能假设：

```text
Blade width = constant
```

检测应采用：

```text
YOLO26n
→ Safe ROI

OpenCV
→ dynamic contour

PCA
→ dynamic blade orientation

Tower-facing branch
→ measurement point
```

然后选定真正需要三维反投影的：

```text
Blade measurement point
```

---

# 22. 不建议直接使用 YOLO BBox Center 做投影

例如：

```text
YOLO BBox
      ↓
BBox Center
      ↓
3D projection
```

这个方法误差会很大。

因为 BBox Center：

- 不是叶尖；
- 不是塔侧最近点；
- 随 Pitch 变化；
- 随 Blade 入画比例变化；
- 随遮挡变化。

正确方法应该是：

```text
YOLO26n
      ↓
Safe ROI
      ↓
Original Resolution
      ↓
OpenCV Local Edge
      ↓
EdgesSubPix
      ↓
Tower-facing Blade Point
      ↓
3D Ray
```

---

# 23. 推荐的完整算法链

```text
RTSP / Video
      ↓
Hardware Decode
      ↓
2560×1440 Original Frame
      ↓
Classic CV Motion / Blade Pass
      ↓
YOLO26n
      ↓
Safe ROI
      ↓
Original-resolution ROI
      ↓
OpenCV Contour
      ↓
PCA / Dynamic Width
      ↓
Tower-facing Boundary
      ↓
EdgesSubPix
      ↓
Blade Pixel Point p=(u,v)
      ↓
Camera Calibration K,D
      ↓
Undistort
      ↓
Camera Ray
      ↓
Camera Extrinsic R,T
      ↓
World Ray
      ↓
Intersect Rotor Plane / Blade Model
      ↓
3D Blade Point P
      ↓
Vertical Projection
      ↓
Ground Point Pg
      ↓
Distance to Tower Axis
      ↓
Tower Radius R_T(Zp)
      ↓
Blade–Tower Clearance
```

---

# 24. 数学流程汇总

## Step 1：像素

$$
p=(u,v)
$$

## Step 2：去畸变

$$
p\rightarrow p_u
$$

## Step 3：生成 Camera Ray

$$
d_c=K^{-1}
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
$$

## Step 4：转换到世界坐标

$$
d_w=R_{wc}d_c
$$

## Step 5：和转子平面求交

$$
\lambda
=
\frac{
n_r^T(H-C)
}{
n_r^Td_w
}
$$

## Step 6：三维叶片点

$$
P=C+\lambda d_w
$$

## Step 7：地面垂直投影

$$
P_g=(X_P,Y_P,0)
$$

## Step 8：到塔轴距离

$$
D_{axis}
=
\sqrt{
(X_P-X_T)^2+
(Y_P-Y_T)^2
}
$$

## Step 9：塔筒表面 Clearance

$$
C_h
=
D_{axis}-R_T(Z_P)
$$

---

# 25. 一个简单的数值例子

假设恢复出的叶片点：

$$
P=(4.20,\;1.30,\;85.0)\;m
$$

塔筒中心：

$$
O_T=(0,0,0)
$$

那么地面投影：

$$
P_g=(4.20,\;1.30,\;0)
$$

投影点到塔筒轴线：

$$
D_{axis}
=
\sqrt{4.20^2+1.30^2}
$$

$$
D_{axis}
\approx4.40m
$$

如果塔筒在：

$$
Z=85m
$$

高度处半径为：

$$
R_T(85)=2.10m
$$

那么水平塔筒表面间距：

$$
C_h
=
4.40-2.10
=
2.30m
$$

这才是比单纯：

```text
4.40 m
```

更接近真实净空的值。

---

# 26. Camera Calibration 是整个方法的基础

要实现上述方法，至少需要：

## Camera Intrinsic

```text
fx
fy
cx
cy
distortion
```

## Camera Extrinsic

```text
Camera position C
Camera orientation R_wc
```

## Turbine Geometry

```text
Tower center
Hub center
Rotor axis
Rotor plane
Tower radius profile
Blade length
```

这些建议全部写入：

```yaml
camera:
  fx:
  fy:
  cx:
  cy:
  distortion:

extrinsic:
  camera_position:
  rotation:

turbine:
  tower_center:
  hub_center:
  rotor_axis:
  blade_radius:
  tower_radius_profile:
```

---

# 27. 摄像机安装后不能随意变化

以下变化都会使外参失效：

```text
Camera mount movement
Zoom change
Focus-related optical change
EIS
Digital crop
Camera rotation
Nacelle camera remount
```

因此推荐：

```text
EIS = OFF
Zoom = LOCK
Focus = LOCK
Camera mount = rigid
```

如果 Camera 相对机舱本身是刚性的，但机舱会 Yaw：

```text
Camera → Nacelle
```

外参固定；

然后利用：

```text
Nacelle yaw angle
```

把坐标进一步转换到：

```text
Tower / World Coordinate
```

---

# 28. 对机舱相机尤其要考虑 Yaw

如果 Camera 安装在 nacelle：

```text
Camera
↓
Nacelle Coordinate
```

是固定的。

但：

```text
Nacelle
```

会绕塔顶 Yaw。

所以：

$$
R_{world-camera}
=
R_{world-nacelle}
R_{nacelle-camera}
$$

其中：

$$
R_{world-nacelle}
$$

由当前：

```text
Yaw angle
```

决定。

因此在线系统最好同步：

```text
timestamp
yaw
pitch
rpm
azimuth
```

---

# 29. 误差主要来自哪里

建议建立误差预算：

| 误差源 | 对结果的影响 |
|---|---|
| Pixel edge error | 改变相机射线方向 |
| Camera intrinsic error | 尺度与方向误差 |
| Lens distortion | 图像边缘区域尤其明显 |
| Camera extrinsic error | 直接造成空间位置系统误差 |
| Camera vibration | 动态外参变化 |
| Rotor-plane assumption | Blade flap 时产生偏差 |
| Hub coordinate error | 三维定位偏差 |
| Tower radius model | Clearance 偏差 |
| Tower deflection | 实际塔筒位置变化 |
| Pitch / azimuth error | Blade 模型误差 |
| Timestamp mismatch | 快速运动时非常严重 |

---

# 30. 时间同步非常重要

叶尖速度如果达到：

$$
V_{tip}=83m/s
$$

25 fps：

$$
\Delta t=0.04s
$$

一帧时间内运动尺度可达：

$$
83\times0.04
=
3.32m
$$

因此：

```text
Video timestamp
SCADA timestamp
Pitch
Yaw
Azimuth
```

必须尽量同步。

否则几帧的错位就可能造成非常大的空间几何误差。

---

# 31. 推荐增加几何有效性 Gate

只有满足以下条件时才输出三维距离：

```text
Camera calibration valid
Camera extrinsic valid
Blade pixel valid
Rotor-plane intersection valid
λ > 0
Blade radius check valid
Tower geometry valid
Timestamp valid
```

否则：

```text
valid = false
```

推荐 failure code：

```text
CAMERA_CALIB_INVALID
CAMERA_EXTRINSIC_INVALID
RAY_PARALLEL_TO_ROTOR_PLANE
RAY_INTERSECTION_BEHIND_CAMERA
BLADE_RADIUS_QC_FAIL
ROTOR_GEOMETRY_INVALID
TOWER_GEOMETRY_INVALID
TIMESTAMP_MISMATCH
```

---

# 32. 推荐输出 CSV

建议保存：

```text
frame_id
timestamp_s
pass_id

blade_u_px
blade_v_px

ray_dx
ray_dy
ray_dz

blade_x_m
blade_y_m
blade_z_m

ground_x_m
ground_y_m

tower_center_x_m
tower_center_y_m

ground_axis_distance_m
tower_radius_at_blade_z_m
horizontal_surface_clearance_m

pitch_deg
yaw_deg
azimuth_deg

geometry_quality
valid
failure_code
```

这样以后每一个 Clearance 数值都能追溯到：

```text
pixel
→ ray
→ 3D point
→ ground projection
→ tower distance
```

形成完整证据链。

---

# 33. 与当前 Blade Clearance 视频方案的关系

这个“地面投影法”并不取代：

```text
YOLO26n
OpenCV
EdgesSubPix
```

而是接在它们后面。

当前第一阶段：

```text
Image
↓
Blade/Tower Boundary
↓
Cpx
```

解决：

```text
像素空间净空
```

新增投影几何后：

```text
Image Pixel
↓
Camera Model
↓
3D Geometry
↓
Ground Projection
↓
Physical Distance
```

解决：

```text
米制空间距离
```

所以完整项目可以分成两级：

```text
Stage 1
Cpx / pixel-space validation

Stage 2
Camera calibration + turbine geometry

Stage 3
C_m / physical clearance
```

---

# 34. 推荐的最终方法

对于你的草图，我建议最终把方法正式定义成：

# **Camera-Ray + Rotor-Geometry + Ground Orthogonal Projection**

而不是简单称为：

```text
像素投影到地面
```

因为真正的数学过程是：

```text
Image Pixel
      ↓
Camera Ray
      ↓
Rotor / Blade Geometry Constraint
      ↓
3D Blade Point
      ↓
Vertical Projection to Ground
      ↓
Horizontal Distance to Tower
```

这个名称能避免后续工程人员误把：

```text
Camera ray ∩ Ground
```

当成：

```text
Blade vertical projection
```

---

# 35. 最终结论

手绘图所表达的“**通过叶片投影到地面，计算投影点到塔筒距离**”在工程上是可行的，而且非常适合风机这种**强几何先验系统**。

但是实现时必须遵守三个关键原则。

## 原则 1：一个像素不能直接得到地面投影

必须先通过：

```text
Camera Ray
+
Rotor / Blade Geometry
```

恢复三维叶片点。

---

## 原则 2：地面投影应是重力方向投影

正确：

$$
P=(X,Y,Z)
\rightarrow
P_g=(X,Y,0)
$$

而不是直接取相机光线和地面的交点。

---

## 原则 3：到塔筒中心的距离还不是最终 Clearance

必须进一步考虑：

$$
R_T(Z)
$$

即塔筒在叶片当前高度处的实际半径：

$$
C_h
=
D_{axis}-R_T(Z_P)
$$

最终推荐完整链路：

```text
YOLO26n / Classic CV
        ↓
Original-resolution Blade Point
        ↓
EdgesSubPix
        ↓
Camera Calibration
        ↓
3D Ray
        ↓
Rotor / Blade Geometry
        ↓
3D Blade Point
        ↓
Vertical Ground Projection
        ↓
Distance to Tower Axis
        ↓
Tower Radius at Height
        ↓
Physical Blade–Tower Clearance
```

这条路线可以作为当前 Blade Clearance 项目从：

```text
Cpx
```

升级到：

```text
C_meter
```

的重要几何方案。
