#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blade 测量边 人工标注工具 (路线 A)

用法:
    .venv/bin/python blade_annotator.py --frames 1617,1620,1625,1630,1635 --out gt_blade.json

操作:
    鼠标左键点击  = 添加折线顶点 (标注 Blade 暗色叶片边界)
    鼠标右键点击  = 结束当前折线, 开始下一条折线
    键盘 s        = 保存标注到 JSON
    键盘 u        = 撤销最后一个顶点
    键盘 d        = 删除最后一条折线
    键盘 n        = 下一帧
    键盘 p        = 上一帧
    键盘 c        = 清空当前帧全部标注
    键盘 r        = 重置视图 (取消缩放/平移)
    键盘 +/-      = 放大/缩小
    键盘 q / ESC  = 退出 (退出前自动保存)
    鼠标滚轮       = 缩放
    鼠标中键拖动   = 平移 (缩放后)

显示: 原始 2560x1440 缩放到屏幕, 坐标自动还原为原始分辨率保存
"""
import av, numpy as np, cv2, os, json, argparse, sys

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")

# ---------- 全局状态 ----------
class Anno:
    def __init__(self):
        self.polylines = []      # 已完成折线: list[list[[x,y],...]]
        self.current = []        # 当前正在画的折线
        self.scale = 0.45        # 显示缩放
        self.pan = [0, 0]        # 平移偏移
        self.frame_idx = 0
        self.frames = []         # 帧号列表
        self.imgs = {}           # 帧号 -> 原始灰度图
        self.dragging = False
        self.drag_start = None

A = Anno()

def load_frames(frames):
    container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"
    want = set(frames); A.frames = sorted(frames)
    for idx, frame in enumerate(container.decode(stream)):
        if idx in want:
            A.imgs[idx] = frame.to_ndarray(format="gray")
            want.discard(idx)
        if not want: break
    container.close()

def screen_to_img(sx, sy):
    x = (sx - A.pan[0]) / A.scale
    y = (sy - A.pan[1]) / A.scale
    return x, y

def img_to_screen(ix, iy):
    return int(ix * A.scale + A.pan[0]), int(iy * A.scale + A.pan[1])

def render():
    g = A.imgs[A.frames[A.frame_idx]]
    disp = cv2.resize(g, None, fx=A.scale, fy=A.scale, interpolation=cv2.INTER_AREA)
    h, w = disp.shape
    canvas = np.full((max(h, 100), max(w, 100), 3), 40, np.uint8)
    # 应用平移
    M = np.float32([[1, 0, A.pan[0]], [0, 1, A.pan[1]]])
    disp_shifted = cv2.warpAffine(disp, M, (canvas.shape[1], canvas.shape[0]))
    canvas[:disp_shifted.shape[0], :disp_shifted.shape[1]] = cv2.cvtColor(disp_shifted, cv2.COLOR_GRAY2BGR)
    # 画已完成折线 (红色)
    for pl in A.polylines:
        pts = np.array([img_to_screen(p[0], p[1]) for p in pl], np.int32).reshape(-1,1,2)
        cv2.polylines(canvas, [pts], False, (0,0,255), 2)
        for p in pts:
            cv2.circle(canvas, tuple(p[0]), 3, (0,0,255), -1)
    # 画当前折线 (绿色)
    if len(A.current) >= 1:
        pts = np.array([img_to_screen(p[0], p[1]) for p in A.current], np.int32).reshape(-1,1,2)
        cv2.polylines(canvas, [pts], False, (0,255,0), 2)
        for p in pts:
            cv2.circle(canvas, tuple(p[0]), 3, (0,255,0), -1)
    # 信息栏
    fn = A.frames[A.frame_idx]
    info = f"frame {fn}  ({A.frame_idx+1}/{len(A.frames)})  scale={A.scale:.2f}  "
    info += f"折线={len(A.polylines)}  当前顶点={len(A.current)}"
    cv2.putText(canvas, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(canvas, "左键加点 | 右键结束折线 | s保存 | n/p切帧 | q退出", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    return canvas

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = screen_to_img(x, y)
        if 0 <= ix < 2560 and 0 <= iy < 1440:
            A.current.append([round(ix, 1), round(iy, 1)])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(A.current) >= 2:
            A.polylines.append(list(A.current))
            A.current = []
        elif len(A.current) == 1:
            A.current = []
    elif event == cv2.EVENT_MBUTTONDOWN:
        A.dragging = True
        A.drag_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and A.dragging:
        if A.drag_start is not None:
            A.pan[0] += x - A.drag_start[0]
            A.pan[1] += y - A.drag_start[1]
            A.drag_start = (x, y)
    elif event == cv2.EVENT_MBUTTONUP:
        A.dragging = False
        A.drag_start = None
    elif event == cv2.EVENT_MOUSEWHEEL:
        factor = 1.1 if flags > 0 else 0.9
        A.scale = min(max(A.scale * factor, 0.1), 3.0)

def save(out_path):
    data = {"video": VIDEO, "frames": {}}
    for fn in A.frames:
        if fn in A.imgs:
            data["frames"][str(fn)] = {"blade_polylines": []}
    # 保存当前帧的标注
    all_pls = A.polylines + ([A.current] if len(A.current) >= 1 else [])
    data["frames"][str(A.frames[A.frame_idx])]["blade_polylines"] = all_pls
    # 读取已有文件合并
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            for k, v in old.get("frames", {}).items():
                if k not in data["frames"]:
                    data["frames"][k] = v
        except Exception:
            pass
    json.dump(data, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[已保存] {out_path}  帧 {A.frames[A.frame_idx]} 折线 {len(all_pls)} 条")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="1617,1620,1625,1630,1635")
    ap.add_argument("--out", default=os.path.expanduser("~/projects/blade-clearance/gt_blade.json"))
    args = ap.parse_args()
    frames = [int(f) for f in args.frames.split(",")]
    load_frames(frames)
    print(f"已加载 {len(A.imgs)} 帧: {A.frames}")
    cv2.namedWindow("Blade Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Blade Annotator", on_mouse)
    while True:
        cv2.imshow("Blade Annotator", render())
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q') or k == 27:
            save(args.out); break
        elif k == ord('s'):
            save(args.out)
        elif k == ord('u') and A.current:
            A.current.pop()
        elif k == ord('d') and A.polylines:
            A.polylines.pop()
        elif k == ord('c'):
            A.polylines = []; A.current = []
        elif k == ord('n') and A.frame_idx < len(A.frames)-1:
            A.polylines = []; A.current = []; A.frame_idx += 1
        elif k == ord('p') and A.frame_idx > 0:
            A.polylines = []; A.current = []; A.frame_idx -= 1
        elif k in (ord('+'), ord('=')):
            A.scale = min(A.scale * 1.15, 3.0)
        elif k == ord('-'):
            A.scale = max(A.scale * 0.87, 0.1)
        elif k == ord('r'):
            A.scale = 0.45; A.pan = [0, 0]
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
