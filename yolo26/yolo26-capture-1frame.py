import argparse

import cv2
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description="YOLO26 single-frame inference")
parser.add_argument("model", help="Path to the YOLO26 model file (.pt)")
parser.add_argument("--drive", required=True, help="Path to the input video file")
parser.add_argument("--frame", type=int, default=0, help="Frame index to process (default: 0)")
parser.add_argument("--out", default=None, help="Output image path (optional, default: not saved)")
args = parser.parse_args()

# Load the YOLO26 model
model = YOLO(args.model)

# Open the video file using OpenCV
cap = cv2.VideoCapture(args.drive)
if not cap.isOpened():
    raise SystemExit(f"无法打开视频: {args.drive}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Seek to the requested frame
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
success, frame = cap.read()
if not success:
    raise SystemExit(f"读取帧 {args.frame} 失败 (视频共 {total} 帧)")

# Run YOLO26 inference on the single frame
results = model(frame)
boxes = results[0].boxes

# Draw green bounding boxes manually (BGR green, consistent with blade detection)
GREEN = (0, 255, 0)
annotated_frame = frame.copy()
if boxes is not None:
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        name = model.names[cls]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), GREEN, 4)
        label = f"{name} {conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

# Print detection results
n = len(boxes) if boxes is not None else 0
print(f"帧 {args.frame}/{total-1}  检出 {n} 个目标")
if boxes is not None and n > 0:
    for b in boxes:
        xyxy = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        name = model.names[cls]
        print(f"  {name:<12} conf={conf:.4f}  "
              f"xyxy=({xyxy[0]:.1f},{xyxy[1]:.1f},{xyxy[2]:.1f},{xyxy[3]:.1f})")

# Save annotated frame if requested
if args.out:
    cv2.imwrite(args.out, annotated_frame)
    print(f"标注图已保存 → {args.out}")

cap.release()
