#!/usr/bin/env python3
"""离线固定绿点：每次叶片经过取距塔筒轮廓最近的已检测叶尖。

输入为逐帧原始候选点 JSON（不能用已保持的坐标），以及原始无绿点视频。
针对当前视频右向左扫过的运动分段；不是通用叶片身份跟踪器。
先看完整段，再从该段首次检测时显示最近点，保持到下段首次检测。
距离是画面像素距离，不能作为真实物理净空。第一/最后段可能不完整。
"""
import argparse
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path


def distance_to_outline(point, outline):
    best = float('inf')
    for a, b in zip(outline, outline[1:] + outline[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        denom = dx * dx + dy * dy
        u = max(0., min(1., ((point[0]-a[0])*dx + (point[1]-a[1])*dy) / denom)) if denom else 0.
        best = min(best, math.hypot(point[0]-a[0]-u*dx, point[1]-a[1]-u*dy))
    return best


def make_passes(rows, outline):
    # Incoming blade enters to the right of the previous outgoing blade.
    rightmost = [max(row, key=lambda p: p[0]) if row else None for row in rows]
    first = next((i for i, p in enumerate(rightmost) if p is not None), None)
    if first is None:
        raise ValueError('No valid blade tip detections')
    starts = [first]
    last, last_i = None, first
    for i, point in enumerate(rightmost):
        if point is None:
            continue
        gap = i - last_i - 1
        restart = last is not None and (
            (point[0]-last[0] > 450 and point[0] > 600)
            or (gap >= 2 and point[0]-last[0] > 250))
        if restart:
            future = [p[0] for p in rightmost[i+1:i+9] if p is not None]
            if (future and max(future) > point[0]-850
                    and (i-starts[-1] > 25 or len(starts) == 1)):
                starts.append(i)
        last, last_i = point, i
    result = []
    for start, end in zip(starts, starts[1:] + [len(rows)]):
        candidates = [(distance_to_outline(p, outline), i, p)
                      for i in range(start, end) for p in rows[i]]
        distance, frame, tip = min(candidates, key=lambda item: item[0])
        result.append(dict(start_frame=start, end_frame_exclusive=end,
                           closest_frame=frame, fixed_tip=tip, distance_px=distance))
    return result


def ass_time(frame, fps):
    # Floor to centiseconds: event activates at the intended frame, not one late.
    ticks = math.floor(frame / fps * 100 + 1e-6)
    return f'{ticks//360000}:{ticks//6000%60:02}:{ticks//100%60:02}.{ticks%100:02}'


def write_ass(path, passes, fps, width, height):
    header = f'''[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dot,Arial,10,&H0000FF00,&H0000FF00,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    # Radius 4 green disk, thin white outline, like the original marker.
    circle = 'm 4 0 b 6.21 0 8 1.79 8 4 b 8 6.21 6.21 8 4 8 b 1.79 8 0 6.21 0 4 b 0 1.79 1.79 0 4 0'
    lines = [header]
    for event in passes:
        x, y = event['fixed_tip']
        tag = '{' + f'\\an7\\pos({x-4},{y-4})\\p1\\bord1\\shad0' + '}'
        lines.append(f"Dialogue: 0,{ass_time(event['start_frame'],fps)},{ass_time(event['end_frame_exclusive'],fps)},Dot,,0,0,0,,{tag}{circle}\n")
    path.write_text(''.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--markers', required=True)
    parser.add_argument('--tower', required=True, help='JSON polygon in full-resolution pixels')
    parser.add_argument('--work', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.markers).read_text())
    outline = json.loads(Path(args.tower).read_text())
    meta = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',args.video]))['streams'][0]
    if int(meta['nb_frames']) != len(rows):
        raise ValueError('Marker frame count does not match video')
    fps = float(Fraction(meta['avg_frame_rate']))
    passes = make_passes(rows, outline)
    (work/'passes.json').write_text(json.dumps(passes, indent=2))
    ass = work/'closest.ass'
    write_ass(ass, passes, fps, meta['width'], meta['height'])
    (work/'provenance.json').write_text(json.dumps(dict(video=str(Path(args.video).resolve()), markers=str(Path(args.markers).resolve()), tower_outline=outline,
        frames=len(rows), fps=fps, detected_passes=len(passes), policy='Offline look-ahead: one closest detected point per pass; fixed from first detection until next pass starts.',
        limitation='Image-space distance, right-to-left segmentation heuristic, approximately one-pixel uncertainty in recovered marker centers.'), indent=2))
    print(f'Prepared {len(passes)} passes, {len(rows)} frames', flush=True)
    if args.prepare_only:
        return
    # Normalize time exactly as the previous OpenCV renderer, retaining every frame.
    # Keep work path shell/filter-safe; subprocess uses argument lists, never a shell.
    if any(c in str(ass) for c in "':,[];\\"):
        raise ValueError('Use a simple work path without filter-special characters')
    cmd = ['ffmpeg','-hide_banner','-loglevel','warning','-y','-threads','2','-i',args.video,
           '-vf',f'setpts=N/({fps:.12f}*TB),ass={ass}', '-filter_threads','2',
           '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','20',
           '-pix_fmt','yuv420p','-r',f'{fps:.12f}','-frames:v',str(len(rows)),
           '-movflags','+faststart','-progress',str(work/'encode.progress'),args.output]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
