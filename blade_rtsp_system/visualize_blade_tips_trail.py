#!/usr/bin/env python3
"""逐帧累积绿点，叶片消失时清空；复用原始候选点，不使用固定/保持坐标。

离线模式：连续至少三帧无检测时，回溯到首个空帧清除；短漏检不清除。
已审核的经过分段也作为清空边界，防止新叶片继承上一片的轨迹。
"""
import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from visualize_blade_tips_closest import write_ass


def build_trails(rows, passes, missing_frames=3):
    if missing_frames < 1:
        raise ValueError('missing_frames must be positive')
    events, segments = [], []
    def finish(start, end):
        if start is None or end <= start:
            return
        seen = set()
        for frame in range(start, end):
            for point in rows[frame]:
                key = tuple(point)
                # Repeated coordinates cover the same pixels; preserve first arrival.
                if key not in seen:
                    seen.add(key)
                    events.append(dict(start_frame=frame, end_frame_exclusive=end,
                                       fixed_tip=point))
        segments.append(dict(start_frame=start, end_frame_exclusive=end,
                             distinct_points=len(seen)))
    for passage in passes:
        start, gap_start = None, None
        for frame in range(passage['start_frame'], passage['end_frame_exclusive']):
            if rows[frame]:
                if start is None:
                    start = frame
                gap_start = None
            elif start is not None:
                if gap_start is None:
                    gap_start = frame
                if frame-gap_start+1 >= missing_frames:
                    finish(start, gap_start)
                    start, gap_start = None, None
        end = passage['end_frame_exclusive']
        # At the end of a known pass even a short final gap is disappearance.
        finish(start, gap_start if gap_start is not None else end)
    return events, segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--markers', required=True)
    parser.add_argument('--passes', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.markers).read_text())
    passes = json.loads(Path(args.passes).read_text())
    meta = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',args.video]))['streams'][0]
    assert int(meta['nb_frames']) == len(rows)
    assert passes[0]['start_frame'] >= 0 and passes[-1]['end_frame_exclusive'] == len(rows)
    assert all(a['end_frame_exclusive'] == b['start_frame'] for a,b in zip(passes,passes[1:]))
    fps = float(Fraction(meta['avg_frame_rate']))
    events, segments = build_trails(rows, passes)
    (work/'trail_events.json').write_text(json.dumps(events))
    (work/'segments.json').write_text(json.dumps(segments, indent=2))
    ass = work/'trail.ass'
    write_ass(ass, events, fps, meta['width'], meta['height'])
    (work/'provenance.json').write_text(json.dumps(dict(video=str(Path(args.video).resolve()), markers=str(Path(args.markers).resolve()), passes=str(Path(args.passes).resolve()), frames=len(rows), fps=fps, segments=len(segments), distinct_point_events=len(events), policy='Each observed tip point persists at its own coordinate until blade disappearance. No lines or predicted points. Clear at the first frame of a >=3-frame detection gap, or at the next pass boundary.', limitation='Uses existing candidate detections and right-to-left pass segmentation; short gaps under 3 frames are treated as detector dropout.'), indent=2))
    print(f'Prepared {len(segments)} trails, {len(events)} point events, {len(rows)} frames', flush=True)
    if args.prepare_only:
        return
    if any(c in str(ass) for c in "':,[];\\"):
        raise ValueError('Use a simple work path without filter-special characters')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-threads','2','-i',args.video,
        '-vf',f'setpts=N/({fps:.12f}*TB),ass={ass}','-filter_threads','2','-an','-c:v','libx264',
        '-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-r',f'{fps:.12f}',
        '-frames:v',str(len(rows)),'-movflags','+faststart','-progress',str(work/'encode.progress'),args.output],check=True)


if __name__ == '__main__':
    main()
