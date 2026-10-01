#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 7/9 号视频完整性(完整解码, 报告帧数或损坏位置)"""
import av

vids = [
    'data/training-data/7-192.168.45.122_01_20250823--北京阴天小雨.mp4',
    'data/training-data/9-192.168.45.122_01_20250821--北京大雨.mp4',
]
for p in vids:
    n = 0
    try:
        c = av.open(p)
        s = c.streams.video[0]
        for f in c.decode(s):
            n += 1
        c.close()
        print(f'{p.split("/")[-1][:30]:32s} OK  {n} 帧', flush=True)
    except Exception as e:
        print(f'{p.split("/")[-1][:30]:32s} 损坏 @约{n} 帧  ({type(e).__name__})', flush=True)
print('DONE', flush=True)
