"""Classic CV candidate detector. No torch, weights, or metrology claims.

Frame difference triggers; current-frame/background difference localizes.
Background updates only while idle, and freezes during a detected pass.
All masks are coarse localization evidence, NOT measurement boundaries.
"""
import time
import cv2
import numpy as np


class ClassicDetector:
    def __init__(self, config=None):
        c = config or {}
        self.warmup = int(c.get('warmup_frames', 30))
        self.trigger = float(c.get('trigger_threshold', 15))
        self.localize = float(c.get('localize_threshold', 30))
        self.min_area = float(c.get('min_component_area', 200))
        self.top_k = int(c.get('top_k', 3))
        self.alpha = float(c.get('background_alpha', .01))
        self.max_fraction = float(c.get('max_foreground_fraction', .65))
        if not (self.warmup >= 1 and 0 < self.trigger < self.localize <= 255
                and self.min_area > 0 and self.top_k >= 1 and 0 < self.alpha <= 1
                and 0 < self.max_fraction <= 1):
            raise ValueError('Invalid classic detector configuration')
        self.reset()

    def reset(self):
        self.background = self.previous = self.center = None
        self.count = 0
        self.active = False
        self.pass_id = 0
        self.warmup_samples = []

    def predict(self, frame):
        start = time.perf_counter()
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.previous is not None and gray.shape != self.previous.shape:
            self.reset()
        self.count += 1
        instances = []
        status = 'NO_MOTION'
        triggered = False
        if self.background is None:
            self.background = gray.astype(np.float32)
        if self.count <= self.warmup:
            self.warmup_samples.append(gray.copy())
            if self.count == self.warmup:
                self.background = np.median(np.stack(self.warmup_samples), axis=0).astype(np.float32)
                self.warmup_samples.clear()
            status = 'BACKGROUND_WARMUP'
        else:
            motion = cv2.absdiff(gray, self.previous) > self.trigger
            triggered = bool(np.count_nonzero(motion) >= self.min_area * .1)
            fg = (cv2.absdiff(gray, cv2.convertScaleAbs(self.background)) > self.localize).astype(np.uint8) * 255
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            if np.count_nonzero(fg) / fg.size > self.max_fraction:
                status = 'FRAME_QUALITY_LOW'  # global lighting change or camera motion
            elif triggered or self.active:
                contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                scored = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    center = np.array([x+w/2, y+h/2])
                    aspect = max(w, h) / max(1, min(w, h))
                    temporal = 0. if self.center is None else 1. / (1.+np.linalg.norm(center-self.center)/max(w,h))
                    score = .4*min(area/(self.min_area*5),1)+.3*min(aspect/3,1)+.3*temporal
                    scored.append((score, area, contour, (x,y,w,h), center))
                for idx, (score, area, contour, (x,y,w,h), center) in enumerate(sorted(scored,key=lambda c:c[0],reverse=True)[:self.top_k]):
                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(mask,[contour],-1,255,-1)
                    instances.append(dict(instance_id=idx,class_id=0,class_name='blade',confidence=float(score),
                        bbox=[x,y,x+w,y+h],mask=mask,mask_area=float(cv2.countNonZero(mask))))
                if scored:
                    self.center = max(scored,key=lambda c:c[0])[4]
                status = 'CANDIDATE_FOUND' if instances else 'BLADE_CANDIDATE_NOT_FOUND'
            # Never absorb foreground or global scene changes into the idle background.
            if not instances and not triggered and np.count_nonzero(fg) < self.min_area:
                cv2.accumulateWeighted(gray,self.background,self.alpha)
        if instances and not self.active:
            self.pass_id += 1
        self.active = bool(instances)
        self.previous = gray
        detection = dict(blade_detected=bool(instances),blade_count=len(instances),blade_instances=instances,
                         primary_instance_id=0 if instances else None,
                         diagnostics=dict(status=status,motion_triggered=triggered,pass_id=self.pass_id,
                         pass_state='TRACKING' if instances else 'IDLE',confidence_kind='heuristic_candidate_score',
                         measurement_valid=False,cpx_euclidean_px=None,failure_code='MEASUREMENT_NOT_IMPLEMENTED'))
        return detection, (time.perf_counter()-start)*1000
