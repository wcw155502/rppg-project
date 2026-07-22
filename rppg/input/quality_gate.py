import cv2
import numpy as np

from rppg.measurement.fusion import extract_skin_rgb_mean
from .types import QualityReport


class InputQualityGate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.previous_gray = None

    def evaluate(self, aligned_face):
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        overexposed = float((gray >= 245).mean())
        underexposed = float((gray <= 20).mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        motion = 0.0
        if self.previous_gray is not None:
            motion = float(cv2.absdiff(gray, self.previous_gray).mean())
        self.previous_gray = gray
        _, skin = extract_skin_rgb_mean(aligned_face, min_skin_ratio=0.0)
        skin_ratio = float(skin["skin_ratio"])

        reasons = []
        if brightness < self.cfg["min_brightness"]:
            reasons.append("too_dark")
        if brightness > self.cfg["max_brightness"]:
            reasons.append("too_bright")
        if overexposed > self.cfg["max_overexposed_ratio"]:
            reasons.append("overexposed")
        if underexposed > self.cfg["max_underexposed_ratio"]:
            reasons.append("underexposed")
        if sharpness < self.cfg["min_sharpness"]:
            reasons.append("blurred")
        if motion > self.cfg["max_motion"]:
            reasons.append("large_motion")
        if skin_ratio < self.cfg["min_skin_ratio"]:
            reasons.append("skin_occluded")
        return QualityReport(
            valid=not reasons,
            reasons=tuple(reasons),
            brightness=brightness,
            overexposed_ratio=overexposed,
            underexposed_ratio=underexposed,
            sharpness=sharpness,
            motion=motion,
            skin_ratio=skin_ratio,
        )

    def reset(self):
        self.previous_gray = None
