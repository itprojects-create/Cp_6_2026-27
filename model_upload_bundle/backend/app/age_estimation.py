"""Chest X-ray age estimation (TorchXRayVision Riken age model).

The dataset has no age metadata. Ages are estimated from the image so that results
can be stratified by age band; the estimates are reviewed visually (see
configs/age_review.yaml) before they are used for evaluation.

The Riken model was trained on chest-framed adult radiographs (NIH ChestX-ray14),
so it is applied to the thorax region found by the lung segmenter rather than to a
whole-body pediatric film.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from tqdm import tqdm

from app.config import get_config
from app.preprocessing import load_image, maybe_invert, pad_to_square, to_xrv_range
from app.segmentation import LungSegmenter, expand_box


class AgeEstimator:
    def __init__(self, device: str | None = None, segmenter: LungSegmenter | None = None):
        import torchxrayvision as xrv

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = xrv.baseline_models.riken.AgeModel().to(self.device).eval()
        self.segmenter = segmenter or LungSegmenter(device=self.device)
        self.segmenter._ensure_loaded()
        self.cfg = get_config()

    @torch.no_grad()
    def estimate_gray(self, gray: np.ndarray, roi_box: list[int] | None = None) -> float:
        if roi_box is None:
            roi_box = self.segmenter.segment_gray(gray)["roi_box"]
        x0, y0, x1, y1 = expand_box(tuple(roi_box), gray.shape, 0.1)
        square, _ = pad_to_square(gray[y0:y1, x0:x1])
        img = cv2.resize(square, (320, 320), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(to_xrv_range(img))[None, None].float().to(self.device)
        return float(self.model(tensor).reshape(-1)[0].cpu())

    def estimate(self, source) -> float:
        gray = maybe_invert(load_image(source), self.cfg["preprocessing"]["invert_threshold"])
        return self.estimate_gray(gray)


def estimate_ages_and_cache(image_ids: list[str], paths: list[str], device: str | None = None):
    """Segment + cache every image once and estimate its age from the thorax ROI.

    Returns (ages, quality_rows).
    """
    from app.seg_cache import segment_and_cache

    estimator = AgeEstimator(device=device)
    ages, rows = [], []
    for image_id, path in tqdm(list(zip(image_ids, paths)), desc="Segmenting + estimating age"):
        res = segment_and_cache(image_id, path, estimator.segmenter)
        ages.append(estimator.estimate_gray(res["gray"], res["seg"]["roi_box"]))
        rows.append(res["quality"])
    return ages, rows
