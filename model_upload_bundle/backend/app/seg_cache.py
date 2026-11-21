"""Segmentation cache: one segmentation per image, reused by training and evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import PROJECT_ROOT, get_config
from app.preprocessing import INPUT_MODES, build_classifier_image, load_image, maybe_invert
from app.segmentation import LungSegmenter

QUALITY_FIELDS = (
    "status", "plausibility", "lung_area_fraction", "lung_box_fraction", "left_right_ratio",
    "mean_confidence", "left_components", "right_components", "refinement_passes", "window_search",
)


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_path(image_id: str) -> Path:
    return Path(get_config()["paths"]["seg_cache"]) / f"{image_id}.npz"


def segment_and_cache(image_id: str, image_path: str, segmenter: LungSegmenter, size: int | None = None) -> dict:
    """Segment one image, write its cache entry and return {gray, seg, quality row}."""
    cfg = get_config()
    size = size or cfg["image"]["input_size"]
    gray = maybe_invert(load_image(str(resolve_image_path(image_path))), cfg["preprocessing"]["invert_threshold"])
    seg = segmenter.segment_gray(gray)
    inputs = np.stack([
        (build_classifier_image(gray, seg["lung_mask"], seg["roi_box"], mode, size) * 255).round().astype(np.uint8)
        for mode in INPUT_MODES
    ])
    h, w = gray.shape
    scale = 512.0 / max(h, w)
    small = cv2.resize(seg["lung_mask"], (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    out = cache_path(image_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, inputs=inputs, lung_mask=small, roi_box=np.array(seg["roi_box"]))
    row = {"image_id": image_id, **{k: seg["quality"][k] for k in QUALITY_FIELDS}}
    row.update(dict(zip(("roi_x0", "roi_y0", "roi_x1", "roi_y1"), seg["roi_box"])))
    return {"gray": gray, "seg": seg, "quality": row}
