"""Metrics and statistical summaries for binary lung-mask evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import cv2


def _binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask.")
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError(f"{name} must contain numeric binary values.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    values = np.unique(array)
    if not np.isin(values, [0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return array.astype(bool)


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return mask & ~eroded


def _boundary_metrics(prediction: np.ndarray, reference: np.ndarray) -> tuple[float | None, float | None]:
    if not prediction.any() or not reference.any():
        return None, None

    pred_surface = _surface(prediction)
    ref_surface = _surface(reference)
    to_reference = cv2.distanceTransform((~ref_surface).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    to_prediction = cv2.distanceTransform((~pred_surface).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    pred_distances = to_reference[pred_surface]
    ref_distances = to_prediction[ref_surface]
    hd95 = max(float(np.percentile(pred_distances, 95)), float(np.percentile(ref_distances, 95)))
    mean_distance = float((pred_distances.sum() + ref_distances.sum()) / (len(pred_distances) + len(ref_distances)))
    return hd95, mean_distance


def _overlap_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    dice_denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    predicted_count = tp + fp
    reference_count = tp + fn
    return {
        "dice": 1.0 if dice_denominator == 0 else 2.0 * tp / dice_denominator,
        "iou": 1.0 if union == 0 else tp / union,
        "precision": 1.0 if predicted_count == 0 and reference_count == 0 else (tp / predicted_count if predicted_count else 0.0),
        "recall": 1.0 if reference_count == 0 and predicted_count == 0 else (tp / reference_count if reference_count else 0.0),
    }


def compute_segmentation_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    predicted_left: np.ndarray | None = None,
    predicted_right: np.ndarray | None = None,
    reference_left: np.ndarray | None = None,
    reference_right: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute per-image overlap and boundary metrics; distances are in pixels.

    Both-empty masks score 1 for overlap metrics. Boundary distances are undefined
    (``None``) if either mask is empty. Left/right Dice is only returned if both
    corresponding predicted and reference masks are provided.
    """
    pred = _binary_mask(prediction, "prediction")
    ref = _binary_mask(reference, "reference")
    if pred.shape != ref.shape:
        raise ValueError(f"prediction and reference dimensions differ: {pred.shape} != {ref.shape}.")

    result: dict[str, float | None] = _overlap_metrics(pred, ref)
    result["hd95_px"], result["mean_surface_distance_px"] = _boundary_metrics(pred, ref)

    side_masks = (
        ("left_lung_dice", predicted_left, reference_left),
        ("right_lung_dice", predicted_right, reference_right),
    )
    for metric, side_prediction, side_reference in side_masks:
        if side_prediction is None and side_reference is None:
            continue
        if side_prediction is None or side_reference is None:
            raise ValueError(f"{metric} requires both predicted and reference masks.")
        pred_side = _binary_mask(side_prediction, f"{metric} prediction")
        ref_side = _binary_mask(side_reference, f"{metric} reference")
        if pred_side.shape != ref_side.shape or pred_side.shape != pred.shape:
            raise ValueError(f"{metric} masks must all have matching dimensions.")
        result[metric] = _overlap_metrics(pred_side, ref_side)["dice"]
    return result


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Return a percentile bootstrap CI for an image-level mean."""
    interval = bootstrap_statistic_interval(
        values,
        statistic="mean",
        confidence=confidence,
        samples=samples,
        seed=seed,
    )
    return {"n": interval["n"], "mean": interval["estimate"], "lower": interval["lower"], "upper": interval["upper"]}


def bootstrap_statistic_interval(
    values: Iterable[float],
    *,
    statistic: str = "mean",
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Return a percentile bootstrap CI for an image-level mean, median, or p95."""
    data = np.asarray(list(values), dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("values must be a one-dimensional sequence of finite numbers.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if samples < 1:
        raise ValueError("samples must be positive.")
    if statistic not in ("mean", "median", "p95"):
        raise ValueError("statistic must be 'mean', 'median', or 'p95'.")
    if data.size == 0:
        return {"n": 0, "estimate": None, "lower": None, "upper": None}
    reducer = {
        "mean": lambda values: values.mean(axis=-1),
        "median": lambda values: np.median(values, axis=-1),
        "p95": lambda values: np.percentile(values, 95, axis=-1),
    }[statistic]
    estimate = float(reducer(data))
    if data.size < 2:
        return {"n": int(data.size), "estimate": estimate, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    statistics = np.empty(samples, dtype=float)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, data.size, size=(stop - start, data.size))
        statistics[start:stop] = reducer(data[indices])
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(statistics, [alpha, 1.0 - alpha])
    return {"n": int(data.size), "estimate": estimate, "lower": float(lower), "upper": float(upper)}


def paired_bootstrap_difference(
    left: Iterable[float],
    right: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Bootstrap the paired mean difference ``left - right`` per image."""
    first = np.asarray(list(left), dtype=float)
    second = np.asarray(list(right), dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("paired metric sequences must be one-dimensional and have equal lengths.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired metric sequences must contain only finite numbers.")
    return bootstrap_mean_interval(first - second, confidence=confidence, samples=samples, seed=seed)


def summarize_metric_rows(rows: list[dict[str, Any]], metric_names: Iterable[str]) -> dict[str, Any]:
    """Aggregate per-image metrics and deterministic 95% bootstrap intervals."""
    summary: dict[str, Any] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary[metric] = bootstrap_mean_interval(values)
    return summary
