"""Lung segmentation using TorchXRayVision PSPNet with thorax-ROI refinement.

Pediatric radiographs (especially infant "babygrams") show the whole body, so the
thorax occupies a small fraction of the frame. Running the 512x512 PSPNet on the
whole image leaves only a few dozen pixels per lung. The segmenter therefore runs a
coarse pass on the full image, locates the thorax, and re-segments a padded crop of
the thorax at full model resolution. All geometry uses padding (never cropping) so
masks map back exactly onto the original image.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.config import get_config
from app.preprocessing import load_image, maybe_invert, pad_to_square, to_xrv_range


class SegmentationModelNotAvailable(Exception):
    """Raised when the segmentation model cannot be loaded."""


Box = tuple[int, int, int, int]

BACKENDS = {
    "pspnet": {
        "lungs": ("Left Lung", "Right Lung"),
        "thorax": ("Heart", "Mediastinum", "Facies Diaphragmatica", "Left Clavicle", "Right Clavicle"),
    },
    "cxas": {
        "lungs": ("left lung", "right lung"),
        "thorax": ("cardiomediastinum", "diaphragm", "clavicles"),
    },
}


def _largest_component(mask: np.ndarray, return_share: bool = False):
    """Keep the largest connected component. Optionally also return its share of all foreground pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        empty = np.zeros_like(mask, dtype=np.uint8)
        return (empty, 0, 0.0) if return_share else (empty, 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    largest = (labels == keep).astype(np.uint8)
    if return_share:
        return largest, n - 1, float(areas.max() / areas.sum())
    return largest, n - 1


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood[1:-1, 1:-1] = mask
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood[1:-1, 1:-1] == 0).astype(np.uint8)
    return (mask | holes).astype(np.uint8)


def _bbox(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(box: Box, shape: tuple[int, int], margin: float) -> Box:
    """Expand a box by `margin` times its larger side on every edge, clipped to the image."""
    h, w = shape
    x0, y0, x1, y1 = box
    pad = int(round(margin * max(x1 - x0, y1 - y0)))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _build_backend(name: str):
    import torchxrayvision as xrv

    if name == "pspnet":
        return xrv.baseline_models.chestx_det.PSPNet()
    if name == "cxas":
        return xrv.baseline_models.chestx_anatomy.UNetResNet50()
    raise ValueError(f"Unknown segmentation backend '{name}'. Expected one of {sorted(BACKENDS)} or 'ensemble'.")


class LungSegmenter:
    """
    Lung segmentation with coarse-to-fine thorax refinement.

    backend (config `segmentation.model`)
        pspnet    TorchXRayVision ChestX-Det PSPNet
        cxas      TorchXRayVision CXAS UNet-ResNet50 (ChestXRayAnatomySegmentation)
        ensemble  mean of the PSPNet and CXAS lung probabilities
    """

    def __init__(self, device: str | None = None, backend: str | None = None):
        self.config = get_config()
        self.device = device or self._resolve_device()
        self.backend = backend or self.config.get("segmentation", {}).get("model", "pspnet")
        self.models: list[tuple[torch.nn.Module, list[int], list[int]]] = []

    @staticmethod
    def _resolve_device() -> str:
        cfg = get_config()
        pref = cfg["inference"].get("device", "auto")
        if pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return pref

    @property
    def seg_cfg(self) -> dict:
        return self.config.get("segmentation", {})

    def load(self) -> None:
        """Load the configured pretrained segmentation backend(s)."""
        try:
            import torchxrayvision  # noqa: F401
        except ImportError as exc:
            raise SegmentationModelNotAvailable(
                "torchxrayvision is required for lung segmentation. "
                "Install with: pip install torchxrayvision"
            ) from exc

        names = ["pspnet", "cxas"] if self.backend == "ensemble" else [self.backend]
        self.models = []
        for name in names:
            net = _build_backend(name).to(self.device).eval()
            targets = list(net.targets)
            spec = BACKENDS[name]
            lungs = [targets.index(t) for t in spec["lungs"]]
            thorax = [targets.index(t) for t in spec["thorax"] if t in targets]
            self.models.append((net, lungs, thorax))

    def _ensure_loaded(self) -> None:
        if not self.models:
            self.load()

    @torch.no_grad()
    def _infer(self, gray: np.ndarray, boxes: list[Box]) -> list[tuple[np.ndarray, np.ndarray, tuple]]:
        """
        Run the segmentation model(s) on square-padded crops of `gray`.

        Returns, per box, (lung_probs [2, S, S], thorax_probs [S, S], geometry) where
        geometry = (side, y_off, x_off) describes how the crop sits in its padded square.
        """
        size = int(self.config["image"]["segmentation_size"])
        batch_size = int(self.seg_cfg.get("batch_size", 4))
        inputs, geoms = [], []
        for x0, y0, x1, y1 in boxes:
            square, (y_off, x_off) = pad_to_square(gray[y0:y1, x0:x1])
            inputs.append(cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA))
            geoms.append((square.shape[0], y_off, x_off))

        results = []
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(to_xrv_range(np.stack(inputs[start : start + batch_size])))[:, None].float().to(self.device)
            lung_sum = torch.zeros((batch.shape[0], 2, size, size), device=self.device)
            thorax_max = torch.zeros((batch.shape[0], size, size), device=self.device)
            for net, lung_idx, thorax_idx in self.models:
                logits = net(batch)[:, lung_idx + thorax_idx]
                if logits.shape[-1] != size:
                    logits = torch.nn.functional.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
                probs = torch.sigmoid(logits)
                lung_sum += probs[:, :2]
                if thorax_idx:
                    thorax_max = torch.maximum(thorax_max, probs[:, 2:].amax(dim=1))
            lung = (lung_sum / len(self.models)).cpu().numpy()
            thorax = thorax_max.cpu().numpy()
            for i in range(batch.shape[0]):
                results.append((lung[i], thorax[i], geoms[start + i]))
        return results

    @staticmethod
    def _to_region(prob: np.ndarray, geom: tuple, box: Box, scale: float) -> np.ndarray:
        """Resample a model-resolution map onto the working-resolution pixels of `box`."""
        side, y_off, x_off = geom
        x0, y0, x1, y1 = box
        wx0, wy0 = int(round(x0 * scale)), int(round(y0 * scale))
        ww = max(1, int(round(x1 * scale)) - wx0)
        wh = max(1, int(round(y1 * scale)) - wy0)
        f = prob.shape[-1] / float(side)
        sx = (x1 - x0) / float(ww)
        sy = (y1 - y0) / float(wh)
        m = np.array([
            [sx * f, 0.0, (0.5 * sx + x_off) * f - 0.5],
            [0.0, sy * f, (0.5 * sy + y_off) * f - 0.5],
        ], dtype=np.float64)
        return cv2.warpAffine(prob, m, (ww, wh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    def _region_probs(self, gray: np.ndarray, boxes: list[Box], scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for (lung, thorax, geom), box in zip(self._infer(gray, boxes), boxes):
            out.append((
                np.stack([self._to_region(p, geom, box, scale) for p in lung]),
                self._to_region(thorax, geom, box, scale),
            ))
        return out

    @staticmethod
    def _clean(lung_probs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """Threshold, resolve overlap, keep the largest component per lung and fill holes."""
        left_p, right_p = lung_probs[0], lung_probs[1]
        left = (left_p > threshold) & (left_p >= right_p)
        right = (right_p > threshold) & (right_p > left_p)

        left, n_left, share_left = _largest_component(left, return_share=True)
        right, n_right, share_right = _largest_component(right, return_share=True)
        left = _fill_holes(left)
        right = _fill_holes(right)

        if left.any() and right.any():
            left_cx = np.nonzero(left)[1].mean()
            right_cx = np.nonzero(right)[1].mean()
            if left_cx < right_cx:
                left, right = right, left
                n_left, n_right = n_right, n_left
                share_left, share_right = share_right, share_left

        right = (right & (1 - left)).astype(np.uint8)
        return left.astype(np.uint8), right.astype(np.uint8), {
            "left_components": int(n_left),
            "right_components": int(n_right),
            "largest_component_share": round(0.5 * (share_left + share_right), 4),
        }

    @staticmethod
    def _score(
        left: np.ndarray,
        right: np.ndarray,
        lung_probs: np.ndarray,
        comp: dict,
        open_edges: tuple[bool, ...] = (True, True, True, True),
    ) -> float:
        """
        Anatomical plausibility of a two-lung segmentation in [0, 1].

        Rewards two confident, coherent (unfragmented), side-by-side lungs that fill a
        reasonable part of the region. Size asymmetry is penalised only mildly because
        consolidation or effusion legitimately shrinks the aerated lung.
        `open_edges` (left, top, right, bottom) marks region edges interior to the
        image; lungs touching them are truncated.
        """
        la, ra = int(left.sum()), int(right.sum())
        if la == 0 or ra == 0:
            return 0.0
        union = (left | right).astype(bool)
        conf = float(np.clip((np.maximum(lung_probs[0], lung_probs[1])[union].mean() - 0.5) / 0.5, 0.0, 1.0))
        ratio = (min(la, ra) / max(la, ra)) ** 0.25
        coherence = float(comp.get("largest_component_share", 1.0)) ** 0.5
        ly, ry = np.nonzero(left)[0], np.nonzero(right)[0]
        lh, rh = np.ptp(ly) + 1, np.ptp(ry) + 1
        align = max(0.0, 1.0 - abs(ly.mean() - ry.mean()) / (0.5 * (lh + rh)))
        x0, y0, x1, y1 = _bbox(union.astype(np.uint8))
        h, w = union.shape
        fill = min(1.0, ((x1 - x0) * (y1 - y0)) / (0.15 * h * w))
        truncated = (open_edges[0] and x0 == 0) or (open_edges[1] and y0 == 0) or (open_edges[2] and x1 == w) or (open_edges[3] and y1 == h)
        return float(conf * ratio * coherence * align * np.sqrt(fill) * (0.6 if truncated else 1.0))

    def _window_boxes(self, h: int, w: int) -> list[Box]:
        short = min(h, w)
        boxes: list[Box] = []
        for frac in self.seg_cfg.get("window_scales", [0.35, 0.55, 0.8]):
            side = max(16, int(round(frac * short)))
            stride = max(1, side // 2)
            ys = sorted(set(list(range(0, max(1, h - side + 1), stride)) + [max(0, h - side)]))
            xs = sorted(set(list(range(0, max(1, w - side + 1), stride)) + [max(0, w - side)]))
            boxes += [(x, y, min(w, x + side), min(h, y + side)) for y in ys for x in xs]
        return boxes

    def _evaluate_box(self, gray: np.ndarray, box: Box, scale: float, threshold: float, probs=None) -> dict:
        h, w = gray.shape
        lung, thorax = probs if probs is not None else self._region_probs(gray, [box], scale)[0]
        left, right, comp = self._clean(lung, threshold)
        x0, y0, x1, y1 = box
        edges = (x0 > 0, y0 > 0, x1 < w, y1 < h)
        return {"box": box, "lung": lung, "thorax": thorax, "left": left, "right": right, "comp": comp,
                "score": self._score(left, right, lung, comp, edges)}

    @staticmethod
    def _lung_box_original(cand: dict, scale: float, shape: tuple[int, int]) -> Box | None:
        b = _bbox(cand["left"] | cand["right"])
        if b is None:
            return None
        h, w = shape
        wx0, wy0 = int(round(cand["box"][0] * scale)), int(round(cand["box"][1] * scale))
        return (
            max(0, int(np.floor((b[0] + wx0) / scale))),
            max(0, int(np.floor((b[1] + wy0) / scale))),
            min(w, int(np.ceil((b[2] + wx0) / scale))),
            min(h, int(np.ceil((b[3] + wy0) / scale))),
        )

    def predict_masks(self, image_input) -> dict[str, np.ndarray]:
        """Return left, right and combined lung masks aligned to the input image."""
        result = self.segment_lungs(image_input)
        return {
            "left_mask": result["left_lung_mask"],
            "right_mask": result["right_lung_mask"],
            "combined": result["lung_mask"],
        }

    def segment_gray(self, gray: np.ndarray) -> dict:
        """
        Segment an already loaded, correctly oriented grayscale image in [0, 1].

        1. Coarse pass on the whole (padded) image.
        2. If the coarse result is not a plausible pair of lungs (e.g. bowel gas
           labelled as lung on a babygram, or a tiny neonatal thorax), search square
           windows at several scales and keep the most plausible one.
        3. Re-segment the thorax box at full model resolution, keeping a refinement
           only if it does not reduce plausibility.
        """
        self._ensure_loaded()
        cfg = self.seg_cfg
        threshold = float(cfg.get("threshold", 0.5))
        margin = float(cfg.get("roi_margin", 0.12))
        max_refinements = int(cfg.get("refinement_passes", 2))
        accept = float(cfg.get("plausibility_threshold", 0.55))
        h, w = gray.shape
        scale = min(1.0, float(cfg.get("working_size", 1024)) / max(h, w))

        full_box: Box = (0, 0, w, h)
        best = self._evaluate_box(gray, full_box, scale, threshold)
        coarse_score = best["score"]
        window_search = False
        if best["score"] < accept:
            window_search = True
            boxes = self._window_boxes(h, w)
            for box, probs in zip(boxes, self._region_probs(gray, boxes, scale)):
                cand = self._evaluate_box(gray, box, scale, threshold, probs)
                if cand["score"] > best["score"]:
                    best = cand

        refinements = 0
        for _ in range(max_refinements):
            lung_box = self._lung_box_original(best, scale, (h, w))
            if lung_box is None:
                break
            roi = expand_box(lung_box, (h, w), margin)
            if roi == best["box"]:
                break
            cand = self._evaluate_box(gray, roi, scale, threshold)
            if cand["score"] < 0.9 * best["score"]:
                break
            best = cand
            refinements += 1

        wh, ww = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        work_lung = np.zeros((2, wh, ww), np.float32)
        work_left = np.zeros((wh, ww), np.uint8)
        work_right = np.zeros((wh, ww), np.uint8)
        bx0, by0 = int(round(best["box"][0] * scale)), int(round(best["box"][1] * scale))
        rh, rw = best["left"].shape
        rh, rw = min(rh, wh - by0), min(rw, ww - bx0)
        work_lung[:, by0 : by0 + rh, bx0 : bx0 + rw] = best["lung"][:, :rh, :rw]
        work_left[by0 : by0 + rh, bx0 : bx0 + rw] = best["left"][:rh, :rw]
        work_right[by0 : by0 + rh, bx0 : bx0 + rw] = best["right"][:rh, :rw]

        def up(mask: np.ndarray) -> np.ndarray:
            if scale == 1.0:
                return mask.astype(np.uint8)
            return (cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

        left, right = up(work_left), up(work_right)
        right = (right & (1 - left)).astype(np.uint8)
        lung_mask = (left | right).astype(np.uint8)

        quality = self._quality(work_left, work_right, work_lung, best["comp"], best["score"], accept)
        quality.update({
            "refinement_passes": refinements,
            "window_search": window_search,
            "coarse_score": round(coarse_score, 4),
            "search_box": [int(v) for v in best["box"]],
        })

        lung_box = _bbox(lung_mask)
        roi_box = expand_box(lung_box, (h, w), float(cfg.get("classifier_roi_margin", 0.08))) if lung_box else full_box

        lung_only = np.where(lung_mask > 0, gray, 0.0).astype(np.float32)

        overlay = np.stack([(gray * 255).astype(np.uint8)] * 3, axis=-1)
        colored = overlay.copy()
        colored[left > 0] = (255, 120, 0)
        colored[right > 0] = (0, 200, 255)
        overlay = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), max(2, int(round(max(h, w) / 400))))

        return {
            "lung_mask": lung_mask,
            "left_lung_mask": left,
            "right_lung_mask": right,
            "lung_only_image": lung_only,
            "overlay": overlay,
            "original_gray": gray,
            "roi_box": [int(v) for v in roi_box],
            "quality": quality,
        }

    @staticmethod
    def _quality(left: np.ndarray, right: np.ndarray, lung_probs: np.ndarray, comp: dict, score: float, accept: float) -> dict:
        la, ra = int(left.sum()), int(right.sum())
        union = (left | right).astype(bool)
        mean_conf = float(np.maximum(lung_probs[0], lung_probs[1])[union].mean()) if union.any() else 0.0
        ratio = float(min(la, ra) / max(la, ra)) if max(la, ra) > 0 else 0.0
        box = _bbox(union.astype(np.uint8))
        box_frac = ((box[2] - box[0]) * (box[3] - box[1]) / float(union.size)) if box else 0.0
        if la == 0 or ra == 0:
            status = "failed"
        elif score >= accept:
            status = "ok"
        else:
            status = "low_confidence"
        return {
            "status": status,
            "plausibility": round(float(score), 4),
            "left_area_fraction": float(la / union.size),
            "right_area_fraction": float(ra / union.size),
            "lung_area_fraction": float(union.mean()),
            "lung_box_fraction": float(box_frac),
            "left_right_ratio": ratio,
            "mean_confidence": mean_conf,
            **comp,
        }

    def segment_lungs(
        self,
        image_input,
        debug: bool = False,
        debug_dir: str | Path | None = None,
    ) -> dict:
        """
        Segment lungs from a chest X-ray.

        Returns masks aligned to the original image size, the lung-only image, an
        overlay, the classifier ROI box and segmentation quality metrics.
        """
        cfg = self.config
        gray = load_image(image_input)
        gray = maybe_invert(gray, threshold=cfg["preprocessing"]["invert_threshold"])
        result = self.segment_gray(gray)

        if debug or self.seg_cfg.get("debug_save"):
            save_dir = Path(debug_dir or self.seg_cfg["debug_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_debug_images(result, save_dir)

        return result

    @staticmethod
    def _save_debug_images(result: dict, save_dir: Path) -> None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Original", "Lung Mask", "Overlay", "Lung-only"]
        images = [
            result["original_gray"],
            result["lung_mask"],
            result["overlay"],
            result["lung_only_image"],
        ]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_dir / "segmentation_debug.png", dpi=150, bbox_inches="tight")
        plt.close()


_default_segmenter: LungSegmenter | None = None


def get_segmenter() -> LungSegmenter:
    global _default_segmenter
    if _default_segmenter is None:
        _default_segmenter = LungSegmenter()
    return _default_segmenter


def segment_lungs(image_input, debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Segment lungs from image. Returns (lung_mask, lung_only_image)."""
    segmenter = get_segmenter()
    result = segmenter.segment_lungs(image_input, debug=debug)
    return result["lung_mask"], result["lung_only_image"]


def segment(image_input, model: str = "auto", debug: bool = False) -> dict:
    """Unified segmentation interface. Supported models: 'auto' (configured), 'pspnet', 'cxas', 'ensemble'."""
    if model not in ("auto", "pspnet", "cxas", "ensemble"):
        return {"status": f"unknown_model:{model}", "model_used": None}
    try:
        segmenter = get_segmenter() if model == "auto" else LungSegmenter(backend=model)
        seg = segmenter.segment_lungs(image_input, debug=debug)
        seg["model_used"] = segmenter.backend
        seg["status"] = "ok"
        return seg
    except Exception as exc:
        return {"status": f"error:{exc}", "model_used": None}
