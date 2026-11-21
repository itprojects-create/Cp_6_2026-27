"""End-to-end inference pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from app.classification import ModelEnsemble, load_model_ensemble, load_trained_classifier
from app.config import get_config
from app.explainability import array_to_base64, generate_gradcam
from app.preprocessing import build_classifier_image, classifier_tensor, load_image, maybe_invert
from app.segmentation import LungSegmenter, get_segmenter

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Research prototype only. This system is not intended to diagnose, "
    "treat, or replace evaluation by a qualified medical professional."
)


def _gray_to_b64(gray: np.ndarray) -> str:
    return array_to_base64((np.clip(gray, 0, 1) * 255).astype(np.uint8))


def _mask_to_b64(mask: np.ndarray) -> str:
    return array_to_base64(np.stack([mask * 255] * 3, axis=-1).astype(np.uint8))


class InferenceEngine:
    """Loads models once at startup and runs the full prediction pipeline."""

    def __init__(self, model_type: str | None = None):
        self.config = get_config()
        self.model_type = model_type or self.config["inference"]["model_type"]
        self.device = self._resolve_device()
        self.classifier: ModelEnsemble | None = None
        self.segmenter: LungSegmenter | None = None
        self.model_loaded = False
        self.segmentation_loaded = False
        self.load_error: str | None = None
        self.model_status = "not_trained"

    @staticmethod
    def _resolve_device() -> str:
        pref = get_config()["inference"].get("device", "auto")
        if pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return pref

    def load_models(self) -> None:
        """Load segmentation and classification models."""
        try:
            self.segmenter = get_segmenter()
            self.segmenter.load()
            self.segmentation_loaded = True
        except Exception as exc:
            logger.warning("Segmentation model load failed: %s", exc)
            self.load_error = f"Segmentation model unavailable: {exc}"

        try:
            self.classifier = load_trained_classifier(self.model_type, self.device)
            self.model_status = "trained"
            self.model_loaded = True
            return
        except FileNotFoundError as exc:
            logger.info("Trained classifier not found: %s", exc)

        demo_path = Path(self.config["paths"].get("demo_model", ""))
        if demo_path.is_file():
            self.classifier = load_model_ensemble(demo_path, self.device)
            self.model_status = "demo"
            self.model_loaded = True
            return

        self.load_error = (
            "Classifier weights not found. Train the model with backend/scripts/train_classifier.py."
        )

    def health_status(self) -> dict:
        return {
            "segmentation_loaded": self.segmentation_loaded,
            "classifier_loaded": self.model_loaded,
            "model_type": self.model_type,
            "device": self.device,
            "ready": self.model_loaded and self.segmentation_loaded,
            "message": self.load_error,
        }

    def _load_gray(self, image_input) -> np.ndarray:
        return maybe_invert(load_image(image_input), self.config["preprocessing"]["invert_threshold"])

    def segment_only(self, image_input) -> dict:
        """Run segmentation without classification."""
        if not self.segmentation_loaded:
            raise RuntimeError(self.load_error or "Segmentation model not loaded.")
        result = self.segmenter.segment_gray(self._load_gray(image_input))
        return {
            "lung_mask_base64": _mask_to_b64(result["lung_mask"]),
            "lung_only_base64": _gray_to_b64(result["lung_only_image"]),
            "overlay_base64": array_to_base64(result["overlay"]),
            "original_base64": _gray_to_b64(result["original_gray"]),
            "segmentation_quality": result["quality"],
        }

    def predict(self, image_input) -> dict:
        """Full pipeline: preprocess -> segment -> classify (all ensemble members) -> Grad-CAM."""
        if not self.model_loaded:
            raise RuntimeError(self.load_error or "Classifier not loaded. Train models with train_classifier.py.")

        modes = self.classifier.input_modes
        gray = self._load_gray(image_input)
        size = self.config["image"]["input_size"]
        needs_lungs = any(m != "full" for m in modes)

        seg = None
        if self.segmentation_loaded:
            seg = self.segmenter.segment_gray(gray)
        elif needs_lungs:
            raise RuntimeError(self.load_error or "Segmentation required but model not loaded.")

        segmentation_failed = seg is not None and seg["quality"]["status"] == "failed"
        if segmentation_failed and needs_lungs:
            logger.warning("Lung segmentation failed; lung-based inputs fall back to the full radiograph.")

        lung_mask = seg["lung_mask"] if seg and not segmentation_failed else None
        roi_box = seg["roi_box"] if seg and not segmentation_failed else None
        images = {
            mode: build_classifier_image(gray, lung_mask, roi_box, mode if lung_mask is not None else "full", size)
            for mode in dict.fromkeys(modes)
        }
        tensors = {mode: classifier_tensor(img).to(self.device) for mode, img in images.items()}
        probs = self.classifier.predict_proba(tensors)

        k = self.classifier.explain_member
        explain_mode = modes[k]
        display = np.stack([(images[explain_mode] * 255).astype(np.uint8)] * 3, axis=-1)
        gradcam = generate_gradcam(
            self.classifier.members[k], tensors[explain_mode], display,
            target_class=1 if probs["prediction"] == "Pneumonia" else 0,
        )

        demo = self.model_status == "demo"
        response = {
            "model_status": "demo_development_model" if demo else "trained",
            "model_label": "DEMO / DEVELOPMENT MODEL" if demo else f"DenseNet-121 ensemble: {self.classifier.architecture}",
            "prediction": probs["prediction"],
            "pneumonia_probability": round(probs["pneumonia_probability"], 4),
            "normal_probability": round(probs["normal_probability"], 4),
            "confidence": round(probs["confidence"], 4),
            "decision_threshold": round(self.classifier.decision_threshold, 4),
            "model_type": self.model_type,
            "input_mode": "+".join(dict.fromkeys(modes)),
            "disclaimer": ("DEMO model - development only; NOT validated for clinical use." if demo else DISCLAIMER),
            "original_image_base64": _gray_to_b64(gray),
            "classifier_input_base64": _gray_to_b64(images[explain_mode]),
            "gradcam_base64": gradcam["overlay_base64"],
        }
        if seg is not None:
            response["lung_mask_base64"] = _mask_to_b64(seg["lung_mask"])
            response["lung_only_base64"] = _gray_to_b64(seg["lung_only_image"])
            response["overlay_base64"] = array_to_base64(seg["overlay"])
            response["segmentation_quality"] = seg["quality"]
        return response


_engine: InferenceEngine | None = None


def get_inference_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.load_models()
    return _engine
