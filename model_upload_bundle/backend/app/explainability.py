"""Grad-CAM explainability for DenseNet-121 predictions."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from app.classification import PneumoniaClassifier
from app.config import get_config


def _get_target_layers(model: PneumoniaClassifier):
    """Resolve Grad-CAM target layer from DenseNet-121."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise ImportError(
            "grad-cam is required for explainability. Install with: pip install grad-cam"
        ) from exc

    # Target the last conv layer in denseblock4
    target_layers = [model.backbone.features.denseblock4.denselayer16.conv2]
    return GradCAM, ClassifierOutputTarget, target_layers


def generate_gradcam(
    model: PneumoniaClassifier,
    input_tensor: torch.Tensor,
    display_rgb: np.ndarray,
    target_class: int | None = None,
) -> dict[str, np.ndarray | str]:
    """
    Generate Grad-CAM heatmap overlaid on the input image.

    Returns dict with heatmap, overlay, and base64-encoded overlay PNG.
    """
    GradCAM, ClassifierOutputTarget, target_layers = _get_target_layers(model)

    if target_class is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target_class = int(logits.argmax(dim=1).item())

    targets = [ClassifierOutputTarget(target_class)]

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

    # Resize CAM to display image size
    h, w = display_rgb.shape[:2]
    cam_resized = cv2.resize(grayscale_cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * display_rgb.astype(np.float32) + 0.4 * heatmap.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return {
        "heatmap": cam_resized,
        "overlay": overlay,
        "overlay_base64": array_to_base64(overlay),
    }


def array_to_base64(image: np.ndarray, fmt: str = "PNG") -> str:
    """Encode numpy RGB image as base64 data URI."""
    pil = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def save_gradcam_figure(
    original: np.ndarray,
    overlay: np.ndarray,
    save_path: str | Path,
    title: str = "Grad-CAM (interpretability only — not medical diagnosis)",
) -> None:
    """Save side-by-side Grad-CAM visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title("Input X-ray")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
