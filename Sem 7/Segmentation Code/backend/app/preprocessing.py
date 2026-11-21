"""Chest X-ray image preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import get_config

PathLike = Union[str, Path]
ImageInput = Union[PathLike, np.ndarray, Image.Image, bytes, BinaryIO]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


class PreprocessingError(Exception):
    """Raised when an image cannot be preprocessed."""


def _load_dicom(source: PathLike | bytes | BinaryIO) -> np.ndarray:
    try:
        import pydicom
    except ImportError as exc:
        raise PreprocessingError(
            "DICOM support requires pydicom. Install with: pip install pydicom"
        ) from exc

    if isinstance(source, (str, Path)):
        ds = pydicom.dcmread(str(source))
    elif isinstance(source, (bytes, bytearray)):
        import io

        ds = pydicom.dcmread(io.BytesIO(source))
    else:
        ds = pydicom.dcmread(source)

    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr - arr.min()
    return arr


def _is_dicom_bytes(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def load_image(source: ImageInput) -> np.ndarray:
    """
    Load an image from path, bytes, PIL, or numpy array.
    Returns grayscale float32 array in [0, 1].
    """
    if isinstance(source, np.ndarray):
        if np.issubdtype(source.dtype, np.floating) and source.size and 0.0 <= float(source.min()) and float(source.max()) <= 1.0:
            arr = source.astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            if float(arr.max()) <= 0:
                raise PreprocessingError("Image appears empty or corrupted.")
            return arr
        arr = source.astype(np.float32)
    elif isinstance(source, Image.Image):
        arr = np.array(source.convert("L"), dtype=np.float32)
    elif isinstance(source, (bytes, bytearray)):
        if _is_dicom_bytes(bytes(source)):
            return to_grayscale(_load_dicom(bytes(source)))
        arr = np.frombuffer(source, dtype=np.uint8)
        arr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise PreprocessingError("Unable to decode image bytes.")
        arr = arr.astype(np.float32)
    elif hasattr(source, "read"):
        data = source.read()
        if _is_dicom_bytes(data):
            return to_grayscale(_load_dicom(data))
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise PreprocessingError("Unable to decode uploaded image.")
        arr = arr.astype(np.float32)
    else:
        path = Path(source)
        if not path.exists():
            raise PreprocessingError(f"Image not found: {path}")
        if path.suffix.lower() == ".dcm":
            arr = _load_dicom(path)
        else:
            arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise PreprocessingError(f"Unable to read image: {path}")
            arr = arr.astype(np.float32)

    return to_grayscale(arr)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert multi-channel or high bit-depth image to grayscale [0, 1]."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2GRAY).astype(np.float32)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            image = image.mean(axis=2)

    image = image.astype(np.float32)
    max_val = float(image.max()) if image.size else 0.0
    if max_val <= 0:
        raise PreprocessingError("Image appears empty or corrupted.")

    if max_val > 255:
        image = (image - image.min()) / (max_val - image.min() + 1e-8)
    else:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def _mediastinum_contrast(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    band = image[int(h * 0.25) : int(h * 0.6)]
    centre = band[:, int(w * 0.45) : int(w * 0.55)].mean()
    left = band[:, int(w * 0.2) : int(w * 0.35)].mean()
    right = band[:, int(w * 0.65) : int(w * 0.8)].mean()
    return float(centre - (left + right) / 2.0)


def is_inverted(image: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Detect inverted X-rays (bright background, dark bones).

    A bright border alone is not sufficient: tightly collimated adult films and
    pediatric cassettes often have bright borders. An image is treated as inverted
    only when the border is bright and the mediastinum is darker than the lung fields.
    """
    border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
    return bool(border.mean() > threshold and _mediastinum_contrast(image) < 0.0)


def maybe_invert(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Invert the image only when it is detected as inverted."""
    if is_inverted(image, threshold):
        return 1.0 - image
    return image


def pad_to_square(image: np.ndarray, pad_value: float = 0.0) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad image to a square canvas. Returns the canvas and the (y, x) offset of the original."""
    h, w = image.shape[:2]
    side = max(h, w)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas = np.full((side, side), pad_value, dtype=np.float32)
    canvas[y_off : y_off + h, x_off : x_off + w] = image
    return canvas, (y_off, x_off)


def resize_with_padding(
    image: np.ndarray,
    target_size: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Resize preserving aspect ratio, pad to square target_size."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size), pad_value, dtype=np.float32)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def normalize_tensor(
    tensor: torch.Tensor,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    """Apply ImageNet-style normalization to a 3-channel tensor."""
    cfg = get_config()
    mean = mean or cfg["image"]["normalize_mean"]
    std = std or cfg["image"]["normalize_std"]
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def preprocess_image(
    source: ImageInput,
    target_size: int | None = None,
    return_rgb: bool = True,
    invert: bool = True,
) -> dict[str, np.ndarray | torch.Tensor]:
    """
    Full preprocessing pipeline.

    Returns dict with:
        - original_gray: grayscale [0,1] original aspect ratio
        - processed_gray: padded/resized grayscale
        - tensor: normalized 3-channel tensor [1, C, H, W] for classifier
        - display_rgb: uint8 RGB for visualization
    """
    cfg = get_config()
    size = target_size or cfg["image"]["input_size"]
    pad_color = cfg["preprocessing"]["pad_color"]
    invert_threshold = cfg["preprocessing"]["invert_threshold"]

    try:
        gray = load_image(source)
    except Exception as exc:
        raise PreprocessingError(str(exc)) from exc

    if invert:
        gray = maybe_invert(gray, threshold=invert_threshold)

    original_gray = gray.copy()
    processed = resize_with_padding(gray, size, pad_value=pad_color / 255.0 if pad_color else 0.0)

    # Build 3-channel tensor for DenseNet
    rgb = np.stack([processed] * 3, axis=0)
    tensor = torch.from_numpy(rgb).float()
    tensor = normalize_tensor(tensor)
    tensor = tensor.unsqueeze(0)

    display_rgb = (processed * 255).astype(np.uint8)
    display_rgb = np.stack([display_rgb] * 3, axis=-1)

    return {
        "original_gray": original_gray,
        "processed_gray": processed,
        "tensor": tensor,
        "display_rgb": display_rgb,
    }


def to_xrv_range(image: np.ndarray) -> np.ndarray:
    """Map a [0, 1] grayscale image to the [-1024, 1024] range used by TorchXRayVision."""
    return (image.astype(np.float32) * 2.0 - 1.0) * 1024.0


def preprocess_for_segmentation(source: ImageInput, invert: bool = True) -> torch.Tensor:
    """
    Preprocess image for TorchXRayVision PSPNet.

    The image is padded (not cropped) to a square so that the predicted mask can be
    mapped back to the full original image without misalignment.
    Returns tensor shape [1, 1, S, S] in the XRV [-1024, 1024] range.
    """
    cfg = get_config()
    seg_size = cfg["image"]["segmentation_size"]

    gray = load_image(source)
    if invert:
        gray = maybe_invert(gray, threshold=cfg["preprocessing"]["invert_threshold"])

    square, _ = pad_to_square(gray)
    resized = cv2.resize(square, (seg_size, seg_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(to_xrv_range(resized))[None, None].float()


INPUT_MODES = ("full", "masked", "roi", "roi_masked")


def soft_lung_mask(lung_mask: np.ndarray, dilation_fraction: float = 0.02) -> np.ndarray:
    """Dilate and feather a binary lung mask so lung borders and costophrenic angles are kept."""
    mask = lung_mask.astype(np.uint8)
    if not mask.any():
        return np.ones_like(mask, dtype=np.float32)
    h, w = mask.shape
    k = max(3, int(round(dilation_fraction * max(h, w))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask, kernel).astype(np.float32)
    return np.clip(cv2.GaussianBlur(dilated, (k, k), 0), 0.0, 1.0)


def build_classifier_image(
    gray: np.ndarray,
    lung_mask: np.ndarray | None,
    roi_box: list[int] | tuple[int, int, int, int] | None,
    mode: str,
    size: int,
) -> np.ndarray:
    """
    Build the square grayscale image fed to the classifier.

    full        whole radiograph
    masked      whole radiograph with non-lung pixels removed
    roi         crop to the lung bounding box (normalises thorax scale across ages)
    roi_masked  lung bounding box crop with non-lung pixels removed
    """
    if mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode '{mode}'. Expected one of {INPUT_MODES}.")
    image = gray.astype(np.float32)
    if mode in ("masked", "roi_masked") and lung_mask is not None:
        image = image * soft_lung_mask(lung_mask)
    if mode in ("roi", "roi_masked") and roi_box is not None:
        x0, y0, x1, y1 = [int(v) for v in roi_box]
        if x1 > x0 and y1 > y0:
            image = image[y0:y1, x0:x1]
    return resize_with_padding(image, size)


def classifier_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a square [0, 1] grayscale image into a normalised [1, 3, H, W] tensor."""
    rgb = np.stack([image] * 3, axis=0)
    return normalize_tensor(torch.from_numpy(rgb).float()).unsqueeze(0)
