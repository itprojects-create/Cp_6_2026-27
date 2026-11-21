"""Read-only benchmark metadata and result loading for the research API."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import get_config

logger = logging.getLogger(__name__)

_METHODS = [
    {
        "id": "cxas",
        "name": "CXAS U-Net–ResNet50",
        "architecture": "TorchXRayVision chestx_anatomy.UNetResNet50",
        "family": "U-Net encoder–decoder + ResNet-50 backbone",
        "description": "Pretrained multi-anatomy semantic segmenter; separate left/right lung channels are used by the live application.",
        "implementation_status": "integrated",
        "literature_evidence": "Exact chest-radiograph accuracy figures for this checkpoint are not asserted here. Local score requires reference masks.",
        "references": [
            {"label": "TorchXRayVision model documentation", "url": "https://mlmed.org/torchxrayvision/models.html"},
            {"label": "TorchXRayVision source", "url": "https://github.com/mlmed/torchxrayvision"},
        ],
        "published_metrics": [],
    },
    {
        "id": "unet",
        "name": "U-Net",
        "architecture": "Symmetric contracting/expanding convolutional encoder-decoder with skip connections",
        "family": "CNN encoder–decoder",
        "description": "Baseline biomedical segmentation design; local PneumoShift implementation is not present.",
        "implementation_status": "literature_only",
        "literature_evidence": "Original study evaluates biomedical microscopy, not chest-radiograph lungs; its metrics are not a CXR benchmark.",
        "references": [
            {"label": "Ronneberger et al., MICCAI 2015", "url": "https://arxiv.org/abs/1505.04597"},
        ],
        "published_metrics": [],
    },
    {
        "id": "unetpp",
        "name": "U-Net++",
        "architecture": "Nested dense skip pathways bridge encoder/decoder semantic gaps",
        "family": "Nested CNN encoder–decoder",
        "description": "U-Net extension designed to connect feature maps at multiple semantic levels.",
        "implementation_status": "not_integrated",
        "literature_evidence": "No same-dataset PneumoShift result; local validation needs reference masks and an implementation.",
        "references": [
            {"label": "Zhou et al., U-Net++", "url": "https://arxiv.org/abs/1807.10165"},
        ],
        "published_metrics": [],
    },
    {
        "id": "attention_unet",
        "name": "Attention U-Net",
        "architecture": "U-Net with attention gates on skip connections",
        "family": "Attention CNN encoder–decoder",
        "description": "Attention gates adaptively emphasize relevant spatial features.",
        "implementation_status": "not_integrated",
        "literature_evidence": "No same-dataset PneumoShift result; local validation needs reference masks and an implementation.",
        "references": [
            {"label": "Oktay et al., Attention U-Net", "url": "https://arxiv.org/abs/1804.03999"},
        ],
        "published_metrics": [],
    },
    {
        "id": "resunet",
        "name": "ResUNet",
        "architecture": "U-Net-style encoder-decoder with residual blocks",
        "family": "Residual CNN encoder–decoder",
        "description": "Residual learning is used in encoder/decoder blocks to support deeper feature extraction.",
        "implementation_status": "not_integrated",
        "literature_evidence": "Architecture family only; no common-dataset local lung-mask score is available.",
        "references": [
            {"label": "He et al., Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/abs/1512.03385"},
        ],
        "published_metrics": [],
    },
    {
        "id": "transunet",
        "name": "TransUNet",
        "architecture": "Transformer encoder with U-Net convolutional decoder",
        "family": "Transformer + CNN encoder–decoder",
        "description": "Uses self-attention for broader context with a U-shaped localization decoder.",
        "implementation_status": "not_integrated",
        "literature_evidence": "No local checkpoint or same-image benchmark in this repository.",
        "references": [
            {"label": "Chen et al., TransUNet", "url": "https://arxiv.org/abs/2102.04306"},
        ],
        "published_metrics": [],
    },
    {
        "id": "transresunet",
        "name": "TransResUNet",
        "architecture": "Transformer attention + residual U-Net",
        "family": "Transformer + residual CNN",
        "description": "Pediatric CXR method combining TransUNet attention and ResUNet residual features.",
        "implementation_status": "literature_only",
        "literature_evidence": "Published Dice is from the paper's multicentre pediatric test set and is not a local or head-to-head score.",
        "references": [
            {
                "label": "Chen et al., Frontiers in Radiology 2023",
                "url": "https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2023.1190745/full",
            }
        ],
        "published_metrics": [
            {
                "metric": "Dice",
                "value": 0.9822,
                "dataset": "Multicentre pediatric chest-radiograph test set in the paper",
                "evidence_type": "published_literature",
            }
        ],
    },
    {
        "id": "dense_unet",
        "name": "Dense U-Net",
        "architecture": "U-Net encoder-decoder with densely connected feature blocks",
        "family": "Dense CNN encoder–decoder",
        "description": "Dense feature reuse is introduced into a U-shaped segmentation network.",
        "implementation_status": "not_integrated",
        "literature_evidence": "Dense U-Net variants differ by implementation and dataset; no single transferable CXR score is displayed.",
        "references": [
            {"label": "Chest-radiograph lung segmentation systematic survey", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8741572/"},
        ],
        "published_metrics": [],
    },
    {
        "id": "pspnet",
        "name": "PSPNet",
        "architecture": "TorchXRayVision chestx_det.PSPNet",
        "family": "CNN + pyramid pooling",
        "description": "Pyramid pooling aggregates multi-scale contextual features for pixel-level prediction.",
        "implementation_status": "integrated_alternative",
        "literature_evidence": "The original paper's scene-parsing scores are not lung-segmentation results and are not compared as such.",
        "references": [
            {
                "label": "Zhao et al., Pyramid Scene Parsing Network, CVPR 2017",
                "url": "https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html",
            },
            {"label": "TorchXRayVision source", "url": "https://github.com/mlmed/torchxrayvision"},
        ],
        "published_metrics": [],
    },
    {
        "id": "deeplabv3plus",
        "name": "DeepLabV3+",
        "architecture": "Atrous spatial pyramid pooling with an encoder-decoder boundary refinement",
        "family": "CNN + atrous spatial pyramid",
        "description": "Multi-scale atrous context with a decoder to refine object boundaries.",
        "implementation_status": "not_integrated",
        "literature_evidence": "No local chest-radiograph checkpoint or reference-mask evaluation.",
        "references": [
            {
                "label": "Chen et al., ECCV 2018",
                "url": "https://openaccess.thecvf.com/content_ECCV_2018/html/Chen_Encoder-Decoder_with_Atrous_Separable_Convolution_for_Semantic_Image_Segmentation_ECCV_2018_paper.html",
            }
        ],
        "published_metrics": [],
    },
    {
        "id": "u2net",
        "name": "U²-Net",
        "architecture": "Nested residual U-blocks arranged as a two-level U-structure",
        "family": "Nested CNN",
        "description": "Designed for salient-object detection; chest-lung adaptation would need independent validation.",
        "implementation_status": "not_integrated",
        "literature_evidence": "Original task is salient-object detection, not a CXR lung benchmark; its speed/size are not shown as lung results.",
        "references": [
            {"label": "Qin et al., U²-Net", "url": "https://arxiv.org/abs/2005.09007"},
        ],
        "published_metrics": [],
    },
    {
        "id": "hybridgnet",
        "name": "HybridGNet",
        "architecture": "Landmark-based graph-convolutional anatomical segmentation",
        "family": "Landmark graph neural network",
        "description": "Predicts anatomical landmarks and uses graph-based modeling to derive anatomical contours.",
        "implementation_status": "not_integrated",
        "literature_evidence": "Used to create CheXmask, a 657,566-mask multi-dataset resource with physician review and automatic QC; not a local benchmark.",
        "references": [
            {
                "label": "Gaggion et al., MICCAI 2021",
                "url": "https://link.springer.com/chapter/10.1007/978-3-030-87193-2_57",
            },
            {
                "label": "CheXmask, Scientific Data 2024",
                "url": "https://www.nature.com/articles/s41597-024-03358-1",
            },
        ],
        "published_metrics": [],
    },
]
_XRV_CHECKPOINTS = {
    "cxas": "baseline_models_cxas_UNet_ResNet50_default_repack.pth",
    "pspnet": "pspnet_chestxray_best_model_4.pth",
}


def _xrv_cache_dir() -> Path:
    if importlib.util.find_spec("torchxrayvision") is None:
        return Path.home() / ".torchxrayvision" / "models_data"
    from torchxrayvision.utils import get_cache_dir

    return Path(get_cache_dir())


def _paths() -> tuple[Path, Path]:
    cfg = get_config()
    return (
        Path(cfg["paths"]["segmentation_reference_manifest"]),
        Path(cfg["paths"]["segmentation_benchmark_results"]),
    )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_segmentation_models() -> dict[str, Any]:
    """Return methods and actual code/config integration status."""
    cfg = get_config()
    backend = cfg.get("segmentation", {}).get("model", "pspnet")
    xrv_installed = importlib.util.find_spec("torchxrayvision") is not None
    methods = []
    cache_dir = _xrv_cache_dir()
    for method in _METHODS:
        entry = dict(method)
        entry["selected"] = entry["id"] == backend
        if entry["id"] in ("cxas", "pspnet"):
            checkpoint_name = _XRV_CHECKPOINTS[entry["id"]]
            checkpoint_path = cache_dir / checkpoint_name
            checkpoint_cached = checkpoint_path.is_file()
            checkpoint_size_bytes = checkpoint_path.stat().st_size if checkpoint_cached else None
            entry["runtime_dependency_available"] = xrv_installed
            entry["checkpoint_file"] = checkpoint_name
            entry["checkpoint_size_bytes"] = checkpoint_size_bytes
            entry["checkpoint_cached"] = checkpoint_cached
            entry["can_run_with_local_assets"] = bool(xrv_installed and checkpoint_cached)
            entry["local_checkpoint"] = (
                f"TorchXRayVision-managed pretrained checkpoint ({checkpoint_name}); "
                f"{'cached locally' if checkpoint_cached else 'not cached locally'}; no project-owned checkpoint."
            )
        else:
            entry["runtime_dependency_available"] = False
            entry["local_checkpoint"] = None
            entry["checkpoint_file"] = None
            entry["checkpoint_size_bytes"] = None
            entry["checkpoint_cached"] = False
            entry["can_run_with_local_assets"] = False
        methods.append(entry)
    return {
        "configured_backend": backend,
        "torchxrayvision_version": _package_version("torchxrayvision"),
        "methods": methods,
    }


def get_reference_status() -> dict[str, Any]:
    """Inspect the explicitly curated reference manifest without exposing local paths."""
    manifest_path, _ = _paths()
    if not manifest_path.is_file():
        return {
            "status": "not_configured",
            "reference_count": 0,
            "manifest_present": False,
            "message": "No independent reference-mask manifest is configured. Pixel-level validation is pending.",
        }
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image_path", "mask_path", "mask_mode"}
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                return {
                    "status": "invalid_manifest",
                    "reference_count": 0,
                    "manifest_present": True,
                    "message": "Reference manifest must include image_path, mask_path, and mask_mode columns.",
                }
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        logger.exception("Could not read segmentation reference manifest")
        return {
            "status": "invalid_manifest",
            "reference_count": 0,
            "manifest_present": True,
            "message": f"Reference manifest could not be read: {exc}",
        }
    return {
        "status": "ready" if rows else "empty",
        "reference_count": len(rows),
        "manifest_present": True,
        "message": "Reference masks are curated annotations; their contents and geometry are checked by the benchmark runner.",
    }


def get_latest_benchmark() -> dict[str, Any]:
    _, result_dir = _paths()
    latest = result_dir / "latest.json"
    if not latest.is_file():
        return {
            "status": "pending_reference_masks",
            "run": None,
            "message": "No benchmark result is available. Dice, IoU, precision, recall and boundary metrics require independent reference masks.",
        }
    try:
        with latest.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict) or not isinstance(result.get("models"), dict) or not result.get("run_id"):
            raise ValueError("Saved benchmark result has an invalid structure.")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("Could not read latest segmentation benchmark")
        return {
            "status": "invalid_result",
            "run": None,
            "message": f"The saved benchmark result could not be read: {exc}",
        }
    return {"status": "available", "run": result, "message": "Locally measured against the reference masks in this run."}


def get_benchmark_run(run_id: str) -> dict[str, Any] | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        return None
    _, result_dir = _paths()
    run_path = result_dir / f"{run_id}.json"
    if not run_path.is_file():
        return None
    try:
        with run_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict) or result.get("run_id") != run_id or not isinstance(result.get("models"), dict):
            raise ValueError("Saved benchmark run has an invalid structure.")
        return result
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("Could not read segmentation benchmark run %s", run_id)
        raise ValueError(f"Saved benchmark run '{run_id}' is invalid: {exc}") from exc


def get_benchmark_example(run_id: str, model: str, sample_id: str, asset: str) -> Path | None:
    if model not in ("cxas", "pspnet") or asset not in ("original", "predicted_mask", "overlay"):
        return None
    if not re.fullmatch(r"[a-f0-9]{16}", sample_id):
        return None
    run = get_benchmark_run(run_id)
    if run is None:
        return None
    examples = run.get("models", {}).get(model, {}).get("visual_examples", [])
    if not any(item.get("sample_id") == sample_id for item in examples):
        return None
    _, result_dir = _paths()
    example_path = result_dir / run_id / "examples" / model / sample_id / f"{asset}.png"
    return example_path if example_path.is_file() else None
