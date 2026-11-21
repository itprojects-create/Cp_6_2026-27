"""DenseNet-121 binary pneumonia classifier."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from app.config import get_config

ARCHITECTURES = ("densenet121", "xrv_densenet121")


class PneumoniaClassifier(nn.Module):
    """
    DenseNet-121 backbone with a two-logit head (Normal=0, Pneumonia=1).

    architecture
        densenet121      ImageNet-pretrained torchvision DenseNet-121
        xrv_densenet121  TorchXRayVision DenseNet-121 pretrained on chest X-rays

    Both variants take the same ImageNet-normalised 3-channel input so preprocessing
    is identical; the XRV variant converts it internally to its 1-channel
    [-1024, 1024] range.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        architecture: str = "densenet121",
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{architecture}'. Expected one of {ARCHITECTURES}.")
        self.config = get_config()
        self.num_classes = num_classes
        self.architecture = architecture
        self.decision_threshold = float(self.config["training"].get("classification_threshold", 0.5))

        backbone = models.densenet121(weights=None)
        if architecture == "densenet121":
            if pretrained:
                backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                import torchxrayvision as xrv

                xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
                backbone.features.load_state_dict(xrv_model.features.state_dict())

        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        self.backbone = backbone

        mean = torch.tensor(self.config["image"]["normalize_mean"]).view(1, 3, 1, 1)
        std = torch.tensor(self.config["image"]["normalize_std"]).view(1, 3, 1, 1)
        self.register_buffer("in_mean", mean, persistent=False)
        self.register_buffer("in_std", std, persistent=False)

        if weights_path and Path(weights_path).exists():
            self.load_checkpoint(weights_path)

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "xrv_densenet121":
            gray = (x * self.in_std + self.in_mean).mean(dim=1, keepdim=True)
            return (gray * 2.0 - 1.0) * 1024.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._prepare(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before classification head."""
        features = self.backbone.features(self._prepare(x))
        out = nn.functional.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def predict_proba(self, x: torch.Tensor) -> dict[str, float | str]:
        """Return class probabilities and the predicted label at the calibrated threshold."""
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(x), dim=1)[0]
        normal_prob = float(probs[0])
        pneumonia_prob = float(probs[1])
        is_pneumonia = pneumonia_prob >= self.decision_threshold
        class_names = self.config["labels"]["class_names"]
        return {
            "prediction": class_names[1] if is_pneumonia else class_names[0],
            "normal_probability": normal_prob,
            "pneumonia_probability": pneumonia_prob,
            "confidence": pneumonia_prob if is_pneumonia else normal_prob,
        }

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
            if "decision_threshold" in checkpoint:
                self.decision_threshold = float(checkpoint["decision_threshold"])
        else:
            self.load_state_dict(checkpoint)


def read_checkpoint_meta(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return {}


def build_classifier(
    weights_path: str | Path | None = None,
    pretrained_backbone: bool | None = None,
    architecture: str | None = None,
) -> PneumoniaClassifier:
    """Factory to build classifier from config."""
    cfg = get_config()
    num_classes = cfg["classification"]["num_classes"]
    pretrained = (
        pretrained_backbone
        if pretrained_backbone is not None
        else cfg["training"]["pretrained_backbone"]
    )
    custom = weights_path or cfg["training"].get("densenet_weights")
    return PneumoniaClassifier(
        num_classes=num_classes,
        pretrained=pretrained and custom is None,
        weights_path=custom,
        architecture=architecture or cfg["classification"]["architecture"],
    )


class ModelEnsemble(nn.Module):
    """
    One or more classifiers, each with its own input mode, whose pneumonia
    probabilities are averaged. A single model is a one-member ensemble.
    """

    def __init__(self, members: list[PneumoniaClassifier], input_modes: list[str], decision_threshold: float = 0.5):
        super().__init__()
        if len(members) != len(input_modes) or not members:
            raise ValueError("Each ensemble member needs exactly one input mode.")
        self.members = nn.ModuleList(members)
        self.input_modes = list(input_modes)
        self.decision_threshold = float(decision_threshold)
        self.config = get_config()

    @property
    def architecture(self) -> str:
        return " + ".join(f"{m.architecture}[{mode}]" for m, mode in zip(self.members, self.input_modes))

    @property
    def explain_member(self) -> int:
        """Member used for Grad-CAM: the first lung-focused member if present."""
        for i, mode in enumerate(self.input_modes):
            if mode != "full":
                return i
        return 0

    @torch.no_grad()
    def pneumonia_probability(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        probs = [torch.softmax(m(tensors[mode]), dim=1)[:, 1] for m, mode in zip(self.members, self.input_modes)]
        return torch.stack(probs).mean(dim=0)

    def predict_proba(self, tensors: dict[str, torch.Tensor]) -> dict[str, float | str]:
        p = float(self.pneumonia_probability(tensors)[0])
        is_pneumonia = p >= self.decision_threshold
        names = self.config["labels"]["class_names"]
        return {
            "prediction": names[1] if is_pneumonia else names[0],
            "pneumonia_probability": p,
            "normal_probability": 1.0 - p,
            "confidence": p if is_pneumonia else 1.0 - p,
        }


def ensemble_checkpoint(members: list[dict], decision_threshold: float, **meta) -> dict:
    """Checkpoint dict for an ensemble. Each member: {model_state_dict, architecture, input_mode}."""
    return {"ensemble": True, "members": members, "decision_threshold": float(decision_threshold), **meta}


def load_model_ensemble(path: str | Path, device: str | None = None) -> ModelEnsemble:
    """Load a single-model or ensemble checkpoint as a ModelEnsemble."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("ensemble"):
        specs = checkpoint["members"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        specs = [checkpoint]
    else:
        specs = [{"model_state_dict": checkpoint}]
    members, modes = [], []
    for spec in specs:
        model = build_classifier(pretrained_backbone=False, architecture=spec.get("architecture", "densenet121"))
        model.load_state_dict(spec["model_state_dict"])
        members.append(model)
        modes.append(spec.get("input_mode", "full"))
    threshold = checkpoint.get("decision_threshold", 0.5) if isinstance(checkpoint, dict) else 0.5
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return ModelEnsemble(members, modes, threshold).to(dev).eval()


def load_trained_classifier(
    model_type: str = "segmented",
    device: str | None = None,
) -> ModelEnsemble:
    """
    Load a trained classifier checkpoint.

    model_type: 'baseline' or 'segmented'
    """
    cfg = get_config()
    path = Path(cfg["paths"]["baseline_model" if model_type == "baseline" else "segmented_model"])
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. "
            "Please train the model first using backend/scripts/train_classifier.py"
        )
    return load_model_ensemble(path, device)
