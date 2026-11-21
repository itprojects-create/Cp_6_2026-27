"""Training utilities shared by the training, cross-validation and evaluation scripts."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.classification import build_classifier
from app.config import get_config
from app.preprocessing import INPUT_MODES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(augment: bool, size: int | None = None):
    """Per-image torchvision transforms (CPU). Training uses the batched GPU equivalent augment_batch."""
    cfg = get_config()
    size = size or cfg["image"]["input_size"]
    mean, std = cfg["image"]["normalize_mean"], cfg["image"]["normalize_std"]
    aug = cfg["augmentation"]
    steps = []
    if augment and aug["enabled"]:
        steps += [
            transforms.RandomAffine(
                degrees=aug["rotation_degrees"],
                translate=(aug["translate_fraction"], aug["translate_fraction"]),
                scale=tuple(aug["scale_range"]),
            ),
            transforms.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
        ]
    steps += [
        transforms.Resize((size, size), antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


def normalize_batch(gray: torch.Tensor) -> torch.Tensor:
    """[B, 1, H, W] grayscale in [0, 1] -> ImageNet-normalised [B, 3, H, W]."""
    cfg = get_config()
    mean = torch.tensor(cfg["image"]["normalize_mean"], device=gray.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["image"]["normalize_std"], device=gray.device).view(1, 3, 1, 1)
    return (gray.expand(-1, 3, -1, -1) - mean) / std


def augment_batch(gray: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Batched GPU RandomAffine + ColorJitter(brightness, contrast) on [B, 1, H, W] in [0, 1]."""
    aug = get_config()["augmentation"]
    if not aug["enabled"]:
        return gray
    b = gray.shape[0]
    dev = gray.device

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(b, device=dev, generator=generator)

    angle = uniform(-aug["rotation_degrees"], aug["rotation_degrees"]) * math.pi / 180.0
    scale = uniform(*aug["scale_range"])
    t = aug["translate_fraction"] * 2.0
    tx, ty = uniform(-t, t), uniform(-t, t)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(gray.shape), align_corners=False)
    out = F.grid_sample(gray, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    out = out * uniform(1 - aug["brightness"], 1 + aug["brightness"]).view(b, 1, 1, 1)
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = (out - mean) * uniform(1 - aug["contrast"], 1 + aug["contrast"]).view(b, 1, 1, 1) + mean
    return out.clamp(0.0, 1.0)


class CachedXrayDataset(Dataset):
    """Serves cached classifier inputs produced by scripts/cache_segmentation.py."""

    def __init__(self, df: pd.DataFrame, mode: str, augment: bool = False, cache_dir: str | Path | None = None):
        cfg = get_config()
        self.df = df.reset_index(drop=True)
        self.mode_idx = INPUT_MODES.index(mode)
        self.transform = get_transforms(augment)
        cache = Path(cache_dir or cfg["paths"]["seg_cache"])
        images = []
        for image_id in self.df["image_id"]:
            path = cache / f"{image_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing cache entry {path}. Run scripts/cache_segmentation.py first.")
            images.append(np.load(path)["inputs"][self.mode_idx])
        self.images = np.stack(images) if images else np.zeros((0, 1, 1), np.uint8)
        self.labels = self.df["label"].astype(int).to_numpy()
        self._device_images: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        pil = Image.fromarray(self.images[idx])
        return self.transform(pil), int(self.labels[idx])

    def device_images(self, device: str) -> torch.Tensor:
        if device not in self._device_images:
            self._device_images[device] = torch.from_numpy(self.images).to(device)[:, None]
        return self._device_images[device]

    def batch(self, idx: torch.Tensor, device: str, generator: torch.Generator | None = None) -> torch.Tensor:
        """Normalised [B, 3, H, W] batch; augmented when a generator is given."""
        gray = self.device_images(device)[idx.to(device)].float() / 255.0
        if generator is not None:
            gray = augment_batch(gray, generator)
        return normalize_batch(gray)


def balanced_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Weights that give every (age band, label) cell equal total weight, mean 1."""
    cells = df["age_band"].astype(str) + "_" + df["label"].astype(str)
    counts = cells.map(cells.value_counts())
    w = 1.0 / counts.to_numpy(dtype=np.float64)
    return w / w.mean()


@torch.no_grad()
def predict(model: nn.Module, data: DataLoader | CachedXrayDataset, device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    if isinstance(data, DataLoader):
        data = data.dataset
    probs = []
    for start in range(0, len(data), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(data)))
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(data.batch(idx, device))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    mode: str,
    architecture: str,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    balance: bool = True,
    patience: int = 6,
    seed: int = 42,
    verbose: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[nn.Module, dict]:
    """Train a classifier and return the best model by validation ROC-AUC (ties: lower loss)."""
    set_seed(seed)
    cfg = get_config()
    train_ds = CachedXrayDataset(train_df, mode, cache_dir=cache_dir)
    val_ds = CachedXrayDataset(val_df, mode, cache_dir=cache_dir)
    order_gen = torch.Generator().manual_seed(seed)
    aug_gen = torch.Generator(device=device).manual_seed(seed)

    model = build_classifier(architecture=architecture).to(device)
    sample_w = balanced_sample_weights(train_ds.df) if balance else np.ones(len(train_ds))
    weights = torch.tensor(sample_w, dtype=torch.float32, device=device)
    labels_all = torch.tensor(train_ds.labels, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = {"auc": -1.0, "loss": float("inf"), "epoch": 0}
    best_state = None
    stale = 0
    history = []
    val_y = val_ds.labels
    val_w = balanced_sample_weights(val_ds.df) if balance else np.ones(len(val_ds))

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        order = torch.randperm(len(train_ds), generator=order_gen)
        for step in range(steps_per_epoch):
            idx = order[step * batch_size : (step + 1) * batch_size]
            images = train_ds.batch(idx, device, aug_gen)
            idx = idx.to(device)
            labels, w = labels_all[idx], weights[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(images)
            loss = (criterion(logits.float(), labels) * w).sum() / w.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(labels)
            n += len(labels)

        p = predict(model, val_ds, device)
        eps = 1e-6
        val_loss = float(-np.average(val_y * np.log(p + eps) + (1 - val_y) * np.log(1 - p + eps), weights=val_w))
        val_auc = float(roc_auc_score(val_y, p, sample_weight=val_w)) if len(np.unique(val_y)) > 1 else 0.0
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": val_loss, "val_auc": val_auc})
        if verbose:
            print(f"  epoch {epoch:02d} train_loss={total / max(n, 1):.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        improved = val_auc > best["auc"] + 1e-4 or (abs(val_auc - best["auc"]) <= 1e-4 and val_loss < best["loss"])
        if improved:
            best = {"auc": val_auc, "loss": val_loss, "epoch": epoch}
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best": best, "history": history}


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    out = {
        "n": int(len(y_true)),
        "n_pneumonia": int(y_true.sum()),
        "n_normal": int((1 - y_true).sum()),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else float("nan"),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": float(threshold),
    }
    return out


def bootstrap_ci(y_true, y_prob, metric: str, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(binary_metrics(y_true[idx], y_prob[idx], threshold)[metric])
    if not vals:
        return float("nan"), float("nan")
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def grouped_metrics(df: pd.DataFrame, prob_col: str = "prob", threshold: float = 0.5, with_ci: bool = True) -> dict:
    """Overall and per age-band metrics for a dataframe with `label`, `age_band` and probabilities."""
    out = {}
    for name, part in [("overall", df)] + [(b, g) for b, g in df.groupby("age_band")]:
        m = binary_metrics(part["label"], part[prob_col], threshold)
        if with_ci and len(part) > 10:
            for metric in ("roc_auc", "balanced_accuracy"):
                m[f"{metric}_ci95"] = bootstrap_ci(part["label"], part[prob_col], metric, threshold)
        out[name] = m
    return out
