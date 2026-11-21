import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "outputs" / "splits" / "manifest_with_splits.csv"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return TRANSFORM(image)


@torch.no_grad()
def extract_features(df: pd.DataFrame, device: torch.device, batch_size: int = 16):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    backbone = models.densenet121(weights=weights)
    backbone.classifier = torch.nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    loader = DataLoader(ImageDataset(df), batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    for images in loader:
        batches.append(backbone(images.to(device)).cpu().numpy())
    return np.vstack(batches)


def add_task_labels(df: pd.DataFrame, task: str):
    df = df.copy()
    if task == "pneumonia":
        label_map = {"Normal": 0, "Pneumonia": 1}
        df["target"] = df["diagnosis"].map(label_map)
        class_names = ["Normal", "Pneumonia"]
    else:
        label_map = {"0-18": 0}
        df["binary_age"] = df["age_group"].apply(
            lambda value: "Pediatric" if value == "0-18" else "Adult"
        )
        label_map = {"Pediatric": 0, "Adult": 1}
        df["target"] = df["binary_age"].map(label_map)
        class_names = ["Pediatric", "Adult"]
    if df["target"].isna().any():
        raise ValueError("The manifest contains an unknown label.")
    return df, class_names


def fit_classifier(x_train, y_train, x_val, y_val, seed):
    best = None
    best_score = -1.0
    best_c = None
    for c_value in (0.1, 1.0, 3.0, 10.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=seed),
        )
        model.fit(x_train, y_train)
        score = balanced_accuracy_score(y_val, model.predict(x_val))
        if score > best_score:
            best, best_score, best_c = model, score, c_value
    return best, best_c, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pneumonia", "pediatric_adult"], required=True)
    parser.add_argument("--output", default="submission/model.pkl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SPLIT_PATH}. Run src/build_manifest.py and src/split_dataset.py first."
        )

    df, class_names = add_task_labels(pd.read_csv(SPLIT_PATH), args.task)
    df = df[df["age_group"] != "unknown"].reset_index(drop=True)

    print(f"Using device: {device}")
    features = extract_features(df, device, args.batch_size)

    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("val").to_numpy()
    test_mask = df["split"].eq("test").to_numpy()
    y = df["target"].to_numpy()

    model, best_c, val_score = fit_classifier(
        features[train_mask], y[train_mask], features[val_mask], y[val_mask], args.seed
    )
    model.fit(np.vstack([features[train_mask], features[val_mask]]),
              np.concatenate([y[train_mask], y[val_mask]]))

    test_pred = model.predict(features[test_mask])
    test_score = balanced_accuracy_score(y[test_mask], test_pred)
    print(f"Best C: {best_c}")
    print(f"Validation balanced accuracy: {val_score:.4f}")
    print(f"Test balanced accuracy: {test_score:.4f}")
    print(classification_report(y[test_mask], test_pred, target_names=class_names, zero_division=0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "classifier": model,
        "task": args.task,
        "class_names": class_names,
        "seed": args.seed,
        "test_balanced_accuracy": float(test_score),
    }
    with output.open("wb") as handle:
        pickle.dump(package, handle)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in package.items() if key != "classifier"}, handle, indent=2)
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
