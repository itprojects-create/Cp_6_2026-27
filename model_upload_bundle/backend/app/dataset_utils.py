"""Shared dataset utilities for manifest building and splitting."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import get_age_group, get_config


PNEUMONIA_KEYWORDS = re.compile(
    r"\bpneumonia\b|\bpneumonitis\b|\bconsolidation\b|\binfiltrat",
    re.IGNORECASE,
)
NORMAL_KEYWORDS = re.compile(
    r"\bno acute\b|\bno focal\b|\bnormal\b|\bclear lungs\b|\bno infiltrate\b|\bno consolidation\b",
    re.IGNORECASE,
)


def load_master_dataset(path: str | Path) -> pd.DataFrame:
    """Load master CSV or Excel dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def infer_label_from_report(report: str, category: str | None = None) -> int | None:
    """Infer binary label from report text. Returns None if ambiguous."""
    if category:
        cat = category.strip().lower()
        if "pneumonia" in cat and "normal" not in cat:
            return 1
        if "normal" in cat:
            return 0

    if not isinstance(report, str) or not report.strip():
        return None

    has_pneumonia = bool(PNEUMONIA_KEYWORDS.search(report))
    has_normal = bool(NORMAL_KEYWORDS.search(report))

    if has_pneumonia and not has_normal:
        return 1
    if has_normal and not has_pneumonia:
        return 0
    return None


def select_patient_studies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply patient-level selection rules:
    - Pneumonia: earliest study with explicit pneumonia in report
    - Normal: earliest eligible normal study
    One image per patient.
    """
    cfg = get_config()
    records = []

    # Expected column mappings (flexible)
    col_map = {
        "patient_id": _find_col(df, ["patient_id", "patientid", "patient id"]),
        "image_path": _find_col(df, ["image_path", "image_index", "image index", "filename"]),
        "age": _find_col(df, ["age", "years"]),
        "sex": _find_col(df, ["sex", "gender"]),
        "study_date": _find_col(df, ["study_date", "study date"]),
        "report": _find_col(df, ["report", "finding_label", "findings"]),
        "category": _find_col(df, ["category", "label"]),
    }

    if not col_map["patient_id"]:
        raise ValueError("Master dataset must contain a patient_id column.")

    df = df.copy()
    df["_patient_id"] = df[col_map["patient_id"]].astype(str)

    if col_map["study_date"]:
        df["_study_date"] = pd.to_datetime(df[col_map["study_date"]], errors="coerce")
    else:
        df["_study_date"] = pd.NaT

    df = df.sort_values(["_patient_id", "_study_date"])

    for patient_id, group in df.groupby("_patient_id"):
        label = None
        selected = None

        # Try pneumonia first — earliest pneumonia study
        for _, row in group.iterrows():
            report = row.get(col_map["report"], "") if col_map["report"] else ""
            category = row.get(col_map["category"], "") if col_map["category"] else ""
            lbl = infer_label_from_report(str(report), str(category) if pd.notna(category) else None)
            if lbl == 1:
                label = 1
                selected = row
                break

        # If no pneumonia, find earliest normal
        if selected is None:
            for _, row in group.iterrows():
                report = row.get(col_map["report"], "") if col_map["report"] else ""
                category = row.get(col_map["category"], "") if col_map["category"] else ""
                lbl = infer_label_from_report(str(report), str(category) if pd.notna(category) else None)
                if lbl == 0:
                    label = 0
                    selected = row
                    break

        if selected is None or label is None:
            continue

        age_val = selected[col_map["age"]] if col_map["age"] else None
        try:
            age = int(float(age_val)) if pd.notna(age_val) else -1
        except (ValueError, TypeError):
            age = -1

        image_path = selected[col_map["image_path"]] if col_map["image_path"] else ""
        sex = selected[col_map["sex"]] if col_map["sex"] else "unknown"

        records.append({
            "patient_id": patient_id,
            "image_path": str(image_path),
            "age": age,
            "age_group": get_age_group(age) if age >= 0 else "unknown",
            "sex": sex,
            "label": label,
            "study_date": selected["_study_date"],
        })

    manifest = pd.DataFrame(records)
    return manifest


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def validate_unique_patients(manifest: pd.DataFrame) -> None:
    """Raise if duplicate patients exist in manifest."""
    dupes = manifest["patient_id"].duplicated().sum()
    if dupes > 0:
        raise ValueError(f"Manifest contains {dupes} duplicate patient IDs.")


def report_quota(manifest: pd.DataFrame) -> pd.DataFrame:
    """Generate quota report vs target distribution."""
    cfg = get_config()
    target = cfg["dataset"]["target_per_age_group"]
    age_groups = [g["name"] for g in cfg["age_groups"]]

    rows = []
    for ag in age_groups:
        subset = manifest[manifest["age_group"] == ag]
        pneumonia_count = (subset["label"] == 1).sum()
        normal_count = (subset["label"] == 0).sum()
        rows.append({
            "age_group": ag,
            "pneumonia": pneumonia_count,
            "normal": normal_count,
            "pneumonia_target": target,
            "normal_target": target,
            "pneumonia_shortage": max(0, target - pneumonia_count),
            "normal_shortage": max(0, target - normal_count),
            "pneumonia_excess": max(0, pneumonia_count - target),
            "normal_excess": max(0, normal_count - target),
        })

    return pd.DataFrame(rows)


def split_patients(
    manifest: pd.DataFrame,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-level stratified train/val/test split."""
    cfg = get_config()
    seed = seed or cfg["project"]["random_seed"]
    train_ratio = cfg["dataset"]["train_ratio"]
    val_ratio = cfg["dataset"]["val_ratio"]
    test_ratio = cfg["dataset"]["test_ratio"]

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Stratify by label + age_group
    manifest = manifest.copy()
    manifest["_strat"] = manifest["label"].astype(str) + "_" + manifest["age_group"]

    train_df, temp_df = train_test_split(
        manifest,
        test_size=(1 - train_ratio),
        stratify=manifest["_strat"],
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["_strat"],
        random_state=seed,
    )

    for df in (train_df, val_df, test_df):
        df.drop(columns=["_strat"], inplace=True)

    # Verify no patient overlap
    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])
    assert not (train_ids & val_ids), "Patient overlap between train and val"
    assert not (train_ids & test_ids), "Patient overlap between train and test"
    assert not (val_ids & test_ids), "Patient overlap between val and test"

    return train_df, val_df, test_df


def assign_splits(
    manifest: pd.DataFrame,
    cv_folds: int = 5,
    test_fraction: float = 0.15,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Assign leakage-safe splits.

    Rows sharing `group_id` (identical image content or the same patient) always land
    in the same partition. Stratification is on label x age band so pediatric and adult
    cases of both classes are represented in every partition.

    Adds columns:
        split    train | val | test
        cv_fold  0..cv_folds-1 for development rows, -1 for the held-out test rows
    """
    from sklearn.model_selection import StratifiedGroupKFold

    cfg = get_config()
    seed = cfg["project"]["random_seed"] if seed is None else seed
    df = manifest.copy().reset_index(drop=True)
    strata = df["label"].astype(str) + "_" + df["age_band"].astype(str)
    groups = df["group_id"].astype(str)

    n_test_splits = max(2, int(round(1.0 / test_fraction)))
    outer = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(df, strata, groups))

    df["split"] = "train"
    df["cv_fold"] = -1
    df.loc[test_idx, "split"] = "test"

    dev = df.iloc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for fold, (_, val_rel) in enumerate(inner.split(dev, strata.iloc[dev_idx], groups.iloc[dev_idx])):
        df.loc[dev.index[val_rel], "cv_fold"] = fold
    df.loc[(df["split"] != "test") & (df["cv_fold"] == 0), "split"] = "val"

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(df.loc[df["split"] == a, "group_id"]) & set(df.loc[df["split"] == b, "group_id"])
        assert not overlap, f"Group overlap between {a} and {b}: {sorted(overlap)[:5]}"
    return df
