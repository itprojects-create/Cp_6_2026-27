"""FastAPI application for pneumonia detection inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_config
from app.inference import get_inference_engine
from app.preprocessing import SUPPORTED_EXTENSIONS, PreprocessingError
from app.schemas import ErrorResponse, HealthResponse, PredictionResponse, SegmentationResponse
from app.segmentation_benchmark import (
    get_benchmark_example,
    get_benchmark_run,
    get_latest_benchmark,
    get_reference_status,
    get_segmentation_models,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Research prototype only. This system is not intended to diagnose, "
    "treat, or replace evaluation by a qualified medical professional."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models at startup...")
    engine = get_inference_engine()
    status = engine.health_status()
    logger.info("Model status: %s", status)
    yield
    logger.info("Shutting down inference engine.")


app = FastAPI(
    title="Pneumonia Detection API",
    description=(
        "Research prototype for chest X-ray pneumonia detection with "
        "lung segmentation and Grad-CAM explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    max_mb = cfg["api"].get("max_upload_mb", 20)
    if file.size and file.size > max_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {max_mb}MB limit.")


@app.get("/")
async def root():
    return {
        "service": "Pneumonia Detection API",
        "version": "1.0.0",
        "disclaimer": DISCLAIMER,
        "endpoints": [
            "/health",
            "/predict",
            "/segment",
            "/research/segmentation/models",
            "/research/segmentation/references",
            "/research/segmentation/benchmark/latest",
        ],
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    engine = get_inference_engine()
    status = engine.health_status()
    return HealthResponse(
        status="ok" if status["ready"] else "degraded",
        segmentation_loaded=status["segmentation_loaded"],
        classifier_loaded=status["classifier_loaded"],
        model_type=status["model_type"],
        device=status["device"],
        ready=status["ready"],
        message=status.get("message"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)):
    _validate_upload(file)
    engine = get_inference_engine()

    if not engine.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classifier model weights not found. "
                "Train the model using: python backend/scripts/train_classifier.py"
            ),
        )

    try:
        contents = await file.read()
        result = engine.predict(contents)
        return PredictionResponse(**result)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@app.post(
    "/segment",
    response_model=SegmentationResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def segment(file: UploadFile = File(...)):
    _validate_upload(file)
    engine = get_inference_engine()

    if not engine.segmentation_loaded:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not available.",
        )

    try:
        contents = await file.read()
        result = engine.segment_only(contents)
        return SegmentationResponse(**result)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Segmentation failed")
        raise HTTPException(status_code=500, detail=f"Segmentation error: {exc}") from exc


@app.post(
    "/analyze",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def analyze(file: UploadFile = File(...)):
    _validate_upload(file)
    engine = get_inference_engine()

    try:
        contents = await file.read()
        result = engine.predict(contents)
        return PredictionResponse(**result)
    except PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze error: {exc}") from exc


@app.get("/research/dataset")
async def research_dataset():
    import pandas as pd

    manifest = Path(get_config()["paths"]["ml_manifest"])
    if not manifest.exists():
        return {"error": "Manifest not found. Run backend/scripts/build_dataset.py."}
    df = pd.read_csv(manifest)
    n_dup = int(df["duplicate_status"].eq("duplicate").sum()) if "duplicate_status" in df else 0
    unique = int(df["content_hash"].nunique()) if "content_hash" in df else int(len(df))
    age_groups = {}
    for g in ["0-18", "19-40", "41-60", "61+"]:
        part = df[df.get("age_group", pd.Series(dtype=str)) == g]
        age_groups[g] = {"pneumonia": int((part["label"] == 1).sum()), "normal": int((part["label"] == 0).sum())}
    age_bands = {}
    if "age_band" in df:
        for band, part in df.groupby("age_band"):
            age_bands[band] = {"pneumonia": int((part["label"] == 1).sum()), "normal": int((part["label"] == 0).sum())}
    splits = df["split"].value_counts().to_dict() if "split" in df else {}
    return {
        "total_images": int(len(df)),
        "pneumonia": int((df["label"] == 1).sum()),
        "normal": int((df["label"] == 0).sum()),
        "duplicates": n_dup,
        "unique_image_contents": unique,
        "age_groups": age_groups,
        "age_bands": age_bands,
        "age_source": "estimated (Riken age model) with visual pediatric/adult review",
        "splits": {k: int(v) for k, v in splits.items()},
        "patient_mapping": {"status": "pending"},
    }


@app.get("/research/domain-shift")
async def research_domain_shift():
    # Return read-only Phase 1 results from a static JSON
    import json
    stat = Path(get_config()["paths"].get("results", "results")) / "research" / "domain_shift.json"
    if stat.exists():
        with open(stat, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback to embedding known values (read-only)
    return {
        "pediatric_vs_adult": {"balanced_accuracy": 0.920},
        "adult_three_class": {"balanced_accuracy": 0.691},
        "19_40_vs_41_60": {"balanced_accuracy": 0.749},
        "19_40_vs_61_plus": {"balanced_accuracy": 0.966},
        "41_60_vs_61_plus": {"balanced_accuracy": 0.804},
        "adult_balanced": {"balanced_accuracy": 0.717},
    }


@app.get("/research/segmentation/models")
async def segmentation_models():
    return get_segmentation_models()


@app.get("/research/segmentation/references")
async def segmentation_references():
    return get_reference_status()


@app.get("/research/segmentation/benchmark/latest")
async def segmentation_benchmark_latest():
    return get_latest_benchmark()


@app.get("/research/segmentation/benchmark/runs/{run_id}")
async def segmentation_benchmark_run(run_id: str):
    try:
        result = get_benchmark_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Segmentation benchmark run not found.")
    return result


@app.get("/research/segmentation/benchmark/runs/{run_id}/examples/{model}/{sample_id}/{asset}")
async def segmentation_benchmark_example(run_id: str, model: str, sample_id: str, asset: str):
    try:
        path = get_benchmark_example(run_id, model, sample_id, asset)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Benchmark example not found.")
    return FileResponse(path, media_type="image/png")
