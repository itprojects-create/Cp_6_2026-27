"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    segmentation_loaded: bool
    classifier_loaded: bool
    model_type: str
    device: str
    ready: bool
    message: str | None = None


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class: Normal or Pneumonia")
    pneumonia_probability: float
    normal_probability: float
    confidence: float
    model_status: str | None = Field(None, description="Model status: demo, trained, or not_trained")
    model_label: str | None = Field(None, description="Human-readable model label (e.g., DEMO / DEVELOPMENT MODEL)")
    model_type: str
    input_mode: str | None = None
    decision_threshold: float | None = None
    disclaimer: str
    original_image_base64: str | None = None
    classifier_input_base64: str | None = None
    lung_mask_base64: str | None = None
    lung_only_base64: str | None = None
    overlay_base64: str | None = None
    gradcam_base64: str | None = None
    segmentation_quality: dict[str, Any] | None = None


class SegmentationResponse(BaseModel):
    original_base64: str
    lung_mask_base64: str
    lung_only_base64: str
    overlay_base64: str
    segmentation_quality: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    detail: str
