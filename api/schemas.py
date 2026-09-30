"""Pydantic v2 schemas for API responses.

Every emitted number has a provenance-friendly companion field so the
frontend can annotate whether it comes from a raw model, a calibrated model,
or a conformal wrapper.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DicomMeta(BaseModel):
    is_dicom: bool = False
    pixel_spacing_mm: Optional[float] = None
    device: Optional[str] = None
    manufacturer: Optional[str] = None
    probe: Optional[str] = None
    probe_frequency_mhz: Optional[float] = None
    body_part: Optional[str] = None
    study_uid: Optional[str] = None
    series_uid: Optional[str] = None
    sop_uid: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None


class UploadResponse(BaseModel):
    scan_id: str
    filename: str
    width: int
    height: int
    preview_url: str
    dicom_meta: DicomMeta


class BoxPix(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    bbox: BoxPix
    confidence: float
    calibrated_confidence: Optional[float] = None
    shadow_score: Optional[float] = None
    conformal_bbox: Optional[BoxPix] = None
    conformal_alpha: Optional[float] = None
    aleatoric: Optional[float] = None
    epistemic: Optional[float] = None
    verdict: Optional[Literal["detected", "review", "clear"]] = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    scan_id: str
    detections: list[Detection]
    calibration_version: Optional[str] = None
    conformal_alpha: Optional[float] = None
    latency_ms: dict[str, float]
    disclaimer: str = (
        "Decision-support output. Findings must be confirmed by a clinician."
    )


class ExplainResponse(BaseModel):
    scan_id: str
    yolo_cam_url: Optional[str] = None
    rfdetr_attention_url: Optional[str] = None
    agreement_map_url: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    rfdetr_loaded: bool
    calibration_ready: bool
    conformal_ready: bool
    device: str


class BenchmarkTable(BaseModel):
    rows: list[dict[str, Any]]
    generated_at: Optional[str] = None
    source_file: Optional[str] = None
