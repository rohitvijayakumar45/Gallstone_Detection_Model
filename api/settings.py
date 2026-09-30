"""Pydantic-settings-driven configuration for the FastAPI backend.

All paths and thresholds are env-driven so nothing is hardcoded. Falls back
to production_models/ for weights + calibration + conformal artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
    _HAS_PSET = True
except ImportError:  # pragma: no cover
    _HAS_PSET = False
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):  # type: ignore[misc]
    # Weights
    yolo_weights: Path = ROOT / "production_models" / "yolo_best.pt"
    rfdetr_weights: Path = ROOT / "production_models" / "rfdetr_best.pth"

    # Calibration artefacts (may be missing until Phase 3 fit script runs)
    calibration_yolo: Path = ROOT / "production_models" / "calibration_yolo.pkl"
    calibration_rfdetr: Path = ROOT / "production_models" / "calibration_rfdetr.pkl"
    isotonic_fusion: Path = ROOT / "production_models" / "isotonic_fusion.pkl"
    selective_thresholds: Path = ROOT / "production_models" / "selective_thresholds.json"

    # Conformal artefacts
    conformal_qhat: Path = ROOT / "production_models" / "conformal_qhat.json"
    conformal_crc: Path = ROOT / "production_models" / "conformal_crc.json"

    # Inference defaults
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    tta_scales: tuple[int, ...] = (640, 800, 1024)
    shadow_threshold: float = 0.7
    ensemble_weights: tuple[float, ...] = (0.5, 0.5)

    # Runtime
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    max_workers: int = 2
    max_upload_mb: int = 20

    # CORS
    cors_origins: tuple[str, ...] = ("*",)

    if _HAS_PSET:
        model_config = SettingsConfigDict(env_prefix="GALLSTONE_", env_file=".env",
                                          case_sensitive=False)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def summary(self) -> dict[str, Any]:
        return {
            "yolo_weights": str(self.yolo_weights),
            "rfdetr_weights": str(self.rfdetr_weights),
            "calibration_ready": all(p.exists() for p in (self.calibration_yolo,
                                                          self.calibration_rfdetr,
                                                          self.isotonic_fusion)),
            "conformal_ready": self.conformal_qhat.exists(),
            "device": self.resolve_device(),
            "shadow_threshold": self.shadow_threshold,
        }


settings = Settings()
