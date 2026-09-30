"""Serve pre-computed benchmark tables + calibration/conformal reports.

Reads from runs/final_eval/*.json produced by scripts/ so the frontend
Benchmarks page shows real numbers regenerable via the fit scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/metrics", tags=["metrics"])

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "runs" / "final_eval"

FILES = {
    "benchmarks": "comprehensive_evaluation.json",
    "shadow_tuning": "shadow_threshold_tuning.json",
    "calibration": "calibration.json",
    "conformal": "conformal.json",
}


def _read(name: str) -> dict:
    path = EVAL_DIR / FILES[name]
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"{name} report not generated yet ({path.name})")
    return json.loads(path.read_text())


@router.get("/benchmarks")
async def benchmarks() -> dict:
    return _read("benchmarks")


@router.get("/calibration")
async def calibration() -> dict:
    return _read("calibration")


@router.get("/conformal")
async def conformal() -> dict:
    return _read("conformal")


@router.get("/shadow_tuning")
async def shadow_tuning() -> dict:
    return _read("shadow_tuning")


@router.get("")
async def all_reports() -> dict:
    out = {}
    for k in FILES:
        try:
            out[k] = _read(k)
        except HTTPException:
            out[k] = None
    return out
