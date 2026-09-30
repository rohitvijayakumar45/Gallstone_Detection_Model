"""
report.py
---------
Builds the structured clinical summary returned by the API.
Derives severity tier, clinical feature guesses, and plain-language explanation.
"""

from typing import Any


def build_report(boxes: list[dict], orig_shape: tuple) -> dict[str, Any]:
    """
    Build the complete structured result from raw detection boxes.

    Severity tiers
    --------------
    Routine           : confidence < 0.50  (or no detection)
    Attention Needed  : 0.50 ≤ confidence < 0.75
    Urgent Review     : confidence ≥ 0.75

    Returns
    -------
    dict matching the API response schema.
    """
    detected = len(boxes) > 0
    conf     = max((b["confidence"] for b in boxes), default=0.0)
    n        = len(boxes)

    # ── Severity ─────────────────────────────────────────────────────────────
    if not detected or conf < 0.50:
        severity       = "Routine"
        severity_color = "neutral"
    elif conf < 0.75:
        severity       = "Attention Needed"
        severity_color = "warning"
    else:
        severity       = "Urgent Review"
        severity_color = "critical"

    # ── Clinical feature estimates ────────────────────────────────────────────
    # These are heuristic guesses based on detection confidence and box geometry.
    # They are NOT clinically validated outputs.
    clinical_features = _estimate_clinical_features(boxes, orig_shape, detected)

    # ── Prediction label & summary ────────────────────────────────────────────
    prediction = "Gallstone Detected" if detected else "No Gallstone Detected"
    summary    = (
        f"Gallstone Detected ({round(conf * 100)}% confidence)"
        if detected
        else "No Gallstone Detected"
    )

    # ── Plain-language explanation ────────────────────────────────────────────
    explanation = _build_explanation(boxes, conf, n, detected, clinical_features)

    return {
        "prediction":        prediction,
        "confidence":        round(conf, 4),
        "boxes":             boxes,
        "summary":           summary,
        "severity":          severity,
        "severity_color":    severity_color,
        "clinical_features": clinical_features,
        "explanation":       explanation,
        "disclaimer": (
            "GallScan AI v3 is an AI-assisted decision-support tool and has not been "
            "approved or cleared as a medical device by any regulatory authority. "
            "All findings must be reviewed and confirmed by a qualified radiologist "
            "or clinician before any clinical action is taken."
        ),
    }


def _estimate_clinical_features(
    boxes: list[dict], orig_shape: tuple, detected: bool
) -> dict[str, Any]:
    H, W = orig_shape

    if not detected:
        return {
            "echogenicity":    "unknown",
            "shadowing":       None,
            "size_estimate_mm": None,
            "multiplicity":    "none",
        }

    # Multiplicity
    n = len(boxes)
    multiplicity = "single" if n == 1 else "multiple"

    # Size estimate — assume ~1 px ≈ 0.25 mm at standard US probe frequency
    # This is a rough approximation; actual scale depends on probe settings.
    px_per_mm = 0.25
    areas_mm2 = []
    for b in boxes:
        area_px  = b.get("mask_area_px") or (b["width"] * b["height"])
        areas_mm2.append(area_px * (px_per_mm ** 2))

    size_mm2 = max(areas_mm2) if areas_mm2 else None

    # Echogenicity — heuristic based on confidence
    conf = max(b["confidence"] for b in boxes)
    if conf >= 0.70:
        echogenicity = "hyperechoic"
        shadowing    = True
    elif conf >= 0.45:
        echogenicity = "mixed"
        shadowing    = None
    else:
        echogenicity = "uncertain"
        shadowing    = None

    return {
        "echogenicity":     echogenicity,
        "shadowing":        shadowing,
        "size_estimate_mm2": round(size_mm2, 1) if size_mm2 else None,
        "multiplicity":     multiplicity,
    }


def _build_explanation(
    boxes: list[dict], conf: float, n: int, detected: bool, cf: dict
) -> str:
    if not detected:
        return (
            "No gallstone was identified in this ultrasound image. "
            "The model found no hyperechoic foci with posterior acoustic shadowing "
            "in the visualised region. "
            "A negative AI result does not exclude pathology — clinical correlation "
            "and expert review remain essential."
        )

    noun       = "gallstone" if n == 1 else f"{n} gallstones"
    conf_label = "high" if conf >= 0.75 else "moderate" if conf >= 0.50 else "low"
    echo_note  = (
        f"The detected region shows {cf['echogenicity']} signal characteristics"
        + (", consistent with posterior acoustic shadowing." if cf["shadowing"] else ".")
    )
    size_note  = (
        f" Estimated lesion area is approximately {cf['size_estimate_mm2']:.0f} mm² "
        "(rough approximation — depends on probe calibration)."
        if cf.get("size_estimate_mm2") else ""
    )

    return (
        f"The model detected {noun} with {conf_label} confidence ({conf:.0%}). "
        f"{echo_note}{size_note} "
        "The segmentation mask outlines the predicted lesion boundary. "
        "The consistency map and stability map show where the model reliably fires "
        "across perturbed copies of the image. "
        "This is a decision-support output and must be confirmed by a qualified clinician."
    )
