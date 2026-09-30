"""Structured clinical report. Real measurements, no canned prose.

Also serves a PDF variant at `/api/report/{scan_id}.pdf` rendered via
reportlab. The PDF encodes only fields backed by real measurements plus a
provenance footer.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.services.session import get_session

router = APIRouter(prefix="/report", tags=["report"])


def _px_to_mm2(area_px: float, pixel_spacing_mm: float | None) -> float | None:
    if pixel_spacing_mm is None or pixel_spacing_mm <= 0:
        return None
    return float(area_px * (pixel_spacing_mm ** 2))


@router.get("/{scan_id}")
async def get_report(scan_id: str) -> dict:
    sess = get_session(scan_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="scan not found")
    pred = sess.get("prediction")
    if pred is None:
        raise HTTPException(status_code=404, detail="not yet predicted")

    meta = sess.get("dicom_meta") or {}
    px_spacing = meta.get("pixel_spacing_mm")

    detections = pred["detections"]
    measurements: list[dict[str, Any]] = []
    for d in detections:
        b = d["bbox"]
        area_px = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        measurements.append({
            "confidence": d.get("calibrated_confidence") or d["confidence"],
            "area_px": area_px,
            "area_mm2": _px_to_mm2(area_px, px_spacing),
            "conformal_area_px": (
                (d["conformal_bbox"]["x2"] - d["conformal_bbox"]["x1"])
                * (d["conformal_bbox"]["y2"] - d["conformal_bbox"]["y1"])
                if d.get("conformal_bbox") else None
            ),
            "verdict": d.get("verdict"),
        })

    n = len(detections)
    finding = "Gallstone(s) detected" if n > 0 else "No gallstone detected"
    if n == 0:
        impression = ("No gallstone identified by the dual-detector ensemble. "
                      "Negative result does not exclude pathology; clinical review required.")
    else:
        best = max(measurements, key=lambda m: m["confidence"])
        alpha = pred.get("conformal_alpha")
        cov_note = (f" Recall guarantee ≥ {1 - alpha:.0%} at α={alpha}." if alpha else "")
        impression = (
            f"{n} candidate lesion(s) with calibrated probability up to "
            f"{best['confidence']:.0%}.{cov_note} Clinician confirmation required."
        )

    return {
        "scan_id": scan_id,
        "finding": finding,
        "impression": impression,
        "measurements": measurements,
        "provenance": {
            "calibration_version": pred.get("calibration_version"),
            "conformal_alpha": pred.get("conformal_alpha"),
            "pixel_spacing_mm": px_spacing,
            "device": meta.get("device"),
            "probe": meta.get("probe"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "disclaimer": (
            "Decision-support output. Wall thickness, CBD calibre, and "
            "adjacent-structure assessment are not produced by the model and "
            "must be independently reviewed by the reporting clinician."
        ),
    }


def _build_pdf(payload: dict[str, Any]) -> bytes:
    """Render report payload as a single-page A4 PDF via reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError as e:
        raise HTTPException(status_code=501,
                            detail=f"reportlab not installed: {e}")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=14 * mm,
        title=f"GallStone AI report {payload['scan_id']}",
        author="GallStone AI v6",
    )
    ss = getSampleStyleSheet()
    label = ParagraphStyle(
        "label", parent=ss["Normal"], fontName="Helvetica-Bold",
        fontSize=8, textColor=colors.HexColor("#8A8A83"),
        leading=10, spaceAfter=2,
    )
    body = ParagraphStyle(
        "body", parent=ss["Normal"], fontName="Helvetica",
        fontSize=10, leading=14, textColor=colors.HexColor("#1A1A1A"),
    )
    heading = ParagraphStyle(
        "h", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, spaceAfter=6, textColor=colors.HexColor("#0F3D33"),
    )
    small = ParagraphStyle(
        "s", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8, leading=10, textColor=colors.HexColor("#5A5A55"),
    )

    story: list = []
    story.append(Paragraph("GALLSTONE AI · DECISION-SUPPORT REPORT", label))
    story.append(Paragraph(f"Scan {payload['scan_id']}", heading))
    story.append(Spacer(1, 4))

    story.append(Paragraph("FINDING", label))
    story.append(Paragraph(payload["finding"], body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("IMPRESSION", label))
    story.append(Paragraph(payload["impression"], body))
    story.append(Spacer(1, 8))

    meas = payload.get("measurements") or []
    if meas:
        story.append(Paragraph("MEASUREMENTS", label))
        rows = [["#", "Prob", "Area px", "Area mm²", "Verdict"]]
        for i, m in enumerate(meas, start=1):
            rows.append([
                str(i),
                f"{(m.get('confidence') or 0) * 100:.1f}%",
                f"{m.get('area_px') or 0:.0f}",
                f"{m.get('area_mm2'):.1f}" if m.get("area_mm2") is not None else "—",
                m.get("verdict") or "—",
            ])
        t = Table(rows, colWidths=[10 * mm, 22 * mm, 24 * mm, 28 * mm, 22 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F3F0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#5A5A55")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E4E4DE")),
            ("GRID", (0, 1), (-1, -1), 0.25, colors.HexColor("#EFEFEA")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    prov = payload.get("provenance") or {}
    prov_line = " · ".join(
        f"{k}={v}" for k, v in prov.items() if v is not None
    )
    story.append(Paragraph("PROVENANCE", label))
    story.append(Paragraph(prov_line or "—", small))
    story.append(Spacer(1, 8))
    story.append(Paragraph("DISCLAIMER", label))
    story.append(Paragraph(payload.get("disclaimer") or "", small))

    doc.build(story)
    return buf.getvalue()


@router.get("/{scan_id}/pdf")
async def get_report_pdf(scan_id: str) -> Response:
    payload = await get_report(scan_id)
    pdf = _build_pdf(payload)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "content-disposition": f'inline; filename="gallstone_{scan_id}.pdf"',
        },
    )
