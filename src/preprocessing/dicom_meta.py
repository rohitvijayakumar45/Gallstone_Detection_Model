"""DICOM metadata extraction for real-unit measurements + provenance.

Returns a plain dict (JSON-serialisable) so downstream API + reports can
attach real-world calibration information (mm/pixel) and provenance
(manufacturer, probe, frequency) instead of the fabricated defaults that
plagued V5.

If pydicom is missing or the file is not a DICOM, returns a dict with all
optional keys set to None so callers can proceed gracefully.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _first(seq: Any) -> Any:
    try:
        return seq[0]
    except Exception:
        return None


def extract_dicom_meta(source: str | Path | bytes | BytesIO) -> dict[str, Any]:
    """Extract clinically-relevant DICOM header fields.

    Returns keys:
      - pixel_spacing_mm    : float | None (from PixelSpacing or ImagerPixelSpacing)
      - pixel_spacing_row   : float | None
      - pixel_spacing_col   : float | None
      - device              : str   | None (ManufacturerModelName)
      - manufacturer        : str   | None
      - probe               : str   | None (TransducerType or TransducerData)
      - probe_frequency_mhz : float | None
      - study_uid           : str   | None
      - series_uid          : str   | None
      - sop_uid             : str   | None
      - patient_age         : str   | None (raw DICOM AS format e.g. "045Y")
      - patient_sex         : str   | None
      - body_part           : str   | None
      - is_dicom            : bool
    """
    out: dict[str, Any] = {
        "pixel_spacing_mm": None,
        "pixel_spacing_row": None,
        "pixel_spacing_col": None,
        "device": None,
        "manufacturer": None,
        "probe": None,
        "probe_frequency_mhz": None,
        "study_uid": None,
        "series_uid": None,
        "sop_uid": None,
        "patient_age": None,
        "patient_sex": None,
        "body_part": None,
        "is_dicom": False,
    }

    try:
        import pydicom
    except ImportError:
        return out

    try:
        if isinstance(source, (str, Path)):
            ds = pydicom.dcmread(str(source), stop_before_pixels=True)
        elif isinstance(source, (bytes, bytearray)):
            ds = pydicom.dcmread(BytesIO(source), stop_before_pixels=True)
        elif isinstance(source, BytesIO):
            source.seek(0)
            ds = pydicom.dcmread(source, stop_before_pixels=True)
        else:
            return out
    except Exception:
        return out

    out["is_dicom"] = True

    # PixelSpacing = [row_spacing_mm, col_spacing_mm]
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        val = getattr(ds, tag, None)
        if val is None:
            continue
        try:
            row = _to_float(val[0])
            col = _to_float(val[1]) if len(val) > 1 else row
            out["pixel_spacing_row"] = row
            out["pixel_spacing_col"] = col
            if row is not None and col is not None:
                out["pixel_spacing_mm"] = (row + col) / 2.0
            break
        except Exception:
            continue

    out["device"] = getattr(ds, "ManufacturerModelName", None) or None
    out["manufacturer"] = getattr(ds, "Manufacturer", None) or None
    out["probe"] = (
        getattr(ds, "TransducerType", None)
        or _first(getattr(ds, "TransducerData", None))
        or None
    )
    out["probe_frequency_mhz"] = _to_float(getattr(ds, "TransducerFrequency", None))
    out["study_uid"] = getattr(ds, "StudyInstanceUID", None) or None
    out["series_uid"] = getattr(ds, "SeriesInstanceUID", None) or None
    out["sop_uid"] = getattr(ds, "SOPInstanceUID", None) or None
    out["patient_age"] = getattr(ds, "PatientAge", None) or None
    out["patient_sex"] = getattr(ds, "PatientSex", None) or None
    out["body_part"] = getattr(ds, "BodyPartExamined", None) or None

    # Coerce str fields
    for k, v in list(out.items()):
        if isinstance(v, bytes):
            try:
                out[k] = v.decode("utf-8", errors="ignore")
            except Exception:
                out[k] = str(v)
    return out
