# GallScan AI v3 — YOLOv26 Gallstone Detection

AI-assisted gallstone detection using **YOLO26 instance segmentation** with three explainability outputs:
EigenCAM · Segmentation Masks · Prediction Stability Map.

## Why YOLOv26?

| Feature                       | Benefit                                     |
| ----------------------------- | ------------------------------------------- |
| NMS-free end-to-end inference | Faster, no post-processing step             |
| No DFL                        | Broader CPU/edge hardware compatibility     |
| MuSGD optimizer               | Better convergence on small datasets        |
| ProgLoss + STAL               | Improved small-object (gallstone) detection |
| Up to 43% faster on CPU       | Practical for CPU-only deployment           |




| Variable               | Default                     | Description                  |
| ---------------------- | --------------------------- | ---------------------------- |
| `WEIGHTS_PATH`         | `weights/gallstone_seg.pt`  | Path to trained weights      |
| `DEVICE`               | `cpu`                       | `cpu`, `cuda`, or `mps`      |
| `CONFIDENCE_THRESHOLD` | `0.25`                      | Minimum detection confidence |
| `ALLOWED_ORIGINS`      | `http://localhost:5173,...` | CORS allowed origins         |

---

## API

Upload an image (JPEG/PNG/WebP/TIFF/DICOM). Returns:

```json
{
  "prediction":    "Gallstone Detected",
  "confidence":    0.87,
  "severity":      "Urgent Review",
  "boxes": [
    { "x": 312, "y": 240, "width": 48, "height": 42,
      "confidence": 0.87, "class_name": "gallstone",
      "mask_polygon": [[x,y], ...], "mask_area_px": 1842 }
  ],
  "clinical_features": {
    "echogenicity": "hyperechoic",
    "shadowing": true,
    "size_estimate_mm2": 115.1,
    "multiplicity": "single"
  },
  "eigencam_url":   "data:image/png;base64,…",
  "segmask_url":    "data:image/png;base64,…",
  "stability_url":  "data:image/png;base64,…",
  "explanation":    "…",
  "summary":        "…",
  "disclaimer":     "…"
}
```

---

### EigenCAM

Uses the first principal component of the YOLO26 backbone (SPPF layer) feature activations.
Works without backpropagation through a classification head — stable for detection models.
From the `pytorch-grad-cam` library (`EigenCAM` class).

### Segmentation Mask

Direct pixel-level output from YOLO26-seg. Each instance rendered with a distinct
semi-transparent colour and contour outline. This is the model's primary output.

### Prediction Stability

Inference on 10 mildly perturbed copies (Gaussian noise + brightness jitter).
Pixel-wise variance of detection outputs shown as a heatmap.
Bright = model is less certain in that region.
Works with any detector without requiring model internals.

---

## Project Structure

```
Gallstone_V3/
├── backend/
│   ├── main.py                   # FastAPI app, /api/detect endpoint
│   ├── train.py                  # YOLO26-seg fine-tuning script
│   ├── convert_boxes_to_masks.py # SAM-based label conversion
│   ├── model/
│   │   ├── detector.py           # YOLO26 inference wrapper
│   │   └── explainability.py     # EigenCAM, seg mask, stability map
│   ├── utils/
│   │   ├── image_utils.py        # Image + DICOM loading
│   │   └── report.py             # Structured report builder
│   ├── weights/                  # .pt file (gitignored)
│   ├── requirements.txt
│   └── .env
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── hooks/useDetection.js
    │   └── components/
    │       ├── Header.jsx
    │       ├── UploadZone.jsx         # Drag-drop + DICOM support
    │       ├── LoadingSkeleton.jsx    # Skeleton loading state
    │       ├── ResultDashboard.jsx    # Full result + PDF export
    │       ├── ClinicalSummary.jsx    # Verdict + clinical features
    │       ├── ImageViewer.jsx        # Zoomable tabbed viewer
    │       ├── ExplainabilityPanel.jsx # 3-mode explainability
    │       ├── DetectionTable.jsx     # Sortable instance table
    │       ├── SeverityBadge.jsx      # Routine/Attention/Urgent
    │       └── Disclaimer.jsx
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

---

## Disclaimer

GallScan AI v3 is a research and educational tool. Not a certified medical device.
All outputs must be reviewed by a qualified clinician before any clinical action.
