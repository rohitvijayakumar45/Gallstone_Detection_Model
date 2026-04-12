# GallScan AI v3 — YOLO26 Gallstone Detection

AI-assisted gallstone detection using **YOLO26 instance segmentation** with three explainability outputs:
EigenCAM · Segmentation Masks · Prediction Stability Map.

---

## Why YOLO26?

| Feature                       | Benefit                                     |
| ----------------------------- | ------------------------------------------- |
| NMS-free end-to-end inference | Faster, no post-processing step             |
| No DFL                        | Broader CPU/edge hardware compatibility     |
| MuSGD optimizer               | Better convergence on small datasets        |
| ProgLoss + STAL               | Improved small-object (gallstone) detection |
| Up to 43% faster on CPU       | Practical for CPU-only deployment           |

---

## Quick Start

### 1. Create environment

```bash
cd Gallstone_V3/backend
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 2. Get dataset from Roboflow

- Go to `universe.roboflow.com`, search `gallstone ultrasound`
- Export → **Format: YOLOv8** · **Augmentation: OFF** · **Split: 80/15/5**
- Unzip into `backend/dataset/`

If your export has detection labels only (no segmentation), convert them:

```bash
# Download SAM checkpoint first:
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
pip install segment-anything
python convert_boxes_to_masks.py --split train
python convert_boxes_to_masks.py --split valid
```

### 3. Train YOLO26-seg

```bash
python train.py                        # CPU, yolo26s-seg, 100 epochs
python train.py --device 0             # GPU
python train.py --model yolo26m-seg    # larger model
python train.py --epochs 150 --batch 16
```

Best weights are automatically copied to `weights/gallstone_seg.pt`.

### 4. Start backend

```bash
uvicorn main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### 5. Start frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## Environment Variables (`backend/.env`)

| Variable               | Default                     | Description                  |
| ---------------------- | --------------------------- | ---------------------------- |
| `WEIGHTS_PATH`         | `weights/gallstone_seg.pt`  | Path to trained weights      |
| `DEVICE`               | `cpu`                       | `cpu`, `cuda`, or `mps`      |
| `CONFIDENCE_THRESHOLD` | `0.25`                      | Minimum detection confidence |
| `ALLOWED_ORIGINS`      | `http://localhost:5173,...` | CORS allowed origins         |

---

## API

### `POST /api/detect`

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

## Explainability Methods

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
