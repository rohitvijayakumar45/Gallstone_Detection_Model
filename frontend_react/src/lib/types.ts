export interface BoxPix {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Provenance {
  raw_source?: string | null;
  calibration?: string | null;
  conformal?: string | null;
  crc_filter?: string | null;
}

export interface Detection {
  bbox: BoxPix;
  confidence: number;
  calibrated_confidence: number | null;
  shadow_score: number | null;
  conformal_bbox: BoxPix | null;
  conformal_alpha: number | null;
  aleatoric: number | null;
  epistemic: number | null;
  verdict: "detected" | "review" | "clear" | null;
  provenance: Provenance;
}

export interface PredictionResponse {
  scan_id: string;
  detections: Detection[];
  per_model_counts?: Record<string, number>;
  fused_count?: number;
  calibration_version: string | null;
  conformal_alpha: number | null;
  latency_ms: Record<string, number>;
  disclaimer: string;
}

export interface DicomMeta {
  is_dicom: boolean;
  pixel_spacing_mm: number | null;
  device: string | null;
  manufacturer: string | null;
  probe: string | null;
  probe_frequency_mhz: number | null;
  body_part: string | null;
}

export interface UploadResponse {
  scan_id: string;
  filename: string;
  width: number;
  height: number;
  preview_url: string;
  dicom_meta: DicomMeta;
}

export interface ExplainResponse {
  scan_id: string;
  yolo_cam_url: string | null;
  rfdetr_attention_url: string | null;
  agreement_map_url: string | null;
  warnings: string[];
}

export interface HealthResponse {
  status: string;
  yolo_loaded: boolean;
  rfdetr_loaded: boolean;
  calibration_ready: boolean;
  conformal_ready: boolean;
  device: string;
}

export type Layer =
  | "raw"
  | "yolo"
  | "rfdetr"
  | "consensus"
  | "conformal"
  | "heatmap";
