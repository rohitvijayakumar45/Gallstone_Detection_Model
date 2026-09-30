import type {
  ExplainResponse,
  HealthResponse,
  PredictionResponse,
  UploadResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = String(body.detail);
    } catch {
      // ignore
    }
    throw new Error(`API ${path}: ${msg}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => jsonFetch<HealthResponse>("/health"),

  upload: async (file: File): Promise<UploadResponse> => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`upload failed: ${res.status}`);
    return res.json() as Promise<UploadResponse>;
  },

  predict: (scanId: string) =>
    jsonFetch<PredictionResponse>(`/predict/${scanId}`, { method: "POST" }),

  getPrediction: (scanId: string) =>
    jsonFetch<PredictionResponse>(`/predict/${scanId}`),

  explain: (scanId: string) =>
    jsonFetch<ExplainResponse>(`/explain/${scanId}`),

  getReport: (scanId: string) => jsonFetch<any>(`/report/${scanId}`),

  benchmarks: () => jsonFetch<any>("/metrics/benchmarks"),
  calibration: () => jsonFetch<any>("/metrics/calibration"),
  conformal: () => jsonFetch<any>("/metrics/conformal"),
};
