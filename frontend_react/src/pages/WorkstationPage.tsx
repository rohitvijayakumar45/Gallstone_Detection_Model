import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import * as Tabs from "@radix-ui/react-tabs";
import { api } from "@/lib/api";
import type {
  Detection,
  ExplainResponse,
  Layer,
  PredictionResponse,
  UploadResponse,
} from "@/lib/types";
import ScanCanvas from "@/components/ScanCanvas";
import UploadDropzone from "@/components/UploadDropzone";
import LayerToggles from "@/components/LayerToggles";
import ConfidenceGauge from "@/components/ConfidenceGauge";
import Provenance from "@/components/Provenance";

const initialLayers: Record<Layer, boolean> = {
  raw: true,
  yolo: true,
  rfdetr: true,
  consensus: true,
  conformal: true,
  heatmap: true,
};

export default function WorkstationPage() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [layers, setLayers] = useState<Record<Layer, boolean>>(initialLayers);
  const [tab, setTab] = useState<"findings" | "uncertainty" | "xai" | "report">("findings");

  const predict = useMutation({
    mutationFn: (scanId: string) => api.predict(scanId),
  });
  const explain = useQuery({
    queryKey: ["explain", upload?.scan_id, predict.isSuccess],
    queryFn: () => (upload ? api.explain(upload.scan_id) : Promise.reject("no scan")),
    // Fire after a successful predict so the heatmap layer is available even
    // before the user opens the XAI tab.
    enabled: !!upload && predict.isSuccess,
    staleTime: 5 * 60_000,
  });
  const report = useQuery({
    queryKey: ["report", upload?.scan_id, tab],
    queryFn: () => (upload ? api.getReport(upload.scan_id) : Promise.reject("no scan")),
    enabled: !!upload && tab === "report" && predict.isSuccess,
  });

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)
        return;
      const map: Record<string, Layer | null> = {
        "0": "raw",
        y: "yolo",
        r: "rfdetr",
        c: "consensus",
        x: "conformal",
        h: "heatmap",
      };
      const k = e.key.toLowerCase();
      if (map[k]) {
        setLayers((s) => ({ ...s, [map[k]!]: !s[map[k]!] }));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const prediction: PredictionResponse | undefined = predict.data;
  const detections: Detection[] = prediction?.detections ?? [];
  const bestConf = useMemo(() => {
    if (!detections.length) return { raw: 0, cal: null as number | null, alpha: null, verdict: null };
    const best = detections.reduce((a, b) =>
      (b.calibrated_confidence ?? b.confidence) > (a.calibrated_confidence ?? a.confidence) ? b : a,
    );
    return {
      raw: best.confidence,
      cal: best.calibrated_confidence,
      alpha: best.conformal_alpha,
      verdict: best.verdict,
    };
  }, [detections]);

  return (
    <div className="h-[calc(100vh-56px-40px)] grid grid-cols-[260px_1fr_440px] grid-rows-[1fr_60px]">
      <aside className="row-span-2 border-r border-border bg-surface-2/50 p-4 overflow-y-auto flex flex-col gap-4">
        <UploadDropzone onUploaded={(u) => { setUpload(u); predict.reset(); }} />
        {upload ? (
          <div className="card p-3">
            <p className="label-caps mb-2">Scan</p>
            <p className="text-[13px] break-all">{upload.filename}</p>
            <p className="num text-[12px] text-text-3 mt-1">
              {upload.width}×{upload.height}
            </p>
            {upload.dicom_meta?.is_dicom ? (
              <dl className="mt-3 grid grid-cols-2 gap-y-1 text-[12px]">
                <dt className="text-text-3">Device</dt>
                <dd className="text-right">{upload.dicom_meta.device ?? "—"}</dd>
                <dt className="text-text-3">Probe</dt>
                <dd className="text-right">{upload.dicom_meta.probe ?? "—"}</dd>
                <dt className="text-text-3">Spacing</dt>
                <dd className="text-right num">
                  {upload.dicom_meta.pixel_spacing_mm != null
                    ? `${upload.dicom_meta.pixel_spacing_mm.toFixed(3)} mm/px`
                    : "—"}
                </dd>
              </dl>
            ) : null}
            <button
              type="button"
              disabled={predict.isPending}
              onClick={() => predict.mutate(upload.scan_id)}
              className="mt-3 w-full h-9 rounded-md bg-accent text-white text-[13px] font-medium disabled:opacity-50 hover:opacity-90 transition-opacity"
            >
              {predict.isPending ? "Analysing…" : predict.isSuccess ? "Re-analyse" : "Analyse"}
            </button>
            {predict.isError ? (
              <p className="text-alarm text-[12px] mt-2">{String(predict.error)}</p>
            ) : null}
          </div>
        ) : null}
      </aside>

      <section className="border-b border-border bg-surface flex flex-col">
        <div className="h-10 border-b border-border px-4 flex items-center justify-between gap-4">
          <LayerToggles
            value={layers}
            onChange={setLayers}
            heatmapAvailable={!!explain.data?.agreement_map_url}
          />
          <div className="text-[12px] text-text-3 num flex items-center gap-4">
            {prediction ? (
              <>
                <span>
                  YOLO{" "}
                  <span style={{ color: (prediction.per_model_counts?.yolo ?? 0) > 0 ? "var(--accent)" : "var(--alarm)" }}>
                    {prediction.per_model_counts?.yolo ?? "—"}
                  </span>
                </span>
                <span>
                  RF-DETR{" "}
                  <span style={{ color: (prediction.per_model_counts?.rfdetr ?? 0) > 0 ? "var(--accent-2)" : "var(--alarm)" }}>
                    {prediction.per_model_counts?.rfdetr ?? "—"}
                  </span>
                </span>
                <span>
                  Fused{" "}
                  <span style={{ color: (prediction.fused_count ?? 0) > 0 ? "var(--consensus)" : "var(--text-3)" }}>
                    {prediction.fused_count ?? "—"}
                  </span>
                </span>
                <span>· fusion {prediction.latency_ms.fusion} ms</span>
              </>
            ) : (
              "no prediction"
            )}
          </div>
        </div>
        <div className="flex-1 min-h-0">
          {upload ? (
            <ScanCanvas
              previewUrl={upload.preview_url}
              detections={detections}
              layers={layers}
              heatmapUrl={explain.data?.agreement_map_url ?? null}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-text-3">
              <p className="text-[13px]">Upload a scan to begin.</p>
            </div>
          )}
        </div>
      </section>

      <aside className="row-span-2 border-l border-border bg-surface overflow-y-auto">
        <Tabs.Root value={tab} onValueChange={(v) => setTab(v as any)} className="h-full flex flex-col">
          <Tabs.List className="flex border-b border-border px-2">
            {[
              { k: "findings", label: "Findings" },
              { k: "uncertainty", label: "Uncertainty" },
              { k: "xai", label: "XAI" },
              { k: "report", label: "Report" },
            ].map((t) => (
              <Tabs.Trigger
                key={t.k}
                value={t.k}
                className="h-10 px-3 text-[12px] label-caps text-text-3 data-[state=active]:text-text data-[state=active]:border-b-2 data-[state=active]:border-accent"
              >
                {t.label}
              </Tabs.Trigger>
            ))}
          </Tabs.List>

          <Tabs.Content value="findings" className="p-4 flex-1 overflow-y-auto">
            {detections.length === 0 ? (
              <p className="text-text-3 text-[13px]">
                {prediction ? "No detections above threshold." : "Awaiting analysis."}
              </p>
            ) : (
              <ul className="space-y-3">
                {detections.map((d, i) => (
                  <li key={i} className="card p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="label-caps">Detection {i + 1}</span>
                      <span className="num text-[12px]" style={{ color: verdictColor(d.verdict) }}>
                        {d.verdict ?? "—"}
                      </span>
                    </div>
                    <dl className="grid grid-cols-2 gap-y-1 text-[13px]">
                      <dt className="text-text-3">Raw</dt>
                      <dd className="text-right num">
                        <Provenance source={d.provenance.raw_source ?? "raw"}>
                          {(d.confidence * 100).toFixed(1)}%
                        </Provenance>
                      </dd>
                      <dt className="text-text-3">Calibrated</dt>
                      <dd className="text-right num">
                        <Provenance source={d.provenance.calibration ?? "—"}>
                          {d.calibrated_confidence != null
                            ? `${(d.calibrated_confidence * 100).toFixed(1)}%`
                            : "—"}
                        </Provenance>
                      </dd>
                      <dt className="text-text-3">Conformal α</dt>
                      <dd className="text-right num">
                        <Provenance source={d.provenance.conformal ?? "—"}>
                          {d.conformal_alpha != null ? d.conformal_alpha.toFixed(2) : "—"}
                        </Provenance>
                      </dd>
                    </dl>
                  </li>
                ))}
              </ul>
            )}
          </Tabs.Content>

          <Tabs.Content value="uncertainty" className="p-4 flex-1 overflow-y-auto">
            <p className="text-text-3 text-[13px]">
              MC-dropout aleatoric measure not yet wired at inference — enable via
              <code className="num mx-1">/api/predict?mc=1</code>.
            </p>
          </Tabs.Content>

          <Tabs.Content value="xai" className="p-4 flex-1 overflow-y-auto space-y-3">
            {upload && !explain.data && !explain.isFetching ? (
              <p className="text-text-3 text-[13px]">Select this tab to load XAI.</p>
            ) : null}
            {explain.isFetching ? <p className="text-text-3 text-[13px]">Rendering…</p> : null}
            {explain.data ? <XAIPanel data={explain.data} /> : null}
          </Tabs.Content>

          <Tabs.Content value="report" className="p-4 flex-1 overflow-y-auto space-y-3">
            {!prediction ? (
              <p className="text-text-3 text-[13px]">Run analysis first.</p>
            ) : report.isFetching ? (
              <p className="text-text-3 text-[13px]">Loading…</p>
            ) : report.data ? (
              <ReportPanel data={report.data} />
            ) : null}
          </Tabs.Content>
        </Tabs.Root>
      </aside>

      <div className="border-t border-border px-4 flex items-center gap-4 text-[12px]">
        {prediction ? (
          <ConfidenceGauge
            raw={bestConf.raw}
            calibrated={bestConf.cal}
            conformalAlpha={bestConf.alpha as number | null}
            verdict={bestConf.verdict}
          />
        ) : (
          <span className="text-text-3">No analysis yet.</span>
        )}
      </div>
    </div>
  );
}

function verdictColor(v: Detection["verdict"]) {
  if (v === "detected") return "var(--ok)";
  if (v === "review") return "var(--warn)";
  if (v === "clear") return "var(--info)";
  return "var(--text-3)";
}

function XAIPanel({ data }: { data: ExplainResponse }) {
  return (
    <div className="space-y-3">
      {(["yolo_cam_url", "rfdetr_attention_url", "agreement_map_url"] as const).map((k) => (
        <figure key={k}>
          <figcaption className="label-caps mb-1">
            {k === "yolo_cam_url"
              ? "YOLO EigenCAM"
              : k === "rfdetr_attention_url"
                ? "RF-DETR cross-attention"
                : "Attention agreement"}
          </figcaption>
          {data[k] ? (
            <img src={data[k]!} alt={k} className="w-full rounded-md border border-border" />
          ) : (
            <p className="text-text-3 text-[12px]">—</p>
          )}
        </figure>
      ))}
      {data.warnings.length ? (
        <div className="text-[12px] text-warn">
          <p className="label-caps mb-1">Warnings</p>
          <ul className="list-disc pl-4 space-y-0.5">
            {data.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function ReportPanel({ data }: { data: any }) {
  return (
    <div className="text-[13px] space-y-3">
      <div>
        <p className="label-caps mb-1">Finding</p>
        <p>{data.finding}</p>
      </div>
      <div>
        <p className="label-caps mb-1">Impression</p>
        <p className="text-text-2 leading-relaxed">{data.impression}</p>
      </div>
      {data.measurements?.length ? (
        <div>
          <p className="label-caps mb-1">Measurements</p>
          <table className="w-full text-[12px] border border-border">
            <thead className="bg-surface-2 text-left">
              <tr>
                <th className="px-2 py-1">#</th>
                <th className="px-2 py-1 text-right">Prob</th>
                <th className="px-2 py-1 text-right">Area px</th>
                <th className="px-2 py-1 text-right">Area mm²</th>
                <th className="px-2 py-1 text-right">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {data.measurements.map((m: any, i: number) => (
                <tr key={i} className="border-t border-divider">
                  <td className="px-2 py-1">{i + 1}</td>
                  <td className="px-2 py-1 text-right num">
                    {(m.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="px-2 py-1 text-right num">{m.area_px?.toFixed(0) ?? "—"}</td>
                  <td className="px-2 py-1 text-right num">
                    {m.area_mm2 != null ? m.area_mm2.toFixed(1) : "—"}
                  </td>
                  <td className="px-2 py-1 text-right num">{m.verdict ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <p className="text-text-3 text-[11px] leading-relaxed">{data.disclaimer}</p>
    </div>
  );
}
