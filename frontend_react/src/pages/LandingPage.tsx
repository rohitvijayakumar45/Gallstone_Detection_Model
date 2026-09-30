import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export default function LandingPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });

  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      <section className="grid grid-cols-12 gap-10 items-end">
        <div className="col-span-12 lg:col-span-7">
          <p className="label-caps mb-4">Clinical decision-support · abdominal ultrasound</p>
          <h1 className="font-display text-4xl md:text-5xl leading-[1.05] tracking-tight text-text">
            Dual-detector gallstone analysis with
            <span className="text-consensus"> calibrated confidence </span>
            and a
            <span className="text-accent"> conformal recall guarantee</span>.
          </h1>
          <p className="mt-6 text-text-2 text-[15px] leading-relaxed max-w-[54ch]">
            YOLOv26L and RF-DETR run in parallel over a shadow-verified fusion
            path. Every result carries a temperature-scaled probability, a
            distribution-free coverage bound, and a provenance trail for
            radiologist review.
          </p>
          <div className="mt-8 flex items-center gap-3">
            <Link
              to="/study"
              className="h-10 px-5 rounded-md inline-flex items-center text-[13px] font-medium bg-accent text-white hover:opacity-90 transition-opacity duration-150 ease-clinical shadow-1"
            >
              Open workstation
            </Link>
            <Link
              to="/benchmarks"
              className="h-10 px-5 rounded-md inline-flex items-center text-[13px] font-medium border border-border bg-surface hover:bg-surface-2 transition-colors duration-150 ease-clinical"
            >
              View benchmarks
            </Link>
          </div>
        </div>
        <div className="col-span-12 lg:col-span-5">
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <p className="label-caps">System status</p>
              <span
                aria-hidden
                className="w-2 h-2 rounded-full"
                style={{ background: health.data?.status === "healthy" ? "var(--ok)" : "var(--warn)" }}
              />
            </div>
            <dl className="grid grid-cols-2 gap-y-3 text-[13px]">
              <dt className="text-text-3">YOLOv26L</dt>
              <dd className="num text-right">{health.data?.yolo_loaded ? "loaded" : "…"}</dd>
              <dt className="text-text-3">RF-DETR</dt>
              <dd className="num text-right">{health.data?.rfdetr_loaded ? "loaded" : "…"}</dd>
              <dt className="text-text-3">Calibration</dt>
              <dd className="num text-right">v1</dd>
              <dt className="text-text-3">Conformal α</dt>
              <dd className="num text-right">0.05</dd>
            </dl>
          </div>
        </div>
      </section>

      <section className="mt-20">
        <p className="label-caps mb-4">Performance</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { k: "Val mAP@50 (YOLO)", v: "0.9598", note: "Optuna best of 19 trials" },
            { k: "Val mAP@50 (RF-DETR)", v: "0.9606", note: "Optuna best of 140 trials" },
            { k: "Fused test (pending)", v: "—", note: "regenerate via scripts/" },
            { k: "Conformal coverage", v: "—", note: "empirical vs α=0.05" },
          ].map((it) => (
            <div key={it.k} className="card p-4">
              <p className="label-caps mb-2">{it.k}</p>
              <p className="num text-2xl">{it.v}</p>
              <p className="text-[12px] text-text-3 mt-1">{it.note}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-20 text-[13px] text-text-2 max-w-3xl">
        <p className="label-caps mb-2">Disclaimer</p>
        <p>
          This system is an investigational decision-support tool. Findings are
          not a substitute for radiologist interpretation. Wall thickness, CBD
          calibre, and adjacent-structure assessment are not produced by the
          model and must be independently reviewed by the reporting clinician.
        </p>
      </section>
    </div>
  );
}
