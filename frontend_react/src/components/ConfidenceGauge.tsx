import Provenance from "./Provenance";

interface Props {
  raw: number;
  calibrated: number | null;
  conformalAlpha: number | null;
  verdict?: "detected" | "review" | "clear" | null;
}

const verdictColor: Record<string, string> = {
  detected: "var(--ok)",
  review: "var(--warn)",
  clear: "var(--info)",
};

function fmtPct(v: number | null): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

export default function ConfidenceGauge({
  raw,
  calibrated,
  conformalAlpha,
  verdict,
}: Props) {
  const dot = verdict ? verdictColor[verdict] : "var(--text-3)";
  return (
    <div className="flex items-center gap-6">
      <div>
        <div className="label-caps mb-1">Raw</div>
        <Provenance source="raw wbf+shadow" detail="Fused confidence before calibration.">
          <span className="num text-lg">{fmtPct(raw)}</span>
        </Provenance>
      </div>
      <div>
        <div className="label-caps mb-1">Calibrated</div>
        <Provenance
          source="isotonic_fusion v1"
          detail="Post-fusion isotonic regression on held-out val fold."
        >
          <span className="num text-lg">{fmtPct(calibrated)}</span>
        </Provenance>
      </div>
      <div>
        <div className="label-caps mb-1">Coverage</div>
        <Provenance
          source="split_conformal_bonferroni"
          detail={
            conformalAlpha != null
              ? `Marginal recall guarantee ≥ ${((1 - conformalAlpha) * 100).toFixed(0)}%`
              : "Conformal artefacts not loaded."
          }
        >
          <span className="num text-lg">
            {conformalAlpha != null ? `≥${((1 - conformalAlpha) * 100).toFixed(0)}%` : "—"}
          </span>
        </Provenance>
      </div>
      {verdict ? (
        <div className="flex items-center gap-2 ml-auto">
          <span aria-hidden className="w-2 h-2 rounded-full" style={{ background: dot }} />
          <span className="label-caps" style={{ color: dot }}>
            {verdict}
          </span>
        </div>
      ) : null}
    </div>
  );
}
