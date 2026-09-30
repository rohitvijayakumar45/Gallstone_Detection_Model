interface Bin {
  lo: number;
  hi: number;
  count: number;
  mean_conf: number;
  empirical_acc: number;
}

interface Props {
  bins: Bin[];
  title?: string;
  width?: number;
  height?: number;
}

// Inline SVG reliability diagram (no plotly dep). Perfect diagonal = well
// calibrated; bars deviating from the diagonal are under/over-confident bins.
export default function ReliabilityDiagram({
  bins,
  title = "Reliability",
  width = 360,
  height = 260,
}: Props) {
  if (!bins?.length) {
    return (
      <div className="card p-4 text-[12px] text-text-3">
        {title}: no data
      </div>
    );
  }
  const pad = { top: 20, right: 12, bottom: 28, left: 32 };
  const iw = width - pad.left - pad.right;
  const ih = height - pad.top - pad.bottom;
  const xy = (v: number) => v;
  const px = (v: number) => pad.left + xy(v) * iw;
  const py = (v: number) => pad.top + (1 - xy(v)) * ih;

  const maxCount = Math.max(1, ...bins.map((b) => b.count));

  return (
    <figure className="card p-3">
      <figcaption className="label-caps mb-2">{title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}>
        {/* axes */}
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + ih}
              stroke="var(--border)" strokeWidth={1} />
        <line x1={pad.left} y1={pad.top + ih} x2={pad.left + iw} y2={pad.top + ih}
              stroke="var(--border)" strokeWidth={1} />
        {/* diagonal (perfect calibration) */}
        <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)}
              stroke="var(--text-3)" strokeDasharray="3 4" strokeWidth={1} />
        {/* bars: bin width x, height = empirical_acc */}
        {bins.map((b, i) => {
          const w = (iw / bins.length) - 2;
          const x = px(b.lo) + 1;
          const y = py(b.empirical_acc);
          const h = pad.top + ih - y;
          const opacity = 0.25 + 0.75 * (b.count / maxCount);
          return (
            <g key={i}>
              <rect x={x} y={y} width={w} height={h}
                    fill="var(--accent)" opacity={opacity} rx={1} />
              {b.count > 0 && (
                <circle cx={px(b.mean_conf)} cy={py(b.empirical_acc)} r={3}
                        fill="var(--consensus)" />
              )}
            </g>
          );
        })}
        {/* axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <text x={px(t)} y={pad.top + ih + 14} textAnchor="middle"
                  fontSize="9" fill="var(--text-3)"
                  fontFamily="JetBrains Mono, ui-monospace">
              {t.toFixed(2)}
            </text>
            <text x={pad.left - 6} y={py(t) + 3} textAnchor="end"
                  fontSize="9" fill="var(--text-3)"
                  fontFamily="JetBrains Mono, ui-monospace">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={pad.left + iw / 2} y={height - 4} textAnchor="middle"
              fontSize="10" fill="var(--text-2)">
          confidence
        </text>
        <text x={10} y={pad.top - 6} fontSize="10" fill="var(--text-2)">
          empirical acc
        </text>
      </svg>
    </figure>
  );
}
