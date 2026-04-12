import SeverityBadge from './SeverityBadge'

/**
 * ClinicalSummary
 * Verdict banner: confidence ring, verdict pill, severity, clinical features, explanation.
 */
export default function ClinicalSummary({ result }) {
  const detected = result.prediction === 'Gallstone Detected'
  const conf     = result.confidence ?? 0
  const confPct  = Math.round(conf * 100)
  const cf       = result.clinical_features ?? {}

  const RADIUS = 32
  const CIRC   = 2 * Math.PI * RADIUS
  const offset = CIRC * (1 - conf)

  const cfRows = [
    { label: 'Echogenicity',  value: cf.echogenicity ?? '—' },
    { label: 'Shadowing',     value: cf.shadowing == null ? '—' : cf.shadowing ? 'Likely present' : 'Not evident' },
    { label: 'Size estimate', value: cf.size_estimate_mm2 ? `~${cf.size_estimate_mm2} mm²` : '—' },
    { label: 'Multiplicity',  value: cf.multiplicity ?? '—' },
  ]

  return (
    <div className="card p-6 space-y-5">

      {/* Top row: ring + verdict */}
      <div className="flex items-start gap-5">

        {/* Confidence ring */}
        <div className="relative shrink-0 flex items-center justify-center">
          <svg width="84" height="84" viewBox="0 0 84 84" className="-rotate-90">
            <circle cx="42" cy="42" r={RADIUS} fill="none" stroke="#E4E1D9" strokeWidth="6"/>
            <circle cx="42" cy="42" r={RADIUS} fill="none"
              stroke={detected ? '#B91C1C' : '#1B6535'}
              strokeWidth="6" strokeLinecap="round"
              strokeDasharray={CIRC}
              strokeDashoffset={offset}
              className="ring-anim"
            />
          </svg>
          <div className="absolute text-center leading-none">
            <p className="text-lg font-bold text-ink-900">{confPct}%</p>
            <p className="text-2xs text-ink-400">conf.</p>
          </div>
        </div>

        {/* Verdict */}
        <div className="flex-1 space-y-2 min-w-0">
          <div className="flex flex-wrap gap-2 items-center">
            <span className={detected ? 'pill-negative' : 'pill-positive'}>
              <span className={`w-1.5 h-1.5 rounded-full ${detected ? 'bg-negative-text' : 'bg-positive-text'}`}/>
              {result.prediction}
            </span>
            <SeverityBadge severity={result.severity} color={result.severity_color} />
            {result.boxes?.length > 0 && (
              <span className="text-xs text-ink-500">
                {result.boxes.length} region{result.boxes.length !== 1 ? 's' : ''} segmented
              </span>
            )}
          </div>
          <h2 className="text-xl font-bold text-ink-900 leading-snug">{result.summary}</h2>
          <p className="text-ink-500 text-sm leading-relaxed">{result.explanation}</p>
        </div>
      </div>

      {/* Clinical features table */}
      {result.boxes?.length > 0 && (
        <div className="divider pt-4">
          <p className="label mb-3">Clinical Feature Estimates</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {cfRows.map(({ label, value }) => (
              <div key={label} className="bg-canvas rounded-xl border border-border p-3">
                <p className="text-2xs text-ink-400 uppercase tracking-wide font-semibold mb-1">{label}</p>
                <p className="text-sm font-semibold text-ink-900 capitalize">{value}</p>
              </div>
            ))}
          </div>
          <p className="text-2xs text-ink-400 mt-2">
            * Feature estimates are heuristic approximations — not clinically validated outputs.
          </p>
        </div>
      )}
    </div>
  )
}
