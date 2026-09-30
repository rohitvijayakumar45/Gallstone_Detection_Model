import { useState } from 'react'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'

const MODES = [
  {
    id:    'eigencam',
    key:   'eigencam_url',
    label: 'Consistency',
    tag:   'Detection consistency map',
    desc:  'Mean detection heat across 10 mildly perturbed copies of the image. Bright (red/yellow) regions are where the model consistently fires regardless of noise — a reliable indicator of true positive detections. Derived from the same stability passes as the variance map.',
    bar:   'linear-gradient(to right, #00008b, #0000ff, #00ffff, #ffff00, #ff4444)',
    barLabel: ['Rarely detected', 'Always detected'],
  },
  {
    id:    'segmask',
    key:   'segmask_url',
    label: 'Seg Mask',
    tag:   'YOLO26 instance segmentation',
    desc:  'Primary YOLO26-seg masks (coloured regions) plus yellow-outlined regions that the stability analysis detected but the primary pass missed. Yellow "?" markers indicate areas worth clinical attention even if below the primary detection threshold.',
    bar:   null,
    barLabel: null,
  },
  {
    id:    'stability',
    key:   'stability_url',
    label: 'Stability',
    tag:   'Prediction variance map',
    desc:  'Runs inference on 10 mildly perturbed copies of the image (Gaussian noise + brightness jitter). High variance (bright, yellow) regions are where the model\'s detections change most — a proxy for epistemic uncertainty that works with any detector without needing model internals.',
    bar:   'linear-gradient(to right, #0d0221, #7b2d8b, #ff6600, #ffff00)',
    barLabel: ['Stable', 'Uncertain'],
  },
]

export default function ExplainabilityPanel({ result }) {
  const [mode, setMode]   = useState('eigencam')
  const [open, setOpen]   = useState(false)

  const active = MODES.find(m => m.id === mode)
  const src    = result[active.key]

  return (
    <div className="card p-5 space-y-4">
      <div>
        <p className="label mb-1">Explainability</p>
        <p className="text-xs text-ink-500">Three independent visualisation methods</p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-1 bg-canvas rounded-xl p-1 border border-border w-fit">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => { setMode(m.id); setOpen(false) }}
            className={[
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150',
              mode === m.id
                ? 'bg-surface shadow-card text-ink-900 border border-border'
                : 'text-ink-500 hover:text-ink-700',
            ].join(' ')}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Tag + colour bar */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <span className="bg-ink-100 text-ink-700 text-2xs font-semibold px-2.5 py-1 rounded-full">
          {active.tag}
        </span>
        {active.bar && (
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-2xs text-ink-400">{active.barLabel[0]}</span>
            <div className="w-20 h-2 rounded-full border border-border" style={{ background: active.bar }} />
            <span className="text-2xs text-ink-400">{active.barLabel[1]}</span>
          </div>
        )}
      </div>

      {/* Image */}
      {src ? (
        <TransformWrapper minScale={0.5} maxScale={10} centerOnInit>
          {({ resetTransform }) => (
            <div className="space-y-1">
              <div className="flex justify-end">
                <button onClick={() => resetTransform()} className="btn-ghost text-xs py-1 px-2.5">
                  Reset zoom
                </button>
              </div>
              <div className="rounded-xl overflow-hidden border border-border bg-canvas cursor-grab active:cursor-grabbing">
                <TransformComponent
                  wrapperStyle={{ width: '100%', maxHeight: '320px', display: 'flex' }}
                  contentStyle={{ width: '100%' }}
                >
                  <img src={src} alt={active.label} className="w-full object-contain max-h-80" draggable={false} />
                </TransformComponent>
              </div>
            </div>
          )}
        </TransformWrapper>
      ) : (
        <div className="h-48 bg-canvas rounded-xl border border-border flex items-center justify-center text-ink-400 text-sm">
          Visualisation unavailable
        </div>
      )}

      {/* "What does this mean?" accordion */}
      <div className="border border-border rounded-xl overflow-hidden">
        <button
          onClick={() => setOpen(o => !o)}
          className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-ink-700 hover:bg-canvas transition-colors"
        >
          What does this mean?
          <svg viewBox="0 0 24 24" fill="none" className={`w-4 h-4 text-ink-400 transition-transform ${open ? 'rotate-180' : ''}`} stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        {open && (
          <div className="px-4 pb-4 pt-1 text-xs text-ink-500 leading-relaxed border-t border-border bg-canvas">
            {active.desc}
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2.5 bg-warn-bg border border-warn-border rounded-xl p-3">
        <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4 text-warn-text shrink-0 mt-px" stroke="currentColor" strokeWidth="2">
          <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <p className="text-warn-text text-xs leading-relaxed">
          All explainability visualisations are for interpretability only.
          They do not constitute a clinical diagnosis and must not inform clinical decisions without expert review.
        </p>
      </div>
    </div>
  )
}
