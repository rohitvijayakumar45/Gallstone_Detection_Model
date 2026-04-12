import { useEffect, useState } from 'react'

const STEPS = [
  { label: 'Uploading image',               ms: 600  },
  { label: 'Running YOLO26 inference',      ms: 2800 },
  { label: 'Extracting segmentation masks', ms: 800  },
  { label: 'Computing EigenCAM',            ms: 1200 },
  { label: 'Building stability map',        ms: 1400 },
  { label: 'Preparing results',             ms: 400  },
]

export default function LoadingSkeleton({ preview }) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    let elapsed = 0
    STEPS.forEach((s, i) => {
      setTimeout(() => setStep(i + 1), elapsed)
      elapsed += s.ms
    })
  }, [])

  return (
    <div className="max-w-6xl mx-auto py-10 space-y-5">

      {/* Top skeleton row */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-4">
          <div className="w-20 h-20 rounded-full bg-ink-100 skeleton shrink-0" />
          <div className="flex-1 space-y-3">
            <div className="h-4 bg-ink-100 rounded-lg w-32 skeleton" />
            <div className="h-7 bg-ink-100 rounded-lg w-64 skeleton" />
            <div className="h-4 bg-ink-100 rounded-lg w-full skeleton" />
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-5">

        {/* Image preview with scan line */}
        <div className="card p-4 relative overflow-hidden min-h-[300px]">
          <div className="relative rounded-xl overflow-hidden bg-canvas h-64">
            {preview
              ? <img src={preview} alt="" className="w-full h-full object-contain opacity-25 grayscale" />
              : <div className="w-full h-full bg-ink-100 skeleton rounded-xl" />
            }
            <div
              className="absolute left-4 right-4 h-px bg-ink-700"
              style={{ animation: 'scan 2s ease-in-out infinite', opacity: 0.45 }}
            />
          </div>
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="bg-surface/80 backdrop-blur-sm border border-border rounded-xl px-4 py-2">
              <p className="text-ink-700 text-sm font-medium">Analysing…</p>
            </div>
          </div>
        </div>

        {/* Progress steps */}
        <div className="card p-6 space-y-5">
          <div>
            <p className="label mb-1">Processing</p>
            <h2 className="text-xl font-bold text-ink-900">Running YOLO26 pipeline</h2>
            <p className="text-ink-500 text-sm mt-1">
              Inference + 3 explainability outputs are computed server-side.
            </p>
          </div>

          <div className="space-y-3">
            {STEPS.map((s, i) => {
              const done    = step > i + 1
              const active  = step === i + 1
              const pending = step < i + 1
              return (
                <div key={i} className="flex items-center gap-3">
                  <div className={[
                    'w-6 h-6 rounded-full flex items-center justify-center shrink-0 border transition-all duration-300',
                    done    ? 'bg-ink-900 border-ink-900 text-white' : '',
                    active  ? 'bg-surface border-ink-900' : '',
                    pending ? 'bg-surface border-border'  : '',
                  ].join(' ')}>
                    {done ? (
                      <svg viewBox="0 0 24 24" fill="none" className="w-3.5 h-3.5" stroke="currentColor" strokeWidth="2.5">
                        <polyline points="20 6 9 17 4 12" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    ) : active ? (
                      <div className="w-2 h-2 rounded-full bg-ink-900 animate-pulse" />
                    ) : (
                      <div className="w-1.5 h-1.5 rounded-full bg-border" />
                    )}
                  </div>
                  <span className={[
                    'text-sm flex-1 transition-colors',
                    done    ? 'text-ink-400 line-through decoration-ink-300' : '',
                    active  ? 'text-ink-900 font-medium' : '',
                    pending ? 'text-ink-300' : '',
                  ].join(' ')}>{s.label}</span>
                  {active && (
                    <div className="w-20 h-1 rounded-full overflow-hidden bg-ink-100">
                      <div className="h-full shimmer rounded-full" />
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          <p className="text-2xs text-ink-300 pt-1 border-t border-border">
            Images are processed server-side and not retained after this request.
          </p>
        </div>
      </div>

      {/* Bottom skeleton (result area placeholder) */}
      <div className="card p-6 space-y-3">
        <div className="h-4 bg-ink-100 rounded-lg w-40 skeleton" />
        <div className="grid grid-cols-3 gap-3">
          {[1,2,3].map(i => (
            <div key={i} className="h-48 bg-ink-100 rounded-xl skeleton" />
          ))}
        </div>
      </div>
    </div>
  )
}
