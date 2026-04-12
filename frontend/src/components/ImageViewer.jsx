import { useState } from 'react'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'

/**
 * ImageViewer
 * Tabbed, zoomable/pannable image panel.
 * Tabs: Original | Segmentation Mask | EigenCAM | Stability Map
 */
export default function ImageViewer({ preview, result }) {
  const [tab, setTab] = useState('original')

  const tabs = [
    { id: 'original',  label: 'Original',    src: preview,               note: 'Uploaded image — no annotations.' },
    { id: 'segmask',   label: 'Seg Mask',     src: result.segmask_url,    note: 'Per-instance segmentation masks predicted by YOLO26.' },
    { id: 'eigencam',  label: 'EigenCAM',     src: result.eigencam_url,   note: 'Principal-component feature importance from the YOLO26 backbone.' },
    { id: 'stability', label: 'Stability',    src: result.stability_url,  note: 'Detection variance across perturbed inputs — bright = uncertain.' },
  ].filter(t => t.src)

  const active = tabs.find(t => t.id === tab) ?? tabs[0]

  const handleDownload = () => {
    if (!active?.src) return
    const a = document.createElement('a')
    a.href     = active.src
    a.download = `gallscan-${active.id}.png`
    a.click()
  }

  return (
    <div className="card p-5 space-y-4">

      {/* Tab bar */}
      <div className="flex gap-1 bg-canvas rounded-xl p-1 w-fit border border-border overflow-x-auto">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={[
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 whitespace-nowrap',
              tab === t.id
                ? 'bg-surface shadow-card text-ink-900 border border-border'
                : 'text-ink-500 hover:text-ink-700',
            ].join(' ')}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Zoomable image */}
      {active?.src && (
        <TransformWrapper minScale={0.5} maxScale={10} centerOnInit>
          {({ resetTransform }) => (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs text-ink-400">{active.note}</p>
                <div className="flex gap-2">
                  <button onClick={() => resetTransform()} className="btn-ghost text-xs py-1 px-2.5">
                    Reset zoom
                  </button>
                  <button onClick={handleDownload} className="btn-ghost text-xs py-1 px-2.5">
                    Download
                  </button>
                </div>
              </div>
              <div className="rounded-xl overflow-hidden border border-border bg-canvas cursor-grab active:cursor-grabbing">
                <TransformComponent
                  wrapperStyle={{ width: '100%', maxHeight: '420px', display: 'flex' }}
                  contentStyle={{ width: '100%' }}
                >
                  <img
                    src={active.src}
                    alt={active.label}
                    className="w-full object-contain max-h-[420px]"
                    draggable={false}
                  />
                </TransformComponent>
              </div>
              <p className="text-2xs text-ink-300 text-center">
                Scroll to zoom · drag to pan · pinch on mobile
              </p>
            </div>
          )}
        </TransformWrapper>
      )}
    </div>
  )
}
