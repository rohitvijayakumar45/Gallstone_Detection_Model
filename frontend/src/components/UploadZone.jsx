import { useState, useRef, useCallback } from 'react'
import clsx from 'clsx'

const ACCEPTED = ['.jpg','.jpeg','.png','.webp','.tif','.tiff','.dcm']
const ACCEPTED_MIME = ['image/jpeg','image/png','image/webp','image/tiff','application/dicom','application/octet-stream']
const MAX_MB = 20

export default function UploadZone({ onFileSelected, onAnalyze, preview }) {
  const [file, setFile]   = useState(null)
  const [drag, setDrag]   = useState(false)
  const [err,  setErr]    = useState('')
  const inputRef = useRef(null)

  const validate = (f) => {
    const ext = f.name.split('.').pop()?.toLowerCase()
    if (!ACCEPTED.includes(`.${ext}`)) return `Unsupported format. Accepted: ${ACCEPTED.join(', ')}`
    if (f.size > MAX_MB * 1024 * 1024) return `File too large (${(f.size/1024/1024).toFixed(1)} MB). Max ${MAX_MB} MB.`
    return ''
  }

  const pick = useCallback((f) => {
    const e = validate(f)
    if (e) { setErr(e); return }
    setErr('')
    setFile(f)
    onFileSelected(f)
  }, [onFileSelected])

  return (
    <div className="max-w-5xl mx-auto space-y-8 pt-10 pb-4">

      {/* Heading */}
      <div>
        <p className="label mb-1">Ultrasound Analysis · YOLO26 Instance Segmentation</p>
        <h1 className="text-3xl font-bold tracking-tight text-ink-900">Gallstone Detection</h1>
        <p className="text-ink-500 text-sm mt-2 max-w-lg">
          Upload an abdominal ultrasound or DICOM file. YOLO26 will segment gallstones
          pixel-accurately and generate three explainability visualisations.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-5">

        {/* Drop zone */}
        <div
          role="button" tabIndex={0} aria-label="Upload image"
          className={clsx(
            'card flex flex-col items-center justify-center gap-5 p-10 cursor-pointer',
            'border-2 border-dashed transition-all duration-200 min-h-[280px]',
            drag ? 'border-ink-900 bg-ink-100/40 drop-active'
                 : 'border-border hover:border-ink-300 hover:bg-canvas',
          )}
          onDragEnter={(e)=>{e.preventDefault();setDrag(true)}}
          onDragOver={(e)=>{e.preventDefault();setDrag(true)}}
          onDragLeave={()=>setDrag(false)}
          onDrop={(e)=>{e.preventDefault();setDrag(false);const f=e.dataTransfer.files?.[0];if(f)pick(f)}}
          onClick={()=>inputRef.current?.click()}
          onKeyDown={(e)=>e.key==='Enter'&&inputRef.current?.click()}
        >
          <input ref={inputRef} type="file"
            accept={ACCEPTED.join(',')}
            className="hidden"
            onChange={(e)=>{const f=e.target.files?.[0];if(f)pick(f)}}
          />

          <div className={clsx(
            'w-14 h-14 rounded-2xl flex items-center justify-center border transition-colors',
            drag ? 'bg-ink-900 border-ink-900' : 'bg-canvas border-border',
          )}>
            <svg viewBox="0 0 24 24" fill="none"
              className={clsx('w-6 h-6', drag ? 'text-white' : 'text-ink-500')}
              stroke="currentColor" strokeWidth="1.8">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round"/>
            </svg>
          </div>

          <div className="text-center space-y-1">
            <p className="text-sm font-semibold text-ink-900">
              {drag ? 'Release to upload' : 'Drag & drop or click to browse'}
            </p>
            <p className="text-xs text-ink-500">
              JPEG · PNG · WebP · TIFF · DICOM &nbsp;·&nbsp; Max {MAX_MB} MB
            </p>
          </div>

          {file && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-positive-bg border border-positive-border text-positive-text text-xs font-medium">
              <svg viewBox="0 0 24 24" fill="none" className="w-3.5 h-3.5" stroke="currentColor" strokeWidth="2.5">
                <polyline points="20 6 9 17 4 12" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span className="truncate max-w-[200px]">{file.name}</span>
            </div>
          )}

          {err && <p className="text-negative-text text-xs text-center max-w-xs">{err}</p>}
        </div>

        {/* Preview + action */}
        <div className="card p-5 flex flex-col gap-4 min-h-[280px]">
          <p className="label">Preview</p>

          {preview ? (
            <div className="flex-1 flex items-center justify-center bg-canvas rounded-xl border border-border overflow-hidden">
              <img src={preview} alt="Uploaded image" className="max-h-56 w-full object-contain" />
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-border text-ink-300">
              <svg viewBox="0 0 24 24" fill="none" className="w-9 h-9 opacity-40" stroke="currentColor" strokeWidth="1.4">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <path d="M21 15l-5-5L5 21" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <p className="text-xs">No image selected</p>
            </div>
          )}

          <button className="btn-primary w-full" disabled={!file||!!err} onClick={onAnalyze}>
            <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/>
              <path d="M21 21l-4.35-4.35" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Analyze with YOLO26
          </button>

          {!file && <p className="text-center text-xs text-ink-300">Select a file to continue</p>}
        </div>
      </div>

      {/* Feature strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'YOLO26-seg',        desc: 'NMS-free instance segmentation' },
          { label: 'EigenCAM',          desc: 'Backbone feature importance map' },
          { label: 'Stability Map',     desc: 'Prediction variance uncertainty'  },
          { label: 'DICOM Support',     desc: 'Native .dcm file processing'      },
        ].map(({ label, desc }) => (
          <div key={label} className="card p-4 space-y-1">
            <p className="text-xs font-semibold text-ink-900">{label}</p>
            <p className="text-2xs text-ink-500 leading-snug">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
