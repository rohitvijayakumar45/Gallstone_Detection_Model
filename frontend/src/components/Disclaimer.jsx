export default function Disclaimer() {
  return (
    <footer className="border-t border-border bg-surface mt-12">
      <div className="container mx-auto px-6 max-w-6xl py-5">
        <div className="flex flex-col md:flex-row items-start md:items-center gap-4">
          <div className="shrink-0 w-7 h-7 rounded-lg bg-warn-bg border border-warn-border flex items-center justify-center">
            <svg viewBox="0 0 24 24" fill="none" className="w-3.5 h-3.5 text-warn-text" stroke="currentColor" strokeWidth="2">
              <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <p className="text-ink-400 text-xs leading-relaxed max-w-4xl">
            <span className="font-semibold text-ink-600">Medical Disclaimer — </span>
            GallScan AI v3 is a research and educational decision-support tool using YOLO26 instance segmentation.
            It has not been approved or cleared as a medical device by any regulatory authority (FDA, CE, CDSCO, or equivalent).
            Results must not be used as the sole basis for any clinical decision.
            All findings must be reviewed by a qualified radiologist or physician.
          </p>
          <p className="md:ml-auto shrink-0 text-2xs text-ink-300 font-mono">
            YOLO26 · FastAPI · © {new Date().getFullYear()}
          </p>
        </div>
      </div>
    </footer>
  )
}
