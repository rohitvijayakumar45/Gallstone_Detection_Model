export default function Header() {
  return (
    <header className="sticky top-0 z-50 bg-surface/90 backdrop-blur-sm border-b border-border">
      <div className="container mx-auto px-6 max-w-6xl h-14 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-ink-900 flex items-center justify-center">
            <svg viewBox="0 0 20 20" fill="none" className="w-4 h-4" stroke="white" strokeWidth="1.8">
              <circle cx="10" cy="10" r="7"/>
              <path d="M7.5 10c0-1.4 1.1-2.5 2.5-2.5s2.5 1.1 2.5 2.5" strokeLinecap="round"/>
              <path d="M10 7.5V6M10 13v1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <span className="font-semibold text-sm tracking-tight text-ink-900">GallScan AI</span>
            <span className="ml-2 text-2xs font-mono text-ink-400 bg-canvas border border-border rounded px-1.5 py-0.5">v3 · YOLO26</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden sm:flex items-center gap-1.5 text-ink-500 text-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-positive-text animate-pulse" />
            Instance segmentation · Decision-support only
          </span>
        </div>
      </div>
    </header>
  )
}
