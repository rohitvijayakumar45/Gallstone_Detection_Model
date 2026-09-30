import { useDetection } from './hooks/useDetection'
import Header          from './components/Header'
import UploadZone      from './components/UploadZone'
import LoadingSkeleton from './components/LoadingSkeleton'
import ResultDashboard from './components/ResultDashboard'
import Disclaimer      from './components/Disclaimer'

export default function App() {
  const detection = useDetection()
  const { phase, preview, result, errorMsg, handleFileSelected, handleAnalyze, handleReset } = detection

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container mx-auto px-4 max-w-6xl">

        {(phase === 'idle' || phase === 'error') && (
          <div className="space-y-4">
            <UploadZone
              onFileSelected={handleFileSelected}
              onAnalyze={handleAnalyze}
              preview={preview}
            />
            {phase === 'error' && (
              <div className="max-w-5xl mx-auto card border-negative-border bg-negative-bg p-4 flex items-start gap-3">
                <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4 text-negative-text shrink-0 mt-0.5" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12" strokeLinecap="round"/><line x1="12" y1="16" x2="12.01" y2="16" strokeLinecap="round"/>
                </svg>
                <div>
                  <p className="text-negative-text font-semibold text-sm">Analysis Failed</p>
                  <p className="text-negative-text/80 text-sm mt-0.5">{errorMsg}</p>
                  <button className="btn-ghost mt-3 text-xs" onClick={handleAnalyze}>
                    Retry
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {phase === 'loading' && <LoadingSkeleton preview={preview} />}

        {phase === 'result' && result && (
          <ResultDashboard
            result={result}
            preview={preview}
            onReset={handleReset}
          />
        )}

      </main>

      <Disclaimer />
    </div>
  )
}
