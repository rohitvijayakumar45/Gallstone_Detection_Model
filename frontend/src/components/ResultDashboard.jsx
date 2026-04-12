import { useRef } from 'react'
import ClinicalSummary    from './ClinicalSummary'
import ImageViewer        from './ImageViewer'
import ExplainabilityPanel from './ExplainabilityPanel'
import DetectionTable     from './DetectionTable'
import Disclaimer         from './Disclaimer'

/**
 * ResultDashboard
 * Full result layout:
 *   ClinicalSummary (full width)
 *   ImageViewer (left) | ExplainabilityPanel (right)
 *   DetectionTable (full width)
 *   Actions bar
 */
export default function ResultDashboard({ result, preview, onReset }) {
  const reportRef = useRef(null)

  const handlePDF = async () => {
    try {
      const [{ default: jsPDF }, { default: html2canvas }] = await Promise.all([
        import('jspdf'),
        import('html2canvas'),
      ])

      const canvas  = await html2canvas(reportRef.current, {
        scale: 2,
        useCORS: true,
        backgroundColor: '#F7F5F0',
        logging: false,
      })
      const imgData = canvas.toDataURL('image/jpeg', 0.92)

      const pdf  = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
      const pw   = pdf.internal.pageSize.getWidth()
      const ph   = pdf.internal.pageSize.getHeight()
      const imgH = (canvas.height / canvas.width) * pw

      // May span multiple pages
      let yPos = 0
      while (yPos < imgH) {
        if (yPos > 0) pdf.addPage()
        pdf.addImage(imgData, 'JPEG', 0, -yPos, pw, imgH)
        yPos += ph
      }

      // Disclaimer page
      pdf.addPage()
      pdf.setFontSize(11); pdf.setTextColor(40)
      pdf.text('GallScan AI v3 — Clinical Report', 14, 20)
      pdf.setFontSize(9);  pdf.setTextColor(100)
      pdf.text(`Generated: ${new Date().toLocaleString()}`, 14, 28)
      pdf.text(`Result: ${result.prediction}`, 14, 36)
      pdf.text(`Confidence: ${Math.round(result.confidence * 100)}%`, 14, 43)
      pdf.text(`Severity: ${result.severity}`, 14, 50)
      pdf.line(14, 55, pw - 14, 55)
      pdf.setFontSize(8)
      const disclaimer = pdf.splitTextToSize(result.disclaimer, pw - 28)
      pdf.text(disclaimer, 14, 62)

      pdf.save('gallscan-v3-report.pdf')
    } catch (err) {
      console.error('PDF export failed:', err)
      alert('PDF export failed. Try downloading the JSON report instead.')
    }
  }

  const handleJSON = () => {
    const blob = new Blob(
      [JSON.stringify({ ...result, eigencam_url: '[base64 omitted]', segmask_url: '[base64 omitted]', stability_url: '[base64 omitted]' }, null, 2)],
      { type: 'application/json' }
    )
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'gallscan-v3-result.json'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="max-w-6xl mx-auto py-8 space-y-5">

      {/* Capturable report area */}
      <div ref={reportRef} className="space-y-5">

        {/* Full-width verdict */}
        <ClinicalSummary result={result} />

        {/* Two-column: image viewer + explainability */}
        <div className="grid lg:grid-cols-2 gap-5">
          <ImageViewer preview={preview} result={result} />
          <ExplainabilityPanel result={result} />
        </div>

        {/* Detection table */}
        <DetectionTable boxes={result.boxes} />

        {/* Disclaimer card */}
        <div className="flex items-start gap-3 bg-warn-bg border border-warn-border rounded-2xl p-4">
          <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4 text-warn-text shrink-0 mt-px" stroke="currentColor" strokeWidth="2">
            <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <p className="text-warn-text text-xs leading-relaxed">{result.disclaimer}</p>
        </div>
      </div>

      {/* Action bar (outside captured area) */}
      <div className="flex flex-wrap gap-3 items-center justify-between pt-2 border-t border-border">
        <button className="btn-ghost" onClick={onReset}>
          <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="2">
            <path d="M3 12a9 9 0 019-9 9.75 9.75 0 016.74 2.74L21 8M21 3v5h-5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          New Analysis
        </button>
        <div className="flex gap-2">
          <button className="btn-ghost" onClick={handleJSON}>
            <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Export JSON
          </button>
          <button className="btn-primary" onClick={handlePDF}>
            <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="14 2 14 8 20 8" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Export PDF Report
          </button>
        </div>
      </div>
    </div>
  )
}
