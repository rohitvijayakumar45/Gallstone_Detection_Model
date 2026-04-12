/**
 * useDetection
 * Custom hook that manages the full detection state machine.
 *
 * States: idle → loading → result | error
 */

import { useState, useCallback } from 'react'

export function useDetection() {
  const [phase,    setPhase]    = useState('idle')     // idle | loading | result | error
  const [preview,  setPreview]  = useState(null)       // object URL for the preview image
  const [result,   setResult]   = useState(null)       // API JSON response
  const [errorMsg, setErrorMsg] = useState('')
  const [file,     setFile]     = useState(null)       // the actual File object

  const handleFileSelected = useCallback((f) => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(URL.createObjectURL(f))
    setFile(f)
    setResult(null)
    setPhase('idle')
    setErrorMsg('')
  }, [preview])

  const handleAnalyze = useCallback(async (f) => {
    if (!f) return
    setPhase('loading')
    setResult(null)
    setErrorMsg('')

    const form = new FormData()
    form.append('file', f)

    try {
      const res = await fetch('/api/detect', { method: 'POST', body: form })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Server error (${res.status})`)
      }

      setResult(await res.json())
      setPhase('result')
    } catch (err) {
      setErrorMsg(err.message || 'Unknown error. Please try again.')
      setPhase('error')
    }
  }, [])

  const handleReset = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setFile(null)
    setResult(null)
    setErrorMsg('')
    setPhase('idle')
  }, [preview])

  return {
    phase, preview, result, errorMsg, file,
    handleFileSelected,
    handleAnalyze: () => handleAnalyze(file),
    handleReset,
  }
}
