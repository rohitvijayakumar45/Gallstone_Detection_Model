/**
 * DetectionTable
 * Per-instance detection detail table with sortable columns.
 */
import { useState } from 'react'

const COLS = [
  { key: 'rank',       label: '#',           sortable: false },
  { key: 'class_name', label: 'Class',        sortable: false },
  { key: 'confidence', label: 'Confidence',   sortable: true  },
  { key: 'mask_area',  label: 'Mask Area',    sortable: true  },
  { key: 'centre',     label: 'Centre (px)',  sortable: false },
  { key: 'size',       label: 'Box (px)',     sortable: false },
]

export default function DetectionTable({ boxes }) {
  const [sortKey, setSortKey] = useState('confidence')
  const [sortDir, setSortDir] = useState('desc')

  if (!boxes?.length) return null

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const sorted = [...boxes].sort((a, b) => {
    const av = sortKey === 'mask_area' ? a.mask_area_px : a.confidence
    const bv = sortKey === 'mask_area' ? b.mask_area_px : b.confidence
    return sortDir === 'asc' ? av - bv : bv - av
  })

  return (
    <div className="card p-5 sm:p-6">
      <p className="label mb-4">Detection Detail · {boxes.length} instance{boxes.length !== 1 ? 's' : ''}</p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left">
              {COLS.map(col => (
                <th
                  key={col.key}
                  className={['pb-3 pr-5 last:pr-0 label', col.sortable ? 'cursor-pointer hover:text-ink-700 select-none' : ''].join(' ')}
                  onClick={col.sortable ? () => toggleSort(col.key) : undefined}
                >
                  <span className="flex items-center gap-1">
                    {col.label}
                    {col.sortable && (
                      <svg viewBox="0 0 24 24" fill="none" className={`w-3 h-3 transition-transform ${sortKey === col.key && sortDir === 'asc' ? 'rotate-180' : ''}`} stroke="currentColor" strokeWidth="2">
                        <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {sorted.map((box, i) => (
              <tr key={i} className="hover:bg-canvas transition-colors">
                <td className="py-3 pr-5 text-ink-400 font-mono text-xs">{i + 1}</td>
                <td className="py-3 pr-5">
                  <span className="pill-negative py-0.5 text-2xs">{box.class_name}</span>
                </td>
                <td className="py-3 pr-5">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-ink-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-negative-text transition-all duration-500"
                        style={{ width: `${box.confidence * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs text-ink-700 font-medium">
                      {(box.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="py-3 pr-5 font-mono text-xs text-ink-500">
                  {box.mask_area_px ? `${box.mask_area_px.toLocaleString()} px` : '—'}
                </td>
                <td className="py-3 pr-5 font-mono text-xs text-ink-500">
                  ({Math.round(box.x)}, {Math.round(box.y)})
                </td>
                <td className="py-3 font-mono text-xs text-ink-500">
                  {Math.round(box.width)} × {Math.round(box.height)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
