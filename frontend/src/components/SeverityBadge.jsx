/**
 * SeverityBadge
 * Three-tier clinical urgency indicator:
 *   Routine | Attention Needed | Urgent Review
 */
export default function SeverityBadge({ severity, color }) {
  const config = {
    neutral:  { cls: 'pill-positive', dot: 'bg-positive-text', label: 'Routine'         },
    warning:  { cls: 'pill-warn',     dot: 'bg-warn-text',     label: 'Attention Needed' },
    critical: { cls: 'pill-critical', dot: 'bg-critical-text', label: 'Urgent Review'    },
  }[color] ?? { cls: 'pill-positive', dot: 'bg-positive-text', label: severity }

  return (
    <span className={config.cls}>
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot} ${color === 'critical' ? 'animate-pulse' : ''}`} />
      {config.label}
    </span>
  )
}
