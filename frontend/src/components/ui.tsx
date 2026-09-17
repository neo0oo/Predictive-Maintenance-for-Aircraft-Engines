import type { ReactNode } from 'react'
import { ALL_MODELS, MODEL_LABELS, type ModelKey, type Status } from '../api'
import { useSelection } from '../selection'

export function DatasetChips({ value, onChange }: { value: string; onChange: (fd: string) => void }) {
  const { datasets } = useSelection()
  return (
    <div className="selector">
      <span className="label">Dataset</span>
      <div className="chips">
        {datasets.map(d => (
          <button key={d.id} className={`chip ${d.id === value ? 'active' : ''}`} onClick={() => onChange(d.id)} title={d.label}>
            {d.id}
          </button>
        ))}
      </div>
    </div>
  )
}

export function ModelChips({ dataset, value, onChange }: { dataset: string; value: ModelKey; onChange: (m: ModelKey) => void }) {
  const { isAvailable } = useSelection()
  return (
    <div className="selector">
      <span className="label">Model</span>
      <div className="chips">
        {ALL_MODELS.map(m => {
          const ok = isAvailable(dataset, m)
          return (
            <button
              key={m}
              className={`chip ${m === value ? 'active' : ''}`}
              disabled={!ok}
              title={ok ? MODEL_LABELS[m] : `${MODEL_LABELS[m]} was not trained for ${dataset}`}
              onClick={() => ok && onChange(m)}
            >
              {MODEL_LABELS[m]}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function StatTile({ label, value, sub, small }: { label: string; value: ReactNode; sub?: ReactNode; small?: boolean }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className={`value ${small ? 'small' : ''}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

export function Legend() {
  return (
    <div className="legend">
      <span><i className="swatch healthy" />Healthy</span>
      <span><i className="swatch degrading" />Degrading</span>
      <span><i className="swatch critical" />Critical</span>
    </div>
  )
}

export function StatusBadge({ status }: { status: Status | null | undefined }) {
  if (!status) return <span className="badge neutral">—</span>
  return <span className={`badge ${status}`}>{status}</span>
}

export function Bars({ rows, best, digits = 2 }: { rows: { label: string; value: number }[]; best?: string; digits?: number }) {
  const max = Math.max(...rows.map(r => r.value), 0) || 1
  return (
    <div className="bars">
      {rows.map(r => (
        <div className="bar-row" key={r.label}>
          <span>{r.label}</span>
          <div className="bar-track">
            <div className={`bar-fill ${r.label === best ? 'best' : ''}`} style={{ width: `${(r.value / max) * 100}%` }} />
          </div>
          <span className="num">{r.value.toFixed(digits)}</span>
        </div>
      ))}
    </div>
  )
}
