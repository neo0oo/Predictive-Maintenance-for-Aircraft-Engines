import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmt, type FleetResponse } from '../api'
import { useSelection } from '../selection'
import { DatasetChips, Legend, ModelChips, StatTile } from '../components/ui'

export default function Fleet() {
  const { dataset, model, setDataset, setModel, current, setFleetSummary, error: ctxError } = useSelection()
  const [data, setData] = useState<FleetResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!current || !current.available_models.includes(model)) return
    let cancelled = false
    setLoading(true)
    api.fleet(dataset, model)
      .then(d => {
        if (cancelled) return
        setData(d)
        setError(null)
        setFleetSummary({ counts: d.status_counts, dataset, model })
      })
      .catch(e => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => { cancelled = true }
  }, [dataset, model, current, setFleetSummary])

  const m = data?.metrics

  return (
    <div className="page">
      <div className="selector-row">
        <DatasetChips value={dataset} onChange={setDataset} />
        <ModelChips dataset={dataset} value={model} onChange={setModel} />
      </div>

      {(error || ctxError) && <p className="error">{error ?? ctxError}</p>}

      <div className="tiles">
        <StatTile label="Engines in test fleet" value={data ? data.engines.length : '—'} sub={current?.label} />
        <StatTile label="Test RMSE (cycles)" value={fmt(m?.rmse)} sub="held-out test set, predicted at each engine's last cycle" />
        <StatTile label="R²" value={fmt(m?.r2, 3)} sub="true RUL capped at 125, as in training" />
        <StatTile label="NASA score" value={m ? Math.round(m.nasa_score).toLocaleString() : '—'} sub="PHM08 asymmetric penalty, summed over engines · lower is better" />
      </div>

      <div className="panel">
        <div className="panel-title">
          <span className="label">Fleet health — click an engine for detail{loading ? ' · loading' : ''}</span>
          <Legend />
        </div>
        <div className="fleet-grid">
          {data?.engines.map(e => (
            <Link
              key={e.unit}
              to={`/engine/${dataset}/${e.unit}`}
              className={`engine-cell ${e.status}`}
              title={`Engine ${e.unit} · ${e.n_cycles} cycles observed · predicted RUL ${e.predicted_rul.toFixed(1)} · true ${e.true_rul.toFixed(0)}`}
            >
              {String(e.unit).padStart(3, '0')}
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
