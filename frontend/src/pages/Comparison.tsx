import { useEffect, useState } from 'react'
import { ALL_MODELS, MODEL_LABELS, api, fmt, type ComparisonResponse, type ModelKey } from '../api'
import { useSelection } from '../selection'
import { Bars, DatasetChips } from '../components/ui'

export default function Comparison() {
  const { dataset, setDataset } = useSelection()
  const [data, setData] = useState<ComparisonResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.comparison().then(setData).catch(e => setError(e.message))
  }, [])

  const entry = data?.[dataset]
  const available = entry ? (Object.keys(entry.models) as ModelKey[]) : []
  const best = available.length
    ? available.reduce((a, b) => (entry!.models[a]!.test.rmse <= entry!.models[b]!.test.rmse ? a : b))
    : null
  const cvModel = available.find(k => entry?.models[k]?.cv)
  const cv = cvModel ? entry!.models[cvModel]!.cv! : null

  return (
    <div className="page">
      <div className="selector-row" style={{ justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <div className="label" style={{ marginBottom: 6 }}>Model comparison</div>
          <div className="muted">Random Forest · XGBoost · LSTM — scored on the held-out test set; validation split shown for reference</div>
        </div>
        <DatasetChips value={dataset} onChange={setDataset} />
      </div>
      {error && <p className="error">{error}</p>}

      <div className="compare-cards">
        {ALL_MODELS.map(k => {
          const m = entry?.models[k]
          return (
            <div key={k} className={`panel compare-card ${m ? '' : 'unavailable'}`}>
              {best === k && <span className="best-tag"><span className="dot healthy" />BEST</span>}
              <div className="name">{MODEL_LABELS[k]}</div>
              {m ? (
                <div className="metric-row">
                  <div className="metric"><div className="label">Test RMSE</div><div className="value">{fmt(m.test.rmse)}</div></div>
                  <div className="metric"><div className="label">R²</div><div className="value">{fmt(m.test.r2, 3)}</div></div>
                  <div className="metric"><div className="label">NASA score</div><div className="value">{Math.round(m.test.nasa_score).toLocaleString()}</div></div>
                </div>
              ) : (
                <div className="muted">Not trained for {dataset}</div>
              )}
              {m?.validation && (
                <div className="sub muted" style={{ marginTop: 12, fontSize: 12 }}>
                  validation split: RMSE {fmt(m.validation.rmse)} · R² {fmt(m.validation.r2, 3)}
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="two-col">
        <div className="panel">
          <div className="label" style={{ marginBottom: 16 }}>Test RMSE (lower is better)</div>
          <Bars rows={available.map(k => ({ label: MODEL_LABELS[k], value: entry!.models[k]!.test.rmse }))} best={best ? MODEL_LABELS[best] : undefined} />
        </div>
        <div className="panel">
          <div className="label" style={{ marginBottom: 16 }}>R² (higher is better)</div>
          <Bars rows={available.map(k => ({ label: MODEL_LABELS[k], value: entry!.models[k]!.test.r2 }))} best={best ? MODEL_LABELS[best] : undefined} digits={3} />
        </div>
      </div>

      <div className="panel">
        <div className="label">
          5-fold cross-validation detail — {cvModel ? MODEL_LABELS[cvModel] : '—'}, {dataset} (grouped by engine unit, validation data)
        </div>
        {cv ? (
          <div className="cv-folds">
            {cv.folds.map((f, i) => (
              <div className="cv-fold" key={i}>
                <div className="label">Fold {i + 1}</div>
                <div className="value">{fmt(f.rmse)}</div>
                <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>R² {fmt(f.r2, 3)}</div>
              </div>
            ))}
            <div className="cv-fold avg">
              <div className="label">Average</div>
              <div className="value">{fmt(cv.rmse_mean)} ± {fmt(cv.rmse_std)}</div>
              <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>R² {fmt(cv.r2_mean, 3)} ± {fmt(cv.r2_std, 2)}</div>
            </div>
          </div>
        ) : (
          <div className="muted" style={{ marginTop: 12 }}>Cross-validation was only run for the tree model on each dataset.</div>
        )}
      </div>
    </div>
  )
}
