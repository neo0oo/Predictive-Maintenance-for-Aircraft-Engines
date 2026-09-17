import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { MODEL_LABELS, api, type EngineResponse, type ModelKey, type PredictResponse } from '../api'
import { useSelection } from '../selection'
import { ModelChips, StatusBadge } from '../components/ui'
import ThreeEngine from '../components/engine/ThreeEngine'
import NNDiagram from '../components/engine/NNDiagram'
import SensorTrend from '../components/engine/SensorTrend'
import ExplainPanel from '../components/engine/ExplainPanel'

const DEBOUNCE_MS = 120

export default function EngineDetail() {
  const { fd = 'FD001', unit: unitParam = '1' } = useParams()
  const unit = Number(unitParam)
  const { model, setModel, setDataset, dataset } = useSelection()

  const [engine, setEngine] = useState<EngineResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [cycle, setCycle] = useState(1)
  const [pred, setPred] = useState<PredictResponse | null>(null)
  const [inflight, setInflight] = useState(false)
  const [pulseKey, setPulseKey] = useState(0)
  const [sensorA, setSensorA] = useState('')
  const [sensorB, setSensorB] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => { if (dataset !== fd) setDataset(fd) }, [fd, dataset, setDataset])

  useEffect(() => {
    setEngine(null); setPred(null); setPulseKey(0)
    api.engine(fd, unit)
      .then(e => {
        setEngine(e)
        setCycle(1)
        const keys = e.sensors.map(s => s.key)
        setSensorA(keys.includes('sensor_3') ? 'sensor_3' : keys[0])
        setSensorB(keys.includes('sensor_9') ? 'sensor_9' : keys[1] ?? keys[0])
      })
      .catch(e => setError(e.message))
  }, [fd, unit])

  const activeModel: ModelKey | null = engine
    ? (engine.available_models.includes(model) ? model : engine.available_models[0])
    : null

  useEffect(() => {
    if (!engine || !activeModel) return
    const timer = setTimeout(() => {
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      setInflight(true)
      api.predictAtCycle(fd, unit, activeModel, cycle, ctrl.signal)
        .then(p => {
          if (ctrl.signal.aborted) return
          setPred(p)
          setError(null)
          setPulseKey(k => k + 1)
        })
        .catch(e => { if (e.name !== 'AbortError') setError(e.message) })
        .finally(() => { if (!ctrl.signal.aborted) setInflight(false) })
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [engine, activeModel, cycle, fd, unit])

  const topInputs = useMemo(() => pred?.explanation.items.slice(0, 6).map(i => i.label) ?? [], [pred])

  if (error && !engine) return <div className="page"><p className="error">{error}</p></div>
  if (!engine) return <div className="page"><div className="loading">loading engine {unit}…</div></div>

  return (
    <div>
      <div className="detail-header">
        <div className="left">
          <Link to="/" className="back">← FLEET</Link>
          <span className="engine-title">ENGINE #{String(unit).padStart(3, '0')}</span>
          <span className="badge neutral">{fd}</span>
          <ModelChips dataset={fd} value={activeModel ?? model} onChange={setModel} />
        </div>
        <div className="rul-readout">
          <div className="label">Predicted remaining useful life</div>
          <div className={`value status-text ${pred?.status ?? ''}`}>
            {pred ? pred.predicted_rul.toFixed(0) : '—'}<span className="unit">cycles</span>
          </div>
          {pred && <div className="muted" style={{ fontSize: 11, fontFamily: 'var(--mono)' }}>true RUL at this cycle: {pred.true_rul.toFixed(0)}</div>}
        </div>
      </div>

      <div className="scrubber-row">
        <span className="label">Cycle {cycle} / {engine.n_cycles}</span>
        <input type="range" min={1} max={engine.n_cycles} value={cycle} onChange={e => setCycle(Number(e.target.value))} />
        <StatusBadge status={pred?.status} />
      </div>
      {error && <p className="error" style={{ padding: '8px 40px 0' }}>{error}</p>}

      <div className="detail-grid">
        <div className="panel">
          <div className="label" style={{ marginBottom: 10 }}>Engine — 3D view</div>
          <div className="fill" style={{ minHeight: 300 }}>
            <ThreeEngine status={pred?.status ?? null} rul={pred?.predicted_rul ?? null} />
          </div>
          <div className="caption">Live turntable — the HPC stage recolors with predicted health as the cycle scrubber moves</div>
        </div>

        <div className="panel">
          <div className="label" style={{ marginBottom: 10 }}>Model thinking — {activeModel ? MODEL_LABELS[activeModel] : ''} forward pass</div>
          <NNDiagram
            inputs={topInputs}
            output={pred?.predicted_rul ?? null}
            pulseKey={pulseKey}
            inflight={inflight}
            modelLabel={activeModel ? MODEL_LABELS[activeModel] : ''}
          />
        </div>

        <div className="panel">
          <div className="label" style={{ marginBottom: 10 }}>Sensor trend — actual readings</div>
          {sensorA && sensorB && (
            <SensorTrend
              history={engine.history}
              sensors={engine.sensors}
              a={sensorA}
              b={sensorB}
              cycle={cycle}
              onChangeA={setSensorA}
              onChangeB={setSensorB}
            />
          )}
        </div>

        <div className="panel">
          <div className="label" style={{ marginBottom: 10 }}>Explainability — top sensors driving this prediction</div>
          <ExplainPanel explanation={pred?.explanation ?? null} />
        </div>
      </div>
    </div>
  )
}
