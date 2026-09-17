import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { MODEL_LABELS, api, type CsvResponse, type ManualResponse, type ModelKey, type RangeStats, type SensorRangesResponse } from '../api'
import { useSelection } from '../selection'
import { DatasetChips, ModelChips, StatusBadge } from '../components/ui'
import ExplainPanel from '../components/engine/ExplainPanel'

const DEBOUNCE_MS = 250

function Slider({ name, desc, value, stats, onChange, digits = 2 }: {
  name: string; desc?: string; value: number; stats: RangeStats; onChange: (v: number) => void; digits?: number
}) {
  const span = stats.max - stats.min || 1
  return (
    <div className="slider-item">
      <div className="head">
        <span className="name">{name}{desc && <span>— {desc}</span>}</span>
        <span className="val">{value.toFixed(digits)}</span>
      </div>
      <input type="range" min={stats.min} max={stats.max} step={span / 500} value={value} onChange={e => onChange(Number(e.target.value))} />
      <div className="range"><span>{stats.min.toFixed(digits)}</span><span>median {stats.median.toFixed(digits)}</span><span>{stats.max.toFixed(digits)}</span></div>
    </div>
  )
}

export default function TryIt() {
  const { dataset, model, setDataset, setModel } = useSelection()
  const [tab, setTab] = useState<'csv' | 'manual'>('manual')
  const [ranges, setRanges] = useState<SensorRangesResponse | null>(null)
  const [regime, setRegime] = useState<number>(0)
  const [sensorValues, setSensorValues] = useState<Record<string, number>>({})
  const [opValues, setOpValues] = useState<Record<string, number>>({})
  const [cycle, setCycle] = useState<number>(100)
  const [faultCluster, setFaultCluster] = useState<'auto' | 0 | 1>('auto')
  const [result, setResult] = useState<ManualResponse | null>(null)
  const [csvResult, setCsvResult] = useState<CsvResponse | null>(null)
  const [csvEngine, setCsvEngine] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [over, setOver] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    setRanges(null); setResult(null); setCsvResult(null); setError(null)
    api.sensorRanges(dataset).then(r => {
      setRanges(r)
      const best = r.regimes ? r.regimes.reduce((a, b) => (a.n_train_rows >= b.n_train_rows ? a : b)).id : 0
      setRegime(best)
      setCycle(r.cycle?.median ?? 100)
      setFaultCluster('auto')
    }).catch(e => setError(e.message))
  }, [dataset])

  const regimeInfo = useMemo(() => ranges?.regimes?.find(r => r.id === regime) ?? null, [ranges, regime])
  const statsFor = useCallback((key: string): RangeStats | undefined =>
    regimeInfo?.sensors[key] ?? ranges?.sensors.find(s => s.key === key), [regimeInfo, ranges])

  useEffect(() => {
    if (!ranges) return
    const sv: Record<string, number> = {}
    ranges.sensors.forEach(s => { sv[s.key] = statsFor(s.key)?.median ?? s.median })
    setSensorValues(sv)
    const ov: Record<string, number> = {}
    if (regimeInfo) Object.assign(ov, regimeInfo.op_settings)
    else ranges.op_settings.forEach(o => { ov[o.key] = o.median })
    setOpValues(ov)
  }, [ranges, regimeInfo, statsFor])

  const activeModel: ModelKey = ranges && !ranges.available_models.includes(model) ? ranges.available_models[0] : model

  useEffect(() => {
    if (!ranges || tab !== 'manual' || Object.keys(sensorValues).length !== ranges.sensors.length) return
    const timer = setTimeout(() => {
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl
      setBusy(true)
      api.predictManual({
        dataset, model: activeModel, sensors: sensorValues, op_settings: opValues,
        cycle: ranges.cycle ? cycle : null,
        fault_cluster: faultCluster === 'auto' ? null : faultCluster,
      }, ctrl.signal)
        .then(r => { if (!ctrl.signal.aborted) { setResult(r); setError(null) } })
        .catch(e => { if (e.name !== 'AbortError') setError(e.message) })
        .finally(() => { if (!ctrl.signal.aborted) setBusy(false) })
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [ranges, tab, sensorValues, opValues, cycle, faultCluster, dataset, activeModel])

  const upload = async (file: File | undefined) => {
    if (!file) return
    setBusy(true); setError(null); setCsvResult(null)
    try {
      const r = await api.predictCsv(dataset, activeModel, file)
      setCsvResult(r); setCsvEngine(0)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const csvSelected = csvResult?.engines[csvEngine]
  const csvChart = useMemo(() => csvSelected
    ? csvSelected.cycles.map((c, i) => {
        const row: Record<string, number> = { cycle: c }
        Object.entries(csvSelected.predictions).forEach(([k, v]) => { row[k] = v[i] })
        return row
      })
    : [], [csvSelected])

  const shown = tab === 'manual' ? result : csvSelected
  const shownRul = tab === 'manual' ? result?.predicted_rul : csvSelected?.final_predicted_rul

  return (
    <div className="page">
      <div className="selector-row" style={{ justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <div className="label" style={{ marginBottom: 6 }}>Try it yourself</div>
          <div className="muted">Live RUL prediction from sensor input, through the same saved preprocessing used in training</div>
        </div>
        <div style={{ display: 'flex', gap: 28 }}>
          <DatasetChips value={dataset} onChange={setDataset} />
          <ModelChips dataset={dataset} value={activeModel} onChange={setModel} />
        </div>
      </div>

      <div className="try-grid">
        <div className="panel">
          <div className="tabs">
            <button className={`tab ${tab === 'csv' ? 'active' : ''}`} onClick={() => setTab('csv')}>Upload CSV</button>
            <button className={`tab ${tab === 'manual' ? 'active' : ''}`} onClick={() => setTab('manual')}>Manual sensor input</button>
          </div>
          {error && <p className="error">{error}</p>}

          {tab === 'manual' && ranges && (
            <div className="slider-list">
              {ranges.regimes && (
                <div className="selector">
                  <span className="label">Operating regime (sets op-settings to the regime centroid and re-centers slider ranges)</span>
                  <div className="chips">
                    {ranges.regimes.map(r => (
                      <button key={r.id} className={`chip ${r.id === regime ? 'active' : ''}`} onClick={() => setRegime(r.id)}
                        title={Object.entries(r.op_settings).map(([k, v]) => `${k}=${v.toFixed(3)}`).join(' · ')}>
                        Regime {r.id}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {!ranges.regimes && ranges.op_settings.map(o => (
                <Slider key={o.key} name={o.key.replace('op_setting_', 'Op setting ')} value={opValues[o.key] ?? o.median} stats={o} digits={4}
                  onChange={v => setOpValues(s => ({ ...s, [o.key]: v }))} />
              ))}
              {ranges.cycle && (
                <Slider name="Cycle" desc="cycles since new (this dataset's tree model uses it directly)" value={cycle}
                  stats={{ ...ranges.cycle, p5: ranges.cycle.min, p95: ranges.cycle.max }} digits={0} onChange={v => setCycle(Math.round(v))} />
              )}
              {ranges.fault_cluster && (
                <div className="selector">
                  <span className="label">Inferred fault mode (cannot be derived from a single snapshot)</span>
                  <div className="chips">
                    {(['auto', 0, 1] as const).map(opt => (
                      <button key={String(opt)} className={`chip ${faultCluster === opt ? 'active' : ''}`} onClick={() => setFaultCluster(opt)}>
                        {opt === 'auto' ? `Auto (majority: cluster ${ranges.fault_cluster!.default})` : `Cluster ${opt}`}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {ranges.sensors.map(s => {
                const st = statsFor(s.key) ?? s
                return (
                  <Slider key={s.key} name={s.name} desc={s.description} value={sensorValues[s.key] ?? st.median} stats={st}
                    onChange={v => setSensorValues(sv => ({ ...sv, [s.key]: v }))} />
                )
              })}
            </div>
          )}

          {tab === 'csv' && (
            <div>
              <label
                className={`dropzone ${over ? 'over' : ''}`}
                onDragOver={e => { e.preventDefault(); setOver(true) }}
                onDragLeave={() => setOver(false)}
                onDrop={e => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files[0]) }}
              >
                <input type="file" accept=".csv,.txt" onChange={e => upload(e.target.files?.[0])} />
                <div className="cta">DROP A TRAJECTORY FILE OR CLICK TO BROWSE</div>
                <div style={{ marginTop: 8, fontSize: 12 }}>
                  Raw 26-column space-separated C-MAPSS format (e.g. a slice of <code>test_{dataset}.txt</code>), or a CSV with a header naming <code>unit</code>, <code>cycle</code>, op settings and <code>sensor_N</code> columns. Multiple engines are scored separately.
                </div>
              </label>
              {csvResult && (
                <div style={{ marginTop: 18 }}>
                  <div className="chips" style={{ marginBottom: 12 }}>
                    {csvResult.engines.map((e, i) => (
                      <button key={e.unit} className={`chip ${i === csvEngine ? 'active' : ''}`} onClick={() => setCsvEngine(i)}>Engine {e.unit}</button>
                    ))}
                  </div>
                  <div className="label" style={{ marginBottom: 8 }}>Predicted RUL over the uploaded trajectory</div>
                  <div style={{ height: 260 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={csvChart} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                        <CartesianGrid stroke="#222a33" strokeDasharray="2 4" />
                        <XAxis dataKey="cycle" stroke="#7f8a96" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
                        <YAxis stroke="#7f8a96" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} width={40} />
                        <Tooltip contentStyle={{ background: '#171c23', border: '1px solid #2e3843', fontFamily: 'IBM Plex Mono', fontSize: 11 }}
                          formatter={(v: unknown, name: unknown) => [typeof v === 'number' ? v.toFixed(1) : String(v), MODEL_LABELS[name as ModelKey] ?? String(name)]} />
                        {Object.keys(csvSelected?.predictions ?? {}).map((k, i) => (
                          <Line key={k} type="monotone" dataKey={k} stroke={i === 0 ? '#3d8bff' : '#b06cff'} strokeWidth={k === activeModel ? 2.2 : 1.2} dot={false} isAnimationActive={false} />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="legend" style={{ marginTop: 8 }}>
                    {Object.keys(csvSelected?.predictions ?? {}).map((k, i) => (
                      <span key={k}><i className="swatch" style={{ background: i === 0 ? '#3d8bff' : '#b06cff' }} />{MODEL_LABELS[k as ModelKey]}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          {!ranges && !error && <div className="loading">loading sensor ranges…</div>}
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="label">Model prediction</div>
          <div className="pred-big">
            <div className="label">Predicted remaining useful life</div>
            <div className={`value status-text ${shown?.status ?? ''}`}>{busy && !shown ? '…' : shownRul === undefined ? '—' : shownRul.toFixed(0)}</div>
            <div className="unit">cycles{tab === 'csv' && csvSelected ? ` · at cycle ${csvSelected.final_cycle} of the uploaded trajectory` : ''}</div>
            <StatusBadge status={shown?.status} />
          </div>

          {shown && (
            <div style={{ marginTop: 18 }}>
              <div className="label" style={{ marginBottom: 10 }}>Top drivers</div>
              <ExplainPanel explanation={shown.explanation} top={6} />
            </div>
          )}

          <div className="note">
            {tab === 'manual' ? (
              <>
                A single snapshot has no history, so it is scored as an engine that has held these exact readings for a full {ranges?.lstm_window ?? ''}-cycle window
                (rolling std = 0, rolling mean = the reading){result?.assumptions.regime !== null && result?.assumptions.regime !== undefined ? `, in operating regime ${result.assumptions.regime}` : ''}
                {result?.assumptions.fault_cluster !== null && result?.assumptions.fault_cluster !== undefined ? `, fault-mode cluster ${result.assumptions.fault_cluster}` : ''}
                {result ? `, at cycle ${result.assumptions.cycle}` : ''}. Slider ranges are the real min/max of the training data{ranges?.regimes ? ' within the selected regime' : ''}.
                Upload a trajectory for a full-fidelity prediction.
              </>
            ) : (
              <>Every cycle of the uploaded trajectory is scored with the same preprocessing used in training (sensor selection, regime normalization, fault-mode inference and rolling features fit on the training set only). The LSTM uses a {ranges?.lstm_window ?? ''}-cycle window ending at each cycle.</>
            )}
          </div>
          <div className="footer-meta"><span>MODEL</span><b>{MODEL_LABELS[activeModel]} · {dataset}</b></div>
        </div>
      </div>
    </div>
  )
}
