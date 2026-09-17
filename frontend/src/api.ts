export type ModelKey = 'random_forest' | 'xgboost' | 'lstm'
export type Status = 'healthy' | 'degrading' | 'critical'

export const MODEL_LABELS: Record<ModelKey, string> = {
  random_forest: 'Random Forest',
  xgboost: 'XGBoost',
  lstm: 'LSTM',
}
export const ALL_MODELS: ModelKey[] = ['random_forest', 'xgboost', 'lstm']

export interface SensorMeta { key: string; name: string; description: string }
export interface DatasetMeta {
  id: string; label: string; conditions: number; fault_modes: number
  available_models: ModelKey[]; n_test_engines: number; sensors: SensorMeta[]; lstm_window: number
}
export interface Metrics { rmse: number; r2: number; nasa_score: number; n_engines: number }
export interface FleetEngine { unit: number; n_cycles: number; predicted_rul: number; true_rul: number; status: Status }
export interface FleetResponse {
  dataset: string; model: ModelKey; metrics: Metrics
  status_counts: Record<Status, number>; engines: FleetEngine[]
}
export interface EngineResponse {
  dataset: string; unit: number; n_cycles: number; true_rul_at_last_cycle: number
  available_models: ModelKey[]; sensors: SensorMeta[]; history: Record<string, number>[]
  regime_per_cycle: number[] | null; fault_cluster: number | null
}
export interface ExplainItem { label: string; description: string; value: number; abs: number; share: number }
export interface Explanation { method: string; items: ExplainItem[] }
export interface PredictResponse {
  dataset: string; unit: number; model: ModelKey; cycle: number
  predicted_rul: number; true_rul: number; true_rul_capped: number; status: Status; explanation: Explanation
}
export interface CvSummary {
  folds: { rmse: number; r2: number }[]
  rmse_mean: number; rmse_std: number; r2_mean: number; r2_std: number
}
export interface ComparisonModel {
  label: string; test: Metrics; validation: { rmse: number; r2: number } | null; cv: CvSummary | null
}
export type ComparisonResponse = Record<string, { label: string; models: Partial<Record<ModelKey, ComparisonModel>> }>
export interface RangeStats { min: number; max: number; p5: number; p95: number; median: number }
export interface RegimeInfo {
  id: number; n_train_rows: number; op_settings: Record<string, number>; sensors: Record<string, RangeStats>
}
export interface SensorRangesResponse {
  dataset: string; sensors: (SensorMeta & RangeStats)[]; op_settings: ({ key: string } & RangeStats)[]
  regimes: RegimeInfo[] | null; cycle: { min: number; max: number; median: number } | null
  fault_cluster: { default: number; options: number[] } | null; lstm_window: number; available_models: ModelKey[]
}
export interface ManualResponse {
  dataset: string; model: ModelKey; predicted_rul: number; status: Status
  assumptions: { steady_state: boolean; cycle: number; regime: number | null; fault_cluster: number | null }
  explanation: Explanation
}
export interface CsvEngine {
  unit: number; n_cycles: number; cycles: number[]; predictions: Record<string, number[]>
  final_predicted_rul: number; final_cycle: number; status: Status; explanation: Explanation
}
export interface CsvResponse { dataset: string; model: ModelKey; engines: CsvEngine[] }

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* keep statusText */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json() as Promise<T>
}

const get = <T,>(url: string, signal?: AbortSignal) => fetch(url, { signal }).then(r => handle<T>(r))

export const api = {
  datasets: () => get<DatasetMeta[]>('/api/datasets'),
  fleet: (fd: string, model: ModelKey) => get<FleetResponse>(`/api/fleet/${fd}/${model}`),
  engine: (fd: string, unit: number) => get<EngineResponse>(`/api/engine/${fd}/${unit}`),
  predictAtCycle: (fd: string, unit: number, model: ModelKey, cycle: number, signal?: AbortSignal) =>
    get<PredictResponse>(`/api/engine/${fd}/${unit}/predict?model=${model}&cycle=${cycle}`, signal),
  comparison: () => get<ComparisonResponse>('/api/comparison'),
  sensorRanges: (fd: string) => get<SensorRangesResponse>(`/api/sensor-ranges/${fd}`),
  predictManual: (body: {
    dataset: string; model: ModelKey; sensors: Record<string, number>
    op_settings: Record<string, number>; cycle?: number | null; fault_cluster?: number | null
  }, signal?: AbortSignal) =>
    fetch('/api/predict/manual', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
    }).then(r => handle<ManualResponse>(r)),
  predictCsv: (fd: string, model: ModelKey, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`/api/predict/csv?dataset=${fd}&model=${model}`, { method: 'POST', body: form })
      .then(r => handle<CsvResponse>(r))
  },
}

export const fmt = (n: number | null | undefined, digits = 2) =>
  n === null || n === undefined || Number.isNaN(n) ? '—' : n.toFixed(digits)
