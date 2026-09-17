import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, type DatasetMeta, type ModelKey, type Status } from './api'

interface FleetSummary { counts: Record<Status, number>; dataset: string; model: ModelKey }

interface SelectionValue {
  datasets: DatasetMeta[]
  dataset: string
  model: ModelKey
  setDataset: (fd: string) => void
  setModel: (m: ModelKey) => void
  current: DatasetMeta | undefined
  isAvailable: (fd: string, m: ModelKey) => boolean
  fleetSummary: FleetSummary | null
  setFleetSummary: (s: FleetSummary | null) => void
  error: string | null
}

const Ctx = createContext<SelectionValue | null>(null)

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([])
  const [dataset, setDatasetState] = useState('FD001')
  const [model, setModelState] = useState<ModelKey>('lstm')
  const [fleetSummary, setFleetSummary] = useState<FleetSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.datasets().then(setDatasets).catch(e => setError(e.message))
  }, [])

  const isAvailable = useCallback(
    (fd: string, m: ModelKey) => datasets.find(d => d.id === fd)?.available_models.includes(m) ?? false,
    [datasets],
  )

  const setDataset = useCallback((fd: string) => {
    setDatasetState(fd)
    const meta = datasets.find(d => d.id === fd)
    if (meta && !meta.available_models.includes(model)) setModelState(meta.available_models[0])
  }, [datasets, model])

  const setModel = useCallback((m: ModelKey) => {
    if (isAvailable(dataset, m)) setModelState(m)
  }, [dataset, isAvailable])

  const current = datasets.find(d => d.id === dataset)

  const value = useMemo<SelectionValue>(() => ({
    datasets, dataset, model, setDataset, setModel, current, isAvailable, fleetSummary, setFleetSummary, error,
  }), [datasets, dataset, model, setDataset, setModel, current, isAvailable, fleetSummary, error])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useSelection() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useSelection outside provider')
  return v
}
