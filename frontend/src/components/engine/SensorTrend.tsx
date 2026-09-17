import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { SensorMeta } from '../../api'

interface Props {
  history: Record<string, number>[]
  sensors: SensorMeta[]
  a: string
  b: string
  cycle: number
  onChangeA: (k: string) => void
  onChangeB: (k: string) => void
}

const A_COLOR = '#3d8bff'
const B_COLOR = '#b06cff'

export default function SensorTrend({ history, sensors, a, b, cycle, onChangeA, onChangeB }: Props) {
  const meta = (k: string) => sensors.find(s => s.key === k)
  const Select = ({ value, onChange }: { value: string; onChange: (k: string) => void }) => (
    <select className="select" value={value} onChange={e => onChange(e.target.value)}>
      {sensors.map(s => <option key={s.key} value={s.key}>{s.name} — {s.description}</option>)}
    </select>
  )
  return (
    <div className="fill" style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
        <Select value={a} onChange={onChangeA} />
        <Select value={b} onChange={onChangeB} />
      </div>
      <div style={{ flex: 1, minHeight: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="#222a33" strokeDasharray="2 4" />
            <XAxis dataKey="cycle" stroke="#7f8a96" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
            <YAxis yAxisId="left" stroke={A_COLOR} tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} domain={['auto', 'auto']} width={58} />
            <YAxis yAxisId="right" orientation="right" stroke={B_COLOR} tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} domain={['auto', 'auto']} width={58} />
            <Tooltip
              contentStyle={{ background: '#171c23', border: '1px solid #2e3843', fontFamily: 'IBM Plex Mono', fontSize: 11 }}
              labelFormatter={(v) => `cycle ${v}`}
              formatter={(v: unknown, name: unknown) => [typeof v === 'number' ? v.toFixed(3) : String(v), meta(String(name))?.name ?? String(name)]}
            />
            <ReferenceLine yAxisId="left" x={cycle} stroke="#7f8a96" strokeDasharray="4 3" />
            <Line yAxisId="left" type="monotone" dataKey={a} stroke={A_COLOR} dot={false} strokeWidth={1.6} isAnimationActive={false} />
            <Line yAxisId="right" type="monotone" dataKey={b} stroke={B_COLOR} dot={false} strokeWidth={1.6} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="legend" style={{ marginTop: 8 }}>
        <span><i className="swatch" style={{ background: A_COLOR }} />{meta(a)?.name} — {meta(a)?.description}</span>
        <span><i className="swatch" style={{ background: B_COLOR }} />{meta(b)?.name} — {meta(b)?.description}</span>
      </div>
    </div>
  )
}
