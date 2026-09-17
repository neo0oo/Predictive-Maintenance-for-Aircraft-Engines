const CSS = `
.nn-svg text { font-family: var(--mono); font-size: 10px; fill: var(--muted); letter-spacing: .08em; }
.nn-svg .node { fill: var(--panel); stroke: var(--blue); stroke-width: 1.5; }
.nn-svg .edge { stroke: var(--blue); stroke-opacity: .28; stroke-width: 1; stroke-dasharray: 3 3; }
.nn-svg .out-node { fill: var(--blue); stroke: var(--blue); }
.nn-fire .in-node { animation: nnGlow .4s ease-out both; }
.nn-fire .e1 { animation: nnEdge .5s .12s linear both; }
.nn-fire .hid-node { animation: nnGlow .4s .5s ease-out both; }
.nn-fire .e2 { animation: nnEdge .5s .62s linear both; }
.nn-fire .out-node { animation: nnOut .6s 1s ease-out both; }
.nn-inflight .in-node { animation: nnIdlePulse 1s ease-in-out infinite; }
@keyframes nnGlow { 0% { fill: var(--panel); } 45% { fill: var(--blue); filter: drop-shadow(0 0 6px var(--blue)); } 100% { fill: var(--panel); } }
@keyframes nnEdge { 0% { stroke-opacity: .28; stroke-dashoffset: 24; } 40% { stroke-opacity: 1; } 100% { stroke-opacity: .28; stroke-dashoffset: 0; } }
@keyframes nnOut { 0% { filter: none; r: 13; } 40% { filter: drop-shadow(0 0 14px var(--blue)); r: 16; } 100% { filter: drop-shadow(0 0 6px var(--blue)); r: 13; } }
@keyframes nnIdlePulse { 0%, 100% { stroke-opacity: 1; } 50% { stroke-opacity: .35; } }
`

interface Props {
  inputs: string[]
  output: number | null
  pulseKey: number
  inflight: boolean
  modelLabel: string
}

export default function NNDiagram({ inputs, output, pulseKey, inflight, modelLabel }: Props) {
  const W = 360, H = 250
  const inX = 52, hidX = 200, outX = 318
  const ins = inputs.length ? inputs : ['—', '—', '—', '—', '—']
  const inY = ins.map((_, i) => 40 + (i * (H - 70)) / Math.max(ins.length - 1, 1))
  const hidY = [70, 115, 160]
  const outY = 115

  return (
    <div className="fill" style={{ display: 'flex', flexDirection: 'column' }}>
      <style>{CSS}</style>
      <svg
        key={pulseKey}
        className={`nn-svg ${pulseKey > 0 ? 'nn-fire' : ''} ${inflight ? 'nn-inflight' : ''}`}
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', flex: 1 }}
      >
        <text x={inX} y={18} textAnchor="middle">SENSORS</text>
        <text x={hidX} y={18} textAnchor="middle">{modelLabel.toUpperCase()}</text>
        <text x={outX} y={18} textAnchor="middle">RUL</text>
        {inY.map((y, i) => hidY.map((hy, j) => <line key={`e1-${i}-${j}`} className="edge e1" x1={inX + 9} y1={y} x2={hidX - 9} y2={hy} />))}
        {hidY.map((hy, j) => <line key={`e2-${j}`} className="edge e2" x1={hidX + 9} y1={hy} x2={outX - 13} y2={outY} />)}
        {inY.map((y, i) => (
          <g key={`in-${i}`}>
            <circle className="node in-node" cx={inX} cy={y} r={8} />
            <text x={inX - 14} y={y + 3} textAnchor="end">{ins[i]}</text>
          </g>
        ))}
        {hidY.map((hy, j) => <circle key={`h-${j}`} className="node hid-node" cx={hidX} cy={hy} r={9} />)}
        <circle className="node out-node" cx={outX} cy={outY} r={13} />
      </svg>
      <div className="nn-output">
        <div className="label">Model output</div>
        <div className="value">{inflight ? 'INFERRING…' : output === null ? '—' : `${output.toFixed(0)} cycles remaining`}</div>
      </div>
    </div>
  )
}
