import type { Explanation } from '../../api'

const METHOD_LABELS: Record<string, string> = {
  shap: 'SHAP values (TreeExplainer), summed per physical sensor',
  gradient_x_input: 'Gradient × input saliency over the LSTM window, summed per sensor',
}

export default function ExplainPanel({ explanation, top = 8 }: { explanation: Explanation | null; top?: number }) {
  if (!explanation) return <div className="loading">awaiting inference…</div>
  const items = explanation.items.slice(0, top)
  const max = items[0]?.abs || 1
  return (
    <div className="fill" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="bars" style={{ flex: 1, justifyContent: 'center' }}>
        {items.map(it => (
          <div className="bar-row" key={it.label} title={`${it.description}: ${it.value >= 0 ? '+' : ''}${it.value.toFixed(2)} cycles`}>
            <span>{it.label}</span>
            <div className="bar-track"><div className="bar-fill" style={{ width: `${(it.abs / max) * 100}%` }} /></div>
            <span className="num">{it.value >= 0 ? '+' : '−'}{it.abs.toFixed(1)}</span>
          </div>
        ))}
      </div>
      <div className="caption" style={{ textAlign: 'left' }}>
        {METHOD_LABELS[explanation.method] ?? explanation.method}. Sign shows push toward longer (+) or shorter (−) RUL.
      </div>
    </div>
  )
}
