import { NavLink, Route, Routes } from 'react-router-dom'
import { SelectionProvider, useSelection } from './selection'
import Fleet from './pages/Fleet'
import EngineDetail from './pages/EngineDetail'
import Comparison from './pages/Comparison'
import TryIt from './pages/TryIt'

function FleetStatus() {
  const { fleetSummary } = useSelection()
  if (!fleetSummary) return <div className="fleet-status muted">—</div>
  const { critical, degrading } = fleetSummary.counts
  const status = critical > 0 ? 'critical' : degrading > 0 ? 'degrading' : 'healthy'
  const text = critical > 0 ? `${critical} CRITICAL` : degrading > 0 ? `${degrading} DEGRADING` : 'FLEET NOMINAL'
  return (
    <div className={`fleet-status status-text ${status}`}>
      <span className={`dot ${status}`} />
      {text}
    </div>
  )
}

function TopBar() {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="title">TURBOFAN RUL</span>
        <span className="subtitle">Predictive Maintenance Console</span>
      </div>
      <nav>
        <NavLink to="/" end>Fleet</NavLink>
        <NavLink to="/compare">Model comparison</NavLink>
        <NavLink to="/try">Try it yourself</NavLink>
      </nav>
      <FleetStatus />
    </header>
  )
}

export default function App() {
  return (
    <SelectionProvider>
      <div className="app">
        <TopBar />
        <Routes>
          <Route path="/" element={<Fleet />} />
          <Route path="/engine/:fd/:unit" element={<EngineDetail />} />
          <Route path="/compare" element={<Comparison />} />
          <Route path="/try" element={<TryIt />} />
        </Routes>
      </div>
    </SelectionProvider>
  )
}
