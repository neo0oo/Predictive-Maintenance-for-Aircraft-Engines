import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { Status } from '../../api'

const STATUS_COLORS: Record<Status, string> = { healthy: '#2fd39a', degrading: '#f2b632', critical: '#ef5350' }
const METAL = '#6b7480'
const DARK = '#2b323b'

function Disc({ x, r, t = 0.12, color = METAL, material }: { x: number; r: number; t?: number; color?: string; material?: React.ReactNode }) {
  return (
    <mesh position={[x, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
      <cylinderGeometry args={[r, r, t, 56]} />
      {material ?? <meshStandardMaterial color={color} metalness={0.75} roughness={0.35} />}
    </mesh>
  )
}

function Cone({ x, r, len, flip = false }: { x: number; r: number; len: number; flip?: boolean }) {
  return (
    <mesh position={[x, 0, 0]} rotation={[0, 0, flip ? Math.PI / 2 : -Math.PI / 2]}>
      <coneGeometry args={[r, len, 48]} />
      <meshStandardMaterial color={DARK} metalness={0.8} roughness={0.3} />
    </mesh>
  )
}

function HpcStage({ status, health }: { status: Status; health: number }) {
  const mat = useRef<THREE.MeshStandardMaterial>(null)
  const target = useMemo(() => new THREE.Color(STATUS_COLORS[status]), [status])
  useFrame(() => {
    if (!mat.current) return
    mat.current.color.lerp(target, 0.08)
    mat.current.emissive.lerp(target, 0.08)
    mat.current.emissiveIntensity += ((0.25 + 0.55 * (1 - health)) - mat.current.emissiveIntensity) * 0.08
  })
  const material = <meshStandardMaterial ref={mat} color={target} emissive={target} emissiveIntensity={0.3} metalness={0.4} roughness={0.45} />
  const discs = [-1.3, -1.05, -0.8, -0.55, -0.3, -0.05]
  return (
    <group>
      {discs.map((x, i) => (
        <mesh key={x} position={[x, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[1.05 - i * 0.05, 1.05 - i * 0.05, 0.14, 56]} />
          {material}
        </mesh>
      ))}
    </group>
  )
}

function EngineModel({ status, health }: { status: Status; health: number }) {
  return (
    <group rotation={[0.15, 0, 0]}>
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.22, 0.22, 6.6, 32]} />
        <meshStandardMaterial color={DARK} metalness={0.9} roughness={0.25} />
      </mesh>
      <Cone x={-3.35} r={0.55} len={0.7} />
      <Disc x={-2.85} r={1.55} t={0.18} color={DARK} />
      <Disc x={-2.6} r={1.45} t={0.1} />
      {[-2.3, -2.05, -1.8, -1.55].map((x, i) => <Disc key={x} x={x} r={1.35 - i * 0.08} />)}
      <HpcStage status={status} health={health} />
      <Disc x={0.35} r={0.95} t={0.5} color={DARK} />
      {[0.85, 1.1].map(x => <Disc key={x} x={x} r={0.85} />)}
      {[1.45, 1.7, 1.95, 2.2].map((x, i) => <Disc key={x} x={x} r={0.9 + i * 0.08} />)}
      <Cone x={2.85} r={1.1} len={1.1} flip />
    </group>
  )
}

export default function ThreeEngine({ status, rul, cap = 125 }: { status: Status | null; rul: number | null; cap?: number }) {
  const s: Status = status ?? 'healthy'
  const health = rul === null ? 1 : Math.max(0, Math.min(1, rul / cap))
  return (
    <Canvas camera={{ position: [0, 2.2, 7.6], fov: 38 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }}>
      <ambientLight intensity={0.55} />
      <directionalLight position={[4, 6, 5]} intensity={1.4} />
      <directionalLight position={[-5, -2, -4]} intensity={0.4} />
      <pointLight position={[0, 3, 0]} intensity={0.6} />
      <EngineModel status={s} health={health} />
      <OrbitControls autoRotate autoRotateSpeed={1.1} enableZoom={false} enablePan={false} />
    </Canvas>
  )
}
