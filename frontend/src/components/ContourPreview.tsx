import { OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import type { ExtrudeGeometryOptions } from 'three'
import * as THREE from 'three'

const VERTICAL_SCALE = 0.5  // exaggerates vertical spacing between layers

interface ContourPreviewProps {
  layers: { points: [number, number][][]; elevation: number }[]
}

function PolygonLayer({
  points,
  elevation,
  baseElevation,
}: {
  points: [number, number][][]
  elevation: number
  baseElevation: number
}) {
  const color = new THREE.Color(`hsl(${(elevation * 50) % 360}, 100%, 50%)`)
  const shapes = points.map((ring) => {
    const shape = new THREE.Shape()
    ring.forEach(([x, y], i) => {
      if (i === 0) shape.moveTo(x, y)
      else shape.lineTo(x, y)
    })
    return shape
  })

  const extrudeSettings: ExtrudeGeometryOptions = {
    depth: 0.2,
    bevelEnabled: false,
  }

  return (
    <group position={[0, (elevation - baseElevation) * VERTICAL_SCALE, 0]}>
      {shapes.map((shape, i) => (
        <mesh key={i} geometry={new THREE.ExtrudeGeometry(shape, extrudeSettings)} position={[0, 0, 0]}>
          <meshStandardMaterial
            color={color}
            wireframe={false}
            polygonOffset
            polygonOffsetFactor={-1}
          />
        </mesh>
      ))}
    </group>
  )
}

export default function ContourPreview({ layers }: ContourPreviewProps) {
  console.log("ContourPreview layers:", layers)
  const minElevation = layers.length > 0 ? Math.min(...layers.map(layer => layer.elevation)) : 0
  return (
    <div style={{ height: '400px', width: '100%', background: '#111' }}>
      <Canvas
        style={{ width: '100%', height: '100%' }}
        camera={{ position: [0, 5, 10], fov: 50 }}
        dpr={[1, 2]}
      >
        <ambientLight />
        <pointLight position={[10, 10, 10]} />
        <directionalLight position={[0, 10, 5]} intensity={0.6} />
        <OrbitControls />
        {layers.length === 0 && (
          <mesh>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial color="orange" />
          </mesh>
        )}
        {layers.map((layer, idx) => (
          <PolygonLayer
            key={idx}
            elevation={layer.elevation}
            points={layer.points}
            baseElevation={minElevation}
          />
        ))}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
          <planeGeometry args={[200, 200]} />
          <meshStandardMaterial color="#222" />
        </mesh>
      </Canvas>
    </div>
  )
}