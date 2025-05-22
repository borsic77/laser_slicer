import { OrbitControls } from '@react-three/drei';
import { Canvas, useThree } from '@react-three/fiber';
import type { MultiPolygon } from 'geojson';
import { useMemo } from 'react';
import * as THREE from 'three';

function CameraController({ layers }: { layers: { geometry: MultiPolygon; elevation: number; thickness?: number }[] }) {
  const { camera } = useThree();

  const target = useMemo(() => {
    const points: [number, number][] = layers.flatMap(layer =>
      layer.geometry.coordinates.flatMap((polygon) =>
        polygon.flat().filter((p): p is [number, number] =>
          Array.isArray(p) && p.length === 2 && typeof p[0] === 'number' && typeof p[1] === 'number'
        )
      )
    );

    if (points.length === 0) return new THREE.Vector3(0, 0, 0);

    const avgX = points.reduce((sum: number, p: [number, number]) => sum + p[0], 0) / points.length;
    const avgY = points.reduce((sum: number, p: [number, number]) => sum + p[1], 0) / points.length;

    return new THREE.Vector3(avgX, avgY, 0);
  }, [layers]);

  useMemo(() => {
    camera.position.set(target.x, target.y, 1); // optional: face top-down
    camera.lookAt(target);
  }, [camera, target]);

  return <OrbitControls target={target.toArray()} />;
}


interface ContourPreviewProps {
  layers: { geometry: MultiPolygon; elevation: number; thickness?: number }[]
}

function PolygonLayer({
  geometry,
  elevation,
  cx,
  cy,
  positionY,
  thickness,
  index,
}: {
  geometry: MultiPolygon
  elevation: number
  cx: number
  cy: number
  positionY: number
  thickness: number
  index: number
}) {
  const debug = false

  if (!geometry || !Array.isArray(geometry.coordinates) || geometry.coordinates.length === 0) {
    console.warn("Skipping invalid or empty geometry", geometry);
    return null;
  }

  const shapes: THREE.Shape[] = []

  geometry.coordinates.forEach(polygon => {
    if (!polygon[0]) return;

    const outer = polygon[0];
    const shape = new THREE.Shape();
    outer.forEach(([x, y], i) => {
      if (i === 0) shape.moveTo(x, y);
      else shape.lineTo(x, y);
    });

    for (let ringIdx = 1; ringIdx < polygon.length; ringIdx++) {
      const holeRing = polygon[ringIdx];
      const holePath = new THREE.Path();
      holeRing.forEach(([x, y], i) => {
        if (i === 0) holePath.moveTo(x, y);
        else holePath.lineTo(x, y);
      });
      shape.holes.push(holePath);
    }

    shapes.push(shape);
  })

  const pathPoints: [number, number][] = [];
  
  for (const shape of shapes) {
    const points = shape.getPoints?.();
    if (!Array.isArray(points)) {
      console.error(`⚠️ shape.getPoints() returned invalid value in layer ${index}:`, points);
      continue;
    }

    for (const pt of points) {
      if (pt && typeof pt.x === "number" && typeof pt.y === "number") {
        pathPoints.push([pt.x, pt.y]);
      } else {
        console.error("⚠️ Invalid point in shape.getPoints():", pt);
      }
    }
  }

  if (
    pathPoints.some(p => !Array.isArray(p) || p.length !== 2 || typeof p[0] !== "number" || typeof p[1] !== "number")
  ) {
    console.error("Invalid shape pathPoints in PolygonLayer", index, pathPoints);
    return null;
  }

  // console.log(`Shape[${index}] path point count:`, pathPoints.length);
  // console.log(`Shape[${index}] path points:`, pathPoints);

  let areaEstimate = 0;

  if (!Array.isArray(pathPoints) || pathPoints.length < 3) {
    console.warn(`Skipping area estimate: not enough valid points in layer ${index}`);
    return null;
  }

  for (let i = 0; i < pathPoints.length; i++) {
    const pt = pathPoints[i];
    if (
      !Array.isArray(pt) ||
      pt.length !== 2 ||
      typeof pt[0] !== "number" ||
      typeof pt[1] !== "number"
    ) {
      console.error(`⚠️ Invalid point at index ${i} in layer ${index}:`, pt);
      console.debug("Full pathPoints:", pathPoints);
      return null;
    }
  }

  for (let i = 0; i < pathPoints.length; i++) {
    const pt1 = pathPoints[i]!;
    const pt2 = pathPoints[(i + 1) % pathPoints.length]!;
    areaEstimate += pt1[0] * pt2[1] - pt2[0] * pt1[1];
  }
  areaEstimate = Math.abs(areaEstimate / 2);

  console.log(`Shape[${index}] estimated area:`, areaEstimate);

  if (shapes.length === 0 || pathPoints.length < 3 || areaEstimate < 1e-10) {
    console.warn(`Skipping degenerate shape at index ${index}`);
    return null;
  }

  const color = new THREE.Color(`hsl(${(elevation * 50) % 360}, 100%, 50%)`)

  let geom: THREE.ExtrudeGeometry;
  try {
    geom = new THREE.ExtrudeGeometry(shapes, {
      depth: isFinite(thickness) && thickness > 0 ? thickness : 0.003,
      bevelEnabled: false,
    });
  } catch (err) {
    console.error(`Failed to extrude layer ${index}:`, err);
    return null;
  }
  // console.log(`Extruded geometry [${index}]`, geom);

  return (
    <group position={[0, 0, positionY]}>
      <mesh geometry={geom}>
        <meshStandardMaterial
          color={color}
          wireframe={false}
        />
      </mesh>
    </group>
  )
}

export default function ContourPreview({ layers }: ContourPreviewProps) {
  // console.log("ContourPreview layers:", layers)
  const minElevation = layers.length > 0 ? Math.min(...layers.map(layer => layer.elevation)) : 0
  const { allPoints, cx, cy, xyExtent } = useMemo(() => {
    const points: [number, number][] = layers
      .flatMap(layer =>
        layer.geometry.coordinates.flatMap(polygon =>
          polygon.flat().filter((p): p is [number, number] => Array.isArray(p) && p.length === 2)
        )
      );
    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const xExtent = Math.max(...xs) - Math.min(...xs);
    const yExtent = Math.max(...ys) - Math.min(...ys);
    const xyExtent = Math.max(xExtent, yExtent);
    return { allPoints: points, cx, cy, xyExtent };
  }, [layers]);

  const cumulativeHeights = useMemo(() => {
    const result: number[] = []
    let sum = 0
    for (const layer of layers) {
      result.push(sum)
      sum += layer.thickness ?? 0.003
    }
    return result
  }, [layers])

  const totalHeight = layers.reduce((sum, l) => sum + (l.thickness ?? 0.003), 0)

  const validLayerCount = layers.length;
  console.log(`Rendering ${validLayerCount} valid contour layers.`);

  return (
    <div style={{ height: '400px', width: '100%', background: '#111' }}>
      <Canvas
        style={{ width: '100%', height: '100%' }}
        orthographic camera={{ zoom: 600, position: [0, 0, 1] }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[0.5, 0.5, 1]} intensity={0.6} castShadow={false} />
          <CameraController layers={layers} />
                {layers.length === 0 && (
                  <mesh>
                    <boxGeometry args={[1, 1, 1]} />
                    <meshStandardMaterial color="orange" />
                  </mesh>
                )}
        {layers.map((layer, idx) => {
          const positionY = cumulativeHeights[idx]
          return (
            <PolygonLayer
              key={`layer-${idx}`}
              index={idx}
              elevation={layer.elevation}
              geometry={layer.geometry}
              cx={cx}
              cy={cy}
              positionY={positionY}
              thickness={layer.thickness ?? 0.003}
            />
          )
        })}

        <axesHelper args={[0.2]} />
        {/* <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
          <planeGeometry args={[200, 200]} />
          <meshStandardMaterial color="#222" />
        </mesh> */}
      </Canvas>
    </div>
  )
}