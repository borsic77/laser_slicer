/**
 * ContourPreview.tsx
 *
 * This file defines a React component that renders a 3D preview of stacked contour layers using Three.js and react-three-fiber.
 * Each contour layer is represented as an extruded polygon mesh, with elevation and thickness indicating its position and height in 3D space.
 * The preview supports orbit controls for interactive rotation and zooming.
 *
 * The main components and functions include:
 * - CameraController: Positions and orients the camera to center the view on the combined geometry of all layers.
 * - PolygonLayer: Converts GeoJSON MultiPolygon geometries into Three.js extruded shapes, handling holes and degenerate cases.
 * - ContourPreview: Orchestrates the rendering of all layers, computes centroids, extents, and cumulative heights for stacking,
 *   and manages the Three.js canvas setup.
 *
 * This component is useful for visualizing layered contour data, such as in laser cutting or 3D printing applications,
 * providing spatial context and depth perception for complex multi-layered shapes.
 */

import { Line, OrbitControls } from '@react-three/drei';
import { Canvas, useThree } from '@react-three/fiber';
import type { MultiLineString, MultiPolygon } from 'geojson';
import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * CameraController
 *
 * Controls the camera orientation and position to focus on the centroid of all contour layers.
 * Uses React Three Fiber's useThree hook to access the camera and updates it based on the layers' geometry.
 *
 * @param layers - Array of contour layers, each containing a MultiPolygon geometry and elevation.
 * @returns OrbitControls component configured to target the computed centroid.
 *
 * The centroid is computed as the average of all valid coordinate points across all polygons,
 * ensuring the camera focuses on the geometric center of the combined contours.
 */
function CameraController({ layers }: { layers: ContourLayer[] }) {
  const { camera } = useThree();

  // Compute the centroid (average position) of all points in all layers to center the camera target.
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

  // Update the camera position and orientation whenever the target changes.
  // Position the camera slightly above the target looking down (z=1) for a top-down view.
  useMemo(() => {
    camera.position.set(target.x, target.y, 1); // optional: face top-down
    camera.lookAt(target);
  }, [camera, target]);

  return <OrbitControls target={target.toArray()} />;
}

interface ContourLayer {
  geometry: MultiPolygon
  elevation: number
  thickness?: number
  roads?: MultiLineString
  waterways?: MultiLineString
  buildings?: MultiPolygon
}

interface ContourPreviewProps {
  layers: ContourLayer[]
}

/**
 * PolygonLayer
 *
 * Renders a single contour layer as an extruded 3D shape based on its MultiPolygon geometry.
 *
 * @param geometry - GeoJSON MultiPolygon defining the shape.
 * @param elevation - Elevation value used for color calculation.
 * @param cx - Center x-coordinate (not used directly here, but could be for positioning).
 * @param cy - Center y-coordinate.
 * @param positionY - Vertical position offset for stacking layers.
 * @param thickness - Extrusion depth (height) of the layer.
 * @param index - Layer index for logging and keys.
 * @returns A Three.js group containing the extruded mesh, or null if geometry is invalid or degenerate.
 *
 * Handles holes by creating THREE.Path objects and adding them as holes to the main shape.
 * Performs validation on points to avoid rendering invalid geometries.
 * Calculates a rough polygon area estimate to skip degenerate shapes that would not render properly.
 * Uses HSL color based on elevation to visually distinguish layers.
 * Defaults thickness to 0.003 if not provided or invalid, ensuring minimal extrusion.
 */
function PolygonLayer({
  geometry,
  elevation,
  cx,
  cy,
  positionY,
  thickness,
  index,
  color: colorProp,
}: {
  geometry: MultiPolygon
  elevation: number
  cx: number
  cy: number
  positionY: number
  thickness: number
  index: number
  color?: string
}) {
  const debug = false

  // Early exit if geometry is missing or malformed to avoid runtime errors.
  if (!geometry || !Array.isArray(geometry.coordinates) || geometry.coordinates.length === 0) {
    console.warn("Skipping invalid or empty geometry", geometry);
    return null;
  }

  const shapes: THREE.Shape[] = []

  // Convert each polygon and its holes into THREE.Shape instances.
  geometry.coordinates.forEach(polygon => {
    if (!polygon[0]) return;

    const outer = polygon[0];
    const shape = new THREE.Shape();
    outer.forEach(([x, y], i) => {
      if (i === 0) shape.moveTo(x, y);
      else shape.lineTo(x, y);
    });

    // Add holes as paths to the shape.
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
  
  // Extract points from shapes for area estimation and validation.
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

  // Validate path points to avoid rendering errors.
  if (
    pathPoints.some(p => !Array.isArray(p) || p.length !== 2 || typeof p[0] !== "number" || typeof p[1] !== "number")
  ) {
    console.error("Invalid shape pathPoints in PolygonLayer", index, pathPoints);
    return null;
  }

  // Area estimation using shoelace formula to detect degenerate polygons.
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

  // Skip rendering if shape is degenerate or too small, preventing unnecessary processing.
  if (shapes.length === 0 || pathPoints.length < 3 || areaEstimate < 1e-10) {
    console.warn(`Skipping degenerate shape at index ${index}`);
    return null;
  }

  // Color chosen via HSL hue cycling based on elevation to visually differentiate layers.
  const color = new THREE.Color(colorProp ?? `hsl(${(index * 50) % 360}, 100%, 50%)`);

  let geom: THREE.ExtrudeGeometry;
  try {
    // Extrude shape with specified thickness, defaulting to 0.003 if invalid.
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

function RoadLines({ geometry, z }: { geometry: MultiLineString; z: number }) {
  if (!geometry || !Array.isArray(geometry.coordinates)) return null;
  return (
    <group>
      {geometry.coordinates.map((coords, i) => {
        const pts = coords.map(([x, y]) => new THREE.Vector3(x, y, z));
        return (
          <Line
            key={i}
            points={pts}
            color="#000"
            lineWidth={1} // world units, adjust if needed
          />
        );
      })}
    </group>
  );
}

function WaterwayLines({ geometry, z }: { geometry: MultiLineString; z: number }) {
  if (!geometry || !Array.isArray(geometry.coordinates)) return null;
  return (
    <group>
      {geometry.coordinates.map((coords, i) => {
        const pts = coords.map(([x, y]) => new THREE.Vector3(x, y, z));
        return (
          <Line
            key={i}
            points={pts}
            color="#00aaff"
            lineWidth={2}
          />
        );
      })}
    </group>
  );
}

function BuildingLayer({ geometry, z }: { geometry: MultiPolygon; z: number }) {
  return (
    <PolygonLayer geometry={geometry} elevation={0} cx={0} cy={0} positionY={z} thickness={0.001} index={0} color="#888" />
  )
}

/**
 * ContourPreview
 *
 * Main component rendering a 3D preview of stacked contour layers.
 * Sets up the Three.js canvas, computes centroids and extents for camera control,
 * and stacks layers vertically based on thickness.
 *
 * @param layers - Array of contour layers with geometry, elevation, and optional thickness.
 * @returns JSX element containing the 3D preview canvas.
 *
 * Uses useMemo hooks to efficiently compute:
 * - The centroid and extent of all points for camera targeting and scaling.
 * - The cumulative heights of layers for proper vertical stacking.
 *
 * Handles empty layers gracefully by rendering a placeholder box.
 */
export default function ContourPreview({ layers }: ContourPreviewProps) {
  // Compute the minimum elevation for potential use (not used currently).
  const minElevation = layers.length > 0 ? Math.min(...layers.map(layer => layer.elevation)) : 0

  /**
   * Compute centroid (cx, cy) and maximum extent (xyExtent) of all points.
   * This is used to center the camera and potentially scale the view.
   */
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

  /**
   * Compute cumulative vertical offsets for each layer to stack them properly.
   * Uses default thickness of 0.003 if not specified.
   */
  const cumulativeHeights = useMemo(() => {
    const result: number[] = []
    let sum = 0
    for (const layer of layers) {
      result.push(sum)
      sum += layer.thickness ?? 0.003
    }
    return result
  }, [layers])

  // Total height of all layers combined (not currently used in rendering).
  const totalHeight = layers.reduce((sum, l) => sum + (l.thickness ?? 0.003), 0)

  const validLayerCount = layers.length;
  console.log(`Rendering ${validLayerCount} valid contour layers.`);

  return (
    <div style={{ height: '400px', width: '100%', background: '#111' }}>
      {/*
        Three.js Canvas setup with orthographic camera for consistent scaling.
        Ambient and directional lights provide basic illumination.
        CameraController centers the view based on layers.
      */}
      <Canvas
        style={{ width: '100%', height: '100%' }}
        orthographic camera={{ zoom: 600, position: [0, 0, 1] }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[0.5, 0.5, 1]} intensity={0.6} castShadow={false} />
          <CameraController layers={layers} />

          {/*
            Render placeholder box if no layers are provided.
            This helps indicate an empty or loading state visually.
          */}
          {layers.length === 0 && (
            <mesh>
              <boxGeometry args={[1, 1, 1]} />
              <meshStandardMaterial color="orange" />
            </mesh>
          )}

          {/*
            Map over each layer to render its extruded polygon mesh.
            Layers are stacked vertically using the cumulative heights computed earlier.
          */}
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

          {/*
            Render additional features like roads and buildings if available in the layer.
            These are positioned at the appropriate height based on their thickness.
          */}
          {layers.map((layer, idx) => {
            const z = cumulativeHeights[idx]
            console.log(`Layer ${idx} roads:`, layer.roads)
            console.log(`Layer ${idx} buildings:`, layer.buildings)
            return (
              <group key={`extras-${idx}`}>
                {layer.roads && <RoadLines geometry={layer.roads} z={z + (layer.thickness ?? 0.003) +0.001} />}
                {layer.waterways && <WaterwayLines geometry={layer.waterways} z={z + (layer.thickness ?? 0.003) +0.001} />}
                {layer.buildings && <BuildingLayer geometry={layer.buildings} z={z + (layer.thickness ?? 0.003) +0.001} />}
              </group>
            )
          })}
          

          {/*
            Render axes helper for orientation reference.
            Commented-out plane mesh could be used as a ground plane if needed.
          */}
          <axesHelper args={[0.2]} />
          {/* <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
            <planeGeometry args={[200, 200]} />
            <meshStandardMaterial color="#222" />
          </mesh> */}
      </Canvas>
    </div>
  )
}