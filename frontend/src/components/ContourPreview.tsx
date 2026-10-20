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
import { useEffect, useMemo } from 'react';
import * as THREE from 'three';

const ROAD_STYLES: Record<string, { color: string; width: number }> = {
  motorway: { color: '#ff0000', width: 3 },
  trunk: { color: '#ff6600', width: 2.5 },
  primary: { color: '#ffa500', width: 2 },
  secondary: { color: '#ffd000', width: 1.5 },
  tertiary: { color: '#ffff00', width: 1.5 },
  residential: { color: '#ffffff', width: 1 },
  service: { color: '#bbbbbb', width: 1 },
};

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
  useEffect(() => {
    camera.position.set(target.x, target.y, 1); // optional: face top-down
    camera.lookAt(target);
  }, [camera, target]);

  return <OrbitControls target={target.toArray()} />;
}

export interface ContourLayer {
  geometry: MultiPolygon
  elevation: number
  thickness?: number
  roads?: Record<string, MultiLineString>
  waterways?: MultiLineString
  buildings?: MultiPolygon
}

interface ContourPreviewProps {
  layers: ContourLayer[]
  darkMode?: boolean
}

/**
 * ContourPreview
 *
 * Main component rendering a 3D preview of stacked contour layers.
 * Sets up the Three.js canvas, computes centroids and extents for camera control,
 * and stacks layers vertically based on thickness.
 *
 * @param layers - Array of contour layers with geometry, elevation, and optional thickness.
 * @param darkMode - Boolean indicating if dark mode is active.
 * @returns JSX element containing the 3D preview canvas.
 *
 * Uses useMemo hooks to efficiently compute:
 * - The centroid and extent of all points for camera targeting and scaling.
 * - The cumulative heights of layers for proper vertical stacking.
 *
 * Handles empty layers gracefully by rendering a placeholder box.
 */
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
  positionY,
  thickness,
  index,
  color: colorProp,
}: {
  geometry: MultiPolygon;
  elevation: number;
  cx: number;
  cy: number;
  positionY: number;
  thickness: number;
  index: number;
  color?: string;
}) {
  const geom = useMemo(() => {
    if (!geometry || !Array.isArray(geometry.coordinates) || geometry.coordinates.length === 0) {
      return null;
    }

    const shapes: THREE.Shape[] = [];

    // Convert each polygon and its holes into THREE.Shape instances.
    for (const polygon of geometry.coordinates) {
      if (!polygon[0] || polygon[0].length < 3) continue;

      const outer = polygon[0];
      const shape = new THREE.Shape();
      outer.forEach(([x, y], i) => {
        if (i === 0) shape.moveTo(x, y);
        else shape.lineTo(x, y);
      });

      // Add holes as paths to the shape.
      for (let ringIdx = 1; ringIdx < polygon.length; ringIdx++) {
        const holeRing = polygon[ringIdx];
        if (!holeRing || holeRing.length < 3) continue;
        const holePath = new THREE.Path();
        holeRing.forEach(([x, y], i) => {
          if (i === 0) holePath.moveTo(x, y);
          else holePath.lineTo(x, y);
        });
        shape.holes.push(holePath);
      }

      // Calculate area per shape using THREE.ShapeUtils.area
      const points = shape.getPoints();
      const area = Math.abs(THREE.ShapeUtils.area(points));
      if (area >= 1e-10) {
        shapes.push(shape);
      }
    }

    if (shapes.length === 0) {
      return null;
    }

    try {
      return new THREE.ExtrudeGeometry(shapes, {
        depth: isFinite(thickness) && thickness > 0 ? thickness : 0.003,
        bevelEnabled: false,
      });
    } catch (err) {
      console.error(`Failed to extrude layer ${index}:`, err);
      return null;
    }
  }, [geometry, thickness, index]);

  useEffect(() => {
    return () => {
      geom?.dispose();
    };
  }, [geom]);

  const color = useMemo(() => {
    return new THREE.Color(colorProp ?? `hsl(${(index * 50) % 360}, 100%, 50%)`);
  }, [colorProp, index]);

  if (!geom) {
    return null;
  }

  return (
    <group position={[0, 0, positionY]}>
      <mesh geometry={geom}>
        <meshStandardMaterial
          color={color}
          wireframe={false}
        />
      </mesh>
    </group>
  );
}

function RoadLines({ roads, z }: { roads: Record<string, MultiLineString>; z: number }) {
  if (!roads) return null;
  const elems: React.ReactElement[] = [];
  Object.entries(roads).forEach(([type, geom]) => {
    if (!Array.isArray(geom.coordinates)) return;
    geom.coordinates.forEach((coords, i) => {
      const pts = coords.map(([x, y]) => new THREE.Vector3(x, y, z));
      const style = ROAD_STYLES[type] || { color: '#000', width: 1 };
      elems.push(
        <Line
          key={`${type}-${i}`}
          points={pts}
          color={style.color}
          lineWidth={style.width}
        />
      );
    });
  });
  return <group>{elems}</group>;
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

export function WaterPlane({ width, height, positionY }: { width: number; height: number; positionY: number }) {
  // Semi-transparent blue plane
  return (
    <mesh position={[0, 0, positionY]} rotation={[0, 0, 0]}>
       <planeGeometry args={[width, height]} />
       <meshStandardMaterial color="#00aaff" transparent opacity={0.3} side={THREE.DoubleSide} />
    </mesh>
  );
}

export default function ContourPreview({ layers, darkMode = true }: ContourPreviewProps) {
  // Compute the minimum elevation for potential use.
  const minElevation = layers.length > 0
    ? layers.reduce((min, l) => Math.min(min, l.elevation), Infinity)
    : 0;

  /**
   * Compute centroid (cx, cy) and maximum extent (xyExtent) of all points.
   * This is used to center the camera and potentially scale the view.
   */
  const { cx, cy, xyExtent } = useMemo(() => {
    const points: [number, number][] = [];
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const layer of layers) {
      if (!layer.geometry || !Array.isArray(layer.geometry.coordinates)) continue;
      for (const polygon of layer.geometry.coordinates) {
        if (!Array.isArray(polygon)) continue;
        for (const ring of polygon) {
          if (!Array.isArray(ring)) continue;
          for (const p of ring) {
            if (Array.isArray(p) && p.length >= 2 && typeof p[0] === 'number' && typeof p[1] === 'number') {
              const x = p[0];
              const y = p[1];
              points.push([x, y]);
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (y < minY) minY = y;
              if (y > maxY) maxY = y;
            }
          }
        }
      }
    }

    if (points.length === 0) {
      return {
        allPoints: points,
        cx: 0,
        cy: 0,
        xyExtent: 1,
        bounds: { minX: 0, maxX: 0, minY: 0, maxY: 0 },
      };
    }

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const xExtent = maxX - minX;
    const yExtent = maxY - minY;
    const xyExtent = Math.max(xExtent, yExtent) || 1;
    const bounds = { minX, maxX, minY, maxY };
    return { allPoints: points, cx, cy, xyExtent, bounds };
  }, [layers]);

  /**
   * Compute cumulative vertical offsets for each layer to stack them properly.
   * Uses default thickness of 0.003 if not specified.
   */
  const cumulativeHeights = useMemo(() => {
    const result: number[] = [];
    let sum = 0;
    for (const layer of layers) {
      result.push(sum);
      sum += layer.thickness ?? 0.003;
    }
    return result;
  }, [layers]);

  // Calculate extents for color interpolation
  const waterStats = useMemo(() => {
    const bathyLayers = layers.filter(l => l.elevation < 0);
    if (!bathyLayers.length) return { min: 0, max: 0 };
    return bathyLayers.reduce(
      (acc, l) => ({ min: Math.min(acc.min, l.elevation), max: Math.max(acc.max, l.elevation) }),
      { min: Infinity, max: -Infinity }
    );
  }, [layers]);

  const landStats = useMemo(() => {
    const landLayers = layers.filter(l => l.elevation >= 0);
    if (!landLayers.length) return { min: 0, max: 0 };
    return landLayers.reduce(
      (acc, l) => ({ min: Math.min(acc.min, l.elevation), max: Math.max(acc.max, l.elevation) }),
      { min: Infinity, max: -Infinity }
    );
  }, [layers]);

  // Color Constants
  const colors = useMemo(() => {
    // ... existing color logic ...
      // Dark Mode Palette (Brighter colors to stand out against dark bg)
      if (darkMode) {
          return {
              bg: '#111',
              waterDeep: new THREE.Color('#1E90FF'),  // Dodger Blue
              waterShallow: new THREE.Color('#00FFFF'), // Aqua
              landLow: new THREE.Color('#98FB98'),    // Pale Green
              landHigh: new THREE.Color('#F4A460'),   // Sandy Brown
              waterPlane: '#00FFFF', // Aqua
          }
      } 
      // Light Mode Palette (Darker colors to stand out against white bg)
      else {
          return {
              bg: '#fff',
              waterDeep: new THREE.Color('#00008B'),  // Dark Blue
              waterShallow: new THREE.Color('#40E0D0'), // Turquoise
              landLow: new THREE.Color('#228B22'),    // Forest Green
              landHigh: new THREE.Color('#8B4513'),   // Saddle Brown
              waterPlane: '#00aaff', // Original blue
          }
      }
  }, [darkMode]);


  return (
    <div style={{ height: '400px', width: '100%', background: colors.bg, borderRadius: '8px', overflow: 'hidden' }}>
      {/* Three.js Canvas setup ... */}
      <Canvas
        style={{ width: '100%', height: '100%' }}
        orthographic camera={{ zoom: 600, position: [0, 0, 1] }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[0.5, 0.5, 1]} intensity={0.6} castShadow={false} />
          <CameraController layers={layers} />

          {/* Render placeholder box if no layers are provided. */}
          {layers.length === 0 && (
            <mesh>
              <boxGeometry args={[1, 1, 1]} />
              <meshStandardMaterial color="orange" />
            </mesh>
          )}

          {/* Map over each layer to render its extruded polygon mesh. */}
          {layers.map((layer, idx) => {
            const positionY = cumulativeHeights[idx]
            
            // Calculate color based on relative elevation within its group (land or water)
            let colorHex = "#888888"; // Default gray
            if (layer.elevation < 0) {
              // Bathymetry
              // Deepest point (min) -> t=0 (Deep Color)
              // Shallowest point (max) -> t=1 (Shallow Color)
              const range = waterStats.max - waterStats.min || 1;
              const t = (layer.elevation - waterStats.min) / range;
              const c = colors.waterDeep.clone().lerp(colors.waterShallow, t);
              colorHex = '#' + c.getHexString();
            } else {
              // Land
              // Lowest point (min) -> t=0 (Low Color)
              // Highest point (max) -> t=1 (High Color)
              const range = landStats.max - landStats.min || 1;
              const t = (layer.elevation - landStats.min) / range;
              const c = colors.landLow.clone().lerp(colors.landHigh, t);
              colorHex = '#' + c.getHexString();
            }

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
                color={colorHex}
              />
            )
          })}
          


          {/*
            Render additional features like roads and buildings if available in the layer.
            These are positioned at the appropriate height based on their thickness.
          */}
          {layers.map((layer, idx) => {
            const z = cumulativeHeights[idx]
            return (
              <group key={`extras-${idx}`}>
                {layer.roads && <RoadLines roads={layer.roads} z={z + (layer.thickness ?? 0.003) + 0.001} />}
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
          
          {/* Render Water Plane if we have negative elevations */}
          {minElevation < 0 && (
            // Find z-position where elevation crosses 0
            // Since layers are sorted, we find the first layer with elevation >= 0
            // and use its Z position.
            (() => {
              const seaLevelIndex = layers.findIndex(l => l.elevation >= 0);
              const z = seaLevelIndex >= 0 ? cumulativeHeights[seaLevelIndex] : cumulativeHeights[cumulativeHeights.length - 1];
              // Use extent of data for plane size
              const width = xyExtent * 1.5; 
              // Center it on cx, cy
              return (
                 <group position={[cx, cy, 0]}>
                    <mesh position={[0, 0, z]} rotation={[0, 0, 0]}>
                       <planeGeometry args={[width, width]} />
                       <meshStandardMaterial color={colors.waterPlane} transparent opacity={0.3} side={THREE.DoubleSide} />
                    </mesh>
                 </group>
              );
            })()
          )}
      </Canvas>
    </div>
  )
}