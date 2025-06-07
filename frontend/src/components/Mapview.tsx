/**
 * MapView.tsx
 *
 * This component renders an interactive Leaflet map with support for:
 * - Displaying and updating a marker based on user clicks or external coordinates.
 * - Dynamically computing and displaying a rectangular selection area within the map bounds.
 * - Enforcing a maximum selectable area size (in km²) with optional square bounding.
 *
 * It handles geospatial projections using UTM zones to accurately compute areas and scale bounding boxes,
 * ensuring that the selection respects a maximum allowed area. The component provides callbacks to
 * notify parent components of bounding box changes.
 *
 * Key responsibilities:
 * - Manage map position and marker state.
 * - Calculate adjusted bounds with geodetic correctness and area constraints.
 * - Synchronize map interactions with external state via callbacks.
 */

import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import proj4 from 'proj4';
import { useEffect, useRef, useState } from 'react';
import { MapContainer, Marker, Rectangle, TileLayer, useMap } from 'react-leaflet';

const MAX_AREA_KM2 = 120; // Maximum selectable area in square kilometers to limit selection size for performance and UX

interface MapViewProps {
  coordinates: [number, number] | null
  onBoundsChange?: (bounds: [[number, number], [number, number]]) => void
  squareOutput?: boolean
  fixMode?: boolean
  setFixMode?: (val: boolean) => void;
  onFixedElevation?: (lat: number, lon: number) => void
}

/**
 * Calculate adjusted bounding box coordinates enforcing a maximum area limit and optional square shape.
 *
 * Uses UTM projection based on the center longitude to convert lat/lng to meters for accurate area calculation.
 * The bounding box is scaled down if it exceeds the max area, preserving center position.
 *
 * @param bounds - Original bounding box as [[south, west], [north, east]] in lat/lng
 * @param square - If true, adjust bounds to be a square with max allowed area
 * @param map - Leaflet map instance (unused here but available for extensions)
 * @returns Adjusted bounding box as [[south, west], [north, east]] in lat/lng coordinates
 */
function getAdjustedBounds(
  bounds: [[number, number], [number, number]],
  square: boolean,
  map: L.Map
): [[number, number], [number, number]] {
  const [[south, west], [north, east]] = bounds;

  const centerLat = (south + north) / 2;
  const centerLon = (west + east) / 2;

  // Calculate UTM zone from center longitude to select appropriate projection
  // UTM zones are 6 degrees wide, EPSG:326xx for northern hemisphere, EPSG:327xx for southern
  const zone = Math.floor((centerLon + 180) / 6) + 1;
  const epsg = centerLat >= 0
    ? `EPSG:326${zone.toString().padStart(2, '0')}`
    : `EPSG:327${zone.toString().padStart(2, '0')}`;

  // Define projection from WGS84 lat/lng to UTM meters
  const proj = proj4('EPSG:4326', epsg);

  // Project southwest and northeast corners to UTM meters for accurate distance and area
  const swProj = proj.forward([west, south]);
  const neProj = proj.forward([east, north]);

  console.debug("UTM SW:", swProj, "UTM NE:", neProj);

  // Calculate width and height in meters, then area in square kilometers
  const width = Math.abs(neProj[0] - swProj[0]);
  const height = Math.abs(neProj[1] - swProj[1]);
  const areaKm2 = (width * height) / 1_000_000;

  // Compute scale factor to enforce max area limit (<= MAX_AREA_KM2)
  // If area is larger, scale down uniformly to keep proportions
  const maxAreaKm2 = MAX_AREA_KM2;
  const scale = Math.min(1.0, Math.sqrt(maxAreaKm2 / areaKm2));

  if (square) {
    // For square output, set bounding box size to the smaller dimension capped by max area side length
    const maxSide = Math.sqrt(maxAreaKm2 * 1_000_000); // max side length in meters
    const size = Math.min(maxSide, Math.min(width, height));

    // Center point in projected coordinates
    const centerProj = proj.forward([centerLon, centerLat]);
    const half = size / 2;

    // Calculate square bounds in projected meters
    const minX = centerProj[0] - half;
    const maxX = centerProj[0] + half;
    const minY = centerProj[1] - half;
    const maxY = centerProj[1] + half;

    // Convert back to lat/lng coordinates
    const newSW = proj.inverse([minX, minY]);
    const newNE = proj.inverse([maxX, maxY]);

    console.debug("Square adjusted UTM box:", [minX, minY], [maxX, maxY]);

    return [
      [newSW[1], newSW[0]],
      [newNE[1], newNE[0]],
    ];
  } else {
    // For non-square output, scale the bounding box uniformly around center to enforce max area
    const centerX = (swProj[0] + neProj[0]) / 2;
    const centerY = (swProj[1] + neProj[1]) / 2;
    const halfWidth = (width * scale) / 2;
    const halfHeight = (height * scale) / 2;

    // Calculate inset bounds in projected meters
    const minX = centerX - halfWidth;
    const maxX = centerX + halfWidth;
    const minY = centerY - halfHeight;
    const maxY = centerY + halfHeight;

    // Convert back to lat/lng coordinates
    const newSW = proj.inverse([minX, minY]);
    const newNE = proj.inverse([maxX, maxY]);

    console.debug("Inset adjusted UTM box:", [minX, minY], [maxX, maxY]);

    return [
      [newSW[1], newSW[0]],
      [newNE[1], newNE[0]],
    ];
  }
}

/**
 * Component to recenter the map view to the provided coordinates without changing zoom or bounds.
 *
 * @param coordinates - Target center coordinates [lat, lng]
 * @param squareOutput - Whether bounding box is square (unused here but passed for consistency)
 */
function RecenterMap({
  coordinates,
  squareOutput,
}: {
  coordinates: [number, number],
  squareOutput: boolean,
}) {
  const map = useMap()
  useEffect(() => {
    // Pan map center to coordinates when they change, without zoom or bounds adjustment
    map.panTo(coordinates);
  }, [coordinates]);
  return null
}

/**
 * Custom hook to track and compute adjusted selection bounds based on current map view.
 *
 * Adds event listeners to update bounds on map move, zoom, or load.
 * Applies an inset factor to keep selection inset from edges (improves UX by avoiding edge clipping).
 * Calls onBoundsChange callback with adjusted bounds.
 *
 * @param map - Leaflet map instance or null
 * @param squareOutput - Whether to enforce square bounding box
 * @param onBoundsChange - Optional callback when bounds change
 * @returns Current selection bounds as [[south, west], [north, east]] or null if not available
 */
function useSelectionBounds(
  map: L.Map | null,
  squareOutput: boolean,
  onBoundsChange?: (bounds: [[number, number], [number, number]]) => void
) {
  // Holds the current adjusted selection bounds for rendering and external use
  const [selectionBounds, setSelectionBounds] = useState<[[number, number], [number, number]] | null>(null);

  useEffect(() => {
    if (!map) return;

    /**
     * Update selection bounds by:
     * - Getting current map bounds
     * - Applying inset to avoid edges (5% inset)
     * - Adjusting bounds to enforce max area and shape constraints
     * - Notifying parent component via callback
     */
    const update = () => {
      const sw = map.getBounds().getSouthWest();
      const ne = map.getBounds().getNorthEast();

      // Inset factor (5%) to shrink selection bounds slightly inside map edges for better UI feedback
      const insetFactor = 0.05;
      const latRange = ne.lat - sw.lat;
      const lngRange = ne.lng - sw.lng;

      const rawBounds: [[number, number], [number, number]] = [
        [sw.lat + latRange * insetFactor, sw.lng + lngRange * insetFactor],
        [ne.lat - latRange * insetFactor, ne.lng - lngRange * insetFactor]
      ];

      // Adjust bounds with projection and area constraints
      const adjustedBounds = getAdjustedBounds(rawBounds, squareOutput, map);

      if (onBoundsChange) onBoundsChange(adjustedBounds);
      setSelectionBounds(adjustedBounds);
    };

    update(); // Run immediately on mount or dependency change

    // Bind event listeners to update bounds on user interaction
    map.on('moveend', update);
    map.on('zoomend', update);
    map.on('load', update);

    // Cleanup event listeners on unmount or dependency change
    return () => {
      map.off('moveend', update);
      map.off('zoomend', update);
      map.off('load', update);
    };
  }, [map, squareOutput, onBoundsChange]);

  return selectionBounds;
}

/**
 * Main MapView component rendering Leaflet map, marker, and selection rectangle.
 *
 * Props:
 * - coordinates: external coordinates to center and mark on the map
 * - onBoundsChange: callback invoked with adjusted selection bounds when map view changes
 * - squareOutput: whether to enforce square bounding box shape
 * - fixMode: if true, enables a mode where clicking on the map sets a fixed elevation without moving the marker
 * - onFixedElevation: callback invoked with lat/lng when fixMode is active and map is clicked
 */
export default function MapView({ coordinates, onBoundsChange, squareOutput = false,  fixMode = false, onFixedElevation }: MapViewProps) {
  // Holds current marker/map center position; initialized to coordinates or default location (Yverdon)
  const [position, setPosition] = useState<[number, number]>(
    coordinates ?? [46.83, 6.86]
  );

  // Ref to Leaflet map instance for imperative API access
  const mapRef = useRef<L.Map | null>(null);

  // Effect to handle fixMode clicks on the map
  // If fixMode is enabled, clicking the map will call onFixedElevation with lat/lng
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!fixMode) return;

    // Handler for fixMode click
    const handleFixClick = (e: L.LeafletMouseEvent) => {
      if (onFixedElevation) {
        onFixedElevation(e.latlng.lat, e.latlng.lng);
      }
      setFixMode(false)
    };    
    map.on('click', handleFixClick);

    // Clean up after one click
    return () => {
      map.off('click', handleFixClick);
    };
  }, [fixMode, onFixedElevation]);

  // Sync internal position state with external coordinates prop changes
  useEffect(() => {
    if (coordinates) {
      setPosition(coordinates)
    }
  }, [coordinates])

  // Custom hook returns adjusted selection bounds based on current map view and props
  const selectionBounds = useSelectionBounds(mapRef.current, squareOutput, onBoundsChange);

  /**
   * LocationMarker component handles user clicks on the map to update marker position.
   * Uses map event listeners to update internal position state.
   */
  function LocationMarker() {
    const map = useMap();

    useEffect(() => {
      // Bind click event to update marker position on user click
      const handleClick = (e: L.LeafletMouseEvent) => {
      // Only allow normal marker movement if NOT in fix mode
        if (!fixMode) setPosition([e.latlng.lat, e.latlng.lng]);
      };

      map.on('click', handleClick);
      return () => {
        map.off('click', handleClick);
      };
    }, [map, fixMode]);

    return <Marker position={position} />;
  }

  return (
    // Leaflet MapContainer initializes the map view centered on current position with default zoom
    <MapContainer
      center={position}
      zoom={13}
      scrollWheelZoom={true}
      style={{ height: '400px', width: '100%' }}
      ref={(instance) => { mapRef.current = instance as L.Map; }}
      whenReady={() => {}}
    >
      {/* TileLayer provides OpenStreetMap tiles with attribution */}
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {/* Recenter map when external coordinates prop changes */}
      {coordinates && <RecenterMap coordinates={coordinates} squareOutput={squareOutput} />}
      {/* Marker showing current selected position */}
      <LocationMarker />
      {/* Rectangle showing current adjusted selection bounds */}
      {selectionBounds && (
        <Rectangle
          bounds={selectionBounds}
          pathOptions={{ color: 'blue', weight: 1 }}
        />
      )}
    </MapContainer>
  )
}