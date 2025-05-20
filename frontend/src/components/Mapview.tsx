import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import proj4 from 'proj4';
import { useEffect, useRef, useState } from 'react';
import { MapContainer, Marker, Rectangle, TileLayer, useMap } from 'react-leaflet';

const MAX_AREA_KM2 = 120;

interface MapViewProps {
  coordinates: [number, number] | null
  onBoundsChange?: (bounds: [[number, number], [number, number]]) => void
  squareOutput?: boolean
}

function getAdjustedBounds(
  bounds: [[number, number], [number, number]],
  square: boolean,
  map: L.Map
): [[number, number], [number, number]] {
  const [[south, west], [north, east]] = bounds;

  const centerLat = (south + north) / 2;
  const centerLon = (west + east) / 2;

  // Estimate UTM zone from center longitude
  const zone = Math.floor((centerLon + 180) / 6) + 1;
  const epsg = centerLat >= 0
    ? `EPSG:326${zone.toString().padStart(2, '0')}`
    : `EPSG:327${zone.toString().padStart(2, '0')}`;

  // Define projection
  const proj = proj4('EPSG:4326', epsg);

  const swProj = proj.forward([west, south]);
  const neProj = proj.forward([east, north]);

  console.debug("UTM SW:", swProj, "UTM NE:", neProj);

  // Enforce max area limit of 25 km²
  const width = Math.abs(neProj[0] - swProj[0]);
  const height = Math.abs(neProj[1] - swProj[1]);
  const areaKm2 = (width * height) / 1_000_000;
  const maxAreaKm2 = MAX_AREA_KM2;
  const scale = Math.min(1.0, Math.sqrt(maxAreaKm2 / areaKm2));

  if (square) {
    const maxSide = Math.sqrt(maxAreaKm2 * 1_000_000); // in meters
    const size = Math.min(maxSide, Math.min(width, height));
    const centerProj = proj.forward([centerLon, centerLat]);
    const half = size / 2;
    const minX = centerProj[0] - half;
    const maxX = centerProj[0] + half;
    const minY = centerProj[1] - half;
    const maxY = centerProj[1] + half;

    const newSW = proj.inverse([minX, minY]);
    const newNE = proj.inverse([maxX, maxY]);

    console.debug("Square adjusted UTM box:", [minX, minY], [maxX, maxY]);

    return [
      [newSW[1], newSW[0]],
      [newNE[1], newNE[0]],
    ];
  } else {
    // Scale to enforce area limit, centered on center
    const centerX = (swProj[0] + neProj[0]) / 2;
    const centerY = (swProj[1] + neProj[1]) / 2;
    const halfWidth = (width * scale) / 2;
    const halfHeight = (height * scale) / 2;

    const minX = centerX - halfWidth;
    const maxX = centerX + halfWidth;
    const minY = centerY - halfHeight;
    const maxY = centerY + halfHeight;

    const newSW = proj.inverse([minX, minY]);
    const newNE = proj.inverse([maxX, maxY]);

    console.debug("Inset adjusted UTM box:", [minX, minY], [maxX, maxY]);

    return [
      [newSW[1], newSW[0]],
      [newNE[1], newNE[0]],
    ];
  }
}

function RecenterMap({
  coordinates,
  squareOutput,
}: {
  coordinates: [number, number],
  squareOutput: boolean,
}) {
  const map = useMap()
  useEffect(() => {
    // Only update map center, never zoom or fit bounds
    map.panTo(coordinates);
  }, [coordinates]);
  return null
}

function useSelectionBounds(
  map: L.Map | null,
  squareOutput: boolean,
  onBoundsChange?: (bounds: [[number, number], [number, number]]) => void
) {
  const [selectionBounds, setSelectionBounds] = useState<[[number, number], [number, number]] | null>(null);

  useEffect(() => {
    if (!map) return;

  const update = () => {
    const sw = map.getBounds().getSouthWest();
    const ne = map.getBounds().getNorthEast();

    const insetFactor = 0.05;
    const latRange = ne.lat - sw.lat;
    const lngRange = ne.lng - sw.lng;

    const rawBounds: [[number, number], [number, number]] = [
      [sw.lat + latRange * insetFactor, sw.lng + lngRange * insetFactor],
      [ne.lat - latRange * insetFactor, ne.lng - lngRange * insetFactor]
    ];

    const adjustedBounds = getAdjustedBounds(rawBounds, squareOutput, map);

    if (onBoundsChange) onBoundsChange(adjustedBounds);
    setSelectionBounds(adjustedBounds);
  };

    update(); // run immediately

    map.on('moveend', update);
    map.on('zoomend', update);
    map.on('load', update);
    return () => {
      map.off('moveend', update);
      map.off('zoomend', update);
      map.off('load', update);
    };
  }, [map, squareOutput, onBoundsChange]);

  return selectionBounds;
}

export default function MapView({ coordinates, onBoundsChange, squareOutput = false }: MapViewProps) {
  const [position, setPosition] = useState<[number, number]>(
    coordinates ?? [46.78, 6.64]
  ) // Default: Yverdon

  const mapRef = useRef<L.Map | null>(null);

  useEffect(() => {
    if (coordinates) {
      setPosition(coordinates)
    }
  }, [coordinates])

  const selectionBounds = useSelectionBounds(mapRef.current, squareOutput, onBoundsChange);

  function LocationMarker() {
    const map = useMap();

    useEffect(() => {
      const handleClick = (e: L.LeafletMouseEvent) => {
        setPosition([e.latlng.lat, e.latlng.lng]);
      };

      map.on('click', handleClick);
      return () => {
        map.off('click', handleClick);
      };
    }, [map]);

    return <Marker position={position} />;
  }

  return (
    <MapContainer
      center={position}
      zoom={13}
      scrollWheelZoom={true}
      style={{ height: '400px', width: '100%' }}
      ref={(instance) => { mapRef.current = instance as L.Map; }}
      whenReady={() => {}}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {coordinates && <RecenterMap coordinates={coordinates} squareOutput={squareOutput} />}
      <LocationMarker />
      {selectionBounds && (
        <Rectangle
          bounds={selectionBounds}
          pathOptions={{ color: 'blue', weight: 1 }}
        />
      )}
    </MapContainer>
  )
}