import { useEffect, useReducer, useState } from 'react';
import './App.css';
import ContourPreview from './components/ContourPreview';
import MapView from './components/Mapview';

// ──────────────────────────────────────────────────────────
// Global reducer for all slicer parameters
type SlicerParams = {
  substrateSize: number;
  layerThickness: number;
  squareOutput: boolean;
  heightPerLayer: number;
  numLayers: number;
  lastChanged: 'height' | 'layers';
};

type Action =
  | { type: 'SET_SUBSTRATE_SIZE'; value: number }
  | { type: 'SET_LAYER_THICKNESS'; value: number }
  | { type: 'SET_SQUARE_OUTPUT'; value: boolean }
  | { type: 'SET_HEIGHT_PER_LAYER'; value: number }
  | { type: 'SET_NUM_LAYERS'; value: number }
  | { type: 'SET_LAST_CHANGED'; value: 'height' | 'layers' };

const initialSlicerParams: SlicerParams = {
  substrateSize: 400,
  layerThickness: 5,
  squareOutput: true,
  heightPerLayer: 250,
  numLayers: 5,
  lastChanged: 'height',
};

function slicerReducer(state: SlicerParams, action: Action): SlicerParams {
  switch (action.type) {
    case 'SET_SUBSTRATE_SIZE':
      return { ...state, substrateSize: action.value };
    case 'SET_LAYER_THICKNESS':
      return { ...state, layerThickness: action.value };
    case 'SET_SQUARE_OUTPUT':
      return { ...state, squareOutput: action.value };
    case 'SET_HEIGHT_PER_LAYER':
      return { ...state, heightPerLayer: action.value };
    case 'SET_NUM_LAYERS':
      return { ...state, numLayers: action.value };
    case 'SET_LAST_CHANGED':
      return { ...state, lastChanged: action.value };
    default:
      return state;
  }
}
// ──────────────────────────────────────────────────────────

function App() {
  const [address, setAddress] = useState('')
  const [coordinates, setCoordinates] = useState<[number, number] | null>(null)
  const [sliced, setSliced] = useState(false)
  const [contourLayers, setContourLayers] = useState<any[]>([])

  const [bounds, setBounds] = useState<[[number, number], [number, number]] | null>(null)
  const [areaStats, setAreaStats] = useState<{ width: number; height: number } | null>(null)
  const [elevationStats, setElevationStats] = useState<{ min: number; max: number } | null>(null)

  const [slicing, setSlicing] = useState(false)

  const [params, dispatch] = useReducer(slicerReducer, initialSlicerParams);

  function getWidthHeightMeters(bounds: [[number, number], [number, number]]): { width: number; height: number } {
    const [latMin, lonMin] = bounds[0]
    const [latMax, lonMax] = bounds[1]

    const R = 6371000
    const φ1 = (latMin * Math.PI) / 180
    const φ2 = (latMax * Math.PI) / 180
    const Δφ = φ2 - φ1
    const Δλ = ((lonMax - lonMin) * Math.PI) / 180
    const φm = (φ1 + φ2) / 2

    const height = R * Δφ
    const width = R * Δλ * Math.cos(φm)

    return {
      width: Math.abs(width),
      height: Math.abs(height),
    }
  }

  async function fetchElevationRange(bounds: [[number, number], [number, number]], signal?: AbortSignal) {
    const body = {
      lat_min: bounds[0][0],
      lon_min: bounds[0][1],
      lat_max: bounds[1][0],
      lon_max: bounds[1][1],
    }

    const res = await fetch("http://localhost:8000/api/elevation-range/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bounds: body }),
      signal,
    })

    if (!res.ok) throw new Error("Failed to get elevation range")
    const data = await res.json()
    setElevationStats({ min: data.min, max: data.max })
  }

  useEffect(() => {
    if (!bounds) return
    const dims = getWidthHeightMeters(bounds)
    setAreaStats(dims)
    const controller = new AbortController();
    fetchElevationRange(bounds, controller.signal).catch((err) => {
      if (err.name !== "AbortError") {
        console.error("Elevation range error:", err);
      }
    });
    return () => controller.abort();
  }, [bounds])

  useEffect(() => {
    if (!elevationStats) return;
    const range = elevationStats.max - elevationStats.min;
    if (params.lastChanged === 'height') {
      const newNum = Math.max(1, Math.round(range / params.heightPerLayer));
      if (newNum !== params.numLayers) {
        dispatch({ type: 'SET_NUM_LAYERS', value: newNum });
      }
    } else {
      const newHeight = Math.max(10, Math.min(5000, range / params.numLayers));
      if (newHeight !== params.heightPerLayer) {
        dispatch({ type: 'SET_HEIGHT_PER_LAYER', value: newHeight });
      }
    }
  }, [elevationStats, params.heightPerLayer, params.numLayers, params.lastChanged]);

  async function fetchCoordinates(address: string, signal?: AbortSignal): Promise<[number, number]> {
    const res = await fetch('http://localhost:8000/api/geocode/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address }),
      signal,
    })

    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Failed to fetch coordinates')
    return [data.lat, data.lon]
  }

  const handleGeocode = () => {
    const controller = new AbortController();
    fetchCoordinates(address, controller.signal)
      .then(setCoordinates)
      .catch((error) => {
        if (error.name !== 'AbortError') {
          alert('Geocoding failed: ' + error);
        }
      });
  }

  const handleSlice = async () => {
    if (!coordinates) {
      alert("Please select a location first.")
      return
    }
    const height = params.heightPerLayer;
    const layers = params.numLayers;

    if (!bounds) {
      alert("Could not read selected bounds.")
      return;
    }

    const controller = new AbortController();

    const body = {
      lat: coordinates[0],
      lon: coordinates[1],
      height_per_layer: height,
      num_layers: layers,
      simplify: Number((document.getElementById('simplify') as HTMLInputElement).value),
      bounds: {
        lat_min: bounds[0][0],
        lon_min: bounds[0][1],
        lat_max: bounds[1][0],
        lon_max: bounds[1][1],
      },
      substrate_size: params.substrateSize,
      layer_thickness: params.layerThickness,
    }

    try {
      setSlicing(true)
      const res = await fetch('http://localhost:8000/api/slice/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      if (!res.ok) throw new Error('Failed to slice contour data')

      const data = await res.json()
      console.log("Received contour layers:", data.layers)
      const layersWithPoints = (data.layers || []).map((layer: any) => ({
        ...layer,
        points: layer.geometry?.coordinates?.[0] ?? [],
      }))
      setContourLayers(layersWithPoints)
      setSliced(true)
      setSlicing(false)
      controller.abort();
    } catch (error) {
      setSlicing(false)
      alert("Slicing failed: " + error)
      controller.abort();
    }
  }

  const handleExport = async () => {
    const controller = new AbortController();

    try {
      const res = await fetch('http://localhost:8000/api/export/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          layers: contourLayers ,
          address,
          coordinates,
          height_per_layer: params.heightPerLayer,   
          num_layers: params.numLayers,     
        }),
        signal: controller.signal,
      })

      if (!res.ok) throw new Error('Failed to export contours')

      const blob = await res.blob();
      const disposition = res.headers.get("Content-Disposition");
      let filename = "contours.zip";
      const match = disposition?.match(/filename="(.+?)"/);
      if (match && match[1]) {
        filename = match[1];
      }
      console.log("Exported filename:", filename);

      const file = new File([blob], filename, { type: "application/zip" });
      const url = URL.createObjectURL(file);

      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      a.remove();
      controller.abort();
    } catch (error) {
      alert("Export failed: " + error)
      controller.abort();
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Lasercut Contour Map Generator</h1>
      </header>
      <div className="content-wrapper">
        <div className="sidebar">
          <div className="controls">
            <input
              type="text"
              placeholder="Enter address or location..."
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleGeocode()
                }
              }}
            />
            <button onClick={handleGeocode}>Locate on Map</button>
          </div>

          <div className="parameters">
            <label>
              Height per layer (m):
              <input
                type="number"
                id="layer-height"
                value={params.heightPerLayer}
                min={10}
                max={5000}
                onChange={(e) => {
                  const val = Math.max(10, Math.min(5000, Number(e.target.value)));
                  dispatch({ type: 'SET_LAST_CHANGED', value: 'height' });
                  dispatch({ type: 'SET_HEIGHT_PER_LAYER', value: val });
                }}
              />
            </label>
            <label>
              Number of layers:
              <input
                type="number"
                id="num-layers"
                value={params.numLayers}
                onChange={(e) => {
                  const val = Math.max(1, Math.floor(Number(e.target.value)));
                  dispatch({ type: 'SET_LAST_CHANGED', value: 'layers' });
                  dispatch({ type: 'SET_NUM_LAYERS', value: val });
                }}
              />
            </label>
            <label>
              Simplify shape:
              <input type="range" id="simplify" min="0" max="1" step="0.05" />
            </label>
            <label>
              Substrate size (mm):
              <input
                type="number"
                id="substrate-size"
                value={params.substrateSize}
                onChange={(e) => dispatch({ type: 'SET_SUBSTRATE_SIZE', value: Number(e.target.value) })}
              />
            </label>
            <label>
              Layer thickness (mm):
              <input
                type="number"
                id="layer-thickness"
                value={params.layerThickness}
                onChange={(e) => dispatch({ type: 'SET_LAYER_THICKNESS', value: Number(e.target.value) })}
              />
            </label>
            <label>
              <input
                type="checkbox"
                checked={params.squareOutput}
                onChange={(e) => dispatch({ type: 'SET_SQUARE_OUTPUT', value: e.target.checked })}
              />
              Square output
            </label>
            <button id="slice-button" onClick={handleSlice}>Slice!</button>
            <button id="export-button" onClick={handleExport} disabled={!sliced}>Export SVGs</button>
            <a id="download-link" href="#" download style={{ display: 'none' }}>
              ⬇️ Download ZIP
            </a>
          </div>
        </div>
        <div className="main-panel">
          <div className="map-container">
            <MapView coordinates={coordinates} onBoundsChange={setBounds} squareOutput={params.squareOutput} />
          </div>
          <div id="preview-3d">
            <h2>3D Preview</h2>
            { slicing ? <p>⏳ Slicing in progress...</p> : <ContourPreview layers={contourLayers} /> }
          </div>
        </div>
        <div className="info-sidebar">
          <h2>Area Info</h2>
          <p><strong>Center:</strong> {coordinates ? `${coordinates[0].toFixed(5)}, ${coordinates[1].toFixed(5)}` : 'N/A'}</p>
          <p><strong>Width:</strong> {areaStats ? `${areaStats.width.toFixed(0)} m` : 'N/A'}</p>
          <p><strong>Height:</strong> {areaStats ? `${areaStats.height.toFixed(0)} m` : 'N/A'}</p>
          <p><strong>Lowest Elevation:</strong> {elevationStats ? `${elevationStats.min.toFixed(0)} m` : '…'}</p>
          <p><strong>Highest Elevation:</strong> {elevationStats ? `${elevationStats.max.toFixed(0)} m` : '…'}</p>
        </div>
      </div>
    </div>
  )
}

export default App
