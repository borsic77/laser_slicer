/**
 * App.tsx
 *
 * Main application component for the Laser Contour Map Generator.
 *
 * This component orchestrates the full workflow of the app:
 *  - Maintains and synchronizes slicer parameters (substrate, layer, etc) using a reducer for coupled state.
 *  - Handles address geocoding and map coordinate selection.
 *  - Communicates with backend APIs for geocoding, elevation range, slicing, and SVG export.
 *  - Manages and displays 3D contour previews and area/elevation info.
 *  - Provides sidebar controls for user interaction and parameter adjustment.
 *  - Coordinates state and cross-component updates for a responsive UX.
 */

import { useEffect, useReducer, useRef, useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';
import ContourPreview from './components/ContourPreview';
import ManualModal from './components/ManualModal';
import MapView from './components/Mapview';

const API_URL = import.meta.env.VITE_API_URL;

// ──────────────────────────────────────────────────────────
/**
 * SlicerParams
 * Holds all user-configurable parameters for slicing.
 * @property {number} substrateSize - Final output side length in millimeters (mm) of the laser substrate.
 * @property {number} layerThickness - Thickness of each physical cut layer in millimeters (mm).
 * @property {boolean} squareOutput - If true, the output area will be forced to a square; otherwise, it matches the selected bounds aspect ratio.
 * @property {number} heightPerLayer - Height in meters (m) of terrain represented by each layer. Interdependent with numLayers.
 * @property {number} numLayers - Number of contour layers to slice. Interdependent with heightPerLayer.
 *
 * Note: heightPerLayer and numLayers are tightly coupled; modifying one will auto-update the other based on elevation range.
 */
type SlicerParams = {
  /** Final output side length in mm (for laser substrate) */
  substrateSize: number;
  /** Thickness of each cut layer in mm */
  layerThickness: number;
  /** If true, output will be forced square; otherwise, matches selected bounds aspect */
  squareOutput: boolean;
  /** Meters of terrain height per layer (auto-updates numLayers if changed) */
  heightPerLayer: number;
  /** Number of layers to slice (auto-updates heightPerLayer if changed) */
  numLayers: number;
};

type Action =
  | { type: 'SET_SUBSTRATE_SIZE'; value: number }
  | { type: 'SET_LAYER_THICKNESS'; value: number }
  | { type: 'SET_SQUARE_OUTPUT'; value: boolean }
  | { type: 'SET_HEIGHT_PER_LAYER'; value: number }
  | { type: 'SET_NUM_LAYERS'; value: number };

const initialSlicerParams: SlicerParams = {
  substrateSize: 400,
  layerThickness: 5,
  squareOutput: true,
  heightPerLayer: 250,
  numLayers: 5,
};

/**
 * Reducer for slicer parameters.
 * Used instead of useState to keep interdependent fields (heightPerLayer, numLayers, etc) in sync and to batch updates atomically.
 * This approach prevents race conditions and simplifies logic for coupled parameter changes.
 */
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
    default:
      return state;
  }
}
// ──────────────────────────────────────────────────────────

function App() {
  // User-inputted address string for geocoding
  const [address, setAddress] = useState('')
  // Selected map center coordinates [latitude, longitude]; shared with MapView and info sidebar
  const [coordinates, setCoordinates] = useState<[number, number] | null>(null)
  // True if slicing has been performed and data is available for export
  const [sliced, setSliced] = useState(false)
  // Array of contour layer data (output polygons from backend); used for preview and export
  const [contourLayers, setContourLayers] = useState<any[]>([])

  // Selected map bounds ([[latMin, lonMin], [latMax, lonMax]]) as chosen by user on MapView
  const [bounds, setBounds] = useState<[[number, number], [number, number]] | null>(null)
  // Physical area dimensions (width/height in meters) of selected bounds; displayed in info sidebar
  const [areaStats, setAreaStats] = useState<{ width: number; height: number } | null>(null)
  // Elevation statistics (min/max in meters) for selected area; used for layer calculations and info
  const [elevationStats, setElevationStats] = useState<{ min: number; max: number } | null>(null)

  // True while waiting for slice API response; disables inputs and shows progress
  const [slicing, setSlicing] = useState(false)

  // Slicer parameters (substrate, thickness, height/layers, etc) managed via reducer for coupled updates
  const [params, dispatch] = useReducer(slicerReducer, initialSlicerParams);
  // Tracks which of heightPerLayer or numLayers was changed last, so the other can be auto-updated after elevation stats arrive
  const lastChanged = useRef<'height' | 'layers' | null>(null);

  // Geometry simplification amount (0=no simplification); affects contour detail level
  const [simplify, setSimplify] = useState(0);
  // Smoothing amount (0=no smoothing); buffers jagged edges in contours
  const [smoothing, setSmoothing] = useState(0);
  // Minimum feature area (cm²); removes polygons below this size after scaling
  const [minArea, setMinArea] = useState(0);
  // Minimum feature width (mm); removes narrow bridges/features after scaling
  const [minFeatureWidth, setMinFeatureWidth] = useState(0);

  // Controls visibility of the manual/help modal dialog
  const [showManual, setShowManual] = useState(false);

  /**
   * Compute the physical width and height (in meters) of a rectangular area given by bounds.
   * Uses the equirectangular approximation for small areas:
   *   width = R * Δλ * cos(φm)
   *   height = R * Δφ
   * where R is Earth's radius, Δλ/Δφ are longitude/latitude differences in radians, φm is the mean latitude.
   * Assumes bounds are [[latMin, lonMin], [latMax, lonMax]].
   * @param bounds - Area bounds as [[latMin, lonMin], [latMax, lonMax]]
   * @returns { width: number, height: number } in meters
   */
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

  /**
   * Fetch the minimum and maximum elevation (in meters) for the selected area bounds from the backend.
   * Makes a POST request to /api/elevation-range/ and updates elevationStats state.
   * @param bounds - Area bounds as [[latMin, lonMin], [latMax, lonMax]]
   * @param signal - Optional AbortSignal for cancellation
   * @returns Promise<void>
   * @sideeffects Updates elevationStats state on success
   * @throws Error if the request fails
   */
  async function fetchElevationRange(bounds: [[number, number], [number, number]], signal?: AbortSignal) {
    const body = {
      lat_min: bounds[0][0],
      lon_min: bounds[0][1],
      lat_max: bounds[1][0],
      lon_max: bounds[1][1],
    }

    const res = await fetch(`${API_URL}/api/elevation-range/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bounds: body }),
      signal,
    })

    if (!res.ok) throw new Error("Failed to get elevation range")
    const data = await res.json()
    setElevationStats({ min: data.min, max: data.max })
  }

  /**
   * React effect: When bounds change, update areaStats and fetch new elevationStats.
   * Triggers: runs whenever bounds changes (user selects a new area on the map).
   * Side effects: Updates areaStats, triggers elevationStats fetch, and sets lastChanged to 'height' to sync layer calculations.
   */
  useEffect(() => {
    if (!bounds) return;
    const dims = getWidthHeightMeters(bounds);
    setAreaStats(dims);
    // When bounds change, update lastChanged to 'height' so numLayers is updated from heightPerLayer after elevation stats are fetched
    lastChanged.current = 'height';
    const controller = new AbortController();

    const fetchData = async () => {
      try {
        await fetchElevationRange(bounds, controller.signal);
      } catch (err) {
        if (err instanceof Error && err.name !== "AbortError") {
          console.error("Elevation range error:", err);
        }
      }
    };

    fetchData();

    return () => {
      controller.abort();
    };
  }, [bounds]);

  /**
   * React effect: When elevationStats or relevant params change, synchronize heightPerLayer and numLayers.
   * Triggers: runs whenever elevationStats, params.heightPerLayer, or params.numLayers change.
   * Side effects: Updates the coupled parameter (numLayers or heightPerLayer) based on which was changed last.
   */
  useEffect(() => {
    if (!elevationStats) return;
    const range = elevationStats.max - elevationStats.min;
    const which = lastChanged.current;
    lastChanged.current = null;

    if (which === 'height') {
      const newNum = Math.max(1, Math.round(range / params.heightPerLayer));
      if (newNum !== params.numLayers) {
        dispatch({ type: 'SET_NUM_LAYERS', value: newNum });
      }
    } else if (which === 'layers') {
      const newHeight = Math.max(10, Math.min(5000, range / params.numLayers));
      if (newHeight !== params.heightPerLayer) {
        dispatch({ type: 'SET_HEIGHT_PER_LAYER', value: newHeight });
      }
    }
  }, [elevationStats, params.heightPerLayer, params.numLayers]);

  /**
   * Fetch latitude/longitude coordinates for a given address from the backend geocoding API.
   * @param address - Address string to geocode
   * @param signal - Optional AbortSignal for cancellation
   * @returns Promise<[number, number]> - [latitude, longitude] on success
   * @throws Error if geocoding fails or returns an error
   */
  async function fetchCoordinates(address: string, signal?: AbortSignal): Promise<[number, number]> {
    const res = await fetch(`${API_URL}/api/geocode/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address }),
      signal,
    })

    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Failed to fetch coordinates')
    return [data.lat, data.lon]
  }

  /**
   * Handler: Geocode the current address string and update coordinates.
   * Sequence:
   *  1. Calls fetchCoordinates with the address (shows error toast on failure).
   *  2. Updates coordinates state on success, which updates the map and info sidebar.
   *  3. Handles abort and error cases gracefully.
   * Side effects: May trigger map recenter and area info update.
   */
  const handleGeocode = () => {
    const controller = new AbortController();
    fetchCoordinates(address, controller.signal)
      .then(setCoordinates)
      .catch((error) => {
        if (error instanceof Error && error.name !== 'AbortError') {
          toast.error('Geocoding failed: ' + error.message);
        } else {
          toast.error('Geocoding failed: An unknown error occurred.');
        }
      });
  }

  /**
   * Handler: Trigger slicing operation for the current area and parameters.
   * Sequence:
   *  1. Validates that coordinates and bounds are set.
   *  2. Sends a POST to /api/slice/ with all relevant parameters.
   *  3. On success, updates contourLayers and sets sliced=true; disables slicing state.
   *  4. On error, shows a toast and disables slicing state.
   * Side effects: Updates preview, enables export, may show error/warning toasts.
   */
  const handleSlice = async () => {
    if (!coordinates) {
      toast.warn("Please select a location first.");
      return
    }
    const height = params.heightPerLayer;
    const layers = params.numLayers;

    if (!bounds) {
      toast.warn("Could not read selected bounds.");
      return;
    }

    const controller = new AbortController();

    const body = {
      lat: coordinates[0],
      lon: coordinates[1],
      height_per_layer: height,
      num_layers: layers,
      simplify,
      smoothing,
      min_area: minArea,
      min_feature_width: minFeatureWidth,
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
      const res = await fetch(`${API_URL}/api/slice/`, {
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
      if (error instanceof Error) {
        toast.error("Slicing failed: " + error.message);
      } else {
        toast.error("Slicing failed: An unknown error occurred.");
      }
      controller.abort();
    }
  }

  /**
   * Handler: Export the current contour layers as SVGs via backend API.
   * Sequence:
   *  1. Sends a POST to /api/export/ with all contour and parameter data.
   *  2. Receives a ZIP file, extracts filename, and triggers download.
   *  3. Handles errors and aborts gracefully, showing a toast on failure.
   * Side effects: Initiates file download for user.
   */
  const handleExport = async () => {
    const controller = new AbortController();

    try {
      const res = await fetch(`${API_URL}/api/export/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          layers: contourLayers ,
          address,
          coordinates,
          height_per_layer: params.heightPerLayer,   
          num_layers: params.numLayers,  
          min_feature_width: minFeatureWidth,
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
      if (error instanceof Error) {
        toast.error("Export failed: " + error.message);
      } else {
        toast.error("Export failed: An unknown error occurred.");
      }
      controller.abort();
    }
  }

  return (
    <div className="container">
      <header>
        <h2>Laser Contour Map Generator</h2>
      </header>
      <div className="content-wrapper">
        {/* ─────────────── Sidebar Controls Section ───────────────
            Contains all user controls for address input, geocoding,
            and all slicer/geometry parameters. Changes here update
            state and drive slicing/export logic. */}
        <div className="sidebar">
          <button onClick={() => setShowManual(true)}>Manual ❓</button>
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
                min={10} // meters per layer
                max={5000} 
                onChange={(e) => {
                  const val = Math.max(10, Math.min(5000, Number(e.target.value)));
                  lastChanged.current = 'height';
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
                min={1} // At least 1 layer
                onChange={(e) => {
                  const val = Math.max(1, Math.floor(Number(e.target.value)));
                  lastChanged.current = 'layers';
                  dispatch({ type: 'SET_NUM_LAYERS', value: val });
                }}
              />
            </label>
            <label title="Reduce geometry complexity by removing small details. 0 = no simplification.">
              Simplify shape:
              <input
                type="range"
                id="simplify"
                min="0"
                max="25"
                step="1"
                value={simplify}
                onChange={(e) => setSimplify(Number(e.target.value))}
              />
            </label>
            <label title="Smooth jagged edges with a small buffer in/out operation. 0 = no smoothing.">
              Smoothing:
              <input
                type="range"
                id="smoothing"
                min="0"
                max="200"
                step="1"
                value={smoothing}
                onChange={(e) => setSmoothing(Number(e.target.value))}
              />
            </label>
            <label>
              Substrate size (mm):
              <input
                type="number"
                id="substrate-size"
                value={params.substrateSize}
                min={10} // Minimum substrate size in mm
                onChange={(e) => dispatch({ type: 'SET_SUBSTRATE_SIZE', value: Number(e.target.value) })}
              />
            </label>
            <label>
              Layer thickness (mm):
              <input
                type="number"
                id="layer-thickness"
                value={params.layerThickness}
                min={0.1} // Minimum thickness in mm
                step={0.1}
                onChange={(e) => dispatch({ type: 'SET_LAYER_THICKNESS', value: Number(e.target.value) })}
              />
            </label>
            <label title="Remove small polygons below this area in square centimeters - measured on the scaled geometry (laser output). 0 = no filtering.">
              Minimum feature size (cm²):
              <input
                type="number"
                id="min-area"
                min="0"
                step="10"
                value={minArea}
                onChange={(e) => setMinArea(Number(e.target.value))}
              />
            </label>
            <label title="Remove narrow features (e.g., bridges, ingresses) below this width in mm. Applied after scaling. 0 = no filtering.">
              Minimum feature width (mm):
              <input
                type="number"
                id="min-feature-width"
                min="0"
                step="0.1"
                value={minFeatureWidth}
                onChange={(e) => setMinFeatureWidth(Number(e.target.value))}
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
        {/* ─────────────── Main Panel Section ───────────────
            Displays the interactive map for selecting area/bounds,
            and the 3D contour preview of the sliced output.
            Updates in response to user selection and slicing. */}
        <div className="main-panel">
          <div className="map-container">
            <MapView coordinates={coordinates} onBoundsChange={setBounds} squareOutput={params.squareOutput} />
          </div>
          <div id="preview-3d">
            <h2>3D Preview</h2>
            { slicing ? <p>⏳ Slicing in progress...</p> : <ContourPreview layers={contourLayers} /> }
          </div>
        </div>
        {/* ─────────────── Info Sidebar Section ───────────────
            Shows summary info for the selected area: center coordinates,
            width/height (meters), and elevation range (meters).
            Updated as user selects new areas or after slicing. */}
        <div className="info-sidebar">
          <h2>Area Info</h2>
          <p><strong>Center:</strong> {coordinates ? `${coordinates[0].toFixed(5)}, ${coordinates[1].toFixed(5)}` : 'N/A'}</p>
          <p><strong>Width:</strong> {areaStats ? `${areaStats.width.toFixed(0)} m` : 'N/A'}</p>
          <p><strong>Height:</strong> {areaStats ? `${areaStats.height.toFixed(0)} m` : 'N/A'}</p>
          <p><strong>Lowest Elevation:</strong> {elevationStats ? `${elevationStats.min.toFixed(0)} m` : '…'}</p>
          <p><strong>Highest Elevation:</strong> {elevationStats ? `${elevationStats.max.toFixed(0)} m` : '…'}</p>
        </div>
      </div>
      <ToastContainer
        aria-label="Notification messages"
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        pauseOnHover
      />
      {showManual && <ManualModal onClose={() => setShowManual(false)} />}
    </div>
  )
}

export default App
