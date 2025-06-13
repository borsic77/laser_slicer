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

import { useEffect, useReducer, useState } from 'react';
// Async job-based workflow state for slicing, elevation, and SVG export
import { useRef } from 'react';
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

  // If no coordinates are set, try to get current location via geolocation API
  // If geolocation fails, fallback to Mont Blanc coordinates
  // (45.832622, 6.864717) as a default location
  // This is used to center the map initially and provide a default location
  // Note: This is only run once on mount, so it won't re-trigger if coordinates change
  useEffect(() => {
    if (coordinates === null) {
      if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            setCoordinates([position.coords.latitude, position.coords.longitude]);
          },
          (error) => {
            setCoordinates([45.832622, 6.864717]); // Mont Blanc fallback
          },
          { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
        );
      } else {
        setCoordinates([45.832622, 6.864717]); // Mont Blanc fallback
      }
    }
    // eslint-disable-next-line
  }, []);

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

  // Async job handling state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<number | null>(null);
  const [jobLog, setJobLog] = useState<string>('');
  const [jobResultUrl, setJobResultUrl] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | null>(null); // For cleanup

  // Separate polling for each job type
  const elevationPollRef = useRef<number | null>(null);
  const slicingPollRef = useRef<number | null>(null);
  const exportPollRef = useRef<number | null>(null);
  const [elevationJobId, setElevationJobId] = useState<string | null>(null);
  const [slicingJobId, setSlicingJobId] = useState<string | null>(null);
  const [exportJobId, setExportJobId] = useState<string | null>(null);

  // Fixed-elevation slice controls
  const [fixMode, setFixMode] = useState(false);               // Armed when button clicked, disables after marker placed
  const [fixedElevation, setFixedElevation] = useState<number | null>(null);
  const [fixedElevationEnabled, setFixedElevationEnabled] = useState(false);
  const [waterPolygon, setWaterPolygon] = useState<any | null>(null);


  // Poll elevation job
  function pollElevationJobStatus(jobId: string) {
    if (elevationPollRef.current) clearInterval(elevationPollRef.current);
    elevationPollRef.current = window.setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/api/jobs/${jobId}/`);
        if (!res.ok) throw new Error('Failed to get elevation job status');
        const data = await res.json();
        if (data.status === "SUCCESS") {
          const result = data.params?.result;
          console.log("Elevation polling jobId=", jobId, "data=", result);
          // Only update if jobId matches the latest elevationJobId
          if (
            result &&
            typeof result.min === "number" &&
            typeof result.max === "number" 
          ) {
            console.log("Setting elevationStats to", { min: result.min, max: result.max });
            setElevationStats({ min: result.min, max: result.max });
          }
          clearInterval(elevationPollRef.current!);
          elevationPollRef.current = null;
        } else if (data.status === "FAILURE") {
          clearInterval(elevationPollRef.current!);
          elevationPollRef.current = null;
          toast.error("Elevation job failed: " + (data.log || 'Unknown error'));
        }
      } catch (err) {
        clearInterval(elevationPollRef.current!);
        elevationPollRef.current = null;
        toast.error("Error polling elevation job status");
      }
    }, 2000);
  }

  // Poll slicing job
  function pollSlicingJobStatus(jobId: string) {
    if (slicingPollRef.current) clearInterval(slicingPollRef.current);
    slicingPollRef.current = window.setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/api/jobs/${jobId}/`);
        if (!res.ok) throw new Error('Failed to get slicing job status');
        const data = await res.json();
        setJobStatus(data.status);
        setJobProgress(data.progress);
        setJobLog(data.log || '');
        setJobResultUrl(data.result_url || null);
        if (data.status === "SUCCESS") {
          console.log("Slicing job layers:", data.params?.layers);
          if (data.params && data.params.layers) {
            setContourLayers(data.params.layers);
          }
          setSliced(true);
          setSlicing(false);
          clearInterval(slicingPollRef.current!);
          slicingPollRef.current = null;
          toast.success("Slicing done!");
        } else if (data.status === "FAILURE") {
          setSlicing(false);
          clearInterval(slicingPollRef.current!);
          slicingPollRef.current = null;
          toast.error("Slicing failed: " + (data.log || 'Unknown error'));
        }
      } catch (err) {
        setSlicing(false);
        clearInterval(slicingPollRef.current!);
        slicingPollRef.current = null;
        toast.error("Error polling slicing job status");
      }
    }, 2000);
  }

  // Poll export job
  function pollExportJobStatus(jobId: string) {
    if (exportPollRef.current) clearInterval(exportPollRef.current);
    exportPollRef.current = window.setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/api/jobs/${jobId}/`);
        if (!res.ok) throw new Error('Failed to get export job status');
        const data = await res.json();
        setJobStatus(data.status);
        setJobProgress(data.progress);
        setJobLog(data.log || '');
        setJobResultUrl(data.result_url || null);
        if (data.status === "SUCCESS") {
          clearInterval(exportPollRef.current!);
          exportPollRef.current = null;
          toast.success("Export ready! Download ZIP below.");
        } else if (data.status === "FAILURE") {
          clearInterval(exportPollRef.current!);
          exportPollRef.current = null;
          toast.error("Export job failed: " + (data.log || 'Unknown error'));
        }
      } catch (err) {
        clearInterval(exportPollRef.current!);
        exportPollRef.current = null;
        toast.error("Error polling export job status");
      }
    }, 2000);
  }

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
   * Submits an elevation job to the backend and polls for min/max results.
   * Updates elevationStats on success.
   */
  async function fetchElevationRange(bounds: [[number, number], [number, number]], signal?: AbortSignal) {
    const body = {
      bounds: {
        lat_min: bounds[0][0],
        lon_min: bounds[0][1],
        lat_max: bounds[1][0],
        lon_max: bounds[1][1],
      }
    };
    const res = await fetch(`${API_URL}/api/elevation-range/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw new Error("Failed to submit elevation job");
    const data = await res.json();
    if (!data.job_id) throw new Error("Invalid job response for elevation");

    // Ensure elevationJobId is set right before polling
    pollElevationJobStatus(data.job_id);    
    setElevationJobId(data.job_id);

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
   * React effect: When elevationStats or numLayers change, synchronize heightPerLayer.
   * Triggers: runs whenever elevationStats or params.numLayers change.
   * Side effects: Updates heightPerLayer based on numLayers and elevationStats.
   */
  useEffect(() => {
    if (!elevationStats) return;
    const range = elevationStats.max - elevationStats.min;
    const newHeight = Math.max(10, Math.min(5000, range / params.numLayers));
    if (newHeight !== params.heightPerLayer) {
      dispatch({ type: 'SET_HEIGHT_PER_LAYER', value: newHeight });
    }
  }, [elevationStats, params.numLayers]);

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
   * Handler: Starts an async contour slicing job and polls for result.
   */
  const handleSlice = async () => {
    if (!coordinates) {
      toast.warn("Please select a location first.");
      return;
    }
    if (!bounds) {
      toast.warn("Could not read selected bounds.");
      return;
    }
    const body = {
      height_per_layer: params.heightPerLayer,
      num_layers: params.numLayers,
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
      fixedElevation: fixedElevationEnabled ? fixedElevation : undefined,
      waterPolygon: waterPolygon ?? undefined,
      waterElevation: fixedElevationEnabled ? fixedElevation ?? undefined : undefined,

    };
    if (fixedElevationEnabled && typeof fixedElevation === 'number') {
      body.fixedElevation = fixedElevation;  
    }      
    try {
      setSlicing(true);
      setContourLayers([]); // clear old result
      const res = await fetch(`${API_URL}/api/slice/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error('Failed to start slicing job');
      const data = await res.json();
      setSlicingJobId(data.job_id);
      setJobId(data.job_id);
      setJobStatus('PENDING');
      setJobProgress(0);
      setJobLog('');
      setJobResultUrl(null);
      pollSlicingJobStatus(data.job_id);
    } catch (error) {
      setSlicing(false);
      toast.error("Failed to start slicing job");
    }
  };

  /**
   * Handler: Starts an async SVG export job and polls for ZIP download URL.
   */
  const handleExport = async () => {
    if (!contourLayers.length) {
      toast.warn("No contours to export.");
      return;
    }
    try {
      const res = await fetch(`${API_URL}/api/export/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layers: contourLayers,
          address,
          coordinates,
          height_per_layer: params.heightPerLayer,
        }),
      });
      if (!res.ok) throw new Error('Failed to start export job');
      const data = await res.json();
      setExportJobId(data.job_id);
      setJobId(data.job_id);
      setJobStatus('PENDING');
      setJobProgress(0);
      setJobLog('');
      setJobResultUrl(null);
      pollExportJobStatus(data.job_id);
    } catch (error) {
      toast.error("Failed to start export job");
    }
  };
  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (elevationPollRef.current) clearInterval(elevationPollRef.current);
      if (slicingPollRef.current) clearInterval(slicingPollRef.current);
      if (exportPollRef.current) clearInterval(exportPollRef.current);
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current); // legacy cleanup
    };
  }, []);

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
          <button onClick={() => setShowManual(true)}>Show Help</button>
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
              Number of layers:
              <input
                type="number"
                id="num-layers"
                value={params.numLayers}
                min={0} // At least 1 layer
                onChange={(e) => {
                  const val = Math.max(1, Math.floor(Number(e.target.value)));
                  dispatch({ type: 'SET_NUM_LAYERS', value: val });
                }}
              />
            </label>
            <label>
              Height per layer:
              <span style={{ marginLeft: '0.5em', fontWeight: 'bold' }}>{params.heightPerLayer.toFixed(1)}</span>
              m<br />
              <br />
            </label>  

              <button
                onClick={() => setFixMode(true)}
                disabled={fixMode}
                title="Click, then place a marker on the map to sample elevation."
                style={{ marginBottom: '0.5em' }}
              >
                {fixMode ? "Select on map..." : "Fix Elevation (lake)"}
              </button>

              <input
                type="number"
                min={elevationStats?.min ?? undefined}
                max={elevationStats?.max ?? undefined}
                step="10"
                placeholder="Elevation (m)"
                value={fixedElevation ?? ''}
                disabled={!fixedElevationEnabled && !fixMode}
                onChange={e => {
                  const v = Number(e.target.value);
                  setFixedElevation(isNaN(v) ? null : v);
                  setFixedElevationEnabled(true);
                }}

              />
              <label style={{ display: "block" }}>
                <input
                  type="checkbox"
                  checked={fixedElevationEnabled}
                  onChange={e => setFixedElevationEnabled(e.target.checked)}
                  disabled={fixedElevation === null}
                />{" "}
                Enable fixed elevation
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
            <button id="export-button" onClick={handleExport} disabled={!sliced || slicing}>Export SVGs</button>
            {jobResultUrl && (
              <a id="download-link" href={API_URL + jobResultUrl} download>
                ⬇️ Download ZIP
              </a>
            )}
            {jobStatus && (
              <div style={{ margin: "0.5em 0" }}>
                <strong>Status:</strong> {jobStatus}<br />
                {jobProgress !== null && <progress value={jobProgress} max={100}>{jobProgress}%</progress>}
                {jobLog && (
                  <pre style={{ fontSize: 'smaller', maxHeight: 100, overflow: 'auto' }}>{jobLog}</pre>
                )}
              </div>
            )}
          </div>
        </div>
        {/* ─────────────── Main Panel Section ───────────────
            Displays the interactive map for selecting area/bounds,
            and the 3D contour preview of the sliced output.
            Updates in response to user selection and slicing. */}
        <div className="main-panel">
          <div className="map-container">
          <MapView
            coordinates={coordinates}
            onBoundsChange={setBounds}
            squareOutput={params.squareOutput}
            fixMode={fixMode}
            setFixMode={setFixMode}
            onFixedElevation={async (lat: number, lon: number) => {
              setFixMode(false);
              try {
                const resp = await fetch(`${API_URL}/api/water-info/?lat=${lat}&lon=${lon}`);
                if (!resp.ok) throw new Error("Failed to fetch water info");
                const data = await resp.json();
                setFixedElevation(data.elevation);
                setWaterPolygon(data.geometry);
                setFixedElevationEnabled(true);

                if (elevationStats && (data.elevation < elevationStats.min || data.elevation > elevationStats.max)) {
                  const msg = `Elevation ${data.elevation}m is outside area bounds (${elevationStats.min}–${elevationStats.max})`;
                  toast.error(msg);
                }
              } catch (err: any) {
                toast.error("Elevation lookup failed");
                setFixedElevation(null);
                setFixedElevationEnabled(false);
              }
            }}
          />
          </div>
          <div id="preview-3d">
            <h2>3D Preview</h2>
            { slicing ? <p>⏳ Slicing in progress...</p> : (contourLayers.length > 0 ? <ContourPreview layers={contourLayers} /> : <p>No contours available.</p>) }
          </div>
        </div>
        {/* ─────────────── Info Sidebar Section ───────────────
            Shows summary info for the selected area: center coordinates,
            width/height (meters), and elevation range (meters).
            Updated as user selects new areas or after slicing. */}
        {(() => {
          console.log("Rendering sidebar, elevationStats =", elevationStats);
          return null;
        })()}
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
      <footer style={{ textAlign: 'center', padding: '1rem', fontSize: '0.9rem', color: '#777' }}>
        © {new Date().getFullYear()} Boris Legradic · <a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener noreferrer">MIT License</a>
      </footer>
    </div>
  )
}

export default App
