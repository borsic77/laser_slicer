/**
 * App.tsx
 *
 * Main application component for the Laser Contour Map Generator.
 * Refactored to use modular components, services, and hooks.
 */

import React, { useEffect, useReducer, useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

import ContourPreview from './components/ContourPreview';
import InfoSidebar from './components/InfoSidebar';
import ManualModal from './components/ManualModal';
import MapController from './components/MapController';
import Sidebar from './components/Sidebar';

import { useElevationJob } from './hooks/useElevationJob';
import { useSlicingJob } from './hooks/useSlicingJob';
import { initialSlicerParams, slicerReducer } from './reducers/slicerReducer';
import { api } from './services/api';

const MapControllerMemo = React.memo(MapController);

function App() {
  // Theme state
  const [darkMode, setDarkMode] = useState(() => {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
       return true;
    }
    return true; 
  });

  useEffect(() => {
    const handler = (e: MediaQueryListEvent) => setDarkMode(e.matches);
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', handler);
    return () => media.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.body.style.backgroundColor = '#242424';
      document.body.style.color = 'rgba(255, 255, 255, 0.87)';
    } else {
      document.body.style.backgroundColor = '#ffffff';
      document.body.style.color = '#213547';
    }
  }, [darkMode]);

  // Core State
  const [address, setAddress] = useState('')
  const [coordinates, setCoordinates] = useState<[number, number] | null>(null)
  const [bounds, setBounds] = useState<[[number, number], [number, number]] | null>(null)
  
  // Params State
  const [params, dispatch] = useReducer(slicerReducer, initialSlicerParams);
  const [simplify, setSimplify] = useState(0);
  const [smoothing, setSmoothing] = useState(5);
  const [minArea, setMinArea] = useState(0);
  const [minFeatureWidth, setMinFeatureWidth] = useState(0);
  
  // Layer Toggles
  const [includeBathymetry, setIncludeBathymetry] = useState(false);
  const [includeRoads, setIncludeRoads] = useState(false);
  const [includeBuildings, setIncludeBuildings] = useState(false);
  const [includeWaterways, setIncludeWaterways] = useState(false);

  // Fixed Elevation State
  const [fixMode, setFixMode] = useState(false);
  const [fixedElevation, setFixedElevation] = useState<number | null>(null);
  const [fixedElevationEnabled, setFixedElevationEnabled] = useState(false);
  const [waterPolygon, setWaterPolygon] = useState<any | null>(null);

  // Reset fixed elevation when bounds change
  useEffect(() => {
    if (!bounds) return;
    setFixedElevation(null);
    setFixedElevationEnabled(false);
    setWaterPolygon(null);
  }, [bounds]);

  // Hooks
  const { areaStats, elevationStats } = useElevationJob(bounds, includeBathymetry);
  const { 
    startSliceJob, startExportJob, 
    jobStatus, jobProgress, jobLog, jobResultUrl, 
    slicing, sliced, contourLayers 
  } = useSlicingJob();

  // Sync Height per Layer with Elevation Stats
  useEffect(() => {
    if (!elevationStats) return;
    const range = elevationStats.max - elevationStats.min;
    const newHeight = Math.max(10, Math.min(5000, range / params.numLayers));
    if (newHeight !== params.heightPerLayer) {
      dispatch({ type: 'SET_HEIGHT_PER_LAYER', value: newHeight });
    }
  }, [elevationStats, params.numLayers]);

  // Handlers
  const handleGeocode = () => {
    const controller = new AbortController();
    api.fetchCoordinates(address, controller.signal)
      .then(setCoordinates)
      .catch((error) => {
        if (error instanceof Error && error.name !== 'AbortError') {
          toast.error('Geocoding failed: ' + error.message);
        } else {
          toast.error('Geocoding failed');
        }
      });
  }

  // Initial Geolocation
  useEffect(() => {
    if (coordinates === null) {
      if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(
          p => setCoordinates([p.coords.latitude, p.coords.longitude]),
          () => setCoordinates([45.832622, 6.864717]),
          { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
        );
      } else {
        setCoordinates([45.832622, 6.864717]);
      }
    }
    // eslint-disable-next-line
  }, []);

  const handleSlice = () => {
    if (!coordinates || !bounds) {
      toast.warn("Please select a location first.");
      return;
    }

    startSliceJob({
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
      fixedElevation: fixedElevationEnabled && typeof fixedElevation === 'number' ? fixedElevation : undefined,
      water_polygon: waterPolygon ?? undefined,
      include_roads: includeRoads,
      include_buildings: includeBuildings,
      include_waterways: includeWaterways,
      include_bathymetry: includeBathymetry,
    });
  };

  const handleExport = () => {
    if (!contourLayers.length || !coordinates) {
        toast.warn("No contours to export.");
        return;
    }
    startExportJob({
        layers: contourLayers,
        address,
        coordinates,
        height_per_layer: params.heightPerLayer,
    });
  };

  const [showManual, setShowManual] = useState(false);

  return (
    <div className={`container ${darkMode ? 'dark-mode' : 'light-mode'}`} style={{
        '--color-background': darkMode ? '#242424' : '#ffffff',
        '--color-text': darkMode ? 'rgba(255, 255, 255, 0.87)' : '#213547',
        '--color-sidebar-bg': darkMode ? '#1a1a1a' : '#f0f0f0',
        '--color-sidebar-text': darkMode ? '#ffffff' : '#000000',
        '--color-input-bg': darkMode ? '#333' : '#fff',
        '--color-input-text': darkMode ? '#fff' : '#000',
        '--color-border': darkMode ? '#444' : '#ccc',
    } as React.CSSProperties}>
      
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>Laser Contour Map Generator</h2>
        <button 
           onClick={() => setDarkMode(!darkMode)}
           style={{ padding: '5px 10px', fontSize: '0.8rem', width: 'auto' }}
        >
          {darkMode ? '☀️ Light' : '🌙 Dark'}
        </button>
      </header>

      <div className="content-wrapper">
        <Sidebar 
          address={address} setAddress={setAddress} handleGeocode={handleGeocode}
          params={params} dispatch={dispatch}
          fixMode={fixMode} setFixMode={setFixMode}
          fixedElevation={fixedElevation} setFixedElevation={setFixedElevation}
          fixedElevationEnabled={fixedElevationEnabled} setFixedElevationEnabled={setFixedElevationEnabled}
          simplify={simplify} setSimplify={setSimplify}
          smoothing={smoothing} setSmoothing={setSmoothing}
          minArea={minArea} setMinArea={setMinArea}
          minFeatureWidth={minFeatureWidth} setMinFeatureWidth={setMinFeatureWidth}
          elevationStats={elevationStats}
          handleSlice={handleSlice} handleExport={handleExport}
          slicing={slicing} sliced={sliced}
          jobStatus={jobStatus} jobProgress={jobProgress} jobLog={jobLog} jobResultUrl={jobResultUrl}
          setShowManual={setShowManual}
        />

        <div className="main-panel">
          <div className="map-container">
            <MapControllerMemo
                coordinates={coordinates}
                setBounds={setBounds}
                squareOutput={params.squareOutput}
                fixMode={fixMode}
                setFixMode={setFixMode}
                setFixedElevation={(val) => { setFixedElevation(val); setFixedElevationEnabled(true); }}
                setFixedElevationEnabled={setFixedElevationEnabled}
                setWaterPolygon={setWaterPolygon}
            />
          </div>
          <div id="preview-3d">
            <h2>3D Preview</h2>
            { slicing ? <p>⏳ Slicing in progress...</p> : (contourLayers.length > 0 ? <ContourPreview layers={contourLayers} darkMode={darkMode} /> : <p>No contours available.</p>) }
          </div>
        </div>

        <InfoSidebar
            coordinates={coordinates}
            bounds={bounds}
            areaStats={areaStats}
            elevationStats={elevationStats}
            includeBathymetry={includeBathymetry} setIncludeBathymetry={setIncludeBathymetry}
            includeRoads={includeRoads} setIncludeRoads={setIncludeRoads}
            includeWaterways={includeWaterways} setIncludeWaterways={setIncludeWaterways}
            includeBuildings={includeBuildings} setIncludeBuildings={setIncludeBuildings}
        />
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
        © {new Date().getFullYear()} Boris Legradic ·
        <a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener noreferrer" style={{ marginLeft: '0.5em' }}>
          MIT License
        </a> ·
        <a href="https://legradic.ch" target="_blank" rel="noopener noreferrer" style={{ marginLeft: '0.5em' }}>
          legradic.ch
        </a> ·
        <a href="https://github.com/borsic77/laser_slicer" target="_blank" rel="noopener noreferrer" style={{ marginLeft: '0.5em' }}>
          GitHub
        </a> ·
        <a href="mailto:info@legradic.ch" style={{ marginLeft: '0.5em' }}>
          info@legradic.ch
        </a>
      </footer>
    </div>
  )
}

export default App
