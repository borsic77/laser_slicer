import React from 'react';
import { Action, SlicerParams } from '../reducers/slicerReducer';

const API_URL = import.meta.env.VITE_API_URL;

interface SidebarProps {
    address: string;
    setAddress: (addr: string) => void;
    handleGeocode: () => void;
    
    params: SlicerParams;
    dispatch: React.Dispatch<Action>;
    
    fixMode: boolean;
    setFixMode: (mode: boolean) => void;
    fixedElevation: number | null;
    setFixedElevation: (val: number | null) => void;
    fixedElevationEnabled: boolean;
    setFixedElevationEnabled: (enabled: boolean) => void;
    
    simplify: number;
    setSimplify: (val: number) => void;
    smoothing: number;
    setSmoothing: (val: number) => void;
    minArea: number;
    setMinArea: (val: number) => void;
    minFeatureWidth: number;
    setMinFeatureWidth: (val: number) => void;
    
    elevationStats: { min: number; max: number } | null;
    
    handleSlice: () => void;
    handleExport: () => void;
    
    slicing: boolean;
    sliced: boolean;
    
    jobStatus: string | null;
    jobProgress: number | null;
    jobLog: string;
    jobResultUrl: string | null;
    
    setShowManual: (show: boolean) => void;
}

export default function Sidebar(props: SidebarProps) {
    const {
        address, setAddress, handleGeocode,
        params, dispatch,
        fixMode, setFixMode,
        fixedElevation, setFixedElevation,
        fixedElevationEnabled, setFixedElevationEnabled,
        simplify, setSimplify,
        smoothing, setSmoothing,
        minArea, setMinArea,
        minFeatureWidth, setMinFeatureWidth,
        elevationStats,
        handleSlice, handleExport,
        slicing, sliced,
        jobStatus, jobProgress, jobLog, jobResultUrl,
        setShowManual
    } = props;

    return (
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
                min={0}
                onChange={(e) => {
                  const val = Math.max(1, Math.floor(Number(e.target.value)));
                  dispatch({ type: 'SET_NUM_LAYERS', value: val });
                }}
              />
            </label>
            <div style={{ fontSize: '0.9rem', color: 'var(--color-text)', opacity: 1, marginBottom: '15px' }}>
               Height per layer: <strong>{params.heightPerLayer.toFixed(1)} m</strong>
            </div>  

              <button
                onClick={() => setFixMode(true)}
                disabled={fixMode}
                title="Click, then place a marker on the map to sample elevation."
                style={{ marginBottom: '5px' }}
              >
                {fixMode ? "Select on map..." : "Fix Elevation (water body)"}
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
              <label style={{ display: "flex", flexDirection: 'row', alignItems: 'center', marginTop: '5px' }}>
                <input
                  type="checkbox"
                  checked={fixedElevationEnabled}
                  onChange={e => setFixedElevationEnabled(e.target.checked)}
                  disabled={fixedElevation === null}
                  style={{ margin: 0 }}
                />{" "}
                Enable fixed elevation
              </label>              
                  
            <label title="Reduce geometry complexity">
              Simplify:
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
            <label title="Smooth jagged edges">
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
            <div className="section-group">
                <label title="Substrate size in mm">
                  Substrate (mm):
                  <input
                    type="number"
                    id="substrate-size"
                    value={params.substrateSize}
                    min={10} 
                    onChange={(e) => dispatch({ type: 'SET_SUBSTRATE_SIZE', value: Number(e.target.value) })}
                  />
                </label>
                <label title="Layer thickness in mm">
                  Thickness (mm):
                  <input
                    type="number"
                    id="layer-thickness"
                    value={params.layerThickness}
                    min={0.1}
                    step={0.1}
                    onChange={(e) => dispatch({ type: 'SET_LAYER_THICKNESS', value: Number(e.target.value) })}
                  />
                </label>
            </div>

            <div className="section-group">
                <label title="Remove features smaller than this area (cm²)">
                  Min Area (cm²):
                  <input
                    type="number"
                    id="min-area"
                    min="0"
                    step="10"
                    value={minArea}
                    onChange={(e) => setMinArea(Number(e.target.value))}
                  />
                </label>
                <label title="Remove features thinner than this width (mm)">
                  Min Width (mm):
                  <input
                    type="number"
                    id="min-feature-width"
                    min="0"
                    step="0.1"
                    value={minFeatureWidth}
                    onChange={(e) => setMinFeatureWidth(Number(e.target.value))}
                  />
                </label>
            </div>
            <label style={{ flexDirection: 'row', alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={params.squareOutput}
                onChange={(e) => dispatch({ type: 'SET_SQUARE_OUTPUT', value: e.target.checked })}
                style={{ margin: 0 }}
              />
              Square output
            </label>
            <hr style={{ width: '100%', border: 'none', borderTop: '1px solid #ccc', margin: '0.5em 0' }} />
            <div className="button-group">
              <button id="slice-button" onClick={handleSlice}>Slice!</button>
              <button id="export-button" onClick={handleExport} disabled={!sliced || slicing}>Export</button>
            </div>
            {jobResultUrl && (
              <a id="download-link" href={API_URL + jobResultUrl} download>
                ⬇️ Download ZIP
              </a>
            )}
            {jobStatus && (
              <div style={{ margin: "0.5em 0" }}>
                <div style={{ marginBottom: '5px', fontWeight: 'bold' }}>
                    {jobLog ? jobLog.split('\n').filter(Boolean).pop()?.replace(/^\[\d+%\]\s*/, '') : jobStatus}
                </div>
                {jobProgress !== null && <progress value={jobProgress} max={100} style={{ width: '100%' }}>{jobProgress}%</progress>}
                {jobLog && (
                  <details>
                    <summary style={{fontSize: '0.8rem', cursor: 'pointer', color: '#888'}}>Show Full Log</summary>
                    <pre style={{ fontSize: 'smaller', maxHeight: 100, overflow: 'auto', marginTop: '5px' }}>{jobLog}</pre>
                  </details>
                )}
              </div>
            )}
          </div>
        </div>
    );
}
