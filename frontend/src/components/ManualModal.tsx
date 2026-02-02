// src/components/ManualModal.tsx
import { useEffect, useRef } from 'react';
import './ManualModal.css';

interface ManualModalProps {
  onClose: () => void;
}

function ManualModal({ onClose }: ManualModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Trap focus inside modal
  useEffect(() => {
    if (modalRef.current) {
      modalRef.current.focus();
    }
  }, []);

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div 
        className="modal-content" 
        onClick={(e) => e.stopPropagation()} 
        ref={modalRef} 
        tabIndex={-1}
      >
        <div className="modal-header">
          <h2 id="modal-title">User Manual & Guide</h2>
          <button className="close-button-top" onClick={onClose} aria-label="Close modal">
            &times;
          </button>
        </div>

        <div className="modal-body">
          <p>Welcome to the <strong>Laser Contour Map Generator</strong>. Create precise, layer-by-layer scale models of any terrain on Earth.</p>
          
          <div className="step-block">
            <p><strong>1. Locate & Frame</strong><br/>
            Search for a location (e.g., "Grand Canyon") or use the locate button. Zoom and pan the map to frame exactly the area you want to slice.</p>
          </div>

          <div className="step-block">
            <p><strong>2. Configure Model Params</strong><br/>
            Set your physical constraints:</p>
            <ul>
                <li><strong>Layers & Height</strong>: Control the vertical resolution. <br/><em>Tip:</em> Standard SRTM data is ~30m resolution. For finer detail (&lt;10m layers), try locations in Switzerland (uses 2m SwissALTI3D).</li>
                <li><strong>Substrate</strong>: The size of your laser cutter bed or material sheet (e.g., 400mm).</li>
                <li><strong>Thickness</strong>: The exact thickness of your plywood or acrylic (e.g., 3mm).</li>
            </ul>
          </div>

          <div className="step-block">
            <p><strong>3. Advanced Features</strong><br/>
            Add detail to your model:</p>
            <ul>
              <li><strong>Bathymetry (Ocean Data)</strong>: Enable this to include underwater terrain. Uses global ETOPO data for deep oceans and trenches.</li>
              <li><strong>Fixed Elevation (Lakes)</strong>: For flat water bodies, use the "Fix Elevation" tool. Click a lake on the map to force a perfectly flat surface at that altitude.</li>
              <li><strong>Overlay Features</strong>: Engrave Roads, Buildings, or Waterways directly onto the corresponding layer surfaces.</li>
            </ul>
          </div>

          <div className="step-block">
            <p><strong>4. Slice & Export</strong><br/>
            Hit <strong>Slice!</strong> to generate the 3D preview. Review the layers, then click <strong>Export</strong> to get a ZIP file of optimized SVGs.</p>
          </div>

          <div className="advanced-section">
            <h3>Tuning & Optimization</h3>
            <p><strong>Simplify:</strong> Reduces the number of nodes in the vector paths. Increase this if your laser cutter chokes on complex curves.</p>
            <p><strong>Smoothing:</strong> Applies a Gaussian blur to the raw terrain data before slicing. Essential for reducing "stepped" artifacts in noisy data.</p>
            <p><strong>Min Area/Width:</strong> Automatically removes tiny islands or fragile bridges that are too small to cut reliably.</p>
          </div>

          <div className="advanced-section">
            <h3>SVG Layer Guide</h3>
            <ul className="legend-list">
              <li className="legend-item">
                <span className="legend-swatch" style={{color: 'black', border: '1px solid #ccc'}}>Black</span>
                <span><strong>Cut</strong> (The boundary of the layer)</span>
              </li>
              <li className="legend-item">
                <span className="legend-swatch" style={{color: 'red', border: '1px solid #ccc'}}>Red</span>
                <span><strong>Score/Engrave</strong> (Alignment guide from the layer above)</span>
              </li>
              <li className="legend-item">
                <span className="legend-swatch" style={{color: 'blue', border: '1px solid #ccc'}}>Blue</span>
                <span><strong>Labels</strong> (Layer ID numbers for assembly)</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="modal-footer">
          <button className="close-button-bottom" onClick={onClose}>Close Manual</button>
        </div>
      </div>
    </div>
  );
}

export default ManualModal;