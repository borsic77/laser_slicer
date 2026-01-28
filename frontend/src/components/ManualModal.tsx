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

  // Trap focus inside modal (simple implementation)
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
          <h2 id="modal-title">User Manual</h2>
          <button className="close-button-top" onClick={onClose} aria-label="Close modal">
            &times;
          </button>
        </div>

        <div className="modal-body">
          <p>Welcome to the <strong>Laser Contour Map Generator</strong>. Turn real-world terrain into laser-cuttable art.</p>
          
          <div className="step-block">
            <p><strong>Step 1: Locate</strong><br/>
            Enter a location in the search bar (e.g., "Zermatt") or use your current location. Move the map to frame your desired area.</p>
          </div>

          <div className="step-block">
            <p><strong>Step 2: Configure Slicing</strong><br/>
            Adjust the physical parameters for your model:</p>
            <ul>
                <li><strong>Number of Layers</strong> / <strong>Height per Layer</strong>: The more layers, the more detailed the model. However beware: SRTM data has a resolution of 30m, so layer heights smaller than 30 m will only work reliably in Switzerland.</li>
                <li><strong>Substrate & Thickness</strong>: Set these to match your material (e.g., 4mm plywood).</li>
                <li><strong>Square Output</strong>: Forces the slicing area to be a perfect square, regardless of your screen's aspect ratio.</li>
            </ul>
          </div>

          <div className="step-block">
            <p><strong>Step 3: Add Features (Optional)</strong><br/>
            Enhance your map with OpenStreetMap data:</p>
            <ul>
              <li><strong>Roads, Buildings, Waterways</strong>: Check these boxes in the sidebar to engrave these features on the corresponding layers.</li>
              <li><strong>Fix Elevation (Water Body)</strong>: Use this for lakes or seas. Click the button, then click on the water body on the map. This forces a flat slice at the water's surface level, fixing noisy data artifacts.</li>
            </ul>
            <p><em>Note: High-resolution SwissALTI3D data (2m) is automatically used for locations in Switzerland. SRTM (30m) is used globally.</em></p>
          </div>

          <div className="step-block">
            <p><strong>Step 4: Generate & Export</strong><br/>
            Click <strong>Slice!</strong> to generate the 3D preview. Once satisfied, click <strong>Export</strong> to download a ZIP file containing optimized SVGs for each layer.</p>
          </div>

          <div className="advanced-section">
            <h3>Advanced Options</h3>
            <p><strong>Simplify:</strong> Reduces geometric complexity (Ramer–Douglas–Peucker). Higher values = faster cutting, less detail.</p>
            <p><strong>Smoothing:</strong> Smooths out jagged terrain and noise from raw elevation data.</p>
            <p><strong>Minimum Area:</strong> Removes tiny polygon islands that are too small to cut or handle.</p>
            <p><strong>Minimum Width:</strong> Removes features thinner than this value (mm) to prevent fragile parts.</p>
          </div>

          <div className="advanced-section">
            <h3>SVG Color Legend</h3>
            <ul className="legend-list">
              <li className="legend-item">
                <span className="legend-swatch" style={{color: 'black', border: '1px solid #ccc'}}>Black</span>
                <span><strong>Cut lines</strong> (Primary contour shape)</span>
              </li>
              <li className="legend-item">
                <span className="legend-swatch" style={{color: 'red', border: '1px solid #ccc'}}>Red</span>
                <span><strong>Alignment</strong> (Outline of the layer above, for easy stacking)</span>
              </li>
              <li className="legend-item">
                <span className="legend-swatch" style={{color: 'blue', border: '1px solid #ccc'}}>Blue</span>
                <span><strong>Visible Labels</strong> (Engrave these for reference)</span>
              </li>
              <li className="legend-item">
                <span className="legend-swatch" style={{color: 'green', border: '1px solid #ccc'}}>Green</span>
                <span><strong>Hidden Labels</strong> (Covered by the layer above)</span>
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