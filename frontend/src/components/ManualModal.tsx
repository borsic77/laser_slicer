// src/components/ManualModal.tsx
import './ManualModal.css';

function ManualModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>User Manual</h2>
        <p style={{ textAlign: 'left' }}>Welcome to the Laser Contour Map Generator.</p>
        <p style={{ textAlign: 'left' }}><strong>Step 1:</strong> Enter a location and click "Locate on Map".</p>
        <p style={{ textAlign: 'left' }}><strong>Step 2:</strong> Adjust slicing parameters to match your project.</p>
        <p style={{ textAlign: 'left' }}><strong>Step 3:</strong> Click "Slice!" and wait for the preview.</p>
        <p style={{ textAlign: 'left' }}><strong>Step 4:</strong> Click "Export SVGs" to download a ZIP for laser cutting.</p>
        <p style={{ textAlign: 'left' }}><strong>Advanced Options:</strong></p>
        <p style={{ textAlign: 'left' }}><strong>Shape Simplification:</strong> Reduces the number of points per contour line using the Ramer–Douglas–Peucker algorithm. Higher values remove more detail but make laser cutting faster.</p>
        <p style={{ textAlign: 'left' }}><strong>Smoothing:</strong> Applies a smoothing filter to the elevation data to reduce sharp edges or noise in the contours. Useful for irregular terrain.</p>
        <p style={{ textAlign: 'left' }}><strong>Minimum Area:</strong> Filters out small polygons below a given area threshold, helping to clean up tiny fragments that might not be meaningful or cuttable.</p>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default ManualModal;