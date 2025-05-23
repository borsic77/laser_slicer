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
        <p style={{ textAlign: 'left' }}><strong>Step 2a (optional):</strong> Set the <strong>Height per Layer</strong> (vertical thickness of each slice) or the <strong>Number of Layers</strong>. Changing one will automatically update the other to cover the entire detected elevation range of your selected area.</p>
        <p style={{ textAlign: 'left' }}><strong>Step 3:</strong> Click "Slice!" and wait for the preview.</p>
        <p style={{ textAlign: 'left' }}><strong>Step 4:</strong> Click "Export SVGs" to download a ZIP for laser cutting.</p>
        <p style={{ textAlign: 'left' }}><strong>Advanced Options:</strong></p>
        <p style={{ textAlign: 'left' }}><strong>Shape Simplification:</strong> Reduces the number of points per contour line using the Ramer–Douglas–Peucker algorithm. Higher values remove more detail but make laser cutting faster.</p>
        <p style={{ textAlign: 'left' }}><strong>Smoothing:</strong> Applies a smoothing filter to the elevation data to reduce sharp edges or noise in the contours. Useful for irregular terrain.</p>
        <p style={{ textAlign: 'left' }}><strong>Minimum Area:</strong> Filters out small polygons below a given area threshold, helping to clean up tiny fragments that might not be meaningful or cuttable.</p>
        <p style={{ textAlign: 'left' }}><strong>Minimum Feature Width:</strong> Removes narrow bridges, protrusions, or gaps below the set width (in mm) from all layers. This helps ensure your exported shapes are robust enough for laser cutting and avoids fragile or uncuttable details. Increasing this value makes sure every cuttable piece is at least as wide as the minimum width.</p>
        <h3 style={{ marginTop: '2em' }}>SVG Color Legend</h3>
        <ul style={{ textAlign: 'left', listStyleType: 'disc', paddingLeft: '1.5em' }}>
          <li>
            <span style={{ backgroundColor: 'white', padding: '0 4px', color: 'black', fontWeight: 500 }}><strong>Black</strong></span>: <em>Cut lines</em> for each slice.
          </li>
          <li>
            <span style={{ backgroundColor: 'white', padding: '0 4px', color: 'red', fontWeight: 500 }}><strong>Red</strong></span>: <em>Alignment outlines</em> of the layer above, used for stacking and assembly guidance.
          </li>
          <li>
            <span style={{ backgroundColor: 'white', padding: '0 4px', color: 'green', fontWeight: 500 }}><strong>Green</strong></span>: <em>Hidden labels</em> (format: #layerNumber_distinctFeatureNumber), drawn beneath a layer so they are visible only during assembly.
          </li>
          <li>
            <span style={{ backgroundColor: 'white', padding: '0 4px', color: 'blue', fontWeight: 500 }}><strong>Blue</strong></span>: <em>Visible labels</em> in the final model (not covered by any layer above) — you may want to avoid engraving these.
          </li>
        </ul>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default ManualModal;