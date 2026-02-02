interface InfoSidebarProps {
    coordinates: [number, number] | null;
    areaStats: { width: number; height: number } | null;
    elevationStats: { min: number; max: number } | null;
    
    includeBathymetry: boolean;
    setIncludeBathymetry: (v: boolean) => void;
    
    includeRoads: boolean;
    setIncludeRoads: (v: boolean) => void;
    includeWaterways: boolean;
    setIncludeWaterways: (v: boolean) => void;
    includeBuildings: boolean;
    setIncludeBuildings: (v: boolean) => void;
}

export default function InfoSidebar(props: InfoSidebarProps) {
    const {
        coordinates, areaStats, elevationStats,
        includeBathymetry, setIncludeBathymetry,
        includeRoads, setIncludeRoads,
        includeWaterways, setIncludeWaterways,
        includeBuildings, setIncludeBuildings
    } = props;

    return (
        <div className="info-sidebar">
          <h2>Area Info</h2>
          <p><strong>Center:</strong> {coordinates ? `${coordinates[0].toFixed(5)}, ${coordinates[1].toFixed(5)}` : 'N/A'}</p>
          <p><strong>Width:</strong> {areaStats ? `${areaStats.width.toFixed(0)} m` : 'N/A'}</p>
          <p><strong>Height:</strong> {areaStats ? `${areaStats.height.toFixed(0)} m` : 'N/A'}</p>
          <p><strong>Lowest Elevation:</strong> {elevationStats ? `${elevationStats.min.toFixed(0)} m` : '…'}</p>
          <p><strong>Highest Elevation:</strong> {elevationStats ? `${elevationStats.max.toFixed(0)} m` : '…'}</p>
          <hr style={{ width: '100%', border: 'none', borderTop: '1px solid #ccc', margin: '1em 0' }} />
          
          <h2>Bathymetry</h2>
          <label style={{display:'block'}}>
            <input type="checkbox" checked={includeBathymetry} onChange={e => setIncludeBathymetry(e.target.checked)} /> Include Ocean Data
          </label>

          <hr style={{ width: '100%', border: 'none', borderTop: '1px solid #ccc', margin: '1em 0' }} />

          <h2>OSM features</h2>
          <label style={{display:'block'}}>
            <input type="checkbox" checked={includeRoads} onChange={e => setIncludeRoads(e.target.checked)} /> Roads
          </label>
          <label style={{display:'block'}}>
            <input type="checkbox" checked={includeWaterways} onChange={e => setIncludeWaterways(e.target.checked)} /> Waterways
          </label>
          <label style={{display:'block'}}>
            <input type="checkbox" checked={includeBuildings} onChange={e => setIncludeBuildings(e.target.checked)} /> Buildings
          </label>
        </div>
    );
}
