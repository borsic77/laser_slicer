import { useRef, useState } from 'react'
import './App.css'
import ContourPreview from './components/ContourPreview'
import MapView from './components/MapView'

function App() {
  const [address, setAddress] = useState('')
  const [coordinates, setCoordinates] = useState<[number, number] | null>(null)
  const [sliced, setSliced] = useState(false)
  const [contourLayers, setContourLayers] = useState<any[]>([])
  const boundsRef = useRef<[[number, number], [number, number]] | null>(null)

  async function fetchCoordinates(address: string): Promise<[number, number]> {
    const res = await fetch('http://localhost:8000/api/geocode/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address })
    })

    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Failed to fetch coordinates')
    return [data.lat, data.lon]
  }

  const handleGeocode = async () => {
    try {
      const coords = await fetchCoordinates(address)
      setCoordinates(coords)
    } catch (error) {
      alert('Geocoding failed: ' + error)
    }
  }

  const handleSlice = async () => {
    if (!coordinates) {
      alert("Please select a location first.")
      return
    }

    const height = Number((document.getElementById('layer-height') as HTMLInputElement).value)
    const numLayers = Number((document.getElementById('num-layers') as HTMLInputElement).value)
    const simplify = Number((document.getElementById('simplify') as HTMLInputElement).value)

    const bounds = boundsRef.current
    if (!bounds) {
      alert("Could not read map bounds.")
      return
    }

    const body = {
      lat: coordinates[0],
      lon: coordinates[1],
      height_per_layer: height,
      num_layers: numLayers,
      simplify: simplify,
      bounds: {
        lat_min: bounds[0][0],
        lon_min: bounds[0][1],
        lat_max: bounds[1][0],
        lon_max: bounds[1][1],
      }
    }

    try {
      const res = await fetch('http://localhost:8000/api/slice/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      if (!res.ok) throw new Error('Failed to slice contour data')

      const data = await res.json()
      console.log("Received contour layers:", data.layers)
      setContourLayers(data.layers || [])
      setSliced(true)
      alert('Slicing complete! Ready to export.')
    } catch (error) {
      alert("Slicing failed: " + error)
    }
  }

  const handleExport = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/export/', {
        method: 'GET',
      })

      if (!res.ok) throw new Error('Failed to export contours')

      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.getElementById('download-link') as HTMLAnchorElement
      link.href = url
      link.download = 'contours.zip'
      link.style.display = 'inline-block'
    } catch (error) {
      alert("Export failed: " + error)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Laser-Cuttable Contour Map Generator</h1>
      </header>

      <div className="controls">
        <input
          type="text"
          placeholder="Enter address or location..."
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        />
        <button onClick={handleGeocode}>Locate on Map</button>
      </div>

      <div className="main-content">
        <div className="map-container">
          <MapView coordinates={coordinates} boundsRef={boundsRef} />
        </div>

        <div className="parameters">
          <label>
            Height per layer (m):
            <input type="number" id="layer-height" defaultValue={10} />
          </label>
          <label>
            Number of layers:
            <input type="number" id="num-layers" defaultValue={5} />
          </label>
          <label>
            Simplify shape:
            <input type="range" id="simplify" min="0" max="1" step="0.05" />
          </label>
          <button id="slice-button" onClick={handleSlice}>Slice!</button>
          <button id="export-button" onClick={handleExport} disabled={!sliced}>Export SVGs</button>
          <a id="download-link" href="#" download style={{ display: 'none' }}>
            ⬇️ Download ZIP
          </a>
        </div>
      </div>

      <div id="preview-3d">
        <h2>3D Preview</h2>
        <ContourPreview layers={contourLayers} />
      </div>
    </div>
  )
}

export default App
