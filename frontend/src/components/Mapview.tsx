import 'leaflet/dist/leaflet.css'
import { useEffect, useState } from 'react'
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet'

interface MapViewProps {
  coordinates: [number, number] | null
  boundsRef?: React.MutableRefObject<[[number, number], [number, number]] | null>
}

function RecenterMap({ coordinates }: { coordinates: [number, number] }) {
  const map = useMap()
  useEffect(() => {
    map.setView(coordinates)
  }, [coordinates, map])
  return null
}

export default function MapView({ coordinates, boundsRef }: MapViewProps) {
  const [position, setPosition] = useState<[number, number]>(
    coordinates ?? [46.78, 6.64]
  ) // Default: Yverdon

  useEffect(() => {
    if (coordinates) {
      setPosition(coordinates)
    }
  }, [coordinates])

  function LocationMarker() {
    useMapEvents({
      click(e) {
        setPosition([e.latlng.lat, e.latlng.lng])
      },
    })

    return <Marker position={position} />
  }

  function TrackBounds({ boundsRef }: { boundsRef?: React.MutableRefObject<[[number, number], [number, number]] | null> }) {
    const map = useMap()

    useEffect(() => {
      if (boundsRef) {
        const update = () => {
          boundsRef.current = [
            [map.getBounds().getSouthWest().lat, map.getBounds().getSouthWest().lng],
            [map.getBounds().getNorthEast().lat, map.getBounds().getNorthEast().lng]
          ]
        }
        update()
        map.on('moveend', update)
        return () => {
          map.off('moveend', update)
        }
      }
    }, [boundsRef, map])

    return null
  }

  return (
    <MapContainer
      center={position}
      zoom={13}
      scrollWheelZoom={true}
      style={{ height: '400px', width: '100%' }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {coordinates && <RecenterMap coordinates={coordinates} />}
      <LocationMarker />
      <TrackBounds boundsRef={boundsRef} />
    </MapContainer>
  )
}