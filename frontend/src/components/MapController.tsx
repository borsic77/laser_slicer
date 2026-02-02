import { toast } from 'react-toastify';
import { api } from '../services/api';
import MapView from './Mapview';

interface MapControllerProps {
    coordinates: [number, number] | null;
    setBounds: (bounds: [[number, number], [number, number]] | null) => void;
    squareOutput: boolean;
    fixMode: boolean;
    setFixMode: (mode: boolean) => void;
    setFixedElevation: (val: number) => void;
    setFixedElevationEnabled: (val: boolean) => void;
    setWaterPolygon: (poly: any) => void;
}

export default function MapController(props: MapControllerProps) {
    const { 
        coordinates, setBounds, squareOutput, 
        fixMode, setFixMode, 
        setFixedElevation, setFixedElevationEnabled, setWaterPolygon 
    } = props;

    const handleFixedElevation = async (lat: number, lon: number) => {
        setFixMode(false);
        try {
            const water = await api.fetchWaterBody(lat, lon);
            const elevation = await api.fetchElevationAtPoint(lat, lon);
            
            setFixedElevation(elevation);
            setFixedElevationEnabled(true);
            
            if (water && water.in_water) {
                setWaterPolygon(water.polygon);
            } else {
                setWaterPolygon(null);
            }
        } catch (error) {
            toast.error("Failed to fetch elevation info");
        }
    };

    return (
        <MapView
            coordinates={coordinates}
            onBoundsChange={setBounds}
            squareOutput={squareOutput}
            fixMode={fixMode}
            setFixMode={setFixMode}
            onFixedElevation={handleFixedElevation}
        />
    );
}
