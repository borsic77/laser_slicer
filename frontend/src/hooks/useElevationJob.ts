import { useEffect, useRef, useState } from 'react';
import { toast } from 'react-toastify';
import { api } from '../services/api';
import { getWidthHeightMeters } from '../utils/geo';

export function useElevationJob(bounds: [[number, number], [number, number]] | null, includeBathymetry: boolean) {
    const [areaStats, setAreaStats] = useState<{ width: number; height: number } | null>(null);
    const [elevationStats, setElevationStats] = useState<{ min: number; max: number } | null>(null);
    const [elevationJobId, setElevationJobId] = useState<string | null>(null);
    const elevationPollRef = useRef<number | null>(null);

    // Poll elevation job
    const pollElevationJobStatus = (jobId: string) => {
        if (elevationPollRef.current) clearInterval(elevationPollRef.current);
        elevationPollRef.current = window.setInterval(async () => {
            try {
                const data = await api.getJobStatus(jobId);
                if (data.status === "SUCCESS") {
                    const result = data.params?.result;
                    if (result && typeof result.min === "number" && typeof result.max === "number") {
                        setElevationStats({ min: result.min, max: result.max });
                    }
                    clearInterval(elevationPollRef.current!);
                    elevationPollRef.current = null;
                } else if (data.status === "FAILURE") {
                    clearInterval(elevationPollRef.current!);
                    elevationPollRef.current = null;
                    toast.error("Elevation job failed: " + (data.log || 'Unknown error'));
                }
            } catch (err) {
                console.error(err);
                clearInterval(elevationPollRef.current!);
                elevationPollRef.current = null;
                // Silent fail on polling error to avoid spam
            }
        }, 2000);
    };

    useEffect(() => {
        if (!bounds) return;
        const dims = getWidthHeightMeters(bounds);
        setAreaStats(dims);
        const controller = new AbortController();

        const fetchData = async () => {
            try {
                const { job_id } = await api.fetchElevationRange({
                    bounds: {
                        lat_min: bounds[0][0],
                        lon_min: bounds[0][1],
                        lat_max: bounds[1][0],
                        lon_max: bounds[1][1],
                    },
                    include_bathymetry: includeBathymetry,
                }, controller.signal);
                
                setElevationJobId(job_id);
                pollElevationJobStatus(job_id);
            } catch (err) {
                if (err instanceof Error && err.name !== "AbortError") {
                    console.error("Elevation range error:", err);
                }
            }
        };

        fetchData();

        return () => {
            controller.abort();
            if (elevationPollRef.current) clearInterval(elevationPollRef.current);
        };
    }, [bounds, includeBathymetry]);

    return { areaStats, elevationStats };
}
