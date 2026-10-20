import { useEffect, useRef, useState } from 'react';
import { toast } from 'react-toastify';
import { api } from '../services/api';
import { getWidthHeightMeters } from '../utils/geo';

export function useElevationJob(bounds: [[number, number], [number, number]] | null, includeBathymetry: boolean) {
    const [areaStats, setAreaStats] = useState<{ width: number; height: number } | null>(null);
    const [elevationStats, setElevationStats] = useState<{ min: number; max: number } | null>(null);
    const elevationPollRef = useRef<number | null>(null);
    const activeJobIdRef = useRef<string | null>(null);
    const inFlightRef = useRef<boolean>(false);

    // Poll elevation job
    const pollElevationJobStatus = (jobId: string) => {
        if (elevationPollRef.current) {
            clearInterval(elevationPollRef.current);
            elevationPollRef.current = null;
        }
        activeJobIdRef.current = jobId;
        inFlightRef.current = false;

        elevationPollRef.current = window.setInterval(async () => {
            if (activeJobIdRef.current !== jobId) {
                return;
            }
            if (inFlightRef.current) {
                return;
            }
            inFlightRef.current = true;
            try {
                const data = await api.getJobStatus(jobId);
                if (activeJobIdRef.current !== jobId) {
                    return;
                }
                if (data.status === "SUCCESS") {
                    const result = data.params?.result;
                    if (result && typeof result.min === "number" && typeof result.max === "number") {
                        setElevationStats({ min: result.min, max: result.max });
                    }
                    if (elevationPollRef.current) {
                        clearInterval(elevationPollRef.current);
                        elevationPollRef.current = null;
                    }
                } else if (data.status === "FAILURE") {
                    if (elevationPollRef.current) {
                        clearInterval(elevationPollRef.current);
                        elevationPollRef.current = null;
                    }
                    toast.error("Elevation job failed: " + (data.log || 'Unknown error'));
                }
            } catch (err) {
                if (activeJobIdRef.current !== jobId) {
                    return;
                }
                console.error(err);
                if (elevationPollRef.current) {
                    clearInterval(elevationPollRef.current);
                    elevationPollRef.current = null;
                }
                // Silent fail on polling error to avoid spam
            } finally {
                inFlightRef.current = false;
            }
        }, 2000);
    };

    useEffect(() => {
        if (elevationPollRef.current) {
            clearInterval(elevationPollRef.current);
            elevationPollRef.current = null;
        }
        activeJobIdRef.current = null;

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
            activeJobIdRef.current = null;
            if (elevationPollRef.current) {
                clearInterval(elevationPollRef.current);
                elevationPollRef.current = null;
            }
        };
    }, [bounds, includeBathymetry]);

    return { areaStats, elevationStats };
}
