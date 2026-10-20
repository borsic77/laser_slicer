import { useEffect, useRef, useState } from 'react';
import { toast } from 'react-toastify';
import type { ContourLayer } from '../components/ContourPreview';
import { api, type ExportJobParams, type SliceJobParams } from '../services/api';

export function useSlicingJob() {
    const [jobId, setJobId] = useState<string | null>(null);
    const [jobStatus, setJobStatus] = useState<string | null>(null);
    const [jobProgress, setJobProgress] = useState<number | null>(null);
    const [jobLog, setJobLog] = useState<string>('');
    const [jobResultUrl, setJobResultUrl] = useState<string | null>(null);
    
    // Slicing specific state
    const [slicing, setSlicing] = useState(false);
    const [sliced, setSliced] = useState(false);
    const [contourLayers, setContourLayers] = useState<ContourLayer[]>([]);

    const slicingPollRef = useRef<number | null>(null);
    const exportPollRef = useRef<number | null>(null);
    const activeSliceJobIdRef = useRef<string | null>(null);
    const activeExportJobIdRef = useRef<string | null>(null);
    const slicingInFlightRef = useRef<boolean>(false);
    const exportInFlightRef = useRef<boolean>(false);

    const clearExistingIntervals = () => {
        if (slicingPollRef.current) {
            clearInterval(slicingPollRef.current);
            slicingPollRef.current = null;
        }
        if (exportPollRef.current) {
            clearInterval(exportPollRef.current);
            exportPollRef.current = null;
        }
        activeSliceJobIdRef.current = null;
        activeExportJobIdRef.current = null;
        slicingInFlightRef.current = false;
        exportInFlightRef.current = false;
    };

    const pollSlicingJobStatus = (id: string) => {
        if (slicingPollRef.current) {
            clearInterval(slicingPollRef.current);
            slicingPollRef.current = null;
        }
        activeSliceJobIdRef.current = id;
        slicingInFlightRef.current = false;

        slicingPollRef.current = window.setInterval(async () => {
            if (activeSliceJobIdRef.current !== id) {
                return;
            }
            if (slicingInFlightRef.current) {
                return;
            }
            slicingInFlightRef.current = true;
            try {
                const data = await api.getJobStatus(id);
                if (activeSliceJobIdRef.current !== id) {
                    return;
                }
                setJobStatus(data.status);
                setJobProgress(data.progress ?? null);
                setJobLog(data.log || '');
                setJobResultUrl(data.result_url || null);

                if (data.status === "SUCCESS") {
                    if (data.params && Array.isArray(data.params.layers)) {
                        setContourLayers(data.params.layers as ContourLayer[]);
                    }
                    setSliced(true);
                    setSlicing(false);
                    if (slicingPollRef.current) {
                        clearInterval(slicingPollRef.current);
                        slicingPollRef.current = null;
                    }
                    toast.success("Slicing done!");
                } else if (data.status === "FAILURE") {
                    setSlicing(false);
                    if (slicingPollRef.current) {
                        clearInterval(slicingPollRef.current);
                        slicingPollRef.current = null;
                    }
                    toast.error("Slicing failed: " + (data.log || 'Unknown error'));
                }
            } catch {
                if (activeSliceJobIdRef.current !== id) {
                    return;
                }
                setSlicing(false);
                if (slicingPollRef.current) {
                    clearInterval(slicingPollRef.current);
                    slicingPollRef.current = null;
                }
                toast.error("Error polling slicing job status");
            } finally {
                slicingInFlightRef.current = false;
            }
        }, 2000);
    };

    const pollExportJobStatus = (id: string) => {
        if (exportPollRef.current) {
            clearInterval(exportPollRef.current);
            exportPollRef.current = null;
        }
        activeExportJobIdRef.current = id;
        exportInFlightRef.current = false;

        exportPollRef.current = window.setInterval(async () => {
            if (activeExportJobIdRef.current !== id) {
                return;
            }
            if (exportInFlightRef.current) {
                return;
            }
            exportInFlightRef.current = true;
            try {
                const data = await api.getJobStatus(id);
                if (activeExportJobIdRef.current !== id) {
                    return;
                }
                setJobStatus(data.status);
                setJobProgress(data.progress ?? null);
                setJobLog(data.log || '');
                setJobResultUrl(data.result_url || null);
                
                if (data.status === "SUCCESS") {
                    if (exportPollRef.current) {
                        clearInterval(exportPollRef.current);
                        exportPollRef.current = null;
                    }
                    toast.success("Export ready! Download ZIP below.");
                } else if (data.status === "FAILURE") {
                    if (exportPollRef.current) {
                        clearInterval(exportPollRef.current);
                        exportPollRef.current = null;
                    }
                    toast.error("Export job failed: " + (data.log || 'Unknown error'));
                }
            } catch {
                if (activeExportJobIdRef.current !== id) {
                    return;
                }
                if (exportPollRef.current) {
                    clearInterval(exportPollRef.current);
                    exportPollRef.current = null;
                }
                toast.error("Error polling export job status");
            } finally {
                exportInFlightRef.current = false;
            }
        }, 2000);
    };

    const startSliceJob = async (params: SliceJobParams) => {
        clearExistingIntervals();
        try {
            setSlicing(true);
            setContourLayers([]);
            const data = await api.startSliceJob(params);
            setJobId(data.job_id);
            setJobStatus('PENDING');
            setJobProgress(0);
            setJobLog('');
            setJobResultUrl(null);
            pollSlicingJobStatus(data.job_id);
        } catch {
            setSlicing(false);
            toast.error("Failed to start slicing job");
        }
    };

    const startExportJob = async (params: ExportJobParams) => {
        clearExistingIntervals();
        try {
            const data = await api.startExportJob(params);
            setJobId(data.job_id);
            setJobStatus('PENDING');
            setJobProgress(0);
            setJobLog('');
            setJobResultUrl(null);
            pollExportJobStatus(data.job_id);
        } catch {
            toast.error("Failed to start export job");
        }
    };

    // Cleanup
    useEffect(() => {
        return () => {
            clearExistingIntervals();
        };
    }, []);

    return {
        startSliceJob,
        startExportJob,
        jobId,
        jobStatus,
        jobProgress,
        jobLog,
        jobResultUrl,
        slicing,
        sliced,
        contourLayers
    };
}
