import { useEffect, useRef, useState } from 'react';
import { toast } from 'react-toastify';
import { api, ExportJobParams, SliceJobParams } from '../services/api';

export function useSlicingJob() {
    const [jobId, setJobId] = useState<string | null>(null);
    const [jobStatus, setJobStatus] = useState<string | null>(null);
    const [jobProgress, setJobProgress] = useState<number | null>(null);
    const [jobLog, setJobLog] = useState<string>('');
    const [jobResultUrl, setJobResultUrl] = useState<string | null>(null);
    
    // Slicing specific state
    const [slicing, setSlicing] = useState(false);
    const [sliced, setSliced] = useState(false);
    const [contourLayers, setContourLayers] = useState<any[]>([]);

    const slicingPollRef = useRef<number | null>(null);
    const exportPollRef = useRef<number | null>(null);

    const pollSlicingJobStatus = (id: string) => {
        if (slicingPollRef.current) clearInterval(slicingPollRef.current);
        slicingPollRef.current = window.setInterval(async () => {
            try {
                const data = await api.getJobStatus(id);
                setJobStatus(data.status);
                setJobProgress(data.progress);
                setJobLog(data.log || '');
                setJobResultUrl(data.result_url || null);

                if (data.status === "SUCCESS") {
                    if (data.params && data.params.layers) {
                        setContourLayers(data.params.layers);
                    }
                    setSliced(true);
                    setSlicing(false);
                    clearInterval(slicingPollRef.current!);
                    slicingPollRef.current = null;
                    toast.success("Slicing done!");
                } else if (data.status === "FAILURE") {
                    setSlicing(false);
                    clearInterval(slicingPollRef.current!);
                    slicingPollRef.current = null;
                    toast.error("Slicing failed: " + (data.log || 'Unknown error'));
                }
            } catch (err) {
                setSlicing(false);
                clearInterval(slicingPollRef.current!);
                slicingPollRef.current = null;
                toast.error("Error polling slicing job status");
            }
        }, 2000);
    };

    const pollExportJobStatus = (id: string) => {
        if (exportPollRef.current) clearInterval(exportPollRef.current);
        exportPollRef.current = window.setInterval(async () => {
             try {
                const data = await api.getJobStatus(id);
                setJobStatus(data.status);
                setJobProgress(data.progress);
                setJobLog(data.log || '');
                setJobResultUrl(data.result_url || null);
                
                if (data.status === "SUCCESS") {
                    clearInterval(exportPollRef.current!);
                    exportPollRef.current = null;
                    toast.success("Export ready! Download ZIP below.");
                } else if (data.status === "FAILURE") {
                    clearInterval(exportPollRef.current!);
                    exportPollRef.current = null;
                    toast.error("Export job failed: " + (data.log || 'Unknown error'));
                }
            } catch (err) {
                clearInterval(exportPollRef.current!);
                exportPollRef.current = null;
                toast.error("Error polling export job status");
            }
        }, 2000);
    };

    const startSliceJob = async (params: SliceJobParams) => {
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
        } catch (error) {
            setSlicing(false);
            toast.error("Failed to start slicing job");
        }
    };

    const startExportJob = async (params: ExportJobParams) => {
         try {
            const data = await api.startExportJob(params);
            setJobId(data.job_id);
            setJobStatus('PENDING');
            setJobProgress(0);
            setJobLog('');
            setJobResultUrl(null);
            pollExportJobStatus(data.job_id);
        } catch (error) {
            toast.error("Failed to start export job");
        }
    };

    // Cleanup
    useEffect(() => {
        return () => {
            if (slicingPollRef.current) clearInterval(slicingPollRef.current);
            if (exportPollRef.current) clearInterval(exportPollRef.current);
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
