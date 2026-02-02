
const API_URL = import.meta.env.VITE_API_URL;

function getCookie(name: string): string {
  const match = document.cookie.match('(^|;)\\s*' + name + '=([^;]*)');
  return match ? decodeURIComponent(match[2]) : '';
}

function fetchWithCsrf(input: RequestInfo | URL, init: RequestInit = {}) {
  const headers = {
    'X-CSRFToken': getCookie('csrftoken'),
    ...(init.headers || {})
  } as HeadersInit;
  return fetch(input, { ...init, credentials: 'include', headers });
}

export interface GeocodeResult {
    lat: number;
    lon: number;
    display_name?: string;
}

export interface ElevationRangeJobParams {
    bounds: {
        lat_min: number;
        lon_min: number;
        lat_max: number;
        lon_max: number;
    };
    include_bathymetry: boolean;
}

export interface SliceJobParams {
    height_per_layer: number;
    num_layers: number;
    simplify: number;
    smoothing: number;
    min_area: number;
    min_feature_width: number;
    bounds: {
        lat_min: number;
        lon_min: number;
        lat_max: number;
        lon_max: number;
    };
    substrate_size: number;
    layer_thickness: number;
    fixedElevation?: number;
    water_polygon?: any;
    include_roads: boolean;
    include_buildings: boolean;
    include_waterways: boolean;
    include_bathymetry: boolean;
}

export interface ExportJobParams {
    layers: any[];
    address: string;
    coordinates: [number, number];
    height_per_layer: number;
}

export const api = {
    baseUrl: API_URL,

    async fetchCoordinates(address: string, signal?: AbortSignal): Promise<[number, number]> {
        const res = await fetchWithCsrf(`${API_URL}/api/geocode/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address }),
            signal,
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to fetch coordinates');
        return [data.lat, data.lon];
    },

    async fetchElevationRange(params: ElevationRangeJobParams, signal?: AbortSignal): Promise<{ job_id: string }> {
        const res = await fetchWithCsrf(`${API_URL}/api/elevation-range/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
            signal,
        });
        if (!res.ok) throw new Error("Failed to submit elevation job");
        const data = await res.json();
        if (!data.job_id) throw new Error("Invalid job response for elevation");
        return { job_id: data.job_id };
    },

    async startSliceJob(params: SliceJobParams): Promise<{ job_id: string }> {
        const res = await fetchWithCsrf(`${API_URL}/api/slice/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!res.ok) throw new Error('Failed to start slicing job');
        return await res.json();
    },

    async startExportJob(params: ExportJobParams): Promise<{ job_id: string }> {
        const res = await fetchWithCsrf(`${API_URL}/api/export/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!res.ok) throw new Error('Failed to start export job');
        return await res.json();
    },

    async getJobStatus(jobId: string): Promise<any> {
        const res = await fetch(`${API_URL}/api/jobs/${jobId}/`);
        if (!res.ok) throw new Error(`Failed to get job status for ${jobId}`);
        return await res.json();
    },

    async fetchWaterBody(lat: number, lon: number): Promise<any | null> {
         try {
            const wres = await fetch(`${API_URL}/api/waterbody/?lat=${lat}&lon=${lon}`);
            if (wres.ok) {
            return await wres.json();
            }
        } catch {}
        return null;
    },

    async fetchElevationAtPoint(lat: number, lon: number): Promise<number> {
        const resp = await fetch(`${API_URL}/api/elevation?lat=${lat}&lon=${lon}`);
        if (!resp.ok) throw new Error("Failed to fetch elevation");
        const { elevation } = await resp.json();
        return elevation;
    }
};
