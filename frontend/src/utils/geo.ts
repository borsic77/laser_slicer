/**
 * Compute the physical width and height (in meters) of a rectangular area given by bounds.
 * Uses the equirectangular approximation for small areas.
 */
export function getWidthHeightMeters(bounds: [[number, number], [number, number]]): { width: number; height: number } {
    const [latMin, lonMin] = bounds[0];
    const [latMax, lonMax] = bounds[1];

    const R = 6371000;
    const φ1 = (latMin * Math.PI) / 180;
    const φ2 = (latMax * Math.PI) / 180;
    const Δφ = φ2 - φ1;
    const Δλ = ((lonMax - lonMin) * Math.PI) / 180;
    const φm = (φ1 + φ2) / 2;

    const height = R * Δφ;
    const width = R * Δλ * Math.cos(φm);

    return {
        width: Math.abs(width),
        height: Math.abs(height),
    };
}
