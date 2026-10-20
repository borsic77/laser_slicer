export interface SwissLakeInfo {
  id: string;
  name: string;
  bbox: [number, number, number, number]; // [lon_min, lat_min, lon_max, lat_max]
  surfaceElevation: number;
}

export const SWISS_BATHY3D_LAKES: SwissLakeInfo[] = [
  { id: "swissbathy3d_aegerisee", name: "Aegerisee", bbox: [8.5846, 47.0976, 8.6516, 47.1433], surfaceElevation: 724.0 },
  { id: "swissbathy3d_baldeggersee", name: "Baldeggersee", bbox: [8.2432, 47.1774, 8.2770, 47.2174], surfaceElevation: 463.0 },
  { id: "swissbathy3d_bielersee", name: "Lake Biel (Bielersee)", bbox: [7.0695, 47.0314, 7.2413, 47.1399], surfaceElevation: 429.0 },
  { id: "swissbathy3d_bodensee", name: "Lake Constance (Bodensee)", bbox: [8.8450, 47.4674, 9.7752, 47.8240], surfaceElevation: 395.0 },
  { id: "swissbathy3d_brienzersee", name: "Lake Brienz (Brienzersee)", bbox: [7.8568, 46.6795, 8.0539, 46.7615], surfaceElevation: 564.0 },
  { id: "swissbathy3d_hallwilersee", name: "Hallwilersee", bbox: [8.1803, 47.2321, 8.2332, 47.3216], surfaceElevation: 449.0 },
  { id: "swissbathy3d_lacdejoux", name: "Lac de Joux", bbox: [6.2493, 46.6120, 6.3421, 46.6760], surfaceElevation: 1004.0 },
  { id: "swissbathy3d_lacleman", name: "Lake Geneva (Lac Leman)", bbox: [6.1223, 46.1968, 6.9334, 46.5272], surfaceElevation: 372.0 },
  { id: "swissbathy3d_lacneuchatel", name: "Lake Neuchatel", bbox: [6.6227, 46.7772, 7.0720, 47.0495], surfaceElevation: 429.0 },
  { id: "swissbathy3d_lagomaggiore", name: "Lake Maggiore", bbox: [8.3146, 45.6798, 8.9107, 46.2594], surfaceElevation: 193.0 },
  { id: "swissbathy3d_lungernsee", name: "Lungerersee", bbox: [8.1499, 46.7827, 8.1757, 46.8135], surfaceElevation: 688.0 },
  { id: "swissbathy3d_murtensee", name: "Lake Murten (Murtensee)", bbox: [7.0313, 46.9053, 7.1368, 46.9597], surfaceElevation: 429.0 },
  { id: "swissbathy3d_rotsee", name: "Rotsee", bbox: [8.2999, 47.0630, 8.3285, 47.0763], surfaceElevation: 419.0 },
  { id: "swissbathy3d_sarnersee", name: "Sarnersee", bbox: [8.1888, 46.8454, 8.2440, 46.8900], surfaceElevation: 469.0 },
  { id: "swissbathy3d_sempachersee", name: "Sempachersee", bbox: [8.1213, 47.1178, 8.1922, 47.1739], surfaceElevation: 504.0 },
  { id: "swissbathy3d_silsersee", name: "Lake Sils (Silsersee)", bbox: [9.6953, 46.4025, 9.7657, 46.4416], surfaceElevation: 1797.0 },
  { id: "swissbathy3d_silvaplanersee", name: "Lake Silvaplana (Silvaplanersee)", bbox: [9.7685, 46.4329, 9.8078, 46.4605], surfaceElevation: 1791.0 },
  { id: "swissbathy3d_thunersee", name: "Lake Thun (Thunersee)", bbox: [7.6215, 46.6535, 7.8444, 46.7531], surfaceElevation: 558.0 },
  { id: "swissbathy3d_vierwaldstaettersee", name: "Lake Lucerne (Vierwaldstaettersee)", bbox: [8.2787, 46.8884, 8.6251, 47.0822], surfaceElevation: 434.0 },
  { id: "swissbathy3d_walensee", name: "Walensee", bbox: [9.0953, 47.0798, 9.3218, 47.1575], surfaceElevation: 419.0 },
  { id: "swissbathy3d_zuerichsee", name: "Lake Zurich (Zuerichsee)", bbox: [8.5207, 47.1946, 8.8288, 47.3688], surfaceElevation: 406.0 },
  { id: "swissbathy3d_zugersee", name: "Lake Zug (Zugersee)", bbox: [8.4389, 47.0538, 8.5339, 47.1896], surfaceElevation: 413.0 },
];

export function getMatchingSwissLake(bounds: [[number, number], [number, number]]): SwissLakeInfo | null {
  const latMin = Math.min(bounds[0][0], bounds[1][0]);
  const latMax = Math.max(bounds[0][0], bounds[1][0]);
  const lonMin = Math.min(bounds[0][1], bounds[1][1]);
  const lonMax = Math.max(bounds[0][1], bounds[1][1]);

  let bestLake: SwissLakeInfo | null = null;
  let bestArea = -1;

  for (const lake of SWISS_BATHY3D_LAKES) {
    const [lMinLon, lMinLat, lMaxLon, lMaxLat] = lake.bbox;
    if (
      lonMax < lMinLon ||
      lonMin > lMaxLon ||
      latMax < lMinLat ||
      latMin > lMaxLat
    ) {
      continue;
    }
    const iMinLon = Math.max(lonMin, lMinLon);
    const iMaxLon = Math.min(lonMax, lMaxLon);
    const iMinLat = Math.max(latMin, lMinLat);
    const iMaxLat = Math.min(latMax, lMaxLat);
    const area = Math.max(0, iMaxLon - iMinLon) * Math.max(0, iMaxLat - iMinLat);
    if (area > bestArea) {
      bestArea = area;
      bestLake = lake;
    }
  }

  return bestLake;
}

export function isGreatLakes(bounds: [[number, number], [number, number]]): boolean {
  const latMin = Math.min(bounds[0][0], bounds[1][0]);
  const latMax = Math.max(bounds[0][0], bounds[1][0]);
  const lonMin = Math.min(bounds[0][1], bounds[1][1]);
  const lonMax = Math.max(bounds[0][1], bounds[1][1]);

  return (
    lonMin <= -75.8 &&
    lonMax >= -92.5 &&
    latMin <= 49.2 &&
    latMax >= 41.2
  );
}

export function isEurope(bounds: [[number, number], [number, number]]): boolean {
  const latMin = Math.min(bounds[0][0], bounds[1][0]);
  const latMax = Math.max(bounds[0][0], bounds[1][0]);
  const lonMin = Math.min(bounds[0][1], bounds[1][1]);
  const lonMax = Math.max(bounds[0][1], bounds[1][1]);

  return (
    lonMin >= -36.0 &&
    lonMax <= 43.0 &&
    latMin >= 24.0 &&
    latMax <= 72.0
  );
}

export interface BathymetryInfo {
  available: boolean;
  datasetName: string;
  description: string;
}

export function getBathymetryDatasetInfo(
  bounds: [[number, number], [number, number]] | null,
  elevationStats: { min: number; max: number } | null
): BathymetryInfo {
  if (!bounds) {
    return {
      available: false,
      datasetName: "None",
      description: "Select an area to view available bathymetry datasets.",
    };
  }

  const swissLake = getMatchingSwissLake(bounds);
  if (swissLake) {
    return {
      available: true,
      datasetName: "SwissBATHY3D",
      description: `Available: ${swissLake.name} (SwissBATHY3D 2m)`,
    };
  }

  if (isGreatLakes(bounds)) {
    return {
      available: true,
      datasetName: "NOAA Great Lakes",
      description: "Available: Great Lakes (NOAA 90m)",
    };
  }

  // If the lowest elevation in this selection is well above sea level, it is inland terrain
  if (elevationStats && elevationStats.min > 15) {
    return {
      available: false,
      datasetName: "Inland",
      description: "No lake bathymetry dataset for this inland area (use Fix Elevation for water bodies).",
    };
  }

  if (isEurope(bounds)) {
    return {
      available: true,
      datasetName: "EMODnet",
      description: "Available: EMODnet Europe (115m)",
    };
  }

  return {
    available: true,
    datasetName: "NOAA Global",
    description: "Available: NOAA Global Bathymetry (450m)",
  };
}
