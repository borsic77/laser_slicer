import { describe, expect, it } from 'vitest';
import { initialSlicerParams, slicerReducer } from './reducers/slicerReducer';
import {
  getBathymetryDatasetInfo,
  getMatchingSwissLake,
  isEurope,
  isGreatLakes,
} from './utils/bathymetry';
import { getWidthHeightMeters } from './utils/geo';

describe('slicerReducer', () => {
  it('updates substrateSize with SET_SUBSTRATE_SIZE', () => {
    const nextState = slicerReducer(initialSlicerParams, {
      type: 'SET_SUBSTRATE_SIZE',
      value: 600,
    });
    expect(nextState.substrateSize).toBe(600);
    expect(nextState.layerThickness).toBe(initialSlicerParams.layerThickness);
  });

  it('updates layerThickness with SET_LAYER_THICKNESS', () => {
    const nextState = slicerReducer(initialSlicerParams, {
      type: 'SET_LAYER_THICKNESS',
      value: 10,
    });
    expect(nextState.layerThickness).toBe(10);
  });

  it('updates squareOutput with SET_SQUARE_OUTPUT', () => {
    const nextState = slicerReducer(initialSlicerParams, {
      type: 'SET_SQUARE_OUTPUT',
      value: false,
    });
    expect(nextState.squareOutput).toBe(false);
  });

  it('updates heightPerLayer with SET_HEIGHT_PER_LAYER', () => {
    const nextState = slicerReducer(initialSlicerParams, {
      type: 'SET_HEIGHT_PER_LAYER',
      value: 100,
    });
    expect(nextState.heightPerLayer).toBe(100);
  });

  it('updates numLayers with SET_NUM_LAYERS', () => {
    const nextState = slicerReducer(initialSlicerParams, {
      type: 'SET_NUM_LAYERS',
      value: 12,
    });
    expect(nextState.numLayers).toBe(12);
  });

  it('returns current state for unhandled action type', () => {
    // @ts-expect-error test invalid action type fallback
    const nextState = slicerReducer(initialSlicerParams, { type: 'UNKNOWN_ACTION' });
    expect(nextState).toEqual(initialSlicerParams);
  });
});

describe('getWidthHeightMeters', () => {
  it('computes width and height in meters for valid coordinates', () => {
    const bounds: [[number, number], [number, number]] = [
      [46.7, 6.5],
      [46.9, 6.7],
    ];
    const { width, height } = getWidthHeightMeters(bounds);
    expect(width).toBeGreaterThan(0);
    expect(height).toBeGreaterThan(0);
    // 0.2 deg lat is approx 22.2 km
    expect(height).toBeCloseTo(22239, -2);
    // 0.2 deg lon at ~46.8N is approx 15.2 km
    expect(width).toBeCloseTo(15230, -2);
  });

  it('returns positive dimensions regardless of coordinate order', () => {
    const bounds: [[number, number], [number, number]] = [
      [10, 20],
      [5, 15],
    ];
    const { width, height } = getWidthHeightMeters(bounds);
    expect(width).toBeGreaterThan(0);
    expect(height).toBeGreaterThan(0);
  });
});

describe('bathymetry utilities', () => {
  it('identifies Swiss lakes and returns SwissBATHY3D info', () => {
    // Rotsee coordinates: lat 47.07, lon 8.31
    const rotseeBounds: [[number, number], [number, number]] = [
      [47.065, 8.305],
      [47.075, 8.325],
    ];
    const lake = getMatchingSwissLake(rotseeBounds);
    expect(lake).not.toBeNull();
    expect(lake?.name).toBe('Rotsee');
    expect(lake?.surfaceElevation).toBe(419.0);

    const info = getBathymetryDatasetInfo(rotseeBounds, { min: 410, max: 550 });
    expect(info.available).toBe(true);
    expect(info.datasetName).toBe('SwissBATHY3D');
    expect(info.description).toContain('Rotsee (SwissBATHY3D 2m)');
  });

  it('identifies Great Lakes and returns NOAA Great Lakes info', () => {
    // Lake Michigan coordinates: lat 42.0, lon -87.0
    const michiganBounds: [[number, number], [number, number]] = [
      [41.5, -87.5],
      [42.5, -86.5],
    ];
    expect(isGreatLakes(michiganBounds)).toBe(true);

    const info = getBathymetryDatasetInfo(michiganBounds, { min: 100, max: 200 });
    expect(info.available).toBe(true);
    expect(info.datasetName).toBe('NOAA Great Lakes');
    expect(info.description).toContain('Great Lakes (NOAA 90m)');
  });

  it('identifies inland mountain areas with no bathymetry dataset', () => {
    // French Alps / Mont Blanc: lat 45.83, lon 6.86, min elev ~1000m
    const alpsBounds: [[number, number], [number, number]] = [
      [45.8, 6.8],
      [45.9, 6.9],
    ];
    expect(isEurope(alpsBounds)).toBe(true);
    const info = getBathymetryDatasetInfo(alpsBounds, { min: 800, max: 4810 });
    expect(info.available).toBe(false);
    expect(info.description).toContain('No lake bathymetry dataset for this inland area');
  });

  it('identifies European coastal waters as EMODnet', () => {
    // Mediterranean coast: lat 43.2, lon 5.3, min elev -200m
    const coastBounds: [[number, number], [number, number]] = [
      [43.1, 5.2],
      [43.3, 5.4],
    ];
    const info = getBathymetryDatasetInfo(coastBounds, { min: -150, max: 50 });
    expect(info.available).toBe(true);
    expect(info.datasetName).toBe('EMODnet');
    expect(info.description).toContain('EMODnet Europe (115m)');
  });

  it('identifies oceanic areas outside Europe as NOAA Global', () => {
    // Hawaii / Pacific: lat 21.3, lon -157.8, min elev -3000m
    const pacificBounds: [[number, number], [number, number]] = [
      [21.0, -158.0],
      [21.5, -157.5],
    ];
    const info = getBathymetryDatasetInfo(pacificBounds, { min: -4000, max: 200 });
    expect(info.available).toBe(true);
    expect(info.datasetName).toBe('NOAA Global');
    expect(info.description).toContain('NOAA Global Bathymetry (450m)');
  });
});
