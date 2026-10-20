import { describe, expect, it } from 'vitest';
import { initialSlicerParams, slicerReducer } from './reducers/slicerReducer';
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
