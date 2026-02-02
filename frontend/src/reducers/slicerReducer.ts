/**
 * SlicerParams
 * Holds all user-configurable parameters for slicing.
 */
export type SlicerParams = {
    /** Final output side length in mm (for laser substrate) */
    substrateSize: number;
    /** Thickness of each cut layer in mm */
    layerThickness: number;
    /** If true, output will be forced square; otherwise, matches selected bounds aspect */
    squareOutput: boolean;
    /** Meters of terrain height per layer (auto-updates numLayers if changed) */
    heightPerLayer: number;
    /** Number of layers to slice (auto-updates heightPerLayer if changed) */
    numLayers: number;
  };
  
  export type Action =
    | { type: 'SET_SUBSTRATE_SIZE'; value: number }
    | { type: 'SET_LAYER_THICKNESS'; value: number }
    | { type: 'SET_SQUARE_OUTPUT'; value: boolean }
    | { type: 'SET_HEIGHT_PER_LAYER'; value: number }
    | { type: 'SET_NUM_LAYERS'; value: number };
  
  export const initialSlicerParams: SlicerParams = {
    substrateSize: 400,
    layerThickness: 5,
    squareOutput: true,
    heightPerLayer: 250,
    numLayers: 5,
  };
  
  /**
   * Reducer for slicer parameters.
   * Used instead of useState to keep interdependent fields (heightPerLayer, numLayers, etc) in sync and to batch updates atomically.
   */
  export function slicerReducer(state: SlicerParams, action: Action): SlicerParams {
    switch (action.type) {
      case 'SET_SUBSTRATE_SIZE':
        return { ...state, substrateSize: action.value };
      case 'SET_LAYER_THICKNESS':
        return { ...state, layerThickness: action.value };
      case 'SET_SQUARE_OUTPUT':
        return { ...state, squareOutput: action.value };
      case 'SET_HEIGHT_PER_LAYER':
        return { ...state, heightPerLayer: action.value };
      case 'SET_NUM_LAYERS':
        return { ...state, numLayers: action.value };
      default:
        return state;
    }
  }
