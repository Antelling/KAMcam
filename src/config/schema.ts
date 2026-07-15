export type SystemType = 'rigid' | 'elastic' | 'nonlinear' | 'sculpture' | 'resonant';
export type VizMode = 'distance' | 'divergence' | 'divergenceDistance' | 'position' | 'neighborDistance' | 'neighborDistanceAccumulated';
export type PerturbDistribution = 'uniform' | 'gaussian';
export type Colormap = 0 | 1 | 2 | 3 | 4 | 5 | 6;
export type ToneMapping = 0 | 1 | 2 | 3;
export type Resolution = 256 | 512 | 1024 | 2048 | 4096;
export type ChunkSize = 256 | 512 | 1024 | 2048;

export type PhaseSpaceDimension =
  | 'angle1'
  | 'velocity1'
  | 'angle2'
  | 'velocity2'
  | 'stretch1'
  | 'stretchRate1'
  | 'stretch2'
  | 'stretchRate2';

export interface PhaseSpaceAxis {
  dimension: PhaseSpaceDimension;
  min: number;
  max: number;
}

export type PhaseSpaceMode = 'manual' | 'tiling';

export interface TileConfig {
  cols: number;
  rows: number;
  toroidal: boolean;
  controlNet: number[][][];
}

export interface PhaseSpaceConfig {
  x: PhaseSpaceAxis;
  y: PhaseSpaceAxis;
  initialValues: Record<PhaseSpaceDimension, number>;
  mode: PhaseSpaceMode;
  tiling: TileConfig;
}

export interface SimulationConfig {
  system: SystemType;
  vizMode: VizMode;
  resolution: Resolution;
  chunkSize: ChunkSize;
  phaseSpace: PhaseSpaceConfig;
  dt: number;
  iterationsPerFrame: number;
  maxIter: number;
  perturb: number;
  perturbDistribution: PerturbDistribution;
  trials: number;
  m1: number;
  m2: number;
  L1: number;
  L2: number;
  k1: number;
  k2: number;
  sculptureWeight: number;
  sculptureRod: number;
  sculptureAxle: number;
  sculptureReduction: number;
  sculptureN: number;
  rpM0: number;
  rpM1: number;
  rpL0: number;
  rpL1: number;
  rpA0: number;
  colormap: Colormap;
  toneMapping: ToneMapping;
  seed: number;
}

export interface SimulationState {
  frameCount: number;
  maxValue: number;
  readIndex: 0 | 1;
  seed: number;
}
