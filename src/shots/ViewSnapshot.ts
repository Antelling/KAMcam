export const SNAPSHOT_VERSION = 1;

export interface ViewSnapshotParams {
  m1?: number;
  m2?: number;
  L1?: number;
  L2?: number;
  k1?: number;
  k2?: number;
  sculptureWeight?: number;
  sculptureRod?: number;
  sculptureAxle?: number;
  sculptureReduction?: number;
  sculptureN?: number;
  rpM0?: number;
  rpM1?: number;
  rpL0?: number;
  rpL1?: number;
  rpA0?: number;
  [key: string]: number | undefined;
}

export interface ViewSnapshotViewport {
  xDimension?: string;
  xMin?: number;
  xMax?: number;
  yDimension?: string;
  yMin?: number;
  yMax?: number;
}

export interface ViewSnapshotRender {
  colormap?: number;
  toneMapping?: number;
  resolution?: number;
  iterations?: number;
  dt?: number;
}

export interface ViewSnapshot {
  version: typeof SNAPSHOT_VERSION;
  system: string;
  params: ViewSnapshotParams;
  initialState: Record<string, number> | null;
  viewport?: ViewSnapshotViewport;
  render?: ViewSnapshotRender;
}
