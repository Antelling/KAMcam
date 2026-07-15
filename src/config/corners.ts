import type { SimulationConfig } from './schema';

export function computeCorners(
  config: SimulationConfig,
  tileCol = 0,
  tileRow = 0,
): [number[], number[], number[], number[]] {
  const ps = config.phaseSpace;
  if (ps.mode === 'tiling') {
    const t = ps.tiling;
    const cols = t.cols;
    const rows = t.rows;
    const i0 = tileCol;
    const i1 = t.toroidal ? (tileCol + 1) % cols : Math.min(tileCol + 1, cols);
    const j0 = tileRow;
    const j1 = t.toroidal ? (tileRow + 1) % rows : Math.min(tileRow + 1, rows);
    const get = (col: number, row: number): number[] => t.controlNet[row][col];
    return [get(i0, j0).slice(), get(i1, j0).slice(), get(i0, j1).slice(), get(i1, j1).slice()];
  }
  const iv: number[] = [];
  for (const d of ['angle1', 'velocity1', 'stretch1', 'stretchRate1', 'angle2', 'velocity2', 'stretch2', 'stretchRate2'] as const) {
    iv.push(ps.initialValues[d]);
  }
  const xDim = ps.x.dimension;
  const yDim = ps.y.dimension;
  const xIdx = ['angle1', 'velocity1', 'stretch1', 'stretchRate1', 'angle2', 'velocity2', 'stretch2', 'stretchRate2'].indexOf(xDim);
  const yIdx = ['angle1', 'velocity1', 'stretch1', 'stretchRate1', 'angle2', 'velocity2', 'stretch2', 'stretchRate2'].indexOf(yDim);
  const xb = new Array(8).fill(0);
  xb[xIdx] = 1;
  const yb = new Array(8).fill(0);
  yb[yIdx] = 1;
  const add = (base: number[], sx: number, sy: number): number[] =>
    base.map((v, i) => v + sx * xb[i] + sy * yb[i]);
  return [
    add(iv, ps.x.min, ps.y.min),
    add(iv, ps.x.max, ps.y.min),
    add(iv, ps.x.min, ps.y.max),
    add(iv, ps.x.max, ps.y.max),
  ];
}

export function bilinearSample(
  corners: [number[], number[], number[], number[]],
  u: number,
  v: number,
): number[] {
  const [c00, c10, c01, c11] = corners;
  return c00.map((_, i) =>
    (1 - u) * (1 - v) * c00[i] + u * (1 - v) * c10[i]
    + (1 - u) * v * c01[i] + u * v * c11[i],
  );
}

export function rigidPack(dir8: number[]): [number, number, number, number] {
  return [dir8[0], dir8[1], dir8[4], dir8[5]];
}

export function elasticPackA(dir8: number[]): [number, number, number, number] {
  return [dir8[0], dir8[1], dir8[2], dir8[3]];
}

export function elasticPackB(dir8: number[]): [number, number, number, number] {
  return [dir8[4], dir8[5], dir8[6], dir8[7]];
}

export function generateTiling(system: string, sculptureN: number, cols: number, rows: number, center: number[]) {
  const availIdx = [0, 1, 4, 5, 2, 3, 6, 7];
  const scale = [Math.PI, 5, Math.PI, 5, 0.5, 5, 0.5, 5];
  const net: number[][][] = [];
  for (let r = 0; r < rows; r++) {
    const rowPts: number[][] = [];
    for (let c = 0; c < cols; c++) {
      const pt = center.slice();
      for (const idx of availIdx) {
        pt[idx] = center[idx] + (Math.random() * 2 - 1) * scale[idx];
      }
      rowPts.push(pt);
    }
    net.push(rowPts);
  }
  return { cols, rows, toroidal: cols >= 2 && rows >= 2, controlNet: net };
}

export function describeTiling(t: { cols: number; rows: number; toroidal: boolean }): string {
  return `${t.cols}x${t.rows}${t.toroidal ? ' torus' : ''}`;
}
