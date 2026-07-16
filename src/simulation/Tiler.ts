import { tgpu, d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../config/schema';
import { createSystemFromConfig } from '../systems/registry';
import { DataCell } from '../systems/rigid/types';

const TilerParams = d.struct({
  resolution: d.f32,
  tileCol: d.f32,
  tileRow: d.f32,
  tileW: d.f32,
  tileH: d.f32,
  _pad: d.f32,
  _pad2: d.f32,
  _pad3: d.f32,
});

export class Tiler {
  private root: TgpuRoot;
  private config: SimulationConfig;
  private cellCount: number;
  private cols: number;
  private rows: number;
  private totalTiles: number;
  private tileW: number;
  private tileH: number;
  private currentTile = 0;
  private tileSteps = 0;
  private tileDone = false;
  private compositeBuf: any;
  private copyPipeline: any;
  private copyBG: any;
  private copyParamsBuf: any;
  private system: any;
  private isDivergence: boolean;

  constructor(root: TgpuRoot, config: SimulationConfig) {
    this.root = root;
    this.config = config;
    this.cols = config.phaseSpace.tiling.cols;
    this.rows = config.phaseSpace.tiling.rows;
    this.totalTiles = this.cols * this.rows;
    this.cellCount = config.resolution * config.resolution;
    this.tileW = Math.floor(config.resolution / this.cols);
    this.tileH = Math.floor(config.resolution / this.rows);
    this.isDivergence = config.vizMode === 'divergence' || config.vizMode === 'divergenceDistance';

    const DataArr = d.arrayOf(DataCell, this.cellCount);
    this.compositeBuf = root.createBuffer(DataArr,
      new Array(this.cellCount).fill({ r: 0, g: 0, b: 0, a: 0 }),
    ).$usage('storage');

    this.copyParamsBuf = root.createBuffer(TilerParams, {
      resolution: config.resolution, tileCol: 0, tileRow: 0,
      tileW: this.tileW, tileH: this.tileH, _pad: 0, _pad2: 0, _pad3: 0,
    }).$usage('uniform');

    const copyLayout = tgpu.bindGroupLayout({
      src: { storage: DataArr },
      dst: { storage: DataArr, access: 'mutable' },
      params: { uniform: TilerParams },
    });
    const cl = copyLayout;
    const tileCopy = (ci: number) => {
      'use gpu';
      const p = cl.$.params;
      const res = d.u32(p.resolution);
      const tCol = d.u32(p.tileCol);
      const tRow = d.u32(p.tileRow);
      const tw = d.u32(p.tileW);
      const th = d.u32(p.tileH);
      const x = ci % res;
      const y = ci / res;
      const dstX = x + tCol * tw;
      const dstY = y + tRow * th;
      if (dstX < res && dstY < res) {
        cl.$.dst[dstY * res + dstX] = DataCell(cl.$.src[ci]);
      }
    };
    this.copyPipeline = root.createGuardedComputePipeline(tileCopy);

    this.system = createSystemFromConfig(config);
    this.system.build(root, config, this.cellCount);

    this.copyBG = root.createBindGroup(copyLayout, {
      src: this.system.data, dst: this.compositeBuf, params: this.copyParamsBuf,
    });

    this.startTile();
  }

  get data() { return this.compositeBuf; }

  getFrameCount(): number { return this.tileSteps; }

  isComplete(): boolean { return this.currentTile >= this.totalTiles; }

  getProgress(): { current: number; total: number } {
    return { current: this.currentTile + (this.tileDone ? 1 : 0), total: this.totalTiles };
  }

  step(): void {
    if (this.isComplete()) return;

    const batchSize = Math.min(this.config.iterationsPerFrame, this.config.maxIter - this.tileSteps);
    for (let i = 0; i < batchSize; i++) {
      if (this.isDivergence) {
        this.system.divergenceStep();
      } else {
        this.system.stepSim();
      }
      this.tileSteps++;
      if (this.tileSteps >= this.config.maxIter) break;
    }
    if (!this.isDivergence) {
      this.system.accumulateSim();
    }

    if (this.tileSteps >= this.config.maxIter) {
      this.compositeTile();
      this.currentTile++;
      if (this.currentTile < this.totalTiles) {
        this.tileSteps = 0;
        this.startTile();
      } else {
        this.tileDone = true;
      }
    }
  }

  private startTile(): void {
    const tc = { ...this.config };
    (tc as any)._tileCol = this.currentTile % this.cols;
    (tc as any)._tileRow = Math.floor(this.currentTile / this.cols);
    this.system.updateParams(tc);
    if (this.isDivergence) {
      this.system.initDivergence(this.config.seed, this.config.perturb);
    } else {
      this.system.initSim();
    }
  }

  private compositeTile(): void {
    const col = this.currentTile % this.cols;
    const row = Math.floor(this.currentTile / this.cols);
    this.copyParamsBuf.write({
      resolution: this.config.resolution, tileCol: col, tileRow: row,
      tileW: this.tileW, tileH: this.tileH, _pad: 0, _pad2: 0, _pad3: 0,
    });
    this.copyPipeline.with(this.copyBG).dispatchThreads(this.cellCount);
  }
}