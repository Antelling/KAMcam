import { tgpu, d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { DataCell } from '../systems/rigid/types';

const REDUCTION_THREADS = 256;

const ReduceParams = d.struct({
  totalCells: d.f32,
  _pad: d.f32,
  _pad2: d.f32,
  _pad3: d.f32,
});

const MaxArray = d.arrayOf(d.f32, REDUCTION_THREADS);

export class Reductions {
  private root: TgpuRoot;
  private layout: any;
  private pipeline: any;
  private bindGroup: any;
  private maxBuffer: any;
  private paramsBuffer: any;
  private dataArrType: any;

  constructor(root: TgpuRoot, dataArray: any, cellCount: number) {
    this.root = root;
    this.dataArrType = d.arrayOf(DataCell, cellCount);
    this.maxBuffer = root.createBuffer(MaxArray, new Array(REDUCTION_THREADS).fill(0)).$usage('storage');
    this.paramsBuffer = root.createBuffer(ReduceParams, { totalCells: cellCount, _pad: 0, _pad2: 0, _pad3: 0 }).$usage('uniform');
    this.layout = tgpu.bindGroupLayout({
      data: { storage: this.dataArrType },
      maxBuf: { storage: MaxArray, access: 'mutable' },
      params: { uniform: ReduceParams },
    });
    this.bindGroup = root.createBindGroup(this.layout, { data: dataArray, maxBuf: this.maxBuffer, params: this.paramsBuffer });

    const reduceLayout = this.layout;
    const reduceMax = (threadIndex: number) => {
      'use gpu';
      const total = d.u32(reduceLayout.$.params.totalCells);
      const chunk = (total + 255) / 256;
      let localMax: number = 0;
      const start = d.u32(threadIndex) * d.u32(chunk);
      for (let i = 0; i < 2048; i++) {
        const idx = start + d.u32(i);
        if (idx >= total) break;
        const val = reduceLayout.$.data[idx].b;
        if (val > localMax) {
          localMax = val;
        }
      }
      reduceLayout.$.maxBuf[threadIndex] = localMax;
    };
    this.pipeline = root.createGuardedComputePipeline(reduceMax);
  }

  updateData(dataArray: any): void {
    this.bindGroup = this.root.createBindGroup(this.layout, { data: dataArray, maxBuf: this.maxBuffer, params: this.paramsBuffer });
  }

  async computeMax(): Promise<number> {
    this.pipeline.with(this.bindGroup).dispatchThreads(REDUCTION_THREADS);
    const mapped = await this.maxBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(mapped);
    let max = 0;
    for (let i = 0; i < REDUCTION_THREADS; i++) {
      if (result[i] > max) max = result[i];
    }
    this.maxBuffer.unmap();
    return max;
  }
}