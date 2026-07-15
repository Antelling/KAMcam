import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, rigidPack } from '../../config/corners';
import { RigidState, RigidParams, DataCell } from './types';
import { computeAccelerations, computeBob2 } from './deriv';
import type { System } from '../System';

export class RigidSystem implements System {
  readonly key = 'rigid' as const;
  readonly stateSize = 16;
  private root!: TgpuRoot;
  private cellCount = 0;
  private StateArray: any = null;
  private DataArray: any = null;
  private initLayout: any = null;
  private stepLayout: any = null;
  private accumulateLayout: any = null;
  private initPipeline: any = null;
  private stepPipeline: any = null;
  private accumulatePipeline: any = null;
  private initBG: any = null;
  private stepFwd: any = null;
  private stepBwd: any = null;
  private accA: any = null;
  private accB: any = null;
  private paramsBuffer: any = null;
  private dataBuf: any = null;
  private _readIndex: 0 | 1 = 0;

  build(root: TgpuRoot, config: SimulationConfig, cellCount: number): void {
    this.root = root;
    this.cellCount = cellCount;
    this.StateArray = d.arrayOf(RigidState, cellCount);
    this.DataArray = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(RigidParams, this.buildParamsData(config)).$usage('uniform');
    const stateA = root.createBuffer(this.StateArray, this.zeroStateArray()).$usage('storage');
    const stateB = root.createBuffer(this.StateArray, this.zeroStateArray()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroDataArray()).$usage('storage');

    this.initLayout = tgpu.bindGroupLayout({
      state: { storage: this.StateArray, access: 'mutable' },
      params: { uniform: RigidParams },
    });
    this.stepLayout = tgpu.bindGroupLayout({
      currentState: { storage: this.StateArray },
      nextState: { storage: this.StateArray, access: 'mutable' },
      params: { uniform: RigidParams },
    });
    this.accumulateLayout = tgpu.bindGroupLayout({
      currentState: { storage: this.StateArray },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: RigidParams },
    });

    const initLayout = this.initLayout;
    const stepLayout = this.stepLayout;
    const accumulateLayout = this.accumulateLayout;

    const initCell = (cellIndex: number) => {
      'use gpu';
      const p = initLayout.$.params;
      const res = p.resolution;
      const x = cellIndex % d.u32(res);
      const y = cellIndex / d.u32(res);
      const u = d.f32(x) / res;
      const v = d.f32(y) / res;
      const t1 = (1 - u) * (1 - v) * p.c00_t1 + u * (1 - v) * p.c10_t1 + (1 - u) * v * p.c01_t1 + u * v * p.c11_t1;
      const w1 = (1 - u) * (1 - v) * p.c00_w1 + u * (1 - v) * p.c10_w1 + (1 - u) * v * p.c01_w1 + u * v * p.c11_w1;
      const t2 = (1 - u) * (1 - v) * p.c00_t2 + u * (1 - v) * p.c10_t2 + (1 - u) * v * p.c01_t2 + u * v * p.c11_t2;
      const w2 = (1 - u) * (1 - v) * p.c00_w2 + u * (1 - v) * p.c10_w2 + (1 - u) * v * p.c01_w2 + u * v * p.c11_w2;
      initLayout.$.state[cellIndex] = RigidState({ theta1: t1, omega1: w1, theta2: t2, omega2: w2 });
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const state = stepLayout.$.currentState[cellIndex];
      const p = stepLayout.$.params;
      const dt = p.dt;
      const a0 = computeAccelerations(state.theta1, state.omega1, state.theta2, state.omega2, p.m1, p.m2, p.L1, p.L2);
      const halfW1 = state.omega1 + 0.5 * dt * a0.x;
      const halfW2 = state.omega2 + 0.5 * dt * a0.y;
      const newT1 = state.theta1 + dt * halfW1;
      const newT2 = state.theta2 + dt * halfW2;
      const a1 = computeAccelerations(newT1, halfW1, newT2, halfW2, p.m1, p.m2, p.L1, p.L2);
      stepLayout.$.nextState[cellIndex] = RigidState({
        theta1: newT1, omega1: halfW1 + 0.5 * dt * a1.x,
        theta2: newT2, omega2: halfW2 + 0.5 * dt * a1.y,
      });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const state = accumulateLayout.$.currentState[cellIndex];
      const data = accumulateLayout.$.data[cellIndex];
      const p = accumulateLayout.$.params;
      const bob = computeBob2(state.theta1, state.theta2, p.L1, p.L2);
      if (data.a > 0.5) {
        const dx = bob.x - data.r;
        const dy = bob.y - data.g;
        const dist = std.sqrt(dx * dx + dy * dy);
        accumulateLayout.$.data[cellIndex] = DataCell({ r: bob.x, g: bob.y, b: data.b + dist, a: 1.0 });
      } else {
        accumulateLayout.$.data[cellIndex] = DataCell({ r: bob.x, g: bob.y, b: 0.0, a: 1.0 });
      }
    };

    this.initPipeline = root.createGuardedComputePipeline(initCell);
    this.stepPipeline = root.createGuardedComputePipeline(stepCell);
    this.accumulatePipeline = root.createGuardedComputePipeline(accumulateCell);
    this.initBG = root.createBindGroup(this.initLayout, { state: stateA, params: this.paramsBuffer });
    this.stepFwd = root.createBindGroup(this.stepLayout, { currentState: stateA, nextState: stateB, params: this.paramsBuffer });
    this.stepBwd = root.createBindGroup(this.stepLayout, { currentState: stateB, nextState: stateA, params: this.paramsBuffer });
    this.accA = root.createBindGroup(this.accumulateLayout, { currentState: stateA, data: this.dataBuf, params: this.paramsBuffer });
    this.accB = root.createBindGroup(this.accumulateLayout, { currentState: stateB, data: this.dataBuf, params: this.paramsBuffer });
    this._readIndex = 0;
  }

  updateParams(config: SimulationConfig): void {
    this.paramsBuffer.write(this.buildParamsData(config));
  }

  get data() { return this.dataBuf; }
  getReadIndex() { return this._readIndex; }

  initSim(): void {
    this.initPipeline.with(this.initBG).dispatchThreads(this.cellCount);
    this._readIndex = 0;
  }

  stepSim(): void {
    if (this._readIndex === 0) {
      this.stepPipeline.with(this.stepFwd).dispatchThreads(this.cellCount);
    } else {
      this.stepPipeline.with(this.stepBwd).dispatchThreads(this.cellCount);
    }
    this._readIndex = this._readIndex === 0 ? 1 : 0;
  }

  accumulateSim(): void {
    if (this._readIndex === 0) {
      this.accumulatePipeline.with(this.accA).dispatchThreads(this.cellCount);
    } else {
      this.accumulatePipeline.with(this.accB).dispatchThreads(this.cellCount);
    }
  }

  private buildParamsData(config: SimulationConfig) {
    const corners = computeCorners(config);
    const c00 = rigidPack(corners[0]);
    const c10 = rigidPack(corners[1]);
    const c01 = rigidPack(corners[2]);
    const c11 = rigidPack(corners[3]);
    return {
      m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
      dt: config.dt, resolution: config.resolution,
      c00_t1: c00[0], c00_w1: c00[1], c00_t2: c00[2], c00_w2: c00[3],
      c10_t1: c10[0], c10_w1: c10[1], c10_t2: c10[2], c10_w2: c10[3],
      c01_t1: c01[0], c01_w1: c01[1], c01_t2: c01[2], c01_w2: c01[3],
      c11_t1: c11[0], c11_w1: c11[1], c11_t2: c11[2], c11_w2: c11[3],
    };
  }

  private zeroStateArray() {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, theta2: 0, omega2: 0 }));
  }

  private zeroDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }
}
