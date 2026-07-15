import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, rigidPack } from '../../config/corners';
import { RigidState, RigidParams, DivParams, DataCell } from './types';
import { computeAccelerations, computeBob2 } from './deriv';
import { hash } from '../shared/hash';
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
  private pertStateA: any = null;
  private pertStateB: any = null;
  private divDataBuf: any = null;
  private divParamsBuffer: any = null;
  private divInitLayout: any = null;
  private divStepLayout: any = null;
  private divInitPipeline: any = null;
  private divStepPipeline: any = null;
  private divInitBG: any = null;
  private divStepFwd: any = null;
  private divStepBwd: any = null;
  private divFrameCounter = 0;

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

    this.pertStateA = root.createBuffer(this.StateArray, this.zeroStateArray()).$usage('storage');
    this.pertStateB = root.createBuffer(this.StateArray, this.zeroStateArray()).$usage('storage');
    this.divDataBuf = root.createBuffer(this.DataArray, this.zeroDivDataArray()).$usage('storage');
    this.divParamsBuffer = root.createBuffer(DivParams, this.buildDivParamsData(config, 0, 0)).$usage('uniform');

    this.divInitLayout = tgpu.bindGroupLayout({
      baseState: { storage: this.StateArray, access: 'mutable' },
      pertState: { storage: this.StateArray, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: DivParams },
    });
    this.divStepLayout = tgpu.bindGroupLayout({
      baseCurrent: { storage: this.StateArray },
      pertCurrent: { storage: this.StateArray },
      baseNext: { storage: this.StateArray, access: 'mutable' },
      pertNext: { storage: this.StateArray, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: DivParams },
    });

    const divInitLayout = this.divInitLayout;
    const divStepLayout = this.divStepLayout;

    const divInitCell = (cellIndex: number) => {
      'use gpu';
      const p = divInitLayout.$.params;
      const res = p.resolution;
      const x = cellIndex % d.u32(res);
      const y = cellIndex / d.u32(res);
      const u = d.f32(x) / res;
      const v = d.f32(y) / res;
      const t1 = (1 - u) * (1 - v) * p.c00_t1 + u * (1 - v) * p.c10_t1 + (1 - u) * v * p.c01_t1 + u * v * p.c11_t1;
      const w1 = (1 - u) * (1 - v) * p.c00_w1 + u * (1 - v) * p.c10_w1 + (1 - u) * v * p.c01_w1 + u * v * p.c11_w1;
      const t2 = (1 - u) * (1 - v) * p.c00_t2 + u * (1 - v) * p.c10_t2 + (1 - u) * v * p.c01_t2 + u * v * p.c11_t2;
      const w2 = (1 - u) * (1 - v) * p.c00_w2 + u * (1 - v) * p.c10_w2 + (1 - u) * v * p.c01_w2 + u * v * p.c11_w2;
      divInitLayout.$.baseState[cellIndex] = RigidState({ theta1: t1, omega1: w1, theta2: t2, omega2: w2 });
      const px = d.f32(x) / res;
      const py = d.f32(y) / res;
      const h1 = (hash(px * 1000 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      const h2 = (hash(px * 1000 + 100 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      divInitLayout.$.pertState[cellIndex] = RigidState({ theta1: t1 + h1, omega1: w1, theta2: t2 + h2, omega2: w2 });
      divInitLayout.$.divData[cellIndex] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const TWO_PI = 2 * std.acos(0);
    const circDiff = (a: number) => {
      'use gpu';
      const da = a - std.floor(a / TWO_PI + 0.5) * TWO_PI;
      return da;
    };

    const divStepCell = (cellIndex: number) => {
      'use gpu';
      const p = divStepLayout.$.params;
      const dt = p.dt;
      const base = divStepLayout.$.baseCurrent[cellIndex];
      const pert = divStepLayout.$.pertCurrent[cellIndex];
      const ba0 = computeAccelerations(base.theta1, base.omega1, base.theta2, base.omega2, p.m1, p.m2, p.L1, p.L2);
      const bHalfW1 = base.omega1 + 0.5 * dt * ba0.x;
      const bHalfW2 = base.omega2 + 0.5 * dt * ba0.y;
      const bNewT1 = base.theta1 + dt * bHalfW1;
      const bNewT2 = base.theta2 + dt * bHalfW2;
      const ba1 = computeAccelerations(bNewT1, bHalfW1, bNewT2, bHalfW2, p.m1, p.m2, p.L1, p.L2);
      divStepLayout.$.baseNext[cellIndex] = RigidState({
        theta1: bNewT1, omega1: bHalfW1 + 0.5 * dt * ba1.x,
        theta2: bNewT2, omega2: bHalfW2 + 0.5 * dt * ba1.y,
      });
      const pa0 = computeAccelerations(pert.theta1, pert.omega1, pert.theta2, pert.omega2, p.m1, p.m2, p.L1, p.L2);
      const pHalfW1 = pert.omega1 + 0.5 * dt * pa0.x;
      const pHalfW2 = pert.omega2 + 0.5 * dt * pa0.y;
      const pNewT1 = pert.theta1 + dt * pHalfW1;
      const pNewT2 = pert.theta2 + dt * pHalfW2;
      const pa1 = computeAccelerations(pNewT1, pHalfW1, pNewT2, pHalfW2, p.m1, p.m2, p.L1, p.L2);
      divStepLayout.$.pertNext[cellIndex] = RigidState({
        theta1: pNewT1, omega1: pHalfW1 + 0.5 * dt * pa1.x,
        theta2: pNewT2, omega2: pHalfW2 + 0.5 * dt * pa1.y,
      });
      const newBase = divStepLayout.$.baseNext[cellIndex];
      const newPert = divStepLayout.$.pertNext[cellIndex];
      const da1 = circDiff(newBase.theta1 - newPert.theta1);
      const dw1 = newBase.omega1 - newPert.omega1;
      const da2 = circDiff(newBase.theta2 - newPert.theta2);
      const dw2 = newBase.omega2 - newPert.omega2;
      const dist = std.sqrt(da1 * da1 + dw1 * dw1 + da2 * da2 + dw2 * dw2);
      const data = divStepLayout.$.divData[cellIndex];
      if (dist > 0.05 && data.g < 0.5) {
        divStepLayout.$.divData[cellIndex] = DataCell({ r: p.frameCounter, g: 1, b: 0, a: 1 });
      }
    };

    this.divInitPipeline = root.createGuardedComputePipeline(divInitCell);
    this.divStepPipeline = root.createGuardedComputePipeline(divStepCell);
    this.divInitBG = root.createBindGroup(this.divInitLayout, {
      baseState: stateA, pertState: this.pertStateA,
      divData: this.divDataBuf, params: this.divParamsBuffer,
    });
    this.divStepFwd = root.createBindGroup(this.divStepLayout, {
      baseCurrent: stateA, pertCurrent: this.pertStateA,
      baseNext: stateB, pertNext: this.pertStateB,
      divData: this.divDataBuf, params: this.divParamsBuffer,
    });
    this.divStepBwd = root.createBindGroup(this.divStepLayout, {
      baseCurrent: stateB, pertCurrent: this.pertStateB,
      baseNext: stateA, pertNext: this.pertStateA,
      divData: this.divDataBuf, params: this.divParamsBuffer,
    });
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

  initDivergence(seed: number, perturb: number): void {
    const cfg = this.lastConfig;
    this.divParamsBuffer.write(this.buildDivParamsData(cfg, seed, perturb));
    this.divFrameCounter = 0;
    this.divInitPipeline.with(this.divInitBG).dispatchThreads(this.cellCount);
    this._readIndex = 0;
  }

  divergenceStep(): void {
    this.divFrameCounter++;
    const cfg = this.lastConfig;
    const data = this.buildDivParamsData(cfg, 0, 0);
    data.frameCounter = this.divFrameCounter;
    this.divParamsBuffer.write(data);
    if (this._readIndex === 0) {
      this.divStepPipeline.with(this.divStepFwd).dispatchThreads(this.cellCount);
    } else {
      this.divStepPipeline.with(this.divStepBwd).dispatchThreads(this.cellCount);
    }
    this._readIndex = this._readIndex === 0 ? 1 : 0;
  }

  private lastConfig!: SimulationConfig;

  private buildParamsData(config: SimulationConfig) {
    this.lastConfig = config;
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

  private buildDivParamsData(config: SimulationConfig, seed: number, perturb: number) {
    const corners = computeCorners(config);
    const c00 = rigidPack(corners[0]);
    const c10 = rigidPack(corners[1]);
    const c01 = rigidPack(corners[2]);
    const c11 = rigidPack(corners[3]);
    return {
      m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
      dt: config.dt, resolution: config.resolution,
      seed: seed, perturb: perturb, frameCounter: 0,
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

  private zeroDivDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 1 }));
  }
}
