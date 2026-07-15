import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { NonlinearStateA, NonlinearStateB, NonlinearParams, DataCell } from './types';
import { systemDeriv, computeBob2 } from './deriv';
import type { System } from '../System';

export class NonlinearSystem implements System {
  readonly key = 'nonlinear' as const;
  readonly stateSize = 32;
  private root!: TgpuRoot;
  private cellCount = 0;
  private StateArrayA: any = null;
  private StateArrayB: any = null;
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
    this.StateArrayA = d.arrayOf(NonlinearStateA, cellCount);
    this.StateArrayB = d.arrayOf(NonlinearStateB, cellCount);
    this.DataArray = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(NonlinearParams, this.buildParamsData(config)).$usage('uniform');
    const stateAa = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateAb = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateBa = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    const stateBb = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroDataArray()).$usage('storage');

    this.initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });
    this.stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' },
      nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });
    this.accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: NonlinearParams },
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
      const omu = 1.0 - u;
      const omv = 1.0 - v;
      initLayout.$.stateA[cellIndex] = NonlinearStateA({
        theta1: omu * omv * p.cA00_th + u * omv * p.cA10_th + omu * v * p.cA01_th + u * v * p.cA11_th,
        omega1: omu * omv * p.cA00_om + u * omv * p.cA10_om + omu * v * p.cA01_om + u * v * p.cA11_om,
        stretch1: omu * omv * p.cA00_r + u * omv * p.cA10_r + omu * v * p.cA01_r + u * v * p.cA11_r,
        stretchRate1: omu * omv * p.cA00_dr + u * omv * p.cA10_dr + omu * v * p.cA01_dr + u * v * p.cA11_dr,
      });
      initLayout.$.stateB[cellIndex] = NonlinearStateB({
        theta2: omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th,
        omega2: omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om,
        stretch2: omu * omv * p.cB00_r + u * omv * p.cB10_r + omu * v * p.cB01_r + u * v * p.cB11_r,
        stretchRate2: omu * omv * p.cB00_dr + u * omv * p.cB10_dr + omu * v * p.cB01_dr + u * v * p.cB11_dr,
      });
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const sa = stepLayout.$.currentA[cellIndex];
      const sb = stepLayout.$.currentB[cellIndex];
      const p = stepLayout.$.params;
      const dt = p.dt;
      const hdt = 0.5 * dt;
      const d1 = systemDeriv(sa.theta1, sa.omega1, sa.stretch1, sa.stretchRate1, sb.theta2, sb.omega2, sb.stretch2, sb.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2);
      const k2sa_th = sa.theta1 + hdt * d1.da_th;
      const k2sa_om = sa.omega1 + hdt * d1.da_om;
      const k2sa_r = sa.stretch1 + hdt * d1.da_r;
      const k2sa_dr = sa.stretchRate1 + hdt * d1.da_dr;
      const k2sb_th = sb.theta2 + hdt * d1.db_th;
      const k2sb_om = sb.omega2 + hdt * d1.db_om;
      const k2sb_r = sb.stretch2 + hdt * d1.db_r;
      const k2sb_dr = sb.stretchRate2 + hdt * d1.db_dr;
      const d2 = systemDeriv(k2sa_th, k2sa_om, k2sa_r, k2sa_dr, k2sb_th, k2sb_om, k2sb_r, k2sb_dr, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2);
      const k3sa_th = sa.theta1 + hdt * d2.da_th;
      const k3sa_om = sa.omega1 + hdt * d2.da_om;
      const k3sa_r = sa.stretch1 + hdt * d2.da_r;
      const k3sa_dr = sa.stretchRate1 + hdt * d2.da_dr;
      const k3sb_th = sb.theta2 + hdt * d2.db_th;
      const k3sb_om = sb.omega2 + hdt * d2.db_om;
      const k3sb_r = sb.stretch2 + hdt * d2.db_r;
      const k3sb_dr = sb.stretchRate2 + hdt * d2.db_dr;
      const d3 = systemDeriv(k3sa_th, k3sa_om, k3sa_r, k3sa_dr, k3sb_th, k3sb_om, k3sb_r, k3sb_dr, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2);
      const k4sa_th = sa.theta1 + dt * d3.da_th;
      const k4sa_om = sa.omega1 + dt * d3.da_om;
      const k4sa_r = sa.stretch1 + dt * d3.da_r;
      const k4sa_dr = sa.stretchRate1 + dt * d3.da_dr;
      const k4sb_th = sb.theta2 + dt * d3.db_th;
      const k4sb_om = sb.omega2 + dt * d3.db_om;
      const k4sb_r = sb.stretch2 + dt * d3.db_r;
      const k4sb_dr = sb.stretchRate2 + dt * d3.db_dr;
      const d4 = systemDeriv(k4sa_th, k4sa_om, k4sa_r, k4sa_dr, k4sb_th, k4sb_om, k4sb_r, k4sb_dr, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2);
      const s6 = dt / 6.0;
      stepLayout.$.nextA[cellIndex] = NonlinearStateA({
        theta1: sa.theta1 + s6 * (d1.da_th + 2.0 * d2.da_th + 2.0 * d3.da_th + d4.da_th),
        omega1: sa.omega1 + s6 * (d1.da_om + 2.0 * d2.da_om + 2.0 * d3.da_om + d4.da_om),
        stretch1: sa.stretch1 + s6 * (d1.da_r + 2.0 * d2.da_r + 2.0 * d3.da_r + d4.da_r),
        stretchRate1: sa.stretchRate1 + s6 * (d1.da_dr + 2.0 * d2.da_dr + 2.0 * d3.da_dr + d4.da_dr),
      });
      stepLayout.$.nextB[cellIndex] = NonlinearStateB({
        theta2: sb.theta2 + s6 * (d1.db_th + 2.0 * d2.db_th + 2.0 * d3.db_th + d4.db_th),
        omega2: sb.omega2 + s6 * (d1.db_om + 2.0 * d2.db_om + 2.0 * d3.db_om + d4.db_om),
        stretch2: sb.stretch2 + s6 * (d1.db_r + 2.0 * d2.db_r + 2.0 * d3.db_r + d4.db_r),
        stretchRate2: sb.stretchRate2 + s6 * (d1.db_dr + 2.0 * d2.db_dr + 2.0 * d3.db_dr + d4.db_dr),
      });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const sa = accumulateLayout.$.currentA[cellIndex];
      const sb = accumulateLayout.$.currentB[cellIndex];
      const data = accumulateLayout.$.data[cellIndex];
      const p = accumulateLayout.$.params;
      const bob = computeBob2(sa.theta1, sb.theta2, p.L1, p.L2, sa.stretch1, sb.stretch2);
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
    this.initBG = root.createBindGroup(this.initLayout, { stateA: stateAa, stateB: stateBa, params: this.paramsBuffer });
    this.stepFwd = root.createBindGroup(this.stepLayout, { currentA: stateAa, currentB: stateBa, nextA: stateAb, nextB: stateBb, params: this.paramsBuffer });
    this.stepBwd = root.createBindGroup(this.stepLayout, { currentA: stateAb, currentB: stateBb, nextA: stateAa, nextB: stateBa, params: this.paramsBuffer });
    this.accA = root.createBindGroup(this.accumulateLayout, { currentA: stateAa, currentB: stateBa, data: this.dataBuf, params: this.paramsBuffer });
    this.accB = root.createBindGroup(this.accumulateLayout, { currentA: stateAb, currentB: stateBb, data: this.dataBuf, params: this.paramsBuffer });
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
    const cA00 = elasticPackA(corners[0]);
    const cA10 = elasticPackA(corners[1]);
    const cA01 = elasticPackA(corners[2]);
    const cA11 = elasticPackA(corners[3]);
    const cB00 = elasticPackB(corners[0]);
    const cB10 = elasticPackB(corners[1]);
    const cB01 = elasticPackB(corners[2]);
    const cB11 = elasticPackB(corners[3]);
    return {
      m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
      k1: config.k1, k2: config.k2, dt: config.dt, resolution: config.resolution,
      cA00_th: cA00[0], cA00_om: cA00[1], cA00_r: cA00[2], cA00_dr: cA00[3],
      cA10_th: cA10[0], cA10_om: cA10[1], cA10_r: cA10[2], cA10_dr: cA10[3],
      cA01_th: cA01[0], cA01_om: cA01[1], cA01_r: cA01[2], cA01_dr: cA01[3],
      cA11_th: cA11[0], cA11_om: cA11[1], cA11_r: cA11[2], cA11_dr: cA11[3],
      cB00_th: cB00[0], cB00_om: cB00[1], cB00_r: cB00[2], cB00_dr: cB00[3],
      cB10_th: cB10[0], cB10_om: cB10[1], cB10_r: cB10[2], cB10_dr: cB10[3],
      cB01_th: cB01[0], cB01_om: cB01[1], cB01_r: cB01[2], cB01_dr: cB01[3],
      cB11_th: cB11[0], cB11_om: cB11[1], cB11_r: cB11[2], cB11_dr: cB11[3],
    };
  }

  private zeroStateA() {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, stretch1: 0, stretchRate1: 0 }));
  }

  private zeroStateB() {
    return Array.from({ length: this.cellCount }, () => ({ theta2: 0, omega2: 0, stretch2: 0, stretchRate2: 0 }));
  }

  private zeroDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }
}
