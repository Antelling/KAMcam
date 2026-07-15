import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { SculptureStateA, SculptureStateB, SculptureParams, DataCell } from './types';
import { systemDeriv, computeSculptureTip } from './deriv';
import type { System } from '../System';

export class SculptureSystem implements System {
  readonly key = 'sculpture' as const;
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
    this.StateArrayA = d.arrayOf(SculptureStateA, cellCount);
    this.StateArrayB = d.arrayOf(SculptureStateB, cellCount);
    this.DataArray = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(SculptureParams, this.buildParamsData(config)).$usage('uniform');
    const stateAa = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateAb = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateBa = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    const stateBb = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroDataArray()).$usage('storage');

    this.initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: SculptureParams },
    });
    this.stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' },
      nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: SculptureParams },
    });
    this.accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: SculptureParams },
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
      initLayout.$.stateA[cellIndex] = SculptureStateA({
        theta0: omu * omv * p.cA00_t0 + u * omv * p.cA10_t0 + omu * v * p.cA01_t0 + u * v * p.cA11_t0,
        omega0: omu * omv * p.cA00_w0 + u * omv * p.cA10_w0 + omu * v * p.cA01_w0 + u * v * p.cA11_w0,
        theta1: omu * omv * p.cA00_t1 + u * omv * p.cA10_t1 + omu * v * p.cA01_t1 + u * v * p.cA11_t1,
        omega1: omu * omv * p.cA00_w1 + u * omv * p.cA10_w1 + omu * v * p.cA01_w1 + u * v * p.cA11_w1,
      });
      initLayout.$.stateB[cellIndex] = SculptureStateB({
        theta2: omu * omv * p.cB00_t2 + u * omv * p.cB10_t2 + omu * v * p.cB01_t2 + u * v * p.cB11_t2,
        omega2: omu * omv * p.cB00_w2 + u * omv * p.cB10_w2 + omu * v * p.cB01_w2 + u * v * p.cB11_w2,
        theta3: omu * omv * p.cB00_t3 + u * omv * p.cB10_t3 + omu * v * p.cB01_t3 + u * v * p.cB11_t3,
        omega3: omu * omv * p.cB00_w3 + u * omv * p.cB10_w3 + omu * v * p.cB01_w3 + u * v * p.cB11_w3,
      });
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const sa = stepLayout.$.currentA[cellIndex];
      const sb = stepLayout.$.currentB[cellIndex];
      const p = stepLayout.$.params;
      const dt = p.dt;
      const h = 0.5 * dt;
      const sM = p.scM0;
      const sL = p.scL0;
      const sA = p.scA0;
      const sR = p.scR;
      const sN = p.scN;
      const d1 = systemDeriv(sa.theta0, sa.omega0, sa.theta1, sa.omega1, sb.theta2, sb.omega2, sb.theta3, sb.omega3, sM, sL, sA, sR, sN);
      const k2t0 = sa.theta0 + h * d1.dt0;
      const k2w0 = sa.omega0 + h * d1.dw0;
      const k2t1 = sa.theta1 + h * d1.dt1;
      const k2w1 = sa.omega1 + h * d1.dw1;
      const k2t2 = sb.theta2 + h * d1.dt2;
      const k2w2 = sb.omega2 + h * d1.dw2;
      const k2t3 = sb.theta3 + h * d1.dt3;
      const k2w3 = sb.omega3 + h * d1.dw3;
      const d2 = systemDeriv(k2t0, k2w0, k2t1, k2w1, k2t2, k2w2, k2t3, k2w3, sM, sL, sA, sR, sN);
      const k3t0 = sa.theta0 + h * d2.dt0;
      const k3w0 = sa.omega0 + h * d2.dw0;
      const k3t1 = sa.theta1 + h * d2.dt1;
      const k3w1 = sa.omega1 + h * d2.dw1;
      const k3t2 = sb.theta2 + h * d2.dt2;
      const k3w2 = sb.omega2 + h * d2.dw2;
      const k3t3 = sb.theta3 + h * d2.dt3;
      const k3w3 = sb.omega3 + h * d2.dw3;
      const d3 = systemDeriv(k3t0, k3w0, k3t1, k3w1, k3t2, k3w2, k3t3, k3w3, sM, sL, sA, sR, sN);
      const k4t0 = sa.theta0 + dt * d3.dt0;
      const k4w0 = sa.omega0 + dt * d3.dw0;
      const k4t1 = sa.theta1 + dt * d3.dt1;
      const k4w1 = sa.omega1 + dt * d3.dw1;
      const k4t2 = sb.theta2 + dt * d3.dt2;
      const k4w2 = sb.omega2 + dt * d3.dw2;
      const k4t3 = sb.theta3 + dt * d3.dt3;
      const k4w3 = sb.omega3 + dt * d3.dw3;
      const d4 = systemDeriv(k4t0, k4w0, k4t1, k4w1, k4t2, k4w2, k4t3, k4w3, sM, sL, sA, sR, sN);
      const s6 = dt / 6.0;
      stepLayout.$.nextA[cellIndex] = SculptureStateA({
        theta0: sa.theta0 + s6 * (d1.dt0 + 2.0 * d2.dt0 + 2.0 * d3.dt0 + d4.dt0),
        omega0: sa.omega0 + s6 * (d1.dw0 + 2.0 * d2.dw0 + 2.0 * d3.dw0 + d4.dw0),
        theta1: sa.theta1 + s6 * (d1.dt1 + 2.0 * d2.dt1 + 2.0 * d3.dt1 + d4.dt1),
        omega1: sa.omega1 + s6 * (d1.dw1 + 2.0 * d2.dw1 + 2.0 * d3.dw1 + d4.dw1),
      });
      stepLayout.$.nextB[cellIndex] = SculptureStateB({
        theta2: sb.theta2 + s6 * (d1.dt2 + 2.0 * d2.dt2 + 2.0 * d3.dt2 + d4.dt2),
        omega2: sb.omega2 + s6 * (d1.dw2 + 2.0 * d2.dw2 + 2.0 * d3.dw2 + d4.dw2),
        theta3: sb.theta3 + s6 * (d1.dt3 + 2.0 * d2.dt3 + 2.0 * d3.dt3 + d4.dt3),
        omega3: sb.omega3 + s6 * (d1.dw3 + 2.0 * d2.dw3 + 2.0 * d3.dw3 + d4.dw3),
      });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const sa = accumulateLayout.$.currentA[cellIndex];
      const sb = accumulateLayout.$.currentB[cellIndex];
      const data = accumulateLayout.$.data[cellIndex];
      const p = accumulateLayout.$.params;
      const tip = computeSculptureTip(
        sa.theta0, sa.theta1, sb.theta2, sb.theta3,
        p.scA0, p.scL0, p.scR, p.scN,
      );
      if (data.a > 0.5) {
        const dx = tip.x - data.r;
        const dy = tip.y - data.g;
        const dist = std.sqrt(dx * dx + dy * dy);
        accumulateLayout.$.data[cellIndex] = DataCell({ r: tip.x, g: tip.y, b: data.b + dist, a: 1.0 });
      } else {
        accumulateLayout.$.data[cellIndex] = DataCell({ r: tip.x, g: tip.y, b: 0.0, a: 1.0 });
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
      scM0: config.sculptureWeight, scL0: config.sculptureRod, scA0: config.sculptureAxle,
      scR: config.sculptureReduction, scN: config.sculptureN,
      dt: config.dt, resolution: config.resolution,
      cA00_t0: cA00[0], cA00_w0: cA00[1], cA00_t1: cA00[2], cA00_w1: cA00[3],
      cA10_t0: cA10[0], cA10_w0: cA10[1], cA10_t1: cA10[2], cA10_w1: cA10[3],
      cA01_t0: cA01[0], cA01_w0: cA01[1], cA01_t1: cA01[2], cA01_w1: cA01[3],
      cA11_t0: cA11[0], cA11_w0: cA11[1], cA11_t1: cA11[2], cA11_w1: cA11[3],
      cB00_t2: cB00[0], cB00_w2: cB00[1], cB00_t3: cB00[2], cB00_w3: cB00[3],
      cB10_t2: cB10[0], cB10_w2: cB10[1], cB10_t3: cB10[2], cB10_w3: cB10[3],
      cB01_t2: cB01[0], cB01_w2: cB01[1], cB01_t3: cB01[2], cB01_w3: cB01[3],
      cB11_t2: cB11[0], cB11_w2: cB11[1], cB11_t3: cB11[2], cB11_w3: cB11[3],
    };
  }

  private zeroStateA() {
    return Array.from({ length: this.cellCount }, () => ({ theta0: 0, omega0: 0, theta1: 0, omega1: 0 }));
  }

  private zeroStateB() {
    return Array.from({ length: this.cellCount }, () => ({ theta2: 0, omega2: 0, theta3: 0, omega3: 0 }));
  }

  private zeroDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }
}
