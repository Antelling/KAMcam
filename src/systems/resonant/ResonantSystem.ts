import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { ResonantStateA, ResonantStateB, ResonantParams, DataCell } from './types';
import { systemDeriv, computeResonantTip } from './deriv';
import type { System } from '../System';

export class ResonantSystem implements System {
  readonly key = 'resonant' as const;
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
    this.StateArrayA = d.arrayOf(ResonantStateA, cellCount);
    this.StateArrayB = d.arrayOf(ResonantStateB, cellCount);
    this.DataArray = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(ResonantParams, this.buildParamsData(config)).$usage('uniform');
    const stateAa = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateAb = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateBa = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    const stateBb = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroDataArray()).$usage('storage');

    this.initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ResonantParams },
    });
    this.stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' },
      nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ResonantParams },
    });
    this.accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: ResonantParams },
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
      initLayout.$.stateA[cellIndex] = ResonantStateA({
        theta0: omu * omv * p.cA00_th + u * omv * p.cA10_th + omu * v * p.cA01_th + u * v * p.cA11_th,
        omega0: omu * omv * p.cA00_om + u * omv * p.cA10_om + omu * v * p.cA01_om + u * v * p.cA11_om,
        dummy1: 0.0,
        dummy2: 0.0,
      });
      initLayout.$.stateB[cellIndex] = ResonantStateB({
        theta1: omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th,
        omega1: omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om,
        dummy1: 0.0,
        dummy2: 0.0,
      });
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const sa = stepLayout.$.currentA[cellIndex];
      const sb = stepLayout.$.currentB[cellIndex];
      const p = stepLayout.$.params;
      const dt = p.dt;
      const hdt = 0.5 * dt;
      const s6 = dt / 6.0;
      const d1 = systemDeriv(sa.theta0, sa.omega0, sb.theta1, sb.omega1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      const k2_th0 = sa.theta0 + hdt * d1.da_th;
      const k2_om0 = sa.omega0 + hdt * d1.da_om;
      const k2_th1 = sb.theta1 + hdt * d1.db_th;
      const k2_om1 = sb.omega1 + hdt * d1.db_om;
      const d2 = systemDeriv(k2_th0, k2_om0, k2_th1, k2_om1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      const k3_th0 = sa.theta0 + hdt * d2.da_th;
      const k3_om0 = sa.omega0 + hdt * d2.da_om;
      const k3_th1 = sb.theta1 + hdt * d2.db_th;
      const k3_om1 = sb.omega1 + hdt * d2.db_om;
      const d3 = systemDeriv(k3_th0, k3_om0, k3_th1, k3_om1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      const k4_th0 = sa.theta0 + dt * d3.da_th;
      const k4_om0 = sa.omega0 + dt * d3.da_om;
      const k4_th1 = sb.theta1 + dt * d3.db_th;
      const k4_om1 = sb.omega1 + dt * d3.db_om;
      const d4 = systemDeriv(k4_th0, k4_om0, k4_th1, k4_om1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      stepLayout.$.nextA[cellIndex] = ResonantStateA({
        theta0: sa.theta0 + s6 * (d1.da_th + 2.0 * d2.da_th + 2.0 * d3.da_th + d4.da_th),
        omega0: sa.omega0 + s6 * (d1.da_om + 2.0 * d2.da_om + 2.0 * d3.da_om + d4.da_om),
        dummy1: 0.0,
        dummy2: 0.0,
      });
      stepLayout.$.nextB[cellIndex] = ResonantStateB({
        theta1: sb.theta1 + s6 * (d1.db_th + 2.0 * d2.db_th + 2.0 * d3.db_th + d4.db_th),
        omega1: sb.omega1 + s6 * (d1.db_om + 2.0 * d2.db_om + 2.0 * d3.db_om + d4.db_om),
        dummy1: 0.0,
        dummy2: 0.0,
      });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const sa = accumulateLayout.$.currentA[cellIndex];
      const sb = accumulateLayout.$.currentB[cellIndex];
      const data = accumulateLayout.$.data[cellIndex];
      const p = accumulateLayout.$.params;
      const tip = computeResonantTip(sa.theta0, sb.theta1, p.rpL0, p.rpL1, p.rpA0);
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
      rpM0: config.rpM0,
      rpM1: config.rpM1,
      rpL0: config.rpL0,
      rpL1: config.rpL1,
      rpA0: config.rpA0,
      dt: config.dt,
      resolution: config.resolution,
      cA00_th: cA00[0], cA00_om: cA00[1], cA00_d1: cA00[2], cA00_d2: cA00[3],
      cA10_th: cA10[0], cA10_om: cA10[1], cA10_d1: cA10[2], cA10_d2: cA10[3],
      cA01_th: cA01[0], cA01_om: cA01[1], cA01_d1: cA01[2], cA01_d2: cA01[3],
      cA11_th: cA11[0], cA11_om: cA11[1], cA11_d1: cA11[2], cA11_d2: cA11[3],
      cB00_th: cB00[0], cB00_om: cB00[1], cB00_d1: cB00[2], cB00_d2: cB00[3],
      cB10_th: cB10[0], cB10_om: cB10[1], cB10_d1: cB10[2], cB10_d2: cB10[3],
      cB01_th: cB01[0], cB01_om: cB01[1], cB01_d1: cB01[2], cB01_d2: cB01[3],
      cB11_th: cB11[0], cB11_om: cB11[1], cB11_d1: cB11[2], cB11_d2: cB11[3],
    };
  }

  private zeroStateA() {
    return Array.from({ length: this.cellCount }, () => ({ theta0: 0, omega0: 0, dummy1: 0, dummy2: 0 }));
  }

  private zeroStateB() {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, dummy1: 0, dummy2: 0 }));
  }

  private zeroDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }
}
